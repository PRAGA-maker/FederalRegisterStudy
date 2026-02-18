#!/usr/bin/env python3
"""
Mine comments from regulations.gov for Federal Register documents.

Reads federal_register_YYYY_comments.csv and extracts comments via regs.gov API.
Uses two-stage stratified sampling for efficient data collection.

Two-stage mining:
1. Collect comment IDs via list endpoint
2. Fetch full details via detail endpoint with smart sampling

Example:
    # CLI usage
    $ python -m stratification_scripts.makeup.mine_comments --year 2024
    
    # Programmatic usage
    >>> from stratification_scripts.makeup.mine_comments import mine_comments_for_year
    >>> from stratification_scripts.config import PipelineConfig
    >>> 
    >>> config = PipelineConfig(year=2024, fetch_strategy="stratified")
    >>> mine_comments_for_year(config)
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import polars as pl
from tqdm import tqdm

from stratification_scripts.config import (
    PipelineConfig,
    get_regs_api_keys,
    get_comments_raw_path,
    get_fr_csv_path,
)
from stratification_scripts.io_utils import sanitize_utf8, extract_pdf_text, extract_docx_text
from stratification_scripts.logging_utils import (
    get_logger,
    log_banner,
    setup_logging,
)
from stratification_scripts.federal_register.client import normalize_docket_id
from stratification_scripts.regulations_gov.client import RegsGovClient

logger = get_logger(__name__)

# Federal Register API endpoint for fallback
FR_DOC_DETAIL_URL = "https://www.federalregister.gov/api/v1/documents/{document_number}.json"


def calculate_sample_size(
    population_size: int,
    confidence: float = 0.95,
    margin: float = 0.05,
) -> int:
    """
    Calculate statistical sample size for proportion estimation.
    
    Uses finite population correction for accurate sample sizing.
    
    Args:
        population_size: Total population size
        confidence: Confidence level (0.95 for 95% CI)
        margin: Margin of error (0.05 for ±5%)
    
    Returns:
        Required sample size (at least population_size for small populations).
    """
    population_size = int(population_size)
    
    if population_size <= 0:
        return 0
    
    # For very small populations, always sample ALL
    if population_size < 10:
        return population_size
    
    # Z-score for confidence level
    Z = 1.96 if confidence == 0.95 else 2.576
    p = 0.5  # Maximum variance assumption
    E = margin
    
    # Sample size for infinite population
    n_infinite = (Z**2 * p * (1 - p)) / (E**2)
    
    # Apply finite population correction
    n_corrected = n_infinite / (1 + (n_infinite - 1) / population_size)
    
    return int(min(int(n_corrected) + 1, population_size))


def extract_comment_fields_from_list(item: dict) -> Dict[str, Any]:
    """
    Extract available fields from /comments list endpoint payload.
    
    The list endpoint returns basic fields without needing a detail call.
    
    Args:
        item: Comment item from list endpoint
    
    Returns:
        Dict with extracted fields and needs_detail_fetch flag.
    """
    if not isinstance(item, dict):
        return {}
    
    comment_id = item.get("id")
    attrs = item.get("attributes") or {}
    
    # Extract available fields — try known field names in order
    comment_text = ""
    for field_name in ("comment", "commentText", "comment_text"):
        val = attrs.get(field_name)
        if val:
            comment_text = val
            if field_name != "comment":
                logger.debug(f"Comment {comment_id}: text from non-standard field '{field_name}'")
            break
    
    # Submitter info
    first_name = attrs.get("firstName")
    last_name = attrs.get("lastName")
    organization = attrs.get("organization")
    submitter_type = attrs.get("submitterType")
    
    # Dates
    posted_date = attrs.get("postedDate")
    receive_date = attrs.get("receiveDate")
    
    # Attachment flag
    has_attachments = attrs.get("hasAttachments", False)
    
    # Decide if we need detail fetch
    text_length = len(str(comment_text or ""))
    needs_detail = (text_length == 0) or (has_attachments and text_length < 3000)
    
    return {
        "comment_id": sanitize_utf8(comment_id),
        "comment_text": sanitize_utf8(comment_text),
        "first_name": sanitize_utf8(first_name),
        "last_name": sanitize_utf8(last_name),
        "organization": sanitize_utf8(organization),
        "submitter_type": sanitize_utf8(submitter_type),
        "posted_date": posted_date,
        "receive_date": receive_date,
        "has_attachments": has_attachments,
        "needs_detail_fetch": needs_detail,
        "_list_payload": item,
    }


def extract_comment_fields_from_detail(
    data: dict,
    client: RegsGovClient,
) -> Dict[str, Any]:
    """
    Extract all fields from full comment detail response.
    
    Includes PDF attachment text extraction.
    
    Args:
        data: Full comment detail response
        client: RegsGovClient for attachment downloads
    
    Returns:
        Dict with all 22 comment fields.
    """
    if not isinstance(data, dict):
        return {}
    
    comment_data = data.get("data") or {}
    attrs = comment_data.get("attributes") or {}
    included = data.get("included") or []
    
    # Core fields
    comment_id = comment_data.get("id")
    comment_text = (
        attrs.get("comment") or 
        attrs.get("commentText") or 
        attrs.get("comment_text") or
        attrs.get("title") or
        ""
    )
    
    # Timestamps
    posted_date = attrs.get("postedDate")
    receive_date = attrs.get("receiveDate")
    postmark_date = attrs.get("postmarkDate")
    
    # Submitter info
    first_name = attrs.get("firstName")
    last_name = attrs.get("lastName")
    organization = attrs.get("organization")
    submitter_type = attrs.get("submitterType")
    
    # Location
    city = attrs.get("city")
    state_province_region = attrs.get("stateProvinceRegion")
    country = attrs.get("country")
    zip_code = attrs.get("zip")
    
    # Government fields
    gov_agency = attrs.get("govAgency")
    gov_agency_type = attrs.get("govAgencyType")
    
    # Attachment processing
    has_attachments = len(included) > 0
    attachment_count = len(included)
    attachment_formats: List[str] = []
    attachment_texts: List[str] = []
    
    if has_attachments:
        for att in included:
            if not isinstance(att, dict):
                continue

            att_attrs = att.get("attributes") or {}
            att_links = att.get("links") or {}

            # Extract format information
            file_formats = att_attrs.get("fileFormats", [])
            if isinstance(file_formats, list) and len(file_formats) > 0:
                for ff in file_formats:
                    if isinstance(ff, dict) and ff.get("format"):
                        attachment_formats.append(str(ff.get("format")))
            elif att_attrs.get("format"):
                attachment_formats.append(str(att_attrs.get("format")))

            # Detect attachment format and download URL
            detected_format = None
            file_url = None

            if isinstance(file_formats, list) and len(file_formats) > 0:
                for ff in file_formats:
                    if isinstance(ff, dict):
                        fmt = str(ff.get("format", "")).lower()
                        if "pdf" in fmt:
                            detected_format = "pdf"
                            file_url = ff.get("fileUrl")
                            break
                        elif "docx" in fmt or "word" in fmt or "document" in fmt:
                            detected_format = "docx"
                            file_url = ff.get("fileUrl")
                            break
                        elif "doc" in fmt:
                            detected_format = "doc"
                            file_url = ff.get("fileUrl")
                            break

            if not detected_format and att_attrs.get("format"):
                fmt = str(att_attrs.get("format", "")).lower()
                if "pdf" in fmt:
                    detected_format = "pdf"
                    file_url = att_attrs.get("fileUrl")
                elif "docx" in fmt or "word" in fmt or "document" in fmt:
                    detected_format = "docx"
                    file_url = att_attrs.get("fileUrl")
                elif "doc" in fmt:
                    detected_format = "doc"
                    file_url = att_attrs.get("fileUrl")

            if not detected_format:
                logger.warning(
                    f"Unsupported attachment format for comment, skipping: "
                    f"formats={attachment_formats}"
                )
                continue

            if not file_url:
                file_url = att_links.get("self")

            if not file_url:
                logger.warning(f"No download URL for {detected_format} attachment")
                continue

            # Download and extract text based on format
            file_bytes = client.download_bytes(file_url, timeout=30)
            if file_bytes:
                extracted_text = None
                if detected_format == "pdf":
                    extracted_text = extract_pdf_text(file_bytes, max_pages=2)
                elif detected_format in ("docx", "doc"):
                    extracted_text = extract_docx_text(file_bytes, max_pages=2)
                if extracted_text:
                    attachment_texts.append(extracted_text)
    
    attachment_formats_str = ",".join(set(attachment_formats)) if attachment_formats else None
    attachment_text = "\n\n---\n\n".join(attachment_texts) if attachment_texts else None
    
    # Other metadata
    duplicate_comments = attrs.get("numItemsRecieved") or attrs.get("duplicateComments")
    page_count = attrs.get("pageCount")
    
    return {
        "comment_id": sanitize_utf8(comment_id),
        "comment_text": sanitize_utf8(comment_text),
        "posted_date": posted_date,
        "receive_date": receive_date,
        "postmark_date": postmark_date,
        "first_name": sanitize_utf8(first_name),
        "last_name": sanitize_utf8(last_name),
        "organization": sanitize_utf8(organization),
        "submitter_type": sanitize_utf8(submitter_type),
        "city": sanitize_utf8(city),
        "state_province_region": sanitize_utf8(state_province_region),
        "country": sanitize_utf8(country),
        "zip": sanitize_utf8(zip_code),
        "gov_agency": sanitize_utf8(gov_agency),
        "gov_agency_type": sanitize_utf8(gov_agency_type),
        "has_attachments": has_attachments,
        "attachment_count": attachment_count,
        "attachment_formats": sanitize_utf8(attachment_formats_str),
        "attachment_text": sanitize_utf8(attachment_text),
        "duplicate_comments": duplicate_comments,
        "page_count": page_count,
    }


def load_documents(
    fr_csv: Path,
    limit_docs: Optional[int],
    requested_year: int,
) -> Optional[pl.DataFrame]:
    """
    Load and filter FR document CSV with year validation.
    
    Args:
        fr_csv: Path to federal_register CSV
        limit_docs: Maximum documents to load
        requested_year: Year being processed
    
    Returns:
        Filtered Polars DataFrame or None if file not found.
    """
    if not fr_csv.exists():
        logger.error(f"{fr_csv} not found")
        logger.error(
            f"Run: python -m stratification_scripts.distribution --year {requested_year}"
        )
        return None

    logger.info(f"Loading documents from: {fr_csv}")
    df_docs = pl.read_csv(str(fr_csv))

    # Validate year
    if "publication_date" in df_docs.columns:
        sample_size = min(1001, len(df_docs))
        sample_dates = df_docs.head(sample_size)["publication_date"].to_list()[1:]
        
        years_in_data = set()
        for date_str in sample_dates[:100]:
            if date_str and isinstance(date_str, str) and len(date_str) >= 4:
                try:
                    year = int(date_str[:4])
                    years_in_data.add(year)
                except (ValueError, TypeError):
                    continue
        
        if years_in_data and requested_year not in years_in_data:
            most_common_year = max(years_in_data) if years_in_data else "unknown"
            logger.warning(
                f"Requested year {requested_year}, but CSV contains data from {most_common_year}"
            )

    # Filter to documents that we can query for comments.
    # Primary path: regs_document_id is set and comment_count > 0
    # Fallback path: docket_id is set (for docs where FR API didn't provide
    # a regs_document_id — we resolve via Regs.gov docket lookup)
    has_regs_id = "regs_document_id" in df_docs.columns
    has_docket = "docket_id" in df_docs.columns
    has_count = "comment_count" in df_docs.columns

    if has_regs_id and has_count and has_docket:
        known_comments = (
            pl.col("regs_document_id").is_not_null() & (pl.col("comment_count") > 0)
        )
        # Docket fallback: docs with docket_id but no regs_document_id
        # (comment_count unknown — Step 1 couldn't query without document_id)
        docket_fallback = (
            pl.col("regs_document_id").is_null()
            & pl.col("docket_id").is_not_null()
            & (pl.col("docket_id") != "")
        )
        n_before = len(df_docs)
        df_docs = df_docs.filter(known_comments | docket_fallback)
        n_known = len(df_docs.filter(known_comments))
        n_docket = len(df_docs.filter(docket_fallback))
        logger.info(
            f"Filtered to {len(df_docs)} mineable documents "
            f"({n_known} with known comments, {n_docket} via docket fallback, "
            f"{n_before - len(df_docs)} excluded)"
        )
    elif has_regs_id and has_count:
        df_docs = df_docs.filter(
            (pl.col("regs_document_id").is_not_null()) & (pl.col("comment_count") > 0)
        )
        logger.info(f"Found {len(df_docs)} documents with comments > 0")
    elif has_regs_id:
        df_docs = df_docs.filter(pl.col("regs_document_id").is_not_null())
    else:
        logger.error("CSV missing regs_document_id column")
        return None

    if limit_docs:
        df_docs = df_docs.head(limit_docs)
        logger.info(f"Limited to first {limit_docs} documents")

    return df_docs


def load_existing_comment_ids(output_csv: Path) -> Set[str]:
    """Load existing comments file and return IDs for deduping."""
    if not output_csv.exists():
        return set()

    df_existing = pl.read_csv(
        str(output_csv),
        infer_schema_length=10000,
        schema_overrides={"zip": pl.Utf8},
    )
    existing_ids = set(df_existing["comment_id"].to_list())
    logger.info(f"Loaded {len(existing_ids)} existing comments")
    return existing_ids


def get_regs_document_id_via_fr(
    document_number: Optional[str],
    retries: int,
) -> Optional[str]:
    """Fetch regs_document_id from Federal Register API."""
    import requests
    
    if not document_number:
        return None
    
    backoff = 1.0
    for _ in range(max(1, retries)):
        try:
            r = requests.get(
                FR_DOC_DETAIL_URL.format(document_number=document_number),
                timeout=20,
            )
        except requests.RequestException:
            time.sleep(backoff)
            backoff = min(backoff * 2, 16)
            continue
        
        if r.status_code == 200:
            j = r.json() or {}
            fr_info = j.get("regulations_dot_gov_info") or {}
            if isinstance(fr_info, dict):
                return fr_info.get("document_id") or fr_info.get("documentId")
            return None
        
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(backoff)
            backoff = min(backoff * 2, 16)
            continue
        
        return None
    
    return None


def collect_comment_ids_for_document(
    row: Dict[str, Any],
    client: RegsGovClient,
    existing_ids: Set[str],
    retries: int,
    max_comments_per_doc: Optional[int],
    cache_payloads: Optional[Dict[str, dict]] = None,
) -> List[str]:
    """
    Collect comment IDs for a single document.
    
    Args:
        row: Document row with regs_document_id
        client: RegsGovClient instance
        existing_ids: Set of already-fetched comment IDs
        retries: Max retries
        max_comments_per_doc: Skip documents with more comments
        cache_payloads: Optional dict to cache list payloads
    
    Returns:
        List of comment IDs to fetch.
    """
    doc_number = row.get("document_number")
    regs_doc_id = row.get("regs_document_id")
    docket_id = normalize_docket_id(row.get("docket_id"))

    if not regs_doc_id:
        regs_doc_id = get_regs_document_id_via_fr(doc_number, retries)

    # Resolve object_id: primary path via document_id, fallback via docket_id
    object_id = None
    if regs_doc_id:
        object_id = client.get_object_id_for_document(regs_doc_id)

    if not object_id and docket_id:
        # Docket fallback: find commentable documents in the docket
        object_ids = client.find_object_ids_for_docket(docket_id)
        if object_ids:
            object_id = object_ids[0]
            logger.debug(
                f"Resolved {doc_number} via docket {docket_id} "
                f"({len(object_ids)} docs found)"
            )

    if not object_id:
        return []

    comment_ids: List[str] = []
    limit_for_doc = max_comments_per_doc if max_comments_per_doc else None

    try:
        for item in client.iter_comments_for_object(object_id, limit_comments=limit_for_doc):
            if not isinstance(item, dict):
                continue
            cid = item.get("id") or item.get("commentId")
            if cid and cid not in existing_ids:
                comment_ids.append(cid)
                if cache_payloads is not None:
                    list_fields = extract_comment_fields_from_list(item)
                    if list_fields:
                        cache_payloads[cid] = list_fields
    except Exception as exc:
        tqdm.write(f"  ERROR collecting IDs for {doc_number}: {exc}")
        return []

    if not comment_ids:
        return []

    if max_comments_per_doc and len(comment_ids) > max_comments_per_doc:
        tqdm.write(
            f"  {doc_number}: Skipping (has {len(comment_ids)} comments, limit is {max_comments_per_doc})"
        )
        return []

    return comment_ids


def build_two_stage_sample_plan(
    df_docs: pl.DataFrame,
    client: RegsGovClient,
    existing_ids: Set[str],
    retries: int,
    max_comments_per_doc: Optional[int],
    sample_docs: bool = True,
) -> Tuple[Dict[str, List[str]], Dict[str, dict], pl.DataFrame]:
    """
    Build two-stage stratified sampling plan.

    Stage 1: Sample documents within each stratum (agency × comment_bin)
              When sample_docs=False, ALL documents are used (no sqrt(n) sampling).
    Stage 2: Sample comments within each sampled document

    Args:
        df_docs: DataFrame with document metadata
        client: RegsGovClient instance
        existing_ids: Already-fetched comment IDs
        retries: Max retries
        max_comments_per_doc: Skip documents with more comments
        sample_docs: If True, use sqrt(n) doc sampling per stratum.
                     If False, process all documents (sample comments only).

    Returns:
        Tuple of:
        - Dict mapping doc_number to sampled comment IDs
        - Dict mapping comment_id to cached list payload
        - Stratification summary DataFrame
    """
    log_banner(logger, "TWO-STAGE STRATIFIED SAMPLING PLAN", width=60)
    
    # Assign comment bins
    logger.info("Assigning documents to comment bins...")
    df_docs = df_docs.with_columns([
        pl.when(pl.col("comment_count") <= 10).then(pl.lit("0-10"))
          .when(pl.col("comment_count") <= 100).then(pl.lit("11-100"))
          .when(pl.col("comment_count") <= 1000).then(pl.lit("101-1000"))
          .when(pl.col("comment_count") <= 10000).then(pl.lit("1001-10000"))
          .otherwise(pl.lit("10000+"))
          .alias("comment_bin")
    ])
    
    # Group by strata
    logger.info("Grouping into strata by agency × comment_bin...")
    strata = df_docs.group_by(["agency", "comment_bin"]).agg([
        pl.col("document_number").alias("doc_numbers"),
        pl.col("comment_count").sum().alias("total_comments"),
        pl.col("document_number").count().alias("num_docs"),
    ])
    
    # Calculate target sample sizes
    logger.info("Calculating sample sizes per stratum...")
    strata = strata.with_columns([
        pl.col("total_comments").map_elements(
            lambda x: calculate_sample_size(x, confidence=0.95, margin=0.05),
            return_dtype=pl.Int64
        ).alias("target_comments")
    ])
    
    strata = strata.sort(["agency", "comment_bin"])
    logger.info(f"Created {len(strata)} strata")
    
    # Stage 1: Sample documents
    log_banner(logger, "STAGE 1: SAMPLING DOCUMENTS", width=60)
    logger.info(f"Doc sampling: {'sqrt(n) per stratum' if sample_docs else 'ALL docs (comments only)'}")

    doc_to_comment_ids: Dict[str, List[str]] = {}
    comment_id_to_payload: Dict[str, dict] = {}
    total_ids_collected = 0

    doc_rows = {row["document_number"]: row for row in df_docs.iter_rows(named=True)}

    for row in tqdm(strata.iter_rows(named=True), total=len(strata), desc="Sampling strata", unit="stratum"):
        agency = row["agency"]
        comment_bin = row["comment_bin"]
        doc_numbers = row["doc_numbers"]
        target_comments = row["target_comments"]
        total_comments = row["total_comments"]
        num_docs = row["num_docs"]

        if not sample_docs:
            # All-docs mode: process every document
            docs_to_sample = doc_numbers
        elif num_docs <= 10 or total_comments <= 25:
            # Small strata: sample ALL
            docs_to_sample = doc_numbers
        else:
            # Two-stage: sample documents then comments
            n_docs_to_sample = min(num_docs, max(int(math.ceil(math.sqrt(num_docs))), 3))
            docs_to_sample = random.sample(list(doc_numbers), n_docs_to_sample)
        
        # Collect comment IDs
        for doc_num in docs_to_sample:
            doc_row = doc_rows.get(doc_num)
            if not doc_row:
                continue

            comment_ids = collect_comment_ids_for_document(
                doc_row, client, existing_ids, retries, max_comments_per_doc,
                cache_payloads=comment_id_to_payload
            )

            if not comment_ids:
                continue
            
            doc_to_comment_ids[doc_num] = comment_ids
            total_ids_collected += len(comment_ids)
    
    logger.info(f"Collected {total_ids_collected:,} comment IDs from {len(doc_to_comment_ids)} documents")
    
    # Stage 2: Sample comments
    log_banner(logger, "STAGE 2: SAMPLING COMMENTS", width=60)
    
    sampled_doc_to_ids: Dict[str, List[str]] = {}
    
    for row in strata.iter_rows(named=True):
        doc_numbers = row["doc_numbers"]
        target_comments = row["target_comments"]
        
        stratum_ids_by_doc: Dict[str, List[str]] = {}
        total_stratum_ids = 0
        
        for doc_num in doc_numbers:
            if doc_num in doc_to_comment_ids:
                ids = doc_to_comment_ids[doc_num]
                stratum_ids_by_doc[doc_num] = ids
                total_stratum_ids += len(ids)
        
        if total_stratum_ids == 0:
            continue

        # Docket-fallback docs have NULL comment_count in the CSV, which
        # produces target_comments=0 via sum(NULLs)=0. Now that we've
        # collected real IDs, recalculate the target from actual counts.
        if target_comments == 0 and total_stratum_ids > 0:
            target_comments = calculate_sample_size(total_stratum_ids)
            logger.info(
                f"Stratum had unknown comment counts -- recalculated "
                f"target from {total_stratum_ids} collected IDs: {target_comments}"
            )

        if target_comments is None or total_stratum_ids <= target_comments:
            for doc_num, ids in stratum_ids_by_doc.items():
                sampled_doc_to_ids[doc_num] = ids
        else:
            sample_rate = target_comments / total_stratum_ids
            for doc_num, ids in stratum_ids_by_doc.items():
                n_to_sample = max(1, int(len(ids) * sample_rate))
                n_to_sample = min(n_to_sample, len(ids))
                sampled = random.sample(ids, n_to_sample)
                sampled_doc_to_ids[doc_num] = sampled
    
    total_sampled = sum(len(ids) for ids in sampled_doc_to_ids.values())
    total_pop = strata['total_comments'].sum()
    
    logger.info(f"Documents sampled: {len(sampled_doc_to_ids)}/{len(df_docs)}")
    logger.info(f"Comments to fetch: {total_sampled:,}")
    if total_pop > 0:
        logger.info(f"Sampling ratio: {100*total_sampled/total_pop:.1f}%")
    
    return sampled_doc_to_ids, comment_id_to_payload, strata


def run_stratified_pipeline(
    df_docs: pl.DataFrame,
    client: RegsGovClient,
    existing_ids: Set[str],
    retries: int,
    max_comments_per_doc: Optional[int],
    concurrent_workers: int = 10,
    sample_docs: bool = True,
) -> List[Dict[str, Any]]:
    """
    Run two-stage stratified sampling pipeline with concurrent fetching.

    Args:
        df_docs: Document DataFrame
        client: RegsGovClient
        existing_ids: Already-fetched IDs
        retries: Max retries
        max_comments_per_doc: Skip large documents
        concurrent_workers: Number of concurrent workers
        sample_docs: If True, use sqrt(n) doc sampling. If False, all docs.

    Returns:
        List of comment records.
    """
    sampled_doc_to_ids, comment_payloads, strata = build_two_stage_sample_plan(
        df_docs, client, existing_ids, retries, max_comments_per_doc,
        sample_docs=sample_docs,
    )
    
    if not sampled_doc_to_ids:
        logger.info("No comments selected for fetching")
        return []

    # Build doc metadata lookup for lifecycle propagation
    doc_meta: Dict[str, Dict[str, Any]] = {}
    for row in df_docs.iter_rows(named=True):
        dn = row.get("document_number")
        if dn:
            doc_meta[dn] = {
                "lifecycle_stage": row.get("lifecycle_stage"),
                "rin": row.get("rin"),
            }

    # Stage 3: Fetch details
    log_banner(logger, "STAGE 3: FETCHING COMMENT DETAILS", width=60)
    
    all_comments: List[Dict[str, Any]] = []
    comments_lock = threading.Lock()
    
    total_to_fetch = sum(len(ids) for ids in sampled_doc_to_ids.values())
    
    # Prepare work items
    work_items = []
    for doc_number, comment_ids in sampled_doc_to_ids.items():
        for comment_id in comment_ids:
            work_items.append((doc_number, comment_id))
    
    def process_comment(doc_number: str, comment_id: str) -> Optional[Dict[str, Any]]:
        # Check cached list payload
        list_fields = comment_payloads.get(comment_id)
        
        if list_fields and not list_fields.get("needs_detail_fetch", True):
            # Use list payload only
            fields = {
                "document_number": sanitize_utf8(doc_number),
                "comment_id": list_fields.get("comment_id"),
                "comment_text": list_fields.get("comment_text"),
                "first_name": list_fields.get("first_name"),
                "last_name": list_fields.get("last_name"),
                "organization": list_fields.get("organization"),
                "submitter_type": list_fields.get("submitter_type"),
                "posted_date": list_fields.get("posted_date"),
                "receive_date": list_fields.get("receive_date"),
                "has_attachments": list_fields.get("has_attachments", False),
                "postmark_date": None,
                "city": None,
                "state_province_region": None,
                "country": None,
                "zip": None,
                "gov_agency": None,
                "gov_agency_type": None,
                "attachment_count": 0,
                "attachment_formats": None,
                "attachment_text": None,
                "duplicate_comments": None,
                "page_count": None,
            }

            meta = doc_meta.get(doc_number, {})
            fields["lifecycle_stage"] = meta.get("lifecycle_stage")
            fields["rin"] = meta.get("rin")

            existing_ids.add(comment_id)
            return fields
        else:
            # Fetch detail
            detail_data = client.fetch_comment_detail(comment_id, include_attachments=True)
            if not detail_data:
                return None
            
            fields = extract_comment_fields_from_detail(detail_data, client)
            if not fields or not fields.get("comment_id"):
                return None
            
            existing_ids.add(comment_id)
            fields["document_number"] = doc_number

            meta = doc_meta.get(doc_number, {})
            fields["lifecycle_stage"] = meta.get("lifecycle_stage")
            fields["rin"] = meta.get("rin")

            return fields
    
    logger.info(f"Using {concurrent_workers} concurrent workers")
    
    with ThreadPoolExecutor(max_workers=concurrent_workers) as executor:
        futures = {
            executor.submit(process_comment, doc_number, comment_id): (doc_number, comment_id)
            for doc_number, comment_id in work_items
        }
        
        with tqdm(total=total_to_fetch, desc="Processing comments", unit="comment") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result:
                    with comments_lock:
                        all_comments.append(result)
                pbar.update(1)
    
    logger.info(f"Processed {len(all_comments):,} comments")
    return all_comments


def write_comment_output(
    all_comments: List[Dict[str, Any]],
    output_csv: Path,
) -> None:
    """Save mined comments and print coverage summary."""
    if not all_comments:
        logger.info("No new comments found")
        return

    df_new = pl.DataFrame(all_comments, infer_schema_length=len(all_comments))
    column_order = [
        "document_number", "comment_id", "comment_text",
        "posted_date", "receive_date", "postmark_date",
        "first_name", "last_name", "organization", "submitter_type",
        "city", "state_province_region", "country", "zip",
        "gov_agency", "gov_agency_type",
        "has_attachments", "attachment_count", "attachment_formats", "attachment_text",
        "duplicate_comments", "page_count",
        "lifecycle_stage", "rin",
    ]

    existing_columns = [c for c in column_order if c in df_new.columns]
    df_new = df_new.select(existing_columns)

    if output_csv.exists():
        df_existing = pl.read_csv(
            str(output_csv),
            infer_schema_length=10000,
            schema_overrides={"zip": pl.Utf8},
        )

        for col in df_new.columns:
            if col not in df_existing.columns:
                df_existing = df_existing.with_columns(
                    pl.lit(None).cast(df_new[col].dtype).alias(col)
                )
            elif df_existing[col].dtype != df_new[col].dtype:
                try:
                    df_existing = df_existing.with_columns(pl.col(col).cast(df_new[col].dtype))
                except Exception:
                    df_new = df_new.with_columns(pl.col(col).cast(df_existing[col].dtype))

        for col in df_existing.columns:
            if col not in df_new.columns:
                df_new = df_new.with_columns(pl.lit(None).cast(df_existing[col].dtype).alias(col))

        df_existing = df_existing.select(df_new.columns)
        df_combined = pl.concat([df_existing, df_new])
    else:
        df_combined = df_new

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_combined.write_csv(str(output_csv))

    log_banner(logger, "SAVED RAW COMMENTS")
    logger.info(f"Mined {len(all_comments)} new comments")
    logger.info(f"Total comments in database: {len(df_combined)}")
    logger.info(f"Output file: {output_csv.absolute()}")


def mine_comments_for_year(config: PipelineConfig) -> None:
    """
    Mine comments for a single year.
    
    This is the main entry point for programmatic use.
    
    Args:
        config: Pipeline configuration
    
    Side Effects:
        Makes HTTP requests to Regulations.gov API
        Writes CSV to data directory
    """
    fr_csv = get_fr_csv_path(config.year)
    output_csv = get_comments_raw_path(config.year)
    
    api_keys = get_regs_api_keys(required=True)
    
    df_docs = load_documents(fr_csv, config.limit_docs, config.year)
    if df_docs is None or len(df_docs) == 0:
        return

    logger.info(f"Mining comments from {len(df_docs)} documents...")
    logger.info(f"Fetch strategy: {config.fetch_strategy}")

    if config.fetch_strategy == "stratified":
        sample_docs = True
    elif config.fetch_strategy == "all":
        sample_docs = False
    else:
        raise ValueError(f"Invalid fetch_strategy: {config.fetch_strategy!r}")

    existing_ids = load_existing_comment_ids(output_csv)

    # Determine concurrent workers
    if config.concurrent_workers is None:
        concurrent_workers = min(10, len(api_keys) * 3)
    else:
        concurrent_workers = config.concurrent_workers

    with RegsGovClient(api_keys, retries=config.retries) as client:
        all_comments = run_stratified_pipeline(
            df_docs, client, existing_ids, config.retries,
            config.max_comments_per_doc, concurrent_workers,
            sample_docs=sample_docs,
        )

    write_comment_output(all_comments, output_csv)


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Mine comments from regulations.gov with multi-key load balancing"
    )
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of documents")
    parser.add_argument("--max-comments-per-doc", type=int, default=None,
                        help="Skip documents with more than this many comments")
    parser.add_argument("--retries", type=int, default=10,
                        help="Max retries for API calls")
    parser.add_argument("--concurrent-workers", type=int, default=None,
                        help="Number of concurrent workers")
    parser.add_argument("--test", action="store_true",
                        help="Test mode: limit to 50 documents")
    parser.add_argument("--fetch-strategy", type=str, default="all",
                        choices=["all", "stratified"],
                        help="Sampling strategy. Default: all")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    setup_logging(verbose=args.verbose, quiet=args.quiet, year=args.year)

    # --test sets default limit of 50; --limit can override
    if args.test and args.limit is None:
        limit_docs = 50
    else:
        limit_docs = args.limit

    config = PipelineConfig(
        year=args.year,
        limit_docs=limit_docs,
        max_comments_per_doc=args.max_comments_per_doc,
        retries=args.retries,
        concurrent_workers=args.concurrent_workers,
        fetch_strategy=args.fetch_strategy,
        verbose=args.verbose,
        quiet=args.quiet,
    )
    
    try:
        mine_comments_for_year(config)
        return 0
    except Exception as e:
        logger.error(f"Mining failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
