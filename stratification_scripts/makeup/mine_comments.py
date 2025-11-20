"""Mine comments from regulations.gov for Federal Register documents.

Reads federal_register_YYYY_comments.csv and extracts all comments via regs.gov API.
Outputs to data/comments_raw_YYYY.csv with Polars with full 22-column metadata.

Uses two-stage mining:
1. Collect comment IDs via list endpoint
2. Fetch full details via detail endpoint with smart sampling
"""
import argparse
import os
import random
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Any

import polars as pl
import requests
from tqdm import tqdm


# API endpoints
REGS_DOC_URL = "https://api.regulations.gov/v4/documents/{documentId}"
REGS_COMMENTS_URL = "https://api.regulations.gov/v4/comments"
REGS_COMMENT_DETAIL_URL = "https://api.regulations.gov/v4/comments/{commentId}"
FR_DOC_DETAIL_URL = "https://www.federalregister.gov/api/v1/documents/{document_number}.json"


@dataclass
class RegsThrottle:
    """Rate limiter for Regulations.gov API calls."""
    rpm: int = 30
    last_call_ts: float = 0.0

    @property
    def min_interval(self) -> float:
        return 60.0 / self.rpm if self.rpm and self.rpm > 0 else 0.0

    def sleep_if_needed(self) -> None:
        if self.min_interval <= 0:
            return
        jitter = random.uniform(-0.05, 0.05) * self.min_interval
        wait = (self.min_interval + jitter) - (time.time() - self.last_call_ts)
        if wait > 0:
            time.sleep(wait)
        self.last_call_ts = time.time()


def calculate_sample_size(population_size: int, confidence: float = 0.95, margin: float = 0.05) -> int:
    """Calculate statistical sample size for proportion estimation with finite population correction.
    
    Args:
        population_size: Total population size
        confidence: Confidence level (0.95 for 95% CI, 0.99 for 99% CI)
        margin: Margin of error (0.05 for ±5%)
    
    Returns:
        Required sample size (always returns at least population_size for small populations)
    """
    if population_size <= 0:
        return 0  # No population
    
    # For very small populations, always sample ALL to ensure coverage
    if population_size < 10:
        return population_size  # 100% sampling for tiny strata
    
    # Z-score for confidence level
    Z = 1.96 if confidence == 0.95 else 2.576  # 95% or 99%
    p = 0.5  # Maximum variance assumption (most conservative)
    E = margin
    
    # Calculate sample size for infinite population
    n_infinite = (Z**2 * p * (1 - p)) / (E**2)
    
    # Apply finite population correction
    n_corrected = n_infinite / (1 + (n_infinite - 1) / population_size)
    
    # Always fetch all if population is smaller than required sample
    return min(int(n_corrected) + 1, population_size)


def build_stratified_sample_plan(
    df_docs: pl.DataFrame,
    all_comment_ids: Dict[str, List[str]],
) -> Tuple[set, pl.DataFrame]:
    """Build stratified sampling plan across (agency × comment_bin).
    
    Creates strata based on agency and comment count bins, calculates statistical
    sample size for each stratum, and returns the set of comment IDs to fetch.
    
    Args:
        df_docs: DataFrame with columns [document_number, agency, comment_count]
        all_comment_ids: Dict mapping document_number -> list of comment IDs
    
    Returns:
        - Set of comment IDs to fetch
        - Stratification summary DataFrame
    """
    print("\n" + "="*60)
    print("STRATIFIED SAMPLING PLAN")
    print("="*60)
    
    # Assign comment bins to each document
    print("\nAssigning documents to comment bins...")
    df_docs = df_docs.with_columns([
        pl.when(pl.col("comment_count") <= 10).then(pl.lit("0-10"))
          .when(pl.col("comment_count") <= 100).then(pl.lit("11-100"))
          .when(pl.col("comment_count") <= 1000).then(pl.lit("101-1000"))
          .when(pl.col("comment_count") <= 10000).then(pl.lit("1001-10000"))
          .otherwise(pl.lit("10000+"))
          .alias("comment_bin")
    ])
    
    # Group by strata (agency × comment_bin)
    print("Grouping into strata by agency × comment_bin...")
    strata = df_docs.group_by(["agency", "comment_bin"]).agg([
        pl.col("document_number").alias("doc_numbers"),
        pl.col("comment_count").sum().alias("total_comments"),
        pl.col("document_number").count().alias("num_docs"),
    ])
    
    # Calculate sample sizes for each stratum
    print("Calculating sample sizes per stratum (95% CI, ±5% margin)...")
    strata = strata.with_columns([
        pl.col("total_comments").map_elements(
            lambda x: calculate_sample_size(x, confidence=0.95, margin=0.05),
            return_dtype=pl.Int64
        ).alias("sample_size")
    ])
    
    # Sort for readable display
    strata = strata.sort(["agency", "comment_bin"])
    
    # Display summary
    print("\nStrata summary:")
    print(strata)
    
    # Build fetch list by pooling IDs from each stratum
    ids_to_fetch = set()
    sampled_strata = 0
    full_strata = 0
    
    print("\nBuilding fetch list per stratum:")
    for row in tqdm(strata.iter_rows(named=True), total=len(strata), desc="Stratifying", unit="stratum"):
        agency = row["agency"]
        comment_bin = row["comment_bin"]
        doc_numbers = row["doc_numbers"]
        sample_size = row["sample_size"]
        total_comments = row["total_comments"]
        num_docs = row["num_docs"]
        
        # Pool all comment IDs from documents in this stratum
        pooled_ids = []
        for doc_num in doc_numbers:
            pooled_ids.extend(all_comment_ids.get(doc_num, []))
        
        # Handle edge case: fewer IDs than expected
        if len(pooled_ids) < total_comments * 0.9:  # Allow 10% discrepancy
            print(f"  WARN: {agency:30s} / {comment_bin:12s} - expected ~{total_comments} IDs, found {len(pooled_ids)}")
        
        # Sample or take all
        if len(pooled_ids) <= sample_size:
            # Fetch all comments in this stratum
            ids_to_fetch.update(pooled_ids)
            full_strata += 1
            action = "ALL"
            pct = "100.0"
        else:
            # Random sample from pooled IDs
            sampled = random.sample(pooled_ids, sample_size)
            ids_to_fetch.update(sampled)
            sampled_strata += 1
            action = f"SAMPLE {sample_size:,}/{len(pooled_ids):,}"
            pct = f"{100*sample_size/len(pooled_ids):.1f}"
        
        print(f"  {action:20s}: {agency:30s} / {comment_bin:12s} - {num_docs:4d} docs, {pct:>5s}%")
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"Stratification complete:")
    print(f"  Total strata: {len(strata)}")
    print(f"  Full sampling (100%): {full_strata}")
    print(f"  Partial sampling: {sampled_strata}")
    print(f"  Total population: {strata['total_comments'].sum():,} comments")
    print(f"  Comments to fetch: {len(ids_to_fetch):,}")
    
    total_pop = strata['total_comments'].sum()
    if total_pop > 0:
        print(f"  Sampling ratio: {100*len(ids_to_fetch)/total_pop:.1f}%")
    
    print(f"{'='*60}\n")
    
    return ids_to_fetch, strata


def download_and_extract_pdf_text(
    pdf_url: str,
    headers: Dict[str, str],
    throttle: RegsThrottle,
    max_pages: int = 2,
) -> Optional[str]:
    """Download PDF from URL and extract text from first N pages using pdftotext.
    
    Returns extracted text or None if extraction fails.
    """
    if not pdf_url:
        return None
    
    try:
        # Download PDF
        throttle.sleep_if_needed()
        response = requests.get(pdf_url, headers=headers, timeout=30)
        if response.status_code != 200:
            return None
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as pdf_file:
            pdf_file.write(response.content)
            pdf_path = pdf_file.name
        
        # Extract text from first N pages
        txt_path = pdf_path.replace('.pdf', '.txt')
        try:
            result = subprocess.run(
                ['pdftotext', '-f', '1', '-l', str(max_pages), pdf_path, txt_path],
                capture_output=True,
                timeout=30,
            )
            
            if result.returncode == 0 and os.path.exists(txt_path):
                with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read().strip()
                os.unlink(txt_path)
                os.unlink(pdf_path)
                return text if text else None
            else:
                # pdftotext failed
                if os.path.exists(txt_path):
                    os.unlink(txt_path)
                os.unlink(pdf_path)
                return None
                
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            # pdftotext not installed or timeout
            if os.path.exists(txt_path):
                os.unlink(txt_path)
            os.unlink(pdf_path)
            return None
            
    except Exception as e:
        # Download or file I/O error
        return None


def get_object_id_for_regs_document(
    regs_document_id: Optional[str],
    headers: Dict[str, str],
    throttle: RegsThrottle,
    retries: int,
) -> Optional[str]:
    """Fetch objectId from a regs.gov document."""
    if not regs_document_id:
        return None
    backoff = 1.0
    for _ in range(max(1, retries)):
        try:
            throttle.sleep_if_needed()
            r = requests.get(
                REGS_DOC_URL.format(documentId=regs_document_id),
                headers=headers,
                timeout=20,
            )
        except requests.RequestException:
            time.sleep(backoff)
            backoff = min(backoff * 2, 16)
            continue
        if r.status_code == 200:
            data = r.json() or {}
            attrs = (data.get("data") or {}).get("attributes") or {}
            obj_id = attrs.get("objectId") or attrs.get("objectID")
            if obj_id:
                return obj_id
            return None
        if r.status_code in (401, 403):
            tqdm.write("ERROR: 401/403 from Regulations.gov; check REGS_API_KEY")
            return None
        if r.status_code == 429:
            try:
                retry_after = int(r.headers.get("Retry-After", "0"))
                if retry_after > 0:
                    time.sleep(min(retry_after, 60))
            except Exception:
                pass
            time.sleep(backoff)
            backoff = min(backoff * 2, 16)
            continue
        if r.status_code in (500, 502, 503, 504):
            time.sleep(backoff)
            backoff = min(backoff * 2, 16)
            continue
        return None
    return None


def get_regs_document_id_via_fr(document_number: Optional[str], retries: int) -> Optional[str]:
    """Fetch regs_document_id from Federal Register API."""
    if not document_number:
        return None
    backoff = 1.0
    for _ in range(max(1, retries)):
        try:
            r = requests.get(FR_DOC_DETAIL_URL.format(document_number=document_number), timeout=20)
        except requests.RequestException:
            time.sleep(backoff)
            backoff = min(backoff * 2, 16)
            continue
        if r.status_code == 200:
            j = r.json() or {}
            fr_info = j.get("regulations_dot_gov_info") or {}
            if isinstance(fr_info, dict):
                doc_id = fr_info.get("document_id") or fr_info.get("documentId")
                if doc_id:
                    return doc_id
            return None
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(backoff)
            backoff = min(backoff * 2, 16)
            continue
        return None
    return None


def iter_comments_for_object(
    object_id: str,
    headers: Dict[str, str],
    throttle: RegsThrottle,
    retries: int,
    limit_comments: Optional[int] = None,
) -> Iterable[dict]:
    """Yield comment records with windowed paging to bypass 5k cap."""
    if not object_id:
        return

    fetched = 0
    cursor_ge: Optional[str] = None
    while True:
        page = 1
        last_seen_ts: Optional[str] = None
        while page <= 20:
            params = {
                "filter[commentOnId]": object_id,
                "page[size]": 250,
                "page[number]": page,
                "sort": "lastModifiedDate,documentId",
            }
            if cursor_ge:
                params["filter[lastModifiedDate][ge]"] = cursor_ge
            backoff = 1.0
            while True:
                try:
                    throttle.sleep_if_needed()
                    r = requests.get(REGS_COMMENTS_URL, headers=headers, params=params, timeout=30)
                except requests.RequestException:
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 16)
                    continue
                if r.status_code == 200:
                    break
                if r.status_code in (401, 403):
                    tqdm.write("ERROR: 401/403 from Regulations.gov /comments; check REGS_API_KEY")
                    return
                if r.status_code == 429:
                    try:
                        retry_after = int(r.headers.get("Retry-After", "0"))
                        if retry_after > 0:
                            time.sleep(min(retry_after, 60))
                    except Exception:
                        pass
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 16)
                    continue
                if r.status_code in (500, 502, 503, 504):
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 16)
                    continue
                return

            data = r.json() or {}
            items: List[dict] = data.get("data") or []
            if not items:
                break

            last_item = items[-1]
            attrs_last = last_item.get("attributes") or {}
            last_seen_ts = attrs_last.get("lastModifiedDate") or attrs_last.get("modifyDate")

            for item in items:
                yield item
                fetched += 1
                if limit_comments is not None and fetched >= limit_comments:
                    return

            page += 1

        if not last_seen_ts or cursor_ge == last_seen_ts:
            return
        cursor_ge = last_seen_ts


def fetch_comment_detail(
    comment_id: str,
    headers: Dict[str, str],
    throttle: RegsThrottle,
    retries: int,
    include_attachments: bool = True,
) -> Optional[dict]:
    """Fetch full comment detail from /v4/comments/{id} endpoint."""
    if not comment_id:
        return None
    
    params = {}
    if include_attachments:
        params["include"] = "attachments"
    
    backoff = 1.0
    for _ in range(max(1, retries)):
        try:
            throttle.sleep_if_needed()
            r = requests.get(
                REGS_COMMENT_DETAIL_URL.format(commentId=comment_id),
                headers=headers,
                params=params,
                timeout=20,
            )
        except requests.RequestException:
            time.sleep(backoff)
            backoff = min(backoff * 2, 16)
            continue
        
        if r.status_code == 200:
            return r.json()
        if r.status_code in (401, 403):
            tqdm.write("ERROR: 401/403 from Regulations.gov; check REGS_API_KEY")
            return None
        if r.status_code == 429:
            try:
                retry_after = int(r.headers.get("Retry-After", "0"))
                if retry_after > 0:
                    time.sleep(min(retry_after, 60))
            except Exception:
                pass
            time.sleep(backoff)
            backoff = min(backoff * 2, 16)
            continue
        if r.status_code in (500, 502, 503, 504):
            time.sleep(backoff)
            backoff = min(backoff * 2, 16)
            continue
        return None
    
    return None


def extract_comment_fields_from_detail(
    data: dict,
    headers: Dict[str, str],
    throttle: RegsThrottle,
) -> Dict[str, Any]:
    """Extract 23 fields from full comment detail response, including PDF attachment text."""
    if not isinstance(data, dict):
        return {}
    
    comment_data = data.get("data") or {}
    attrs = comment_data.get("attributes") or {}
    included = data.get("included") or []
    
    # Core fields
    comment_id = comment_data.get("id")
    # Try multiple field names for comment text
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
    
    # Attachment info
    has_attachments = len(included) > 0
    attachment_count = len(included)
    attachment_formats = []
    attachment_texts = []
    
    for att in included:
        if isinstance(att, dict):
            att_attrs = att.get("attributes") or {}
            att_links = att.get("links") or {}
            
            # Extract format
            fmt = att_attrs.get("format") or att_attrs.get("fileFormats")
            if fmt:
                if isinstance(fmt, list):
                    attachment_formats.extend([str(f) for f in fmt])
                elif isinstance(fmt, dict):
                    attachment_formats.append(str(fmt.get("name") or fmt.get("type") or "unknown"))
                else:
                    attachment_formats.append(str(fmt))
            
            # Extract text from PDF attachments
            # Check if this is a PDF attachment
            is_pdf = False
            if fmt:
                fmt_str = str(fmt).lower()
                is_pdf = 'pdf' in fmt_str
            
            if is_pdf:
                # Get PDF URL from links
                pdf_url = att_links.get("self") or att_attrs.get("fileUrl")
                if pdf_url:
                    extracted_text = download_and_extract_pdf_text(pdf_url, headers, throttle)
                    if extracted_text:
                        attachment_texts.append(extracted_text)
    
    attachment_formats_str = ",".join(set(attachment_formats)) if attachment_formats else None
    attachment_text = "\n\n---\n\n".join(attachment_texts) if attachment_texts else None
    
    # Other metadata
    duplicate_comments = attrs.get("numItemsRecieved") or attrs.get("duplicateComments")
    page_count = attrs.get("pageCount")
    
    return {
        "comment_id": comment_id,
        "comment_text": str(comment_text or ""),
        "posted_date": posted_date,
        "receive_date": receive_date,
        "postmark_date": postmark_date,
        "first_name": first_name,
        "last_name": last_name,
        "organization": organization,
        "submitter_type": submitter_type,
        "city": city,
        "state_province_region": state_province_region,
        "country": country,
        "zip": zip_code,
        "gov_agency": gov_agency,
        "gov_agency_type": gov_agency_type,
        "has_attachments": has_attachments,
        "attachment_count": attachment_count,
        "attachment_formats": attachment_formats_str,
        "attachment_text": attachment_text,
        "duplicate_comments": duplicate_comments,
        "page_count": page_count,
    }


def should_sample_comments(comment_count: int, fetch_strategy: str) -> Tuple[bool, float]:
    """Decide whether to sample and what fraction based on strategy.
    
    Returns (should_sample, sample_rate)
    """
    if fetch_strategy == "all":
        return False, 1.0
    elif fetch_strategy == "sample":
        # Always sample 10%
        return True, 0.10
    elif fetch_strategy == "smart":
        # Fetch all if ≤1000, sample 10% if >1000, cap at 5000
        if comment_count <= 1000:
            return False, 1.0
        else:
            # Sample to get ~10% but max 5000
            target_sample = min(5000, int(comment_count * 0.10))
            sample_rate = target_sample / comment_count
            return True, sample_rate
    else:
        return False, 1.0


def mine_comments(
    fr_csv: Path,
    output_csv: Path,
    api_key: Optional[str],
    rpm: int,
    retries: int,
    limit_docs: Optional[int],
    fetch_strategy: str,
    max_comments_per_doc: Optional[int] = None,
) -> None:
    """Mine all comments from Federal Register documents with two-stage approach.
    
    Args:
        max_comments_per_doc: If set, skip documents with more comments than this limit
    """
    
    # Read FR documents
    if not fr_csv.exists():
        print(f"ERROR: {fr_csv} not found")
        return
    
    df_docs = pl.read_csv(str(fr_csv))
    
    # Filter to documents with regs_document_id AND comment_count > 0
    if "regs_document_id" in df_docs.columns and "comment_count" in df_docs.columns:
        df_docs = df_docs.filter(
            (pl.col("regs_document_id").is_not_null()) & (pl.col("comment_count") > 0)
        )
        print(f"Found {len(df_docs)} documents with comments > 0")
    elif "regs_document_id" in df_docs.columns:
        df_docs = df_docs.filter(pl.col("regs_document_id").is_not_null())
    else:
        print("ERROR: CSV missing regs_document_id column")
        return
    
    if limit_docs:
        df_docs = df_docs.head(limit_docs)
    
    print(f"Mining comments from {len(df_docs)} documents...")
    print(f"Fetch strategy: {fetch_strategy}")
    
    # Setup API
    headers = {"X-Api-Key": api_key} if api_key else {}
    throttle = RegsThrottle(rpm=rpm)
    
    # Load existing comments to avoid duplicates
    existing_ids = set()
    if output_csv.exists():
        df_existing = pl.read_csv(str(output_csv))
        existing_ids = set(df_existing["comment_id"].to_list())
        print(f"Loaded {len(existing_ids)} existing comments")
    
    # STRATIFIED SAMPLING PATH
    if fetch_strategy == "stratified":
        print("\n" + "="*60)
        print("STAGE 1: COLLECTING COMMENT IDs")
        print("="*60 + "\n")
        
        # Stage 1: Collect ALL comment IDs for ALL documents
        all_comment_ids = {}  # document_number -> [comment_id1, comment_id2, ...]
        doc_metadata = {}  # document_number -> {agency, comment_count, ...}
        
        for row in tqdm(df_docs.iter_rows(named=True), total=len(df_docs), desc="Collecting IDs"):
            doc_number = row.get("document_number")
            regs_doc_id = row.get("regs_document_id")
            agency = row.get("agency", "Unknown")
            comment_count = row.get("comment_count", 0)
            
            # Get regs_document_id if missing
            if not regs_doc_id and doc_number:
                regs_doc_id = get_regs_document_id_via_fr(doc_number, retries)
                if not regs_doc_id:
                    continue
            
            if not regs_doc_id:
                continue
            
            # Get objectId
            object_id = get_object_id_for_regs_document(regs_doc_id, headers, throttle, retries)
            if not object_id:
                continue
            
            # Collect comment IDs from list endpoint
            comment_ids = []
            limit_for_doc = max_comments_per_doc if max_comments_per_doc else None
            
            try:
                for item in iter_comments_for_object(object_id, headers, throttle, retries, limit_comments=limit_for_doc):
                    if not isinstance(item, dict):
                        continue
                    cid = item.get("id") or item.get("commentId")
                    if cid and cid not in existing_ids:
                        comment_ids.append(cid)
            except Exception as e:
                tqdm.write(f"  ERROR collecting IDs for {doc_number}: {e}")
                continue
            
            if not comment_ids:
                continue
            
            # Skip documents with too many comments if limit is set
            if max_comments_per_doc and len(comment_ids) > max_comments_per_doc:
                tqdm.write(f"  {doc_number}: Skipping (has {len(comment_ids)} comments, limit is {max_comments_per_doc})")
                continue
            
            # Store IDs and metadata
            all_comment_ids[doc_number] = comment_ids
            doc_metadata[doc_number] = {
                "agency": agency,
                "comment_count": len(comment_ids),
            }
            
            tqdm.write(f"  {doc_number}: Collected {len(comment_ids)} comment IDs (agency: {agency})")
        
        print(f"\nStage 1 complete: Collected {sum(len(ids) for ids in all_comment_ids.values()):,} comment IDs from {len(all_comment_ids)} documents")
        
        # Build DataFrame for stratification
        strat_data = []
        for doc_num, ids in all_comment_ids.items():
            meta = doc_metadata[doc_num]
            strat_data.append({
                "document_number": doc_num,
                "agency": meta["agency"],
                "comment_count": meta["comment_count"],
            })
        
        if not strat_data:
            print("No documents with comments found")
            return
        
        df_for_stratification = pl.DataFrame(strat_data)
        
        # Stage 2: Stratified sampling
        print("\n" + "="*60)
        print("STAGE 2: STRATIFIED SAMPLING")
        print("="*60 + "\n")
        
        ids_to_fetch, strata_summary = build_stratified_sample_plan(df_for_stratification, all_comment_ids)
        
        if not ids_to_fetch:
            print("No comments selected for fetching")
            return
        
        # Create reverse mapping: comment_id -> document_number
        comment_to_doc = {}
        for doc_num, ids in all_comment_ids.items():
            for cid in ids:
                if cid in ids_to_fetch:
                    comment_to_doc[cid] = doc_num
        
        # Stage 3: Fetch details for selected comments
        print("\n" + "="*60)
        print("STAGE 3: FETCHING COMMENT DETAILS")
        print("="*60 + "\n")
        
        all_comments = []
        failed_fetches = 0
        
        for comment_id in tqdm(ids_to_fetch, desc="Fetching details", unit="comment"):
            try:
                detail_data = fetch_comment_detail(comment_id, headers, throttle, retries)
                if not detail_data:
                    failed_fetches += 1
                    continue
                
                fields = extract_comment_fields_from_detail(detail_data, headers, throttle)
                if not fields or not fields.get("comment_id"):
                    failed_fetches += 1
                    continue
                
                # Add document_number
                doc_num = comment_to_doc.get(comment_id, "Unknown")
                fields["document_number"] = doc_num
                all_comments.append(fields)
                existing_ids.add(comment_id)
                
            except Exception as e:
                tqdm.write(f"  ERROR fetching {comment_id}: {e}")
                failed_fetches += 1
                continue
        
        print(f"\nStage 3 complete: Fetched {len(all_comments):,} comments ({failed_fetches} failed)")
    
    # LEGACY SAMPLING PATHS (smart/all/sample)
    else:
        all_comments = []
        
        for row in tqdm(df_docs.iter_rows(named=True), total=len(df_docs), desc="Documents"):
            doc_number = row.get("document_number")
            regs_doc_id = row.get("regs_document_id")
            comment_count = row.get("comment_count", 0)
            
            # Get regs_document_id if missing
            if not regs_doc_id and doc_number:
                regs_doc_id = get_regs_document_id_via_fr(doc_number, retries)
                if not regs_doc_id:
                    continue
            
            if not regs_doc_id:
                continue
            
            # Get objectId
            object_id = get_object_id_for_regs_document(regs_doc_id, headers, throttle, retries)
            if not object_id:
                continue
            
            # Stage 1: Collect comment IDs from list endpoint
            comment_ids = []
            # Use max_comments_per_doc as limit if specified
            limit_for_doc = max_comments_per_doc if max_comments_per_doc else None
            for item in iter_comments_for_object(object_id, headers, throttle, retries, limit_comments=limit_for_doc):
                if not isinstance(item, dict):
                    continue
                cid = item.get("id") or item.get("commentId")
                if cid and cid not in existing_ids:
                    comment_ids.append(cid)
            
            if not comment_ids:
                continue
            
            # Skip documents with too many comments if limit is set
            if max_comments_per_doc and len(comment_ids) > max_comments_per_doc:
                tqdm.write(f"  {doc_number}: Skipping (has {len(comment_ids)} comments, limit is {max_comments_per_doc})")
                continue
            
            # Stage 2: Decide sampling strategy
            should_sample, sample_rate = should_sample_comments(len(comment_ids), fetch_strategy)
            
            if should_sample:
                n_sample = int(len(comment_ids) * sample_rate)
                n_sample = max(1, min(n_sample, len(comment_ids)))
                sampled_ids = random.sample(comment_ids, n_sample)
                tqdm.write(f"  {doc_number}: Sampling {n_sample}/{len(comment_ids)} comments ({sample_rate*100:.1f}%)")
            else:
                sampled_ids = comment_ids
                tqdm.write(f"  {doc_number}: Fetching all {len(comment_ids)} comments")
            
            # Stage 3: Fetch full details for selected comments
            for comment_id in sampled_ids:
                detail_data = fetch_comment_detail(comment_id, headers, throttle, retries)
                if not detail_data:
                    continue
                
                fields = extract_comment_fields_from_detail(detail_data, headers, throttle)
                if not fields or not fields.get("comment_id"):
                    continue
                
                # Add document_number
                fields["document_number"] = doc_number
                all_comments.append(fields)
                existing_ids.add(comment_id)
    
    if not all_comments:
        print("No new comments found")
        return
    
    # Write to CSV with all 22 columns
    # Use large infer_schema_length to handle varying field types across all rows
    df_new = pl.DataFrame(all_comments, infer_schema_length=len(all_comments))
    
    # Reorder columns to put document_number first
    column_order = [
        "document_number",
        "comment_id",
        "comment_text",
        "posted_date",
        "receive_date",
        "postmark_date",
        "first_name",
        "last_name",
        "organization",
        "submitter_type",
        "city",
        "state_province_region",
        "country",
        "zip",
        "gov_agency",
        "gov_agency_type",
        "has_attachments",
        "attachment_count",
        "attachment_formats",
        "attachment_text",
        "duplicate_comments",
        "page_count",
    ]
    
    # Keep only columns that exist
    existing_columns = [c for c in column_order if c in df_new.columns]
    df_new = df_new.select(existing_columns)
    
    if output_csv.exists():
        df_existing = pl.read_csv(str(output_csv))
        
        # Ensure both DataFrames have same columns with proper types
        for col in df_new.columns:
            if col not in df_existing.columns:
                # Add missing column to existing with same type as new
                df_existing = df_existing.with_columns(pl.lit(None).cast(df_new[col].dtype).alias(col))
            else:
                # Column exists in both - ensure types match
                if df_existing[col].dtype != df_new[col].dtype:
                    # Cast existing to match new (new data schema is authoritative)
                    try:
                        df_existing = df_existing.with_columns(pl.col(col).cast(df_new[col].dtype))
                    except Exception as e:
                        print(f"  WARNING: Could not cast {col} from {df_existing[col].dtype} to {df_new[col].dtype}: {e}")
                        # Fallback: cast new to match existing
                        df_new = df_new.with_columns(pl.col(col).cast(df_existing[col].dtype))
        
        for col in df_existing.columns:
            if col not in df_new.columns:
                # Add missing column to new with same type as existing
                df_new = df_new.with_columns(pl.lit(None).cast(df_existing[col].dtype).alias(col))
        
        # Reorder columns to match
        df_existing = df_existing.select(df_new.columns)
        df_combined = pl.concat([df_existing, df_new])
    else:
        df_combined = df_new
    
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_combined.write_csv(str(output_csv))
    
    print(f"\nMined {len(all_comments)} new comments")
    print(f"Total comments in database: {len(df_combined)}")
    print(f"Output: {output_csv}")
    
    # Print metadata statistics
    if len(df_new) > 0:
        print("\nMetadata coverage:")
        org_count = df_new.filter(pl.col("organization").is_not_null()).shape[0]
        print(f"  Organization: {org_count}/{len(df_new)} ({100*org_count/len(df_new):.1f}%)")
        name_count = df_new.filter(pl.col("first_name").is_not_null() | pl.col("last_name").is_not_null()).shape[0]
        print(f"  Name: {name_count}/{len(df_new)} ({100*name_count/len(df_new):.1f}%)")
        attach_count = df_new.filter(pl.col("has_attachments") == True).shape[0]
        print(f"  Attachments: {attach_count}/{len(df_new)} ({100*attach_count/len(df_new):.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine comments from regulations.gov")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of documents")
    parser.add_argument("--max-comments-per-doc", type=int, default=None, help="Skip documents with more than this many comments (cost control)")
    parser.add_argument("--rpm", type=int, default=80, help="Requests per minute")
    parser.add_argument("--retries", type=int, default=10, help="Max retries for API calls")
    parser.add_argument(
        "--fetch-strategy",
        type=str,
        default="stratified",
        choices=["all", "sample", "smart", "stratified"],
        help="Sampling strategy: stratified (agency×comment_bin power calc), smart (<=1000 all else 10%%), all, sample (10%%)",
    )
    args = parser.parse_args()
    
    # Setup paths
    script_dir = Path(__file__).parent
    fr_csv = script_dir.parent / "output" / f"federal_register_{args.year}_comments.csv"
    output_csv = script_dir / "data" / f"comments_raw_{args.year}.csv"
    
    # Get API key from environment
    api_key = os.environ.get("REGS_API_KEY")
    if not api_key:
        print("WARNING: REGS_API_KEY not set, API calls may be rate limited")
    
    mine_comments(fr_csv, output_csv, api_key, args.rpm, args.retries, args.limit, args.fetch_strategy, args.max_comments_per_doc)


if __name__ == "__main__":
    main()


# learnings from building the comment mining pipeline:
#
# api design:
# - using the detail endpoint instead of just list endpoint is crucial for getting full metadata
#   like organization, submitter_type, attachments, etc. the list endpoint only returns minimal
#   fields (id, comment text, dates)
# - the two-stage approach (collect IDs first, then fetch details selectively) is necessary
#   because fetching every detail on large dockets (10k+ comments) would take hours and burn
#   through API quota
# - always include ?include=attachments in detail endpoint to get attachment data in the
#   "included" array - this is required for pdf text extraction
#
# sampling strategies:
# - stratified sampling (default) bins by agency × comment_count and uses power calculation to
#   determine sample size per stratum (95% CI, ±5% margin = ~384 samples per stratum)
# - bins are 0-10, 11-100, 101-1k, 1k-10k, 10k+ to capture different docket sizes
# - CRITICAL: never skip small strata - if a stratum has <10 comments, we sample ALL of them to
#   ensure representative coverage across the distribution (that's the whole point of stratification)
# - stratified approach guarantees representative coverage across all agencies and prevents
#   mega-dockets from dominating the sample
# - typical sampling ratio is 50-70% of total population due to many small strata requiring
#   100% sampling (when population < 384)
# - smart sampling (legacy: fetch all for <=1000 comments, sample 10% otherwise) doesn't
#   guarantee cross-agency coverage but is faster for quick tests
# - --max-comments-per-doc is essential for cost control - skips documents with too many comments
#   before burning api quota and openai credits
# - use --limit to cap number of documents for testing/development
#
# rate limiting:
# - throttling is critical - regs.gov officially limits to 1000/hr but can handle up to 5000/hr
#   with API key (use --rpm 80 for 4800/hr to stay safe)
# - default is 80 rpm which gives ~15 hours for ID collection stage and reasonable detail fetch times
# - exponential backoff on 429/5xx is essential for reliability
# - rate limiting applies to both comment list calls AND attachment downloads
#
# attachment handling:
# - pdf attachments can be huge - only extract first 2 pages with pdftotext to keep token counts
#   low and processing fast
# - pdftotext command: pdftotext -f 1 -l 2 <input.pdf> <output.txt>
# - non-pdf attachments are skipped entirely - images, videos, word docs, etc don't add value
#   and are expensive to process
# - attachment urls come from included[].links.self or included[].attributes.fileUrl in the
#   detail response
# - format detection: check if 'pdf' in str(format).lower() to identify pdf attachments
# - always use temp files for pdf downloads and clean them up immediately after extraction
# - download failures are common (broken urls, access denied) - gracefully return None and
#   continue without crashing
#
# pagination workarounds:
# - the windowed pagination in iter_comments bypasses the 5000 result cap by using
#   lastModifiedDate filters to "restart" pagination windows
# - this technique is essential for dockets with >5000 comments
#
# data quality:
# - metadata fields are often null/missing so always have fallbacks
# - comment_text field name varies: can be "comment", "commentText", "comment_text", or "title"
# - attachment_text is stored alongside comment_text and used as fallback during classification
# - empty submissions (no text, no attachments, no metadata) should be excluded entirely rather
#   than marked as "undecided" - we want a high-quality sample not a population with noise
#
# polars gotchas:
# - polars is way faster than pandas for large CSVs but syntax takes getting used to
# - when concatenating dataframes with new columns, you need to explicitly add missing columns
#   with pl.lit(None).cast(dtype) to match schemas and cast existing columns to match new schema
# - use infer_schema_length=len(data) when creating dataframes from dicts to handle varying
#   field types across rows (e.g., some rows have strings, others have None)
# - use .iter_rows(named=True) to iterate as dicts for easier field access
#
# performance tips:
# - use persistent http session for connection pooling
# - cache results (existing_ids) to avoid duplicate fetches
# - incremental saves prevent data loss on crashes
# - subprocess.run with timeout prevents hanging on corrupted pdfs

