#!/usr/bin/env python3
"""
Track federal agency responses to public comments using Gemini API.

This module identifies whether agencies responded to public comments and
classifies their responses (accept/reject/uncertain) using Gemini 2.0 Flash
with thinking mode and Google Search grounding.

Example:
    # CLI usage
    $ python -m stratification_scripts.makeup.track_responses --year 2024
    
    # Programmatic usage
    >>> from stratification_scripts.makeup.track_responses import track_responses_for_year
    >>> from stratification_scripts.config import PipelineConfig
    >>> 
    >>> config = PipelineConfig(year=2024)
    >>> track_responses_for_year(config)
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

import polars as pl

from stratification_scripts.config import (
    PipelineConfig,
    get_gemini_api_key,
    get_xai_api_key,
    get_comments_raw_path,
    get_makeup_data_path,
    get_fr_csv_path,
    get_agency_responses_path,
)
from stratification_scripts.logging_utils import (
    get_logger,
    log_banner,
    setup_logging,
)
from stratification_scripts.gemini_client import GeminiResponseTracker
from stratification_scripts.openai_response_client import OpenAIResponseTracker
from stratification_scripts.xai_response_client import XAIResponseTracker
from stratification_scripts.io_utils import extract_pdf_text
from stratification_scripts.regulations_gov.client import RegsGovClient
from stratification_scripts.config import get_regs_api_keys
from stratification_scripts.makeup.mine_comments import calculate_sample_size

logger = get_logger(__name__)


def extract_full_comment_text(
    comment_row: dict,
    client: Optional[RegsGovClient] = None,
    max_pages: int = 30,
) -> str:
    """
    Extract full comment text, re-downloading attachments if needed.
    
    Logic:
    1. Get comment_text (always available)
    2. Check attachment_text:
       - If None or empty: return comment_text only
       - If short (< 2800 chars): use existing attachment_text
       - If long or appears truncated: re-download PDF with max_pages=30
    3. Combine comment_text + full_attachment_text
    
    Args:
        comment_row: Dict with comment data (comment_text, attachment_text, etc.)
        client: Optional RegsGovClient for re-downloading attachments
        max_pages: Maximum pages to extract from PDFs
    
    Returns:
        Combined comment text string
    """
    comment_text = comment_row.get("comment_text", "")
    attachment_text = comment_row.get("attachment_text", "")
    
    # Always start with comment_text
    parts = []
    if comment_text and str(comment_text).strip() and str(comment_text).strip().lower() != "none":
        parts.append(str(comment_text).strip())
    
    # Handle attachment text
    if attachment_text and str(attachment_text).strip() and str(attachment_text).strip().lower() != "none":
        attachment_str = str(attachment_text).strip()
        
        # If attachment text is long (likely truncated at 2 pages), try to re-download
        # Threshold: 2800 chars suggests it might be truncated
        if len(attachment_str) > 2800 and client is not None:
            # Try to re-download with more pages
            comment_id = comment_row.get("comment_id")
            if comment_id:
                try:
                    logger.debug(f"Re-downloading attachment for {comment_id} with max_pages={max_pages}")
                    detail_data = client.fetch_comment_detail(comment_id, include_attachments=True)
                    
                    if detail_data:
                        included = detail_data.get("included") or []
                        new_attachment_texts = []
                        
                        for att in included:
                            if not isinstance(att, dict):
                                continue
                            
                            att_attrs = att.get("attributes") or {}
                            att_links = att.get("links") or {}
                            file_formats = att_attrs.get("fileFormats", [])
                            
                            # Check for PDF
                            is_pdf = False
                            pdf_url = None
                            
                            if isinstance(file_formats, list):
                                for ff in file_formats:
                                    if isinstance(ff, dict):
                                        fmt = str(ff.get("format", "")).lower()
                                        if "pdf" in fmt:
                                            is_pdf = True
                                            pdf_url = ff.get("fileUrl")
                                            break
                            
                            if not is_pdf and att_attrs.get("format"):
                                if "pdf" in str(att_attrs.get("format", "")).lower():
                                    is_pdf = True
                                    pdf_url = att_attrs.get("fileUrl")
                            
                            if not is_pdf:
                                continue
                            
                            if not pdf_url:
                                pdf_url = att_links.get("self")
                            
                            if not pdf_url:
                                continue
                            
                            # Download and extract with max_pages
                            pdf_bytes = client.download_bytes(pdf_url, timeout=30)
                            if pdf_bytes:
                                extracted = extract_pdf_text(pdf_bytes, max_pages=max_pages)
                                if extracted:
                                    new_attachment_texts.append(extracted)
                        
                        if new_attachment_texts:
                            # Use newly extracted text
                            parts.append("\n\n---\n\n".join(new_attachment_texts))
                        else:
                            # Fallback to existing
                            parts.append(attachment_str)
                    else:
                        # Fallback to existing
                        parts.append(attachment_str)
                        
                except Exception as e:
                    logger.debug(f"Failed to re-download attachment for {comment_id}: {e}")
                    # Fallback to existing attachment text
                    parts.append(attachment_str)
            else:
                # No comment_id, use existing
                parts.append(attachment_str)
        else:
            # Attachment is short enough, use existing
            parts.append(attachment_str)
    
    return "\n\n".join(parts) if parts else ""


def strip_error_rows(responses_csv: Path) -> int:
    """
    Remove rows with API errors from the responses CSV so they get reprocessed.

    Identifies error rows by checking the 'reasoning' column for known error
    patterns (API errors, retries exhausted, empty responses, validation errors).

    Args:
        responses_csv: Path to agency_responses CSV.

    Returns:
        Number of error rows removed.
    """
    if not responses_csv.exists():
        return 0

    try:
        df = pl.read_csv(str(responses_csv), infer_schema_length=None)
    except Exception as e:
        logger.warning(f"Failed to read responses CSV for error stripping: {e}")
        return 0

    if "reasoning" not in df.columns:
        return 0

    original_count = len(df)

    # Identify error rows by reasoning patterns
    error_mask = (
        pl.col("reasoning").str.starts_with("API error:")
        | pl.col("reasoning").str.starts_with("API retries exhausted")
        | pl.col("reasoning").str.starts_with("Empty model response")
        | pl.col("reasoning").str.starts_with("VALIDATION_ERROR:")
    )

    df_errors = df.filter(error_mask)
    error_count = len(df_errors)

    if error_count == 0:
        logger.info("No error rows found in responses CSV")
        return 0

    # Backup before modifying
    import shutil
    backup_path = responses_csv.with_suffix(".pre_retry_backup.csv")
    shutil.copy2(responses_csv, backup_path)
    logger.info(f"Backed up {original_count} rows to {backup_path.name}")

    # Keep only non-error rows
    df_clean = df.filter(~error_mask)
    df_clean.write_csv(str(responses_csv))
    logger.info(
        f"Stripped {error_count} error rows from responses CSV "
        f"({len(df_clean)} rows remaining). These comments will be reprocessed."
    )

    return error_count


def load_existing_responses(responses_csv: Path) -> Set[str]:
    """Load existing response tracking results and return processed comment IDs."""
    if not responses_csv.exists():
        return set()

    try:
        df_existing = pl.read_csv(str(responses_csv), infer_schema_length=None)
        processed_ids = set(df_existing["comment_id"].to_list())
        logger.info(f"Loaded {len(processed_ids)} already-processed comments")
        return processed_ids
    except Exception as e:
        logger.warning(f"Failed to load existing responses: {e}")
        return set()


def sample_comments_for_response_tracking(
    df_unprocessed: pl.DataFrame,
    census_threshold: int = 30,
    seed: Optional[int] = None,
) -> tuple[pl.DataFrame, Dict[str, float]]:
    """
    Stratified sample of comments for response tracking (Cochran + FPC).

    Stratifies by ``category`` (commenter type). Categories with N <= census_threshold
    are taken as census (weight=1.0). Larger categories use Cochran sample sizing.
    After sampling, guarantees at least 1 comment per document_number.

    Args:
        df_unprocessed: DataFrame of comments to sample from (must have
            ``comment_id``, ``category``, ``document_number`` columns).
        census_threshold: Take all if stratum N is at or below this value.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (sampled DataFrame, weight_map ``{comment_id: weight}``).
    """
    import random as _random

    weight_map: Dict[str, float] = {}
    sampled_frames: list[pl.DataFrame] = []

    # Fallback category for rows missing a category value
    FALLBACK_CAT = "UNKNOWN"

    df_work = df_unprocessed.with_columns(
        pl.col("category").fill_null(FALLBACK_CAT).alias("category")
    )

    categories = df_work["category"].unique().to_list()

    rng = _random.Random(seed)

    for cat in sorted(categories):
        stratum = df_work.filter(pl.col("category") == cat)
        N = len(stratum)

        if N <= census_threshold:
            # Census — take all
            sampled_frames.append(stratum)
            for cid in stratum["comment_id"].to_list():
                weight_map[str(cid)] = 1.0
            logger.info(f"  Sampling '{cat}': N={N} <= {census_threshold} -> census (weight=1.0)")
        else:
            n = calculate_sample_size(N)
            n = min(n, N)
            weight = N / n

            # Sample n rows
            indices = list(range(N))
            rng.shuffle(indices)
            selected_indices = sorted(indices[:n])
            stratum_sampled = stratum[selected_indices]

            sampled_frames.append(stratum_sampled)
            for cid in stratum_sampled["comment_id"].to_list():
                weight_map[str(cid)] = weight
            logger.info(f"  Sampling '{cat}': N={N} -> n={n} (weight={weight:.2f})")

    if sampled_frames:
        df_sampled = pl.concat(sampled_frames)
    else:
        df_sampled = df_unprocessed.head(0)

    # Document coverage guarantee: force-include 1 comment per doc if missing
    if "document_number" in df_unprocessed.columns:
        sampled_docs = set(df_sampled["document_number"].to_list())
        all_docs = set(df_unprocessed["document_number"].to_list())
        missing_docs = all_docs - sampled_docs

        if missing_docs:
            force_rows: list[pl.DataFrame] = []
            for doc_num in missing_docs:
                doc_rows = df_unprocessed.filter(pl.col("document_number") == doc_num)
                if len(doc_rows) > 0:
                    row = doc_rows.head(1)
                    force_rows.append(row)
                    cid = str(row["comment_id"][0])
                    # Weight = total comments for this doc / 1 (single force-include)
                    weight_map[cid] = float(len(doc_rows))

            if force_rows:
                df_force = pl.concat(force_rows)
                df_sampled = pl.concat([df_sampled, df_force])
                logger.info(f"  Force-included {len(force_rows)} comments for document coverage ({len(missing_docs)} docs)")

    total_N = len(df_unprocessed)
    total_n = len(df_sampled)
    logger.info(f"Sampling complete: {total_N} -> {total_n} comments ({total_n/total_N*100:.1f}% of total)")

    return df_sampled, weight_map


def propagate_responses_to_duplicates(
    results: List[Dict],
    df_comments: pl.DataFrame,
) -> List[Dict]:
    """
    Propagate response tracking results to duplicate comments.
    
    For each canonical comment result, find all duplicates in the same group
    and create result entries for them with the same response tracking data.
    
    Args:
        results: List of result dicts with comment_id and response fields
        df_comments: DataFrame with duplicate_group_id and canonical_comment_id columns
    
    Returns:
        Expanded list of results including duplicates
    """
    if "duplicate_group_id" not in df_comments.columns:
        # No deduplication columns, return as-is
        return results
    
    # Build mapping from canonical_comment_id to result
    canonical_to_result = {}
    for r in results:
        comment_id = r.get("comment_id")
        if comment_id:
            canonical_to_result[str(comment_id)] = r
    
    # Find all duplicates that need results propagated
    expanded_results = list(results)  # Start with canonical results
    
    # Group by duplicate_group_id to find duplicates
    duplicate_groups = df_comments.filter(
        pl.col("duplicate_group_id").is_not_null() & 
        (pl.col("is_canonical") == False)
    )
    
    if len(duplicate_groups) > 0:
        for row in duplicate_groups.iter_rows(named=True):
            duplicate_id = str(row["comment_id"])
            canonical_id = str(row["canonical_comment_id"])
            
            # If canonical was processed, propagate to duplicate
            if canonical_id in canonical_to_result:
                canonical_result = canonical_to_result[canonical_id].copy()
                # Update comment_id to the duplicate's ID
                canonical_result["comment_id"] = duplicate_id
                expanded_results.append(canonical_result)
    
    return expanded_results


def save_responses_incremental(
    responses_csv: Path,
    new_results: List[Dict],
    df_comments: Optional[pl.DataFrame] = None,
) -> None:
    """
    Save response tracking results incrementally, optionally propagating to duplicates.
    
    Args:
        responses_csv: Path to output CSV
        new_results: List of result dicts to append
        df_comments: Optional DataFrame with comments (for duplicate propagation)
    """
    if not new_results:
        return
    
    # Propagate results to duplicates if comments DataFrame provided
    if df_comments is not None:
        new_results = propagate_responses_to_duplicates(new_results, df_comments)
    
    df_new = pl.DataFrame(new_results, infer_schema_length=None)

    if responses_csv.exists():
        try:
            df_existing = pl.read_csv(str(responses_csv), infer_schema_length=None)

            # Ensure schema compatibility — handle missing columns AND type mismatches
            for col in df_new.columns:
                if col not in df_existing.columns:
                    df_existing = df_existing.with_columns(
                        pl.lit(None).cast(df_new[col].dtype).alias(col)
                    )
                elif df_existing[col].dtype != df_new[col].dtype:
                    # Type mismatch: cast both to Utf8 (safest common type)
                    df_existing = df_existing.with_columns(pl.col(col).cast(pl.Utf8))
                    df_new = df_new.with_columns(pl.col(col).cast(pl.Utf8))

            for col in df_existing.columns:
                if col not in df_new.columns:
                    df_new = df_new.with_columns(
                        pl.lit(None).cast(df_existing[col].dtype).alias(col)
                    )

            # Align column order
            df_existing = df_existing.select(df_new.columns)
            df_combined = pl.concat([df_existing, df_new])
        except Exception as e:
            # Backup existing file before overwriting to prevent data loss
            backup_path = responses_csv.with_suffix(f".backup.csv")
            import shutil
            shutil.copy2(responses_csv, backup_path)
            logger.warning(
                f"Failed to merge with existing data: {e}. "
                f"Backed up existing file to {backup_path.name}, writing new data only."
            )
            df_combined = df_new
    else:
        df_combined = df_new
    
    responses_csv.parent.mkdir(parents=True, exist_ok=True)
    df_combined.write_csv(str(responses_csv))
    logger.info(f"Saved {len(new_results)} new responses (total: {len(df_combined)})")


async def process_responses_async(
    tracker,
    comments_to_process: List[Dict],
    responses_csv: Path,
    max_concurrency: int,
    regs_client: Optional[RegsGovClient] = None,
    max_comment_pages: int = 30,
    df_comments: Optional[pl.DataFrame] = None,
    batch_size: int = 1000,
    weight_map: Optional[Dict[str, float]] = None,
) -> None:
    """
    Process comments asynchronously in batches.

    Args:
        tracker: Response tracker instance (Gemini, OpenAI, or xAI)
        comments_to_process: List of comment dicts to process
        responses_csv: Path to save results
        max_concurrency: Max concurrent API calls
        regs_client: Optional RegsGovClient for re-downloading attachments
        max_comment_pages: Max pages to extract from PDFs
        df_comments: Optional DataFrame with comments (for duplicate propagation)
        batch_size: Number of comments to process per batch (default: 1000)
        weight_map: Optional dict mapping comment_id -> sampling weight
    """
    total_comments = len(comments_to_process)
    
    logger.info(f"Processing {total_comments} comments in batches of {batch_size}")
    
    for batch_start in range(0, total_comments, batch_size):
        batch_end = min(batch_start + batch_size, total_comments)
        batch = comments_to_process[batch_start:batch_end]
        
        logger.info(f"Processing batch {batch_start+1}-{batch_end} of {total_comments}")
        
        # Prepare batch for Gemini
        gemini_batch = []
        # Track full_text length per comment_id so we don't re-call
        # extract_full_comment_text() later without regs_client (which
        # would lose re-downloaded attachment text).
        comment_text_lengths: Dict[str, int] = {}
        for comment in batch:
            # Extract full comment text
            full_text = extract_full_comment_text(
                comment,
                client=regs_client,
                max_pages=max_comment_pages,
            )

            cid = str(comment.get("comment_id", "unknown"))
            comment_text_lengths[cid] = len(full_text)

            metadata = {
                "comment_id": cid,
                "document_number": str(comment.get("document_number", "N/A")),
                "agency": str(comment.get("agency", "N/A")),
                "commenter_type": str(comment.get("category", "N/A")),
                "submission_date": str(comment.get("posted_date", "N/A")),
                # Lifecycle tracking fields
                "lifecycle_stage": str(comment.get("lifecycle_stage", "UNKNOWN")),
                "rin": str(comment.get("rin", "N/A")),
            }

            gemini_batch.append((full_text, metadata))
        
        # Track responses
        results = await tracker.track_batch(gemini_batch, max_concurrency=max_concurrency)
        
        # Format results for CSV
        csv_results = []
        for comment_id, parsed_response, raw_response in results:
            # Find original comment to get metadata
            original_comment = next(
                (c for c in batch if str(c.get("comment_id")) == comment_id),
                {}
            )
            
            row = {
                "comment_id": comment_id,
                "document_number": str(original_comment.get("document_number") or "N/A"),
                "agency": str(original_comment.get("agency") or "N/A"),
                "response_found": parsed_response.get("response_found", "uncertain"),
                "agency_decision": parsed_response.get("agency_decision", "uncertain"),
                "response_text": parsed_response.get("response_text", "N/A"),
                "response_location": parsed_response.get("response_location", "N/A"),
                "reasoning": parsed_response.get("reasoning", "N/A"),
                "processed_at": datetime.now().isoformat(),
                "model": tracker.model,
                "comment_text_length": comment_text_lengths.get(comment_id, 0),
                "has_attachment": bool(original_comment.get("attachment_text")),
                # Lifecycle tracking fields
                "lifecycle_stage": str(original_comment.get("lifecycle_stage") or "UNKNOWN"),
                "rin": str(original_comment.get("rin") or "N/A"),
                # Sampling weight (1.0 if no sampling or census)
                "response_sample_weight": weight_map.get(comment_id, 1.0) if weight_map else 1.0,
            }
            csv_results.append(row)
        
        # Save incrementally (with duplicate propagation if df_comments provided)
        save_responses_incremental(responses_csv, csv_results, df_comments)


def track_responses_for_year(config: PipelineConfig, limit: Optional[int] = None, batch_size: int = 1000) -> None:
    """
    Track agency responses for a single year.
    
    This is the main entry point for programmatic use.
    
    Args:
        config: Pipeline configuration
        limit: Optional limit on number of comments to process (for testing)
        batch_size: Number of comments to process per batch (default: 1000)
    
    Side Effects:
        Makes API calls to Gemini
        Optionally makes API calls to Regulations.gov for re-downloading attachments
        Writes response tracking results to CSV
    """
    makeup_data_csv = get_makeup_data_path(config.year)
    comments_raw_csv = get_comments_raw_path(config.year)
    fr_csv = get_fr_csv_path(config.year)
    responses_csv = get_agency_responses_path(config.year)

    provider = getattr(config, "response_provider", "xai")

    # Always strip error rows so they get reprocessed
    stripped = strip_error_rows(responses_csv)
    if stripped > 0:
        logger.info(f"Retry-errors: {stripped} error rows will be reprocessed")

    # Get API key based on provider
    if provider == "xai":
        api_key = get_xai_api_key(required=True)
    elif provider == "openai":
        from stratification_scripts.config import get_openai_api_key
        api_key = get_openai_api_key(required=True)
    else:
        api_key = get_gemini_api_key(required=True)
    
    # Check input files
    if not makeup_data_csv.exists():
        logger.error(f"{makeup_data_csv} not found. Run classify step first.")
        return
    
    if not comments_raw_csv.exists():
        logger.error(f"{comments_raw_csv} not found. Run mine step first.")
        return
    
    log_banner(logger, f"TRACKING AGENCY RESPONSES - YEAR {config.year}")
    
    # Load makeup data (contains all comments including duplicates after classification propagation)
    logger.info(f"Loading makeup data from: {makeup_data_csv}")
    df_makeup = pl.read_csv(str(makeup_data_csv), infer_schema_length=None)
    logger.info(f"Loaded {len(df_makeup)} classified comments")

    # Join with comments_raw to get full metadata (including deduplication columns if present)
    logger.info(f"Loading raw comments from: {comments_raw_csv}")
    df_raw = pl.read_csv(str(comments_raw_csv), infer_schema_length=None, schema_overrides={"zip": pl.Utf8})

    # Check if deduplication was performed
    has_deduplication = "is_canonical" in df_raw.columns
    
    if has_deduplication:
        canonical_count = df_raw.filter(pl.col("is_canonical") == True).shape[0]
        total_count = len(df_raw)
        duplicate_count = total_count - canonical_count
        logger.info(f"Deduplication detected: {canonical_count:,} canonical, {duplicate_count:,} duplicates")
    
    # Join on comment_id
    df_joined = df_makeup.join(df_raw, on="comment_id", how="left")
    logger.info(f"Joined data: {len(df_joined)} comments")
    
    # Join with FR data to get agency info and lifecycle stage if not already present
    if fr_csv.exists():
        logger.info(f"Loading FR data from: {fr_csv}")
        df_fr = pl.read_csv(
            str(fr_csv),
            infer_schema_length=None,
            schema_overrides={"cfr_titles": pl.Utf8, "topics": pl.Utf8, "abstract": pl.Utf8},
        )

        # Determine which columns to join
        join_cols = ["document_number"]
        if "agency" not in df_joined.columns and "agency" in df_fr.columns:
            join_cols.append("agency")
        if "lifecycle_stage" not in df_joined.columns and "lifecycle_stage" in df_fr.columns:
            join_cols.append("lifecycle_stage")
        if "rin" not in df_joined.columns and "rin" in df_fr.columns:
            join_cols.append("rin")

        if len(join_cols) > 1:  # Have columns beyond document_number
            df_joined = df_joined.join(
                df_fr.select(join_cols),
                on="document_number",
                how="left",
            )
            logger.info(f"Joined FR data columns: {join_cols[1:]}")
    
    # Load existing responses
    processed_ids = load_existing_responses(responses_csv)
    
    # Filter to unprocessed comments
    # If deduplication enabled, only process canonical comments (duplicates will be propagated)
    if has_deduplication:
        df_unprocessed = df_joined.filter(
            (pl.col("is_canonical") == True) &
            (~pl.col("comment_id").is_in(list(processed_ids)))
        )
        logger.info(f"Filtering to canonical comments only: {len(df_unprocessed):,} to process")
    else:
        df_unprocessed = df_joined.filter(
            ~pl.col("comment_id").is_in(list(processed_ids))
        )
        logger.info(f"Processing {len(df_unprocessed)} unprocessed comments")
    
    # Initialize tracker based on provider
    if provider == "xai":
        logger.info(f"Initializing XAIResponseTracker with model={config.xai_model}")
        tracker = XAIResponseTracker(
            api_key=api_key,
            model=config.xai_model,
            max_retries=5,
            enable_search=True,
        )
        max_concurrency = config.xai_max_concurrency
        logger.info(f"XAIResponseTracker initialized successfully with model={tracker.model}")
    elif provider == "openai":
        logger.info(f"Initializing OpenAIResponseTracker with model=gpt-5-mini")
        tracker = OpenAIResponseTracker(
            api_key=api_key,
            model="gpt-5-mini",
            max_retries=5,
            enable_search=True,
        )
        max_concurrency = config.max_concurrency
        logger.info(f"OpenAIResponseTracker initialized successfully with model={tracker.model}")
    else:
        logger.info(f"Initializing GeminiResponseTracker with model={config.gemini_model}")
        tracker = GeminiResponseTracker(
            api_key=api_key,
            model=config.gemini_model,
            max_retries=10,
            enable_search=config.enable_search_grounding,
            thinking_level=config.gemini_thinking_level,
        )
        max_concurrency = config.gemini_max_concurrency
        logger.info(f"GeminiResponseTracker initialized successfully with model={tracker.model}")

    # ========================================================================
    # TIER 1: Web search grounding for explicit agency responses
    # ========================================================================
    weight_map: Optional[Dict[str, float]] = None

    if len(df_unprocessed) == 0:
        logger.info("Tier 1: All comments already processed")
    else:
        # Stratified sampling (always on — Cochran+FPC by commenter type)
        if "category" in df_unprocessed.columns:
            log_banner(logger, "RESPONSE TRACKING SAMPLING")
            df_to_process, weight_map = sample_comments_for_response_tracking(
                df_unprocessed,
                census_threshold=30,
                seed=config.sampling_seed,
            )
        else:
            df_to_process = df_unprocessed
            logger.info("No 'category' column -- processing all comments (no sampling)")

        # Convert to list of dicts
        comments_to_process = df_to_process.to_dicts()

        # Apply limit if specified (for testing)
        if limit:
            comments_to_process = comments_to_process[:limit]
            logger.info(f"Limited to {len(comments_to_process)} comments for testing")

        # Initialize RegsGovClient for re-downloading attachments if needed
        regs_client = None
        try:
            api_keys = get_regs_api_keys(required=False)
            if api_keys:
                regs_client = RegsGovClient(api_keys, retries=config.retries)
                logger.info("RegsGovClient initialized for attachment re-downloading")
        except Exception as e:
            logger.warning(f"Could not initialize RegsGovClient: {e}")
            logger.warning("Will use existing attachment text only")

        # Process comments (pass df_raw for duplicate propagation if deduplication enabled)
        try:
            asyncio.run(process_responses_async(
                tracker,
                comments_to_process,
                responses_csv,
                max_concurrency,
                regs_client,
                config.max_comment_pages,
                df_raw if has_deduplication else None,
                batch_size,
                weight_map,
            ))
        finally:
            if regs_client:
                regs_client.close()

    log_banner(logger, "TIER 1 RESPONSE TRACKING COMPLETE")
    logger.info(f"Output file: {responses_csv.absolute()}")

    # Print Tier 1 summary statistics
    _print_response_summary(responses_csv, "TIER 1")

    # ========================================================================
    # TIER 2: NPRM vs FINAL RULE TEXT COMPARISON
    # ========================================================================
    # Tier 2 doesn't use web search — it compares NPRM vs Final Rule text
    # directly via Gemini. When using OpenAI for Tier 1, create a separate
    # Gemini tracker for Tier 2 (if Gemini key available), or skip.
    # ========================================================================
    tier2_tracker = tracker  # Same tracker if Gemini was used for Tier 1
    if provider in ("openai", "xai"):
        # Need a Gemini tracker for Tier 2 (no web search needed)
        gemini_key = get_gemini_api_key(required=False)
        if gemini_key:
            logger.info("Creating separate Gemini tracker for Tier 2 (no search grounding)")
            tier2_tracker = GeminiResponseTracker(
                api_key=gemini_key,
                model=config.gemini_model,
                max_retries=5,
                enable_search=False,
                thinking_level=config.gemini_thinking_level,
            )
        else:
            logger.info(f"Tier 2: Skipping (no Gemini API key available, {provider} used for Tier 1)")
            tier2_tracker = None

    if tier2_tracker is not None:
        _run_tier2_comparison(
            config=config,
            tracker=tier2_tracker,
            fr_csv=fr_csv,
            responses_csv=responses_csv,
            df_raw=df_raw if has_deduplication else None,
        )

    log_banner(logger, "ALL RESPONSE TRACKING COMPLETE (TIER 1 + TIER 2)")
    _print_response_summary(responses_csv, "FINAL (TIER 1 + TIER 2)")


def _print_response_summary(responses_csv: Path, label: str = "") -> None:
    """Print summary statistics from the responses CSV."""
    if not responses_csv.exists():
        return

    df_responses = pl.read_csv(str(responses_csv), infer_schema_length=None)
    total = len(df_responses)

    logger.info(f"\n{label} Total responses tracked: {total}")

    if total == 0:
        return

    logger.info(f"\n{label} Response found breakdown:")
    for value in ["yes", "no", "uncertain"]:
        count = df_responses.filter(pl.col("response_found") == value).shape[0]
        pct = 100.0 * count / total
        logger.info(f"  {value}: {count} ({pct:.1f}%)")

    # Agency decision breakdown (for responses found)
    responses_found = df_responses.filter(pl.col("response_found") == "yes")
    if len(responses_found) > 0:
        logger.info(f"\n{label} Agency decision breakdown (for found responses):")
        for value in ["accept", "reject", "partial", "uncertain"]:
            count = responses_found.filter(pl.col("agency_decision") == value).shape[0]
            pct = 100.0 * count / len(responses_found)
            logger.info(f"  {value}: {count} ({pct:.1f}%)")

    # Tier 2 summary (if columns exist)
    if "tier2_acceptance_status" in df_responses.columns:
        tier2_processed = df_responses.filter(
            pl.col("tier2_acceptance_status").is_not_null()
            & (pl.col("tier2_acceptance_status") != "")
            & (pl.col("tier2_acceptance_status") != "None")
        )
        if len(tier2_processed) > 0:
            logger.info(f"\n{label} Tier 2 acceptance status breakdown ({len(tier2_processed)} comments):")
            for value in ["ACCEPTED", "REJECTED", "PARTIAL", "UNCLEAR"]:
                count = tier2_processed.filter(pl.col("tier2_acceptance_status") == value).shape[0]
                pct = 100.0 * count / len(tier2_processed)
                logger.info(f"  {value}: {count} ({pct:.1f}%)")

    # Lifecycle-stratified response metrics
    if "lifecycle_stage" in df_responses.columns:
        log_banner(logger, f"RESPONSE RATES BY LIFECYCLE STAGE ({label})")

        all_stages = df_responses["lifecycle_stage"].unique().to_list()
        stages = [s for s in all_stages if s and s not in ("N/A", "UNKNOWN", "None", None)]

        if stages:
            for stage in sorted(stages):
                stage_data = df_responses.filter(pl.col("lifecycle_stage") == stage)
                stage_total = len(stage_data)

                if stage_total == 0:
                    continue

                found_yes = stage_data.filter(pl.col("response_found") == "yes").shape[0]
                found_rate = 100.0 * found_yes / stage_total

                stage_found = stage_data.filter(pl.col("response_found") == "yes")
                accept = stage_found.filter(pl.col("agency_decision") == "accept").shape[0] if len(stage_found) > 0 else 0
                reject = stage_found.filter(pl.col("agency_decision") == "reject").shape[0] if len(stage_found) > 0 else 0
                partial = stage_found.filter(pl.col("agency_decision") == "partial").shape[0] if len(stage_found) > 0 else 0

                logger.info(f"\n{stage}:")
                logger.info(f"  Total comments: {stage_total}")
                logger.info(f"  Response found: {found_yes} ({found_rate:.1f}%)")
                if len(stage_found) > 0:
                    accept_rate = 100.0 * accept / len(stage_found)
                    reject_rate = 100.0 * reject / len(stage_found)
                    partial_rate = 100.0 * partial / len(stage_found)
                    logger.info(f"  Accept: {accept} ({accept_rate:.1f}%) | Reject: {reject} ({reject_rate:.1f}%) | Partial: {partial} ({partial_rate:.1f}%)")
        else:
            logger.info("No lifecycle stage data found for stratification")


def _run_tier2_comparison(
    config: PipelineConfig,
    tracker: "GeminiResponseTracker",
    fr_csv: Path,
    responses_csv: Path,
    df_raw: Optional[pl.DataFrame] = None,
) -> None:
    """
    Tier 2: Compare NPRM vs Final Rule text for comments where Tier 1 found no response.

    Loads the FR CSV to get document linking columns (nprm_document_number,
    final_rule_document_number), identifies eligible comments, fetches document
    texts from the FR API (with per-document caching), and sends each eligible
    comment to Gemini for NPRM vs Final Rule comparison.

    Results are merged back into the agency_responses CSV as new columns:
    tier2_acceptance_status, tier2_confidence, tier2_text_change_summary.

    Args:
        config: Pipeline configuration
        tracker: Already-initialized GeminiResponseTracker
        fr_csv: Path to federal_register_{year}_comments.csv
        responses_csv: Path to agency_responses_{year}.csv
        df_raw: Optional raw comments DataFrame for duplicate propagation
    """
    from stratification_scripts.federal_register.client import FederalRegisterClient

    log_banner(logger, "TIER 2: NPRM vs FINAL RULE TEXT COMPARISON")

    # ------------------------------------------------------------------
    # Step 1: Load data and identify Tier 2-eligible comments
    # ------------------------------------------------------------------
    if not responses_csv.exists():
        logger.warning("Tier 2: No responses CSV found. Skipping Tier 2.")
        return

    if not fr_csv.exists():
        logger.warning("Tier 2: No FR CSV found. Cannot perform document linking. Skipping Tier 2.")
        return

    df_responses = pl.read_csv(str(responses_csv), infer_schema_length=None)
    df_fr = pl.read_csv(
        str(fr_csv),
        infer_schema_length=None,
        schema_overrides={"cfr_titles": pl.Utf8, "topics": pl.Utf8, "abstract": pl.Utf8},
    )

    # Check required columns exist in FR CSV
    for col in ["nprm_document_number", "final_rule_document_number"]:
        if col not in df_fr.columns:
            logger.warning(f"Tier 2: Column '{col}' not found in FR CSV. Skipping Tier 2.")
            logger.warning("Tier 2: Run distribution.py Stage 4 (document linking) first.")
            return

    # Find comments where Tier 1 did NOT find a response
    no_response = df_responses.filter(pl.col("response_found") != "yes")
    total_no_response = len(no_response)
    logger.info(f"Tier 2: {total_no_response} comments with response_found != 'yes'")

    if total_no_response == 0:
        logger.info("Tier 2: All comments have Tier 1 responses. Nothing to do.")
        return

    # Skip comments already processed by Tier 2 (idempotency)
    if "tier2_acceptance_status" in df_responses.columns:
        already_tier2 = df_responses.filter(
            pl.col("tier2_acceptance_status").is_not_null()
            & (pl.col("tier2_acceptance_status") != "")
            & (pl.col("tier2_acceptance_status") != "None")
        )
        already_tier2_ids = set(already_tier2["comment_id"].to_list())
        if already_tier2_ids:
            logger.info(f"Tier 2: {len(already_tier2_ids)} comments already have Tier 2 results, skipping them")
            no_response = no_response.filter(~pl.col("comment_id").is_in(list(already_tier2_ids)))

    if len(no_response) == 0:
        logger.info("Tier 2: All eligible comments already processed. Nothing to do.")
        return

    # Join no-response comments with FR CSV to get document linking columns
    doc_linking = df_fr.select([
        "document_number",
        "nprm_document_number",
        "final_rule_document_number",
    ])

    no_response_with_links = no_response.join(
        doc_linking,
        on="document_number",
        how="left",
    )

    # Filter to comments where BOTH nprm and final rule document numbers exist
    def _is_valid(col_name: str) -> pl.Expr:
        return (
            pl.col(col_name).is_not_null()
            & (pl.col(col_name) != "")
            & (pl.col(col_name) != "None")
            & (pl.col(col_name) != "null")
        )

    eligible = no_response_with_links.filter(
        _is_valid("nprm_document_number") & _is_valid("final_rule_document_number")
    )

    total_eligible = len(eligible)
    logger.info(
        f"Tier 2: {total_eligible} comments eligible "
        f"(have both NPRM and Final Rule document numbers, out of {total_no_response} without Tier 1 response)"
    )

    if total_eligible == 0:
        logger.info("Tier 2: No eligible comments found. Skipping.")
        # Log why: breakdown of no-response comments
        no_nprm = no_response_with_links.filter(~_is_valid("nprm_document_number")).shape[0]
        no_final = no_response_with_links.filter(~_is_valid("final_rule_document_number")).shape[0]
        logger.info(f"Tier 2: Breakdown of ineligible: {no_nprm} missing NPRM doc, {no_final} missing Final Rule doc")
        return

    # ------------------------------------------------------------------
    # Step 2: Fetch and cache document texts
    # ------------------------------------------------------------------
    # Identify unique document pairs to fetch
    unique_pairs = (
        eligible
        .select(["nprm_document_number", "final_rule_document_number"])
        .unique()
    )

    logger.info(
        f"Tier 2: {len(unique_pairs)} unique document pair(s) to fetch "
        f"(serving {total_eligible} comments)"
    )

    # Cache: (document_number) -> text
    doc_text_cache: Dict[str, Optional[str]] = {}

    # Collect all unique document numbers that need fetching
    all_doc_nums: Set[str] = set()
    for row in unique_pairs.iter_rows(named=True):
        all_doc_nums.add(row["nprm_document_number"])
        all_doc_nums.add(row["final_rule_document_number"])

    logger.info(f"Tier 2: Fetching full text for {len(all_doc_nums)} unique document(s) from FR API")

    fr_client = FederalRegisterClient(
        max_retries=config.retries,
        sleep_between=0.5,  # Rate limit: 0.5s between requests
    )

    fetch_failures: List[str] = []

    try:
        for doc_num in sorted(all_doc_nums):
            if doc_num in doc_text_cache:
                continue

            logger.info(f"Tier 2: Fetching full text for document {doc_num}")
            text = fr_client.fetch_document_full_text(doc_num, max_chars=50000)

            if text is None:
                logger.warning(f"Tier 2: FAILED to fetch text for document {doc_num}")
                fetch_failures.append(doc_num)
                doc_text_cache[doc_num] = None
            else:
                logger.info(f"Tier 2: Fetched {len(text)} chars for document {doc_num}")
                doc_text_cache[doc_num] = text
    finally:
        fr_client.close()

    if fetch_failures:
        logger.warning(f"Tier 2: Failed to fetch text for {len(fetch_failures)} document(s): {fetch_failures}")

    # ------------------------------------------------------------------
    # Step 3: Prepare Tier 2 batch for Gemini
    # ------------------------------------------------------------------
    # Load raw comments for full text extraction
    comments_raw_csv = get_comments_raw_path(config.year)
    df_comments_raw = pl.read_csv(str(comments_raw_csv), infer_schema_length=None, schema_overrides={"zip": pl.Utf8}) if comments_raw_csv.exists() else None

    tier2_batch: List[tuple] = []  # (comment_text, nprm_text, final_text, metadata)
    skipped_no_text: List[str] = []

    for row in eligible.iter_rows(named=True):
        comment_id = str(row["comment_id"])
        nprm_doc_num = row["nprm_document_number"]
        final_doc_num = row["final_rule_document_number"]

        nprm_text = doc_text_cache.get(nprm_doc_num)
        final_text = doc_text_cache.get(final_doc_num)

        if nprm_text is None or final_text is None:
            logger.warning(
                f"Tier 2: Skipping comment {comment_id} -- "
                f"missing text for {'NPRM ' + nprm_doc_num if nprm_text is None else ''}"
                f"{'Final Rule ' + final_doc_num if final_text is None else ''}"
            )
            skipped_no_text.append(comment_id)
            continue

        # Get the comment text
        comment_text_str = ""
        if df_comments_raw is not None:
            comment_row = df_comments_raw.filter(pl.col("comment_id") == comment_id)
            if len(comment_row) > 0:
                comment_dict = comment_row.to_dicts()[0]
                comment_text_str = extract_full_comment_text(comment_dict)

        if not comment_text_str:
            # Fallback: try from the joined data
            ct = row.get("comment_text", "")
            if ct and str(ct).strip() and str(ct).strip().lower() != "none":
                comment_text_str = str(ct).strip()

        if not comment_text_str:
            logger.warning(f"Tier 2: No comment text found for {comment_id}, skipping")
            skipped_no_text.append(comment_id)
            continue

        metadata = {
            "comment_id": comment_id,
            "document_number": str(row.get("document_number", "N/A")),
            "agency": str(row.get("agency", "N/A")),
        }

        tier2_batch.append((comment_text_str, nprm_text, final_text, metadata))

    logger.info(
        f"Tier 2: Prepared {len(tier2_batch)} comments for Gemini comparison"
        f" (skipped {len(skipped_no_text)} due to missing text)"
    )

    if not tier2_batch:
        logger.info("Tier 2: No comments to process after text filtering. Done.")
        return

    # ------------------------------------------------------------------
    # Step 4: Send to Gemini for Tier 2 comparison
    # ------------------------------------------------------------------
    logger.info(f"Tier 2: Sending {len(tier2_batch)} comments to Gemini for NPRM vs Final Rule comparison")

    tier2_results = asyncio.run(
        tracker.track_tier2_batch(tier2_batch, max_concurrency=config.gemini_max_concurrency)
    )

    # ------------------------------------------------------------------
    # Step 5: Merge Tier 2 results back into the responses CSV
    # ------------------------------------------------------------------
    logger.info(f"Tier 2: Processing {len(tier2_results)} results")

    # Build a mapping: comment_id -> tier2 result dict
    tier2_map: Dict[str, Dict[str, str]] = {}
    for comment_id, parsed, raw in tier2_results:
        tier2_map[comment_id] = {
            "tier2_acceptance_status": parsed.get("acceptance_status", "UNCLEAR"),
            "tier2_confidence": parsed.get("confidence", "0.0"),
            "tier2_text_change_summary": parsed.get("text_change_summary", ""),
        }

    # Propagate Tier 2 results to duplicates if applicable
    if df_raw is not None and "is_canonical" in df_raw.columns:
        duplicate_groups = df_raw.filter(
            pl.col("duplicate_group_id").is_not_null()
            & (pl.col("is_canonical") == False)
        )
        for row in duplicate_groups.iter_rows(named=True):
            dup_id = str(row["comment_id"])
            canonical_id = str(row["canonical_comment_id"])
            if canonical_id in tier2_map and dup_id not in tier2_map:
                tier2_map[dup_id] = tier2_map[canonical_id].copy()

    # Reload the responses CSV and add Tier 2 columns
    df_responses = pl.read_csv(str(responses_csv), infer_schema_length=None)

    # Initialize Tier 2 columns if they don't exist
    for col in ["tier2_acceptance_status", "tier2_confidence", "tier2_text_change_summary"]:
        if col not in df_responses.columns:
            df_responses = df_responses.with_columns(
                pl.lit(None).cast(pl.Utf8).alias(col)
            )

    # Update rows that have Tier 2 results
    tier2_updated_count = 0
    # Build new column values using when/then
    comment_ids_with_tier2 = list(tier2_map.keys())

    if comment_ids_with_tier2:
        # Create a small DataFrame with tier2 results for joining
        tier2_rows = [
            {
                "comment_id": cid,
                "t2_acceptance_status": vals["tier2_acceptance_status"],
                "t2_confidence": vals["tier2_confidence"],
                "t2_text_change_summary": vals["tier2_text_change_summary"],
            }
            for cid, vals in tier2_map.items()
        ]
        df_tier2 = pl.DataFrame(tier2_rows)

        # Join and coalesce
        df_merged = df_responses.join(df_tier2, on="comment_id", how="left")

        df_merged = df_merged.with_columns([
            pl.coalesce([
                pl.col("t2_acceptance_status"),
                pl.col("tier2_acceptance_status"),
            ]).alias("tier2_acceptance_status"),
            pl.coalesce([
                pl.col("t2_confidence"),
                pl.col("tier2_confidence"),
            ]).alias("tier2_confidence"),
            pl.coalesce([
                pl.col("t2_text_change_summary"),
                pl.col("tier2_text_change_summary"),
            ]).alias("tier2_text_change_summary"),
        ])

        # Drop temporary join columns
        df_merged = df_merged.drop(["t2_acceptance_status", "t2_confidence", "t2_text_change_summary"])

        tier2_updated_count = len(df_tier2.filter(pl.col("comment_id").is_in(df_responses["comment_id"].to_list())))

        df_responses = df_merged

    # Backup existing CSV before Tier 2 merge to prevent data loss
    import shutil
    if responses_csv.exists():
        backup_path = responses_csv.with_suffix(f".pre_tier2_backup.csv")
        shutil.copy2(responses_csv, backup_path)
        logger.info(f"Tier 2: Backed up existing responses to {backup_path.name}")

    # Write back
    df_responses.write_csv(str(responses_csv))
    logger.info(f"Tier 2: Updated {tier2_updated_count} comment(s) in {responses_csv.name}")

    # ------------------------------------------------------------------
    # Step 6: Log Tier 2 summary
    # ------------------------------------------------------------------
    log_banner(logger, "TIER 2 SUMMARY")
    logger.info(f"  Comments with no Tier 1 response: {total_no_response}")
    logger.info(f"  Tier 2 eligible (have NPRM + Final Rule doc pairs): {total_eligible}")
    logger.info(f"  Skipped (missing document text): {len(skipped_no_text)}")
    logger.info(f"  Processed by Gemini: {len(tier2_results)}")
    logger.info(f"  Updated in CSV: {tier2_updated_count}")

    if tier2_results:
        # Acceptance status distribution
        status_counts: Dict[str, int] = {}
        for _, parsed, _ in tier2_results:
            status = parsed.get("acceptance_status", "UNCLEAR")
            status_counts[status] = status_counts.get(status, 0) + 1

        logger.info(f"\n  Tier 2 acceptance status distribution:")
        for status in ["ACCEPTED", "REJECTED", "PARTIAL", "UNCLEAR"]:
            count = status_counts.get(status, 0)
            pct = 100.0 * count / len(tier2_results) if tier2_results else 0
            logger.info(f"    {status}: {count} ({pct:.1f}%)")


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Track federal agency responses to public comments"
    )
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--max-concurrency", type=int, default=100)
    parser.add_argument("--model", type=str, default="gemini-3-flash-preview",
                        help="Gemini model (used for Tier 2 and when provider=gemini)")
    parser.add_argument("--provider", type=str, default="xai", choices=["xai", "openai", "gemini"],
                        help="LLM provider for Tier 1 response tracking (default: xai)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of comments to process (for testing)")
    parser.add_argument("--max-comment-pages", type=int, default=30,
                        help="Max pages to extract from PDF attachments")
    parser.add_argument("--disable-search", action="store_true",
                        help="Disable search grounding")
    parser.add_argument("--thinking-level", type=str, default=None,
                        help='Gemini 3 thinking level: minimal|low|medium|high')
    parser.add_argument("--batch-size", type=int, default=1000,
                        help="Number of comments to process per batch (default: 1000)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    
    args = parser.parse_args()
    
    setup_logging(verbose=args.verbose, quiet=args.quiet, year=args.year)
    
    config = PipelineConfig(
        year=args.year,
        response_provider=args.provider,
        gemini_model=args.model,
        gemini_max_concurrency=args.max_concurrency,
        enable_search_grounding=not args.disable_search,
        max_comment_pages=args.max_comment_pages,
        gemini_thinking_level=args.thinking_level,
        verbose=args.verbose,
        quiet=args.quiet,
    )
    
    try:
        track_responses_for_year(config, limit=args.limit, batch_size=args.batch_size)
        return 0
    except Exception as e:
        logger.error(f"Response tracking failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

