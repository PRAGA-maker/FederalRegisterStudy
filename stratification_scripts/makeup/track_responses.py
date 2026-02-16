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
    get_comments_raw_path,
    get_makeup_data_path,
    get_fr_csv_path,
)
from stratification_scripts.logging_utils import (
    get_logger,
    log_banner,
    setup_logging,
)
from stratification_scripts.gemini_client import GeminiResponseTracker
from stratification_scripts.io_utils import extract_pdf_text
from stratification_scripts.regulations_gov.client import RegsGovClient
from stratification_scripts.config import get_regs_api_keys

logger = get_logger(__name__)


def get_agency_responses_path(year: int) -> Path:
    """Get path to agency_responses_{year}.csv"""
    from stratification_scripts.config import get_data_dir
    return get_data_dir() / f"agency_responses_{year}.csv"


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


def load_existing_responses(responses_csv: Path) -> Set[str]:
    """Load existing response tracking results and return processed comment IDs."""
    if not responses_csv.exists():
        return set()
    
    try:
        df_existing = pl.read_csv(str(responses_csv))
        processed_ids = set(df_existing["comment_id"].to_list())
        logger.info(f"Loaded {len(processed_ids)} already-processed comments")
        return processed_ids
    except Exception as e:
        logger.warning(f"Failed to load existing responses: {e}")
        return set()


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
    
    df_new = pl.DataFrame(new_results)
    
    if responses_csv.exists():
        try:
            df_existing = pl.read_csv(str(responses_csv))
            
            # Ensure schema compatibility
            for col in df_new.columns:
                if col not in df_existing.columns:
                    df_existing = df_existing.with_columns(
                        pl.lit(None).cast(df_new[col].dtype).alias(col)
                    )
            
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
    tracker: GeminiResponseTracker,
    comments_to_process: List[Dict],
    responses_csv: Path,
    max_concurrency: int,
    regs_client: Optional[RegsGovClient] = None,
    max_comment_pages: int = 30,
    df_comments: Optional[pl.DataFrame] = None,
    batch_size: int = 1000,
) -> None:
    """
    Process comments asynchronously in batches.
    
    Args:
        tracker: GeminiResponseTracker instance
        comments_to_process: List of comment dicts to process
        responses_csv: Path to save results
        max_concurrency: Max concurrent API calls
        regs_client: Optional RegsGovClient for re-downloading attachments
        max_comment_pages: Max pages to extract from PDFs
        df_comments: Optional DataFrame with comments (for duplicate propagation)
        batch_size: Number of comments to process per batch (default: 1000)
    """
    total_comments = len(comments_to_process)
    
    logger.info(f"Processing {total_comments} comments in batches of {batch_size}")
    
    for batch_start in range(0, total_comments, batch_size):
        batch_end = min(batch_start + batch_size, total_comments)
        batch = comments_to_process[batch_start:batch_end]
        
        logger.info(f"Processing batch {batch_start+1}-{batch_end} of {total_comments}")
        
        # Prepare batch for Gemini
        gemini_batch = []
        for comment in batch:
            # Extract full comment text
            full_text = extract_full_comment_text(
                comment,
                client=regs_client,
                max_pages=max_comment_pages,
            )
            
            metadata = {
                "comment_id": str(comment.get("comment_id", "unknown")),
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
            
            csv_results.append({
                "comment_id": comment_id,
                "document_number": original_comment.get("document_number", "N/A"),
                "agency": original_comment.get("agency", "N/A"),
                "response_found": parsed_response.get("response_found", "uncertain"),
                "agency_decision": parsed_response.get("agency_decision", "uncertain"),
                "response_text": parsed_response.get("response_text", "N/A"),
                "response_location": parsed_response.get("response_location", "N/A"),
                "reasoning": parsed_response.get("reasoning", "N/A"),
                "processed_at": datetime.now().isoformat(),
                "model": tracker.model,
                "comment_text_length": len(extract_full_comment_text(original_comment)),
                "has_attachment": bool(original_comment.get("attachment_text")),
                # Lifecycle tracking fields
                "lifecycle_stage": original_comment.get("lifecycle_stage", "UNKNOWN"),
                "rin": original_comment.get("rin", "N/A"),
            })
        
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
    df_makeup = pl.read_csv(str(makeup_data_csv))
    logger.info(f"Loaded {len(df_makeup)} classified comments")
    
    # Join with comments_raw to get full metadata (including deduplication columns if present)
    logger.info(f"Loading raw comments from: {comments_raw_csv}")
    df_raw = pl.read_csv(str(comments_raw_csv))
    
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
        df_fr = pl.read_csv(str(fr_csv))

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
    
    if len(df_unprocessed) == 0:
        logger.info("All comments already processed")
        return
    
    # Convert to list of dicts
    comments_to_process = df_unprocessed.to_dicts()
    
    # Apply limit if specified (for testing)
    if limit:
        comments_to_process = comments_to_process[:limit]
        logger.info(f"Limited to {len(comments_to_process)} comments for testing")
    
    # Initialize Gemini tracker
    logger.info(f"Initializing GeminiResponseTracker with model={config.gemini_model}")
    tracker = GeminiResponseTracker(
        api_key=api_key,
        model=config.gemini_model,
        max_retries=10,
        enable_search=config.enable_search_grounding,
        thinking_level=config.gemini_thinking_level,
    )
    logger.info(f"GeminiResponseTracker initialized successfully with model={tracker.model}")
    
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
            config.gemini_max_concurrency,
            regs_client,
            config.max_comment_pages,
            df_raw if has_deduplication else None,
            batch_size,
        ))
    finally:
        if regs_client:
            regs_client.close()
    
    log_banner(logger, "RESPONSE TRACKING COMPLETE")
    logger.info(f"Output file: {responses_csv.absolute()}")
    
    # Print summary statistics
    if responses_csv.exists():
        df_responses = pl.read_csv(str(responses_csv))
        total = len(df_responses)
        
        logger.info(f"\nTotal responses tracked: {total}")
        
        if total > 0:
            logger.info("\nResponse found breakdown:")
            for value in ["yes", "no", "uncertain"]:
                count = df_responses.filter(pl.col("response_found") == value).shape[0]
                pct = 100.0 * count / total
                logger.info(f"  {value}: {count} ({pct:.1f}%)")
            
            # Agency decision breakdown (for responses found)
            responses_found = df_responses.filter(pl.col("response_found") == "yes")
            if len(responses_found) > 0:
                logger.info("\nAgency decision breakdown (for found responses):")
                for value in ["accept", "reject", "uncertain"]:
                    count = responses_found.filter(pl.col("agency_decision") == value).shape[0]
                    pct = 100.0 * count / len(responses_found)
                    logger.info(f"  {value}: {count} ({pct:.1f}%)")

            # Lifecycle-stratified response metrics
            if "lifecycle_stage" in df_responses.columns:
                log_banner(logger, "RESPONSE RATES BY LIFECYCLE STAGE")

                # Get unique stages, excluding N/A and UNKNOWN
                all_stages = df_responses["lifecycle_stage"].unique().to_list()
                stages = [s for s in all_stages if s and s not in ("N/A", "UNKNOWN", "None", None)]

                if stages:
                    for stage in sorted(stages):
                        stage_data = df_responses.filter(pl.col("lifecycle_stage") == stage)
                        stage_total = len(stage_data)

                        if stage_total == 0:
                            continue

                        # Response found rate
                        found_yes = stage_data.filter(pl.col("response_found") == "yes").shape[0]
                        found_no = stage_data.filter(pl.col("response_found") == "no").shape[0]
                        found_rate = 100.0 * found_yes / stage_total

                        # Decision rates (for found responses)
                        stage_found = stage_data.filter(pl.col("response_found") == "yes")
                        accept = stage_found.filter(pl.col("agency_decision") == "accept").shape[0] if len(stage_found) > 0 else 0
                        reject = stage_found.filter(pl.col("agency_decision") == "reject").shape[0] if len(stage_found) > 0 else 0

                        logger.info(f"\n{stage}:")
                        logger.info(f"  Total comments: {stage_total}")
                        logger.info(f"  Response found: {found_yes} ({found_rate:.1f}%)")
                        if len(stage_found) > 0:
                            accept_rate = 100.0 * accept / len(stage_found)
                            reject_rate = 100.0 * reject / len(stage_found)
                            logger.info(f"  Accept: {accept} ({accept_rate:.1f}%) | Reject: {reject} ({reject_rate:.1f}%)")
                else:
                    logger.info("No lifecycle stage data found for stratification")


# ============================================================================
# FUTURE STEP 2: HANDLE NO-RESPONSE CASES
# ============================================================================
# TODO: This section will be implemented by me after Jennifer's work is done.
# 
# Goal: For comments where response_found == "no" or "uncertain",
#       perform additional processing to attempt finding responses
#       through alternative channels or methods.
#
# Placeholder for:
# - Alternative search strategies
# - Manual verification workflows
# - Extended search timeframes
# ============================================================================


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Track federal agency responses to public comments"
    )
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--max-concurrency", type=int, default=100)
    parser.add_argument("--model", type=str, default="gemini-3-flash-preview")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of comments to process (for testing)")
    parser.add_argument("--max-comment-pages", type=int, default=30,
                        help="Max pages to extract from PDF attachments")
    parser.add_argument("--disable-search", action="store_true",
                        help="Disable Google Search grounding")
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

