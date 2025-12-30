#!/usr/bin/env python3
"""
Fetch and enrich Federal Register documents with comment metadata.

This module fetches proposed rules and notices from the Federal Register,
enriches them with Regulations.gov comment counts, and outputs a CSV
for downstream pipeline steps.

Two-stage enrichment:
1. Stage 1: Fetch FR document details (comment URLs, regs.gov IDs)
2. Stage 2: Fetch accurate comment counts from Regulations.gov

Example:
    # CLI usage
    $ python -m stratification_scripts.distribution --year 2024
    
    # Programmatic usage
    >>> from stratification_scripts.distribution import fetch_and_enrich_documents
    >>> from stratification_scripts.config import PipelineConfig
    >>> 
    >>> config = PipelineConfig(year=2024, limit_docs=100)
    >>> df = fetch_and_enrich_documents(config)
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from stratification_scripts.config import (
    PipelineConfig,
    get_regs_api_keys,
    get_fr_csv_path,
)
from stratification_scripts.federal_register.client import (
    FederalRegisterClient,
    classify_submission_channel,
)
from stratification_scripts.logging_utils import (
    get_logger,
    log_banner,
    setup_logging,
)
from stratification_scripts.regulations_gov.client import RegsGovClient

logger = get_logger(__name__)


def enrich_fr_detail(
    rec: dict,
    fr_client: FederalRegisterClient,
) -> Optional[dict]:
    """
    Stage 1: Enrich document with FR detail data.
    
    Fetches FR detail and extracts metadata. For regs.gov docs, comment_count
    will be a preliminary value updated in Stage 2.
    
    Args:
        rec: FR document record from list API
        fr_client: FederalRegisterClient instance
    
    Returns:
        Enriched record dict, or None if doc is ineligible for comments.
    
    Side Effects:
        Makes HTTP request to FR API.
    """
    doc_number = rec.get("document_number")
    comment_url = None
    comments_close_on = None
    regs_document_id = None
    count_source = "unknown"
    comment_count: Optional[int] = None
    
    if doc_number:
        details = fr_client.fetch_document_details(doc_number)
        if details:
            comment_url = details.get("comment_url")
            comments_close_on = details.get("comments_close_on")
            fr_info = details.get("regulations_dot_gov_info", {})
            if isinstance(fr_info, dict):
                regs_document_id = fr_info.get("document_id")
                
                # For regs.gov docs, use FR embedded count as temporary value
                if regs_document_id:
                    cc_fr = fr_info.get("comments_count")
                    if isinstance(cc_fr, int) and cc_fr >= 0:
                        comment_count = cc_fr
                        count_source = "federalregister"
        else:
            logger.debug(f"FR API failed for {doc_number}")

    agencies = rec.get("agencies") or []
    agency_names = ", ".join([
        a.get("name") for a in agencies
        if isinstance(a, dict) and a.get("name") is not None
    ])

    # Eligibility and channel
    is_prorule = rec.get("type") == "Proposed Rule"
    has_comment_mechanism = bool(comment_url or comments_close_on or regs_document_id)
    eligibility_reason = None
    
    if is_prorule:
        eligibility_reason = "prorule"
    elif has_comment_mechanism:
        eligibility_reason = "notice-with-comment-period"
    
    submission_channel = classify_submission_channel(
        comment_url, regs_document_id
    ) if eligibility_reason else None

    if not eligibility_reason:
        return None

    return {
        "document_number": rec.get("document_number"),
        "title": rec.get("title"),
        "agency": agency_names,
        "publication_date": rec.get("publication_date"),
        "comment_url": comment_url,
        "comments_close_on": comments_close_on,
        "regs_document_id": regs_document_id,
        "comment_count": comment_count,
        "count_source": count_source,
        "eligibility_reason": eligibility_reason,
        "submission_channel": submission_channel,
    }


def enrich_regs_count(
    base_rec: dict,
    regs_client: RegsGovClient,
) -> dict:
    """
    Stage 2: Enrich regs.gov docs with accurate comment counts.
    
    Only called for docs with regs_document_id. Overwrites the FR embedded
    count with accurate data from Regulations.gov API.
    
    Args:
        base_rec: Record from Stage 1 enrichment
        regs_client: RegsGovClient instance
    
    Returns:
        Updated record with accurate comment count.
    
    Side Effects:
        Makes HTTP request to Regulations.gov API.
    """
    regs_document_id = base_rec.get("regs_document_id")
    if not regs_document_id:
        return base_rec
    
    comment_count = base_rec.get("comment_count")
    count_source = base_rec.get("count_source", "unknown")
    
    # Fetch accurate count from Regs.gov
    regs_detail = regs_client.get_document_detail(regs_document_id)
    if regs_detail:
        attrs = (regs_detail.get("data") or {}).get("attributes") or {}
        cc = attrs.get("commentCount")
        if isinstance(cc, int) and cc >= 0:
            comment_count = cc
            count_source = "regulations.gov"
        else:
            obj_id = attrs.get("objectId")
            if obj_id:
                mt = regs_client.get_comment_total_by_object_id(obj_id)
                if isinstance(mt, int):
                    comment_count = mt
                    count_source = "regulations.gov-meta"
    
    # Ensure we have a valid count
    if not isinstance(comment_count, int) or comment_count < 0:
        comment_count = 0
    
    base_rec["comment_count"] = comment_count
    base_rec["count_source"] = count_source
    return base_rec


def fetch_and_enrich_documents(
    config: PipelineConfig,
) -> pd.DataFrame:
    """
    Fetch and enrich Federal Register documents for a year.
    
    This is the main entry point for programmatic use. It:
    1. Fetches all PRORULE/NOTICE documents from Federal Register
    2. Enriches with FR detail data (Stage 1)
    3. Enriches regs.gov docs with accurate counts (Stage 2)
    4. Returns a DataFrame ready for downstream processing
    
    Args:
        config: Pipeline configuration with year and API settings
    
    Returns:
        DataFrame with enriched document data.
    
    Side Effects:
        Makes HTTP requests to FR and Regulations.gov APIs.
        Logs progress to configured logger.
    """
    year = config.year
    api_keys = get_regs_api_keys(required=False)
    num_keys = len(api_keys)
    
    # Determine worker counts
    workers_stage1 = config.concurrent_workers or 3
    if config.concurrent_workers is not None:
        workers_stage2 = config.concurrent_workers
    elif num_keys > 1:
        workers_stage2 = min(8, num_keys * 2)
    elif num_keys == 1:
        workers_stage2 = 4
    else:
        workers_stage2 = 1
    
    if not config.quiet:
        if num_keys > 0:
            logger.info(f"Using {num_keys} Regs.gov API key(s)")
            logger.info(f"Stage 1 workers (FR API): {workers_stage1}")
            logger.info(f"Stage 2 workers (Regs.gov API): {workers_stage2}")
        else:
            logger.info(f"No Regs.gov API keys (FR detail only)")

    # Fetch FR documents day-by-day
    fr_client = FederalRegisterClient(
        max_retries=config.retries,
        sleep_between=config.fr_detail_sleep,
    )
    
    all_results: List[dict] = list(tqdm(
        fr_client.iter_documents_by_day(
            year,
            fr_sleep=config.fr_sleep,
            limit=config.limit_docs,
        ),
        desc="FR days",
        unit="doc",
    ))
    
    if not config.quiet:
        logger.info(f"Total {year} docs fetched (PRORULE + NOTICE): {len(all_results)}")

    # Stage 1: FR detail enrichment
    log_banner(logger, "STAGE 1: FR DETAIL ENRICHMENT")
    logger.info(f"Using {workers_stage1} workers with {config.fr_detail_sleep}s rate limit")
    
    stage1_results = []
    failed_count = 0
    
    with ThreadPoolExecutor(max_workers=workers_stage1) as executor:
        futures = {
            executor.submit(enrich_fr_detail, rec, fr_client): rec
            for rec in all_results
        }
        
        for future in tqdm(as_completed(futures), total=len(all_results),
                          desc="Stage 1: FR details", unit="doc"):
            try:
                partial = future.result()
                if partial:
                    stage1_results.append(partial)
                else:
                    failed_count += 1
            except Exception as e:
                failed_count += 1
                logger.debug(f"Unexpected error: {e}")
    
    fr_client.close()
    
    # Split by whether they need Regs.gov enrichment
    regs_docs = [p for p in stage1_results if p.get("regs_document_id")]
    non_regs_docs = [p for p in stage1_results if not p.get("regs_document_id")]
    
    logger.info(f"Stage 1 complete: {len(stage1_results)} comment-eligible docs")
    if failed_count > 0:
        logger.warning(f"{failed_count} docs failed FR detail fetch")
    logger.info(f"  {len(regs_docs)} with regs_document_id (need Stage 2)")
    logger.info(f"  {len(non_regs_docs)} without regs_document_id")
    
    # Stage 2: Regs.gov count enrichment
    rows: List[dict] = []
    
    if regs_docs and api_keys:
        log_banner(logger, "STAGE 2: REGS.GOV COUNT ENRICHMENT")
        logger.info(f"Using {workers_stage2} workers")
        
        regs_client = RegsGovClient(api_keys, retries=config.retries)
        
        with ThreadPoolExecutor(max_workers=workers_stage2) as executor:
            futures = {
                executor.submit(enrich_regs_count, rec, regs_client): rec
                for rec in regs_docs
            }
            
            for future in tqdm(as_completed(futures), total=len(regs_docs),
                              desc="Stage 2: Regs.gov counts", unit="doc"):
                try:
                    enriched = future.result()
                    if enriched:
                        rows.append(enriched)
                except Exception as e:
                    if not config.quiet:
                        logger.debug(f"Error in Stage 2: {e}")
        
        regs_client.close()
    elif regs_docs:
        # No API keys - use Stage 1 data as-is
        rows.extend(regs_docs)
    
    # Add non-regs.gov docs
    rows.extend(non_regs_docs)
    
    if not rows:
        logger.warning("No data collected - check API parameters")
        return pd.DataFrame()
    
    df = pd.DataFrame(rows)
    
    if not config.quiet:
        logger.info(f"Filtered to {len(rows)} comment-eligible documents")
        
        # Log breakdowns
        for col in ["eligibility_reason", "count_source", "submission_channel"]:
            if col in df.columns:
                logger.info(f"\n{col} breakdown:")
                for val, count in df[col].value_counts().items():
                    logger.info(f"  {val}: {count}")
    
    return df


def save_documents(
    df: pd.DataFrame,
    config: PipelineConfig,
) -> Path:
    """
    Save enriched documents to CSV.
    
    Args:
        df: DataFrame to save
        config: Pipeline configuration
    
    Returns:
        Path to saved file.
    
    Side Effects:
        Writes CSV to output directory.
    """
    csv_path = get_fr_csv_path(config.year)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    
    log_banner(logger, "SAVED FEDERAL REGISTER DOCUMENTS")
    logger.info(f"Output file: {csv_path.absolute()}")
    logger.info(f"Documents saved: {len(df)}")
    
    return csv_path


def print_analysis_summary(
    df: pd.DataFrame,
    config: PipelineConfig,
) -> None:
    """
    Print summary statistics for the enriched documents.
    
    Args:
        df: Enriched documents DataFrame
        config: Pipeline configuration
    """
    if len(df) == 0:
        logger.error("No data to analyze")
        return
    
    df = df.copy()
    df["comment_count"] = df["comment_count"].fillna(0).astype(int)
    total_docs = len(df)
    
    stats = df["comment_count"].describe(percentiles=[0.25, 0.5, 0.75])
    
    logger.info(
        f"\nOVERALL - Comment-eligible docs: {total_docs}; "
        f"min={int(stats['min'])}, p25={int(stats['25%'])}, median={int(stats['50%'])}, "
        f"mean={stats['mean']:.2f}, p75={int(stats['75%'])}, max={int(stats['max'])}"
    )

    # Stats for Regs.gov channel only
    regs_mapped = df[df["submission_channel"] == "regs.gov"]
    if len(regs_mapped) > 0:
        regs_stats = regs_mapped["comment_count"].describe(percentiles=[0.25, 0.5, 0.75])
        logger.info(
            f"\nREGS.GOV-MAPPED - Docs: {len(regs_mapped)}; "
            f"min={int(regs_stats['min'])}, p25={int(regs_stats['25%'])}, "
            f"median={int(regs_stats['50%'])}, mean={regs_stats['mean']:.2f}, "
            f"p75={int(regs_stats['75%'])}, max={int(regs_stats['max'])}"
        )
        
        # Top 3 by comment count
        top3 = regs_mapped.sort_values("comment_count", ascending=False).head(3)
        logger.info("Top 3 by comment_count (regs.gov):")
        for _, r in top3.iterrows():
            title = str(r['title'])[:80] if r['title'] else ""
            logger.info(f"  {r['document_number']}: {r['comment_count']} - {title}")


def main() -> int:
    """
    CLI entry point for document fetching and enrichment.
    
    Returns:
        Exit code (0 for success, 1 for failure).
    """
    parser = argparse.ArgumentParser(
        description="Distribution of comments for Federal Register documents by year",
        allow_abbrev=False,
    )
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of documents to process")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: stratification_scripts/output)")
    parser.add_argument("--concurrent-workers", type=int, default=None,
                        help="Number of concurrent enrichment workers")
    parser.add_argument("--fr-sleep", type=float, default=0.2,
                        help="Sleep seconds between FR page fetches")
    parser.add_argument("--fr-detail-sleep", type=float, default=0.5,
                        help="Sleep seconds between FR detail API calls")
    parser.add_argument("--retries", type=int, default=10,
                        help="Max retries for 429/5xx responses")
    parser.add_argument("--quiet", action="store_true",
                        help="Reduce log verbosity")
    parser.add_argument("--verbose", action="store_true",
                        help="Increase log verbosity")
    
    args = parser.parse_args()
    
    # Override year from environment if set
    env_year = os.environ.get("FR_YEAR")
    if env_year:
        try:
            args.year = int(env_year)
        except ValueError:
            pass
    
    # Setup logging
    setup_logging(verbose=args.verbose, quiet=args.quiet, year=args.year)
    
    # Create config
    config = PipelineConfig(
        year=args.year,
        limit_docs=args.limit,
        concurrent_workers=args.concurrent_workers,
        fr_sleep=args.fr_sleep,
        fr_detail_sleep=args.fr_detail_sleep,
        retries=args.retries,
        quiet=args.quiet,
        verbose=args.verbose,
    )
    
    if args.output_dir:
        config.output_dir = Path(args.output_dir)
    
    logger.info(f"Starting Federal Register Study Pipeline for {args.year}")
    
    try:
        df = fetch_and_enrich_documents(config)
        
        if len(df) == 0:
            logger.error("No documents collected")
            return 1
        
        save_documents(df, config)
        print_analysis_summary(df, config)
        
        log_banner(logger, "NEXT STEP: Mine comments")
        logger.info(
            f"Run: python -m stratification_scripts.makeup.mine_comments --year {args.year}"
        )
        
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

