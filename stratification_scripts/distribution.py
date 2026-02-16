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
import json
import os
import sys
from collections import Counter
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from stratification_scripts.config import (
    PipelineConfig,
    get_data_dir,
    get_regs_api_keys,
    get_fr_csv_path,
)
from stratification_scripts.federal_register.client import (
    FederalRegisterClient,
    classify_submission_channel,
    extract_rin,
    extract_all_rins,
    extract_docket_id,
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

    Fetches FR detail and extracts metadata including RIN and docket_id.
    For regs.gov docs, comment_count will be a preliminary value updated in Stage 2.

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

    # RIN and docket_id extraction
    rin: Optional[str] = None
    rin_all: List[str] = []
    docket_id: Optional[str] = None
    doc_type: Optional[str] = rec.get("type")

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

            # Extract RIN and docket_id (already done by fetch_document_details)
            rin = details.get("rin")
            rin_all = details.get("rin_all", [])
            docket_id = details.get("docket_id")

            # Also extract from regs_info if not found
            if not docket_id and isinstance(fr_info, dict):
                docket_id = fr_info.get("docket_id")
        else:
            logger.debug(f"FR API failed for {doc_number}")

    agencies = rec.get("agencies") or []
    agency_names = ", ".join([
        a.get("name") for a in agencies
        if isinstance(a, dict) and a.get("name") is not None
    ])

    # Eligibility and channel
    # Comment-eligible first: only admit docs with a comment mechanism.
    # RIN is used for lifecycle enrichment later, NOT as standalone eligibility.
    is_prorule = rec.get("type") == "Proposed Rule"
    has_comment_mechanism = bool(comment_url or comments_close_on or regs_document_id)
    eligibility_reason = None

    if is_prorule:
        eligibility_reason = "prorule"
    elif has_comment_mechanism:
        eligibility_reason = "notice-with-comment-period"

    submission_channel = classify_submission_channel(
        comment_url, regs_document_id
    ) if (is_prorule or has_comment_mechanism) else None

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
        # New lifecycle tracking fields
        "rin": rin,
        "rin_all": ",".join(rin_all) if len(rin_all) > 1 else None,
        "docket_id": docket_id,
        "doc_type": doc_type,
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
    
    document_types = ["PRORULE", "NOTICE", "RULE"]

    all_results: List[dict] = list(tqdm(
        fr_client.iter_documents_by_day(
            year,
            fr_sleep=config.fr_sleep,
            limit=config.limit_docs,
            document_types=document_types,
        ),
        desc="FR days",
        unit="doc",
    ))

    if not config.quiet:
        logger.info(f"Total {year} docs fetched (PRORULE + NOTICE + RULE): {len(all_results)}")

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

    # Stage 3: Lifecycle Enrichment (if enabled)
    if getattr(config, "enable_lifecycle_tracking", True):
        rows = _enrich_lifecycle_stage(rows, config)

    df = pd.DataFrame(rows)

    if not config.quiet:
        logger.info(f"Filtered to {len(rows)} comment-eligible documents")

        # Log breakdowns
        for col in ["eligibility_reason", "count_source", "submission_channel", "lifecycle_stage"]:
            if col in df.columns:
                non_null = df[col].dropna()
                if len(non_null) > 0:
                    logger.info(f"\n{col} breakdown:")
                    for val, count in non_null.value_counts().items():
                        logger.info(f"  {val}: {count}")

    return df


def _enrich_lifecycle_stage(
    rows: List[dict],
    config: PipelineConfig,
) -> List[dict]:
    """
    Stage 3: Enrich documents with lifecycle stage from Unified Agenda.

    Fetches Unified Agenda data for each unique RIN and determines the
    lifecycle stage based on agenda stage + document metadata.

    Args:
        rows: List of document records from Stages 1 & 2
        config: Pipeline configuration

    Returns:
        Updated rows with lifecycle_stage, unified_agenda_stage, rin_timetable fields.
    """
    # Import here to avoid circular imports and allow module to work without reginfo
    try:
        from stratification_scripts.reginfo.client import RegInfoClient
    except ImportError:
        logger.warning("RegInfo client not available - skipping lifecycle enrichment")
        for rec in rows:
            rec["lifecycle_stage"] = "UNKNOWN"
            rec["unified_agenda_stage"] = None
            rec["rin_timetable"] = None
        return rows

    log_banner(logger, "STAGE 3: LIFECYCLE ENRICHMENT")

    # Collect unique RINs
    rins = set()
    for rec in rows:
        rin = rec.get("rin")
        if rin:
            rins.add(rin)

    logger.info(f"Found {len(rins)} unique RINs to enrich")

    if not rins:
        logger.info("No RINs found - setting all lifecycle stages to NO_RIN")
        for rec in rows:
            rec["lifecycle_stage"] = "NO_RIN"
            rec["unified_agenda_stage"] = None
            rec["rin_timetable"] = None
        return rows

    # Fetch Unified Agenda data for each RIN
    reginfo_client = RegInfoClient(
        sleep_between=getattr(config, "unified_agenda_sleep", 1.0),
        timeout=getattr(config, "unified_agenda_timeout", 30.0),
        failed_log_path=get_data_dir() / f"failed_rins_{config.year}.txt",
    )

    agenda_cache: Dict[str, dict] = {}
    failed_rins = 0

    for rin in tqdm(rins, desc="Fetching lifecycle stages", unit="rin"):
        try:
            agenda = reginfo_client.fetch_unified_agenda(rin)
            if agenda:
                agenda_cache[rin] = agenda
            else:
                failed_rins += 1
        except Exception as e:
            logger.debug(f"Error fetching agenda for {rin}: {e}")
            failed_rins += 1

    reginfo_client.close()
    reginfo_client.flush_failed_rins()

    logger.info(f"Fetched agenda data for {len(agenda_cache)}/{len(rins)} RINs")
    if failed_rins > 0:
        logger.warning(f"{failed_rins} RINs not found in Unified Agenda")

    # Enrich documents with lifecycle stage
    for rec in rows:
        rin = rec.get("rin")

        if not rin:
            rec["lifecycle_stage"] = "NO_RIN"
            rec["unified_agenda_stage"] = None
            rec["rin_timetable"] = None
            rec.update(RegInfoClient._empty_timeline_columns())
            continue

        agenda = agenda_cache.get(rin)

        if not agenda:
            rec["lifecycle_stage"] = "AGENDA_NOT_FOUND"
            rec["unified_agenda_stage"] = None
            rec["rin_timetable"] = None
            rec.update(RegInfoClient._empty_timeline_columns())
            continue

        # Determine lifecycle stage
        doc_type = rec.get("doc_type") or rec.get("eligibility_reason", "")
        has_open_comments = bool(rec.get("comment_url"))

        lifecycle_stage = reginfo_client.determine_lifecycle_stage(
            agenda_data=agenda,
            doc_type=doc_type,
            has_open_comments=has_open_comments,
        )

        rec["lifecycle_stage"] = lifecycle_stage
        rec["unified_agenda_stage"] = agenda.get("stage_raw") or agenda.get("stage")
        rec["rin_timetable"] = json.dumps(agenda.get("timetable", []))
        timeline = RegInfoClient.extract_structured_timeline(agenda.get("timetable", []))
        # UA stage gives us a lifecycle label even when timetable is empty;
        # upgrade from "none" to "label_only" for these docs.
        if timeline["timeline_confidence"] == "none":
            timeline["timeline_confidence"] = "label_only"
        rec.update(timeline)

    # Compute comment_period_days from FR metadata (all docs, not just RIN-bearing)
    for rec in rows:
        close_str = rec.get("comments_close_on")
        pub_str = rec.get("publication_date")
        if close_str and pub_str:
            try:
                close_dt = datetime.fromisoformat(str(close_str)[:10])
                pub_dt = datetime.fromisoformat(str(pub_str)[:10])
                rec["comment_period_days"] = (close_dt - pub_dt).days
            except (ValueError, TypeError):
                rec["comment_period_days"] = None
        else:
            rec["comment_period_days"] = None

    # Docket ID fallback: recover lifecycle data for NO_RIN docs that share
    # a docket_id with an enriched doc that has a RIN.
    docket_to_rin = {}
    for rec in rows:
        if rec.get("rin") and rec.get("docket_id") and rec["lifecycle_stage"] not in ("NO_RIN", "AGENDA_NOT_FOUND"):
            docket_to_rin.setdefault(rec["docket_id"], rec["rin"])

    no_rin_rows = [r for r in rows if r.get("lifecycle_stage") == "NO_RIN" and r.get("docket_id")]
    recovered = 0
    for rec in no_rin_rows:
        matched_rin = docket_to_rin.get(rec["docket_id"])
        if matched_rin and matched_rin in agenda_cache:
            agenda = agenda_cache[matched_rin]
            doc_type = rec.get("doc_type") or rec.get("eligibility_reason", "")
            has_open_comments = bool(rec.get("comment_url"))
            rec["lifecycle_stage"] = reginfo_client.determine_lifecycle_stage(
                agenda_data=agenda,
                doc_type=doc_type,
                has_open_comments=has_open_comments,
            )
            rec["unified_agenda_stage"] = agenda.get("stage_raw") or agenda.get("stage")
            rec["rin_timetable"] = json.dumps(agenda.get("timetable", []))
            timeline = RegInfoClient.extract_structured_timeline(agenda.get("timetable", []))
            if timeline["timeline_confidence"] == "none":
                timeline["timeline_confidence"] = "label_only"
            rec.update(timeline)
            rec["rin"] = matched_rin  # Inherit the RIN
            recovered += 1
            logger.info(
                f"Recovered {rec.get('document_number')}: "
                f"docket {rec['docket_id']} -> RIN {matched_rin} ({rec['lifecycle_stage']})"
            )

    if no_rin_rows:
        logger.info(
            f"Docket ID fallback: {recovered}/{len(no_rin_rows)} NO_RIN docs "
            f"recovered via shared docket_id ({len(docket_to_rin)} docket-to-RIN mappings available)"
        )

    # Log lifecycle stage distribution
    stage_counts = Counter(rec.get("lifecycle_stage") for rec in rows)
    logger.info("\nLifecycle stage breakdown:")
    for stage, count in stage_counts.most_common():
        logger.info(f"  {stage}: {count}")

    return rows


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

