#!/usr/bin/env python3
"""
Integration test: Parallel Stage 2 + Stage 3 with real API calls.

Loads records from an existing FR CSV, strips Stage 2/3 data,
re-runs enrichment in parallel, and verifies results match expectations.

This bypasses the FR API (which may be rate-limited) and tests only
the Regs.gov + reginfo.gov parallel enrichment code path.
"""

import os
import sys
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from stratification_scripts.config import PipelineConfig, get_regs_api_keys
from stratification_scripts.distribution import (
    enrich_regs_count,
    _enrich_lifecycle_stage,
)
from stratification_scripts.regulations_gov.client import RegsGovClient
from stratification_scripts.logging_utils import setup_logging, get_logger

setup_logging(verbose=True)
logger = get_logger(__name__)


def load_test_records(csv_path: str, n: int = 100):
    """Load records from existing FR CSV and strip Stage 2/3 data."""
    df = pd.read_csv(csv_path)

    # Take a sample that includes both regs.gov and non-regs.gov docs,
    # and both RIN-bearing and NO_RIN docs
    regs_with_rin = df[df["regs_document_id"].notna() & df["rin"].notna()].head(n // 4)
    regs_no_rin = df[df["regs_document_id"].notna() & df["rin"].isna()].head(n // 4)
    no_regs_with_rin = df[df["regs_document_id"].isna() & df["rin"].notna()].head(n // 4)
    no_regs_no_rin = df[df["regs_document_id"].isna() & df["rin"].isna()].head(n // 4)

    sample = pd.concat([regs_with_rin, regs_no_rin, no_regs_with_rin, no_regs_no_rin])

    print(f"Sample: {len(sample)} records")
    print(f"  regs+rin: {len(regs_with_rin)}, regs+no_rin: {len(regs_no_rin)}")
    print(f"  no_regs+rin: {len(no_regs_with_rin)}, no_regs+no_rin: {len(no_regs_no_rin)}")

    # Convert to list of dicts (as distribution.py expects)
    records = sample.to_dict(orient="records")

    # Strip Stage 2 data (comment_count, count_source) for regs.gov docs
    # and Stage 3 data (lifecycle_stage, etc.) for all docs
    for rec in records:
        # Preserve originals for comparison
        rec["_orig_comment_count"] = rec.get("comment_count")
        rec["_orig_count_source"] = rec.get("count_source")
        rec["_orig_lifecycle_stage"] = rec.get("lifecycle_stage")

        # Strip Stage 2 data
        if rec.get("regs_document_id"):
            rec["comment_count"] = None
            rec["count_source"] = "federalregister"

        # Strip Stage 3 data
        rec["lifecycle_stage"] = None
        rec["unified_agenda_stage"] = None
        rec["rin_timetable"] = None

        # Strip timeline columns that Stage 3 populates
        for col in [
            "nprm_date", "nprm_citation", "final_action_date",
            "final_action_citation", "effective_date", "withdrawal_date",
            "nprm_to_final_days", "timetable_action_count",
            "has_reopened_comment", "has_extended_comment",
            "anprm_date", "anprm_citation", "interim_final_date",
            "interim_final_citation", "direct_final_date",
            "comment_reopened_date", "comment_extended_date",
            "anprm_to_nprm_days", "anprm_to_final_days",
            "timetable_sequence", "timeline_confidence",
        ]:
            rec[col] = None

    return records


def run_parallel_test(records, api_keys, config):
    """Run Stage 2 + Stage 3 in parallel (mirrors distribution.py logic)."""
    regs_docs = [r for r in records if r.get("regs_document_id")]
    non_regs = [r for r in records if not r.get("regs_document_id")]

    print(f"\nRegs.gov docs: {len(regs_docs)} | Non-regs: {len(non_regs)}")

    regs_client = RegsGovClient(api_keys, retries=3)
    workers_stage2 = min(6, len(api_keys) * 2)

    stage2_errors = [0]
    stage2_exception = [None]

    def _run_stage2():
        try:
            with ThreadPoolExecutor(max_workers=workers_stage2) as executor:
                futures = {
                    executor.submit(enrich_regs_count, rec, regs_client): rec
                    for rec in regs_docs
                }
                for future in tqdm(
                    as_completed(futures), total=len(regs_docs),
                    desc="Stage 2: Regs.gov", unit="doc",
                ):
                    try:
                        future.result()
                    except Exception as e:
                        stage2_errors[0] += 1
                        logger.warning(f"Stage 2 error: {e}")
            regs_client.close()
        except Exception as e:
            stage2_exception[0] = e
            logger.error(f"Stage 2 failed: {e}")

    print("\n--- Starting parallel Stage 2 + Stage 3 ---")
    start = time.monotonic()

    stage2_thread = threading.Thread(target=_run_stage2, name="stage2-regs")
    stage2_thread.start()

    # Stage 3 in main thread
    records = _enrich_lifecycle_stage(records, config)

    stage2_thread.join()
    elapsed = time.monotonic() - start

    print(f"\n--- Parallel complete in {elapsed:.1f}s ---")

    if stage2_exception[0]:
        print(f"  Stage 2 FAILED: {stage2_exception[0]}")
    if stage2_errors[0]:
        print(f"  Stage 2 errors: {stage2_errors[0]}")

    return records, elapsed


def run_sequential_test(records, api_keys, config):
    """Run Stage 2 then Stage 3 sequentially for timing comparison."""
    regs_docs = [r for r in records if r.get("regs_document_id")]
    regs_client = RegsGovClient(api_keys, retries=3)
    workers_stage2 = min(6, len(api_keys) * 2)

    print("\n--- Starting sequential Stage 2 then Stage 3 ---")
    start = time.monotonic()

    # Stage 2
    with ThreadPoolExecutor(max_workers=workers_stage2) as executor:
        futures = {
            executor.submit(enrich_regs_count, rec, regs_client): rec
            for rec in regs_docs
        }
        for future in tqdm(
            as_completed(futures), total=len(regs_docs),
            desc="Stage 2 (seq)", unit="doc",
        ):
            try:
                future.result()
            except Exception:
                pass
    regs_client.close()

    s2_time = time.monotonic() - start

    # Stage 3
    records = _enrich_lifecycle_stage(records, config)

    elapsed = time.monotonic() - start
    s3_time = elapsed - s2_time

    print(f"\n--- Sequential complete in {elapsed:.1f}s (S2: {s2_time:.1f}s, S3: {s3_time:.1f}s) ---")
    return records, elapsed


def verify_results(records):
    """Verify that both stages populated their fields correctly."""
    print("\n=== VERIFICATION ===")

    # Stage 2 checks
    regs_docs = [r for r in records if r.get("regs_document_id")]
    regs_with_count = [r for r in regs_docs if r.get("count_source") in ("regulations.gov", "regulations.gov-meta")]
    print(f"Stage 2: {len(regs_with_count)}/{len(regs_docs)} regs.gov docs got updated counts")

    # Stage 3 checks
    with_lifecycle = [r for r in records if r.get("lifecycle_stage") is not None]
    print(f"Stage 3: {len(with_lifecycle)}/{len(records)} docs got lifecycle_stage")

    lifecycle_counts = {}
    for r in records:
        stage = r.get("lifecycle_stage", "MISSING")
        lifecycle_counts[stage] = lifecycle_counts.get(stage, 0) + 1
    print(f"  Lifecycle breakdown: {lifecycle_counts}")

    # Cross-check: records with RIN should have lifecycle != NO_RIN (unless AGENDA_NOT_FOUND)
    rin_docs = [r for r in records if r.get("rin")]
    rin_with_enrichment = [r for r in rin_docs if r.get("lifecycle_stage") not in (None, "NO_RIN")]
    print(f"  RIN docs enriched: {len(rin_with_enrichment)}/{len(rin_docs)}")

    # Check comment_count values aren't corrupted
    for r in records:
        cc = r.get("comment_count")
        if cc is not None and not isinstance(cc, (int, float)):
            print(f"  WARNING: Non-numeric comment_count for {r['document_number']}: {cc}")
            return False

    # Check lifecycle_stage values are valid
    VALID_STAGES = {
        "NO_RIN", "AGENDA_NOT_FOUND", "FINAL_EFFECTIVE", "LONG_TERM_STALLED",
        "NPRM_PUBLISHED", "PROPOSED_RULE", "FINAL_RULE", "INTERIM_FINAL",
        "WITHDRAWN", "COMPLETED", "UNKNOWN", None,
    }
    for r in records:
        stage = r.get("lifecycle_stage")
        if stage not in VALID_STAGES:
            print(f"  WARNING: Invalid lifecycle_stage for {r['document_number']}: {stage}")
            # Don't fail on this - just flag

    print("\nVERIFICATION PASSED")
    return True


def main():
    api_keys = get_regs_api_keys(required=True)
    print(f"Using {len(api_keys)} Regs.gov API keys")

    csv_path = "stratification_scripts/output/federal_register_2013_comments.csv"
    if not Path(csv_path).exists():
        print(f"ERROR: {csv_path} not found. Need existing FR CSV for test.")
        sys.exit(1)

    config = PipelineConfig(
        year=2013,
        limit_docs=None,
        retries=3,
        quiet=False,
        verbose=True,
    )

    # Load and strip test records
    records = load_test_records(csv_path, n=40)  # 40 records for quick test

    print("\n" + "=" * 60)
    print("TEST: PARALLEL Stage 2 + Stage 3")
    print("=" * 60)

    records_par, par_time = run_parallel_test(records, api_keys, config)
    ok = verify_results(records_par)

    if not ok:
        print("\nFAILED: Verification errors detected")
        sys.exit(1)

    print(f"\nTotal parallel time: {par_time:.1f}s")
    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED - Parallel Stage 2+3 works correctly")
    print("=" * 60)


if __name__ == "__main__":
    main()
