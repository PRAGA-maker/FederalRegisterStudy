#!/usr/bin/env python3
"""
Diagnostic script: Verify Gemini empty-response storm fix.

Runs 15 comments (stratified by text length) through the FIXED GeminiResponseTracker
and measures per-comment timing and retry behavior.

Expected: empty-response comments should now resolve in ~2 retries (~10s extra)
instead of 10 retries (~486s extra).
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

from stratification_scripts.gemini_client import GeminiResponseTracker
from stratification_scripts.logging_utils import setup_logging, get_logger

setup_logging(verbose=True)
logger = get_logger(__name__)


async def timed_track_response(tracker, comment_text, metadata, semaphore):
    """Wrap track_response with per-comment timing."""
    comment_id = metadata.get("comment_id", "unknown")
    start = time.monotonic()

    try:
        result = await tracker.track_response(comment_text, metadata, semaphore)
        elapsed = time.monotonic() - start

        _, parsed, raw = result
        return {
            "comment_id": comment_id,
            "elapsed_seconds": round(elapsed, 2),
            "response_found": parsed.get("response_found", "unknown"),
            "agency_decision": parsed.get("agency_decision", "unknown"),
            "raw_prefix": raw[:300] if raw else "EMPTY",
            "is_error": raw.startswith("ERROR:") if raw else False,
            "is_empty_response": raw.startswith("EMPTY_RESPONSE:") if raw else False,
            "comment_text_length": len(comment_text),
        }
    except Exception as e:
        elapsed = time.monotonic() - start
        return {
            "comment_id": comment_id,
            "elapsed_seconds": round(elapsed, 2),
            "response_found": "exception",
            "agency_decision": "exception",
            "raw_prefix": f"EXCEPTION: {type(e).__name__}: {str(e)[:200]}",
            "is_error": True,
            "is_empty_response": False,
            "comment_text_length": len(comment_text),
        }


async def run_diagnostic(comments_to_test, api_key, max_concurrency=5):
    """Run diagnostic with sequential processing for clean timing."""
    tracker = GeminiResponseTracker(
        api_key=api_key,
        model="gemini-3-flash-preview",
        max_retries=10,  # Match production — the fix caps EMPTY retries at 2 internally
        enable_search=True,
    )

    semaphore = asyncio.Semaphore(max_concurrency)

    results = []
    for i, (comment_text, metadata) in enumerate(comments_to_test):
        result = await timed_track_response(tracker, comment_text, metadata, semaphore)
        results.append(result)
        marker = ""
        if result["is_empty_response"]:
            marker = " [EMPTY_RESPONSE_CAPPED]"
        elif result["is_error"]:
            marker = " [ERROR]"
        print(
            f"[{i+1}/{len(comments_to_test)}] {result['comment_id']}: "
            f"{result['elapsed_seconds']}s | "
            f"found={result['response_found']} | "
            f"text_len={result['comment_text_length']}{marker}"
        )

    return results


def select_test_comments(n=15):
    """Select stratified sample of comments for testing."""
    df_raw = pl.read_csv(
        "C:/Users/prapa/Documents/GitHub/FederalRegisterStudy/stratification_scripts/makeup/data/comments_raw_2025.csv"
    )
    df_raw = df_raw.with_columns(
        pl.col("comment_text").str.len_chars().alias("text_length")
    )

    # Intentionally over-sample tiny comments (most likely to trigger storms)
    tiny = df_raw.filter(
        pl.col("comment_text").is_null() | (pl.col("text_length") < 20)
    ).sample(min(5, n // 3), seed=42)

    short = df_raw.filter(
        (pl.col("text_length") >= 20) & (pl.col("text_length") < 200)
    ).sample(min(5, n // 3), seed=42)

    long = df_raw.filter(
        pl.col("text_length") >= 200
    ).sample(min(5, n // 3), seed=42)

    selected = pl.concat([tiny, short, long])

    print(f"Selected {len(selected)} comments:")
    print(f"  Tiny (<20 chars): {len(tiny)}")
    print(f"  Short (20-200): {len(short)}")
    print(f"  Long (>200): {len(long)}")

    comments = []
    for row in selected.iter_rows(named=True):
        text = str(row.get("comment_text", "") or "")
        metadata = {
            "comment_id": str(row.get("comment_id", "unknown")),
            "document_number": str(row.get("document_number", "N/A")),
            "agency": "Test Agency",
            "commenter_type": str(row.get("submitter_type", "N/A")),
            "submission_date": str(row.get("posted_date", "N/A")),
        }
        comments.append((text, metadata))

    return comments


def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: Set GEMINI_API_KEY environment variable")
        sys.exit(1)

    print("=" * 70)
    print("GEMINI STORM FIX VERIFICATION")
    print("=" * 70)
    print(f"Fix: MAX_EMPTY_RESPONSE_RETRIES = {GeminiResponseTracker.MAX_EMPTY_RESPONSE_RETRIES}")
    print(f"Empty responses now capped at 2 retries with short backoff (~3.5s)")
    print(f"Previously: 10 retries with exponential backoff (~486s)")
    print()

    comments = select_test_comments(n=15)

    print(f"\nRunning {len(comments)} comments through Gemini (sequential)...")
    print()

    start_total = time.monotonic()
    results = asyncio.run(run_diagnostic(comments, api_key, max_concurrency=1))
    total_time = time.monotonic() - start_total

    # Analyze
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    times = [r["elapsed_seconds"] for r in results]
    errors = [r for r in results if r["is_error"]]
    empty_capped = [r for r in results if r["is_empty_response"]]

    results_sorted = sorted(results, key=lambda r: r["elapsed_seconds"], reverse=True)

    print(f"\nTotal wall time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Total comments: {len(results)}")
    print(f"Errors: {len(errors)}")
    print(f"Empty response (capped at 2 retries): {len(empty_capped)}")

    print(f"\nPer-comment timing distribution:")
    print(f"  min:    {min(times):.1f}s")
    print(f"  median: {sorted(times)[len(times)//2]:.1f}s")
    print(f"  mean:   {sum(times)/len(times):.1f}s")
    print(f"  max:    {max(times):.1f}s")

    STORM_THRESHOLD_S = 30
    storms = [r for r in results if r["elapsed_seconds"] > STORM_THRESHOLD_S]
    storm_time = sum(r["elapsed_seconds"] for r in storms)
    total_comment_time = sum(times)

    print(f"\nSlow comments (>{STORM_THRESHOLD_S}s):")
    print(f"  Count: {len(storms)}/{len(results)} ({100*len(storms)/len(results):.1f}%)")
    print(f"  Time: {storm_time:.1f}s / {total_comment_time:.1f}s ({100*storm_time/total_comment_time:.1f}% of total)")

    if empty_capped:
        print(f"\nEmpty-response details (CAPPED by fix):")
        for r in empty_capped:
            print(f"  {r['comment_id']}: {r['elapsed_seconds']}s | text_len={r['comment_text_length']}")
            print(f"    raw: {r['raw_prefix'][:200]}")

    print(f"\nTop 5 slowest:")
    for r in results_sorted[:5]:
        tag = " [EMPTY_CAPPED]" if r["is_empty_response"] else (" [ERROR]" if r["is_error"] else "")
        print(f"  {r['comment_id']}: {r['elapsed_seconds']}s | len={r['comment_text_length']} | found={r['response_found']}{tag}")

    # Estimate savings
    if empty_capped:
        old_time_per_storm = 486  # 10 retries with exponential backoff
        new_time_per_storm = sum(r["elapsed_seconds"] for r in empty_capped) / len(empty_capped)
        savings_per_storm = old_time_per_storm - new_time_per_storm
        storm_rate = len(empty_capped) / len(results)
        print(f"\nEstimated production savings (per 1000 comments):")
        print(f"  Storm rate: {storm_rate*100:.1f}%")
        print(f"  Old time per storm: ~{old_time_per_storm}s")
        print(f"  New time per storm: ~{new_time_per_storm:.0f}s")
        print(f"  Savings per storm: ~{savings_per_storm:.0f}s")
        print(f"  Expected storms per 1000: {storm_rate*1000:.0f}")
        print(f"  Total savings per 1000: ~{savings_per_storm*storm_rate*1000/3600:.1f} hours")

    # Save results
    output_path = "diagnose_storms_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "total_wall_time_s": round(total_time, 2),
            "total_comment_time_s": round(total_comment_time, 2),
            "comments_tested": len(results),
            "empty_response_capped": len(empty_capped),
            "errors": len(errors),
            "max_empty_retries": GeminiResponseTracker.MAX_EMPTY_RESPONSE_RETRIES,
            "results": results,
        }, f, indent=2)
    print(f"\nFull results saved to {output_path}")


if __name__ == "__main__":
    main()
