"""
Analyze 2014 full-year lifecycle data from Agent 10's production-scale fetch.
Run: python analyze_2014.py

This script answers key open questions:
1. Lifecycle stage distribution at full-year scale (29K+ docs)
2. Timetable step count distribution (answers "why <2 steps?")
3. Enrichment funnel from total docs to full timelines
4. NPRM-to-Final duration distribution with real N
5. Agency-level enrichment breakdown
"""

import csv
import sys
from collections import Counter
from pathlib import Path

CSV_PATH = Path("stratification_scripts/output/federal_register_2014_comments.csv")


def load_data():
    if not CSV_PATH.exists():
        print(f"ERROR: {CSV_PATH} not found. Agent 10 may not have finished yet.")
        sys.exit(1)

    with open(CSV_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Loaded {len(rows)} documents from {CSV_PATH.name}")
    print(f"Columns: {len(rows[0].keys())}")
    return rows


def enrichment_funnel(rows):
    print("\n" + "=" * 70)
    print("ENRICHMENT FUNNEL")
    print("=" * 70)

    total = len(rows)
    print(f"Total FR docs fetched: {total}")

    # Comment-eligible (all rows in the CSV should be eligible)
    eligible = len(rows)
    print(f"Comment-eligible: {eligible} (100% — CSV only has eligible docs)")

    # Doc type breakdown
    types = Counter(r.get("doc_type", "unknown") for r in rows)
    print(f"\n  Doc type breakdown:")
    for dtype, count in types.most_common():
        print(f"    {dtype}: {count} ({100*count/total:.1f}%)")

    # Has RIN
    has_rin = [r for r in rows if r.get("rin") and r["rin"].strip()]
    print(f"\nHas RIN: {len(has_rin)} ({100*len(has_rin)/total:.1f}%)")
    print(f"No RIN: {total - len(has_rin)} ({100*(total-len(has_rin))/total:.1f}%)")

    # RIN in Unified Agenda (has lifecycle data beyond NO_RIN)
    not_no_rin = [r for r in rows if r.get("lifecycle_stage") not in ("NO_RIN", "")]
    print(f"\nRIN found in UA (has lifecycle): {len(not_no_rin)} ({100*len(not_no_rin)/total:.1f}%)")

    agenda_not_found = [r for r in has_rin if r.get("lifecycle_stage") == "AGENDA_NOT_FOUND"]
    print(f"  RIN but AGENDA_NOT_FOUND: {len(agenda_not_found)} ({100*len(agenda_not_found)/len(has_rin):.1f}% of RIN-bearing)" if has_rin else "")

    # Has timetable
    has_timetable = [r for r in rows if int(float(r.get("timetable_action_count", "0"))) > 0]
    print(f"\nHas timetable (action_count > 0): {len(has_timetable)} ({100*len(has_timetable)/total:.1f}%)")

    # Full timeline
    full_timeline = [r for r in rows if r.get("timeline_confidence") == "full"]
    print(f"Full timeline (confidence=full): {len(full_timeline)} ({100*len(full_timeline)/total:.1f}%)")

    # Has FR citations
    has_nprm_citation = [r for r in rows if r.get("nprm_citation") and r["nprm_citation"].strip()]
    has_final_citation = [r for r in rows if r.get("final_action_citation") and r["final_action_citation"].strip()]
    has_both_citation = [r for r in rows if
                         (r.get("nprm_citation") and r["nprm_citation"].strip()) and
                         (r.get("final_action_citation") and r["final_action_citation"].strip())]
    print(f"\nHas NPRM citation: {len(has_nprm_citation)}")
    print(f"Has Final Action citation: {len(has_final_citation)}")
    print(f"Has BOTH citations: {len(has_both_citation)}")

    # Document linking
    has_nprm_doc = [r for r in rows if r.get("nprm_document_number") and r["nprm_document_number"].strip()]
    has_final_doc = [r for r in rows if r.get("final_rule_document_number") and r["final_rule_document_number"].strip()]
    print(f"\nnprm_document_number resolved: {len(has_nprm_doc)}")
    print(f"final_rule_document_number resolved: {len(has_final_doc)}")

    # For RIN-bearing docs specifically
    if has_rin:
        rin_full = [r for r in has_rin if r.get("timeline_confidence") == "full"]
        rin_partial = [r for r in has_rin if r.get("timeline_confidence") == "partial"]
        rin_label = [r for r in has_rin if r.get("timeline_confidence") == "label_only"]
        print(f"\n--- RIN-bearing docs ({len(has_rin)}) ---")
        print(f"  full: {len(rin_full)} ({100*len(rin_full)/len(has_rin):.1f}%)")
        print(f"  partial: {len(rin_partial)} ({100*len(rin_partial)/len(has_rin):.1f}%)")
        print(f"  label_only: {len(rin_label)} ({100*len(rin_label)/len(has_rin):.1f}%)")
        rin_none = len(has_rin) - len(rin_full) - len(rin_partial) - len(rin_label)
        print(f"  none: {rin_none} ({100*rin_none/len(has_rin):.1f}%)")


def lifecycle_distribution(rows):
    print("\n" + "=" * 70)
    print("LIFECYCLE STAGE DISTRIBUTION")
    print("=" * 70)

    stages = Counter(r.get("lifecycle_stage", "MISSING") for r in rows)
    for stage, count in stages.most_common():
        pct = 100 * count / len(rows)
        print(f"  {stage}: {count} ({pct:.1f}%)")


def timetable_analysis(rows):
    print("\n" + "=" * 70)
    print("TIMETABLE ACTION COUNT DISTRIBUTION")
    print("=" * 70)

    action_counts = Counter(int(float(r.get("timetable_action_count", "0"))) for r in rows)
    for ac, count in sorted(action_counts.items()):
        pct = 100 * count / len(rows)
        print(f"  {ac} actions: {count} docs ({pct:.1f}%)")

    print("\n--- Among docs WITH timetable (actions > 0) ---")
    with_tt = [r for r in rows if int(float(r.get("timetable_action_count", "0"))) > 0]
    if with_tt:
        action_counts_tt = Counter(int(float(r.get("timetable_action_count", "0"))) for r in with_tt)
        for ac, count in sorted(action_counts_tt.items()):
            pct = 100 * count / len(with_tt)
            print(f"  {ac} actions: {count} docs ({pct:.1f}%)")
        avg = sum(int(float(r.get("timetable_action_count", "0"))) for r in with_tt) / len(with_tt)
        print(f"  Average: {avg:.1f} actions")

    print("\n--- Top timetable sequences ---")
    sequences = Counter(r.get("timetable_sequence", "") for r in rows if r.get("timetable_sequence"))
    for seq, count in sequences.most_common(15):
        print(f"  {seq}: {count}")


def duration_analysis(rows):
    print("\n" + "=" * 70)
    print("NPRM-TO-FINAL DURATION DISTRIBUTION")
    print("=" * 70)

    durations = []
    for r in rows:
        val = r.get("nprm_to_final_days", "")
        if val and val.strip():
            try:
                durations.append(float(val))
            except ValueError:
                pass

    if not durations:
        print("  No NPRM-to-Final data")
        return

    durations.sort()
    n = len(durations)
    print(f"  N: {n}")
    print(f"  Min: {durations[0]:.0f} days ({durations[0]/365:.1f} years)")
    print(f"  Max: {durations[-1]:.0f} days ({durations[-1]/365:.1f} years)")
    print(f"  Mean: {sum(durations)/n:.0f} days ({sum(durations)/n/365:.1f} years)")
    print(f"  Median: {durations[n//2]:.0f} days ({durations[n//2]/365:.1f} years)")
    print(f"  25th: {durations[n//4]:.0f} days")
    print(f"  75th: {durations[3*n//4]:.0f} days")

    # Buckets
    print("\n  Duration buckets:")
    buckets = [(0, 90), (91, 180), (181, 365), (366, 730), (731, 1460), (1461, 99999)]
    labels = ["<3mo", "3-6mo", "6-12mo", "1-2yr", "2-4yr", "4+yr"]
    for (lo, hi), label in zip(buckets, labels):
        count = sum(1 for d in durations if lo <= d <= hi)
        pct = 100 * count / n
        print(f"    {label}: {count} ({pct:.1f}%)")


def comment_analysis(rows):
    print("\n" + "=" * 70)
    print("COMMENT COUNT ANALYSIS")
    print("=" * 70)

    comments = []
    for r in rows:
        cc = r.get("comment_count", "0")
        try:
            comments.append(float(cc))
        except (ValueError, TypeError):
            comments.append(0)

    total = sum(comments)
    with_comments = sum(1 for c in comments if c > 0)
    print(f"  Total comments across all docs: {total:,.0f}")
    print(f"  Docs with >0 comments: {with_comments} ({100*with_comments/len(rows):.1f}%)")
    print(f"  Docs with 0 comments: {len(rows)-with_comments} ({100*(len(rows)-with_comments)/len(rows):.1f}%)")

    if with_comments > 0:
        nonzero = sorted([c for c in comments if c > 0])
        print(f"\n  Among docs with comments (N={len(nonzero)}):")
        print(f"    Min: {nonzero[0]:.0f}")
        print(f"    Max: {nonzero[-1]:,.0f}")
        print(f"    Median: {nonzero[len(nonzero)//2]:.0f}")
        print(f"    Mean: {sum(nonzero)/len(nonzero):,.0f}")

    # By lifecycle stage
    print("\n  Comments by lifecycle stage:")
    stage_comments = {}
    for r in rows:
        stage = r.get("lifecycle_stage", "MISSING")
        cc = float(r.get("comment_count", "0") or "0")
        if stage not in stage_comments:
            stage_comments[stage] = {"count": 0, "total_comments": 0, "with_comments": 0}
        stage_comments[stage]["count"] += 1
        stage_comments[stage]["total_comments"] += cc
        if cc > 0:
            stage_comments[stage]["with_comments"] += 1

    for stage in sorted(stage_comments, key=lambda s: -stage_comments[s]["total_comments"]):
        info = stage_comments[stage]
        pct_with = 100 * info["with_comments"] / info["count"] if info["count"] > 0 else 0
        print(f"    {stage}: {info['count']} docs, {info['total_comments']:,.0f} comments, "
              f"{pct_with:.0f}% have >0")


def agency_enrichment(rows):
    print("\n" + "=" * 70)
    print("AGENCY-LEVEL ENRICHMENT (top 20 by doc count)")
    print("=" * 70)

    agency_data = {}
    for r in rows:
        agency = r.get("agency", "unknown")
        if agency not in agency_data:
            agency_data[agency] = {"total": 0, "has_rin": 0, "full_timeline": 0}
        agency_data[agency]["total"] += 1
        if r.get("rin") and r["rin"].strip():
            agency_data[agency]["has_rin"] += 1
        if r.get("timeline_confidence") == "full":
            agency_data[agency]["full_timeline"] += 1

    sorted_agencies = sorted(agency_data.items(), key=lambda x: -x[1]["total"])
    print(f"  {'Agency':<55} | {'Docs':>5} | {'RIN%':>5} | {'Full%':>5}")
    print(f"  {'-'*55}-+-{'-'*5}-+-{'-'*5}-+-{'-'*5}")
    for agency, data in sorted_agencies[:20]:
        rin_pct = 100 * data["has_rin"] / data["total"]
        full_pct = 100 * data["full_timeline"] / data["total"]
        name = agency[:55]
        print(f"  {name:<55} | {data['total']:>5} | {rin_pct:>4.0f}% | {full_pct:>4.0f}%")


def main():
    rows = load_data()
    enrichment_funnel(rows)
    lifecycle_distribution(rows)
    timetable_analysis(rows)
    duration_analysis(rows)
    comment_analysis(rows)
    agency_enrichment(rows)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
