"""
Phase 2.5A/B Enrichment Funnel Validation Script.

Reads a Step 1 output CSV and produces:
- Enrichment funnel with counts at each stage
- Lifecycle stage distribution
- New field population checks
- Duration outlier analysis
- Agency enrichment breakdown
- Proposed Rules focus
- Spot-check sample selection

Usage:
    python stratification_scripts/output/validate_enrichment.py <csv_path>
"""

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median


def load_csv(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def is_nonempty(val: str | None) -> bool:
    return val is not None and val.strip() != ""


def print_section(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def enrichment_funnel(rows: list[dict]):
    print_section("ENRICHMENT FUNNEL")
    total = len(rows)
    # All rows should be comment-eligible (pipeline filters before saving)
    eligible = total  # by design

    has_rin = [r for r in rows if is_nonempty(r.get("rin"))]
    rin_in_ua = [r for r in has_rin if r["lifecycle_stage"] not in ("NO_RIN", "AGENDA_NOT_FOUND")]
    has_timetable = [r for r in rin_in_ua if int(r.get("timetable_action_count", 0) or 0) > 0]
    full_timeline = [r for r in rin_in_ua if r.get("timeline_confidence") == "full"]
    has_citations = [
        r for r in rin_in_ua
        if is_nonempty(r.get("nprm_citation")) or is_nonempty(r.get("final_action_citation"))
    ]

    stages = [
        ("Total FR docs (comment-eligible)", eligible, total),
        ("Has RIN", len(has_rin), total),
        ("RIN found in Unified Agenda", len(rin_in_ua), total),
        ("Has timetable (action_count > 0)", len(has_timetable), total),
        ("Has full timeline (confidence=full)", len(full_timeline), total),
        ("Has FR citations in timetable", len(has_citations), total),
    ]

    for label, count, denom in stages:
        pct = (count / denom * 100) if denom else 0
        print(f"  {label}: {count}/{denom} ({pct:.1f}%)")

    # Also show % of RIN-bearing docs at each stage
    if has_rin:
        print(f"\n  --- Among RIN-bearing docs ({len(has_rin)}): ---")
        print(f"  RIN in UA: {len(rin_in_ua)}/{len(has_rin)} ({len(rin_in_ua)/len(has_rin)*100:.1f}%)")
        print(f"  Has timetable: {len(has_timetable)}/{len(has_rin)} ({len(has_timetable)/len(has_rin)*100:.1f}%)")
        print(f"  Full timeline: {len(full_timeline)}/{len(has_rin)} ({len(full_timeline)/len(has_rin)*100:.1f}%)")
        print(f"  Has citations: {len(has_citations)}/{len(has_rin)} ({len(has_citations)/len(has_rin)*100:.1f}%)")


def lifecycle_distribution(rows: list[dict]):
    print_section("LIFECYCLE STAGE DISTRIBUTION")
    counts = Counter(r["lifecycle_stage"] for r in rows)
    for stage, count in counts.most_common():
        pct = count / len(rows) * 100
        print(f"  {stage}: {count} ({pct:.1f}%)")


def timeline_confidence_distribution(rows: list[dict]):
    print_section("TIMELINE CONFIDENCE DISTRIBUTION")
    counts = Counter(r.get("timeline_confidence", "missing") for r in rows)
    for conf, count in counts.most_common():
        pct = count / len(rows) * 100
        print(f"  {conf}: {count} ({pct:.1f}%)")


def new_field_population(rows: list[dict]):
    print_section("NEW FIELD POPULATION (Agent B additions)")

    # effective_date
    has_eff = [r for r in rows if is_nonempty(r.get("effective_date"))]
    print(f"\n  effective_date populated: {len(has_eff)}/{len(rows)}")

    # Where effective_date differs from final_action_date
    print("  Docs where effective_date != final_action_date:")
    for r in has_eff:
        fad = r.get("final_action_date", "")
        ed = r.get("effective_date", "")
        if ed != fad:
            print(f"    {r['document_number']} (RIN {r['rin']}): final={fad}, effective={ed}")

    # has_extended_comment
    extended = [r for r in rows if r.get("has_extended_comment") == "True"]
    print(f"\n  has_extended_comment=True: {len(extended)}/{len(rows)}")
    for r in extended:
        print(f"    {r['document_number']} (RIN {r['rin']}): extended_date={r.get('comment_extended_date')}")

    # has_reopened_comment
    reopened = [r for r in rows if r.get("has_reopened_comment") == "True"]
    print(f"\n  has_reopened_comment=True: {len(reopened)}/{len(rows)}")
    for r in reopened:
        print(f"    {r['document_number']} (RIN {r['rin']}): reopened_date={r.get('comment_reopened_date')}")

    # timetable_sequence unique values
    print("\n  timetable_sequence unique values:")
    seqs = Counter(r.get("timetable_sequence", "") or "EMPTY" for r in rows)
    for seq, count in seqs.most_common():
        print(f"    [{count}x] {seq}")


def duration_outlier_check(rows: list[dict]):
    print_section("DURATION ANALYSIS: nprm_to_final_days")

    durations = []
    for r in rows:
        val = r.get("nprm_to_final_days", "")
        if is_nonempty(val):
            try:
                durations.append((float(val), r))
            except ValueError:
                print(f"  WARNING: non-numeric nprm_to_final_days: {val} (doc {r['document_number']})")

    if not durations:
        print("  No durations found.")
        return

    vals = [d[0] for d in durations]
    print(f"  Count: {len(vals)}")
    print(f"  Min: {min(vals):.0f} days")
    print(f"  Max: {max(vals):.0f} days")
    print(f"  Mean: {mean(vals):.0f} days")
    print(f"  Median: {median(vals):.0f} days")

    # Outliers > 5000 days (~14 years)
    outliers = [(v, r) for v, r in durations if v > 5000]
    print(f"\n  Outliers (> 5000 days): {len(outliers)}")
    for v, r in outliers:
        print(f"    {r['document_number']} (RIN {r['rin']}): {v:.0f} days")
        print(f"      nprm_date={r['nprm_date']}, final_date={r['final_action_date']}")
        print(f"      title: {r['title'][:80]}")

    # Also flag anything > 3000 as notable
    notable = [(v, r) for v, r in durations if 3000 < v <= 5000]
    if notable:
        print(f"\n  Notable (3000-5000 days): {len(notable)}")
        for v, r in notable:
            print(f"    {r['document_number']} (RIN {r['rin']}): {v:.0f} days")


def agency_breakdown(rows: list[dict]):
    print_section("AGENCY ENRICHMENT BREAKDOWN")

    agency_data = defaultdict(lambda: {"total": 0, "has_rin": 0, "full_timeline": 0})
    for r in rows:
        agency = r["agency"].split(",")[0].strip()  # first agency only
        agency_data[agency]["total"] += 1
        if is_nonempty(r.get("rin")):
            agency_data[agency]["has_rin"] += 1
        if r.get("timeline_confidence") == "full":
            agency_data[agency]["full_timeline"] += 1

    # Sort by total docs descending
    sorted_agencies = sorted(agency_data.items(), key=lambda x: x[1]["total"], reverse=True)
    print(f"  {'Agency':<45} {'Total':>5} {'RIN%':>6} {'Full%':>6}")
    print(f"  {'-'*45} {'-'*5} {'-'*6} {'-'*6}")
    for agency, d in sorted_agencies:
        rin_pct = d["has_rin"] / d["total"] * 100 if d["total"] else 0
        full_pct = d["full_timeline"] / d["total"] * 100 if d["total"] else 0
        print(f"  {agency[:45]:<45} {d['total']:>5} {rin_pct:>5.0f}% {full_pct:>5.0f}%")


def proposed_rules_focus(rows: list[dict]):
    print_section("PROPOSED RULES FOCUS")

    prorules = [r for r in rows if r["doc_type"] == "Proposed Rule"]
    total_pr = len(prorules)
    has_rin_pr = [r for r in prorules if is_nonempty(r.get("rin"))]
    rin_in_ua_pr = [r for r in has_rin_pr if r["lifecycle_stage"] not in ("NO_RIN", "AGENDA_NOT_FOUND")]
    full_pr = [r for r in prorules if r.get("timeline_confidence") == "full"]

    print(f"  Total Proposed Rules: {total_pr}")
    print(f"  Has RIN: {len(has_rin_pr)}/{total_pr} ({len(has_rin_pr)/total_pr*100:.1f}%)" if total_pr else "  No Proposed Rules found")
    print(f"  RIN in UA: {len(rin_in_ua_pr)}/{total_pr} ({len(rin_in_ua_pr)/total_pr*100:.1f}%)" if total_pr else "")
    print(f"  Full timeline: {len(full_pr)}/{total_pr} ({len(full_pr)/total_pr*100:.1f}%)" if total_pr else "")

    print(f"\n  Individual Proposed Rules:")
    for r in prorules:
        rin = r.get("rin", "none") or "none"
        conf = r.get("timeline_confidence", "?")
        stage = r.get("lifecycle_stage", "?")
        print(f"    {r['document_number']} RIN={rin} stage={stage} conf={conf}")
        print(f"      {r['title'][:70]}")


def spot_check_selection(rows: list[dict]):
    print_section("SPOT-CHECK SAMPLE SELECTION")
    print("  For each lifecycle stage: up to 3 RIN-bearing docs for manual reginfo.gov verification")
    print()

    enriched = [r for r in rows if is_nonempty(r.get("rin")) and r["lifecycle_stage"] not in ("NO_RIN",)]
    by_stage = defaultdict(list)
    for r in enriched:
        by_stage[r["lifecycle_stage"]].append(r)

    for stage, docs in sorted(by_stage.items()):
        print(f"  --- {stage} ({len(docs)} docs) ---")
        for r in docs[:3]:
            print(f"    doc={r['document_number']}  RIN={r['rin']}")
            print(f"    title: {r['title'][:70]}")
            print(f"    timetable_seq: {r.get('timetable_sequence', 'NONE')}")
            print(f"    nprm_date={r.get('nprm_date', '')}  final={r.get('final_action_date', '')}  eff={r.get('effective_date', '')}")
            print(f"    URL: https://www.reginfo.gov/public/do/eAgendaViewRule?pubId=current&RIN={r['rin']}")
            print()


def main():
    if len(sys.argv) < 2:
        csv_path = Path(__file__).parent / "federal_register_2015_comments.csv"
    else:
        csv_path = Path(sys.argv[1])

    if not csv_path.exists():
        print(f"ERROR: CSV not found at {csv_path}")
        sys.exit(1)

    rows = load_csv(str(csv_path))
    print(f"Loaded {len(rows)} rows from {csv_path.name}")
    print(f"Columns: {len(rows[0]) if rows else 0}")

    enrichment_funnel(rows)
    lifecycle_distribution(rows)
    timeline_confidence_distribution(rows)
    new_field_population(rows)
    duration_outlier_check(rows)
    agency_breakdown(rows)
    proposed_rules_focus(rows)
    spot_check_selection(rows)


if __name__ == "__main__":
    main()
