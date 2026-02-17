"""
Docket-Based Comment Counting Investigation
============================================
Hypothesis: Docket-based comment counting is partially reliable (1:1 dockets OK,
shared dockets not) but the ceiling on recoverable comment volume is tiny because
92.3% of NO_RIN docs get zero comments. Prior expectation: confirms this is NOT
worth building into the pipeline.

Uses 2014 VERIFIED CSV as READ-ONLY profiling data (10,406 docs).
Makes targeted Regs.gov API calls for 15-20 sample dockets + 5 FRDOC_0001.

Run:  python investigation_docket_counting.py
"""

import csv
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "stratification_scripts" / "output"
CSV_2014 = DATA_DIR / "federal_register_2014_comments_VERIFIED.csv"

# Regs.gov API
REGS_API_KEYS = os.environ.get("REGS_API_KEYS", "").split(",")
if not REGS_API_KEYS or not REGS_API_KEYS[0]:
    # Fallback: read from keys file
    keys_file = Path(__file__).parent / "prd" / "keys.txt"
    if keys_file.exists():
        for line in keys_file.read_text().splitlines():
            if "REGS_API_KEYS" in line:
                # Parse PowerShell format: $env:REGS_API_KEYS="key1:rpm,key2:rpm,..."
                match = re.search(r'"([^"]+)"', line)
                if match:
                    raw = match.group(1)
                    REGS_API_KEYS = [entry.split(":")[0] for entry in raw.split(",")]
                    break

API_KEY = REGS_API_KEYS[0] if REGS_API_KEYS else ""
REGS_BASE = "https://api.regulations.gov/v4"

# Docket normalization regex (same as federal_register/client.py:150-153)
_DOCKET_PREFIX_RE = re.compile(
    r"^(?:DHS\s+)?(?:Docket|Doc\.?|Document)\s*(?:No\.?|Number|#:?)\s*",
    re.IGNORECASE,
)

# Regs.gov-style docket ID pattern: AGENCY-YEAR-TYPE-NUMBER or AGENCY_FRDOC_0001
_REGS_DOCKET_RE = re.compile(r"^[A-Z]{2,10}[-_][A-Z0-9]")

# FRDOC_0001 catch-all pattern
_FRDOC_RE = re.compile(r"_FRDOC_0001$")

# Rate limiting
_last_call = 0.0
RATE_LIMIT_SECONDS = 0.8  # ~75 RPM, conservative


def normalize_docket_id(raw):
    """Strip common FR API prefixes from docket IDs for Regs.gov lookup."""
    if not raw or not isinstance(raw, str):
        return raw
    cleaned = _DOCKET_PREFIX_RE.sub("", raw).strip()
    return cleaned if cleaned else raw


def is_regs_format(docket_id):
    """Check if a normalized docket ID looks like a Regs.gov format."""
    if not docket_id:
        return False
    return bool(_REGS_DOCKET_RE.match(docket_id))


def is_frdoc_0001(docket_id):
    """Check if a docket ID is a FRDOC_0001 catch-all."""
    if not docket_id:
        return False
    return bool(_FRDOC_RE.search(docket_id))


def regs_api_get(endpoint, params=None):
    """Make a rate-limited Regs.gov API call."""
    global _last_call
    elapsed = time.time() - _last_call
    if elapsed < RATE_LIMIT_SECONDS:
        time.sleep(RATE_LIMIT_SECONDS - elapsed)

    if params is None:
        params = {}
    params["api_key"] = API_KEY

    url = f"{REGS_BASE}/{endpoint}"
    try:
        resp = requests.get(url, params=params, timeout=30)
        _last_call = time.time()
        if resp.status_code == 429:
            print(f"  [RATE LIMITED] Waiting 60s...")
            time.sleep(60)
            return regs_api_get(endpoint, {k: v for k, v in params.items() if k != "api_key"})
        if resp.status_code != 200:
            return None
        return resp.json()
    except Exception as e:
        print(f"  [ERROR] {e}")
        return None


def query_docket_docs(docket_id):
    """Count non-submission documents in a docket."""
    data = regs_api_get("documents", {
        "filter[docketId]": docket_id,
        "page[size]": 250,  # max allowed
    })
    if not data:
        return 0, []

    docs = []
    for item in (data.get("data") or []):
        attrs = item.get("attributes") or {}
        doc_type = attrs.get("documentType", "")
        if doc_type == "Public Submission":
            continue
        docs.append({
            "id": item.get("id"),
            "type": doc_type,
            "title": (attrs.get("title") or "")[:80],
            "comments": attrs.get("commentCount"),
        })
    total_elements = (data.get("meta") or {}).get("totalElements", len(data.get("data", [])))
    return total_elements, docs


def query_docket_comments(docket_id):
    """Get total comment count for a docket."""
    data = regs_api_get("comments", {
        "filter[docketId]": docket_id,
        "page[size]": 5,
    })
    if not data:
        return None
    return (data.get("meta") or {}).get("totalElements")


# ---------------------------------------------------------------------------
# Part A: Profile the blind spot
# ---------------------------------------------------------------------------
def load_and_profile():
    """Load 2014 VERIFIED CSV and profile the blind spot."""
    print("=" * 70)
    print("PART A: BLIND SPOT PROFILE (2014 VERIFIED CSV)")
    print("=" * 70)

    if not CSV_2014.exists():
        print(f"ERROR: {CSV_2014} not found")
        sys.exit(1)

    rows = []
    with open(CSV_2014, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    total = len(rows)
    print(f"\nTotal docs: {total:,}")

    # Find blind spot docs
    has_docket = [r for r in rows if r.get("docket_id", "").strip()]
    has_regs_id = [r for r in rows if r.get("regs_document_id", "").strip()]
    blind_spot = [
        r for r in rows
        if r.get("docket_id", "").strip() and not r.get("regs_document_id", "").strip()
    ]

    print(f"Docs with docket_id: {len(has_docket):,} ({100*len(has_docket)/total:.1f}%)")
    print(f"Docs with regs_document_id: {len(has_regs_id):,} ({100*len(has_regs_id)/total:.1f}%)")
    print(f"BLIND SPOT (docket_id YES, regs_document_id NO): {len(blind_spot):,} ({100*len(blind_spot)/total:.1f}%)")

    # Normalize and categorize
    categories = {"regs_format": [], "frdoc_0001": [], "non_regs": [], "empty": []}

    for r in blind_spot:
        raw = r["docket_id"]
        normalized = normalize_docket_id(raw)

        if not normalized or normalized == raw and not is_regs_format(normalized):
            # Normalization didn't help and it's not Regs format
            pass

        if is_frdoc_0001(normalized):
            categories["frdoc_0001"].append((r, normalized))
        elif is_regs_format(normalized):
            categories["regs_format"].append((r, normalized))
        elif not normalized:
            categories["empty"].append((r, normalized))
        else:
            categories["non_regs"].append((r, normalized))

    print(f"\n--- Normalization Results ---")
    print(f"Valid Regs.gov format (queryable): {len(categories['regs_format']):,}")
    print(f"FRDOC_0001 catch-all:              {len(categories['frdoc_0001']):,}")
    print(f"Non-Regs.gov format:               {len(categories['non_regs']):,}")
    print(f"Empty after normalization:          {len(categories['empty']):,}")

    # Show non-Regs samples
    if categories["non_regs"]:
        print(f"\n--- Sample non-Regs.gov docket formats (first 15) ---")
        seen = set()
        count = 0
        for r, norm in categories["non_regs"]:
            if norm not in seen and count < 15:
                seen.add(norm)
                print(f"  raw: {r['docket_id']!r:50} -> normalized: {norm!r}")
                count += 1

    # Show FRDOC_0001 distribution
    if categories["frdoc_0001"]:
        frdoc_prefixes = Counter(norm for _, norm in categories["frdoc_0001"])
        print(f"\n--- FRDOC_0001 dockets ({len(categories['frdoc_0001'])} docs, {len(frdoc_prefixes)} distinct) ---")
        for prefix, cnt in frdoc_prefixes.most_common(20):
            print(f"  {prefix}: {cnt} docs")

    # Doc type breakdown of blind spot
    type_counts = Counter(r.get("doc_type", "unknown") for r, _ in
                          categories["regs_format"] + categories["frdoc_0001"] + categories["non_regs"])
    print(f"\n--- Blind spot by doc_type ---")
    for dtype, cnt in type_counts.most_common():
        print(f"  {dtype}: {cnt:,}")

    # Comment count info for blind spot
    comment_counts = []
    for r in blind_spot:
        cc = r.get("comment_count", "")
        try:
            cc_val = float(cc) if cc else 0.0
            comment_counts.append(cc_val)
        except ValueError:
            comment_counts.append(0.0)

    zero_comments = sum(1 for c in comment_counts if c == 0)
    print(f"\n--- Blind spot comment counts ---")
    print(f"Zero comments: {zero_comments:,} of {len(blind_spot):,} ({100*zero_comments/len(blind_spot):.1f}%)")
    print(f"Non-zero comments: {len(blind_spot) - zero_comments:,}")
    if comment_counts:
        print(f"Total comments across blind spot: {sum(comment_counts):,.0f}")
        nonzero = [c for c in comment_counts if c > 0]
        if nonzero:
            print(f"Non-zero: mean={sum(nonzero)/len(nonzero):.1f}, max={max(nonzero):.0f}")

    return categories, blind_spot


# ---------------------------------------------------------------------------
# Part B: API sampling of queryable dockets
# ---------------------------------------------------------------------------
def sample_queryable_dockets(categories):
    """Query 15-20 dockets from the 'regs_format' category."""
    print("\n" + "=" * 70)
    print("PART B: API SAMPLING (15-20 QUERYABLE DOCKETS)")
    print("=" * 70)

    if not API_KEY:
        print("ERROR: No REGS_API_KEY found. Set REGS_API_KEYS env var or check prd/keys.txt")
        return []

    queryable = categories["regs_format"]
    if not queryable:
        print("No queryable dockets found.")
        return []

    # Deduplicate by normalized docket_id, sample up to 20
    seen = {}
    for r, norm in queryable:
        if norm not in seen:
            seen[norm] = r

    sample_ids = list(seen.keys())[:20]
    print(f"Sampling {len(sample_ids)} unique dockets from {len(queryable)} queryable docs\n")

    results = []
    for i, docket_id in enumerate(sample_ids):
        fr_doc = seen[docket_id]
        print(f"[{i+1}/{len(sample_ids)}] {docket_id}")
        print(f"  FR doc: {fr_doc.get('document_number')} - {(fr_doc.get('title') or '')[:60]}")

        total_all_docs, docs = query_docket_docs(docket_id)
        total_comments = query_docket_comments(docket_id)

        non_submission_count = len(docs)

        # Categorize
        if total_comments == 0 or total_comments is None:
            category = "D_moot"
        elif non_submission_count == 0:
            category = "D_moot"
        elif non_submission_count == 1:
            category = "A_trustworthy"
        elif non_submission_count <= 3:
            category = "B_ambiguous"
        else:
            category = "C_untrustworthy"

        result = {
            "docket_id": docket_id,
            "fr_document": fr_doc.get("document_number"),
            "fr_title": (fr_doc.get("title") or "")[:60],
            "total_elements_in_search": total_all_docs,
            "non_submission_docs": non_submission_count,
            "total_comments": total_comments,
            "category": category,
            "docs": docs,
        }
        results.append(result)

        if total_comments is None:
            total_comments_str = "API error"
        else:
            total_comments_str = f"{total_comments:,}"

        print(f"  Docs in docket (non-submission): {non_submission_count} (total elements: {total_all_docs})")
        print(f"  Total comments on docket: {total_comments_str}")
        print(f"  Category: {category}")

        if docs and len(docs) <= 5:
            for d in docs:
                print(f"    - [{d['type']}] {d['id']}: {d['title']} ({d['comments']} comments)")
        elif docs:
            print(f"    (showing first 3 of {len(docs)})")
            for d in docs[:3]:
                print(f"    - [{d['type']}] {d['id']}: {d['title']} ({d['comments']} comments)")
        print()

    return results


# ---------------------------------------------------------------------------
# Part C: FRDOC_0001 deep-dive
# ---------------------------------------------------------------------------
def frdoc_deep_dive(categories):
    """Query 5 FRDOC_0001 dockets to measure their scale."""
    print("=" * 70)
    print("PART C: FRDOC_0001 DEEP-DIVE (5 CATCH-ALL DOCKETS)")
    print("=" * 70)

    if not API_KEY:
        print("ERROR: No API key")
        return

    frdoc_dockets = set(norm for _, norm in categories["frdoc_0001"])
    sample = sorted(frdoc_dockets)[:5]

    if not sample:
        print("No FRDOC_0001 dockets found in blind spot.")
        return

    print(f"Testing {len(sample)} FRDOC_0001 dockets\n")

    for docket_id in sample:
        # Count how many of our FR docs use this docket
        our_docs = [r for r, norm in categories["frdoc_0001"] if norm == docket_id]
        print(f"--- {docket_id} ({len(our_docs)} of our FR docs use this) ---")

        total_all, docs = query_docket_docs(docket_id)
        total_comments = query_docket_comments(docket_id)

        print(f"  Total elements in Regs.gov search: {total_all}")
        print(f"  Non-submission docs returned (page 1): {len(docs)}")
        print(f"  Total comments on entire docket: {total_comments}")
        if docs:
            print(f"  Sample docs (first 3):")
            for d in docs[:3]:
                print(f"    - [{d['type']}] {d['id']}: {d['title']}")
        print(f"  VERDICT: {'CATCH-ALL (untrustworthy)' if total_all and total_all > 10 else 'small docket'}")
        print()


# ---------------------------------------------------------------------------
# Part D: Cross-validation of trustworthy dockets
# ---------------------------------------------------------------------------
def cross_validate(results):
    """For Category A dockets, verify document-level vs docket-level counts."""
    print("=" * 70)
    print("PART D: CROSS-VALIDATION (CATEGORY A DOCKETS)")
    print("=" * 70)

    cat_a = [r for r in results if r["category"] == "A_trustworthy"]
    if not cat_a:
        print("No Category A (1:1 trustworthy) dockets found to validate.")
        return

    print(f"\nValidating {len(cat_a)} trustworthy dockets:\n")
    for r in cat_a[:5]:
        print(f"  {r['docket_id']}:")
        print(f"    Docket total comments: {r['total_comments']}")
        if r["docs"]:
            doc = r["docs"][0]
            doc_comments = doc.get("comments")
            print(f"    Single doc ({doc['id']}) commentCount: {doc_comments}")
            if doc_comments is not None and r["total_comments"] is not None:
                match = "MATCH" if doc_comments == r["total_comments"] else f"MISMATCH (diff={r['total_comments'] - doc_comments})"
                print(f"    Validation: {match}")
        print()


# ---------------------------------------------------------------------------
# Part E: Summary report
# ---------------------------------------------------------------------------
def print_report(categories, blind_spot, results):
    """Print the final structured report."""
    print("\n" + "=" * 70)
    print("FINAL REPORT: DOCKET-BASED COMMENT COUNTING RELIABILITY")
    print("=" * 70)

    total_docs = 10406  # known from exploration
    bs = len(blind_spot)

    print(f"\n1. SCOPE")
    print(f"   Total eligible docs (2014): {total_docs:,}")
    print(f"   Blind spot (docket_id, no regs_document_id): {bs:,} ({100*bs/total_docs:.1f}%)")

    print(f"\n2. NORMALIZATION")
    print(f"   Queryable (valid Regs.gov format): {len(categories['regs_format']):,}")
    print(f"   FRDOC_0001 (catch-all, never trustworthy): {len(categories['frdoc_0001']):,}")
    print(f"   Non-Regs.gov (FERC/FTC/OMB/etc): {len(categories['non_regs']):,}")
    queryable_pct = 100 * len(categories['regs_format']) / bs if bs else 0
    print(f"   Queryable as % of blind spot: {queryable_pct:.1f}%")

    if results:
        cat_counts = Counter(r["category"] for r in results)
        total_recoverable = sum(r["total_comments"] or 0 for r in results if r["category"] == "A_trustworthy")
        total_ambiguous = sum(r["total_comments"] or 0 for r in results if r["category"] == "B_ambiguous")

        print(f"\n3. API SAMPLING ({len(results)} dockets tested)")
        print(f"   Category A (1:1 trustworthy): {cat_counts.get('A_trustworthy', 0)}")
        print(f"   Category B (2-3 docs, ambiguous): {cat_counts.get('B_ambiguous', 0)}")
        print(f"   Category C (4+ docs, untrustworthy): {cat_counts.get('C_untrustworthy', 0)}")
        print(f"   Category D (0 comments, moot): {cat_counts.get('D_moot', 0)}")
        print(f"\n   Comments recoverable from Category A: {total_recoverable:,}")
        print(f"   Comments in Category B (ambiguous): {total_ambiguous:,}")

        # Extrapolation
        n_queryable = len(categories['regs_format'])
        n_sampled = len(results)
        if n_sampled > 0 and n_queryable > 0:
            trustworthy_rate = cat_counts.get('A_trustworthy', 0) / n_sampled
            avg_comments_trustworthy = total_recoverable / max(cat_counts.get('A_trustworthy', 0), 1)
            estimated_total = trustworthy_rate * n_queryable * avg_comments_trustworthy
            print(f"\n4. EXTRAPOLATION (rough)")
            print(f"   Trustworthy rate: {100*trustworthy_rate:.0f}% of queryable")
            print(f"   Avg comments per trustworthy docket: {avg_comments_trustworthy:.0f}")
            print(f"   Estimated total recoverable: ~{estimated_total:,.0f} comments")
            print(f"   For context: RIN-bearing docs have 4.9M+ comments")
            if estimated_total > 0:
                print(f"   Recovery as % of RIN-bearing: {100*estimated_total/4_900_000:.3f}%")

    print(f"\n5. RECOMMENDATION")
    print(f"   [To be filled based on actual results above]")
    print(f"   Prior expectation: 92.3% of NO_RIN docs get zero comments.")
    print(f"   Even with perfect resolution, ceiling on recoverable volume is tiny")
    print(f"   relative to 4.9M+ comments on RIN-bearing docs.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Docket-Based Comment Counting Investigation")
    print(f"Data: {CSV_2014}")
    print(f"API key: {API_KEY[:8]}..." if API_KEY else "API key: NOT FOUND")
    print()

    categories, blind_spot = load_and_profile()

    results = sample_queryable_dockets(categories)

    frdoc_deep_dive(categories)

    cross_validate(results)

    print_report(categories, blind_spot, results)


if __name__ == "__main__":
    main()
