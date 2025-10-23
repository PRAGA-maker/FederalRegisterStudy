import argparse
import os
import time
from typing import Optional, Dict

import pandas as pd
import requests
from tqdm import tqdm


# API Endpoints
FR_DOCS_URL = "https://www.federalregister.gov/api/v1/documents.json"
REGS_DOC_URL = "https://api.regulations.gov/v4/documents/{documentId}"

# HTTP session (connection pooling)
SESSION = requests.Session()



def fetch_document_details(document_number: str, max_retries: int = 3) -> Optional[dict]:
    """Fetch detailed information for a Federal Register document including comment data."""
    if not document_number:
        return None
    backoff = 1.0
    for _ in range(max_retries):
        try:
            r = SESSION.get(
                f"https://www.federalregister.gov/api/v1/documents/{document_number}.json",
                timeout=30,
            )
        except requests.RequestException:
            time.sleep(backoff)
            backoff = min(backoff * 2, 16)
            continue
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(backoff)
            backoff = min(backoff * 2, 16)
            continue
        return None
    return None


def regs_comment_count(document_id: str, api_key: Optional[str], max_retries: int = 3) -> Optional[int]:
    if not document_id:
        return None
    headers = {"X-Api-Key": api_key} if api_key else {}
    backoff = 1.0
    for _ in range(max_retries):
        try:
            r = SESSION.get(
                REGS_DOC_URL.format(documentId=document_id),
                headers=headers,
                params={"fields": "commentCount"},
                timeout=20,
            )
        except requests.RequestException:
            time.sleep(backoff)
            backoff = min(backoff * 2, 16)
            continue
        if r.status_code == 200:
            data = r.json()
            val = data.get("data", {}).get("attributes", {}).get("commentCount")
            return int(val) if isinstance(val, (int, str)) and str(val).isdigit() else None
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(backoff)
            backoff = min(backoff * 2, 16)
            continue
        return None
    return None


def regs_doc_detail(document_id: str, api_key: Optional[str], max_retries: int = 3) -> Optional[dict]:
    """Fetch full document details from Regulations.gov including openForComment status."""
    if not document_id:
        return None
    headers = {"X-Api-Key": api_key} if api_key else {}
    backoff = 1.0
    for _ in range(max_retries):
        try:
            r = SESSION.get(
                REGS_DOC_URL.format(documentId=document_id),
                headers=headers,
                timeout=20,
            )
        except requests.RequestException:
            time.sleep(backoff)
            backoff = min(backoff * 2, 16)
            continue
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(backoff)
            backoff = min(backoff * 2, 16)
            continue
        return None
    return None


def regs_comment_total_by_object_id(object_id: str, api_key: Optional[str], max_retries: int = 3) -> Optional[int]:
    """Fetch total comment count via comments search API (authoritative via meta.totalElements)."""
    if not object_id:
        return None
    headers = {"X-Api-Key": api_key} if api_key else {}
    backoff = 1.0
    for _ in range(max_retries):
        try:
            r = SESSION.get(
                "https://api.regulations.gov/v4/comments",
                headers=headers,
                params={"filter[commentOnId]": object_id, "page[size]": 1},
                timeout=20,
            )
        except requests.RequestException:
            time.sleep(backoff)
            backoff = min(backoff * 2, 16)
            continue
        if r.status_code == 200:
            data = r.json()
            total = data.get("meta", {}).get("totalElements")
            return int(total) if isinstance(total, (int, str)) and str(total).isdigit() else None
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(backoff)
            backoff = min(backoff * 2, 16)
            continue
        return None
    return None


def main() -> None:
    # Disable abbreviation so flags like --end aren't misparsed
    parser = argparse.ArgumentParser(description="Distribution of comments for Federal Register documents by year", allow_abbrev=False)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=os.path.join(os.path.dirname(__file__), "output"))
    parser.add_argument("--regs-api-key", type=str, default=os.environ.get("REGS_API_KEY"))
    parser.add_argument("--regs-rpm", type=int, default=30, help="Throttle Regulations.gov lookups to requests per minute (<=50 recommended)")
    parser.add_argument("--fr-sleep", type=float, default=0.2, help="Sleep seconds between FR page fetches")
    parser.add_argument("--retries", type=int, default=10, help="Max retries for 429/5xx responses")
    args = parser.parse_args()

    # If FR_YEAR env is present, enforce it as the year to avoid any CLI/env mismatch
    env_year = os.environ.get("FR_YEAR")
    if env_year and str(args.year) != str(env_year):
        try:
            args.year = int(env_year)
            print(f"Note: Overriding --year with FR_YEAR={env_year} to ensure consistency with runner")
        except ValueError:
            print(f"Warning: FR_YEAR={env_year} is not a valid integer; using --year={args.year}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Fetch FR documents that allow commenting for the specified year
    # Query each day individually to avoid hitting the 10,000 result limit
    all_results = []
    dates = pd.date_range(start=f"{args.year}-01-01", end=f"{args.year}-12-31").strftime("%Y-%m-%d").tolist()
    reached_limit = False
    
    # If limiting documents, use an open-ended bar so we can stop early before 366 days
    daybar_total = None if args.limit is not None else len(dates)
    with tqdm(total=daybar_total, desc="FR days", unit="day") as daybar:
        for day in dates:
            page = 1
            total_pages = None
            while True:
                params = {
                    "per_page": 1000,
                    "page": page,
                    "conditions[publication_date][is]": day,
                    "conditions[type][]": ["PRORULE", "NOTICE"],
                }
                
                # retry on failure per page
                backoff = 1.0
                r = None
                for _ in range(max(1, args.retries)):
                    try:
                        r = SESSION.get(FR_DOCS_URL, params=params, timeout=30)
                    except requests.RequestException:
                        time.sleep(backoff)
                        backoff = min(backoff * 2, 16)
                        continue
                    if r.status_code == 200:
                        break
                    if r.status_code in (429, 500, 502, 503, 504):
                        time.sleep(backoff)
                        backoff = min(backoff * 2, 16)
                        continue
                    break
                if r is None or r.status_code != 200:
                    # Skip this day on persistent error
                    break
                j = r.json()
                if total_pages is None:
                    try:
                        total_pages = int(j.get("total_pages", 1))
                    except Exception:
                        total_pages = 1
                results = j.get("results", [])
                if not results:
                    break
                all_results.extend(results)
                # Stop early if we've collected enough documents for a test run
                if args.limit is not None and len(all_results) >= args.limit:
                    reached_limit = True
                    break
                page += 1
                if args.fr_sleep > 0:
                    time.sleep(args.fr_sleep)
                if total_pages is not None and page > total_pages:
                    break
            daybar.update(1)
            if reached_limit:
                break

    if args.limit is not None:
        all_results = all_results[: args.limit]

    # Include all PRORULE and NOTICE documents for enrichment
    # Will filter after enrichment to keep only those that actually accept comments
    comment_eligible = all_results
    print(f"Total {args.year} docs fetched (PRORULE + NOTICE): {len(all_results)}")

    # Enrich comment counts from FR data directly, fallback to Regulations.gov
    rows = []
    regs_cache: Dict[str, int] = {}
    min_interval = 60.0 / args.regs_rpm if args.regs_rpm and args.regs_rpm > 0 else 0.0
    last_call_ts = 0.0
    
    for rec in tqdm(comment_eligible, desc="Enriching", unit="doc"):
        # Fetch document details to get comment information
        doc_number = rec.get("document_number")
        comment_count = None
        count_source = "unknown"
        eligibility_reason = None
        comment_url = None
        comments_close_on = None
        regs_document_id = None
        
        if doc_number:
            details = fetch_document_details(doc_number, max_retries=args.retries)
            if details:
                # Extract comment information from document details
                comment_url = details.get("comment_url")
                comments_close_on = details.get("comments_close_on")
                
                # Get Regs document ID
                fr_info = details.get("regulations_dot_gov_info", {})
                if isinstance(fr_info, dict):
                    regs_document_id = fr_info.get("document_id")
                
                # ALWAYS prefer Regulations.gov when we can map the document
                if regs_document_id:
                    # Check cache first
                    if regs_document_id in regs_cache:
                        comment_count = regs_cache[regs_document_id]
                        count_source = "regulations.gov-cached"
                    else:
                        # Throttle API calls
                        now = time.time()
                        wait = min_interval - (now - last_call_ts)
                        if wait > 0:
                            time.sleep(wait)
                        
                        # Try document detail endpoint first
                        regs_detail = regs_doc_detail(regs_document_id, args.regs_api_key, max_retries=args.retries)
                        last_call_ts = time.time()
                        
                        if regs_detail:
                            attrs = (regs_detail.get("data") or {}).get("attributes") or {}
                            cc = attrs.get("commentCount")
                            if isinstance(cc, int) and cc >= 0:
                                comment_count = cc
                                count_source = "regulations.gov"
                                regs_cache[regs_document_id] = cc
                            else:
                                # Fallback: use comments search API (authoritative via meta.totalElements)
                                obj_id = attrs.get("objectId")
                                if obj_id:
                                    # Additional throttle for second call
                                    now = time.time()
                                    wait = min_interval - (now - last_call_ts)
                                    if wait > 0:
                                        time.sleep(wait)
                                    mt = regs_comment_total_by_object_id(obj_id, args.regs_api_key, max_retries=args.retries)
                                    last_call_ts = time.time()
                                    if isinstance(mt, int):
                                        comment_count = mt
                                        count_source = "regulations.gov-meta"
                                        regs_cache[regs_document_id] = mt
                
                # Only if still unknown, consider FR's embedded count as a hint
                if comment_count is None and isinstance(fr_info, dict):
                    cc = fr_info.get("comments_count")
                    if isinstance(cc, int) and cc >= 0:
                        comment_count = cc
                        count_source = "federalregister"
        
        # Default to 0 if still unknown
        if not isinstance(comment_count, int) or comment_count < 0:
            comment_count = 0

        agencies = rec.get("agencies") or []
        agency_names = ", ".join([a.get("name") for a in agencies if isinstance(a, dict) and a.get("name")])

        # Determine eligibility and reason
        doc_type = rec.get("type")
        if doc_type == "Proposed Rule":
            eligibility_reason = "prorule"
        elif regs_document_id and (comment_url or comments_close_on):
            # NOTICE with Regs.gov mapping AND comment mechanism
            if comment_url and "regulations.gov" in (comment_url or "").lower():
                eligibility_reason = "regs.gov"
            elif comments_close_on:
                eligibility_reason = "regs.gov"
            else:
                eligibility_reason = "external"
        elif comment_url or comments_close_on:
            # Has comment mechanism but no Regs.gov mapping (likely external)
            eligibility_reason = "external"
        
        # Include if we determined eligibility
        if eligibility_reason:
            rows.append(
                {
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
                }
            )

    print(f"\nFiltered to {len(rows)} comment-eligible documents (from {len(all_results)} total)")

    # Count by eligibility reason
    if rows:
        df_temp = pd.DataFrame(rows)
        print("\nEligibility breakdown:")
        for reason, count in df_temp["eligibility_reason"].value_counts().items():
            print(f"  {reason}: {count}")
        
        print("\nCount source breakdown:")
        for source, count in df_temp["count_source"].value_counts().items():
            print(f"  {source}: {count}")
        
        print(f"\nSample row: {rows[0]}")
    else:
        print("WARNING: No data collected - check API parameters and network connection")
        return
    
    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.output_dir, f"federal_register_{args.year}_comments.csv")
    df.to_csv(csv_path, index=False)

    # Analysis summary
    if len(df) == 0:
        print("ERROR: No data to analyze")
        return
        
    df["comment_count"] = df["comment_count"].fillna(0).astype(int)
    total_docs = len(df)
    stats = df["comment_count"].describe(percentiles=[0.25, 0.5, 0.75])
    
    # Overall stats
    print(
        f"\nOVERALL - Comment-eligible docs: {total_docs}; "
        f"min={int(stats['min'])}, p25={int(stats['25%'])}, median={int(stats['50%'])}, "
        f"mean={stats['mean']:.2f}, p75={int(stats['75%'])}, max={int(stats['max'])}"
    )

    # Stats for Regs.gov-mapped documents only (more reliable counts)
    regs_mapped = df[df["eligibility_reason"].isin(["prorule", "regs.gov"])]
    if len(regs_mapped) > 0:
        regs_stats = regs_mapped["comment_count"].describe(percentiles=[0.25, 0.5, 0.75])
        print(
            f"\nREGS.GOV-MAPPED - Docs: {len(regs_mapped)}; "
            f"min={int(regs_stats['min'])}, p25={int(regs_stats['25%'])}, median={int(regs_stats['50%'])}, "
            f"mean={regs_stats['mean']:.2f}, p75={int(regs_stats['75%'])}, max={int(regs_stats['max'])}"
        )

    # Stats for external comment mechanisms (counts likely unavailable/0)
    external = df[df["eligibility_reason"] == "external"]
    if len(external) > 0:
        print(f"\nEXTERNAL - Docs: {len(external)} (counts likely unavailable via API)")

    # Plotting removed per request; only CSV output is generated


if __name__ == "__main__":
    main()

"""
LESSONS LEARNED

what works:
- use documents.json not public_inspection_documents.json for published docs
- conditions[publication_date][is]=YYYY-MM-DD - must include [is] operator for date filter to work!
- daily loop through 365 days avoids 10k result limit on year-long queries
- filter by type[]=PRORULE,NOTICE to get comment-eligible docs server-side
- use persistent http session for connection pooling
- cache regulations.gov lookups to avoid duplicate api calls
- throttle regs.gov calls with rpm limits (30-50 recommended)
- use exponential backoff for 429/5xx responses
- fetch detail endpoint for each doc to get comment fields (list api lacks them)
- filter after enrichment not before - need detail data to know if doc accepts comments
- client-side filter: keep PRORULE OR has comment_url/comments_close_on/regs_document_id
- ALWAYS prefer regs.gov api over fr embedded counts (fr counts often stale/0)
- use regs.gov document detail commentCount when available
- fallback to regs.gov comments search meta.totalElements for authoritative totals
- only use fr embedded count if regs.gov unavailable (last resort)
- track count_source and eligibility_reason for data quality audits
- use tqdm for progress bars on network-bound operations

what doesnt work:
- conditions[publication_date] without [is] operator - api silently ignores it and returns current docs
- filtering before enrichment - list api response lacks comment fields so you cant tell what accepts comments
- assuming all NOTICE docs accept comments - many are informational only
- assuming PRORULE always has comment_url in list response - it doesnt, need detail fetch
- requesting fields[] parameter - breaks the query and returns 0 results
- public inspection docs rarely have comment counts or regs.gov mappings
- single large paginated queries can hit api limits or timeout
- fetching all fr docs then filtering is inefficient vs server-side filtering
- hardcoding api keys in public code
- not handling rate limits leads to 429 errors
- not using connection pooling wastes time on tcp handshakes
- not caching duplicate regs.gov lookups wastes api quota

functionality needed:
- server-side filtering by publication date and document type
- client-side filtering for comment eligibility indicators
- fallback from fr comment counts to regs.gov api when needed
- proper error handling and retries for network issues
- rate limiting for external apis
- progress indication for long-running operations
- csv output with comment counts and sources
- histogram visualization with specified colors
- summary statistics (min, p25, median, mean, p75, max)

api endpoints and fields:
- fr list: documents.json returns basic doc info (title, type, agencies, publication_date) but NOT comment fields
- fr detail: documents/{docId}.json returns full doc including comments_close_on, regulations_dot_gov_info.comments_count
- regs.gov v4: /documents/{documentId} with commentCount field, requires x-api-key header

data quality issues:
- many fr docs dont have comment periods
- pi docs often lack regs.gov mappings
- comment counts may be missing or stale
- some docs have multiple regs.gov documents in dockets (can overcount)
- broken date filter returns current year data regardless of requested year - validate dates in results!

typical results per year:
- ~24-27k total docs fetched (PRORULE + NOTICE types)
- ~6-8k comment-eligible after filtering (PRORULE + NOTICE with comment mechanisms)
- ~2-2.5k PRORULE docs (almost all accept comments)
- ~4-5k NOTICE docs with comment periods
- trump years (2017-2020) have fewer regulations overall

"""


