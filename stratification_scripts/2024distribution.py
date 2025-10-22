import argparse
import os
import re
import time
from typing import Optional, Dict

import pandas as pd
import requests
from tqdm import tqdm


# Endpoints
FR_DOCS_URL = "https://www.federalregister.gov/api/v1/documents.json"
REGS_DOC_URL = "https://api.regulations.gov/v4/documents/{documentId}"
FR_DOCS_BULK_URL_TMPL = "https://www.federalregister.gov/api/v1/documents/{docnums}.json"

# HTTP session (connection pooling)
SESSION = requests.Session()

# Precompiled regex for regs.gov IDs
REG_PATH_RE = re.compile(r"/document/([A-Za-z0-9-]+)")
REG_QUERY_RE = re.compile(r"[?&]D=([A-Za-z0-9-]+)")



def extract_regsid(record: dict) -> Optional[str]:
    candidates = []
    for k in (
        "comment_url",
        "comments_url",
        "regulations_dot_gov_url",
        "regulations_dot_gov_docket_url",
        "action_comments_url",
        "comment_url_html",
    ):
        v = record.get(k)
        if isinstance(v, str):
            candidates.append(v)
    for k, v in record.items():
        if isinstance(v, str) and "regulations.gov" in v:
            candidates.append(v)
    r_path = REG_PATH_RE
    r_query = REG_QUERY_RE
    for url in candidates:
        m1 = r_path.search(url)
        if m1:
            return m1.group(1)
        m2 = r_query.search(url)
        if m2:
            return m2.group(1)
    for k in ("regulations_dot_gov_document_id", "regs_document_id"):
        v = record.get(k)
        if isinstance(v, str) and v:
            return v
    return None


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
    parser.add_argument("--include-dockets", action="store_true", default=False, help="Also fetch FR dockets (second pass) for missing counts")
    parser.add_argument("--fr-batch-size", type=int, default=100, help="Batch size for FR document details fetch")
    parser.add_argument("--sampling", type=str, default="quarterly", choices=["daily", "quarterly"], help="Sampling strategy: daily (full year) or quarterly (efficient sampling)")
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

    # Fetch FR documents that allow commenting in 2024
    all_results = []
    if args.sampling == "quarterly":
        # Use quarterly sampling for efficiency
        dates = pd.date_range(start=f"{args.year}-01-01", end=f"{args.year}-12-31", freq="QS").strftime("%Y-%m-%d").tolist()
        # Add a few more strategic dates throughout the year
        dates.extend([
            f"{args.year}-03-15", f"{args.year}-06-15", f"{args.year}-09-15", f"{args.year}-12-15"
        ])
        dates = sorted(set(dates))  # Remove duplicates and sort
    else:
        # Use daily sampling (original approach)
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
                    "conditions[publication_date]": day,
                    "conditions[type][]": ["PRORULE", "NOTICE"],
                    # Removed fields[] parameter - FR API returns reasonable defaults without it
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

    # Filter to only documents that allow commenting (client-side filter)
    comment_eligible = []
    prorule_count = 0
    explicit_comment_count = 0
    
    for rec in all_results:
        # Always include PRORULE documents as they typically accept comments
        if rec.get("type") == "Proposed Rule":
            comment_eligible.append(rec)
            prorule_count += 1
        # Also include if any explicit comment indicators present
        elif (
            rec.get("accepting_comments_on_regulations_dot_gov") is True
            or rec.get("comment_url") is not None
            or rec.get("comments_close_on") is not None
        ):
            comment_eligible.append(rec)
            explicit_comment_count += 1
    
    print(f"Total {args.year} docs: {len(all_results)}; Comment-eligible: {len(comment_eligible)}")
    print(f"  - PRORULE documents: {prorule_count}")
    print(f"  - Explicit comment indicators: {explicit_comment_count}")

    # Enrich comment counts from FR data directly, fallback to Regulations.gov
    rows = []
    regs_cache: Dict[str, int] = {}
    min_interval = 60.0 / args.regs_rpm if args.regs_rpm and args.regs_rpm > 0 else 0.0
    last_call_ts = 0.0
    fr_source_count = 0
    regs_source_count = 0
    
    for rec in tqdm(comment_eligible, desc="Enriching", unit="doc"):
        # Fetch document details to get comment information
        doc_number = rec.get("document_number")
        comment_count = 0
        source = "unknown"
        comment_url = None
        comments_close_on = None
        regs_document_id = None
        
        if doc_number:
            details = fetch_document_details(doc_number, max_retries=args.retries)
            if details:
                # Extract comment information from document details
                comment_url = details.get("comment_url")
                comments_close_on = details.get("comments_close_on")
                
                # Get comment count from regulations_dot_gov_info
                fr_info = details.get("regulations_dot_gov_info", {})
                if isinstance(fr_info, dict):
                    comment_count = fr_info.get("comments_count", 0)
                    regs_document_id = fr_info.get("document_id")
                    if isinstance(comment_count, int) and comment_count >= 0:
                        source = "federalregister"
                        fr_source_count += 1
                
                # If still no comment count, try Regulations.gov API fallback
                if source == "unknown" and regs_document_id:
                    cc = None
                    if regs_document_id in regs_cache:
                        cc = regs_cache[regs_document_id]
                    else:
                        now = time.time()
                        wait = min_interval - (now - last_call_ts)
                        if wait > 0:
                            time.sleep(wait)
                        val = regs_comment_count(regs_document_id, args.regs_api_key, max_retries=args.retries)
                        last_call_ts = time.time()
                        if isinstance(val, int):
                            regs_cache[regs_document_id] = val
                            cc = val
                    if isinstance(cc, int):
                        comment_count = cc
                        source = "regulations.gov"
                        regs_source_count += 1
        
        if not isinstance(comment_count, int) or comment_count < 0:
            comment_count = 0

        agencies = rec.get("agencies") or []
        agency_names = ", ".join([a.get("name") for a in agencies if isinstance(a, dict) and a.get("name")])

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
                "source": source,
            }
        )

    print(f"Total rows: {len(rows)}")
    print(f"Data sources: FR={fr_source_count}, Regs.gov={regs_source_count}, Unknown={len(rows)-fr_source_count-regs_source_count}")
    if rows:
        print(f"Sample row: {rows[0]}")
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
    print(
        f"Comment-eligible docs: {total_docs}; "
        f"min={int(stats['min'])}, p25={int(stats['25%'])}, median={int(stats['50%'])}, "
        f"mean={stats['mean']:.2f}, p75={int(stats['75%'])}, max={int(stats['max'])}"
    )

    # Plotting removed per request; only CSV output is generated


if __name__ == "__main__":
    main()

"""
LESSONS LEARNED

what works:
- use documents.json not public_inspection_documents.json for published docs
- daily loop with publication_date=YYYY-MM-DD for precise date filtering
- filter by type[]=PRORULE,NOTICE to get comment-eligible docs
- request specific fields to reduce payload size
- use persistent http session for connection pooling
- precompile regex patterns for performance
- cache regulations.gov lookups to avoid duplicate api calls
- throttle regs.gov calls with rpm limits (30-50 recommended)
- use exponential backoff for 429/5xx responses
- client-side filter for comment eligibility: accepting_comments_on_regulations_dot_gov=true OR comment_url present OR comments_close_on not null
- prefer fr regulations_dot_gov_info.comments_count over regs.gov api when available
- use tqdm for progress bars on network-bound operations
- batch fr detail fetches only when absolutely needed

what doesnt work:
- public inspection docs rarely have comment counts or regs.gov mappings
- single large paginated queries can hit api limits or timeout
- fetching all fr docs then filtering is inefficient vs server-side filtering
- hardcoding api keys in public code
- not handling rate limits leads to 429 errors
- not using connection pooling wastes time on tcp handshakes
- regex compilation in loops is slow
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
- fr documents.json: publication_date, type[], comment_url, accepting_comments_on_regulations_dot_gov, comments_close_on, regulations_dot_gov_info.comments_count
- regs.gov v4: /documents/{documentId} with commentCount field, requires x-api-key header
- fr bulk details: /documents/{doc1,doc2,...}.json for batch fetching

data quality issues:
- many fr docs dont have comment periods
- pi docs often lack regs.gov mappings
- comment counts may be missing or stale
- some docs have multiple regs.gov documents in dockets (can overcount)

"""


