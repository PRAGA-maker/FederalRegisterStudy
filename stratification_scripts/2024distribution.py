import argparse
import os
import time
from typing import Optional, Dict, Iterable, List

import pandas as pd
import requests
from tqdm import tqdm


# API Endpoints
FR_DOCS_URL = "https://www.federalregister.gov/api/v1/documents.json"
FR_DOC_DETAIL_URL = "https://www.federalregister.gov/api/v1/documents/{document_number}.json"
REGS_DOC_URL = "https://api.regulations.gov/v4/documents/{documentId}"
REGS_COMMENTS_URL = "https://api.regulations.gov/v4/comments"

# HTTP session (connection pooling)
SESSION = requests.Session()



def fetch_document_details(document_number: str, max_retries: int = 3) -> Optional[dict]:
    """Fetch detailed information for a Federal Register document including comment data."""
    if not document_number:
        return None
    backoff = 1.0
    for _ in range(max_retries):
        try:
            r = SESSION.get(FR_DOC_DETAIL_URL.format(document_number=document_number), timeout=30)
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

class RegsClient:
    """Thin Regulations.gov v4 client with shared throttle and caches."""

    def __init__(self, api_key: Optional[str], rpm: int = 30) -> None:
        self.headers = {"X-Api-Key": api_key} if api_key else {}
        self.min_interval = 60.0 / rpm if rpm and rpm > 0 else 0.0
        self.last_call_ts = 0.0
        self.doc_detail_by_document_id: Dict[str, dict] = {}
        self.total_by_object_id: Dict[str, int] = {}

    def _throttle(self) -> None:
        if self.min_interval <= 0:
            return
        wait = self.min_interval - (time.time() - self.last_call_ts)
        if wait > 0:
            time.sleep(wait)
        self.last_call_ts = time.time()

    def _handle_retry_after(self, r: requests.Response) -> None:
        try:
            retry_after = int(r.headers.get("Retry-After", "0"))
            if retry_after > 0:
                time.sleep(min(retry_after, 60))
        except Exception:
            pass

    def get_document_detail(self, document_id: str, max_retries: int = 3) -> Optional[dict]:
        if not document_id:
            return None
        if document_id in self.doc_detail_by_document_id:
            return self.doc_detail_by_document_id[document_id]
        backoff = 1.0
        for _ in range(max_retries):
            try:
                self._throttle()
                r = SESSION.get(
                    REGS_DOC_URL.format(documentId=document_id),
                    headers=self.headers,
                    timeout=20,
                )
            except requests.RequestException:
                time.sleep(backoff)
                backoff = min(backoff * 2, 16)
                continue
            if r.status_code == 200:
                data = r.json()
                self.doc_detail_by_document_id[document_id] = data
                return data
            if r.status_code == 429:
                self._handle_retry_after(r)
                time.sleep(backoff)
                backoff = min(backoff * 2, 16)
                continue
            if r.status_code in (500, 502, 503, 504):
                time.sleep(backoff)
                backoff = min(backoff * 2, 16)
                continue
            return None
        return None

    def get_comment_total_by_object_id(self, object_id: str, max_retries: int = 3) -> Optional[int]:
        if not object_id:
            return None
        if object_id in self.total_by_object_id:
            return self.total_by_object_id[object_id]
        backoff = 1.0
        for _ in range(max_retries):
            try:
                self._throttle()
                r = SESSION.get(
                    REGS_COMMENTS_URL,
                    headers=self.headers,
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
                if isinstance(total, int) or (isinstance(total, str) and str(total).isdigit()):
                    val = int(total)
                    self.total_by_object_id[object_id] = val
                    return val
                return None
            if r.status_code == 429:
                self._handle_retry_after(r)
                time.sleep(backoff)
                backoff = min(backoff * 2, 16)
                continue
            if r.status_code in (500, 502, 503, 504):
                time.sleep(backoff)
                backoff = min(backoff * 2, 16)
                continue
            return None
        return None


def classify_submission_channel(comment_url: Optional[str], regs_document_id: Optional[str]) -> str:
    """Classify how comments are submitted for a document.
    Returns one of: 'regs.gov', 'fcc-ecfs', 'ferc-portal', 'sec-portal', 'email', 'other-portal', 'unknown'.
    """
    if regs_document_id:
        return "regs.gov"
    if not comment_url:
        return "unknown"
    url_lower = (comment_url or "").lower()
    if "regulations.gov" in url_lower:
        return "regs.gov"
    agency_portals = {
        "ecfs.fcc.gov": "fcc-ecfs",
        "efiling.ferc.gov": "ferc-portal",
        "www.sec.gov/comments": "sec-portal",
    }
    for domain, channel in agency_portals.items():
        if domain in url_lower:
            return channel
    if url_lower.startswith("mailto:"):
        return "email"
    if url_lower.startswith("http"):
        return "other-portal"
    return "unknown"


def iter_fr_documents_by_day(year: int, fr_sleep: float, limit: Optional[int], retries: int) -> Iterable[dict]:
    """Yield FR documents for the given year, day by day, with pagination and validation."""
    yielded = 0
    dates = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31").strftime("%Y-%m-%d").tolist()
    with tqdm(total=None if limit else len(dates), desc="FR days", unit="day") as daybar:
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
                backoff = 1.0
                r = None
                for _ in range(max(1, retries)):
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
                # Validate day match to avoid wrong-year data
                for rec in results:
                    if rec.get("publication_date") == day:
                        yield rec
                        yielded += 1
                        if limit and yielded >= limit:
                            daybar.update(1)
                            return
                page += 1
                if fr_sleep > 0:
                    time.sleep(fr_sleep)
                if total_pages is not None and page > total_pages:
                    break
            daybar.update(1)


def enrich_record(rec: dict, regs: RegsClient, retries: int) -> Optional[dict]:
    """Enrich a single FR record with detail, counts, and classification."""
    doc_number = rec.get("document_number")
    comment_url = None
    comments_close_on = None
    regs_document_id = None
    count_source = "unknown"
    comment_count: Optional[int] = None
    
    if doc_number:
        details = fetch_document_details(doc_number, max_retries=retries)
        if details:
            comment_url = details.get("comment_url")
            comments_close_on = details.get("comments_close_on")
            fr_info = details.get("regulations_dot_gov_info", {})
            if isinstance(fr_info, dict):
                regs_document_id = fr_info.get("document_id")

            # Prefer Regs.gov
            if regs_document_id:
                regs_detail = regs.get_document_detail(regs_document_id, max_retries=retries)
                if regs_detail:
                    attrs = (regs_detail.get("data") or {}).get("attributes") or {}
                    cc = attrs.get("commentCount")
                    if isinstance(cc, int) and cc >= 0:
                        comment_count = cc
                        count_source = "regulations.gov"
                    else:
                        obj_id = attrs.get("objectId")
                        if obj_id:
                            mt = regs.get_comment_total_by_object_id(obj_id, max_retries=retries)
                            if isinstance(mt, int):
                                comment_count = mt
                                count_source = "regulations.gov-meta"

            # Last resort: FR embedded count
            if comment_count is None and isinstance(fr_info, dict):
                cc_fr = fr_info.get("comments_count")
                if isinstance(cc_fr, int) and cc_fr >= 0:
                    comment_count = cc_fr
                    count_source = "federalregister"
        
    if not isinstance(comment_count, int) or comment_count < 0:
        comment_count = 0

    agencies = rec.get("agencies") or []
    agency_names = ", ".join([a.get("name") for a in agencies if isinstance(a, dict) and a.get("name")])

    # Eligibility and channel
    is_prorule = rec.get("type") == "Proposed Rule"
    has_comment_mechanism = bool(comment_url or comments_close_on or regs_document_id)
    eligibility_reason = None
    if is_prorule:
        eligibility_reason = "prorule"
    elif has_comment_mechanism:
        eligibility_reason = "notice-with-comment-period"
    submission_channel = classify_submission_channel(comment_url, regs_document_id) if eligibility_reason else None

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
    parser.add_argument("--min-age-hours", type=int, default=0, help="Exclude docs newer than this many hours from stats only")
    parser.add_argument("--quiet", action="store_true", help="Reduce log verbosity")
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

    # Fetch FR documents day-by-day
    all_results: List[dict] = list(iter_fr_documents_by_day(args.year, args.fr_sleep, args.limit, args.retries))

    # Include all PRORULE and NOTICE documents for enrichment
    # Will filter after enrichment to keep only those that actually accept comments
    comment_eligible = all_results
    if not args.quiet:
        print(f"Total {args.year} docs fetched (PRORULE + NOTICE): {len(all_results)}")

    # Enrich
    rows: List[dict] = []
    regs = RegsClient(api_key=args.regs_api_key, rpm=args.regs_rpm)
    for rec in tqdm(comment_eligible, desc="Enriching", unit="doc"):
        enriched = enrich_record(rec, regs, args.retries)
        if enriched:
            rows.append(enriched)

    if not args.quiet:
        print(f"\nFiltered to {len(rows)} comment-eligible documents (from {len(all_results)} total)")

    # Count by eligibility reason
    if rows and not args.quiet:
        df_temp = pd.DataFrame(rows)
        print("\nEligibility breakdown:")
        for reason, count in df_temp["eligibility_reason"].value_counts().items():
            print(f"  {reason}: {count}")
        
        print("\nCount source breakdown:")
        for source, count in df_temp["count_source"].value_counts().items():
            print(f"  {source}: {count}")
        
        if "submission_channel" in df_temp.columns:
            print("\nSubmission channel breakdown:")
            for ch, count in df_temp["submission_channel"].value_counts().items():
                print(f"  {ch}: {count}")
        
        print(f"\nSample row: {rows[0]}")
    elif not rows:
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
    stats_df = df
    # Apply min-age-hours filter for stats only
    if args.min_age_hours and args.min_age_hours > 0:
        try:
            df_dates = pd.to_datetime(df["publication_date"], errors="coerce")
            age_days = (pd.Timestamp.utcnow().normalize() - df_dates).dt.days
            min_days = max(args.min_age_hours // 24, 0)
            stats_df = df.loc[age_days > min_days]
        except Exception:
            pass
    stats = stats_df["comment_count"].describe(percentiles=[0.25, 0.5, 0.75])
    
    # Overall stats
    print(
        f"\nOVERALL - Comment-eligible docs: {total_docs}; "
        f"min={int(stats['min'])}, p25={int(stats['25%'])}, median={int(stats['50%'])}, "
        f"mean={stats['mean']:.2f}, p75={int(stats['75%'])}, max={int(stats['max'])}"
    )

    # Stats for Regs.gov channel only (authoritative counts)
    regs_mapped = stats_df[stats_df["submission_channel"] == "regs.gov"]
    if len(regs_mapped) > 0:
        regs_stats = regs_mapped["comment_count"].describe(percentiles=[0.25, 0.5, 0.75])
        print(
            f"\nREGS.GOV-MAPPED - Docs: {len(regs_mapped)}; "
            f"min={int(regs_stats['min'])}, p25={int(regs_stats['25%'])}, median={int(regs_stats['50%'])}, "
            f"mean={regs_stats['mean']:.2f}, p75={int(regs_stats['75%'])}, max={int(regs_stats['max'])}"
        )
        # Top sniff test
        top3 = regs_mapped.sort_values("comment_count", ascending=False).head(3)
        print("Top 3 by comment_count (regs.gov):")
        for _, r in top3.iterrows():
            print(f"  {r['document_number']}: {r['comment_count']} - {r['title'][:80]}")

    # Channel breakdown (counts likely unavailable outside regs.gov)
    if "submission_channel" in df.columns:
        print("\nSubmission channel breakdown:")
        for ch, count in df["submission_channel"].value_counts().items():
            print(f"  {ch}: {count}")

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


"""


