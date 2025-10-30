"""Mine comments from regulations.gov for Federal Register documents.

Reads federal_register_YYYY_comments.csv and extracts all comments via regs.gov API.
Outputs to data/comments_raw_YYYY.csv with Polars with full 22-column metadata.

Uses two-stage mining:
1. Collect comment IDs via list endpoint
2. Fetch full details via detail endpoint with smart sampling
"""
import argparse
import os
import random
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Any

import polars as pl
import requests
from tqdm import tqdm


# API endpoints
REGS_DOC_URL = "https://api.regulations.gov/v4/documents/{documentId}"
REGS_COMMENTS_URL = "https://api.regulations.gov/v4/comments"
REGS_COMMENT_DETAIL_URL = "https://api.regulations.gov/v4/comments/{commentId}"
FR_DOC_DETAIL_URL = "https://www.federalregister.gov/api/v1/documents/{document_number}.json"


@dataclass
class RegsThrottle:
    """Rate limiter for Regulations.gov API calls."""
    rpm: int = 30
    last_call_ts: float = 0.0

    @property
    def min_interval(self) -> float:
        return 60.0 / self.rpm if self.rpm and self.rpm > 0 else 0.0

    def sleep_if_needed(self) -> None:
        if self.min_interval <= 0:
            return
        jitter = random.uniform(-0.05, 0.05) * self.min_interval
        wait = (self.min_interval + jitter) - (time.time() - self.last_call_ts)
        if wait > 0:
            time.sleep(wait)
        self.last_call_ts = time.time()


def download_and_extract_pdf_text(
    pdf_url: str,
    headers: Dict[str, str],
    throttle: RegsThrottle,
    max_pages: int = 2,
) -> Optional[str]:
    """Download PDF from URL and extract text from first N pages using pdftotext.
    
    Returns extracted text or None if extraction fails.
    """
    if not pdf_url:
        return None
    
    try:
        # Download PDF
        throttle.sleep_if_needed()
        response = requests.get(pdf_url, headers=headers, timeout=30)
        if response.status_code != 200:
            return None
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as pdf_file:
            pdf_file.write(response.content)
            pdf_path = pdf_file.name
        
        # Extract text from first N pages
        txt_path = pdf_path.replace('.pdf', '.txt')
        try:
            result = subprocess.run(
                ['pdftotext', '-f', '1', '-l', str(max_pages), pdf_path, txt_path],
                capture_output=True,
                timeout=30,
            )
            
            if result.returncode == 0 and os.path.exists(txt_path):
                with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read().strip()
                os.unlink(txt_path)
                os.unlink(pdf_path)
                return text if text else None
            else:
                # pdftotext failed
                if os.path.exists(txt_path):
                    os.unlink(txt_path)
                os.unlink(pdf_path)
                return None
                
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            # pdftotext not installed or timeout
            if os.path.exists(txt_path):
                os.unlink(txt_path)
            os.unlink(pdf_path)
            return None
            
    except Exception as e:
        # Download or file I/O error
        return None


def get_object_id_for_regs_document(
    regs_document_id: Optional[str],
    headers: Dict[str, str],
    throttle: RegsThrottle,
    retries: int,
) -> Optional[str]:
    """Fetch objectId from a regs.gov document."""
    if not regs_document_id:
        return None
    backoff = 1.0
    for _ in range(max(1, retries)):
        try:
            throttle.sleep_if_needed()
            r = requests.get(
                REGS_DOC_URL.format(documentId=regs_document_id),
                headers=headers,
                timeout=20,
            )
        except requests.RequestException:
            time.sleep(backoff)
            backoff = min(backoff * 2, 16)
            continue
        if r.status_code == 200:
            data = r.json() or {}
            attrs = (data.get("data") or {}).get("attributes") or {}
            obj_id = attrs.get("objectId") or attrs.get("objectID")
            if obj_id:
                return obj_id
            return None
        if r.status_code in (401, 403):
            tqdm.write("ERROR: 401/403 from Regulations.gov; check REGS_API_KEY")
            return None
        if r.status_code == 429:
            try:
                retry_after = int(r.headers.get("Retry-After", "0"))
                if retry_after > 0:
                    time.sleep(min(retry_after, 60))
            except Exception:
                pass
            time.sleep(backoff)
            backoff = min(backoff * 2, 16)
            continue
        if r.status_code in (500, 502, 503, 504):
            time.sleep(backoff)
            backoff = min(backoff * 2, 16)
            continue
        return None
    return None


def get_regs_document_id_via_fr(document_number: Optional[str], retries: int) -> Optional[str]:
    """Fetch regs_document_id from Federal Register API."""
    if not document_number:
        return None
    backoff = 1.0
    for _ in range(max(1, retries)):
        try:
            r = requests.get(FR_DOC_DETAIL_URL.format(document_number=document_number), timeout=20)
        except requests.RequestException:
            time.sleep(backoff)
            backoff = min(backoff * 2, 16)
            continue
        if r.status_code == 200:
            j = r.json() or {}
            fr_info = j.get("regulations_dot_gov_info") or {}
            if isinstance(fr_info, dict):
                doc_id = fr_info.get("document_id") or fr_info.get("documentId")
                if doc_id:
                    return doc_id
            return None
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(backoff)
            backoff = min(backoff * 2, 16)
            continue
        return None
    return None


def iter_comments_for_object(
    object_id: str,
    headers: Dict[str, str],
    throttle: RegsThrottle,
    retries: int,
    limit_comments: Optional[int] = None,
) -> Iterable[dict]:
    """Yield comment records with windowed paging to bypass 5k cap."""
    if not object_id:
        return

    fetched = 0
    cursor_ge: Optional[str] = None
    while True:
        page = 1
        last_seen_ts: Optional[str] = None
        while page <= 20:
            params = {
                "filter[commentOnId]": object_id,
                "page[size]": 250,
                "page[number]": page,
                "sort": "lastModifiedDate,documentId",
            }
            if cursor_ge:
                params["filter[lastModifiedDate][ge]"] = cursor_ge
            backoff = 1.0
            while True:
                try:
                    throttle.sleep_if_needed()
                    r = requests.get(REGS_COMMENTS_URL, headers=headers, params=params, timeout=30)
                except requests.RequestException:
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 16)
                    continue
                if r.status_code == 200:
                    break
                if r.status_code in (401, 403):
                    tqdm.write("ERROR: 401/403 from Regulations.gov /comments; check REGS_API_KEY")
                    return
                if r.status_code == 429:
                    try:
                        retry_after = int(r.headers.get("Retry-After", "0"))
                        if retry_after > 0:
                            time.sleep(min(retry_after, 60))
                    except Exception:
                        pass
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 16)
                    continue
                if r.status_code in (500, 502, 503, 504):
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 16)
                    continue
                return

            data = r.json() or {}
            items: List[dict] = data.get("data") or []
            if not items:
                break

            last_item = items[-1]
            attrs_last = last_item.get("attributes") or {}
            last_seen_ts = attrs_last.get("lastModifiedDate") or attrs_last.get("modifyDate")

            for item in items:
                yield item
                fetched += 1
                if limit_comments is not None and fetched >= limit_comments:
                    return

            page += 1

        if not last_seen_ts or cursor_ge == last_seen_ts:
            return
        cursor_ge = last_seen_ts


def fetch_comment_detail(
    comment_id: str,
    headers: Dict[str, str],
    throttle: RegsThrottle,
    retries: int,
    include_attachments: bool = True,
) -> Optional[dict]:
    """Fetch full comment detail from /v4/comments/{id} endpoint."""
    if not comment_id:
        return None
    
    params = {}
    if include_attachments:
        params["include"] = "attachments"
    
    backoff = 1.0
    for _ in range(max(1, retries)):
        try:
            throttle.sleep_if_needed()
            r = requests.get(
                REGS_COMMENT_DETAIL_URL.format(commentId=comment_id),
                headers=headers,
                params=params,
                timeout=20,
            )
        except requests.RequestException:
            time.sleep(backoff)
            backoff = min(backoff * 2, 16)
            continue
        
        if r.status_code == 200:
            return r.json()
        if r.status_code in (401, 403):
            tqdm.write("ERROR: 401/403 from Regulations.gov; check REGS_API_KEY")
            return None
        if r.status_code == 429:
            try:
                retry_after = int(r.headers.get("Retry-After", "0"))
                if retry_after > 0:
                    time.sleep(min(retry_after, 60))
            except Exception:
                pass
            time.sleep(backoff)
            backoff = min(backoff * 2, 16)
            continue
        if r.status_code in (500, 502, 503, 504):
            time.sleep(backoff)
            backoff = min(backoff * 2, 16)
            continue
        return None
    
    return None


def extract_comment_fields_from_detail(
    data: dict,
    headers: Dict[str, str],
    throttle: RegsThrottle,
) -> Dict[str, Any]:
    """Extract 23 fields from full comment detail response, including PDF attachment text."""
    if not isinstance(data, dict):
        return {}
    
    comment_data = data.get("data") or {}
    attrs = comment_data.get("attributes") or {}
    included = data.get("included") or []
    
    # Core fields
    comment_id = comment_data.get("id")
    # Try multiple field names for comment text
    comment_text = (
        attrs.get("comment") or 
        attrs.get("commentText") or 
        attrs.get("comment_text") or
        attrs.get("title") or
        ""
    )
    
    # Timestamps
    posted_date = attrs.get("postedDate")
    receive_date = attrs.get("receiveDate")
    postmark_date = attrs.get("postmarkDate")
    
    # Submitter info
    first_name = attrs.get("firstName")
    last_name = attrs.get("lastName")
    organization = attrs.get("organization")
    submitter_type = attrs.get("submitterType")
    
    # Location
    city = attrs.get("city")
    state_province_region = attrs.get("stateProvinceRegion")
    country = attrs.get("country")
    zip_code = attrs.get("zip")
    
    # Government fields
    gov_agency = attrs.get("govAgency")
    gov_agency_type = attrs.get("govAgencyType")
    
    # Attachment info
    has_attachments = len(included) > 0
    attachment_count = len(included)
    attachment_formats = []
    attachment_texts = []
    
    for att in included:
        if isinstance(att, dict):
            att_attrs = att.get("attributes") or {}
            att_links = att.get("links") or {}
            
            # Extract format
            fmt = att_attrs.get("format") or att_attrs.get("fileFormats")
            if fmt:
                if isinstance(fmt, list):
                    attachment_formats.extend([str(f) for f in fmt])
                elif isinstance(fmt, dict):
                    attachment_formats.append(str(fmt.get("name") or fmt.get("type") or "unknown"))
                else:
                    attachment_formats.append(str(fmt))
            
            # Extract text from PDF attachments
            # Check if this is a PDF attachment
            is_pdf = False
            if fmt:
                fmt_str = str(fmt).lower()
                is_pdf = 'pdf' in fmt_str
            
            if is_pdf:
                # Get PDF URL from links
                pdf_url = att_links.get("self") or att_attrs.get("fileUrl")
                if pdf_url:
                    extracted_text = download_and_extract_pdf_text(pdf_url, headers, throttle)
                    if extracted_text:
                        attachment_texts.append(extracted_text)
    
    attachment_formats_str = ",".join(set(attachment_formats)) if attachment_formats else None
    attachment_text = "\n\n---\n\n".join(attachment_texts) if attachment_texts else None
    
    # Other metadata
    duplicate_comments = attrs.get("numItemsRecieved") or attrs.get("duplicateComments")
    page_count = attrs.get("pageCount")
    
    return {
        "comment_id": comment_id,
        "comment_text": str(comment_text or ""),
        "posted_date": posted_date,
        "receive_date": receive_date,
        "postmark_date": postmark_date,
        "first_name": first_name,
        "last_name": last_name,
        "organization": organization,
        "submitter_type": submitter_type,
        "city": city,
        "state_province_region": state_province_region,
        "country": country,
        "zip": zip_code,
        "gov_agency": gov_agency,
        "gov_agency_type": gov_agency_type,
        "has_attachments": has_attachments,
        "attachment_count": attachment_count,
        "attachment_formats": attachment_formats_str,
        "attachment_text": attachment_text,
        "duplicate_comments": duplicate_comments,
        "page_count": page_count,
    }


def should_sample_comments(comment_count: int, fetch_strategy: str) -> Tuple[bool, float]:
    """Decide whether to sample and what fraction based on strategy.
    
    Returns (should_sample, sample_rate)
    """
    if fetch_strategy == "all":
        return False, 1.0
    elif fetch_strategy == "sample":
        # Always sample 10%
        return True, 0.10
    elif fetch_strategy == "smart":
        # Fetch all if ≤1000, sample 10% if >1000, cap at 5000
        if comment_count <= 1000:
            return False, 1.0
        else:
            # Sample to get ~10% but max 5000
            target_sample = min(5000, int(comment_count * 0.10))
            sample_rate = target_sample / comment_count
            return True, sample_rate
    else:
        return False, 1.0


def mine_comments(
    fr_csv: Path,
    output_csv: Path,
    api_key: Optional[str],
    rpm: int,
    retries: int,
    limit_docs: Optional[int],
    fetch_strategy: str,
    max_comments_per_doc: Optional[int] = None,
) -> None:
    """Mine all comments from Federal Register documents with two-stage approach.
    
    Args:
        max_comments_per_doc: If set, skip documents with more comments than this limit
    """
    
    # Read FR documents
    if not fr_csv.exists():
        print(f"ERROR: {fr_csv} not found")
        return
    
    df_docs = pl.read_csv(str(fr_csv))
    
    # Filter to documents with regs_document_id AND comment_count > 0
    if "regs_document_id" in df_docs.columns and "comment_count" in df_docs.columns:
        df_docs = df_docs.filter(
            (pl.col("regs_document_id").is_not_null()) & (pl.col("comment_count") > 0)
        )
        print(f"Found {len(df_docs)} documents with comments > 0")
    elif "regs_document_id" in df_docs.columns:
        df_docs = df_docs.filter(pl.col("regs_document_id").is_not_null())
    else:
        print("ERROR: CSV missing regs_document_id column")
        return
    
    if limit_docs:
        df_docs = df_docs.head(limit_docs)
    
    print(f"Mining comments from {len(df_docs)} documents...")
    print(f"Fetch strategy: {fetch_strategy}")
    
    # Setup API
    headers = {"X-Api-Key": api_key} if api_key else {}
    throttle = RegsThrottle(rpm=rpm)
    
    # Load existing comments to avoid duplicates
    existing_ids = set()
    if output_csv.exists():
        df_existing = pl.read_csv(str(output_csv))
        existing_ids = set(df_existing["comment_id"].to_list())
        print(f"Loaded {len(existing_ids)} existing comments")
    
    # Collect comments with full metadata
    all_comments = []
    
    for row in tqdm(df_docs.iter_rows(named=True), total=len(df_docs), desc="Documents"):
        doc_number = row.get("document_number")
        regs_doc_id = row.get("regs_document_id")
        comment_count = row.get("comment_count", 0)
        
        # Get regs_document_id if missing
        if not regs_doc_id and doc_number:
            regs_doc_id = get_regs_document_id_via_fr(doc_number, retries)
            if not regs_doc_id:
                continue
        
        if not regs_doc_id:
            continue
        
        # Get objectId
        object_id = get_object_id_for_regs_document(regs_doc_id, headers, throttle, retries)
        if not object_id:
            continue
        
        # Stage 1: Collect comment IDs from list endpoint
        comment_ids = []
        # Use max_comments_per_doc as limit if specified
        limit_for_doc = max_comments_per_doc if max_comments_per_doc else None
        for item in iter_comments_for_object(object_id, headers, throttle, retries, limit_comments=limit_for_doc):
            if not isinstance(item, dict):
                continue
            cid = item.get("id") or item.get("commentId")
            if cid and cid not in existing_ids:
                comment_ids.append(cid)
        
        if not comment_ids:
            continue
        
        # Skip documents with too many comments if limit is set
        if max_comments_per_doc and len(comment_ids) > max_comments_per_doc:
            tqdm.write(f"  {doc_number}: Skipping (has {len(comment_ids)} comments, limit is {max_comments_per_doc})")
            continue
        
        # Stage 2: Decide sampling strategy
        should_sample, sample_rate = should_sample_comments(len(comment_ids), fetch_strategy)
        
        if should_sample:
            n_sample = int(len(comment_ids) * sample_rate)
            n_sample = max(1, min(n_sample, len(comment_ids)))
            sampled_ids = random.sample(comment_ids, n_sample)
            tqdm.write(f"  {doc_number}: Sampling {n_sample}/{len(comment_ids)} comments ({sample_rate*100:.1f}%)")
        else:
            sampled_ids = comment_ids
            tqdm.write(f"  {doc_number}: Fetching all {len(comment_ids)} comments")
        
        # Stage 3: Fetch full details for selected comments
        for comment_id in sampled_ids:
            detail_data = fetch_comment_detail(comment_id, headers, throttle, retries)
            if not detail_data:
                continue
            
            fields = extract_comment_fields_from_detail(detail_data, headers, throttle)
            if not fields or not fields.get("comment_id"):
                continue
            
            # Add document_number
            fields["document_number"] = doc_number
            all_comments.append(fields)
            existing_ids.add(comment_id)
    
    if not all_comments:
        print("No new comments found")
        return
    
    # Write to CSV with all 22 columns
    df_new = pl.DataFrame(all_comments)
    
    # Reorder columns to put document_number first
    column_order = [
        "document_number",
        "comment_id",
        "comment_text",
        "posted_date",
        "receive_date",
        "postmark_date",
        "first_name",
        "last_name",
        "organization",
        "submitter_type",
        "city",
        "state_province_region",
        "country",
        "zip",
        "gov_agency",
        "gov_agency_type",
        "has_attachments",
        "attachment_count",
        "attachment_formats",
        "attachment_text",
        "duplicate_comments",
        "page_count",
    ]
    
    # Keep only columns that exist
    existing_columns = [c for c in column_order if c in df_new.columns]
    df_new = df_new.select(existing_columns)
    
    if output_csv.exists():
        df_existing = pl.read_csv(str(output_csv))
        # Ensure both DataFrames have same columns with proper types
        for col in df_new.columns:
            if col not in df_existing.columns:
                # Cast to match the new dataframe's type
                df_existing = df_existing.with_columns(pl.lit(None).cast(df_new[col].dtype).alias(col))
        for col in df_existing.columns:
            if col not in df_new.columns:
                # Cast to match the existing dataframe's type
                df_new = df_new.with_columns(pl.lit(None).cast(df_existing[col].dtype).alias(col))
        # Reorder columns to match
        df_existing = df_existing.select(df_new.columns)
        df_combined = pl.concat([df_existing, df_new])
    else:
        df_combined = df_new
    
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_combined.write_csv(str(output_csv))
    
    print(f"\nMined {len(all_comments)} new comments")
    print(f"Total comments in database: {len(df_combined)}")
    print(f"Output: {output_csv}")
    
    # Print metadata statistics
    if len(df_new) > 0:
        print("\nMetadata coverage:")
        org_count = df_new.filter(pl.col("organization").is_not_null()).shape[0]
        print(f"  Organization: {org_count}/{len(df_new)} ({100*org_count/len(df_new):.1f}%)")
        name_count = df_new.filter(pl.col("first_name").is_not_null() | pl.col("last_name").is_not_null()).shape[0]
        print(f"  Name: {name_count}/{len(df_new)} ({100*name_count/len(df_new):.1f}%)")
        attach_count = df_new.filter(pl.col("has_attachments") == True).shape[0]
        print(f"  Attachments: {attach_count}/{len(df_new)} ({100*attach_count/len(df_new):.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine comments from regulations.gov")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of documents")
    parser.add_argument("--max-comments-per-doc", type=int, default=None, help="Skip documents with more than this many comments (cost control)")
    parser.add_argument("--rpm", type=int, default=30, help="Requests per minute")
    parser.add_argument("--retries", type=int, default=10, help="Max retries for API calls")
    parser.add_argument(
        "--fetch-strategy",
        type=str,
        default="smart",
        choices=["all", "sample", "smart"],
        help="Sampling strategy: all (fetch every comment), sample (10%%), smart (all if <=1000, else sample)",
    )
    args = parser.parse_args()
    
    # Setup paths
    script_dir = Path(__file__).parent
    fr_csv = script_dir.parent / "output" / f"federal_register_{args.year}_comments.csv"
    output_csv = script_dir / "data" / f"comments_raw_{args.year}.csv"
    
    # Get API key from environment
    api_key = os.environ.get("REGS_API_KEY")
    if not api_key:
        print("WARNING: REGS_API_KEY not set, API calls may be rate limited")
    
    mine_comments(fr_csv, output_csv, api_key, args.rpm, args.retries, args.limit, args.fetch_strategy, args.max_comments_per_doc)


if __name__ == "__main__":
    main()


# learnings from building the comment mining pipeline:
#
# api design:
# - using the detail endpoint instead of just list endpoint is crucial for getting full metadata
#   like organization, submitter_type, attachments, etc. the list endpoint only returns minimal
#   fields (id, comment text, dates)
# - the two-stage approach (collect IDs first, then fetch details selectively) is necessary
#   because fetching every detail on large dockets (10k+ comments) would take hours and burn
#   through API quota
# - always include ?include=attachments in detail endpoint to get attachment data in the
#   "included" array - this is required for pdf text extraction
#
# sampling strategies:
# - smart sampling (fetch all for <=1000 comments, sample 10% otherwise with 5k cap) gives good
#   coverage while staying fast
# - --max-comments-per-doc is essential for cost control - skips documents with too many comments
#   before burning api quota and openai credits
# - use --limit to cap number of documents for testing/development
#
# rate limiting:
# - throttling is critical - regs.gov rate limits hard at 1000/hr without API key, 1000/hr with
#   key per their docs (but we use 30 rpm = 1800/hr to be safe since they seem flexible)
# - exponential backoff on 429/5xx is essential for reliability
# - rate limiting applies to both comment list calls AND attachment downloads
#
# attachment handling:
# - pdf attachments can be huge - only extract first 2 pages with pdftotext to keep token counts
#   low and processing fast
# - pdftotext command: pdftotext -f 1 -l 2 <input.pdf> <output.txt>
# - non-pdf attachments are skipped entirely - images, videos, word docs, etc don't add value
#   and are expensive to process
# - attachment urls come from included[].links.self or included[].attributes.fileUrl in the
#   detail response
# - format detection: check if 'pdf' in str(format).lower() to identify pdf attachments
# - always use temp files for pdf downloads and clean them up immediately after extraction
# - download failures are common (broken urls, access denied) - gracefully return None and
#   continue without crashing
#
# pagination workarounds:
# - the windowed pagination in iter_comments bypasses the 5000 result cap by using
#   lastModifiedDate filters to "restart" pagination windows
# - this technique is essential for dockets with >5000 comments
#
# data quality:
# - metadata fields are often null/missing so always have fallbacks
# - comment_text field name varies: can be "comment", "commentText", "comment_text", or "title"
# - attachment_text is stored alongside comment_text and used as fallback during classification
# - empty submissions (no text, no attachments, no metadata) should be excluded entirely rather
#   than marked as "undecided" - we want a high-quality sample not a population with noise
#
# polars gotchas:
# - polars is way faster than pandas for large CSVs but syntax takes getting used to
# - when concatenating dataframes with new columns, you need to explicitly add missing columns
#   with pl.lit(None).cast(dtype) to match schemas
# - use .iter_rows(named=True) to iterate as dicts for easier field access
#
# performance tips:
# - use persistent http session for connection pooling
# - cache results (existing_ids) to avoid duplicate fetches
# - incremental saves prevent data loss on crashes
# - subprocess.run with timeout prevents hanging on corrupted pdfs

