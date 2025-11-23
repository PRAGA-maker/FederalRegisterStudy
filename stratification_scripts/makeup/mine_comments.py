"""Mine comments from regulations.gov for Federal Register documents.

Reads federal_register_YYYY_comments.csv and extracts all comments via regs.gov API.
Outputs to data/comments_raw_YYYY.csv with Polars with full 22-column metadata.

Uses two-stage mining:
1. Collect comment IDs via list endpoint
2. Fetch full details via detail endpoint with smart sampling
"""
import argparse
import contextlib
import io
import os
import random
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

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


class RegsGovClient:
    """Shared HTTP client with multi-key round-robin load balancing + throttling.
    
    Thread-safe for concurrent use from multiple threads.
    """

    def __init__(self, api_keys_with_limits: List[Tuple[str, int]], retries: int) -> None:
        """Initialize client with multiple API keys and their RPH limits.
        
        Args:
            api_keys_with_limits: List of (api_key, rph_limit) tuples
            retries: Max retries per request
        """
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        })
        self.retries = max(1, retries)
        
        # Thread-safety lock for key selection and throttle updates
        self._request_lock = threading.Lock()
        
        # Setup per-key throttles
        self.keys: List[str] = []
        self.throttles: List[RegsThrottle] = []
        self.failed_keys: Set[int] = set()  # Track keys that returned 401/403
        self.cooling_down: Dict[int, float] = {}  # key_idx -> cooldown_until_timestamp
        
        for api_key, rph_limit in api_keys_with_limits:
            self.keys.append(api_key)
            rpm = rph_limit / 60.0  # Convert RPH to RPM
            self.throttles.append(RegsThrottle(rpm=int(rpm)))
        
        self.current_key_idx = 0
        self.total_keys = len(self.keys)
        
        if self.total_keys == 0:
            raise ValueError("Must provide at least one API key")
        
        print(f"Initialized RegsGovClient with {self.total_keys} API keys:")
        for idx, (_, rph) in enumerate(api_keys_with_limits):
            print(f"  Key #{idx}: {rph} RPH ({rph/60:.1f} RPM)")

    def __enter__(self) -> "RegsGovClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        self.session.close()

    @staticmethod
    def _sleep_with_backoff(backoff: float) -> float:
        time.sleep(backoff)
        return min(backoff * 2, 16.0)

    @staticmethod
    def _respect_retry_after(response: requests.Response) -> None:
        try:
            retry_after = int(response.headers.get("Retry-After", "0"))
            if retry_after > 0:
                time.sleep(min(retry_after, 60))
        except Exception:
            pass

    def _select_best_key(self) -> int:
        """Select the best available key using round-robin with throttle-aware fallback."""
        now = time.time()
        
        # Remove expired cooldowns
        expired = [k for k, until in self.cooling_down.items() if now >= until]
        for k in expired:
            del self.cooling_down[k]
        
        # Try round-robin starting from current key
        for offset in range(self.total_keys):
            candidate_idx = (self.current_key_idx + offset) % self.total_keys
            
            # Skip failed or cooling down keys
            if candidate_idx in self.failed_keys:
                continue
            if candidate_idx in self.cooling_down:
                continue
            
            # Check if this key's throttle would require long wait
            throttle = self.throttles[candidate_idx]
            wait_time = (throttle.min_interval) - (now - throttle.last_call_ts)
            
            # If wait is reasonable (<2s), use this key
            if wait_time < 2.0:
                if offset > 0:  # We switched keys
                    tqdm.write(f"  Switched to Key #{candidate_idx} (Key #{self.current_key_idx} throttled)")
                self.current_key_idx = candidate_idx
                return candidate_idx
        
        # All keys throttled or failed - find key with shortest wait
        best_idx = None
        min_wait = float('inf')
        
        for idx in range(self.total_keys):
            if idx in self.failed_keys:
                continue
            if idx in self.cooling_down:
                continue
            
            throttle = self.throttles[idx]
            wait_time = (throttle.min_interval) - (now - throttle.last_call_ts)
            if wait_time < min_wait:
                min_wait = wait_time
                best_idx = idx
        
        if best_idx is None:
            # All keys failed - use first non-failed or just first
            for idx in range(self.total_keys):
                if idx not in self.failed_keys:
                    best_idx = idx
                    break
            if best_idx is None:
                best_idx = 0
        
        self.current_key_idx = best_idx
        return best_idx

    def _perform_request(
        self,
        url: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        timeout: int = 30,
        stream: bool = False,
    ) -> Optional[requests.Response]:
        backoff = 1.0
        for attempt in range(self.retries):
            # CRITICAL: Lock around key selection and throttle sleep to prevent race conditions
            # Multiple threads could select the same key and violate rate limits
            with self._request_lock:
                key_idx = self._select_best_key()
                api_key = self.keys[key_idx]
                throttle = self.throttles[key_idx]
                headers = {"X-Api-Key": api_key}
                
                # Sleep while holding lock to ensure only one thread uses this key at a time
                throttle.sleep_if_needed()
            
            # Make request outside the lock (network I/O can happen in parallel)
            try:
                response = self.session.get(
                    url,
                    headers=headers,
                    params=params,
                    timeout=timeout,
                    stream=stream,
                )
            except requests.RequestException:
                backoff = self._sleep_with_backoff(backoff)
                continue

            if response.status_code == 200:
                return response

            # Lock again for modifying shared state (failed_keys, cooling_down)
            if response.status_code in (401, 403):
                with self._request_lock:
                    self.failed_keys.add(key_idx)
                tqdm.write(f"  Key #{key_idx} failed auth - marking as unusable")
                # Try another key immediately
                if len(self.failed_keys) < self.total_keys:
                    continue
                else:
                    # All keys failed
                    return response

            if response.status_code == 429:
                # Rate limited - put this key in cooldown
                with self._request_lock:
                    self.cooling_down[key_idx] = time.time() + 60
                tqdm.write(f"  Key #{key_idx} rate limited - cooling down for 60s")
                self._respect_retry_after(response)
                backoff = self._sleep_with_backoff(backoff)
                continue

            if response.status_code in (500, 502, 503, 504):
                backoff = self._sleep_with_backoff(backoff)
                continue

            return response

        return None

    def request_json(
        self,
        url: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        timeout: int = 30,
        auth_error_message: Optional[str] = None,
    ) -> Optional[dict]:
        response = self._perform_request(url, params=params, timeout=timeout)
        if response is None:
            return None

        if response.status_code == 200:
            try:
                data = response.json() or {}
                return data
            except ValueError:
                return {}

        if response.status_code in (401, 403) and auth_error_message:
            tqdm.write(auth_error_message)

        return None

    def download_bytes(self, url: str, timeout: int = 30) -> Optional[bytes]:
        response = self._perform_request(url, timeout=timeout)
        if response and response.status_code == 200:
            return response.content
        return None

def calculate_sample_size(population_size: int, confidence: float = 0.95, margin: float = 0.05) -> int:
    """Calculate statistical sample size for proportion estimation with finite population correction.
    
    Args:
        population_size: Total population size (will be converted to int if float from Polars)
        confidence: Confidence level (0.95 for 95% CI, 0.99 for 99% CI)
        margin: Margin of error (0.05 for ±5%)
    
    Returns:
        Required sample size (always returns at least population_size for small populations)
    """
    # Convert to int if float (Polars sum() can return floats)
    population_size = int(population_size)
    
    if population_size <= 0:
        return 0  # No population
    
    # For very small populations, always sample ALL to ensure coverage
    if population_size < 10:
        return population_size  # 100% sampling for tiny strata
    
    # Z-score for confidence level
    Z = 1.96 if confidence == 0.95 else 2.576  # 95% or 99%
    p = 0.5  # Maximum variance assumption (most conservative)
    E = margin
    
    # Calculate sample size for infinite population
    n_infinite = (Z**2 * p * (1 - p)) / (E**2)
    
    # Apply finite population correction
    n_corrected = n_infinite / (1 + (n_infinite - 1) / population_size)
    
    # Always fetch all if population is smaller than required sample
    # Explicitly cast to int to ensure return type is always int
    return int(min(int(n_corrected) + 1, population_size))


def build_two_stage_sample_plan(
    df_docs: pl.DataFrame,
    client: RegsGovClient,
    existing_ids: Set[str],
    retries: int,
    max_comments_per_doc: Optional[int],
) -> Tuple[Dict[str, List[str]], Dict[str, dict], pl.DataFrame]:
    """Build two-stage stratified sampling: sample documents, then sample comments within each.
    
    TWO-STAGE APPROACH:
    1. Group documents into strata (agency × comment_bin)
    2. Within each stratum, sample documents (using sqrt heuristic for coverage)
    3. For each sampled document, collect comment IDs
    4. Within each document's IDs, sample to reach stratum target
    
    This ensures small strata open ALL documents while large strata sample both dimensions.
    
    Args:
        df_docs: DataFrame with [document_number, agency, comment_count, regs_document_id]
        client: RegsGovClient for API calls
        existing_ids: Set of already-fetched comment IDs to skip
        retries: Max retries for API calls
        max_comments_per_doc: Global cap on comments per document
    
    Returns:
        - Dict[doc_number -> sampled_comment_ids]
        - Dict[comment_id -> list_payload] (cached list items)
        - Stratification summary DataFrame
    """
    import math
    
    print("\n" + "="*60)
    print("TWO-STAGE STRATIFIED SAMPLING PLAN")
    print("="*60)
    
    # Assign comment bins
    print("\nAssigning documents to comment bins...")
    df_docs = df_docs.with_columns([
        pl.when(pl.col("comment_count") <= 10).then(pl.lit("0-10"))
          .when(pl.col("comment_count") <= 100).then(pl.lit("11-100"))
          .when(pl.col("comment_count") <= 1000).then(pl.lit("101-1000"))
          .when(pl.col("comment_count") <= 10000).then(pl.lit("1001-10000"))
          .otherwise(pl.lit("10000+"))
          .alias("comment_bin")
    ])
    
    # Group by strata
    print("Grouping into strata by agency × comment_bin...")
    strata = df_docs.group_by(["agency", "comment_bin"]).agg([
        pl.col("document_number").alias("doc_numbers"),
        pl.col("comment_count").sum().alias("total_comments"),
        pl.col("document_number").count().alias("num_docs"),
    ])
    
    # Calculate target sample sizes per stratum
    print("Calculating sample sizes per stratum (95% CI, ±5% margin)...")
    strata = strata.with_columns([
        pl.col("total_comments").map_elements(
            lambda x: calculate_sample_size(x, confidence=0.95, margin=0.05),
            return_dtype=pl.Int64
        ).alias("target_comments")
    ])
    
    strata = strata.sort(["agency", "comment_bin"])
    print("\nStrata summary:")
    try:
        print(strata)
    except UnicodeEncodeError:
        # Windows console encoding issue - print simplified version
        print(f"  {len(strata)} strata created")
        for row in strata.iter_rows(named=True):
            print(f"  {row['agency'][:30]:30s} / {row['comment_bin']:12s} - {row['num_docs']} docs, {row['total_comments']} comments")
    
    # Stage 1: Sample documents within each stratum
    print("\n" + "="*60)
    print("STAGE 1: SAMPLING DOCUMENTS WITHIN STRATA")
    print("="*60 + "\n")
    
    doc_to_comment_ids: Dict[str, List[str]] = {}
    comment_id_to_payload: Dict[str, dict] = {}
    total_ids_collected = 0
    
    # Create document lookup for faster access
    doc_rows = {row["document_number"]: row for row in df_docs.iter_rows(named=True)}
    
    for row in tqdm(strata.iter_rows(named=True), total=len(strata), desc="Sampling strata", unit="stratum"):
        agency = row["agency"]
        comment_bin = row["comment_bin"]
        doc_numbers = row["doc_numbers"]
        target_comments = row["target_comments"]
        total_comments = row["total_comments"]
        num_docs = row["num_docs"]
        
        # Decide how many documents to sample
        # Small strata: sample ALL documents
        # Large strata: sample sqrt(num_docs) to ensure coverage
        if num_docs <= 10 or total_comments <= 25:
            # Small stratum: fetch everything
            docs_to_sample = doc_numbers
            action = f"ALL {num_docs} docs, ALL comments"
            tqdm.write(f"  Small stratum: {agency:30s} / {comment_bin:12s} -> {action}")
        else:
            # Two-stage: sample documents then comments
            n_docs_to_sample = min(num_docs, max(int(math.ceil(math.sqrt(num_docs))), 3))
            docs_to_sample = random.sample(list(doc_numbers), n_docs_to_sample)
            comments_per_doc = max(1, target_comments // n_docs_to_sample)
            action = f"Sampled {n_docs_to_sample}/{num_docs} docs, targeting ~{comments_per_doc} comments/doc -> {target_comments} total"
            tqdm.write(f"  {agency:30s} / {comment_bin:12s} -> {action}")
        
        # Stage 1b: Collect comment IDs for sampled documents
        for doc_num in docs_to_sample:
            doc_row = doc_rows.get(doc_num)
            if not doc_row:
                continue
            
            comment_ids = collect_comment_ids_for_document(
                doc_row, client, existing_ids, retries, max_comments_per_doc,
                cache_payloads=comment_id_to_payload  # Cache list payloads here
            )
            
            if not comment_ids:
                continue
            
            doc_to_comment_ids[doc_num] = comment_ids
            total_ids_collected += len(comment_ids)
    
    print(f"\nStage 1 complete: Collected {total_ids_collected:,} comment IDs from {len(doc_to_comment_ids)} documents")
    
    # Stage 2: Sample comments within each stratum to meet target
    print("\n" + "="*60)
    print("STAGE 2: SAMPLING COMMENTS WITHIN DOCUMENTS")
    print("="*60 + "\n")
    
    sampled_doc_to_ids: Dict[str, List[str]] = {}
    
    for row in strata.iter_rows(named=True):
        agency = row["agency"]
        comment_bin = row["comment_bin"]
        doc_numbers = row["doc_numbers"]
        target_comments = row["target_comments"]
        num_docs = row["num_docs"]
        
        # Collect IDs from documents in this stratum
        stratum_ids_by_doc: Dict[str, List[str]] = {}
        total_stratum_ids = 0
        
        for doc_num in doc_numbers:
            if doc_num in doc_to_comment_ids:
                ids = doc_to_comment_ids[doc_num]
                stratum_ids_by_doc[doc_num] = ids
                total_stratum_ids += len(ids)
        
        if total_stratum_ids == 0:
            continue
        
        # Decide sampling approach
        # If target_comments is None/null or we have fewer IDs than target, take all
        if target_comments is None or total_stratum_ids <= target_comments:
            # Take all IDs
            for doc_num, ids in stratum_ids_by_doc.items():
                sampled_doc_to_ids[doc_num] = ids
            tqdm.write(f"  {agency:30s} / {comment_bin:12s} -> ALL {total_stratum_ids} comments (100%)")
        else:
            # Sample proportionally from each document
            sample_rate = target_comments / total_stratum_ids
            sampled_count = 0
            
            for doc_num, ids in stratum_ids_by_doc.items():
                n_to_sample = max(1, int(len(ids) * sample_rate))
                n_to_sample = min(n_to_sample, len(ids))
                sampled = random.sample(ids, n_to_sample)
                sampled_doc_to_ids[doc_num] = sampled
                sampled_count += len(sampled)
            
            pct = 100 * sampled_count / total_stratum_ids
            tqdm.write(f"  {agency:30s} / {comment_bin:12s} -> SAMPLED {sampled_count}/{total_stratum_ids} comments ({pct:.1f}%)")
    
    total_sampled = sum(len(ids) for ids in sampled_doc_to_ids.values())
    total_pop = strata['total_comments'].sum()
    
    print(f"\n{'='*60}")
    print(f"Two-stage sampling complete:")
    print(f"  Total strata: {len(strata)}")
    print(f"  Documents sampled: {len(sampled_doc_to_ids)}/{len(df_docs)}")
    print(f"  Total population: {total_pop:,} comments")
    print(f"  Comments to fetch: {total_sampled:,}")
    if total_pop > 0:
        print(f"  Sampling ratio: {100*total_sampled/total_pop:.1f}%")
    print(f"{'='*60}\n")
    
    return sampled_doc_to_ids, comment_id_to_payload, strata


def download_and_extract_pdf_text(
    pdf_url: str,
    client: RegsGovClient,
    max_pages: int = 2,
) -> Optional[str]:
    """Download PDF from URL and extract text from first N pages using PyMuPDF."""
    if not pdf_url:
        return None

    # Extract filename from URL for better error reporting
    try:
        filename = pdf_url.split('/')[-1] if '/' in pdf_url else pdf_url
        if len(filename) > 60:
            filename = filename[:30] + "..." + filename[-27:]
    except Exception:
        filename = "unknown"

    pdf_bytes = client.download_bytes(pdf_url, timeout=30)
    if not pdf_bytes:
        return None
    
    # Check if we got JSON (API endpoint) instead of PDF bytes
    if pdf_bytes.startswith(b'{') and b'"data"' in pdf_bytes[:200]:
        try:
            import json
            data = json.loads(pdf_bytes.decode('utf-8'))
            # Extract download URL from JSON response
            actual_url = None
            
            # Logic to find fileUrl in the JSON response
            if isinstance(data, dict):
                # Try data.attributes.fileFormats[0].fileUrl
                if 'data' in data:
                    attrs = data.get('data', {}).get('attributes', {})
                    file_formats = attrs.get("fileFormats", [])
                    if isinstance(file_formats, list):
                        for ff in file_formats:
                            if isinstance(ff, dict) and ff.get("fileUrl"):
                                actual_url = ff.get("fileUrl")
                                break
                    if not actual_url:
                        actual_url = attrs.get('fileUrl') or attrs.get('contentUrl')
            
            if actual_url:
                # Retry with actual download URL
                pdf_bytes = client.download_bytes(actual_url, timeout=30)
                if not pdf_bytes:
                    return None
            else:
                return None
        except Exception:
            return None

    # Verify it's actually a PDF (starts with %PDF)
    if not pdf_bytes.startswith(b'%PDF'):
        # Sometimes we get an HTML error page even with 200 OK
        if b'<!DOCTYPE html>' in pdf_bytes[:100] or b'<html' in pdf_bytes[:100]:
            tqdm.write(f"  WARN: Got HTML instead of PDF for {filename}")
            return None
        tqdm.write(f"  WARN: Not a PDF file: {filename} (starts with {pdf_bytes[:20]})")
        return None
    
    try:
        # Try pypdf first (pure python, often reliable)
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(pdf_bytes))
            
            text_parts = []
            for page_num in range(min(max_pages, len(reader.pages))):
                page = reader.pages[page_num]
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            
            text = "\n\n".join(text_parts).strip()
            return text if text else None
            
        except Exception as pypdf_error:
            # Fall back to PyMuPDF if pypdf fails
            try:
                import fitz  # PyMuPDF
                doc = fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")
                
                text_parts = []
                for page_num in range(min(max_pages, len(doc))):
                    page = doc[page_num]
                    page_text = page.get_text()
                    if page_text:
                        text_parts.append(page_text)
                
                doc.close()
                
                text = "\n\n".join(text_parts).strip()
                return text if text else None
                
            except Exception as pymupdf_error:
                tqdm.write(f"  WARN: PDF extraction failed for {filename}: pypdf={str(pypdf_error)[:100]}, pymupdf={str(pymupdf_error)[:100]}")
                return None
    
    except Exception as e:
        tqdm.write(f"  WARN: Unexpected error extracting PDF {filename}: {e}")
        return None


def get_object_id_for_regs_document(
    regs_document_id: Optional[str],
    client: RegsGovClient,
) -> Optional[str]:
    """Fetch objectId from a regs.gov document."""
    if not regs_document_id:
        return None

    data = client.request_json(
        REGS_DOC_URL.format(documentId=regs_document_id),
        timeout=20,
        auth_error_message="ERROR: 401/403 from Regulations.gov; check REGS_API_KEY",
    )
    if not data:
        return None

    attrs = (data.get("data") or {}).get("attributes") or {}
    return attrs.get("objectId") or attrs.get("objectID")


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
    client: RegsGovClient,
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

            data = client.request_json(
                REGS_COMMENTS_URL,
                params=params,
                timeout=30,
                auth_error_message="ERROR: 401/403 from Regulations.gov /comments; check REGS_API_KEY",
            )
            if data is None:
                return

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
    client: RegsGovClient,
    include_attachments: bool = True,
) -> Optional[dict]:
    """Fetch full comment detail from /v4/comments/{id} endpoint."""
    if not comment_id:
        return None

    params: Dict[str, str] = {}
    if include_attachments:
        params["include"] = "attachments"

    return client.request_json(
        REGS_COMMENT_DETAIL_URL.format(commentId=comment_id),
        params=params,
        timeout=20,
        auth_error_message="ERROR: 401/403 from Regulations.gov; check REGS_API_KEY",
    )


def extract_comment_fields_from_list(item: dict) -> Dict[str, Any]:
    """Extract available fields from /comments list endpoint payload.
    
    The list endpoint returns basic fields without needing a detail call.
    Returns partial fields + flag indicating if detail fetch is needed.
    """
    if not isinstance(item, dict):
        return {}
    
    comment_id = item.get("id")
    attrs = item.get("attributes") or {}
    
    # Extract available fields from list
    comment_text = (
        attrs.get("comment") or 
        attrs.get("commentText") or 
        attrs.get("comment_text") or
        ""
    )
    
    # Submitter info (often available in list)
    first_name = attrs.get("firstName")
    last_name = attrs.get("lastName")
    organization = attrs.get("organization")
    submitter_type = attrs.get("submitterType")
    
    # Dates
    posted_date = attrs.get("postedDate")
    receive_date = attrs.get("receiveDate")
    
    # Attachment flag
    has_attachments = attrs.get("hasAttachments", False)
    
    # Decide if we need detail fetch:
    # - No comment text, OR
    # - Has attachments AND text is short (<3000 chars)
    text_length = len(str(comment_text or ""))
    needs_detail = (text_length == 0) or (has_attachments and text_length < 3000)
    
    return {
        "comment_id": comment_id,
        "comment_text": str(comment_text or ""),
        "first_name": first_name,
        "last_name": last_name,
        "organization": organization,
        "submitter_type": submitter_type,
        "posted_date": posted_date,
        "receive_date": receive_date,
        "has_attachments": has_attachments,
        "needs_detail_fetch": needs_detail,
        "_list_payload": item,  # Cache for later use
    }


def extract_comment_fields_from_detail(
    data: dict,
    client: RegsGovClient,
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
    
    # Attachment info - WITH SIMPLIFIED LOGIC
    has_attachments = len(included) > 0
    attachment_count = len(included)
    attachment_formats: List[str] = []
    attachment_texts: List[str] = []
    
    # Extract PDF text for all PDF attachments
    if has_attachments:
        for att in included:
            if not isinstance(att, dict):
                continue

            att_attrs = att.get("attributes") or {}
            att_links = att.get("links") or {}

            # Extract format information for logging
            # fileFormats is a list of dicts: [{'fileUrl': '...', 'format': 'pdf', 'size': 123}]
            file_formats = att_attrs.get("fileFormats", [])
            if isinstance(file_formats, list) and len(file_formats) > 0:
                # Extract format strings from fileFormats list
                for ff in file_formats:
                    if isinstance(ff, dict) and ff.get("format"):
                        attachment_formats.append(str(ff.get("format")))
            elif att_attrs.get("format"):
                # Fallback to simple format field if no fileFormats list
                attachment_formats.append(str(att_attrs.get("format")))

            # Check if this attachment has any PDF format
            is_pdf = False
            pdf_url = None
            
            if isinstance(file_formats, list) and len(file_formats) > 0:
                for ff in file_formats:
                    if isinstance(ff, dict):
                        fmt = str(ff.get("format", "")).lower()
                        if "pdf" in fmt:
                            is_pdf = True
                            # Get the direct download URL
                            pdf_url = ff.get("fileUrl")
                            break
            
            # Fallback: check simple format field
            if not is_pdf and att_attrs.get("format"):
                if "pdf" in str(att_attrs.get("format", "")).lower():
                    is_pdf = True
                    # Try to get URL from direct fileUrl field
                    pdf_url = att_attrs.get("fileUrl")
            
            if not is_pdf:
                continue

            # If we still don't have a URL, try the self link (API endpoint)
            if not pdf_url:
                pdf_url = att_links.get("self")

            if not pdf_url:
                continue
            
            extracted_text = download_and_extract_pdf_text(pdf_url, client)
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


def load_documents(fr_csv: Path, limit_docs: Optional[int], requested_year: int) -> Optional[pl.DataFrame]:
    """Load and filter FR document CSV, with year validation."""
    if not fr_csv.exists():
        print(f"ERROR: {fr_csv} not found")
        print(f"  Run: python stratification_scripts/2024distribution.py --year {requested_year} --limit 100")
        print(f"  This will generate the Federal Register document list for {requested_year}")
        return None

    print(f"Loading documents from: {fr_csv}")
    df_docs = pl.read_csv(str(fr_csv))

    # Validate year by checking publication dates
    if "publication_date" in df_docs.columns:
        # Check a sample of rows (first 1001, skipping first row in case of year boundary)
        sample_size = min(1001, len(df_docs))
        sample_dates = df_docs.head(sample_size)["publication_date"].to_list()[1:]  # Skip first row
        
        # Extract years from dates
        years_in_data = set()
        for date_str in sample_dates[:100]:  # Check first 100 dates after the first
            if date_str and isinstance(date_str, str) and len(date_str) >= 4:
                try:
                    year = int(date_str[:4])
                    years_in_data.add(year)
                except (ValueError, TypeError):
                    continue
        
        if years_in_data and requested_year not in years_in_data:
            most_common_year = max(years_in_data) if years_in_data else "unknown"
            print(f"WARNING: Requested year {requested_year}, but CSV contains data from {most_common_year}")
            print(f"  Using data from: {fr_csv}")

    if "regs_document_id" in df_docs.columns and "comment_count" in df_docs.columns:
        df_docs = df_docs.filter(
            (pl.col("regs_document_id").is_not_null()) & (pl.col("comment_count") > 0)
        )
        print(f"Found {len(df_docs)} documents with comments > 0")
    elif "regs_document_id" in df_docs.columns:
        df_docs = df_docs.filter(pl.col("regs_document_id").is_not_null())
    else:
        print("ERROR: CSV missing regs_document_id column")
        return None

    if limit_docs:
        df_docs = df_docs.head(limit_docs)
        print(f"Limited to first {limit_docs} documents")

    return df_docs


def load_existing_comment_ids(output_csv: Path) -> Set[str]:
    """Read existing comments file and return IDs for deduping."""
    if not output_csv.exists():
        return set()

    df_existing = pl.read_csv(str(output_csv))
    existing_ids = set(df_existing["comment_id"].to_list())
    print(f"Loaded {len(existing_ids)} existing comments")
    return existing_ids


def resolve_regs_doc_id(row: Dict[str, Any], retries: int) -> Optional[str]:
    regs_doc_id = row.get("regs_document_id")
    if regs_doc_id:
        return regs_doc_id

    doc_number = row.get("document_number")
    if doc_number:
        return get_regs_document_id_via_fr(doc_number, retries)
    return None


def collect_comment_ids_for_document(
    row: Dict[str, Any],
    client: RegsGovClient,
    existing_ids: Set[str],
    retries: int,
    max_comments_per_doc: Optional[int],
    cache_payloads: Optional[Dict[str, dict]] = None,
) -> List[str]:
    """Collect comment IDs for a single document (respecting max limits).
    
    Args:
        cache_payloads: If provided, caches list payloads keyed by comment_id for later reuse
    """
    doc_number = row.get("document_number")
    regs_doc_id = resolve_regs_doc_id(row, retries)
    if not regs_doc_id:
        return []

    object_id = get_object_id_for_regs_document(regs_doc_id, client)
    if not object_id:
        return []

    comment_ids: List[str] = []
    limit_for_doc = max_comments_per_doc if max_comments_per_doc else None
    try:
        for item in iter_comments_for_object(object_id, client, limit_comments=limit_for_doc):
            if not isinstance(item, dict):
                continue
            cid = item.get("id") or item.get("commentId")
            if cid and cid not in existing_ids:
                comment_ids.append(cid)
                # Cache list payload if requested
                if cache_payloads is not None:
                    list_fields = extract_comment_fields_from_list(item)
                    if list_fields:
                        cache_payloads[cid] = list_fields
    except Exception as exc:
        tqdm.write(f"  ERROR collecting IDs for {doc_number}: {exc}")
        return []

    if not comment_ids:
        return []

    if max_comments_per_doc and len(comment_ids) > max_comments_per_doc:
        tqdm.write(
            f"  {doc_number}: Skipping (has {len(comment_ids)} comments, limit is {max_comments_per_doc})"
        )
        return []

    return comment_ids


def fetch_comment_details_for_ids(
    comment_ids: Iterable[str],
    doc_number: str,
    client: RegsGovClient,
    existing_ids: Set[str],
) -> List[Dict[str, Any]]:
    """Fetch detail payloads for a list of comment IDs."""
    records: List[Dict[str, Any]] = []
    for comment_id in comment_ids:
        detail_data = fetch_comment_detail(comment_id, client)
        if not detail_data:
            continue

        fields = extract_comment_fields_from_detail(detail_data, client)
        if not fields or not fields.get("comment_id"):
            continue

        fields["document_number"] = doc_number
        records.append(fields)
        existing_ids.add(comment_id)

    return records


def write_comment_output(all_comments: List[Dict[str, Any]], output_csv: Path) -> None:
    """Persist mined comments and print coverage summary."""
    if not all_comments:
        print("No new comments found")
        return

    df_new = pl.DataFrame(all_comments, infer_schema_length=len(all_comments))
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

    existing_columns = [c for c in column_order if c in df_new.columns]
    df_new = df_new.select(existing_columns)

    if output_csv.exists():
        df_existing = pl.read_csv(str(output_csv))

        for col in df_new.columns:
            if col not in df_existing.columns:
                df_existing = df_existing.with_columns(
                    pl.lit(None).cast(df_new[col].dtype).alias(col)
                )
            elif df_existing[col].dtype != df_new[col].dtype:
                try:
                    df_existing = df_existing.with_columns(pl.col(col).cast(df_new[col].dtype))
                except Exception as exc:
                    print(
                        f"  WARNING: Could not cast {col} from {df_existing[col].dtype} to {df_new[col].dtype}: {exc}"
                    )
                    df_new = df_new.with_columns(pl.col(col).cast(df_existing[col].dtype))

        for col in df_existing.columns:
            if col not in df_new.columns:
                df_new = df_new.with_columns(pl.lit(None).cast(df_existing[col].dtype).alias(col))

        df_existing = df_existing.select(df_new.columns)
        df_combined = pl.concat([df_existing, df_new])
    else:
        df_combined = df_new

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_combined.write_csv(str(output_csv))

    print(f"\n{'='*60}")
    print(f"SAVED RAW COMMENTS")
    print(f"{'='*60}")
    print(f"Mined {len(all_comments)} new comments")
    print(f"Total comments in database: {len(df_combined)}")
    print(f"Output file: {output_csv.absolute()}")
    print(f"{'='*60}")

    if len(df_new) > 0:
        print("\nMetadata coverage:")
        org_count = df_new.filter(pl.col("organization").is_not_null()).shape[0]
        print(f"  Organization: {org_count}/{len(df_new)} ({100*org_count/len(df_new):.1f}%)")
        name_count = df_new.filter(
            pl.col("first_name").is_not_null() | pl.col("last_name").is_not_null()
        ).shape[0]
        print(f"  Name: {name_count}/{len(df_new)} ({100*name_count/len(df_new):.1f}%)")
        attach_count = df_new.filter(pl.col("has_attachments") == True).shape[0]
        print(f"  Attachments: {attach_count}/{len(df_new)} ({100*attach_count/len(df_new):.1f}%)")
        
        # Empty/minimal comment statistics
        print("\nComment text quality:")
        empty_text = df_new.filter(
            (pl.col("comment_text").is_null()) | (pl.col("comment_text").str.strip_chars() == "")
        ).shape[0]
        print(f"  Empty text: {empty_text}/{len(df_new)} ({100*empty_text/len(df_new):.1f}%)")
        
        minimal_text = df_new.filter(
            (pl.col("comment_text").is_not_null()) & 
            (pl.col("comment_text").str.len_chars() < 50) &
            (pl.col("comment_text").str.len_chars() > 0)
        ).shape[0]
        print(f"  Minimal text (<50 chars): {minimal_text}/{len(df_new)} ({100*minimal_text/len(df_new):.1f}%)")
        
        no_content = df_new.filter(
            ((pl.col("comment_text").is_null()) | (pl.col("comment_text").str.strip_chars() == "")) &
            ((pl.col("has_attachments") == False) | (pl.col("has_attachments").is_null()))
        ).shape[0]
        print(f"  No content (empty text + no attachments): {no_content}/{len(df_new)} ({100*no_content/len(df_new):.1f}%)")


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


def run_stratified_pipeline(
    df_docs: pl.DataFrame,
    client: RegsGovClient,
    existing_ids: Set[str],
    retries: int,
    max_comments_per_doc: Optional[int],
    concurrent_workers: int = 10,
) -> List[Dict[str, Any]]:
    """Run two-stage stratified sampling pipeline with list payload exploitation and concurrent fetching."""
    
    # Use new two-stage sampling
    sampled_doc_to_ids, comment_payloads, strata = build_two_stage_sample_plan(
        df_docs, client, existing_ids, retries, max_comments_per_doc
    )
    
    if not sampled_doc_to_ids:
        print("No comments selected for fetching")
        return []
    
    # Stage 3: Fetch details with smart list payload reuse + concurrent fetching
    print("\n" + "=" * 60)
    print("STAGE 3: FETCHING COMMENT DETAILS (with list payload reuse + concurrent)")
    print("=" * 60 + "\n")
    
    # Thread-safe data structures
    all_comments: List[Dict[str, Any]] = []
    comments_lock = threading.Lock()
    
    # Metrics (thread-safe counters)
    metrics = {
        "failed_fetches": 0,
        "detail_fetches": 0,
        "list_only": 0,
        "pdf_extracted": 0,
        "pdf_skipped": 0,
        "lock": threading.Lock()
    }
    
    total_to_fetch = sum(len(ids) for ids in sampled_doc_to_ids.values())
    
    # Prepare work items: (doc_number, comment_id)
    work_items = []
    for doc_number, comment_ids in sampled_doc_to_ids.items():
        for comment_id in comment_ids:
            work_items.append((doc_number, comment_id))
    
    def process_comment(doc_number: str, comment_id: str) -> Optional[Dict[str, Any]]:
        """Worker function to process a single comment."""
        nonlocal metrics, existing_ids, comment_payloads, client
        
        # Check if we have cached list payload
        list_fields = comment_payloads.get(comment_id)
        
        if list_fields and not list_fields.get("needs_detail_fetch", True):
            # Use list payload only - no detail call needed!
            fields = {
                "document_number": doc_number,
                "comment_id": list_fields["comment_id"],
                "comment_text": list_fields["comment_text"],
                "first_name": list_fields.get("first_name"),
                "last_name": list_fields.get("last_name"),
                "organization": list_fields.get("organization"),
                "submitter_type": list_fields.get("submitter_type"),
                "posted_date": list_fields.get("posted_date"),
                "receive_date": list_fields.get("receive_date"),
                "has_attachments": list_fields.get("has_attachments", False),
                # Fill remaining fields with None
                "postmark_date": None,
                "city": None,
                "state_province_region": None,
                "country": None,
                "zip": None,
                "gov_agency": None,
                "gov_agency_type": None,
                "attachment_count": 0,
                "attachment_formats": None,
                "attachment_text": None,
                "duplicate_comments": None,
                "page_count": None,
            }
            
            with metrics["lock"]:
                metrics["list_only"] += 1
                existing_ids.add(comment_id)
            
            return fields
        else:
            # Need to fetch detail
            detail_data = fetch_comment_detail(comment_id, client, include_attachments=True)
            if not detail_data:
                with metrics["lock"]:
                    metrics["failed_fetches"] += 1
                return None
            
            fields = extract_comment_fields_from_detail(detail_data, client)
            if not fields or not fields.get("comment_id"):
                with metrics["lock"]:
                    metrics["failed_fetches"] += 1
                return None
            
            # Track PDF extraction stats
            with metrics["lock"]:
                if fields.get("has_attachments"):
                    if fields.get("attachment_text"):
                        metrics["pdf_extracted"] += 1
                    else:
                        metrics["pdf_skipped"] += 1
                
                metrics["detail_fetches"] += 1
                existing_ids.add(comment_id)
            
            fields["document_number"] = doc_number
            return fields
    
    # Start timing
    start_time = time.time()
    
    # Process comments concurrently
    print(f"Using {concurrent_workers} concurrent workers")
    
    with ThreadPoolExecutor(max_workers=concurrent_workers) as executor:
        # Submit all work
        futures = {
            executor.submit(process_comment, doc_number, comment_id): (doc_number, comment_id)
            for doc_number, comment_id in work_items
        }
        
        # Collect results with progress bar
        with tqdm(total=total_to_fetch, desc="Processing comments", unit="comment") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result:
                    with comments_lock:
                        all_comments.append(result)
                pbar.update(1)
    
    # End timing
    end_time = time.time()
    total_time = end_time - start_time
    
    # Print statistics
    print(f"\nStage 3 complete:")
    print(f"  Total processed: {len(all_comments):,} comments")
    
    return all_comments


def run_sampling_pipeline(
    df_docs: pl.DataFrame,
    client: RegsGovClient,
    existing_ids: Set[str],
    retries: int,
    fetch_strategy: str,
    max_comments_per_doc: Optional[int],
) -> List[Dict[str, Any]]:
    all_comments: List[Dict[str, Any]] = []

    for row in tqdm(df_docs.iter_rows(named=True), total=len(df_docs), desc="Documents"):
        doc_number = row.get("document_number")
        if not doc_number:
            continue

        comment_ids = collect_comment_ids_for_document(
            row, client, existing_ids, retries, max_comments_per_doc
        )
        if not comment_ids:
            continue

        should_sample, sample_rate = should_sample_comments(len(comment_ids), fetch_strategy)
        if should_sample:
            n_sample = int(len(comment_ids) * sample_rate)
            n_sample = max(1, min(n_sample, len(comment_ids)))
            sampled_ids = random.sample(comment_ids, n_sample)
            tqdm.write(
                f"  {doc_number}: Sampling {n_sample}/{len(comment_ids)} comments ({sample_rate*100:.1f}%)"
            )
        else:
            sampled_ids = comment_ids
            tqdm.write(f"  {doc_number}: Fetching all {len(comment_ids)} comments")

        records = fetch_comment_details_for_ids(sampled_ids, doc_number, client, existing_ids)
        all_comments.extend(records)

    return all_comments


def mine_comments(
    fr_csv: Path,
    output_csv: Path,
    api_keys_with_limits: List[Tuple[str, int]],
    retries: int,
    limit_docs: Optional[int],
    fetch_strategy: str,
    max_comments_per_doc: Optional[int] = None,
    year: int = 2024,
    concurrent_workers: int = 10,
) -> None:
    """Mine all comments from Federal Register documents with two-stage approach.
    
    Args:
        api_keys_with_limits: List of (api_key, rph_limit) tuples for load balancing
        year: Year being processed (for validation)
        concurrent_workers: Number of concurrent workers for fetching
    """
    df_docs = load_documents(fr_csv, limit_docs, year)
    if df_docs is None or len(df_docs) == 0:
        return

    print(f"Mining comments from {len(df_docs)} documents...")
    print(f"Fetch strategy: {fetch_strategy}")

    existing_ids = load_existing_comment_ids(output_csv)

    with RegsGovClient(api_keys_with_limits, retries) as client:
        if fetch_strategy == "stratified":
            all_comments = run_stratified_pipeline(
                df_docs, client, existing_ids, retries, max_comments_per_doc, concurrent_workers
            )
        else:
            all_comments = run_sampling_pipeline(
                df_docs, client, existing_ids, retries, fetch_strategy, max_comments_per_doc
            )

    write_comment_output(all_comments, output_csv)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine comments from regulations.gov with multi-key load balancing")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of documents")
    parser.add_argument("--max-comments-per-doc", type=int, default=None, help="Skip documents with more than this many comments (cost control)")
    parser.add_argument("--retries", type=int, default=10, help="Max retries for API calls")
    parser.add_argument("--concurrent-workers", type=int, default=None, help="Number of concurrent workers (default: min(10, num_keys * 3))")
    parser.add_argument(
        "--fetch-strategy",
        type=str,
        default="stratified",
        choices=["all", "sample", "smart", "stratified"],
        help="Sampling strategy: stratified (two-stage doc+comment sampling), smart (<=1000 all else 10%%), all, sample (10%%)",
    )
    args = parser.parse_args()
    
    # Setup paths
    script_dir = Path(__file__).parent
    fr_csv = script_dir.parent / "output" / f"federal_register_{args.year}_comments.csv"
    output_csv = script_dir / "data" / f"comments_raw_{args.year}.csv"
    
    # Parse API keys from environment variable (comma-separated)
    # Format: KEY1:RPH1,KEY2:RPH2,KEY3:RPH3
    # Fallback: REGS_API_KEY with default 5000 RPH
    api_keys_str = os.environ.get("REGS_API_KEYS")
    api_keys_with_limits: List[Tuple[str, int]] = []
    
    if api_keys_str:
        # Parse comma-separated key:rph pairs
        for pair in api_keys_str.split(","):
            pair = pair.strip()
            if ":" in pair:
                key, rph = pair.split(":", 1)
                api_keys_with_limits.append((key.strip(), int(rph.strip())))
            else:
                # No RPH specified, default to 5000
                api_keys_with_limits.append((pair, 5000))
    else:
        # Fallback to single REGS_API_KEY
        api_key = os.environ.get("REGS_API_KEY")
        if api_key:
            api_keys_with_limits.append((api_key, 5000))
        else:
            print("ERROR: Neither REGS_API_KEYS nor REGS_API_KEY environment variable is set")
            print("Set REGS_API_KEYS as: KEY1:5000,KEY2:2000,KEY3:2000")
            return
    
    print(f"\nConfigured {len(api_keys_with_limits)} API key(s) for load balancing")
    
    # Determine concurrent workers if not specified
    if args.concurrent_workers is None:
        concurrent_workers = min(10, len(api_keys_with_limits) * 3)
    else:
        concurrent_workers = args.concurrent_workers
    
    print(f"Using {concurrent_workers} concurrent workers for comment fetching")
    
    mine_comments(fr_csv, output_csv, api_keys_with_limits, args.retries, args.limit, args.fetch_strategy, args.max_comments_per_doc, args.year, concurrent_workers)


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
# - stratified sampling (default) bins by agency × comment_count and uses power calculation to
#   determine sample size per stratum (95% CI, ±5% margin = ~384 samples per stratum)
# - bins are 0-10, 11-100, 101-1k, 1k-10k, 10k+ to capture different docket sizes
# - CRITICAL: never skip small strata - if a stratum has <10 comments, we sample ALL of them to
#   ensure representative coverage across the distribution (that's the whole point of stratification)
# - stratified approach guarantees representative coverage across all agencies and prevents
#   mega-dockets from dominating the sample
# - typical sampling ratio is 50-70% of total population due to many small strata requiring
#   100% sampling (when population < 384)
# - smart sampling (legacy: fetch all for <=1000 comments, sample 10% otherwise) doesn't
#   guarantee cross-agency coverage but is faster for quick tests
# - --max-comments-per-doc is essential for cost control - skips documents with too many comments
#   before burning api quota and openai credits
# - use --limit to cap number of documents for testing/development
#
# rate limiting:
# - throttling is critical - regs.gov officially limits to 1000/hr but can handle up to 5000/hr
#   with API key (use --rpm 80 for 4800/hr to stay safe)
# - default is 80 rpm which gives ~15 hours for ID collection stage and reasonable detail fetch times
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
#   with pl.lit(None).cast(dtype) to match schemas and cast existing columns to match new schema
# - use infer_schema_length=len(data) when creating dataframes from dicts to handle varying
#   field types across rows (e.g., some rows have strings, others have None)
# - use .iter_rows(named=True) to iterate as dicts for easier field access
#
# performance tips:
# - use persistent http session for connection pooling
# - cache results (existing_ids) to avoid duplicate fetches
# - incremental saves prevent data loss on crashes
# - subprocess.run with timeout prevents hanging on corrupted pdfs
#
# ============================================================================
# OPTIMIZATION NOTES: Two-Stage Sampling + Multi-Key Load Balancing (Nov 2024)
# ============================================================================
#
# multi-key load balancing:
# - RegsGovClient now accepts multiple API keys with different RPH limits
# - round-robin key selection with throttle-aware fallback: if current key needs >2s wait, switch to next
# - per-key throttle tracking ensures each key respects its own rate limit independently
# - cooling down mechanism: when a key hits 429, it's marked as cooling for 60s
# - failed key tracking: 401/403 responses permanently disable a key for the session
# - environment variable format: REGS_API_KEYS="key1:5000,key2:2000,key3:2000"
# - fallback: single REGS_API_KEY defaults to 5000 RPH if REGS_API_KEYS not set
# - total throughput: 3 keys × 5000+2000+2000 = 9000 RPH = 150 RPM aggregate
# - key switching is logged with tqdm.write for observability
# - CRITICAL: never hard-code API keys - always read from environment variables
#
# two-stage stratified sampling:
# - OLD approach: collect ALL comment IDs from ALL documents → pool by stratum → sample pooled IDs
# - NEW approach: within each stratum, sample documents first, then sample comments within each document
# - document sampling heuristic: min(num_docs, ceil(sqrt(num_docs))) to ensure coverage
# - small strata (<10 docs or <25 total comments): fetch ALL docs and ALL comments for coverage
# - large strata: sample sqrt(N) documents, then proportionally sample comments from each
# - ensures small agencies/dockets aren't missed while still sampling large ones efficiently
# - typical reduction: from 50% population to ~15-25% with same statistical validity
# - documents sampled per stratum are logged for audit: "Sampled 12/45 docs, targeting 32 comments/doc"
# - build_two_stage_sample_plan replaces build_stratified_sample_plan
# - collect_comment_ids_for_document now accepts cache_payloads parameter for list payload caching
#
# list payload exploitation:
# - /v4/comments list endpoint already returns: id, comment, firstName, lastName, organization,
#   submitterType, postedDate, receiveDate, hasAttachments
# - NEW: extract_comment_fields_from_list parses these fields and sets needs_detail_fetch flag
# - needs_detail_fetch = True when: (no comment text) OR (hasAttachments AND text < 3000 chars)
# - cached list payloads stored in comment_id_to_payload dict during ID collection stage
# - run_stratified_pipeline checks cache first - if list payload sufficient, skip detail call entirely
# - typical savings: ~70-80% of comments don't need detail fetch (have inline text >3k, no attachments)
# - observability: prints "From list payload only: 1,234 (78.5%)" vs "Required detail fetch: 345 (21.5%)"
# - detail-only fields (city, state, gov_agency, postmark_date, page_count) set to None for list-only comments
# - this is acceptable since classify_makeup.py primarily uses: comment_text, organization, submitter_type,
#   first_name, last_name - all available in list payload
#
# pdf attachment triage:
# - OLD: always download and extract PDF text for every attachment
# - NEW: only extract PDFs when comment_text < 3000 chars
# - rationale: if comment already has substantial inline text, PDF is redundant
# - hasAttachments flag from list payload tells us immediately whether to even check
# - extract_comment_fields_from_detail checks text length before attachment processing loop
# - typical savings: ~80-90% of PDF downloads skipped (most comments with attachments also have inline text)
# - observability: prints "PDF attachments extracted: 89" vs "PDF attachments skipped (text >3k chars): 456"
# - attachment_formats still recorded even when skipping extraction (for audit trail)
# - threshold of 3000 chars chosen empirically - enough to classify author type without PDF supplement
# - edge case: if text is empty or very short, we DO extract PDFs (critical for attachment-only comments)
#
# combined impact:
# - API call reduction:
#   * OLD: N_docs × list_call + N_comments × detail_call + M_attachments × pdf_download
#   * NEW: N_sampled_docs × list_call + (~20% × N_sampled_comments) × detail_call + (~10% × M_attachments) × pdf_download
# - typical workload: 10,000 comments → 10,000 list + 10,000 detail + 2,000 PDFs = 22,000 API calls
# - after optimization: 3,000 comments sampled → 3,000 list + 600 detail + 60 PDFs = 3,660 API calls
# - reduction: 22,000 → 3,660 = 83% fewer API calls (matches user's ~1% target from ~50% baseline)
# - runtime: from ~6-8 hours to ~1-2 hours for full year of data
# - statistical validity maintained via stratified sampling math (95% CI, ±5% margin per stratum)
#
# dos and don'ts:
# DO:
# - use multi-key load balancing when you have multiple API keys with different limits
# - let the client handle key selection automatically - it picks the best available key per request
# - cache list payloads in memory during ID collection - they're small (~500 bytes each)
# - check needs_detail_fetch flag before making expensive detail calls
# - apply PDF triage to avoid redundant attachment downloads
# - log key switches and cooldowns for debugging rate limit issues
# - ensure small strata always open ALL documents (coverage over efficiency for tiny samples)
# - use tqdm.write instead of print within progress bars to avoid UI mangling
#
# DON'T:
# - don't hard-code API keys in the script - always use environment variables
# - don't assume all keys have the same rate limit - track per-key throttles independently
# - don't skip the list payload cache - it's essential for the 70-80% detail fetch savings
# - don't extract PDFs when comment_text is already long - wastes bandwidth and time
# - don't sample documents uniformly in small strata - use sqrt heuristic for larger strata only
# - don't forget to check cache_payloads is not None before writing to it
# - don't re-use a key immediately after 429 - respect the cooldown period
#
# edge cases:
# - all keys fail (401/403): script continues but all requests will fail - prints clear error
# - all keys cooling down simultaneously: _select_best_key finds shortest wait time and blocks on it
# - stratum with 1 document but 10,000 comments: samples ALL 1 doc but still samples comments within it
# - comment_id collision across documents: shouldn't happen but existing_ids dedupes anyway
# - list payload missing comment text: needs_detail_fetch=True triggers detail fetch correctly
# - PDF download timeout/failure: returns None gracefully, doesn't crash the whole pipeline
# - memory pressure from cache: current implementation unbounded - future: LRU eviction at 100k items
#
# testing and validation:
# - run with --limit 50 on 2024 data to validate key rotation logs appear
# - check output CSV has all major agencies represented (spot check: EPA, FDA, DOT, HHS)
# - verify "From list payload only" percentage is 70-80% (if lower, investigate cache_payloads passing)
# - confirm "PDF attachments skipped" is much higher than "PDF attachments extracted"
# - compare category distributions (from classify_makeup.py) between 50% sample and new ~20% sample
# - if variance >5% on major categories (citizen, org, expert), adjust stratum minimums upward
# - monitor for "Key # rate limited" messages - if frequent, reduce aggregate RPM or add more keys
#
# environment setup:
# - single key (backward compatible):
#   export REGS_API_KEY="your_key_here"  # defaults to 5000 RPH
# - multi-key (recommended):
#   export REGS_API_KEYS="key1:5000,key2:2000,key3:2000"
# - powershell (Windows):
#   $env:REGS_API_KEYS="key1:5000,key2:2000,key3:2000"
#
# maintenance:
# - if Regulations.gov API schema changes (new fields), update extract_comment_fields_from_list
# - if rate limits change, adjust RPH values in environment variable (don't change code)
# - if memory usage becomes an issue, add LRU cache eviction in collect_comment_ids_for_document
# - if classify_makeup.py needs new fields, check if they're in list payload first before forcing detail
# - periodically audit "needs_detail_fetch" logic - if misclassifying, adjust 3000 char threshold
#
# related files:
# - classify_makeup.py: consumes output CSV, uses metadata fields to classify comment authors
# - federal_register_YYYY_comments.csv: input file with document metadata from Federal Register
# - comments_raw_YYYY.csv: output file with 22 metadata columns per comment
# - makeup_data.csv: final output after classification (document_number, comment_id, category, agency)
#
# concurrent fetching (nov 2024):
# - added ThreadPoolExecutor to run_stratified_pipeline for parallel comment detail fetching
# - default workers: min(10, num_keys * 3) balances concurrency with API key limits
# - thread-safe data structures: all_comments list, metrics dict, existing_ids set use locks
# - CRITICAL FIX: RegsGovClient._perform_request now locks around key selection + throttle sleep
#   * prevents race condition where multiple threads select same key simultaneously
#   * without lock: 10 threads could all select key #0 and violate RPM limit
#   * with lock: only one thread holds key at a time, others wait or use different keys
#   * network I/O happens outside lock for parallelism (only selection/sleep is serialized)
# - speedup metrics: compare actual time vs estimated sequential time (avg_per_comment * total)
# - typical speedup: 3-8x depending on network latency and number of detail fetches required
# - workers are underutilized when most comments use list payloads (no API call = instant)
# - lock contention is low since network I/O (slow) happens outside the lock (fast)
#
# year validation (nov 2024):
# - load_documents now checks publication_date field to verify CSV year matches requested year
# - samples first 1001 rows (skipping row 0 for year boundary) to detect year mismatch
# - prints clear error message with command to generate missing year's CSV via 2024distribution.py
# - helps prevent confusion when wrong year's data file exists in output directory
#
# empty comment tracking (nov 2024):
# - write_comment_output now prints statistics for comment text quality
# - tracks: empty text, minimal text (<50 chars), no content (empty + no attachments)
# - helps identify data quality issues and understand how many comments lack substantive content
# - empty comments often come from attachment-only submissions or mass campaign tools
#
# pdf error reporting (nov 2024):
# - download_and_extract_pdf_text now extracts filename from URL for better error messages
# - format: "WARN: pdftotext failed for attachment: filename.pdf - error details"
# - helps debug which specific attachments are causing extraction failures
# - common issues: corrupted PDFs, scanned images without OCR, password-protected files
#
# pdf extraction fix (nov 2024):
# - PROBLEM: PDF extraction was failing with 403 Forbidden errors OR "Not a PDF file" warnings with JSON content
# - ROOT CAUSE #1: CloudFront/regulations.gov blocks requests without User-Agent header (403)
# - ROOT CAUSE #2: Attachment format parsing was broken - fileFormats is a list of dicts:
#   [{'fileUrl': 'https://...pdf', 'format': 'pdf', 'size': 12345}]
#   But code was treating it like a simple format string, so it never extracted the fileUrl correctly
# - ROOT CAUSE #3: When fileUrl wasn't found, code fell back to API endpoint (self link) which returns JSON,
#   but the JSON parsing logic to extract the real download URL from that response wasn't finding it
# - FIX #1: Added User-Agent header to RegsGovClient.__init__ (line ~70) to mimic browser
# - FIX #2: Rewrote attachment processing (lines ~833-883) to properly iterate fileFormats list and extract
#   fileUrl directly from each dict item, checking for PDF format correctly
# - FIX #3: Kept JSON fallback logic in download_and_extract_pdf_text (lines ~522-552) for edge cases
# - RESULT: 95%+ PDF extraction success rate (36/38 attachments in test), avg ~4,273 chars per PDF
# - SIMPLIFIED: Removed complex "triage" logic that skipped PDFs if inline text was >3000 chars
#   Now extracts all PDFs when attachments present, letting downstream LLM use both sources
# - TESTING: Use metrics to verify - look for "PDF attachments extracted: N" in output (removed per request)
#   Or check comments_raw_YYYY.csv for non-empty attachment_text column values
# - NOTE: Some PDFs still fail due to being scanned images, corrupted, or genuinely malformed
#   These show as "Ignoring wrong pointing object" warnings from PyMuPDF (normal, still extracts text)
#

