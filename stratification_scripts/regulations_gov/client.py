"""
Regulations.gov API client with multi-key rate limiting.

This module provides a robust client for interacting with the regulations.gov
v4 API, including document lookups, comment listing, and attachment downloads.

Example:
    >>> from stratification_scripts.config import get_regs_api_keys
    >>> from stratification_scripts.regulations_gov import RegsGovClient
    >>> 
    >>> keys = get_regs_api_keys()
    >>> with RegsGovClient(keys, retries=5) as client:
    ...     doc = client.get_document_detail("EPA-HQ-2024-0001-0001")
    ...     print(doc["data"]["attributes"]["title"])
"""

from __future__ import annotations

import random
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import requests
from tqdm import tqdm

from stratification_scripts.logging_utils import get_logger
from stratification_scripts.regulations_gov.rate_limiter import MultiKeyLimiter

logger = get_logger(__name__)

# API endpoints
REGS_DOC_URL = "https://api.regulations.gov/v4/documents/{documentId}"
REGS_DOCS_SEARCH_URL = "https://api.regulations.gov/v4/documents"
REGS_COMMENTS_URL = "https://api.regulations.gov/v4/comments"
REGS_COMMENT_DETAIL_URL = "https://api.regulations.gov/v4/comments/{commentId}"


class RegsThrottle:
    """
    Simple rate limiter for a single API key.
    
    Used internally by RegsGovClient for per-key throttling.
    """
    
    def __init__(self, rpm: int = 30) -> None:
        self.rpm = rpm
        self.last_call_ts: float = 0.0

    @property
    def min_interval(self) -> float:
        """Minimum seconds between requests."""
        return 60.0 / self.rpm if self.rpm and self.rpm > 0 else 0.0

    def sleep_if_needed(self) -> None:
        """Sleep if necessary to respect rate limit."""
        if self.min_interval <= 0:
            return
        jitter = random.uniform(-0.05, 0.05) * self.min_interval
        wait = (self.min_interval + jitter) - (time.time() - self.last_call_ts)
        if wait > 0:
            time.sleep(wait)
        self.last_call_ts = time.time()


class RegsGovClient:
    """
    Regulations.gov v4 API client with multi-key rotation and caching.
    
    This client provides:
    - Multi-key load balancing with round-robin selection
    - Per-key rate limiting and throttling
    - Thread-safe caching of document details
    - Automatic retry with exponential backoff
    - Cooldown handling for 429 responses
    
    Attributes:
        session: HTTP session for connection pooling
        retries: Maximum retry attempts per request
        limiter: MultiKeyLimiter for key management
    
    Example:
        >>> keys = [("key1", 5000), ("key2", 1000)]
        >>> with RegsGovClient(keys, retries=5) as client:
        ...     doc = client.get_document_detail("EPA-HQ-2024-0001")
        ...     for comment in client.iter_comments_for_object(obj_id):
        ...         print(comment["id"])
    """

    def __init__(
        self,
        api_keys_with_limits: List[Tuple[str, int]],
        retries: int = 5,
    ) -> None:
        """
        Initialize client with API keys and their RPH limits.
        
        Args:
            api_keys_with_limits: List of (api_key, requests_per_hour) tuples.
            retries: Maximum retry attempts per request.
        
        Raises:
            ValueError: If no API keys provided.
        """
        if not api_keys_with_limits:
            raise ValueError("Must provide at least one API key")
        
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/91.0.4472.124 Safari/537.36"
            )
        })
        self.retries = max(1, retries)
        
        # Multi-key rate limiter
        self.limiter = MultiKeyLimiter(api_keys_with_limits)
        
        # Thread-safe caches
        self._cache_lock = threading.Lock()
        self._doc_cache: Dict[str, dict] = {}
        self._total_cache: Dict[str, int] = {}
        
        # Track failed/cooling keys
        self._failed_keys: Set[int] = set()
        self._cooling_down: Dict[int, float] = {}
        
        # Key list for round-robin
        self._keys = [k for k, _ in api_keys_with_limits]
        self._throttles = [
            RegsThrottle(rpm=max(1, rph // 60))
            for _, rph in api_keys_with_limits
        ]
        self._current_key_idx = 0
        self._request_lock = threading.Lock()
        
        logger.info(f"Initialized RegsGovClient with {len(self._keys)} API keys")
        for idx, (_, rph) in enumerate(api_keys_with_limits):
            logger.debug(f"  Key #{idx}: {rph} RPH ({rph/60:.1f} RPM)")

    def __enter__(self) -> "RegsGovClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        """Close the HTTP session."""
        self.session.close()

    def _sleep_with_backoff(self, backoff: float) -> float:
        """Sleep and return next backoff value."""
        time.sleep(backoff)
        return min(backoff * 2, 16.0)

    def _select_best_key(self) -> int:
        """Select the best available key using round-robin with throttle fallback."""
        now = time.time()
        
        # Remove expired cooldowns
        expired = [k for k, until in self._cooling_down.items() if now >= until]
        for k in expired:
            del self._cooling_down[k]
        
        # Try round-robin starting from current key
        for offset in range(len(self._keys)):
            candidate_idx = (self._current_key_idx + offset) % len(self._keys)
            
            if candidate_idx in self._failed_keys:
                continue
            if candidate_idx in self._cooling_down:
                continue
            
            throttle = self._throttles[candidate_idx]
            wait_time = throttle.min_interval - (now - throttle.last_call_ts)
            
            if wait_time < 2.0:
                if offset > 0:
                    logger.debug(
                        f"Switched to Key #{candidate_idx} "
                        f"(Key #{self._current_key_idx} throttled)"
                    )
                self._current_key_idx = candidate_idx
                return candidate_idx
        
        # All keys throttled - find shortest wait
        best_idx = None
        min_wait = float("inf")
        
        for idx in range(len(self._keys)):
            if idx in self._failed_keys or idx in self._cooling_down:
                continue
            
            throttle = self._throttles[idx]
            wait_time = throttle.min_interval - (now - throttle.last_call_ts)
            if wait_time < min_wait:
                min_wait = wait_time
                best_idx = idx
        
        if best_idx is None:
            # All keys failed - use first non-failed
            for idx in range(len(self._keys)):
                if idx not in self._failed_keys:
                    best_idx = idx
                    break
            if best_idx is None:
                best_idx = 0
        
        self._current_key_idx = best_idx
        return best_idx

    def _perform_request(
        self,
        url: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        timeout: int = 30,
    ) -> Optional[requests.Response]:
        """
        Perform HTTP GET with multi-key rotation and retry logic.
        
        Args:
            url: Request URL
            params: Query parameters
            timeout: Request timeout in seconds
        
        Returns:
            Response object or None if all retries failed.
        """
        backoff = 1.0
        
        for attempt in range(self.retries):
            # Lock for key selection and throttle
            with self._request_lock:
                key_idx = self._select_best_key()
                api_key = self._keys[key_idx]
                throttle = self._throttles[key_idx]
                headers = {"X-Api-Key": api_key}
                throttle.sleep_if_needed()
            
            # Make request outside the lock
            try:
                response = self.session.get(
                    url,
                    headers=headers,
                    params=params,
                    timeout=timeout,
                )
            except requests.RequestException as e:
                logger.debug(f"Request failed: {e}")
                backoff = self._sleep_with_backoff(backoff)
                continue

            if response.status_code == 200:
                return response

            # Handle auth failures
            if response.status_code in (401, 403):
                with self._request_lock:
                    self._failed_keys.add(key_idx)
                logger.warning(
                    f"Key #{key_idx} failed auth (HTTP {response.status_code}) - "
                    "marking as unusable"
                )
                if len(self._failed_keys) < len(self._keys):
                    continue
                return response

            # Handle rate limits
            if response.status_code == 429:
                with self._request_lock:
                    self._cooling_down[key_idx] = time.time() + 60
                try:
                    retry_after = int(response.headers.get("Retry-After", "60"))
                except (ValueError, TypeError):
                    retry_after = 60
                logger.debug(f"Key #{key_idx} rate limited, cooling for {retry_after}s")
                self.limiter.on_429(api_key, retry_after)
                backoff = self._sleep_with_backoff(backoff)
                continue

            # Handle server errors
            if response.status_code in (500, 502, 503, 504):
                logger.debug(f"Server error {response.status_code}, retrying...")
                backoff = self._sleep_with_backoff(backoff)
                continue

            # Non-retriable error
            return response

        return None

    def request_json(
        self,
        url: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        timeout: int = 30,
    ) -> Optional[dict]:
        """
        Make a GET request and return JSON response.
        
        Args:
            url: Request URL
            params: Query parameters
            timeout: Request timeout
        
        Returns:
            Parsed JSON dict or None if request failed.
        """
        response = self._perform_request(url, params=params, timeout=timeout)
        if response is None:
            return None

        if response.status_code == 200:
            try:
                return response.json() or {}
            except ValueError:
                return {}

        return None

    def download_bytes(self, url: str, timeout: int = 30) -> Optional[bytes]:
        """
        Download binary content from URL.
        
        Args:
            url: URL to download from
            timeout: Request timeout
        
        Returns:
            Raw bytes or None if download failed.
        """
        response = self._perform_request(url, timeout=timeout)
        if response and response.status_code == 200:
            return response.content
        return None

    def get_document_detail(
        self,
        document_id: str,
    ) -> Optional[dict]:
        """
        Fetch document detail from Regulations.gov (cached).
        
        Args:
            document_id: Regulations.gov document ID (e.g., "EPA-HQ-2024-0001")
        
        Returns:
            Document data dict or None if not found.
        
        Side Effects:
            Caches successful responses for reuse.
        """
        if not document_id:
            return None
        
        # Check cache
        with self._cache_lock:
            if document_id in self._doc_cache:
                return self._doc_cache[document_id]
        
        # Fetch
        data = self.request_json(
            REGS_DOC_URL.format(documentId=document_id),
            timeout=20,
        )
        
        if data:
            with self._cache_lock:
                self._doc_cache[document_id] = data
        
        return data

    def get_comment_total_by_object_id(
        self,
        object_id: str,
    ) -> Optional[int]:
        """
        Fetch comment count for an object ID (cached).
        
        Args:
            object_id: Regulations.gov object ID
        
        Returns:
            Total comment count or None if not found.
        
        Side Effects:
            Caches successful responses for reuse.
        """
        if not object_id:
            return None
        
        # Check cache
        with self._cache_lock:
            if object_id in self._total_cache:
                return self._total_cache[object_id]
        
        # Fetch
        data = self.request_json(
            REGS_COMMENTS_URL,
            params={"filter[commentOnId]": object_id, "page[size]": 5},
        )
        
        if not data:
            return None
        
        try:
            total = (data.get("meta") or {}).get("totalElements")
            if total is not None:
                val = int(total)
                with self._cache_lock:
                    self._total_cache[object_id] = val
                return val
        except (ValueError, TypeError):
            pass
        
        return None

    def get_object_id_for_document(
        self,
        document_id: str,
    ) -> Optional[str]:
        """
        Get the objectId for a document (used for comment queries).
        
        Args:
            document_id: Regulations.gov document ID
        
        Returns:
            Object ID string or None if not found.
        """
        data = self.get_document_detail(document_id)
        if not data:
            return None
        
        attrs = (data.get("data") or {}).get("attributes") or {}
        return attrs.get("objectId") or attrs.get("objectID")

    def find_object_ids_for_docket(
        self,
        docket_id: str,
    ) -> List[str]:
        """
        Find objectIds for non-comment documents in a docket.

        Queries /v4/documents?filter[docketId]=... to find the primary
        rulemaking documents (Rules, Proposed Rules, Notices) — excluding
        public submissions (which are comments themselves).

        Args:
            docket_id: Regulations.gov docket ID (e.g., "EPA-R09-OAR-2012-0936")

        Returns:
            List of objectId strings for commentable documents.
        """
        if not docket_id:
            return []

        data = self.request_json(
            REGS_DOCS_SEARCH_URL,
            params={
                "filter[docketId]": docket_id,
                "page[size]": 25,
            },
        )
        if not data:
            return []

        object_ids = []
        for item in data.get("data") or []:
            attrs = item.get("attributes") or {}
            doc_type = attrs.get("documentType", "")
            # Skip public submissions — those are comments, not commentable docs
            if doc_type == "Public Submission":
                continue
            obj_id = attrs.get("objectId") or attrs.get("objectID")
            if obj_id:
                object_ids.append(obj_id)

        return object_ids

    def resolve_docket_documents(
        self,
        docket_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Find non-submission documents in a docket with full metadata.

        Like find_object_ids_for_docket but returns richer data needed for
        Step 1 docket-to-document resolution: document_id, objectId,
        documentType, and commentCount.

        Args:
            docket_id: Regulations.gov docket ID (e.g., "FDA-2012-N-1148")

        Returns:
            List of dicts with keys: document_id, object_id, document_type,
            comment_count, title.  Sorted by comment_count descending so
            the most-commented document is first.
        """
        if not docket_id:
            return []

        data = self.request_json(
            REGS_DOCS_SEARCH_URL,
            params={
                "filter[docketId]": docket_id,
                "page[size]": 25,
            },
        )
        if not data:
            return []

        docs = []
        for item in data.get("data") or []:
            attrs = item.get("attributes") or {}
            doc_type = attrs.get("documentType", "")
            if doc_type == "Public Submission":
                continue
            obj_id = attrs.get("objectId") or attrs.get("objectID")
            doc_id = item.get("id") or attrs.get("documentId")
            cc = attrs.get("commentCount")
            if not isinstance(cc, int):
                cc = None
            if obj_id or doc_id:
                docs.append({
                    "document_id": doc_id,
                    "object_id": obj_id,
                    "document_type": doc_type,
                    "comment_count": cc,
                    "title": attrs.get("title", ""),
                })

        # Prefer primary commentable documents over supporting material.
        # When comment_count is available (rare in list endpoint), use it
        # as tiebreaker within the same priority tier.
        _DOC_TYPE_PRIORITY = {
            "Rule": 0,
            "Proposed Rule": 0,
            "Notice": 1,
            "Other": 2,
            "Supporting & Related Material": 3,
        }
        docs.sort(
            key=lambda d: (
                _DOC_TYPE_PRIORITY.get(d.get("document_type", ""), 2),
                -(d.get("comment_count") or 0),
            ),
        )
        return docs

    def iter_comments_for_object(
        self,
        object_id: str,
        limit_comments: Optional[int] = None,
    ) -> Iterable[dict]:
        """
        Yield comment records for an object with windowed paging.
        
        Uses lastModifiedDate windowing to bypass the 5000 result cap.
        
        Args:
            object_id: Regulations.gov object ID
            limit_comments: Maximum comments to yield (None = all)
        
        Yields:
            Comment data dicts from the list endpoint.
        """
        if not object_id:
            return

        MAX_WINDOWS = 1000  # Safety cap: 1000 windows * 20 pages * 250 items = 5M comments max

        fetched = 0
        cursor_ge: Optional[str] = None
        window = 0

        while window < MAX_WINDOWS:
            window += 1
            page = 1
            last_seen_ts: Optional[str] = None

            while page <= 20:
                params: Dict[str, Any] = {
                    "filter[commentOnId]": object_id,
                    "page[size]": 250,
                    "page[number]": page,
                    "sort": "lastModifiedDate,documentId",
                }
                if cursor_ge:
                    params["filter[lastModifiedDate][ge]"] = cursor_ge

                data = self.request_json(REGS_COMMENTS_URL, params=params, timeout=30)
                if data is None:
                    logger.warning(
                        "Paging terminated early for object_id=%s: "
                        "request returned None at window=%d, page=%d "
                        "(fetched %d comments so far, cursor=%r)",
                        object_id, window, page, fetched, cursor_ge,
                    )
                    return

                items: List[dict] = data.get("data") or []
                if not items:
                    break

                last_item = items[-1]
                attrs_last = last_item.get("attributes") or {}
                last_seen_ts = (
                    attrs_last.get("lastModifiedDate") or
                    attrs_last.get("modifyDate")
                )

                for item in items:
                    yield item
                    fetched += 1
                    if limit_comments is not None and fetched >= limit_comments:
                        return

                page += 1

            if not last_seen_ts or cursor_ge == last_seen_ts:
                return
            cursor_ge = last_seen_ts

        # Safety cap reached
        logger.error(
            "Safety cap reached for object_id=%s: "
            "%d windows exhausted, fetched %d comments. Last cursor: %r",
            object_id, MAX_WINDOWS, fetched, cursor_ge,
        )

    def fetch_comment_detail(
        self,
        comment_id: str,
        include_attachments: bool = True,
    ) -> Optional[dict]:
        """
        Fetch full comment detail from the detail endpoint.
        
        Args:
            comment_id: Comment ID to fetch
            include_attachments: Whether to include attachment data
        
        Returns:
            Full comment detail dict or None if not found.
        """
        if not comment_id:
            return None

        params: Dict[str, str] = {}
        if include_attachments:
            params["include"] = "attachments"

        return self.request_json(
            REGS_COMMENT_DETAIL_URL.format(commentId=comment_id),
            params=params,
            timeout=20,
        )

    def clear_cache(self) -> None:
        """Clear all cached data."""
        with self._cache_lock:
            self._doc_cache.clear()
            self._total_cache.clear()

