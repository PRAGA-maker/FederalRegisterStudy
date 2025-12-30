"""
Federal Register API client.

This module provides a client for interacting with the Federal Register API
to fetch document metadata, including comment information.

Example:
    >>> from stratification_scripts.federal_register import FederalRegisterClient
    >>> 
    >>> client = FederalRegisterClient()
    >>> doc = client.fetch_document_details("2024-12345")
    >>> print(doc["title"])
    >>> 
    >>> for doc in client.iter_documents_by_day(2024):
    ...     print(doc["document_number"])
"""

from __future__ import annotations

import time
from typing import Iterable, List, Optional

import pandas as pd
import requests

from stratification_scripts.logging_utils import get_logger

logger = get_logger(__name__)

# API Endpoints
FR_DOCS_URL = "https://www.federalregister.gov/api/v1/documents.json"
FR_DOC_DETAIL_URL = "https://www.federalregister.gov/api/v1/documents/{document_number}.json"


class FederalRegisterClient:
    """
    Client for the Federal Register API.
    
    This client provides methods for:
    - Fetching document details by document number
    - Iterating through documents by publication date
    - Rate limiting to avoid 403 blocks
    
    The Federal Register API has aggressive IP-based rate limiting,
    so requests are throttled by default.
    
    Attributes:
        session: HTTP session for connection pooling
        max_retries: Maximum retry attempts per request
        sleep_between: Seconds to wait between requests
    
    Example:
        >>> client = FederalRegisterClient(sleep_between=0.5)
        >>> 
        >>> # Fetch a single document
        >>> doc = client.fetch_document_details("2024-12345")
        >>> 
        >>> # Iterate through all 2024 documents
        >>> for doc in client.iter_documents_by_day(2024, limit=100):
        ...     print(f"{doc['document_number']}: {doc['title']}")
    """

    def __init__(
        self,
        max_retries: int = 10,
        sleep_between: float = 0.5,
    ) -> None:
        """
        Initialize the Federal Register client.
        
        Args:
            max_retries: Maximum retry attempts for failed requests.
            sleep_between: Seconds to wait between requests to avoid rate limiting.
        """
        self.session = requests.Session()
        self.max_retries = max(1, max_retries)
        self.sleep_between = max(0.0, sleep_between)
        
        logger.debug(
            f"Initialized FederalRegisterClient with "
            f"retries={max_retries}, sleep={sleep_between}s"
        )

    def __enter__(self) -> "FederalRegisterClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        """Close the HTTP session."""
        self.session.close()

    def fetch_document_details(
        self,
        document_number: str,
    ) -> Optional[dict]:
        """
        Fetch detailed information for a Federal Register document.
        
        This fetches the full document record including:
        - comment_url
        - comments_close_on
        - regulations_dot_gov_info (with document_id and comments_count)
        
        Args:
            document_number: The FR document number (e.g., "2024-12345")
        
        Returns:
            Document data dict or None if fetch failed.
        
        Side Effects:
            Sleeps for self.sleep_between seconds before making request.
        """
        if not document_number:
            return None
        
        # Rate limit
        if self.sleep_between > 0:
            time.sleep(self.sleep_between)
        
        backoff = 1.0
        url = FR_DOC_DETAIL_URL.format(document_number=document_number)
        
        for attempt in range(self.max_retries):
            try:
                r = self.session.get(url, timeout=30)
            except requests.RequestException as e:
                if attempt == self.max_retries - 1:
                    logger.debug(f"FR API error for {document_number}: {type(e).__name__}")
                time.sleep(backoff)
                backoff = min(backoff * 2, 16)
                continue
            
            if r.status_code == 200:
                return r.json()
            
            # Handle rate limiting and server errors
            if r.status_code in (403, 429, 500, 502, 503, 504):
                if attempt == self.max_retries - 1:
                    logger.debug(
                        f"FR API HTTP {r.status_code} for {document_number} "
                        f"after {self.max_retries} retries"
                    )
                time.sleep(backoff)
                backoff = min(backoff * 2, 16)
                continue
            
            # Non-retriable error
            if attempt == self.max_retries - 1:
                logger.debug(f"FR API HTTP {r.status_code} for {document_number}")
            return None
        
        return None

    def iter_documents_by_day(
        self,
        year: int,
        *,
        fr_sleep: float = 0.2,
        limit: Optional[int] = None,
        document_types: Optional[List[str]] = None,
    ) -> Iterable[dict]:
        """
        Yield FR documents for the given year, day by day.
        
        Uses daily pagination to avoid the 10,000 result limit on year-long queries.
        
        Args:
            year: Year to fetch documents for.
            fr_sleep: Seconds to sleep between page requests.
            limit: Maximum number of documents to yield (None = all).
            document_types: Types to filter (default: ["PRORULE", "NOTICE"]).
        
        Yields:
            Document data dicts from the list endpoint.
        
        Note:
            This only yields documents matching the requested year based on
            publication_date to avoid wrong-year data from API edge cases.
        """
        if document_types is None:
            document_types = ["PRORULE", "NOTICE"]
        
        yielded = 0
        dates = pd.date_range(
            start=f"{year}-01-01",
            end=f"{year}-12-31"
        ).strftime("%Y-%m-%d").tolist()
        
        for day in dates:
            page = 1
            total_pages = None
            
            while True:
                params = {
                    "per_page": 1000,
                    "page": page,
                    "conditions[publication_date][is]": day,
                    "conditions[type][]": document_types,
                }
                
                backoff = 1.0
                r = None
                
                for _ in range(max(1, self.max_retries)):
                    try:
                        r = self.session.get(FR_DOCS_URL, params=params, timeout=30)
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
                    except (ValueError, TypeError):
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
                            return
                
                page += 1
                if fr_sleep > 0:
                    time.sleep(fr_sleep)
                if total_pages is not None and page > total_pages:
                    break


def classify_submission_channel(
    comment_url: Optional[str],
    regs_document_id: Optional[str],
) -> str:
    """
    Classify how comments are submitted for a document.
    
    Args:
        comment_url: URL where comments can be submitted
        regs_document_id: Regulations.gov document ID if available
    
    Returns:
        One of: 'regs.gov', 'fcc-ecfs', 'ferc-portal', 'sec-portal',
        'email', 'other-portal', 'unknown'
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

