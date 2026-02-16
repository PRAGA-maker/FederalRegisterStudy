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

import math
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests

from stratification_scripts.logging_utils import get_logger

logger = get_logger(__name__)


def extract_rin(details: dict) -> Optional[str]:
    """
    Extract primary RIN from FR API document details.

    Checks multiple fields where RIN might be stored:
    - regulation_id_numbers (array)
    - rin (legacy string field)
    - regulation_id_number_info.rin

    Args:
        details: Full document details dict from FR API

    Returns:
        Primary RIN string (e.g., "0648-AU02") or None if not found.
    """
    if not details:
        return None

    # Try regulation_id_numbers array (most common)
    rins = details.get("regulation_id_numbers")
    if rins and isinstance(rins, list) and len(rins) > 0:
        first_rin = rins[0]
        if isinstance(first_rin, str) and first_rin.strip():
            return first_rin.strip().upper()

    # Try rin field (legacy)
    rin = details.get("rin")
    if isinstance(rin, str) and rin.strip():
        return rin.strip().upper()

    # Try regulation_id_number_info
    rin_info = details.get("regulation_id_number_info")
    if isinstance(rin_info, dict):
        rin = rin_info.get("rin")
        if isinstance(rin, str) and rin.strip():
            return rin.strip().upper()

    return None


def extract_all_rins(details: dict) -> List[str]:
    """
    Extract all RINs from FR API document details.

    Used for joint rulemakings where a document may have multiple RINs.

    Args:
        details: Full document details dict from FR API

    Returns:
        List of RIN strings, empty list if none found.
    """
    if not details:
        return []

    rins = []

    # Get from regulation_id_numbers array
    rin_list = details.get("regulation_id_numbers")
    if rin_list and isinstance(rin_list, list):
        for r in rin_list:
            if isinstance(r, str) and r.strip():
                rins.append(r.strip().upper())

    # Add from other sources if not already present
    for rin in [details.get("rin"), (details.get("regulation_id_number_info") or {}).get("rin")]:
        if isinstance(rin, str) and rin.strip():
            rin_upper = rin.strip().upper()
            if rin_upper not in rins:
                rins.append(rin_upper)

    return rins


def extract_docket_id(details: dict) -> Optional[str]:
    """
    Extract primary docket ID from FR API document details.

    Checks multiple fields where docket ID might be stored:
    - docket_ids (array)
    - docket_id (string)
    - regulations_dot_gov_info.docket_id

    Args:
        details: Full document details dict from FR API

    Returns:
        Primary docket ID string or None if not found.
    """
    if not details:
        return None

    # Try docket_ids array
    docket_ids = details.get("docket_ids")
    if docket_ids and isinstance(docket_ids, list) and len(docket_ids) > 0:
        first_id = docket_ids[0]
        if isinstance(first_id, str) and first_id.strip():
            return first_id.strip()

    # Try docket_id field
    docket_id = details.get("docket_id")
    if isinstance(docket_id, str) and docket_id.strip():
        return docket_id.strip()

    # Try regulations_dot_gov_info
    regs_info = details.get("regulations_dot_gov_info")
    if isinstance(regs_info, dict):
        docket_id = regs_info.get("docket_id")
        if isinstance(docket_id, str) and docket_id.strip():
            return docket_id.strip()

    return None


_FR_CITATION_RE = re.compile(r"(\d+)\s+FR\s+(\d+)")


def parse_fr_citation(citation: str) -> Optional[Tuple[int, int]]:
    """
    Parse a Federal Register citation string into (volume, page).

    Args:
        citation: String like "80 FR 69589" or "81 FR 57854".

    Returns:
        Tuple of (volume, page) as integers, or None if parsing fails.

    Examples:
        >>> parse_fr_citation("80 FR 69589")
        (80, 69589)
        >>> parse_fr_citation("")
        None
    """
    if not citation:
        return None
    m = _FR_CITATION_RE.search(citation)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def fr_volume_to_year(volume: int) -> int:
    """
    Convert Federal Register volume number to calendar year.

    The Federal Register began in 1936 as Volume 1.
    Each calendar year gets a new volume number.

    Examples:
        >>> fr_volume_to_year(80)
        2015
        >>> fr_volume_to_year(89)
        2024
    """
    return volume + 1935


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
        enrich_identifiers: bool = True,
    ) -> Optional[dict]:
        """
        Fetch detailed information for a Federal Register document.

        This fetches the full document record including:
        - comment_url
        - comments_close_on
        - regulations_dot_gov_info (with document_id and comments_count)
        - rin (Regulation Identifier Number, extracted)
        - rin_all (all RINs for joint rulemakings)
        - docket_id (extracted from various fields)

        Args:
            document_number: The FR document number (e.g., "2024-12345")
            enrich_identifiers: If True, extract and add rin/docket_id fields

        Returns:
            Document data dict or None if fetch failed.
            When enrich_identifiers=True, adds:
                - rin: Primary RIN string or None
                - rin_all: List of all RINs (for joint rulemakings)
                - docket_id: Primary docket ID or None

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
                data = r.json()

                # Enrich with extracted identifiers
                if enrich_identifiers and data:
                    data["rin"] = extract_rin(data)
                    data["rin_all"] = extract_all_rins(data)
                    data["docket_id"] = extract_docket_id(data)

                return data

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

    def lookup_document_by_citation(
        self,
        citation: str,
        publication_date: Optional[str] = None,
    ) -> Optional[dict]:
        """
        Look up a Federal Register document by its FR citation.

        Parses the citation (e.g., "81 FR 57854") into volume and page,
        then queries the FR API to find the matching document.

        When publication_date is provided (from a timetable entry), queries
        that exact day for efficiency. Otherwise, estimates the month from
        volume/page and searches within a window.

        Args:
            citation: FR citation string (e.g., "81 FR 57854").
            publication_date: Optional ISO date string (e.g., "2016-08-24")
                from the timetable entry. Greatly improves lookup efficiency.

        Returns:
            Dict with document_number, title, type, publication_date, citation
            for the matched document, or None if not found.
        """
        parsed = parse_fr_citation(citation)
        if not parsed:
            logger.warning(f"Cannot parse FR citation: {citation!r}")
            return None

        volume, page = parsed
        year = fr_volume_to_year(volume)

        # Rate limit
        if self.sleep_between > 0:
            time.sleep(self.sleep_between)

        fields = [
            "document_number", "title", "type", "start_page",
            "end_page", "citation", "volume", "publication_date",
        ]

        if publication_date:
            # Exact day query — most efficient
            date_windows = [(publication_date, publication_date)]
        else:
            # Estimate month from page number (FR pages increase through year)
            # Typical volume has ~80000 pages; estimate month proportionally
            est_month = max(1, min(12, math.ceil(page / 80000 * 12)))
            # Search a 60-day window centered on the estimated month
            mid_day = min(28, 15)  # middle of month
            from datetime import datetime, timedelta
            center = datetime(year, est_month, mid_day)
            start = (center - timedelta(days=30)).strftime("%Y-%m-%d")
            end = (center + timedelta(days=30)).strftime("%Y-%m-%d")
            date_windows = [(start, end)]

        for date_start, date_end in date_windows:
            params: Dict[str, Any] = {"per_page": 1000, "page": 1}
            for f in fields:
                params[f"fields[]"] = fields  # requests lib handles list
                break
            params["fields[]"] = fields

            if date_start == date_end:
                params["conditions[publication_date][is]"] = date_start
            else:
                params["conditions[publication_date][gte]"] = date_start
                params["conditions[publication_date][lte]"] = date_end

            backoff = 1.0
            for attempt in range(self.max_retries):
                try:
                    r = self.session.get(FR_DOCS_URL, params=params, timeout=30)
                except requests.RequestException as e:
                    if attempt == self.max_retries - 1:
                        logger.warning(f"FR API error looking up {citation}: {e}")
                        return None
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 16)
                    continue

                if r.status_code == 200:
                    break
                if r.status_code in (403, 429, 500, 502, 503, 504):
                    if attempt == self.max_retries - 1:
                        logger.warning(
                            f"FR API HTTP {r.status_code} looking up {citation} "
                            f"after {self.max_retries} retries"
                        )
                        return None
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 16)
                    continue
                logger.warning(f"FR API HTTP {r.status_code} looking up {citation}")
                return None
            else:
                return None  # exhausted retries without break

            data = r.json()
            results = data.get("results", [])

            # Scan for matching start_page
            for doc in results:
                if doc.get("start_page") == page or str(doc.get("start_page")) == str(page):
                    return {
                        "document_number": doc.get("document_number"),
                        "title": doc.get("title"),
                        "type": doc.get("type"),
                        "publication_date": doc.get("publication_date"),
                        "citation": doc.get("citation"),
                    }

            # If exact-day query returned no match, something is off
            if publication_date:
                logger.info(
                    f"Citation {citation} not found on date {publication_date} "
                    f"({len(results)} docs checked)"
                )

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
            document_types = ["PRORULE", "NOTICE", "RULE"]
        
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

