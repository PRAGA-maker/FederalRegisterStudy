"""
Reginfo.gov client for Unified Agenda data.

This module provides a client for fetching lifecycle stage data from the
Unified Agenda of Federal Regulatory and Deregulatory Actions on reginfo.gov.

The Unified Agenda is published semi-annually (Spring and Fall) and tracks
rulemakings through stages: Prerule, Proposed Rule, Final Rule, Long-Term
Actions, and Completed Actions.

Example:
    >>> from stratification_scripts.reginfo import RegInfoClient
    >>>
    >>> client = RegInfoClient()
    >>> agenda = client.fetch_unified_agenda("0648-AU02")
    >>> print(agenda["stage"])  # "COMPLETED"
    >>> print(agenda["timetable"])  # List of actions with dates
"""

from __future__ import annotations

import json
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from stratification_scripts.logging_utils import get_logger

logger = get_logger(__name__)

# URL patterns for reginfo.gov
# XML endpoint for individual RIN lookup
UNIFIED_AGENDA_XML_URL = "https://www.reginfo.gov/public/do/eAgendaViewRule"
# Alternative XML endpoint
UNIFIED_AGENDA_XML_ALT_URL = "https://www.reginfo.gov/public/xml/Spring{year}/RIN/{rin}.xml"

# Stage mappings
STAGE_MAP = {
    "Prerule Stage": "PRERULE",
    "Proposed Rule Stage": "PROPOSED",
    "Final Rule Stage": "FINAL",
    "Long-Term Actions": "LONG_TERM",
    "Completed Actions": "COMPLETED",
    # Alternative spellings found in API
    "Prerule": "PRERULE",
    "Proposed Rule": "PROPOSED",
    "Final Rule": "FINAL",
    "Long-Term": "LONG_TERM",
    "Completed": "COMPLETED",
}

# Lifecycle stage enum values
LIFECYCLE_STAGES = [
    # Pre-rule stages
    "PRERULE_ANPRM",      # Advance Notice of Proposed Rulemaking
    "PRERULE_RFI",        # Request for Information

    # Proposed rule stages
    "NPRM_ACTIVE",        # Comment period open
    "NPRM_CLOSED",        # Comment period closed, awaiting final

    # Final rule stages
    "INTERIM_FINAL",      # Interim final rule (effective, still accepting comments)
    "FINAL_PENDING",      # Final rule published, not yet effective
    "FINAL_EFFECTIVE",    # Final rule in effect

    # Terminal stages
    "WITHDRAWN",          # Rule withdrawn
    "LONG_TERM_STALLED",  # In Unified Agenda "Long-Term Actions"

    # Unknown/missing data
    "NO_RIN",             # Document has no RIN
    "AGENDA_NOT_FOUND",   # RIN not found in Unified Agenda
    "UNKNOWN",            # Could not determine stage
]


# Accumulates all unique timetable action types seen across the entire run.
# Inspect after pipeline completes: from stratification_scripts.reginfo.client import ALL_OBSERVED_ACTIONS
ALL_OBSERVED_ACTIONS: set = set()


class RegInfoClient:
    """
    Client for fetching Unified Agenda data from reginfo.gov.

    The Unified Agenda tracks the lifecycle of federal rulemakings using
    Regulation Identifier Numbers (RINs). This client fetches and parses
    the agenda data to determine a document's lifecycle stage.

    Attributes:
        session: HTTP session for connection pooling
        sleep_between: Seconds to wait between requests
        max_retries: Maximum retry attempts per request
        cache: In-memory cache of agenda data by RIN

    Example:
        >>> client = RegInfoClient()
        >>>
        >>> # Fetch lifecycle data for a RIN
        >>> agenda = client.fetch_unified_agenda("0648-AU02")
        >>> if agenda:
        ...     print(f"Stage: {agenda['stage']}")
        ...     print(f"Timetable: {agenda['timetable']}")
        >>>
        >>> # Determine specific lifecycle stage
        >>> stage = client.determine_lifecycle_stage(
        ...     agenda, doc_type="Proposed Rule", has_open_comments=True
        ... )
        >>> print(stage)  # "NPRM_ACTIVE"
    """

    def __init__(
        self,
        sleep_between: float = 1.0,
        max_retries: int = 3,
        timeout: float = 30.0,
        cache_enabled: bool = True,
        failed_log_path: Optional[Path] = None,
    ) -> None:
        """
        Initialize the RegInfo client.

        Args:
            sleep_between: Seconds to wait between requests to avoid rate limiting.
            max_retries: Maximum retry attempts for failed requests.
            timeout: Request timeout in seconds.
            cache_enabled: Whether to cache agenda data in memory.
            failed_log_path: Path to write failed RINs for manual review.
        """
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "FederalRegisterStudy/1.0 (Research)",
            "Accept": "application/xml, text/xml, text/html",
        })
        self.sleep_between = max(0.0, sleep_between)
        self.max_retries = max(1, max_retries)
        self.timeout = timeout
        self._cache: Dict[str, dict] = {} if cache_enabled else None
        self._failed_log_path = failed_log_path
        self._failed_rins: List[str] = []

        logger.debug(
            f"Initialized RegInfoClient with "
            f"sleep={sleep_between}s, retries={max_retries}, timeout={timeout}s"
        )

    def __enter__(self) -> "RegInfoClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        """Close the HTTP session."""
        self.session.close()

    def fetch_unified_agenda(self, rin: str) -> Optional[dict]:
        """
        Fetch Unified Agenda entry for a RIN.

        Queries reginfo.gov for the Unified Agenda data associated with
        the given Regulation Identifier Number (RIN).

        Args:
            rin: The RIN to look up (e.g., "0648-AU02" or "1234-AB56")

        Returns:
            Dict with agenda data if found:
                {
                    "rin": "0648-AU02",
                    "stage": "COMPLETED",  # Normalized stage
                    "stage_raw": "Completed Actions",  # Original stage text
                    "title": "Spinner Dolphins...",
                    "abstract": "This rule...",
                    "timetable": [
                        {"action": "ANPRM", "date": "2005-12-15", "citation": "70 FR 75183"},
                        {"action": "NPRM", "date": "2016-08-24", "citation": "81 FR 57854"},
                    ],
                    "agency": "0648",
                    "cfr_citations": "15 CFR 922",
                    "withdrawn": False,
                }
            None if RIN not found or request failed.

        Side Effects:
            - Makes HTTP request to reginfo.gov
            - Sleeps for self.sleep_between seconds before request
            - Caches result if cache_enabled
        """
        if not rin:
            return None

        # Normalize RIN format (should be XXXX-XXXX)
        rin = rin.strip().upper()
        if not re.match(r"^\d{4}-[A-Z0-9]{4}$", rin):
            logger.debug(f"Invalid RIN format: {rin}")
            return None

        # Check cache
        if self._cache is not None and rin in self._cache:
            return self._cache[rin]

        # Rate limit
        if self.sleep_between > 0:
            time.sleep(self.sleep_between)

        # Try fetching from reginfo.gov
        result = self._fetch_from_reginfo(rin)

        if not result:
            self._failed_rins.append(rin)

        # Cache result
        if result and self._cache is not None:
            self._cache[rin] = result

        return result

    def _fetch_from_reginfo(self, rin: str) -> Optional[dict]:
        """
        Fetch agenda data from reginfo.gov HTML/XML endpoint.

        The eAgendaViewRule endpoint without pubId returns a list of
        Unified Agendas that contain the RIN. We need to follow a link
        to get the actual rule data.
        """
        # First, get the list of agendas containing this RIN
        url = f"{UNIFIED_AGENDA_XML_URL}?pubId=&RIN={rin}"

        backoff = 1.0

        for attempt in range(self.max_retries):
            try:
                r = self.session.get(url, timeout=self.timeout)
            except requests.RequestException as e:
                logger.debug(f"Request failed for RIN {rin}: {type(e).__name__}")
                time.sleep(backoff)
                backoff = min(backoff * 2, 16)
                continue

            if r.status_code == 200:
                # Check if this is a listing page (contains links to specific agendas)
                # Pattern: /public/do/eAgendaViewRule?pubId=YYYYMM&RIN=...
                pub_ids = re.findall(r'pubId=(\d{6})&RIN=' + re.escape(rin), r.text)

                if pub_ids:
                    # Get the most recent agenda (highest pubId)
                    latest_pub_id = max(pub_ids)
                    logger.debug(f"Found {len(pub_ids)} agendas for RIN {rin}, using latest: {latest_pub_id}")

                    # Fetch the specific agenda entry (with retry + backoff)
                    specific_url = f"{UNIFIED_AGENDA_XML_URL}?pubId={latest_pub_id}&RIN={rin}"
                    specific_backoff = 1.0

                    for specific_attempt in range(self.max_retries):
                        time.sleep(0.5 if specific_attempt == 0 else specific_backoff)

                        try:
                            r2 = self.session.get(specific_url, timeout=self.timeout)
                        except requests.RequestException as e:
                            logger.debug(
                                f"Retry {specific_attempt + 1}/{self.max_retries} "
                                f"for specific agenda RIN {rin}: {type(e).__name__}"
                            )
                            specific_backoff = min(specific_backoff * 2, 16)
                            continue

                        if r2.status_code == 200:
                            result = self._parse_html_response(r2.text, rin)
                            if result:
                                return result
                            # 200 but no parseable data — not retriable
                            break

                        if r2.status_code in (429, 500, 502, 503, 504):
                            logger.debug(
                                f"Retry {specific_attempt + 1}/{self.max_retries} "
                                f"for specific agenda RIN {rin}: HTTP {r2.status_code}"
                            )
                            specific_backoff = min(specific_backoff * 2, 16)
                            continue

                        # Non-retriable status (e.g. 404, 403)
                        logger.debug(
                            f"Non-retriable HTTP {r2.status_code} "
                            f"for specific agenda RIN {rin}"
                        )
                        break

                # Try parsing the original page in case it has inline data
                result = self._parse_html_response(r.text, rin)
                if result:
                    return result

                # Fallback: try XML parsing
                result = self._parse_agenda_xml(r.text, rin)
                if result:
                    return result

                logger.debug(f"Could not parse response for RIN {rin}")
                return None

            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(backoff)
                backoff = min(backoff * 2, 16)
                continue

            # 404 or other client error - RIN not found
            if r.status_code == 404:
                logger.debug(f"RIN {rin} not found in Unified Agenda")
                return None

            logger.debug(f"HTTP {r.status_code} for RIN {rin}")
            return None

        return None

    def _parse_html_response(self, html: str, rin: str) -> Optional[dict]:
        """
        Parse HTML response from eAgendaViewRule endpoint.

        The HTML contains structured data we can extract using regex patterns.
        """
        result = {
            "rin": rin,
            "stage": "UNKNOWN",
            "stage_raw": "",
            "title": "",
            "abstract": "",
            "timetable": [],
            "agency": rin[:4] if len(rin) >= 4 else "",
            "cfr_citations": "",
            "withdrawn": False,
        }

        # Check if this is a "no results" page
        if "No data found" in html or "RIN not found" in html:
            return None

        # Extract stage from various possible locations
        stage_patterns = [
            r'Rule Stage[:\s]*</td>\s*<td[^>]*>([^<]+)',
            r'Stage of Rulemaking[:\s]*</td>\s*<td[^>]*>([^<]+)',
            r'class="stage"[^>]*>([^<]+)',
            r'>Stage:\s*([^<]+)<',
            # Additional patterns for reginfo.gov HTML structure
            r'>(Completed Actions)<',
            r'>(Final Rule Stage)<',
            r'>(Proposed Rule Stage)<',
            r'>(Prerule Stage)<',
            r'>(Long-Term Actions)<',
        ]

        for pattern in stage_patterns:
            match = re.search(pattern, html, re.IGNORECASE)
            if match:
                raw_stage = match.group(1).strip()
                result["stage_raw"] = raw_stage
                result["stage"] = self._normalize_stage(raw_stage)
                break

        # If no stage found yet, infer from timetable/content
        if result["stage"] == "UNKNOWN":
            # Check if this is a completed/final rule based on timetable
            html_lower = html.lower()
            if "completed actions" in html_lower:
                result["stage"] = "COMPLETED"
                result["stage_raw"] = "Completed Actions"
            elif "final rule stage" in html_lower or "final action" in html_lower:
                result["stage"] = "FINAL"
                result["stage_raw"] = "Final Rule Stage"
            elif "proposed rule stage" in html_lower:
                result["stage"] = "PROPOSED"
                result["stage_raw"] = "Proposed Rule Stage"
            elif "prerule stage" in html_lower:
                result["stage"] = "PRERULE"
                result["stage_raw"] = "Prerule Stage"
            elif "long-term actions" in html_lower or "long term actions" in html_lower:
                result["stage"] = "LONG_TERM"
                result["stage_raw"] = "Long-Term Actions"

        # Extract title
        title_patterns = [
            r'Rule Title[:\s]*</td>\s*<td[^>]*>([^<]+)',
            r'<title>([^<]+)</title>',
            r'class="title"[^>]*>([^<]+)',
        ]

        for pattern in title_patterns:
            match = re.search(pattern, html, re.IGNORECASE)
            if match:
                result["title"] = match.group(1).strip()
                break

        # Extract abstract
        abstract_match = re.search(
            r'Abstract[:\s]*</td>\s*<td[^>]*>(.*?)</td>',
            html, re.IGNORECASE | re.DOTALL
        )
        if abstract_match:
            result["abstract"] = re.sub(r'<[^>]+>', '', abstract_match.group(1)).strip()

        # Extract timetable entries from HTML table rows:
        #   <td>Action</td><td>Date</td><td>Citation</td>
        td_pattern = r"<td[^>]*>\s*([\w][^<]*?)\s*(?:&nbsp;)?\s*</td>\s*<td[^>]*>\s*(\d{1,2}/\d{1,2}/\d{4})\s*(?:&nbsp;)?\s*</td>"
        # Reference set of action types the parser previously recognized.
        # Used ONLY for logging — no filtering. All valid <td>action</td><td>date</td>
        # pairs are now accepted regardless of action name.
        _previously_known_actions = {
            "anprm", "nprm", "final rule", "final action", "interim final",
            "withdrawn", "direct final",
            "nprm comment period reopened", "nprm comment period end",
            "nprm comment period extended", "nprm comment period extended end",
            "nprm comment period reopened end",
            "nprm extended comment period end",
            "nprm comment period reopened; corrections",
            "nprm comment period reopened; corrections end",
            "second nprm", "supplemental nprm",
            "second nprm comment period end", "supplemental nprm comment period end",
            "final action effective", "final rule effective",
            "final rule; delay of effective date", "other/delayed effective date",
            "notice of availability",
            "actions will continue through",
        }
        existing_entries = set()

        for match in re.finditer(td_pattern, html, re.IGNORECASE | re.DOTALL):
            raw_action = re.sub(r'<[^>]+>', '', match.group(1)).strip()
            date_str = match.group(2).strip()

            if not raw_action:
                continue

            action_lower = raw_action.lower()
            ALL_OBSERVED_ACTIONS.add(raw_action.upper())

            # Log previously-unknown action types at DEBUG level for tracking
            if action_lower not in _previously_known_actions and not action_lower.startswith("duplicate of"):
                logger.debug(f"Timetable action (new): {raw_action!r}")

            # Handle 00-day placeholder (month-level precision from omnibus RINs)
            date_parts = date_str.split("/")
            if len(date_parts) == 3 and date_parts[1] == "00":
                date_str = f"{date_parts[0]}/01/{date_parts[2]}"

            try:
                date_obj = datetime.strptime(date_str, "%m/%d/%Y")
                date_iso = date_obj.strftime("%Y-%m-%d")
            except ValueError:
                logger.warning(f"Malformed date {date_str!r} for action {raw_action!r}, using raw string")
                date_iso = date_str

            key = (raw_action.upper(), date_iso)
            if key not in existing_entries:
                # Extract citation from the next <td> if present
                remainder = html[match.end():]
                cite_match = re.search(r'<td[^>]*>(.*?)</td>', remainder, re.DOTALL)
                citation = ""
                if cite_match:
                    cite_text = re.sub(r'<[^>]+>', '', cite_match.group(1)).strip()
                    cite_text = cite_text.replace('&nbsp;', '').strip()
                    fr_match = re.search(r'(\d+ FR \d+)', cite_text)
                    if fr_match:
                        citation = fr_match.group(1)

                existing_entries.add(key)
                result["timetable"].append({
                    "action": raw_action.upper(),
                    "date": date_iso,
                    "citation": citation,
                })

        # Check for withdrawal
        if "withdrawn" in html.lower():
            for entry in result["timetable"]:
                if "WITHDRAWN" in entry["action"].upper():
                    result["withdrawn"] = True
                    break

        # For Long-Term Actions with empty timetable, check for "Next Action
        # Undetermined" text. These RINs have real data on reginfo.gov but the
        # timetable regex can't match them because there's no parseable date.
        # Store a synthetic entry so timetable_action_count > 0.
        if result["stage"] == "LONG_TERM" and not result["timetable"]:
            if re.search(r"next\s+action\s+undetermined", html, re.IGNORECASE):
                result["timetable"].append({
                    "action": "NEXT ACTION UNDETERMINED",
                    "date": "",
                    "citation": "",
                })
                logger.debug(f"RIN {rin}: Long-Term Actions with 'Next Action Undetermined' — stored synthetic entry")
            elif re.search(r"to\s+be\s+determined", html, re.IGNORECASE):
                result["timetable"].append({
                    "action": "NEXT ACTION UNDETERMINED",
                    "date": "",
                    "citation": "",
                })
                logger.debug(f"RIN {rin}: Long-Term Actions with 'To Be Determined' — stored synthetic entry")

        # Extract CFR citations
        cfr_match = re.search(r'CFR Citation[:\s]*</td>\s*<td[^>]*>([^<]+)', html, re.IGNORECASE)
        if cfr_match:
            result["cfr_citations"] = cfr_match.group(1).strip()

        # Only return if we found meaningful data
        if result["stage"] != "UNKNOWN" or result["timetable"]:
            return result

        return None

    def _parse_agenda_xml(self, xml_text: str, rin: str) -> Optional[dict]:
        """
        Parse XML response from Unified Agenda endpoint.

        Args:
            xml_text: Raw XML text from API response
            rin: The RIN being looked up

        Returns:
            Parsed agenda dict or None if parsing failed.
        """
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return None

        result = {
            "rin": rin,
            "stage": "UNKNOWN",
            "stage_raw": "",
            "title": "",
            "abstract": "",
            "timetable": [],
            "agency": "",
            "cfr_citations": "",
            "withdrawn": False,
        }

        # Try various XML element paths
        # Path 1: Standard Unified Agenda XML structure
        stage = root.findtext(".//RULE_STAGE") or root.findtext(".//RuleStage")
        if stage:
            result["stage_raw"] = stage
            result["stage"] = self._normalize_stage(stage)

        result["title"] = root.findtext(".//TITLE") or root.findtext(".//Title") or ""
        result["abstract"] = root.findtext(".//ABSTRACT") or root.findtext(".//Abstract") or ""
        result["agency"] = root.findtext(".//AGENCY_CODE") or root.findtext(".//AgencyCode") or rin[:4]
        result["cfr_citations"] = root.findtext(".//CFR") or root.findtext(".//CFRCitation") or ""

        # Parse timetable entries
        for entry in root.findall(".//TIMETABLE_ENTRY") + root.findall(".//TimetableEntry"):
            action = entry.findtext("ACTION") or entry.findtext("Action") or ""
            date = entry.findtext("FR_DATE") or entry.findtext("Date") or entry.findtext("DATE") or ""
            citation = entry.findtext("FR_CITE") or entry.findtext("Citation") or ""

            if action or date:
                result["timetable"].append({
                    "action": action.upper(),
                    "date": date,
                    "citation": citation,
                })

                if "WITHDRAWN" in action.upper():
                    result["withdrawn"] = True

        return result if result["stage"] != "UNKNOWN" or result["timetable"] else None

    def _normalize_stage(self, raw_stage: str) -> str:
        """
        Normalize stage string to enum value.

        Args:
            raw_stage: Raw stage text from API response

        Returns:
            Normalized stage: PRERULE, PROPOSED, FINAL, LONG_TERM, COMPLETED, or UNKNOWN
        """
        if not raw_stage:
            return "UNKNOWN"

        raw_stage = raw_stage.strip()

        # Direct mapping
        if raw_stage in STAGE_MAP:
            return STAGE_MAP[raw_stage]

        # Case-insensitive search
        raw_lower = raw_stage.lower()
        for key, value in STAGE_MAP.items():
            if key.lower() == raw_lower:
                return value

        # Partial matching
        if "prerule" in raw_lower:
            return "PRERULE"
        if "proposed" in raw_lower:
            return "PROPOSED"
        if "final" in raw_lower:
            return "FINAL"
        if "long-term" in raw_lower or "long term" in raw_lower:
            return "LONG_TERM"
        if "completed" in raw_lower:
            return "COMPLETED"

        return "UNKNOWN"

    def determine_lifecycle_stage(
        self,
        agenda_data: Optional[dict],
        doc_type: str,
        has_open_comments: bool,
        effective_date: Optional[str] = None,
    ) -> str:
        """
        Determine specific lifecycle stage from agenda + document metadata.

        This method combines Unified Agenda stage information with Federal Register
        document metadata to assign a precise lifecycle stage.

        Args:
            agenda_data: Dict from fetch_unified_agenda(), or None
            doc_type: FR document type (e.g., "Proposed Rule", "Rule", "Notice")
            has_open_comments: Whether the comment period is currently open
            effective_date: Optional effective date string (ISO format)

        Returns:
            One of the LIFECYCLE_STAGES values:
            - PRERULE_ANPRM, PRERULE_RFI (pre-proposal stages)
            - NPRM_ACTIVE, NPRM_CLOSED (proposed rule stages)
            - INTERIM_FINAL, FINAL_PENDING, FINAL_EFFECTIVE (final stages)
            - WITHDRAWN, LONG_TERM_STALLED (terminal/stalled)
            - NO_RIN, AGENDA_NOT_FOUND, UNKNOWN (error cases)

        Example:
            >>> stage = client.determine_lifecycle_stage(
            ...     agenda_data={"stage": "PROPOSED", "timetable": [...]},
            ...     doc_type="Proposed Rule",
            ...     has_open_comments=True,
            ... )
            >>> print(stage)  # "NPRM_ACTIVE"
        """
        # Handle missing data cases
        if agenda_data is None:
            return "AGENDA_NOT_FOUND"

        stage = agenda_data.get("stage", "UNKNOWN")
        timetable = agenda_data.get("timetable", [])
        withdrawn = agenda_data.get("withdrawn", False)

        doc_type_lower = (doc_type or "").lower()

        # Check for withdrawal
        if withdrawn or stage == "COMPLETED":
            # Check timetable for withdrawal vs completion
            for entry in timetable:
                if "withdrawn" in entry.get("action", "").lower():
                    return "WITHDRAWN"
            # If COMPLETED but not withdrawn, it's finalized
            if stage == "COMPLETED":
                return "FINAL_EFFECTIVE"
            # withdrawn=True but no timetable entry matched — still treat as withdrawn
            if withdrawn:
                return "WITHDRAWN"

        # Pre-rule stages
        if stage == "PRERULE":
            if "advance" in doc_type_lower or "anprm" in doc_type_lower:
                return "PRERULE_ANPRM"
            if "request for" in doc_type_lower or "rfi" in doc_type_lower:
                return "PRERULE_RFI"
            return "PRERULE_ANPRM"  # Default pre-rule type

        # Proposed rule stages
        if stage == "PROPOSED":
            if has_open_comments:
                return "NPRM_ACTIVE"
            return "NPRM_CLOSED"

        # Final rule stages
        if stage == "FINAL":
            # Check for interim final
            if "interim" in doc_type_lower:
                return "INTERIM_FINAL"

            # Check effective date
            if effective_date:
                try:
                    eff_date = datetime.fromisoformat(effective_date.replace("Z", "+00:00"))
                    if eff_date > datetime.now(eff_date.tzinfo) if eff_date.tzinfo else datetime.now():
                        return "FINAL_PENDING"
                except ValueError:
                    pass

            return "FINAL_EFFECTIVE"

        # Long-term stalled
        if stage == "LONG_TERM":
            return "LONG_TERM_STALLED"

        # Try to infer from document type if stage is unknown
        if stage == "UNKNOWN":
            if "proposed" in doc_type_lower:
                return "NPRM_ACTIVE" if has_open_comments else "NPRM_CLOSED"
            if "rule" in doc_type_lower and "proposed" not in doc_type_lower:
                if "interim" in doc_type_lower:
                    return "INTERIM_FINAL"
                return "FINAL_EFFECTIVE"
            if "advance" in doc_type_lower or "notice" in doc_type_lower:
                return "PRERULE_ANPRM"

        return "UNKNOWN"

    def get_timetable_summary(self, agenda_data: Optional[dict]) -> List[Tuple[str, str, str]]:
        """
        Extract a summary of the rulemaking timetable.

        Args:
            agenda_data: Dict from fetch_unified_agenda()

        Returns:
            List of (action, date, citation) tuples in chronological order.
        """
        if not agenda_data:
            return []

        timetable = agenda_data.get("timetable", [])

        # Sort by date
        sorted_entries = sorted(
            timetable,
            key=lambda x: x.get("date", "9999-99-99")
        )

        return [
            (entry.get("action", ""), entry.get("date", ""), entry.get("citation", ""))
            for entry in sorted_entries
        ]

    # Map verbose timetable actions to condensed sequence tokens
    _ACTION_CONDENSED = {
        "ANPRM": "ANPRM",
        "NPRM": "NPRM",
        "SECOND NPRM": "SECOND_NPRM",
        "SUPPLEMENTAL NPRM": "SUPPLEMENTAL_NPRM",
        "NPRM COMMENT PERIOD REOPENED": "NPRM_REOPENED",
        "NPRM COMMENT PERIOD END": "NPRM_CLOSED",
        "NPRM COMMENT PERIOD EXTENDED": "NPRM_EXTENDED",
        "NPRM COMMENT PERIOD EXTENDED END": "NPRM_EXTENDED_END",
        "NPRM COMMENT PERIOD REOPENED END": "NPRM_REOPENED_END",
        "NPRM EXTENDED COMMENT PERIOD END": "NPRM_EXTENDED_END",
        "NPRM COMMENT PERIOD REOPENED; CORRECTIONS": "NPRM_REOPENED_CORRECTIONS",
        "NPRM COMMENT PERIOD REOPENED; CORRECTIONS END": "NPRM_REOPENED_CORRECTIONS_END",
        "SECOND NPRM COMMENT PERIOD END": "SECOND_NPRM_CLOSED",
        "SUPPLEMENTAL NPRM COMMENT PERIOD END": "SUPPLEMENTAL_NPRM_CLOSED",
        "NOTICE OF AVAILABILITY": "NOA",
        "FINAL ACTION": "FINAL",
        "FINAL RULE": "FINAL",
        "FINAL ACTION EFFECTIVE": "FINAL_EFFECTIVE",
        "FINAL RULE EFFECTIVE": "FINAL_EFFECTIVE",
        "FINAL RULE; DELAY OF EFFECTIVE DATE": "FINAL_DELAYED",
        "OTHER/DELAYED EFFECTIVE DATE": "FINAL_DELAYED",
        "INTERIM FINAL": "INTERIM_FINAL",
        "DIRECT FINAL": "DIRECT_FINAL",
        "WITHDRAWN": "WITHDRAWN",
        "NEXT ACTION UNDETERMINED": "UNDETERMINED",
        "ACTIONS WILL CONTINUE THROUGH": "CONTINUING",
    }

    @staticmethod
    def _empty_timeline_columns() -> dict:
        """Return a dict of structured timeline columns with empty defaults."""
        return {
            "nprm_date": None,
            "nprm_citation": None,
            "final_action_date": None,
            "final_action_citation": None,
            "effective_date": None,
            "withdrawal_date": None,
            "nprm_to_final_days": None,
            "timetable_action_count": 0,
            "has_reopened_comment": False,
            "has_extended_comment": False,
            "anprm_date": None,
            "anprm_citation": None,
            "interim_final_date": None,
            "interim_final_citation": None,
            "direct_final_date": None,
            "comment_reopened_date": None,
            "comment_extended_date": None,
            "anprm_to_nprm_days": None,
            "anprm_to_final_days": None,
            "timetable_sequence": None,
            "timeline_confidence": "none",
        }

    @staticmethod
    def _safe_days_between(date_a: Optional[str], date_b: Optional[str]) -> Optional[int]:
        """Compute (date_b - date_a) in days, or None if either is missing/malformed.

        Skips computation if:
        - date_b is in the future (projected timetable entry, not an actual event)
        - date_b < date_a (reversed-order events, e.g. Direct Final before NPRM)
        """
        if not date_a or not date_b:
            return None
        try:
            dt_a = datetime.fromisoformat(date_a)
            dt_b = datetime.fromisoformat(date_b)
            if dt_b > datetime.now():
                logger.debug(
                    f"Skipping duration: end date {date_b} is in the future (projected)"
                )
                return None
            days = (dt_b - dt_a).days
            if days < 0:
                logger.debug(
                    f"Skipping duration: negative result ({days} days) "
                    f"suggests reversed event order ({date_a} -> {date_b})"
                )
                return None
            return days
        except ValueError:
            return None

    @staticmethod
    def extract_structured_timeline(timetable: List[dict]) -> dict:
        """
        Extract structured timeline columns from a timetable action list.

        Given a list of timetable entries (each with action, date, citation),
        extracts key milestones into flat columns suitable for CSV output.

        Args:
            timetable: List of dicts with keys "action", "date", "citation".
                       Expected to be the value from agenda["timetable"].

        Returns:
            Dict with milestone dates/citations, duration metrics,
            and a condensed timetable_sequence string.
        """
        result = RegInfoClient._empty_timeline_columns()

        if not timetable:
            return result

        # Count only entries with a parseable date — placeholders like
        # "NEXT ACTION UNDETERMINED" (no date) should not inflate the count.
        result["timetable_action_count"] = sum(
            1 for e in timetable if e.get("date", "").strip()
        )

        # Sort chronologically (same pattern as get_timetable_summary)
        sorted_entries = sorted(
            timetable,
            key=lambda x: x.get("date", "9999-99-99")
        )

        # Track candidates for each milestone
        anprm_entry = None         # First ANPRM (earliest)
        nprm_entry = None          # First NPRM (earliest, excludes Second/Supplemental)
        final_entry = None         # Latest FINAL-type action (excludes effective dates)
        final_citation_backup = None  # Citation from any FINAL entry (fallback)
        effective_entry = None     # Latest effective date action
        interim_final_entry = None # First INTERIM FINAL
        direct_final_entry = None  # First DIRECT FINAL
        withdrawal_entry = None    # First WITHDRAWN
        reopened_entry = None      # First REOPENED
        extended_entry = None      # First EXTENDED

        # Actions that represent effective dates, NOT final action dates
        _EFFECTIVE_DATE_ACTIONS = {
            "FINAL ACTION EFFECTIVE", "FINAL RULE EFFECTIVE",
            "FINAL RULE; DELAY OF EFFECTIVE DATE", "OTHER/DELAYED EFFECTIVE DATE",
        }

        sequence_tokens = []

        for entry in sorted_entries:
            action = entry.get("action", "")
            action_upper = action.upper()
            date_str = entry.get("date", "")
            citation = entry.get("citation", "") or None  # "" -> None

            # Build sequence token; for "DUPLICATE OF ..." use a generic token
            if action_upper.startswith("DUPLICATE OF"):
                token = "DUPLICATE"
            else:
                token = RegInfoClient._ACTION_CONDENSED.get(action_upper, action_upper)
            sequence_tokens.append(token)

            # ANPRM
            if action_upper == "ANPRM":
                if anprm_entry is None:
                    anprm_entry = {"date": date_str, "citation": citation}

            # NPRM: exactly "NPRM" only (not Second/Supplemental, not comment period variants)
            if action_upper == "NPRM":
                if nprm_entry is None:  # Take earliest
                    nprm_entry = {"date": date_str, "citation": citation}

            # EFFECTIVE DATE actions: track separately, do NOT overwrite final_entry
            if action_upper in _EFFECTIVE_DATE_ACTIONS:
                # Always overwrite -> keeps latest by date (entries are sorted)
                effective_entry = {"date": date_str, "citation": citation}
            # FINAL: "Final Action", "Final Rule", "Interim Final", "Direct Final"
            # but NOT effective date variants
            elif "FINAL" in action_upper:
                # Always overwrite -> keeps latest by date (entries are sorted)
                final_entry = {"date": date_str, "citation": citation}
                if citation:
                    final_citation_backup = citation

            # INTERIM FINAL (specifically)
            if "INTERIM" in action_upper and "FINAL" in action_upper:
                if interim_final_entry is None:
                    interim_final_entry = {"date": date_str, "citation": citation}

            # DIRECT FINAL
            if "DIRECT" in action_upper and "FINAL" in action_upper:
                if direct_final_entry is None:
                    direct_final_entry = {"date": date_str}

            # WITHDRAWN
            if "WITHDRAWN" in action_upper:
                if withdrawal_entry is None:
                    withdrawal_entry = {"date": date_str}

            # REOPENED
            if "REOPENED" in action_upper:
                result["has_reopened_comment"] = True
                if reopened_entry is None:
                    reopened_entry = {"date": date_str}

            # EXTENDED comment period
            if "EXTENDED" in action_upper and "COMMENT" in action_upper:
                result["has_extended_comment"] = True
                if extended_entry is None:
                    extended_entry = {"date": date_str}

        # --- Extract milestone fields ---

        if anprm_entry:
            result["anprm_date"] = anprm_entry["date"] or None
            result["anprm_citation"] = anprm_entry["citation"]

        if nprm_entry:
            result["nprm_date"] = nprm_entry["date"] or None
            result["nprm_citation"] = nprm_entry["citation"]

        if final_entry:
            result["final_action_date"] = final_entry["date"] or None
            result["final_action_citation"] = final_entry["citation"] or final_citation_backup
        elif effective_entry:
            # No explicit Final Action/Rule, but we have an effective date.
            # Use it as final_action_date (better than nothing).
            result["final_action_date"] = effective_entry["date"] or None
            result["final_action_citation"] = effective_entry["citation"]

        if effective_entry:
            result["effective_date"] = effective_entry["date"] or None

        if interim_final_entry:
            result["interim_final_date"] = interim_final_entry["date"] or None
            result["interim_final_citation"] = interim_final_entry["citation"]

        if direct_final_entry:
            result["direct_final_date"] = direct_final_entry["date"] or None

        if withdrawal_entry:
            result["withdrawal_date"] = withdrawal_entry["date"] or None

        if reopened_entry:
            result["comment_reopened_date"] = reopened_entry["date"] or None

        if extended_entry:
            result["comment_extended_date"] = extended_entry["date"] or None

        # --- Compute durations ---

        result["nprm_to_final_days"] = RegInfoClient._safe_days_between(
            result["nprm_date"], result["final_action_date"]
        )
        result["anprm_to_nprm_days"] = RegInfoClient._safe_days_between(
            result["anprm_date"], result["nprm_date"]
        )
        result["anprm_to_final_days"] = RegInfoClient._safe_days_between(
            result["anprm_date"], result["final_action_date"]
        )

        if (result["nprm_date"] and result["final_action_date"]
                and result["nprm_to_final_days"] is None):
            logger.warning(
                f"Malformed date in timetable — cannot compute durations: "
                f"nprm_date={result['nprm_date']!r}, "
                f"final_action_date={result['final_action_date']!r}"
            )

        # --- Build timetable_sequence ---
        if sequence_tokens:
            result["timetable_sequence"] = "->".join(sequence_tokens)

        # --- Derive timeline_confidence ---
        # "full": both NPRM and Final dates present
        # "partial": at least one milestone date (nprm, final, or withdrawal)
        # "label_only": timetable was provided but no milestone dates extracted
        has_nprm = bool(result["nprm_date"])
        has_final = bool(result["final_action_date"])
        has_withdrawal = bool(result["withdrawal_date"])

        if has_nprm and (has_final or has_withdrawal):
            result["timeline_confidence"] = "full"
        elif has_nprm or has_final or has_withdrawal:
            result["timeline_confidence"] = "partial"
        elif result["timetable_action_count"] > 0:
            result["timeline_confidence"] = "partial"
        else:
            # Timetable was provided but empty/placeholder-only.
            # Caller (distribution.py) upgrades to "label_only" when
            # lifecycle_stage is set from UA stage metadata.
            result["timeline_confidence"] = "label_only"

        return result

    def clear_cache(self) -> None:
        """Clear the in-memory agenda cache."""
        if self._cache is not None:
            self._cache.clear()
            logger.debug("RegInfo cache cleared")

    def flush_failed_rins(self) -> None:
        """Write accumulated failed RINs to log file, then clear the list."""
        if not self._failed_rins:
            return

        if self._failed_log_path is None:
            logger.warning(
                f"{len(self._failed_rins)} RINs failed but no failed_log_path configured"
            )
            return

        self._failed_log_path.parent.mkdir(parents=True, exist_ok=True)

        # Deduplicate while preserving order
        seen = set()
        unique_rins = []
        for rin in self._failed_rins:
            if rin not in seen:
                seen.add(rin)
                unique_rins.append(rin)

        # Write mode (not append) — each run produces a clean file
        from datetime import datetime as _dt
        with open(self._failed_log_path, "w", encoding="utf-8") as f:
            f.write(f"# Failed RINs — {_dt.now().isoformat()}\n")
            for rin in unique_rins:
                f.write(rin + "\n")

        logger.info(
            f"Wrote {len(unique_rins)} unique failed RINs "
            f"(from {len(self._failed_rins)} total) to {self._failed_log_path}"
        )
        self._failed_rins.clear()
