"""Cross-row document cache."""

from __future__ import annotations

from typing import Dict, Optional

from ..makeup.fr_response_extractor import ResponseExtract, extract_response_section

_MISS = object()


class DocumentCache:
    """Memoize FR document details, full-text XML, and response extracts."""

    def __init__(self, fr_client) -> None:
        self._fr = fr_client
        self._details: Dict[str, object] = {}
        self._xml: Dict[str, object] = {}
        self._extract: Dict[str, object] = {}
        self._stats = {
            "details_hits": 0, "details_misses": 0,
            "xml_hits": 0, "xml_misses": 0,
        }

    @property
    def stats(self) -> dict:
        return dict(self._stats)

    def details(self, document_number: Optional[str]) -> Optional[dict]:
        key = (document_number or "").strip()
        if not key:
            return None
        if key in self._details:
            self._stats["details_hits"] += 1
            value = self._details[key]
            return None if value is _MISS else value
        self._stats["details_misses"] += 1
        fetched = self._fr.fetch_document_details(key, enrich_identifiers=True)
        self._details[key] = fetched if fetched else _MISS
        return fetched or None

    def xml(self, document_number: Optional[str]) -> Optional[str]:
        key = (document_number or "").strip()
        if not key:
            return None
        if key in self._xml:
            self._stats["xml_hits"] += 1
            value = self._xml[key]
            return None if value is _MISS else value
        self._stats["xml_misses"] += 1
        fetched = self._fr.fetch_document_full_text_xml(key)
        self._xml[key] = fetched if fetched else _MISS
        return fetched or None

    def extract(self, document_number: Optional[str]) -> Optional[ResponseExtract]:
        key = (document_number or "").strip()
        if not key:
            return None
        if key not in self._extract:
            xml = self.xml(key)
            self._extract[key] = extract_response_section(xml) if xml else _MISS
        value = self._extract[key]
        return None if value is _MISS else value
