from unittest.mock import MagicMock

from stratification_scripts.federal_register.client import FederalRegisterClient


def test_fetch_full_text_xml_uses_details_url(monkeypatch):
    c = FederalRegisterClient(max_retries=1, sleep_between=0)
    monkeypatch.setattr(c, "fetch_document_details",
                        lambda dn, enrich_identifiers=False: {"full_text_xml_url": "http://x/doc.xml"})
    resp = MagicMock(status_code=200, text="<RULE><SUPLINF>hi</SUPLINF></RULE>")
    monkeypatch.setattr(c.session, "get", lambda url, timeout=60: resp)
    assert "<SUPLINF>" in c.fetch_document_full_text_xml("2024-19696")


def test_fetch_full_text_xml_missing_url_returns_none(monkeypatch):
    c = FederalRegisterClient(max_retries=1, sleep_between=0)
    monkeypatch.setattr(c, "fetch_document_details", lambda dn, enrich_identifiers=False: {})
    assert c.fetch_document_full_text_xml("2024-19696") is None
