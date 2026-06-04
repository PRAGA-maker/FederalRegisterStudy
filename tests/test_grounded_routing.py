import polars as pl

import stratification_scripts.makeup.track_responses as tr
from stratification_scripts.makeup.track_responses import build_grounded_cache

_XML = ('<SUPLINF><HD SOURCE="HD1">Response to Comments</HD><P>'
        + ("x " * 2000) + 'we agree with the commenter.</P></SUPLINF>')


def test_build_grounded_cache_maps_doc_to_final_rule(monkeypatch):
    fr = pl.DataFrame({"document_number": ["2024-NPRM"],
                       "final_rule_document_number": ["2024-FINAL"]})
    monkeypatch.setattr(tr, "_fetch_final_rule_xml", lambda dn: _XML)
    cache = build_grounded_cache(["2024-NPRM"], fr, grounded_max_chars=100000)
    assert "2024-NPRM" in cache
    assert "we agree with the commenter" in cache["2024-NPRM"].grounded_text


def test_doc_that_is_itself_a_rule_is_grounded(monkeypatch):
    fr = pl.DataFrame({"document_number": ["2024-RULE"],
                       "final_rule_document_number": [None],
                       "doc_type": ["Rule"]})
    captured = {}
    def fake(dn):
        captured["dn"] = dn
        return _XML
    monkeypatch.setattr(tr, "_fetch_final_rule_xml", fake)
    cache = build_grounded_cache(["2024-RULE"], fr, grounded_max_chars=100000)
    assert captured["dn"] == "2024-RULE"      # fetched its own doc as the final rule
    assert "2024-RULE" in cache


def test_unlinked_doc_is_absent_from_cache(monkeypatch):
    fr = pl.DataFrame({"document_number": ["2024-NOTICE"],
                       "final_rule_document_number": [None],
                       "doc_type": ["Notice"]})
    monkeypatch.setattr(tr, "_fetch_final_rule_xml", lambda dn: _XML)
    cache = build_grounded_cache(["2024-NOTICE"], fr, grounded_max_chars=100000)
    assert cache == {}


def test_failed_fetch_skips_gracefully(monkeypatch):
    fr = pl.DataFrame({"document_number": ["2024-NPRM"],
                       "final_rule_document_number": ["2024-FINAL"]})
    monkeypatch.setattr(tr, "_fetch_final_rule_xml", lambda dn: None)
    assert build_grounded_cache(["2024-NPRM"], fr) == {}
