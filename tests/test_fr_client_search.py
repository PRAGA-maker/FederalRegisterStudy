from unittest.mock import MagicMock

from stratification_scripts.federal_register.client import (
    SEARCH_FIELDS, FederalRegisterClient,
)


def _client():
    return FederalRegisterClient(max_retries=1, sleep_between=0)


def _ok(payload):
    response = MagicMock(status_code=200)
    response.json.return_value = payload
    return response


def test_search_by_rin_sends_the_rin_condition_and_fields(monkeypatch):
    captured = {}

    def fake_get(url, params=None, timeout=30):
        captured["url"] = url
        captured["params"] = params
        return _ok({"count": 1, "results": [{"document_number": "2024-27333",
                                             "action": "Direct final rule."}]})

    client = _client()
    monkeypatch.setattr(client.session, "get", fake_get)
    docs = client.search_by_rin("1004-AF01")
    assert docs == [{"document_number": "2024-27333", "action": "Direct final rule."}]
    assert captured["params"]["conditions[regulation_id_number]"] == "1004-AF01"
    assert captured["params"]["fields[]"] == SEARCH_FIELDS


def test_search_by_docket_zero_results_is_empty_list_not_none(monkeypatch):
    client = _client()
    monkeypatch.setattr(client.session, "get",
                        lambda url, params=None, timeout=30: _ok({"count": 0}))
    assert client.search_by_docket("NOAA-NMFS-2023-0125") == []


def test_search_full_text_quotes_the_identifier(monkeypatch):
    captured = {}

    def fake_get(url, params=None, timeout=30):
        captured["params"] = params
        return _ok({"count": 0})

    client = _client()
    monkeypatch.setattr(client.session, "get", fake_get)
    client.search_full_text("1808-IFC")
    assert captured["params"]["conditions[term]"] == '"1808-IFC"'


def test_search_returns_none_on_persistent_http_error(monkeypatch):
    client = _client()
    monkeypatch.setattr(client.session, "get",
                        lambda url, params=None, timeout=30: MagicMock(status_code=503))
    assert client.search_by_rin("1004-AF01") is None


def test_search_returns_none_on_request_exception(monkeypatch):
    import requests

    def boom(url, params=None, timeout=30):
        raise requests.RequestException("network down")

    client = _client()
    monkeypatch.setattr(client.session, "get", boom)
    assert client.search_full_text("1808-IFC") is None


def test_blank_query_short_circuits_to_empty(monkeypatch):
    client = _client()
    monkeypatch.setattr(client.session, "get",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no request")))
    assert client.search_by_rin("") == []
    assert client.search_by_docket(None) == []
    assert client.search_full_text("  ") == []
