from stratification_scripts.resolution.cache import DocumentCache


class FakeFR:
    def __init__(self):
        self.detail_calls = []
        self.xml_calls = []

    def fetch_document_details(self, document_number, enrich_identifiers=True):
        self.detail_calls.append(document_number)
        return {"document_number": document_number, "action": "Final rule."}

    def fetch_document_full_text_xml(self, document_number):
        self.xml_calls.append(document_number)
        return "<RULE><SUPLINF>Comment: x Response: we agree</SUPLINF></RULE>"


def test_details_fetched_once_per_document():
    fr = FakeFR()
    cache = DocumentCache(fr)
    assert cache.details("2024-15931")["action"] == "Final rule."
    assert cache.details("2024-15931")["action"] == "Final rule."
    assert fr.detail_calls == ["2024-15931"]
    assert cache.stats["details_hits"] == 1


def test_xml_and_extract_share_one_fetch():
    fr = FakeFR()
    cache = DocumentCache(fr)
    assert "SUPLINF" in cache.xml("2024-29990")
    extract = cache.extract("2024-29990")
    assert extract is not None and extract.grounded_text
    cache.extract("2024-29990")
    assert fr.xml_calls == ["2024-29990"]


def test_failed_fetch_is_cached_as_a_miss_not_retried():
    class FailingFR(FakeFR):
        def fetch_document_full_text_xml(self, document_number):
            self.xml_calls.append(document_number)
            return None

    fr = FailingFR()
    cache = DocumentCache(fr)
    assert cache.xml("2024-00000") is None
    assert cache.xml("2024-00000") is None
    assert fr.xml_calls == ["2024-00000"]


def test_blank_document_number_never_hits_the_client():
    fr = FakeFR()
    cache = DocumentCache(fr)
    assert cache.details("") is None
    assert cache.xml(None) is None
    assert fr.detail_calls == [] and fr.xml_calls == []
