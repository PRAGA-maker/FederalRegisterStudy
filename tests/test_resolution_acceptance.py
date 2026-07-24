"""The six observed topologies, replayed offline from recorded fixtures."""

import gzip
import json
import re
from pathlib import Path

import pytest

from stratification_scripts.resolution.models import CommentRef, Status
from stratification_scripts.resolution.resolver import (
    DocumentResolver, qualifying_candidates,
)

FIXTURES = Path(__file__).parent / "fixtures" / "resolution"
ROWS = [
    "NOAA-NMFS-2023-0125-0016",
    "BLM-2024-0001-0003",
    "FBI-2024-0002-0006",
    "EPA-HQ-OAR-2022-0491-0022",
    "DOT-OST-2024-0090-0049",
    "CMS-2024-0131-6043",
]


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


class ReplayFRClient:
    def __init__(self, directory: Path) -> None:
        self.dir = directory

    def _load(self, name: str):
        path = self.dir / name
        if not path.exists():
            raise AssertionError(f"unrecorded request: {name}")
        return json.loads(path.read_text())

    def fetch_document_details(self, document_number, enrich_identifiers=True):
        path = self.dir / f"fr_doc_{_slug(document_number)}.json"
        return json.loads(path.read_text()) if path.exists() else None

    def fetch_document_full_text_xml(self, document_number):
        path = self.dir / f"xml_{_slug(document_number)}.xml.gz"
        return gzip.decompress(path.read_bytes()).decode() if path.exists() else None

    def search_by_rin(self, rin):
        return self._load(f"fr_rin_{_slug(rin)}.json")

    def search_by_docket(self, docket_id):
        return self._load(f"fr_docket_{_slug(docket_id or '')}.json")

    def search_full_text(self, identifier):
        return self._load(f"fr_term_{_slug(identifier)}.json")


class ReplayRegInfoClient:
    def __init__(self, directory: Path) -> None:
        self.dir = directory

    def fetch_unified_agenda(self, rin):
        path = self.dir / f"agenda_{_slug(rin)}.json"
        return json.loads(path.read_text()) if path.exists() else None


def _ref(directory: Path) -> CommentRef:
    data = json.loads((directory / "input.json").read_text())
    return CommentRef(
        comment_id=data["comment_id"], comment_date=data["comment_date"],
        source_document=data["source_document"], agency=data["agency"],
        rins=tuple(data["rins"]), docket_id=data["docket_id"],
        packet_final_document=data["packet_final_document"],
    )


@pytest.mark.parametrize("comment_id", ROWS)
def test_topology_fixture(comment_id):
    directory = FIXTURES / comment_id
    expected = json.loads((directory / "expected.json").read_text())
    resolver = DocumentResolver(
        fr_client=ReplayFRClient(directory),
        reginfo_client=ReplayRegInfoClient(directory),
    )
    result = resolver.resolve(_ref(directory))

    assert result.status.value == expected["status"], result.to_dict()
    actual_reason = result.absence_reason.value if result.absence_reason else None
    assert actual_reason == expected["absence_reason"], result.to_dict()
    assert [c.document_number for c in qualifying_candidates(result)] == \
        expected["qualifying_document_numbers"], result.to_dict()

    by_number = {c.document_number: c.to_dict() for c in result.candidates}
    for assertion in expected["candidate_assertions"]:
        number = assertion["document_number"]
        assert number in by_number, f"{number} missing from candidates"
        for key, value in assertion.items():
            if key == "document_number":
                continue
            assert by_number[number][key] == value, (number, key, by_number[number])


def test_dot_row_is_the_header_flag_regression():
    directory = FIXTURES / "DOT-OST-2024-0090-0049"
    resolver = DocumentResolver(
        fr_client=ReplayFRClient(directory),
        reginfo_client=ReplayRegInfoClient(directory),
    )
    result = resolver.resolve(_ref(directory))
    final = [c for c in result.candidates if c.document_number == "2024-29990"][0]
    assert final.response_header_matched is False
    assert result.status is Status.FOUND


def test_no_fixture_directory_is_missing():
    assert sorted(p.name for p in FIXTURES.iterdir() if p.is_dir()) == sorted(ROWS)
