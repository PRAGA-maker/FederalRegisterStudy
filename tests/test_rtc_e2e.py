"""Real-data tests for the RTC parser.

`test_real_slice_*` run against a committed text slice of the actual CCL5 PDF
(deterministic, no binary, no network). `test_full_pdf_*` run the whole parse
against the local gitignored PDF and skip when it is absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from stratification_scripts import config
from stratification_scripts.rtc_parser import cli, crosswalk, exhibit2, responses, topics

_SLICE = Path(__file__).parent / "fixtures" / "rtc_ccl5_slice.txt"
_FULL_PDF = config.get_rtc_inputs_dir() / "ccl5_rtc.pdf"


def _parse_slice():
    text = _SLICE.read_text()
    commenters = exhibit2.parse_commenters(text)
    sec2 = text.index("2. Comments and EPA Responses by Topic", text.index("Exhibit 2"))
    body = text[sec2:]
    blocks = responses.split_comment_blocks(body)
    discussions = topics.split_topic_discussions(body)
    return commenters, crosswalk.assemble(commenters, blocks, discussions), discussions


def test_real_slice_parses_expected_structure():
    commenters, records, discussions = _parse_slice()
    assert len(commenters) == 54
    assert list(discussions) == ["General Comments"]
    assert {r.commenter_number for r in records} == {52, 54, 58, 59}


def test_real_slice_joins_document_id_topic_and_discussion():
    _, records, _ = _parse_slice()
    r52 = next(r for r in records if r.commenter_number == 52)
    assert r52.document_id == "EPA-HQ-OW-2018-0594-0052"
    assert r52.first_name == "Alicia" and r52.last_name == "Johnston"
    assert [t.canonical for t in r52.topic_refs] == ["General Comments"]
    assert r52.topic_discussions["General Comments"].startswith("EPA received many general comments")


@pytest.mark.skipif(not _FULL_PDF.exists(), reason="full CCL5 PDF not present in rtc/inputs/")
def test_full_pdf_parses_end_to_end():
    commenters, records, discussions, page_count = cli.parse_pdf(_FULL_PDF)
    assert page_count == 159
    assert len(commenters) == 54
    assert len(records) == 114  # 114 excerpts across 52 distinct commenter numbers
    assert len({r.commenter_number for r in records}) == 52
    assert len(discussions) == 22
    # exactly one orphan excerpt (the 114-vs-113 off-by-one)
    assert sum(not r.has_individual_response for r in records) == 1
    # the vast majority of cross-refs resolve; a handful are honestly flagged
    total = sum(len(r.topic_refs) for r in records)
    unresolved = sum(1 for r in records for t in r.topic_refs if not t.resolved)
    assert total > 200 and unresolved < 15
