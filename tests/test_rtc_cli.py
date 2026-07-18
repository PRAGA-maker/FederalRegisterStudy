from __future__ import annotations

import json

import pytest

from stratification_scripts import config
from stratification_scripts.rtc_parser import cli
from stratification_scripts.rtc_parser.models import Commenter, CommentRecord, TopicRef


def test_rtc_paths_are_project_siblings():
    root = config.get_project_root()
    assert config.get_rtc_dir() == root / "rtc"
    assert config.get_rtc_output_path("ccl5") == root / "rtc" / "ccl5"
    assert config.get_rtc_inputs_dir() == root / "rtc" / "inputs"


def _fixture_outputs():
    commenters = [Commenter(52, "EPA-HQ-OW-2018-0594-0052", "A", "B", "Org")]
    records = [
        CommentRecord(
            commenter_number=52,
            document_id="EPA-HQ-OW-2018-0594-0052",
            first_name="A",
            last_name="B",
            organization="Org",
            comment_excerpt="excerpt",
            has_individual_response=True,
            topic_refs=[
                TopicRef("General Comments", "General Comments", True),
                TopicRef("1,4-Dioxane", None, False),
            ],
            individual_response_supplemental="supp",
            topic_discussions={"General Comments": "GC text"},
        )
    ]
    discussions = {"General Comments": "GC text"}
    manifest = {
        "source_pdf_sha256": "x",
        "source_url": "u",
        "page_count": 1,
        "counts": {"commenters": 1, "comments": 1, "topics": 1, "unresolved_refs": 1},
    }
    return commenters, records, discussions, manifest


def test_write_outputs_produces_all_artifacts(tmp_path):
    commenters, records, discussions, manifest = _fixture_outputs()
    out = tmp_path / "ccl5"
    cli.write_outputs(
        commenters=commenters,
        records=records,
        discussions=discussions,
        manifest=manifest,
        out_dir=out,
    )
    for name in [
        "crosswalk.jsonl",
        "crosswalk.csv",
        "commenters.csv",
        "topic_discussions.json",
        "parse_manifest.json",
    ]:
        assert (out / name).exists(), f"missing {name}"

    row = json.loads((out / "crosswalk.jsonl").read_text().splitlines()[0])
    assert row["document_id"] == "EPA-HQ-OW-2018-0594-0052"
    assert row["topic_refs"][1] == {"raw": "1,4-Dioxane", "canonical": None, "resolved": False}
    assert row["topic_discussions"]["General Comments"] == "GC text"


def test_write_outputs_refuses_to_overwrite(tmp_path):
    commenters, records, discussions, manifest = _fixture_outputs()
    out = tmp_path / "ccl5"
    cli.write_outputs(
        commenters=commenters, records=records, discussions=discussions,
        manifest=manifest, out_dir=out,
    )
    with pytest.raises(FileExistsError):
        cli.write_outputs(
            commenters=commenters, records=records, discussions=discussions,
            manifest=manifest, out_dir=out,
        )
