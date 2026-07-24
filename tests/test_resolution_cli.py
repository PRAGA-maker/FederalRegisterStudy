import json

from stratification_scripts.resolution import cli
from stratification_scripts.resolution.models import (
    AgendaStatus, Channel, ResolutionResult, Status,
)


def _result(comment_id: str) -> ResolutionResult:
    return ResolutionResult(
        comment_id=comment_id, comment_date="2024-09-23",
        source_document="2024-18496",
        status=Status.FOUND, absence_reason=None, candidates=[],
        agenda=AgendaStatus("2105-AF05", "FINAL", [], False, False,
                            "2026-07-23T00:00:00", True),
        channels_run={c: "ok" for c in Channel},
        resolved_at="2026-07-23T00:00:00",
    )


def test_write_run_emits_jsonl_summary_and_manifest(tmp_path):
    out = cli.write_run(
        results=[_result("A-1"), _result("B-2")],
        manifest={"snapshot": "2026-07-15-ce44ac5", "year": 2024, "rows": 2,
                  "cache_stats": {}, "started_at": "t0", "finished_at": "t1"},
        out_dir=tmp_path / "run1",
    )
    assert (out / "resolutions.jsonl").exists()
    assert (out / "summary.csv").exists()
    assert (out / "manifest.json").exists()
    lines = (out / "resolutions.jsonl").read_text().strip().splitlines()
    assert [json.loads(line)["comment_id"] for line in lines] == ["A-1", "B-2"]
    assert json.loads((out / "manifest.json").read_text())["rows"] == 2


def test_write_run_refuses_to_overwrite_a_non_empty_directory(tmp_path):
    target = tmp_path / "run1"
    target.mkdir()
    (target / "resolutions.jsonl").write_text("existing")
    try:
        cli.write_run(results=[_result("A-1")], manifest={}, out_dir=target)
    except FileExistsError:
        return
    raise AssertionError("expected FileExistsError")


def test_main_requires_a_snapshot():
    assert cli.main(["resolve"]) != 0
