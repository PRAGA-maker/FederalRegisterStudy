from pathlib import Path

from stratification_scripts import config


def test_get_frozen_dir_is_repo_root_frozen():
    frozen = config.get_frozen_dir()
    assert frozen == config.get_project_root() / "frozen"
    assert frozen.is_dir()  # mkdir side effect


def test_get_frozen_snapshot_path_joins_id():
    p = config.get_frozen_snapshot_path("2026-07-15-abc1234")
    assert p == config.get_frozen_dir() / "2026-07-15-abc1234"
