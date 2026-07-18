from __future__ import annotations

from stratification_scripts import config


def test_rtc_paths_are_project_siblings():
    root = config.get_project_root()
    assert config.get_rtc_dir() == root / "rtc"
    assert config.get_rtc_output_path("ccl5") == root / "rtc" / "ccl5"
    assert config.get_rtc_inputs_dir() == root / "rtc" / "inputs"
