from __future__ import annotations

from stratification_scripts.rtc_parser import clean

PAGE = (
    "EPA-OGWDW\nDraft CCL 5 Response to Comments\nEPA 815-R-22-001\n"
    "October 2022\nPage 14 of 159\n\nReal body line\n"
)


def test_strip_page_headers_removes_noise_keeps_body():
    out = clean.strip_page_headers([PAGE, PAGE])
    assert "Real body line" in out
    for noise in ["EPA-OGWDW", "EPA 815-R-22-001", "Page 14 of 159", "October 2022"]:
        assert noise not in out


def test_strip_running_headers_removes_section_headers():
    txt = "Agency Discussion on General Comments\nkept\nComments Received on PFAS\nalso kept\n"
    out = clean.strip_running_headers(txt)
    assert "kept" in out and "also kept" in out
    assert "Agency Discussion on" not in out
    assert "Comments Received on" not in out
