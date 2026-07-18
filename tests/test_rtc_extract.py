from __future__ import annotations

import hashlib
from pathlib import Path

import fitz

from stratification_scripts.rtc_parser import extract


def _make_pdf(tmp_path: Path) -> Path:
    doc = fitz.open()
    for txt in ["PAGE ONE hello", "PAGE TWO world"]:
        page = doc.new_page()
        page.insert_text((72, 72), txt)
    p = tmp_path / "tiny.pdf"
    doc.save(p)
    doc.close()
    return p


def test_extract_pages_returns_text_per_page(tmp_path):
    pages = extract.extract_pages(_make_pdf(tmp_path))
    assert len(pages) == 2
    assert "hello" in pages[0]
    assert "world" in pages[1]


def test_pdf_sha256_matches_hashlib(tmp_path):
    p = _make_pdf(tmp_path)
    assert extract.pdf_sha256(p) == hashlib.sha256(p.read_bytes()).hexdigest()
