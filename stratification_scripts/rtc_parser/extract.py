"""PDF -> per-page text, and PDF hashing. The only io/fitz unit in the parser.

Kept deliberately thin so all the structural parsing logic downstream is pure
functions over strings, testable on text fixtures without a PDF.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def extract_pages(pdf_path: Path | str) -> list[str]:
    """Return the embedded text of each page, in order."""
    import fitz  # PyMuPDF

    doc = fitz.open(pdf_path)
    try:
        return [doc[i].get_text() for i in range(doc.page_count)]
    finally:
        doc.close()


def pdf_sha256(pdf_path: Path | str) -> str:
    """Hex sha256 of the raw PDF bytes (recorded in parse_manifest.json)."""
    return hashlib.sha256(Path(pdf_path).read_bytes()).hexdigest()
