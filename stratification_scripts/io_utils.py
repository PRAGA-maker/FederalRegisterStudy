"""
I/O utilities for the Federal Register Study pipeline.

This module provides:
- UTF-8 text sanitization for API responses
- PDF text extraction

Example:
    >>> from stratification_scripts.io_utils import sanitize_utf8, extract_pdf_text
    >>>
    >>> # Clean text with invalid UTF-8
    >>> clean_text = sanitize_utf8(dirty_text)
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Optional

from stratification_scripts.logging_utils import get_logger

logger = get_logger(__name__)


def sanitize_utf8(text: Any) -> Optional[str]:
    """
    Sanitize text to remove invalid UTF-8 characters.

    Removes surrogate characters (U+D800 to U+DFFF) that can't be encoded
    to valid UTF-8, which can cause issues when writing to CSV or databases.

    Args:
        text: Input text (str, bytes, or None)

    Returns:
        Sanitized UTF-8 string or None if input is None/empty.

    Example:
        >>> sanitize_utf8("Hello \\ud83d\\ude00 World")  # Valid emoji
        'Hello World'
        >>> sanitize_utf8(None)
        None
    """
    if text is None:
        return None

    if isinstance(text, bytes):
        try:
            text = text.decode('utf-8', errors='replace')
        except Exception:
            return None

    if not isinstance(text, str):
        text = str(text)

    if not text:
        return None

    # Remove surrogate characters that can't be encoded to UTF-8
    # Surrogates are in the range U+D800 to U+DFFF
    sanitized_chars = []
    for char in text:
        code_point = ord(char)
        # Skip surrogate characters
        if 0xD800 <= code_point <= 0xDFFF:
            continue
        sanitized_chars.append(char)

    sanitized = ''.join(sanitized_chars)

    if not sanitized:
        return None

    # Final check: ensure it can be encoded to UTF-8
    try:
        sanitized.encode('utf-8')
        return sanitized
    except (UnicodeEncodeError, UnicodeError):
        # Last resort: use error handling
        try:
            return sanitized.encode('utf-8', errors='replace').decode('utf-8', errors='replace') or None
        except Exception:
            return None


def extract_pdf_text(
    pdf_bytes: bytes,
    max_pages: int = 2,
) -> Optional[str]:
    """
    Extract text from PDF bytes.

    Tries pypdf first, falls back to PyMuPDF if that fails.

    Args:
        pdf_bytes: Raw PDF file content
        max_pages: Maximum number of pages to extract (default: 2)

    Returns:
        Extracted text or None if extraction failed.
    """
    if not pdf_bytes:
        return None

    # Verify it's a PDF
    if not pdf_bytes.startswith(b'%PDF'):
        if b'<!DOCTYPE html>' in pdf_bytes[:100] or b'<html' in pdf_bytes[:100]:
            logger.debug("Got HTML instead of PDF")
            return None
        logger.debug(f"Not a PDF file (starts with {pdf_bytes[:20]})")
        return None

    # Try pypdf first (pure python, often reliable)
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(pdf_bytes))

        text_parts = []
        for page_num in range(min(max_pages, len(reader.pages))):
            page = reader.pages[page_num]
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)

        text = "\n\n".join(text_parts).strip()
        return sanitize_utf8(text) if text else None

    except Exception as pypdf_error:
        # Fall back to PyMuPDF
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")

            text_parts = []
            for page_num in range(min(max_pages, len(doc))):
                page = doc[page_num]
                page_text = page.get_text()
                if page_text:
                    text_parts.append(page_text)

            doc.close()

            text = "\n\n".join(text_parts).strip()
            return sanitize_utf8(text) if text else None

        except Exception as pymupdf_error:
            logger.debug(
                f"PDF extraction failed: pypdf={pypdf_error}, "
                f"pymupdf={pymupdf_error}"
            )
            return None
