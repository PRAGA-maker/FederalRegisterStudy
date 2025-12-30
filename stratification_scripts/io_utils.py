"""
I/O utilities for the Federal Register Study pipeline.

This module provides:
- UTF-8 text sanitization for API responses
- CSV reading/writing with consistent handling
- Path utilities for year-specific files

Example:
    >>> from stratification_scripts.io_utils import sanitize_utf8, read_csv_polars
    >>> 
    >>> # Clean text with invalid UTF-8
    >>> clean_text = sanitize_utf8(dirty_text)
    >>> 
    >>> # Read CSV with polars
    >>> df = read_csv_polars("data/comments.csv")
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Optional, Union

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
        'Hello 😀 World'
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


def read_csv_polars(
    path: Union[str, Path],
    *,
    infer_schema_length: Optional[int] = None,
) -> "pl.DataFrame":
    """
    Read a CSV file using Polars.
    
    Args:
        path: Path to CSV file
        infer_schema_length: Number of rows to use for schema inference.
                            None uses Polars default.
    
    Returns:
        Polars DataFrame
    
    Raises:
        FileNotFoundError: If the file doesn't exist.
    
    Example:
        >>> df = read_csv_polars("data/comments.csv")
        >>> print(df.shape)
    """
    import polars as pl
    
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    
    kwargs = {}
    if infer_schema_length is not None:
        kwargs["infer_schema_length"] = infer_schema_length
    
    return pl.read_csv(str(path), **kwargs)


def write_csv_polars(
    df: "pl.DataFrame",
    path: Union[str, Path],
    *,
    mkdir: bool = True,
) -> Path:
    """
    Write a Polars DataFrame to CSV.
    
    Args:
        df: DataFrame to write
        path: Output path
        mkdir: Create parent directories if needed
    
    Returns:
        Path to written file
    
    Side Effects:
        Creates parent directories if mkdir=True.
        Writes file to disk.
    """
    path = Path(path)
    
    if mkdir:
        path.parent.mkdir(parents=True, exist_ok=True)
    
    df.write_csv(str(path))
    logger.debug(f"Wrote {len(df)} rows to {path}")
    
    return path


def read_csv_pandas(
    path: Union[str, Path],
) -> "pd.DataFrame":
    """
    Read a CSV file using Pandas.
    
    Args:
        path: Path to CSV file
    
    Returns:
        Pandas DataFrame
    
    Raises:
        FileNotFoundError: If the file doesn't exist.
    """
    import pandas as pd
    
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    
    return pd.read_csv(path)


def write_csv_pandas(
    df: "pd.DataFrame",
    path: Union[str, Path],
    *,
    index: bool = False,
    mkdir: bool = True,
) -> Path:
    """
    Write a Pandas DataFrame to CSV.
    
    Args:
        df: DataFrame to write
        path: Output path
        index: Whether to include index column
        mkdir: Create parent directories if needed
    
    Returns:
        Path to written file
    
    Side Effects:
        Creates parent directories if mkdir=True.
        Writes file to disk.
    """
    path = Path(path)
    
    if mkdir:
        path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(path, index=index)
    logger.debug(f"Wrote {len(df)} rows to {path}")
    
    return path


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path
    
    Returns:
        Path object for the directory
    
    Side Effects:
        Creates directory if it doesn't exist.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def format_year_path(
    template: str,
    year: int,
) -> Path:
    """
    Format a path template with a year value.
    
    Args:
        template: Path template with {year} placeholder
        year: Year to substitute
    
    Returns:
        Path with year substituted
    
    Example:
        >>> format_year_path("data/comments_{year}.csv", 2024)
        Path('data/comments_2024.csv')
    """
    return Path(template.format(year=year))


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

