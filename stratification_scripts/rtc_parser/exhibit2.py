"""Parse Exhibit 2 -- the commenter table that maps Commenter Number <-> Document ID.

The table linearizes (fitz) into fixed-position cells per row: a bare-integer
Commenter Number line, the Document ID line, then First, Last, and Organization
(the last wrapping over 1+ lines). Blank name cells render as ' '. The table
header repeats at each page break and is dropped.

Rows are anchored on the (integer line -> Document ID line) pair so Document IDs
that appear in surrounding prose are never mistaken for rows. The region is
bounded to `[Exhibit 2 header .. the Section 2 marker that follows it]`.
"""

from __future__ import annotations

import re

from stratification_scripts.rtc_parser.models import Commenter

_EXHIBIT2_HEADER = "Exhibit 2: List of Public Commenters"
_SECTION2_MARKER = "2. Comments and EPA Responses by Topic"

# Header/column-label lines that repeat at page breaks; dropped verbatim.
_TABLE_LABELS = {
    _EXHIBIT2_HEADER,
    "Comment Information",
    "Submitter Information",
    "Commenter",
    "Number",
    "Document ID",
    "First Name",
    "Last Name",
    "Organization Name",
}

_INT = re.compile(r"^\d+$")


def _region(text: str) -> str:
    """Slice from the Exhibit 2 header to the Section 2 marker that follows it.

    The Section 2 marker also appears in the Table of Contents, so the end anchor
    is searched *after* the Exhibit 2 header, not from the top of the document.
    """
    lo = text.find(_EXHIBIT2_HEADER)
    if lo == -1:
        return ""
    hi = text.find(_SECTION2_MARKER, lo)
    return text[lo:hi] if hi != -1 else text[lo:]


def parse_commenters(
    text: str, docket_prefix: str = "EPA-HQ-OW-2018-0594"
) -> list[Commenter]:
    """Return one Commenter per Exhibit 2 row, in document order."""
    doc_id = re.compile(rf"^{re.escape(docket_prefix)}-\d{{4}}$")
    # Keep blank lines (they hold positional first/last cells); drop only labels.
    lines = [
        ln.rstrip()
        for ln in _region(text).splitlines()
        if ln.strip() not in _TABLE_LABELS
    ]

    # Find row-start indices: a bare-int line immediately followed by a doc-id line.
    starts: list[int] = []
    for i in range(len(lines) - 1):
        if _INT.match(lines[i].strip()) and doc_id.match(lines[i + 1].strip()):
            starts.append(i)

    rows: list[Commenter] = []
    for s, start in enumerate(starts):
        end = starts[s + 1] if s + 1 < len(starts) else len(lines)
        number = int(lines[start].strip())
        document_id = lines[start + 1].strip()
        block = lines[start + 2 : end]  # positional: [First, Last, Org...]
        first = block[0].strip() if len(block) > 0 else ""
        last = block[1].strip() if len(block) > 1 else ""
        org = " ".join(p.strip() for p in block[2:] if p.strip()).strip()
        rows.append(Commenter(number, document_id, first, last, org))
    return rows
