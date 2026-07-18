"""Strip the RTC PDF's repeating noise so the structural anchors stand alone.

`strip_page_headers` removes the 5-line EPA/page-number block that fitz emits at
the top of every page, and joins pages into one text. `strip_running_headers`
removes the per-page `Agency Discussion on ...` / `Comments Received on ...`
running headers -- applied to body captures only, because their FIRST occurrence
is the anchor the topic/comment splitters key on.
"""

from __future__ import annotations

import re

_NOISE = re.compile(
    r"^(?:EPA-OGWDW|Draft CCL 5 Response to Comments|EPA 815-R-22-001|"
    r"October 2022|Page \d+ of \d+)\s*$"
)
_RUNNING = re.compile(r"^(?:Agency Discussion on|Comments Received on) .+$")


def strip_page_headers(pages: list[str]) -> str:
    """Drop the per-page EPA/page-number noise lines; return one joined text."""
    kept: list[str] = []
    for page in pages:
        for line in page.splitlines():
            if not _NOISE.match(line.strip()):
                kept.append(line)
    return "\n".join(kept)


def strip_running_headers(text: str) -> str:
    """Drop `Agency Discussion on ...` / `Comments Received on ...` header lines."""
    return "\n".join(line for line in text.splitlines() if not _RUNNING.match(line.strip()))
