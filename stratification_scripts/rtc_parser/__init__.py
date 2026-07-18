"""
EPA Response-to-Comments (RTC) crosswalk parser — Tier-0, standalone.

Parses an agency RTC PDF (the CCL5 Draft Response to Comments, EPA 815-R-22-001)
into a structured per-comment crosswalk: regulations.gov Document ID -> topic(s)
-> agency disposition text, plus the Exhibit 2 commenter table and the per-topic
Agency Topic Discussions.

This is a NEW coded-table parser. It is deliberately NOT an extension of the
prose response extractor (makeup/track_responses.py). It is never wired into the
pipeline cli.py; run it only as `python -m stratification_scripts.rtc_parser`.

See docs/superpowers/specs/2026-07-18-epa-rtc-crosswalk-parser-design.md.
"""
