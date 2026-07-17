"""
Gold-set sampling — the reproducible stratified draw from a frozen snapshot.

The frame is the subpopulation where the pipeline's "no response" claim is
*checkable*: a final rule provably exists (FINAL_EFFECTIVE), the pipeline said
"no" (response_found == no), and the answer came from a source a human can
re-derive (web_search or fr_preamble). Within that frame we draw n per source
with a seeded RNG, so the same (seed, snapshot) reproduces the sample forever.

Standalone: NOT imported by cli.py, NOT a pipeline step (like freeze).
"""

from __future__ import annotations

import polars as pl

from stratification_scripts import config

FRAME_LIFECYCLE = "FINAL_EFFECTIVE"
FRAME_RESPONSE_FOUND = "no"
FRAME_SOURCES = ("web_search", "fr_preamble")


def frame_from_agency_responses(df: pl.DataFrame) -> pl.DataFrame:
    """The checkable-"no" frame: FINAL_EFFECTIVE ∧ response_found==no ∧ source∈{web_search, fr_preamble}."""
    return df.filter(
        (pl.col("lifecycle_stage") == FRAME_LIFECYCLE)
        & (pl.col("response_found") == FRAME_RESPONSE_FOUND)
        & (pl.col("response_source").is_in(list(FRAME_SOURCES)))
    )


def load_frame(snapshot_id: str, *, year: int = 2024) -> pl.DataFrame:
    """Read agency_responses_<year>.csv from the pinned snapshot and apply the frame filter."""
    base = config.get_frozen_snapshot_path(snapshot_id)
    src = base / f"makeup/data/agency_responses_{year}.csv"
    df = pl.read_csv(src, infer_schema_length=0)  # all-string, matching the freeze convention
    return frame_from_agency_responses(df)
