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

import hashlib
import random
from datetime import datetime, timezone

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


def make_label_row_id(seed: int, snapshot_id: str, comment_id: str) -> str:
    """Opaque, stable id for a sampled comment. Encodes no stratum or position."""
    digest = hashlib.sha256(f"{seed}|{snapshot_id}|{comment_id}".encode()).hexdigest()
    return digest[:12]


def draw_sample(
    frame: pl.DataFrame,
    *,
    snapshot_id: str,
    seed: int,
    n: int = 15,
    overlap: int = 10,
) -> pl.DataFrame:
    """Seeded stratified draw of n rows per response_source, interleaved.

    Determinism: within each stratum rows are sorted by comment_id (canonical
    order independent of file/read order), then a single seeded RNG picks the
    sample and shuffles the combined result so strata interleave. Same
    (seed, snapshot_id) ⇒ identical output, forever.
    """
    rng = random.Random(seed)
    picked_frames: list[pl.DataFrame] = []
    for src in FRAME_SOURCES:
        stratum = frame.filter(pl.col("response_source") == src).sort("comment_id")
        if stratum.height < n:
            raise ValueError(
                f"stratum {src!r} has only {stratum.height} rows; cannot draw n={n}"
            )
        idx = sorted(rng.sample(range(stratum.height), n))
        picked_frames.append(stratum[idx])

    # Interleave: canonicalize (comment_id-sorted) then shuffle with the same RNG.
    combined = pl.concat(picked_frames).sort("comment_id")
    order = list(range(combined.height))
    rng.shuffle(order)
    combined = combined[order]

    label_ids = [
        make_label_row_id(seed, snapshot_id, cid)
        for cid in combined["comment_id"].to_list()
    ]
    if len(set(label_ids)) != len(label_ids):
        raise ValueError("label_row_id collision — widen the id hash")

    overlap_ids = set(rng.sample(label_ids, min(overlap, len(label_ids))))
    return combined.with_columns(
        pl.Series("label_row_id", label_ids),
        pl.Series("overlap_candidate", [lid in overlap_ids for lid in label_ids]),
    )


def _iso_z(moment: datetime) -> str:
    return (
        moment.astimezone(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def make_seed_id(snapshot_id: str, moment: datetime) -> str:
    """Seed-run directory id: '<YYYY-MM-DD>-<snapshot-short>' (mirrors the freeze id)."""
    snapshot_short = snapshot_id.rsplit("-", 1)[-1]
    return f"{moment.date().isoformat()}-{snapshot_short}"


def _weight_mass(df: pl.DataFrame) -> float:
    return float(df["response_sample_weight"].cast(pl.Float64).sum())


def build_sample_manifest(
    frame: pl.DataFrame,
    sampled: pl.DataFrame,
    *,
    snapshot_id: str,
    seed: int,
    n: int,
    overlap: int,
    moment: datetime,
) -> dict:
    """Provenance for a seed run: the frame shape, per-stratum weight mass, and the sampled rows.

    frame_weight_mass is the Σ weight over ALL rows in each stratum of the frame
    (not just the sampled rows) — the grader's projection denominator.
    """
    strata = {}
    for src in FRAME_SOURCES:
        sfx = frame.filter(pl.col("response_source") == src)
        strata[src] = {
            "frame_rows": sfx.height,
            "frame_weight_mass": round(_weight_mass(sfx), 4),
            "allocated": n,
        }
    sampled_rows = [
        {
            "label_row_id": r["label_row_id"],
            "comment_id": r["comment_id"],
            "response_source": r["response_source"],
            "response_sample_weight": float(r["response_sample_weight"]),
            "overlap_candidate": r["overlap_candidate"],
        }
        for r in sampled.iter_rows(named=True)
    ]
    return {
        "snapshot_id": snapshot_id,
        "created_at": _iso_z(moment),
        "seed": seed,
        "n_per_stratum": n,
        "overlap": overlap,
        "frame_total_rows": frame.height,
        "strata": strata,
        "sampled": sampled_rows,
    }
