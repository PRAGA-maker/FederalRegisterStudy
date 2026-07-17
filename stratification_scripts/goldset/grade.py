"""
Grade returned human labels into per-source web-search false-negative rates.

FN rate = P(true == yes | pred == no). Every frame row has pred == no by
construction, so within a stratum the FN rate is just the share the labeler
marked "yes" (a response the pipeline missed). Reported unweighted with a Wilson
CI and as an HT-weighted point estimate, per response_source, with the frame
caveat restated in the output itself.

Validation fails loud: a "yes" without an evidence quote is a guess, not a label.
"""

from __future__ import annotations

import math

import polars as pl

from stratification_scripts.goldset.sample import FRAME_SOURCES

VALID_TRUE_FOUND = {"yes", "no", "uncertain"}
VALID_DECISION = {"accept", "partial", "reject", "uncertain"}


def load_labels(path) -> pl.DataFrame:
    """Read a filled labeling sheet, all columns as strings (empty cells → '')."""
    return pl.read_csv(path, infer_schema_length=0).fill_null("")


def validate_labels(labels: pl.DataFrame, key: pl.DataFrame) -> None:
    """Raise ValueError unless every packet row is labeled exactly once with valid values."""
    label_ids = labels["label_row_id"].to_list()
    key_ids = set(key["label_row_id"].to_list())

    unknown = [i for i in label_ids if i not in key_ids]
    if unknown:
        raise ValueError(f"unknown label_row_id(s) not in the key: {sorted(set(unknown))}")

    missing = key_ids - set(label_ids)
    if missing:
        raise ValueError(f"missing labels for {len(missing)} packet row(s): {sorted(missing)}")

    if len(label_ids) != len(set(label_ids)):
        raise ValueError("duplicate label_row_id(s) in the returned labels")

    for r in labels.iter_rows(named=True):
        tf = (r.get("true_response_found") or "").strip().lower()
        if tf not in VALID_TRUE_FOUND:
            raise ValueError(f"{r['label_row_id']}: true_response_found={tf!r} not in {VALID_TRUE_FOUND}")
        dec = (r.get("true_agency_decision") or "").strip().lower()
        if dec and dec not in VALID_DECISION:
            raise ValueError(f"{r['label_row_id']}: true_agency_decision={dec!r} not in {VALID_DECISION}")
        if tf == "yes" and not (r.get("evidence_quote") or "").strip():
            raise ValueError(f"{r['label_row_id']}: true_response_found=yes requires an evidence_quote")


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n. n==0 → (0.0, 1.0)."""
    if n == 0:
        return (0.0, 1.0)
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _stratum_stats(rows: list[dict], frame_weight_mass: float) -> dict:
    """rows: joined (label ⋈ key) dicts for one response_source stratum."""
    n = len(rows)
    is_yes = [(r["true_response_found"].strip().lower() == "yes") for r in rows]
    yes = sum(is_yes)
    uncertain = sum(1 for r in rows if r["true_response_found"].strip().lower() == "uncertain")
    weights = [float(r["response_sample_weight"]) for r in rows]

    fn_unweighted = yes / n if n else 0.0
    lo, hi = wilson_ci(yes, n)

    wsum = sum(weights)
    wyes = sum(w for w, y in zip(weights, is_yes) if y)
    fn_weighted = (wyes / wsum) if wsum else 0.0

    return {
        "n": n,
        "yes": yes,
        "uncertain": uncertain,
        "fn_unweighted": fn_unweighted,
        "fn_unweighted_ci95": [lo, hi],
        "fn_weighted": fn_weighted,
        "frame_weight_mass": frame_weight_mass,
        "projected_missed": fn_weighted * frame_weight_mass,
    }


def compute_stats(labels: pl.DataFrame, key: pl.DataFrame, manifest: dict) -> dict:
    """Per-source FN stats + the web-vs-fr contrast. Assumes labels already validated.

    FN denominator is every sampled row in the stratum (all pred==no); `uncertain`
    labels stay in the denominator (a conservative FN estimate) and are also
    reported explicitly so the reader can see how many were unresolved.
    """
    joined = key.join(labels, on="label_row_id", how="left")
    per_source = {}
    for src in FRAME_SOURCES:
        rows = [r for r in joined.iter_rows(named=True) if r["response_source"] == src]
        mass = float(manifest["strata"].get(src, {}).get("frame_weight_mass", 0.0))
        per_source[src] = _stratum_stats(rows, mass)

    contrast = per_source["web_search"]["fn_unweighted"] - per_source["fr_preamble"]["fn_unweighted"]
    return {"strata": per_source, "contrast_web_minus_fr": contrast}
