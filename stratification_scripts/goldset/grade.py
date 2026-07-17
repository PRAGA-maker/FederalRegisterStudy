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

import polars as pl

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
