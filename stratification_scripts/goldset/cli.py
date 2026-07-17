"""
Gold-set CLI — standalone, never wired into the pipeline cli.py.

    python -m stratification_scripts.goldset sample --snapshot <id> [--n 15] [--seed 0] [--overlap 10]
    python -m stratification_scripts.goldset grade <seed-id> --labels <path>
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from stratification_scripts import config
from stratification_scripts.goldset import grade, packet, sample


def write_seed_run(*, packet: pl.DataFrame, key: pl.DataFrame, manifest: dict, seed_id: str) -> Path:
    """Write manifest + packet + key into goldset/<seed_id>/; refuse to overwrite."""
    seed_dir = config.get_goldset_seed_path(seed_id)
    if seed_dir.exists():
        raise FileExistsError(f"seed run already exists: {seed_dir}")
    seed_dir.mkdir(parents=True)
    (seed_dir / "sample_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    packet.write_csv(seed_dir / "labeling_packet.csv")
    key.write_csv(seed_dir / "prediction_key.csv")
    return seed_dir


def cmd_sample(args) -> int:
    frame = sample.load_frame(args.snapshot, year=args.year)
    sampled = sample.draw_sample(
        frame, snapshot_id=args.snapshot, seed=args.seed, n=args.n, overlap=args.overlap
    )
    moment = datetime.now(timezone.utc)
    seed_id = sample.make_seed_id(args.snapshot, moment)
    manifest = sample.build_sample_manifest(
        frame, sampled, snapshot_id=args.snapshot, seed=args.seed,
        n=args.n, overlap=args.overlap, moment=moment,
    )
    manifest["seed_id"] = seed_id
    pkt, key = packet.build_packet_and_key(sampled, snapshot_id=args.snapshot, year=args.year)
    seed_dir = write_seed_run(packet=pkt, key=key, manifest=manifest, seed_id=seed_id)
    print(f"wrote seed run {seed_id} ({pkt.height} rows) → {seed_dir}")
    print("next: label labeling_packet.csv in a spreadsheet, save as labels_returned.csv, then `grade`.")
    return 0


def cmd_grade(args) -> int:
    seed_dir = config.get_goldset_seed_path(args.seed_id)
    manifest = json.loads((seed_dir / "sample_manifest.json").read_text())
    key = pl.read_csv(seed_dir / "prediction_key.csv", infer_schema_length=0)
    labels = grade.load_labels(args.labels)
    grade.validate_labels(labels, key)
    stats = grade.compute_stats(labels, key, manifest)
    grade.write_results(stats, seed_dir, n_per_stratum=manifest.get("n_per_stratum", args.n))
    print((seed_dir / "results.md").read_text())
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="goldset",
        description="Draw a blind gold-set labeling packet from a frozen snapshot, then grade returned labels.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_sample = sub.add_parser("sample", help="Draw a stratified sample and write the blind packet.")
    p_sample.add_argument("--snapshot", required=True, help="Frozen snapshot id, e.g. 2026-07-15-ce44ac5.")
    p_sample.add_argument("--year", type=int, default=2024)
    p_sample.add_argument("--n", type=int, default=15, help="Rows per stratum.")
    p_sample.add_argument("--seed", type=int, default=0, help="RNG seed (reproducibility).")
    p_sample.add_argument("--overlap", type=int, default=10, help="Rows flagged for double-labeling.")

    p_grade = sub.add_parser("grade", help="Grade returned labels for a seed run.")
    p_grade.add_argument("seed_id", help="Seed run id (the goldset/<seed-id> dir name).")
    p_grade.add_argument("--labels", required=True, help="Path to the filled labels_returned.csv.")
    p_grade.add_argument("--n", type=int, default=15, help="Fallback n if absent from the manifest.")

    args = parser.parse_args(argv)
    try:
        if args.cmd == "sample":
            return cmd_sample(args)
        if args.cmd == "grade":
            return cmd_grade(args)
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 1  # unreachable: subparser required


if __name__ == "__main__":
    sys.exit(main())
