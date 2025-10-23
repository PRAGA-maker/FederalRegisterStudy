"""
Efficiency frontier visualization for Federal Register comments.

Reads CSV created by 2024distribution.py and produces:
- Primary coverage curve ("efficiency frontier"): share of comments captured by focusing on the top x% of documents
- Lorenz curve + Gini coefficient
- Rank-size (Zipf-style) view
Also emits a concise metrics JSON and console summary.

Usage:
    python stratification_scripts/output/efficiency_frontier.py --year 2024

Requirements: pandas, numpy, matplotlib
"""

import argparse
import json
import os
import sys
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# Use non-interactive backend for headless environments
import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def load_data(output_dir: str, year: int) -> pd.DataFrame:
    csv_path = os.path.join(output_dir, f"federal_register_{year}_comments.csv")
    if not os.path.exists(csv_path):
        print(
            f"ERROR: Data CSV not found at {csv_path}.\n"
            "Run: python stratification_scripts/2024distribution.py --year 2024"
        )
        sys.exit(1)
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        print(f"ERROR: Failed to read CSV {csv_path}: {exc}")
        sys.exit(1)
    return df


def filter_channel(df: pd.DataFrame, include_all_channels: bool) -> pd.DataFrame:
    if include_all_channels:
        return df.copy()
    # Default: use regs.gov only, where counts are authoritative
    if "submission_channel" not in df.columns:
        return df.copy()
    return df.loc[df["submission_channel"] == "regs.gov"].copy()


def get_counts(df: pd.DataFrame) -> np.ndarray:
    if "comment_count" not in df.columns:
        print("ERROR: 'comment_count' column missing in dataset")
        sys.exit(1)
    counts = (
        pd.to_numeric(df["comment_count"], errors="coerce")
        .fillna(0)
        .astype(int)
        .clip(lower=0)
    )
    return counts.to_numpy()


def compute_frontier(counts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if counts.size == 0:
        print("ERROR: No rows after filtering")
        sys.exit(1)
    ordered = np.sort(counts)[::-1]
    total = ordered.sum()
    if total <= 0:
        print("ERROR: No comments in dataset after filtering")
        sys.exit(1)
    cum = np.cumsum(ordered)
    n = ordered.size
    x_share_docs = np.arange(1, n + 1) / n
    y_share_comments = cum / total
    return x_share_docs, y_share_comments


def compute_lorenz_and_gini(counts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    asc = np.sort(np.maximum(counts, 0))
    total = asc.sum()
    if total <= 0:
        # Degenerate case; return equality line
        x = np.linspace(0, 1, len(asc) if len(asc) > 1 else 2)
        y = x.copy()
        return x, y, 0.0
    lorenz_y = np.cumsum(asc) / total
    lorenz_x = np.linspace(0, 1, len(asc), endpoint=True)
    gini = float(1 - 2 * np.trapz(lorenz_y, lorenz_x))
    return lorenz_x, lorenz_y, gini


def compute_rank_size(counts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(counts)[::-1]
    ranks = np.arange(1, len(ordered) + 1)
    return ranks, ordered


def compute_threshold_metrics(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    n = len(x)
    # Coverage at top p% of docs
    tops = {}
    for t in [0.01, 0.05, 0.10]:
        idx = max(0, int(np.ceil(t * n)) - 1)
        tops[f"coverage_top_{int(t*100)}pct_docs"] = float(y[idx])

    # Docs needed to reach target share of comments
    needed = {}
    for target in [0.5, 0.8, 0.9, 0.95]:
        k = int(np.searchsorted(y, target, side="left") + 1)
        k = min(max(1, k), n)
        needed[f"docs_share_for_{int(target*100)}pct_comments"] = float(k / n)

    return {**tops, **needed}


def format_pct(value: float) -> str:
    return f"{value*100:.1f}%"


def render_figure(
    out_png: str,
    x: np.ndarray,
    y: np.ndarray,
    lorenz_x: np.ndarray,
    lorenz_y: np.ndarray,
    gini: float,
    ranks: np.ndarray,
    ordered_counts: np.ndarray,
    year: int,
    include_all_channels: bool,
    show_lorenz: bool,
    show_zipf: bool,
    annotations_on: bool,
) -> None:
    # Layout: Top = Frontier (full width). Bottom-left = Lorenz. Bottom-right = Rank-Size
    fig = plt.figure(figsize=(12, 8), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.25, 1])

    # Frontier (primary)
    ax_frontier = fig.add_subplot(grid[0, :])
    ax_frontier.plot(x * 100, y * 100, label="Coverage frontier", color="#1f77b4", lw=2)
    ax_frontier.plot([0, 100], [0, 100], ls="--", color="#777777", label="Uniform attention (y = x)")
    ax_frontier.set_xlim(0, 100)
    ax_frontier.set_ylim(0, 100)
    ax_frontier.set_xlabel("Share of documents selected (best-first)")
    ax_frontier.set_ylabel("Share of total comments captured")
    chan = "All channels" if include_all_channels else "Regs.gov only"
    ax_frontier.set_title(f"Public attention is concentrated: 2024 {chan}")
    ax_frontier.grid(True, alpha=0.2)

    if annotations_on:
        # Annotate target comment shares with needed doc shares
        for target, color in zip([0.5, 0.8, 0.9, 0.95], ["#2ca02c", "#ff7f0e", "#d62728", "#9467bd"]):
            idx = int(np.searchsorted(y, target, side="left"))
            idx = min(max(0, idx), len(x) - 1)
            x_pct = x[idx] * 100
            y_pct = y[idx] * 100
            ax_frontier.scatter([x_pct], [y_pct], color=color, s=40, zorder=3)
            ax_frontier.axvline(x_pct, color=color, ls=":", alpha=0.6)
            ax_frontier.text(
                x_pct,
                y_pct,
                f"  {format_pct(x[idx])} docs → {format_pct(target)} comments",
                va="bottom",
                ha="left",
                fontsize=9,
                color=color,
            )

        # Annotate coverage at top doc shares (1%, 5%, 10%)
        for share, color in zip([0.01, 0.05, 0.10], ["#17becf", "#bcbd22", "#8c564b"]):
            idx = max(0, int(np.ceil(share * len(x))) - 1)
            ax_frontier.scatter([share * 100], [y[idx] * 100], color=color, s=30)
            ax_frontier.text(
                share * 100,
                y[idx] * 100,
                f"  top {int(share*100)}% docs → {format_pct(y[idx])} comments",
                va="bottom",
                ha="left",
                fontsize=9,
                color=color,
            )

    ax_frontier.legend(loc="lower right")

    # Lorenz
    if show_lorenz:
        ax_lorenz = fig.add_subplot(grid[1, 0])
        ax_lorenz.plot(lorenz_x * 100, lorenz_y * 100, color="#1f77b4", lw=2)
        ax_lorenz.plot([0, 100], [0, 100], ls="--", color="#777777")
        ax_lorenz.set_xlim(0, 100)
        ax_lorenz.set_ylim(0, 100)
        ax_lorenz.set_xlabel("Cumulative share of documents (worst → best)")
        ax_lorenz.set_ylabel("Cumulative share of comments")
        ax_lorenz.set_title(f"Lorenz curve (Gini = {gini:.2f})")
        ax_lorenz.grid(True, alpha=0.2)

    # Rank-size (Zipf)
    if show_zipf:
        ax_zipf = fig.add_subplot(grid[1, 1])
        ax_zipf.plot(ranks, ordered_counts, color="#1f77b4")
        ax_zipf.set_xscale("log")
        ax_zipf.set_yscale("log")
        ax_zipf.set_xlabel("Rank (1 = most commented)")
        ax_zipf.set_ylabel("Comment count")
        ax_zipf.set_title("Heavy-tail: a few blockbusters, many ignored")
        ax_zipf.grid(True, which="both", alpha=0.2)

    fig.suptitle(f"Federal Register comments concentration ({year})", fontsize=14, y=0.99)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Efficiency frontier of Federal Register comments",
        allow_abbrev=False,
    )
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.dirname(__file__),
        help="Directory containing input CSV and where outputs will be written",
    )
    parser.add_argument(
        "--all-channels",
        action="store_true",
        help="Include all submission channels (default is regs.gov only)",
    )
    parser.add_argument("--no-lorenz", action="store_true", help="Hide Lorenz subplot")
    parser.add_argument("--no-zipf", action="store_true", help="Hide Zipf subplot")
    parser.add_argument(
        "--annotations",
        choices=["on", "off"],
        default="on",
        help="Toggle on-figure annotations",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce console output")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df_all = load_data(args.output_dir, args.year)
    df = filter_channel(df_all, include_all_channels=args.all_channels)

    if not args.quiet:
        total_docs = len(df_all)
        used_docs = len(df)
        channel_msg = "all channels" if args.all_channels else "regs.gov only"
        print(
            f"Loaded {total_docs} rows; using {used_docs} for analysis ({channel_msg})."
        )

    counts = get_counts(df)
    x, y = compute_frontier(counts)
    lorenz_x, lorenz_y, gini = compute_lorenz_and_gini(counts)
    ranks, ordered_counts = compute_rank_size(counts)

    metrics = {
        "year": args.year,
        "docs_total": int(len(counts)),
        "comments_total": int(np.maximum(counts, 0).sum()),
        "gini": float(round(gini, 4)),
    }
    metrics.update(compute_threshold_metrics(x, y))

    # Save figure
    out_png = os.path.join(args.output_dir, f"efficiency_frontier_{args.year}.png")
    render_figure(
        out_png=out_png,
        x=x,
        y=y,
        lorenz_x=lorenz_x,
        lorenz_y=lorenz_y,
        gini=gini,
        ranks=ranks,
        ordered_counts=ordered_counts,
        year=args.year,
        include_all_channels=args.all_channels,
        show_lorenz=not args.no_lorenz,
        show_zipf=not args.no_zipf,
        annotations_on=(args.annotations == "on"),
    )

    # Save metrics
    out_json = os.path.join(args.output_dir, f"efficiency_frontier_{args.year}_metrics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    if not args.quiet:
        print("\nKey metrics:")
        print(json.dumps(metrics, indent=2))
        print(f"\nSaved figure → {out_png}")
        print(f"Saved metrics → {out_json}")


if __name__ == "__main__":
    main()

