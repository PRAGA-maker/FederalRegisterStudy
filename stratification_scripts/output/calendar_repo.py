import argparse
import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# Palette: muted GitHub-like greens (low to high)
PALETTE = [
    "#ebedf0",  # very light grey-green (Less)
    "#9be9a8",
    "#40c463",
    "#30a14e",
    "#216e39",  # dark green (More)
]

# Overlay dot for days where zero-share exceeds threshold
ZERO_SHARE_MARK = "#111217"


@dataclass
class CalendarArgs:
    year: int
    csv_path: str
    out_path: str
    dpi: int
    zero_thresh: float
    no_title: bool


def resolve_args() -> CalendarArgs:
    parser = argparse.ArgumentParser(
        description="Static GitHub-style calendar heatmap of daily total comments",
        allow_abbrev=False,
    )
    default_year = int(os.environ.get("FR_YEAR", 2024))
    parser.add_argument("--year", type=int, default=default_year)
    parser.add_argument("--csv", type=str, default=None, help="Override input CSV path")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output PNG path (default: output/calendar_{year}.png)",
    )
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--zero-thresh",
        type=float,
        default=0.10,
        help="Zero-share threshold for dot overlay (e.g., 0.10 for 10%)",
    )
    parser.add_argument(
        "--no-title",
        action="store_true",
        help="Hide the small header; keep only Jan/Dec labels",
    )
    ns = parser.parse_args()

    default_csv = os.path.join(
        os.path.dirname(__file__), f"federal_register_{ns.year}_comments.csv"
    )
    csv_path = ns.csv or default_csv

    default_out = os.path.join(os.path.dirname(__file__), f"calendar_{ns.year}.png")
    out_path = ns.out or default_out

    return CalendarArgs(
        year=int(ns.year),
        csv_path=csv_path,
        out_path=out_path,
        dpi=int(ns.dpi),
        zero_thresh=float(ns.zero_thresh),
        no_title=bool(ns.no_title),
    )


def load_daily_metrics(csv_path: str, year: int) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise SystemExit(
            f"Input CSV not found: {csv_path}. Run 2024distribution.py first to generate it."
        )

    df = pd.read_csv(csv_path)
    if "publication_date" not in df.columns or "comment_count" not in df.columns:
        raise SystemExit(
            "CSV missing required columns 'publication_date' and 'comment_count'"
        )

    df["publication_date"] = pd.to_datetime(df["publication_date"], errors="coerce")
    df = df.dropna(subset=["publication_date"]).copy()
    df = df[df["publication_date"].dt.year == year]

    # Daily totals and zero-share metric
    df["is_zero"] = (df["comment_count"].fillna(0).astype(int) == 0).astype(int)
    daily_total = (
        df.groupby(df["publication_date"].dt.date)["comment_count"].sum().astype(int)
    )
    zero_share = df.groupby(df["publication_date"].dt.date)["is_zero"].mean()

    # Reindex to all days of the year for a full grid
    idx = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq="D").date
    daily = pd.DataFrame(
        {
            "total": daily_total.reindex(idx, fill_value=0),
            "zero_share": zero_share.reindex(idx).fillna(0.0),
        },
        index=pd.Index(idx, name="date"),
    )
    return daily


def _monday_on_or_before(d: pd.Timestamp) -> pd.Timestamp:
    weekday = (d.weekday() + 7) % 7  # Monday=0
    return d - pd.Timedelta(days=weekday)


def _sunday_on_or_after(d: pd.Timestamp) -> pd.Timestamp:
    weekday = (d.weekday() + 7) % 7
    return d + pd.Timedelta(days=(6 - weekday))


def build_github_grid(daily: pd.DataFrame, year: int, zero_thresh: float) -> Tuple[np.ndarray, np.ndarray, pd.Timestamp, int]:
    # Build a rectangular grid from first Monday to last Sunday covering the year
    jan1 = pd.Timestamp(year=year, month=1, day=1)
    dec31 = pd.Timestamp(year=year, month=12, day=31)
    start = _monday_on_or_before(jan1)
    end = _sunday_on_or_after(dec31)

    all_days = pd.date_range(start=start, end=end, freq="D")
    num_weeks = int(np.ceil(len(all_days) / 7.0))

    values = np.full((7, num_weeks), np.nan)
    zero_flag = np.zeros((7, num_weeks), dtype=bool)

    # Map each day into grid position (weekday, week_index)
    for offset, day_ts in enumerate(all_days):
        week = offset // 7
        weekday = (day_ts.weekday() + 7) % 7  # Monday=0..Sunday=6
        day_date = day_ts.date()
        if day_date in daily.index:
            values[weekday, week] = float(daily.loc[day_date, "total"])  # type: ignore[index]
            zero_flag[weekday, week] = bool(daily.loc[day_date, "zero_share"] >= zero_thresh)  # type: ignore[index]
        else:
            values[weekday, week] = 0.0
            zero_flag[weekday, week] = False

    return values, zero_flag, start, num_weeks


def bin_values(values: np.ndarray) -> np.ndarray:
    # Assign each value to 5 discrete levels using quantile thresholds
    flat = values[~np.isnan(values)]
    if flat.size == 0:
        return np.zeros_like(values, dtype=int)

    q25, q50, q75, q95 = np.quantile(flat, [0.25, 0.5, 0.75, 0.95])

    # Ensure strictly increasing thresholds; if degenerate, fall back to linear bins
    thresholds = np.array([q25, q50, q75, q95], dtype=float)
    if np.any(np.diff(np.concatenate(([0.0], np.unique(thresholds)))) == 0.0):
        vmin, vmax = float(np.min(flat)), float(np.max(flat))
        if vmax <= vmin:
            thresholds = np.array([0, 1, 2, 3], dtype=float)
        else:
            thresholds = np.linspace(vmin + (vmax - vmin) * 0.2, vmax, 4)

    binned = np.digitize(np.nan_to_num(values, nan=0.0), thresholds, right=True)
    binned = np.clip(binned, 0, 4)
    return binned.astype(int)


def plot_calendar(
    values: np.ndarray,
    zero_flag: np.ndarray,
    start: pd.Timestamp,
    num_weeks: int,
    out_path: str,
    dpi: int,
    year: int,
    show_title: bool,
) -> None:
    # Figure sizing (GitHub-like): small squares with even gutters
    cell_in = 0.24
    gutter = 0.18  # fraction of cell size used as gutter in data units
    width = max(8.0, num_weeks * cell_in + 1.4)
    height = 7 * cell_in + 0.9

    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    levels = bin_values(values)
    # Prepare palette
    from matplotlib.colors import to_rgb

    palette_rgb = np.array([to_rgb(c) for c in PALETTE])

    # Draw per-day squares as rectangles to control gutters
    nrows, ncols = levels.shape
    size = 1.0 - gutter
    offset = gutter / 2.0
    for r in range(nrows):
        for c in range(ncols):
            color = palette_rgb[levels[r, c]]
            rect = Rectangle((c + offset, r + offset), size, size, linewidth=0, facecolor=color)
            ax.add_patch(rect)

    # Overlay small centered black dot where zero_flag is true
    ys, xs = np.where(zero_flag)
    ax.scatter(xs + 0.5, ys + 0.5, s=8, c=ZERO_SHARE_MARK, marker="o", linewidths=0, zorder=3)

    # Axes limits and orientation: Monday at top
    ax.set_xlim(0, ncols)
    ax.set_ylim(7, 0)
    ax.set_aspect("equal")

    # Ticks and labels: Mon, Wed, Fri only (left)
    ax.set_xticks([])
    ax.set_yticks([0.5, 2.5, 4.5])
    ax.set_yticklabels(["Mon", "Wed", "Fri"], fontsize=9)

    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Month labels: Jan left, Dec right (axes coords for robust placement)
    ax.text(0.02, 1.03, "Jan", transform=ax.transAxes, fontsize=10, weight="bold", ha="left", va="bottom")
    ax.text(0.98, 1.03, "Dec", transform=ax.transAxes, fontsize=10, weight="bold", ha="right", va="bottom")

    # Optional tiny header (off by default per --no-title)
    if not show_title:
        pass
    else:
        ax.text(0.02, 1.12, f"Daily total comments • {year}", transform=ax.transAxes, fontsize=11, weight="bold", ha="left", va="bottom")

    # Legend bottom-right in axes coordinates
    # Swatch sizes tuned to visually match GitHub legend
    base_x = 0.78
    base_y = -0.12
    sw_w = 0.018
    sw_h = 0.04
    for i, color in enumerate(PALETTE):
        ax.add_patch(
            Rectangle((base_x + i * (sw_w + 0.008), base_y), sw_w, sw_h, transform=ax.transAxes, facecolor=color, edgecolor="none", clip_on=False)
        )
    ax.text(base_x - 0.01, base_y + sw_h / 2, "Less", transform=ax.transAxes, ha="right", va="center", fontsize=9)
    ax.text(base_x + 5 * (sw_w + 0.008) + 0.004, base_y + sw_h / 2, "More", transform=ax.transAxes, ha="left", va="center", fontsize=9)

    # Dot legend
    dot_x = base_x + 5 * (sw_w + 0.008) + 0.10
    dot_y = base_y + sw_h / 2
    ax.scatter([dot_x], [dot_y], transform=ax.transAxes, s=14, c=ZERO_SHARE_MARK, marker="o", clip_on=False)
    ax.text(dot_x + 0.02, dot_y, "≥10% zero-comment day", transform=ax.transAxes, ha="left", va="center", fontsize=9)

    plt.subplots_adjust(left=0.07, right=0.99, top=0.90, bottom=0.20)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = resolve_args()
    daily = load_daily_metrics(args.csv_path, args.year)
    values, zero_flag, start, num_weeks = build_github_grid(daily, args.year, args.zero_thresh)
    plot_calendar(
        values,
        zero_flag,
        start,
        num_weeks,
        args.out_path,
        args.dpi,
        args.year,
        show_title=not args.no_title,
    )
    print(f"Saved calendar heatmap → {args.out_path}")


if __name__ == "__main__":
    main()
