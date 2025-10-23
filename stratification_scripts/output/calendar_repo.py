import argparse
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Palette: muted GitHub-like greens (low to high)
PALETTE = [
    "#ebedf0",  # very light grey-green (Less)
    "#9be9a8",
    "#40c463",
    "#30a14e",
    "#216e39",  # dark green (More)
]

# Overlay marker for days where >=50% documents have zero comments
ZERO_SHARE_MARK = "#1b1032"


@dataclass
class CalendarArgs:
    year: int
    csv_path: str
    out_path: str
    dpi: int


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
    ns = parser.parse_args()

    default_csv = os.path.join(
        os.path.dirname(__file__), f"federal_register_{ns.year}_comments.csv"
    )
    csv_path = ns.csv or default_csv

    default_out = os.path.join(os.path.dirname(__file__), f"calendar_{ns.year}.png")
    out_path = ns.out or default_out

    return CalendarArgs(year=int(ns.year), csv_path=csv_path, out_path=out_path, dpi=int(ns.dpi))


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


def build_github_grid(daily: pd.DataFrame, year: int) -> Tuple[np.ndarray, np.ndarray, pd.Timestamp, int]:
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
            zero_flag[weekday, week] = bool(daily.loc[day_date, "zero_share"] >= 0.5)  # type: ignore[index]
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


def plot_calendar(values: np.ndarray, zero_flag: np.ndarray, start: pd.Timestamp, num_weeks: int, out_path: str, dpi: int, year: int) -> None:
    # Figure sizing approximates GitHub proportions
    cell = 0.26  # inches per cell width
    width = max(8.0, num_weeks * cell + 2.0)
    height = 7 * cell + 1.2

    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)

    levels = bin_values(values)
    color_grid = np.empty(levels.shape + (3,), dtype=float)
    # Map levels to palette colors
    palette_rgb = np.array([plt.colors.to_rgb(c) if hasattr(plt, "colors") else plt.cm.get_cmap()(0) for c in PALETTE])
    # Fallback for older Matplotlib attribute
    try:
        from matplotlib.colors import to_rgb
        palette_rgb = np.array([to_rgb(c) for c in PALETTE])
    except Exception:
        pass
    for idx in range(5):
        color_grid[levels == idx] = palette_rgb[idx]

    ax.imshow(
        color_grid,
        aspect="equal",
        interpolation="none",
        origin="upper",  # row 0 (Monday) at the top
    )

    # Overlay markers where zero-share >= 0.5
    ys, xs = np.where(zero_flag)
    ax.scatter(xs, ys, s=9, c=ZERO_SHARE_MARK, marker="s", linewidths=0)

    # Ticks and labels: show Mon, Wed, Fri on the left only
    ax.set_xticks([])
    ax.set_yticks([0, 2, 4])
    ax.set_yticklabels(["Mon", "Wed", "Fri"], fontsize=9)

    # Outline and background styling
    ax.set_frame_on(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Month labels: only Jan on the first column and Dec on the last column
    ax.text(0, -0.9, "Jan", fontsize=10, weight="bold", ha="left", va="bottom")
    ax.text(num_weeks - 1, -0.9, "Dec", fontsize=10, weight="bold", ha="right", va="bottom")

    # Legend: Less <> More with 5 swatches
    legend_x = num_weeks - 11
    legend_y = 7.75
    sw = 0.7
    for i, color in enumerate(PALETTE):
        ax.add_patch(
            plt.Rectangle((legend_x + i * (sw + 0.15), legend_y), sw, 0.6, color=color, transform=ax.transData, clip_on=False)
        )
    ax.text(legend_x - 0.5, legend_y + 0.3, "Less", ha="right", va="center", fontsize=9)
    ax.text(legend_x + 5 * (sw + 0.15) + 0.1, legend_y + 0.3, "More", ha="left", va="center", fontsize=9)

    # Zero-share legend marker
    ax.add_patch(
        plt.Rectangle((legend_x + 6.5, legend_y), sw, 0.6, color=ZERO_SHARE_MARK, transform=ax.transData, clip_on=False)
    )
    ax.text(legend_x + 6.5 + sw + 0.2, legend_y + 0.3, "≥50% zero-comment day", ha="left", va="center", fontsize=9)

    # Title
    ax.text(0, -2.0, f"Daily total comments • {year}", fontsize=12, weight="bold", ha="left", va="bottom")

    # Tight layout without cropping annotations
    plt.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.08)

    # Save
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = resolve_args()
    daily = load_daily_metrics(args.csv_path, args.year)
    values, zero_flag, start, num_weeks = build_github_grid(daily, args.year)
    plot_calendar(values, zero_flag, start, num_weeks, args.out_path, args.dpi, args.year)
    print(f"Saved calendar heatmap → {args.out_path}")


if __name__ == "__main__":
    main()
