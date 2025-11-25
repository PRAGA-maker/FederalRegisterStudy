"""
Federal Register calendar visualizations showing the engagement problem.
Three modes:
  1. Overall engagement calendar (default): Shows % of docs with 10+ comments per day
  2. Agency comparison: Side-by-side comparison of high vs low engagement agencies
  3. Stacked view: Multiple agencies stacked to show aggregate vs individual patterns
"""
import argparse
import os
from dataclasses import dataclass
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# Green palette (GitHub-style)
PALETTE = [
    "#ebedf0",  # very light grey (0-10%)
    "#9be9a8",  # light green (10-25%)
    "#40c463",  # medium green (25-40%)
    "#30a14e",  # dark green (40-60%)
    "#216e39",  # darkest green (60%+)
]


@dataclass
class CalendarArgs:
    year: int
    csv_path: str
    out_path: str
    dpi: int
    mode: str
    show_title: bool
    high_agency: Optional[str]
    low_agency: Optional[str]
    comment_threshold: int


def resolve_args() -> CalendarArgs:
    parser = argparse.ArgumentParser(
        description="Federal Register engagement calendar visualizations",
        allow_abbrev=False,
    )
    default_year = int(os.environ.get("FR_YEAR", 2024))
    parser.add_argument("--year", type=int, default=default_year)
    parser.add_argument("--csv", type=str, default=None, help="Override input CSV path")
    parser.add_argument("--out", type=str, default=None, help="Output PNG path")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--mode",
        type=str,
        default="engagement",
        choices=["engagement", "comparison", "stacked"],
        help="Visualization mode: 'engagement' (default), 'comparison', or 'stacked'",
    )
    parser.add_argument(
        "--title",
        action="store_true",
        help="Show title header",
    )
    parser.add_argument("--high-agency", type=str, default=None, 
                       help="Agency name for high engagement (comparison mode)")
    parser.add_argument("--low-agency", type=str, default=None,
                       help="Agency name for low engagement (comparison mode)")
    parser.add_argument("--threshold", type=int, default=10,
                       help="Comment count threshold (default: 10)")
    ns = parser.parse_args()

    default_csv = os.path.join(
        os.path.dirname(__file__), f"federal_register_{ns.year}_comments.csv"
    )
    csv_path = ns.csv or default_csv

    # Auto-generate output path in makeup/{year}/ subdirectory
    makeup_dir = os.path.join(os.path.dirname(__file__), "makeup", str(ns.year))
    
    if ns.out:
        out_path = ns.out
    else:
        if ns.mode == "comparison":
            out_path = os.path.join(makeup_dir, f"agency_comparison.png")
        elif ns.mode == "stacked":
            out_path = os.path.join(makeup_dir, f"agency_stacked.png")
        else:
            out_path = os.path.join(makeup_dir, f"engagement_calendar.png")

    return CalendarArgs(
        year=int(ns.year),
        csv_path=csv_path,
        out_path=out_path,
        dpi=int(ns.dpi),
        mode=ns.mode,
        show_title=bool(ns.title),
        high_agency=ns.high_agency,
        low_agency=ns.low_agency,
        comment_threshold=int(ns.threshold),
    )


# ============================================================================
# MODE 1: Overall Engagement Calendar
# ============================================================================

def load_daily_engagement(csv_path: str, year: int, threshold: int = 10) -> pd.DataFrame:
    """Load and compute daily engagement rate (% of docs with threshold+ comments)."""
    if not os.path.exists(csv_path):
        raise SystemExit(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "publication_date" not in df.columns or "comment_count" not in df.columns:
        raise SystemExit(
            "CSV missing required columns 'publication_date' and 'comment_count'"
        )

    df["publication_date"] = pd.to_datetime(df["publication_date"], errors="coerce")
    df = df.dropna(subset=["publication_date"]).copy()
    df = df[df["publication_date"].dt.year == year]

    # Compute engagement rate: % of documents with comment_count >= threshold
    df["has_comments"] = (df["comment_count"].fillna(0).astype(float) >= threshold).astype(int)
    
    daily = df.groupby(df["publication_date"].dt.date).agg(
        num_docs=("document_number", "count"),
        docs_with_comments=("has_comments", "sum"),
        total_comments=("comment_count", "sum"),
    )
    daily["engagement_rate"] = (daily["docs_with_comments"] / daily["num_docs"] * 100).round(1)

    # Reindex to all days of the year for a full grid
    idx = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq="D").date
    daily = daily.reindex(idx, fill_value=0)
    # Set engagement_rate to NaN for days with no documents (weekends, holidays)
    daily.loc[daily["num_docs"] == 0, "engagement_rate"] = np.nan
    
    return daily


def _monday_on_or_before(d: pd.Timestamp) -> pd.Timestamp:
    weekday = d.weekday()
    return d - pd.Timedelta(days=weekday)


def _sunday_on_or_after(d: pd.Timestamp) -> pd.Timestamp:
    weekday = d.weekday()
    return d + pd.Timedelta(days=(6 - weekday))


def build_github_grid(
    daily: pd.DataFrame, year: int
) -> Tuple[np.ndarray, pd.Timestamp, int]:
    """Build a rectangular grid from first Monday to last Sunday covering the year."""
    jan1 = pd.Timestamp(year=year, month=1, day=1)
    dec31 = pd.Timestamp(year=year, month=12, day=31)
    start = _monday_on_or_before(jan1)
    end = _sunday_on_or_after(dec31)

    all_days = pd.date_range(start=start, end=end, freq="D")
    num_weeks = int(np.ceil(len(all_days) / 7.0))

    values = np.full((7, num_weeks), np.nan)

    # Map each day into grid position (weekday, week_index)
    for offset, day_ts in enumerate(all_days):
        week = offset // 7
        weekday = day_ts.weekday()
        day_date = day_ts.date()
        if day_date in daily.index:
            values[weekday, week] = daily.loc[day_date, "engagement_rate"]

    return values, start, num_weeks


def bin_engagement_values(values: np.ndarray) -> np.ndarray:
    """
    Bin engagement rates (0-100%) into 5 levels.
    Uses fixed thresholds that make sense for engagement:
    0: 0-10% engagement (almost nothing)
    1: 10-25% engagement (low)
    2: 25-40% engagement (below average)
    3: 40-60% engagement (average)
    4: 60%+ engagement (good, rare!)
    """
    thresholds = np.array([10, 25, 40, 60], dtype=float)
    binned = np.digitize(np.nan_to_num(values, nan=0.0), thresholds, right=False)
    binned = np.clip(binned, 0, 4)
    return binned.astype(int)


def plot_engagement_calendar(
    values: np.ndarray,
    start: pd.Timestamp,
    num_weeks: int,
    out_path: str,
    dpi: int,
    year: int,
    show_title: bool,
    threshold: int,
) -> None:
    """Plot the overall engagement calendar."""
    from matplotlib.colors import to_rgb
    
    palette_rgb = np.array([to_rgb(c) for c in PALETTE])
    
    # Figure sizing
    cell_in = 0.24
    gutter = 0.18
    width = max(8.0, num_weeks * cell_in + 1.4)
    height = 7 * cell_in + 1.2

    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    levels = bin_engagement_values(values)

    # Draw per-day squares
    nrows, ncols = levels.shape
    size = 1.0 - gutter
    offset = gutter / 2.0
    for r in range(nrows):
        for c in range(ncols):
            if np.isnan(values[r, c]):
                # No data (weekend/holiday) - show very light gray
                color = to_rgb("#f0f0f0")
            else:
                color = palette_rgb[levels[r, c]]
            rect = Rectangle(
                (c + offset, r + offset), size, size, linewidth=0, facecolor=color
            )
            ax.add_patch(rect)

    # Axes limits and orientation: Monday at top
    ax.set_xlim(0, ncols)
    ax.set_ylim(7, 0)
    ax.set_aspect("equal")

    # Ticks and labels
    ax.set_xticks([])
    ax.set_yticks([0.5, 2.5, 4.5])
    ax.set_yticklabels(["Mon", "Wed", "Fri"], fontsize=9)

    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Month labels
    ax.text(
        0.02, 1.03, "Jan", transform=ax.transAxes, fontsize=10,
        weight="bold", ha="left", va="bottom",
    )
    ax.text(
        0.98, 1.03, "Dec", transform=ax.transAxes, fontsize=10,
        weight="bold", ha="right", va="bottom",
    )

    # Title
    if show_title:
        ax.text(
            0.02, 1.12, f"Federal Register Engagement • {year}",
            transform=ax.transAxes, fontsize=11, weight="bold",
            ha="left", va="bottom",
        )
    
    # Main descriptive text
    ax.text(
        0.02, -0.02, f"% of documents that received at least {threshold} comments",
        transform=ax.transAxes, fontsize=10, ha="left", va="top",
        style="italic",
    )

    # Legend bottom-right
    base_x = 0.65
    base_y = -0.16
    sw_w = 0.018
    sw_h = 0.04
    
    for i, color in enumerate(PALETTE):
        ax.add_patch(
            Rectangle(
                (base_x + i * (sw_w + 0.008), base_y), sw_w, sw_h,
                transform=ax.transAxes, facecolor=color,
                edgecolor="none", clip_on=False,
            )
        )
    
    # Legend labels with actual percentages
    ax.text(
        base_x - 0.01, base_y + sw_h / 2, "0%",
        transform=ax.transAxes, ha="right", va="center", fontsize=9,
    )
    ax.text(
        base_x + 5 * (sw_w + 0.008) + 0.004, base_y + sw_h / 2, "60%+",
        transform=ax.transAxes, ha="left", va="center", fontsize=9,
    )

    plt.subplots_adjust(left=0.07, right=0.99, top=0.90, bottom=0.22)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    
    # Print summary stats
    flat = values[~np.isnan(values)]
    if flat.size > 0:
        print(f"\nEngagement Statistics for {year} (>={threshold} comments):")
        print(f"  Mean engagement: {np.mean(flat):.1f}%")
        print(f"  Median engagement: {np.median(flat):.1f}%")
        print(f"  Min: {np.min(flat):.1f}%  Max: {np.max(flat):.1f}%")
        print(f"  Days with <25% engagement: {np.sum(flat < 25)} / {len(flat)}")


# ============================================================================
# MODE 2: Agency Comparison
# ============================================================================

def find_representative_agencies(df: pd.DataFrame, min_docs: int = 50, threshold: int = 10) -> Tuple[str, str, pd.DataFrame]:
    """Find a high-engagement and low-engagement agency with sufficient documents."""
    df["has_comments"] = (df["comment_count"].fillna(0) >= threshold).astype(int)
    
    agency_stats = df.groupby("agency").agg(
        num_docs=("document_number", "count"),
        docs_with_comments=("has_comments", "sum"),
    )
    agency_stats["engagement_rate"] = (
        agency_stats["docs_with_comments"] / agency_stats["num_docs"] * 100
    )
    
    # Filter agencies with enough documents for meaningful visualization
    agency_stats = agency_stats[agency_stats["num_docs"] >= min_docs].sort_values(
        "engagement_rate", ascending=False
    )
    
    if len(agency_stats) < 2:
        raise SystemExit("Not enough agencies with sufficient documents for comparison")
    
    high_agency = agency_stats.index[0]
    low_agency = agency_stats.index[-1]
    
    return high_agency, low_agency, agency_stats


def load_agency_daily_engagement(
    csv_path: str, year: int, agency: str, threshold: int = 10
) -> Tuple[pd.DataFrame, float]:
    """Load daily engagement data for a specific agency."""
    df = pd.read_csv(csv_path)
    df["publication_date"] = pd.to_datetime(df["publication_date"], errors="coerce")
    df = df.dropna(subset=["publication_date"]).copy()
    df = df[df["publication_date"].dt.year == year]
    
    # Filter to specific agency
    df = df[df["agency"] == agency]
    
    if len(df) == 0:
        raise SystemExit(f"No documents found for agency: {agency}")

    # Compute engagement rate
    df["has_comments"] = (df["comment_count"].fillna(0) >= threshold).astype(int)
    
    daily = df.groupby(df["publication_date"].dt.date).agg(
        num_docs=("document_number", "count"),
        docs_with_comments=("has_comments", "sum"),
    )
    daily["engagement_rate"] = (daily["docs_with_comments"] / daily["num_docs"] * 100)
    
    # Overall stats
    overall_engagement = df["has_comments"].mean() * 100
    
    # Reindex to all days of the year
    idx = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq="D").date
    daily = daily.reindex(idx, fill_value=0)
    daily.loc[daily["num_docs"] == 0, "engagement_rate"] = np.nan
    
    return daily, overall_engagement


def plot_comparison_calendar(
    values_high: np.ndarray,
    values_low: np.ndarray,
    start: pd.Timestamp,
    num_weeks: int,
    high_agency: str,
    low_agency: str,
    high_engagement: float,
    low_engagement: float,
    out_path: str,
    dpi: int,
    year: int,
    threshold: int,
) -> None:
    """Plot side-by-side calendar comparison with better spacing."""
    from matplotlib.colors import to_rgb
    
    palette_rgb = np.array([to_rgb(c) for c in PALETTE])
    
    # Figure sizing - two calendars stacked vertically with more spacing
    cell_in = 0.24
    gutter = 0.18
    width = max(10.0, num_weeks * cell_in + 1.8)
    height = 2 * (7 * cell_in) + 3.2  # Increased spacing between calendars
    
    fig = plt.figure(figsize=(width, height), dpi=dpi)
    fig.patch.set_facecolor("white")
    
    # Create two subplots with more vertical spacing
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.65)
    
    for idx, (values, agency, engagement, title_prefix) in enumerate([
        (values_high, high_agency, high_engagement, "HIGH ENGAGEMENT"),
        (values_low, low_agency, low_engagement, "LOW ENGAGEMENT"),
    ]):
        ax = fig.add_subplot(gs[idx])
        ax.set_facecolor("white")
        
        levels = bin_engagement_values(values)
        
        # Draw squares
        nrows, ncols = levels.shape
        size = 1.0 - gutter
        offset = gutter / 2.0
        
        for r in range(nrows):
            for c in range(ncols):
                if np.isnan(values[r, c]):
                    color = to_rgb("#f0f0f0")
                else:
                    color = palette_rgb[levels[r, c]]
                rect = Rectangle(
                    (c + offset, r + offset), size, size, linewidth=0, facecolor=color
                )
                ax.add_patch(rect)
        
        # Axes setup
        ax.set_xlim(0, ncols)
        ax.set_ylim(7, 0)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([0.5, 2.5, 4.5])
        ax.set_yticklabels(["Mon", "Wed", "Fri"], fontsize=9)
        
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Month labels
        ax.text(0.02, 1.05, "Jan", transform=ax.transAxes, fontsize=10, 
                weight="bold", ha="left", va="bottom")
        ax.text(0.98, 1.05, "Dec", transform=ax.transAxes, fontsize=10,
                weight="bold", ha="right", va="bottom")
        
        # Title with agency name (truncate if too long) - increased spacing
        agency_short = agency if len(agency) < 75 else agency[:72] + "..."
        ax.text(0.02, 1.20, f"{title_prefix}: {engagement:.1f}%",
                transform=ax.transAxes, fontsize=11, weight="bold",
                ha="left", va="bottom")
        ax.text(0.02, 1.11, agency_short,
                transform=ax.transAxes, fontsize=9,
                ha="left", va="bottom", style="italic")
    
    # Overall title - adjusted positioning
    fig.text(0.5, 0.98, f"Federal Register Engagement Comparison • {year}",
             ha="center", va="top", fontsize=13, weight="bold")
    fig.text(0.5, 0.95, f"% of documents that received at least {threshold} comments",
             ha="center", va="top", fontsize=10, style="italic")
    
    # Legend at bottom
    ax_legend = fig.add_axes([0.35, 0.015, 0.3, 0.035])
    ax_legend.set_xlim(0, 1)
    ax_legend.set_ylim(0, 1)
    ax_legend.axis("off")
    
    sw_w = 0.12
    base_x = 0.15
    for i, color in enumerate(PALETTE):
        rect = Rectangle((base_x + i * (sw_w + 0.02), 0.2), sw_w, 0.6,
                        facecolor=color, edgecolor="none")
        ax_legend.add_patch(rect)
    
    ax_legend.text(0.05, 0.5, "0%", ha="right", va="center", fontsize=9)
    ax_legend.text(0.95, 0.5, "60%+", ha="left", va="center", fontsize=9)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# MODE 3: Stacked View (Multiple Agencies)
# ============================================================================

def plot_stacked_calendars(
    csv_path: str,
    year: int,
    out_path: str,
    dpi: int,
    threshold: int,
    num_agencies: int = 6,
) -> None:
    """
    Plot 3D isometric view of multiple agency calendars stacked vertically.
    Uses 1 comment threshold to show engagement sparsity.
    """
    from matplotlib.colors import to_rgb
    from matplotlib.patches import Polygon
    
    # Force threshold to 1 for this visualization
    threshold = 1
    
    # Load data
    df = pd.read_csv(csv_path)
    df["publication_date"] = pd.to_datetime(df["publication_date"], errors="coerce")
    df = df.dropna(subset=["publication_date"])
    df = df[df["publication_date"].dt.year == year]
    
    # Find top agencies by document count
    df["has_comments"] = (df["comment_count"].fillna(0) >= threshold).astype(int)
    agency_stats = df.groupby("agency").agg(
        num_docs=("document_number", "count"),
        docs_with_comments=("has_comments", "sum"),
        days_without_comments=("has_comments", lambda x: (x == 0).sum())
    )
    agency_stats["engagement_rate"] = (
        agency_stats["docs_with_comments"] / agency_stats["num_docs"] * 100
    )
    
    # Get diverse agencies by volume
    agency_stats = agency_stats[agency_stats["num_docs"] >= 50].sort_values("num_docs", ascending=False)
    selected_agencies = agency_stats.head(num_agencies).index.tolist()
    
    # Compute overall aggregate
    daily_all = load_daily_engagement(csv_path, year, threshold)
    values_all, start, num_weeks = build_github_grid(daily_all, year)
    
    palette_rgb = np.array([to_rgb(c) for c in PALETTE])
    
    # Isometric projection parameters
    iso_angle = np.radians(30)  # 30-degree isometric angle
    z_scale = 0.85  # Vertical spacing between layers
    cell_size = 0.15  # Base cell size
    x_offset = 4  # Center offset for better positioning
    
    # Figure setup
    fig, ax = plt.subplots(figsize=(16, 12), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_aspect("equal")
    ax.axis("off")
    
    # Isometric transformation function
    def iso_transform(x, y, z):
        """Convert 3D coordinates to 2D isometric projection"""
        iso_x = (x - y) * np.cos(iso_angle) + x_offset
        iso_y = (x + y) * np.sin(iso_angle) + z
        return iso_x, iso_y
    
    def draw_iso_rect(ax, x, y, z, width, height, color, alpha=1.0, edgecolor=None):
        """Draw a rectangle in isometric view"""
        # Define the 4 corners of the rectangle
        corners = np.array([
            [x, y, z],
            [x + width, y, z],
            [x + width, y + height, z],
            [x, y + height, z]
        ])
        # Transform to isometric
        iso_corners = np.array([iso_transform(c[0], c[1], c[2]) for c in corners])
        # Draw polygon
        poly = Polygon(iso_corners, facecolor=color, alpha=alpha, 
                      edgecolor=edgecolor if edgecolor else color, linewidth=0.5)
        ax.add_patch(poly)
        return iso_corners
    
    # Calculate grid dimensions
    nrows, ncols = 7, num_weeks
    
    # Store layer info for side annotations
    layer_info = []
    
    # Draw layers from bottom to top (agencies first, then aggregate)
    all_layers = list(enumerate(selected_agencies)) + [(num_agencies, "AGGREGATE")]
    
    for layer_idx, agency_or_label in all_layers:
        z = layer_idx * z_scale
        
        # Get data for this layer
        if agency_or_label == "AGGREGATE":
            values = values_all
            daily_data = daily_all
        else:
            agency = agency_or_label
            daily_data, _ = load_agency_daily_engagement(csv_path, year, agency, threshold)
            values, _, _ = build_github_grid(daily_data, year)
        
        # Calculate days without comments for this layer
        flat_vals = values[~np.isnan(values)]
        days_with_zero = np.sum(flat_vals == 0) if flat_vals.size > 0 else 0
        total_days = len(flat_vals) if flat_vals.size > 0 else 1
        
        # Bin the values
        levels = bin_engagement_values(values)
        
        # Determine alpha based on layer (lower layers more transparent)
        # Gradual transparency from bottom (0.4) to top (1.0)
        if agency_or_label == "AGGREGATE":
            base_alpha = 1.0
        else:
            # Gradual alpha increase from bottom layer to top
            base_alpha = 0.4 + (layer_idx / num_agencies) * 0.6
        
        # Draw calendar grid for this layer
        for r in range(nrows):
            for c in range(ncols):
                # Skip some cells for lower layers for partial visibility effect
                # But use gradual probability instead of hard cutoff
                if agency_or_label != "AGGREGATE":
                    # Lower layers show fewer cells (sparse sampling)
                    show_prob = 0.3 + (layer_idx / num_agencies) * 0.7
                    if (c + r * ncols) % 3 >= show_prob * 3:
                        continue
                
                # Get color
                if np.isnan(values[r, c]):
                    color = to_rgb("#f0f0f0")
                else:
                    color = palette_rgb[levels[r, c]]
                
                # Draw isometric cell with gradual alpha
                x = c * cell_size
                y = r * cell_size
                draw_iso_rect(ax, x, y, z, cell_size * 0.9, cell_size * 0.9, 
                            color, alpha=base_alpha)
        
        # Store layer info
        if agency_or_label == "AGGREGATE":
            layer_info.append({
                'z': z,
                'label': 'ALL AGENCIES',
                'days_without': days_with_zero,
                'total_days': total_days,
                'is_aggregate': True
            })
        else:
            agency_short = agency if len(agency) < 40 else agency[:37] + "..."
            layer_info.append({
                'z': z,
                'label': agency_short,
                'days_without': days_with_zero,
                'total_days': total_days,
                'is_aggregate': False
            })
    
    # Add agency labels on the side for each layer
    for info in layer_info:
        label_x, label_y = iso_transform(ncols * cell_size, nrows * cell_size / 2, info['z'])
        
        if info['is_aggregate']:
            ax.text(label_x + 0.5, label_y + 0.2, "ALL AGENCIES", 
                   fontsize=10, weight='bold', ha='left', va='center')
        else:
            # Truncate agency name if too long
            agency_name = info['label'] if len(info['label']) < 50 else info['label'][:47] + "..."
            ax.text(label_x + 0.5, label_y, agency_name, 
                   fontsize=7, ha='left', va='center', color='#555')
    
    # Add title at top
    ax.text(0.5, 0.97, f"Federal Register Engagement • {year}",
           transform=ax.transAxes, fontsize=16, weight='bold', ha='center', va='top')
    ax.text(0.5, 0.94, "% of documents that received at least 1 comment",
           transform=ax.transAxes, fontsize=10, ha='center', va='top')
    
    # Add month labels on bottom layer
    jan_x, jan_y = iso_transform(0, 0, -0.5)
    dec_x, dec_y = iso_transform(ncols * cell_size, 0, -0.5)
    ax.text(jan_x, jan_y, "Jan", fontsize=9, weight='bold', ha='left', va='top')
    ax.text(dec_x, dec_y, "Dec", fontsize=9, weight='bold', ha='right', va='top')
    
    # Legend - centered below the visualization
    legend_y = -1.8
    legend_x_start = x_offset + 1
    sw_size = 0.3
    for i, color in enumerate(PALETTE):
        rect_x = legend_x_start + i * (sw_size + 0.1)
        rect = Rectangle((rect_x, legend_y), sw_size, sw_size, 
                        facecolor=color, edgecolor='none')
        ax.add_patch(rect)
    
    ax.text(legend_x_start - 0.3, legend_y + sw_size/2, "0%", 
           fontsize=8, ha='right', va='center')
    ax.text(legend_x_start + 5 * (sw_size + 0.1) + 0.1, legend_y + sw_size/2, "60%+",
           fontsize=8, ha='left', va='center')
    
    # Set axis limits - centered around the visualization with space for labels
    max_x = x_offset + ncols * cell_size * np.cos(iso_angle) + 2
    ax.set_xlim(-1, max_x + 5)  # More space on right for agency names
    ax.set_ylim(-2.5, (num_agencies + 1) * z_scale + 2.5)
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    
    print(f"\nShowing {num_agencies} agencies (1+ comment threshold):")
    for agency in selected_agencies:
        stats = agency_stats.loc[agency]
        print(f"  {stats['days_without_comments']:.0f} days with 0 comments • {agency}")


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    args = resolve_args()
    
    if args.mode == "engagement":
        # Mode 1: Overall engagement calendar
        daily = load_daily_engagement(args.csv_path, args.year, args.comment_threshold)
        values, start, num_weeks = build_github_grid(daily, args.year)
        plot_engagement_calendar(
            values, start, num_weeks, args.out_path, args.dpi,
            args.year, args.show_title, args.comment_threshold
        )
        print(f"Saved engagement calendar -> {args.out_path}")
    
    elif args.mode == "comparison":
        # Mode 2: Agency comparison
        df_full = pd.read_csv(args.csv_path)
        df_full["publication_date"] = pd.to_datetime(df_full["publication_date"], errors="coerce")
        df_full = df_full.dropna(subset=["publication_date"])
        df_full = df_full[df_full["publication_date"].dt.year == args.year]
        
        # Auto-detect agencies if not specified
        if args.high_agency is None or args.low_agency is None:
            high_agency, low_agency, agency_stats = find_representative_agencies(
                df_full, threshold=args.comment_threshold
            )
            if args.high_agency is not None:
                high_agency = args.high_agency
            if args.low_agency is not None:
                low_agency = args.low_agency
            
            print(f"\nSelected agencies:")
            print(f"  HIGH: {high_agency}")
            print(f"  LOW:  {low_agency}")
            print(f"\nTop 5 agencies by engagement:")
            print(agency_stats.head(5))
            print(f"\nBottom 5 agencies by engagement:")
            print(agency_stats.tail(5))
        else:
            high_agency = args.high_agency
            low_agency = args.low_agency
        
        # Load daily data for each agency
        daily_high, engagement_high = load_agency_daily_engagement(
            args.csv_path, args.year, high_agency, args.comment_threshold
        )
        daily_low, engagement_low = load_agency_daily_engagement(
            args.csv_path, args.year, low_agency, args.comment_threshold
        )
        
        # Build grids
        values_high, start, num_weeks = build_github_grid(daily_high, args.year)
        values_low, _, _ = build_github_grid(daily_low, args.year)
        
        # Plot
        plot_comparison_calendar(
            values_high, values_low, start, num_weeks,
            high_agency, low_agency,
            engagement_high, engagement_low,
            args.out_path, args.dpi, args.year, args.comment_threshold,
        )
        
        print(f"\nSaved comparison calendar -> {args.out_path}")
        print(f"\nEngagement rates (>={args.comment_threshold} comments):")
        print(f"  {high_agency}: {engagement_high:.1f}%")
        print(f"  {low_agency}: {engagement_low:.1f}%")
        print(f"  Difference: {engagement_high - engagement_low:.1f} percentage points")
    
    elif args.mode == "stacked":
        # Mode 3: Stacked view showing aggregate vs individual patterns
        plot_stacked_calendars(
            args.csv_path, args.year, args.out_path, args.dpi,
            args.comment_threshold, num_agencies=6
        )
        print(f"\nSaved stacked calendar -> {args.out_path}")


if __name__ == "__main__":
    main()
