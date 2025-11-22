"""
Federal Register Comment Makeup Visualization Suite

Generates policy-brief visualizations showing who participates in federal rulemaking.
Produces both simple 3-chart sets and complex continuum pages with ribbon analysis.

Optional dependency: adjustText (pip install adjusttext) for better label positioning
"""

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patheffects import withStroke
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ============================================================================
# CONSTANTS & CONFIGURATION
# ============================================================================

CATEGORY_ORDER = [
    "Undecided/Anonymous",
    "Ordinary Citizen",
    "Large Organization/Corporation",
    "Academic/Industry/Expert (incl. small/local business)",
    "Political Consultant/Lobbyist",
]

CATEGORY_MAPPING = {
    "Undecided/Anonymous": "Undecided/Anonymous",
    "undecided": "Undecided/Anonymous",
    "Ordinary Citizen": "Ordinary Citizen",
    "citizen": "Ordinary Citizen",
    "Organization/Corporation": "Large Organization/Corporation",
    "organization": "Large Organization/Corporation",
    "corporation": "Large Organization/Corporation",
    "Academic/Industry/Expert (incl. small/local business)": "Academic/Industry/Expert (incl. small/local business)",
    "expert": "Academic/Industry/Expert (incl. small/local business)",
    "academic": "Academic/Industry/Expert (incl. small/local business)",
    "Political Consultant/Lobbyist": "Political Consultant/Lobbyist",
    "lobbyist": "Political Consultant/Lobbyist",
    "consultant": "Political Consultant/Lobbyist",
}

COLOR_PALETTE = {
    "Undecided/Anonymous": "#DDE3EA",  # paper
    "Ordinary Citizen": "#6B7D6D",      # sage
    "Large Organization/Corporation": "#12161A",  # ink
    "Academic/Industry/Expert (incl. small/local business)": "#9A8153",  # ochre
    "Political Consultant/Lobbyist": "#6D7581",  # slate
}

# Design tokens
FONT_TITLE = 18
FONT_LABEL = 13
FONT_LEGEND = 12
GRID_COLOR = "#CCCCCC"
GRID_ALPHA = 0.3


# ============================================================================
# DATA PREPROCESSING
# ============================================================================

def normalize_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Map categories to standardized names."""
    df = df.copy()
    if "category" in df.columns:
        df["category"] = df["category"].map(CATEGORY_MAPPING).fillna(df["category"])
        # Filter to only known categories
        df = df[df["category"].isin(CATEGORY_ORDER)]
    return df


def explode_agencies(df: pd.DataFrame) -> pd.DataFrame:
    """Split multi-agency entries into separate rows."""
    df = df.copy()
    if "agency" not in df.columns or df["agency"].isna().all():
        df["agency"] = "Unknown"
        return df
    
    # Split on comma and expand
    df["agency"] = df["agency"].fillna("Unknown")
    df["agency_list"] = df["agency"].str.split(",")
    df = df.explode("agency_list")
    df["agency"] = df["agency_list"].str.strip()
    df = df.drop(columns=["agency_list"])
    
    # Filter out empty agencies
    df = df[df["agency"].notna() & (df["agency"] != "")]
    return df


def deduplicate_comments(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate comments by comment_id."""
    if "comment_id" in df.columns:
        df = df.drop_duplicates(subset=["comment_id"])
    return df


def compute_agency_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate agency-level metrics for compass positioning.
    Returns DataFrame with columns: agency, total, X, Y, plus share/count columns.
    """
    # Group by agency and category
    grp = df.groupby(["agency", "category"])["comment_id"].count().rename("n").reset_index()
    totals = grp.groupby("agency")["n"].sum().rename("total")
    merged = grp.merge(totals, on="agency", how="left")
    merged["share"] = merged["n"] / merged["total"].where(merged["total"] > 0, 1)
    
    # Pivot to wide format
    wide = merged.pivot(index="agency", columns="category", values=["share", "n"]).fillna(0.0)
    
    # Flatten column names
    wide.columns = ["_".join(col).strip() for col in wide.columns.values]
    wide = wide.reset_index()
    
    # Add total column
    wide = wide.merge(totals, on="agency", how="left")
    
    # Ensure all category share/count columns exist
    for cat in CATEGORY_ORDER:
        share_col = f"share_{cat}"
        count_col = f"n_{cat}"
        if share_col not in wide.columns:
            wide[share_col] = 0.0
        if count_col not in wide.columns:
            wide[count_col] = 0
    
    # Compute compass coordinates
    oc_share = wide.get(f"share_Ordinary Citizen", 0.0)
    org_share = wide.get(f"share_Large Organization/Corporation", 0.0)
    exp_share = wide.get(f"share_Academic/Industry/Expert (incl. small/local business)", 0.0)
    lob_share = wide.get(f"share_Political Consultant/Lobbyist", 0.0)
    
    wide["X"] = (oc_share - org_share).clip(-1, 1)
    wide["Y"] = (exp_share + lob_share).clip(0, 1)
    
    return wide


# ============================================================================
# WEIGHTED STATISTICS
# ============================================================================

def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Calculate weighted quantile."""
    if len(values) == 0:
        return np.nan
    
    values = np.array(values)
    weights = np.array(weights)
    
    # Sort by values
    idx = np.argsort(values)
    values = values[idx]
    weights = weights[idx]
    
    # Cumulative weights
    cum_weights = np.cumsum(weights)
    total_weight = cum_weights[-1]
    
    if total_weight == 0:
        return np.nan
    
    # Find quantile
    target = q * total_weight
    idx = np.searchsorted(cum_weights, target)
    
    if idx >= len(values):
        return values[-1]
    return values[idx]


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Calculate weighted median."""
    return weighted_quantile(values, weights, 0.5)


def compute_ribbon_band(df_metrics: pd.DataFrame, n_bins: int = 30) -> pd.DataFrame:
    """
    Compute weighted median and IQR band across X continuum.
    Returns DataFrame with columns: x_center, y_median, y_q25, y_q75
    """
    if df_metrics.empty or "X" not in df_metrics.columns or "Y" not in df_metrics.columns:
        return pd.DataFrame(columns=["x_center", "y_median", "y_q25", "y_q75"])
    
    # Create bins
    x_min, x_max = df_metrics["X"].min(), df_metrics["X"].max()
    if x_min == x_max:
        return pd.DataFrame(columns=["x_center", "y_median", "y_q25", "y_q75"])
    
    bins = np.linspace(x_min, x_max, n_bins + 1)
    df_metrics["x_bin"] = pd.cut(df_metrics["X"], bins=bins, include_lowest=True)
    
    ribbon_data = []
    for bin_label, group in df_metrics.groupby("x_bin", observed=True):
        if len(group) == 0:
            continue
        
        x_center = group["X"].mean()
        weights = group["total"].values
        y_values = group["Y"].values
        
        y_median = weighted_median(y_values, weights)
        y_q25 = weighted_quantile(y_values, weights, 0.25)
        y_q75 = weighted_quantile(y_values, weights, 0.75)
        
        ribbon_data.append({
            "x_center": x_center,
            "y_median": y_median,
            "y_q25": y_q25,
            "y_q75": y_q75,
        })
    
    return pd.DataFrame(ribbon_data).sort_values("x_center")


# ============================================================================
# STYLING UTILITIES
# ============================================================================

def apply_clean_style(ax: plt.Axes) -> None:
    """Apply consistent clean styling to axis."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID_COLOR)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.tick_params(labelsize=FONT_LABEL, colors="#5C6670")
    ax.grid(True, alpha=GRID_ALPHA, color=GRID_COLOR, linestyle="-", linewidth=0.8)


def get_category_colors() -> List[str]:
    """Get colors in category order."""
    return [COLOR_PALETTE[cat] for cat in CATEGORY_ORDER]


# ============================================================================
# SIMPLE CHART SET
# ============================================================================

def plot_composition_donut(df: pd.DataFrame, year: int, outdir: Path) -> None:
    """
    Chart 1: Clean donut chart of comment composition.
    """
    counts = df["category"].value_counts().reindex(CATEGORY_ORDER, fill_value=0)
    if counts.sum() == 0:
        print(f"Warning: No data for donut chart ({year})")
        return
    
    percentages = 100 * counts / counts.sum()
    colors = get_category_colors()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create donut
    wedges, texts = ax.pie(
        counts.values,
        colors=colors,
        startangle=90,
        wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2)
    )
    
    # Title
    ax.set_title(f"Who commented in {year}?", fontsize=FONT_TITLE, fontweight="bold", pad=20)
    
    # Legend with percentages
    legend_labels = [f"{cat}: {pct:.1f}%" for cat, pct in zip(CATEGORY_ORDER, percentages)]
    ax.legend(
        legend_labels,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        fontsize=FONT_LEGEND,
        frameon=False
    )
    
    plt.tight_layout()
    fig.savefig(outdir / f"comment_makeup_{year}_donut.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: comment_makeup_{year}_donut.png")


def plot_agency_compass(df: pd.DataFrame, year: int, outdir: Path, top_n: int = 25) -> None:
    """
    Chart 2: Agency compass with quadrants.
    X = OC - ORG, Y = EXP + LOB
    """
    df_metrics = compute_agency_metrics(df)
    
    if df_metrics.empty:
        print(f"Warning: No agency metrics for compass ({year})")
        return
    
    # Identify top agencies
    top_agencies = df_metrics.nlargest(top_n, "total")["agency"].values
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Scatter all agencies (bubble size = comment volume)
    ax.scatter(
        df_metrics["X"],
        df_metrics["Y"],
        s=df_metrics["total"] / df_metrics["total"].max() * 400 + 30,
        alpha=0.4,
        color="#5C6670",
        edgecolors="white",
        linewidths=0.5
    )
    
    # Prepare labels for top agencies (using adjustText if available)
    try:
        from adjustText import adjust_text
        texts = []
        for _, row in df_metrics[df_metrics["agency"].isin(top_agencies)].iterrows():
            label = str(row["agency"])[:40]
            text = ax.text(
                row["X"], 
                row["Y"], 
                label,
                fontsize=9,
                alpha=0.85,
                ha="center",
                path_effects=[withStroke(linewidth=3, foreground="white")]
            )
            texts.append(text)
        
        # Adjust text positions to avoid overlap
        adjust_text(
            texts, 
            ax=ax,
            arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.5),
            expand_points=(1.2, 1.3),
            force_text=(0.5, 0.75)
        )
    except ImportError:
        # Fallback to basic annotation if adjustText not available
        for _, row in df_metrics[df_metrics["agency"].isin(top_agencies)].iterrows():
            label = str(row["agency"])[:40]
            ax.annotate(
                label,
                (row["X"], row["Y"]),
                fontsize=9,
                alpha=0.85,
                ha="center",
                path_effects=[withStroke(linewidth=3, foreground="white")]
            )
    
    # Quadrant grid
    ax.axhline(0.5, color=GRID_COLOR, linewidth=1.5, alpha=0.6, zorder=0)
    ax.axvline(0, color=GRID_COLOR, linewidth=1.5, alpha=0.6, zorder=0)
    
    # Quadrant labels (consistent format)
    label_props = dict(fontsize=11, alpha=0.6, style="italic", weight="bold")
    ax.text(0.5, 0.75, "High Citizen\nHigh Professional", ha="center", va="center", **label_props)
    ax.text(-0.5, 0.75, "High Corporate\nHigh Professional", ha="center", va="center", **label_props)
    ax.text(0.5, 0.25, "High Citizen\nLow Professional", ha="center", va="center", **label_props)
    ax.text(-0.5, 0.25, "High Corporate\nLow Professional", ha="center", va="center", **label_props)
    
    # Styling
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Corporate ← → Citizen", fontsize=FONT_LABEL, fontweight="bold")
    ax.set_ylabel("Low ← → High Professional Input", fontsize=FONT_LABEL, fontweight="bold")
    ax.set_title(f"Agency Positioning ({year})", fontsize=FONT_TITLE, fontweight="bold", pad=20)
    
    # Add note about bubble size
    ax.text(
        0.02, 0.98, 
        "Bubble size = comment volume",
        transform=ax.transAxes,
        fontsize=9,
        alpha=0.6,
        va="top",
        style="italic"
    )
    
    apply_clean_style(ax)
    
    plt.tight_layout()
    fig.savefig(outdir / f"agency_compass_{year}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: agency_compass_{year}.png")


def plot_workload_vs_makeup(df: pd.DataFrame, year: int, outdir: Path) -> None:
    """
    Chart 3: Workload vs makeup with multi-series scatter and trend lines.
    Shows total comments vs. counts by category with log-binned median lines.
    """
    df_metrics = compute_agency_metrics(df)
    
    if df_metrics.empty:
        print(f"Warning: No agency metrics for workload chart ({year})")
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # For each category, plot scatter and trend line
    for i, cat in enumerate(CATEGORY_ORDER):
        count_col = f"n_{cat}"
        if count_col not in df_metrics.columns:
            continue
        
        color = COLOR_PALETTE[cat]
        
        # Scatter background
        ax.scatter(
            df_metrics["total"],
            df_metrics[count_col],
            s=30,
            alpha=0.2,
            color=color,
            edgecolors="none"
        )
        
        # Compute log-binned medians for trend line
        df_metrics["log_total"] = np.log10(df_metrics["total"].clip(lower=1))
        log_min, log_max = df_metrics["log_total"].min(), df_metrics["log_total"].max()
        
        if log_min < log_max:
            n_bins = 10
            bins = np.linspace(log_min, log_max, n_bins + 1)
            df_metrics["log_bin"] = pd.cut(df_metrics["log_total"], bins=bins, include_lowest=True)
            
            trend_x, trend_y = [], []
            for bin_label, group in df_metrics.groupby("log_bin", observed=True):
                if len(group) >= 2:
                    trend_x.append(group["total"].median())
                    trend_y.append(group[count_col].median())
            
            if len(trend_x) > 1:
                ax.plot(trend_x, trend_y, color=color, linewidth=2.5, alpha=0.9, label=cat)
    
    # Styling
    ax.set_xscale("log")
    ax.set_xlabel("Total Comments (log scale)", fontsize=FONT_LABEL, fontweight="bold")
    ax.set_ylabel("Count by Category", fontsize=FONT_LABEL, fontweight="bold")
    ax.set_title(f"Workload vs. Who Shows Up ({year})", fontsize=FONT_TITLE, fontweight="bold", pad=20)
    
    ax.legend(fontsize=FONT_LEGEND - 1, frameon=True, loc="upper left", framealpha=0.9)
    apply_clean_style(ax)
    
    plt.tight_layout()
    fig.savefig(outdir / f"workload_vs_makeup_{year}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: workload_vs_makeup_{year}.png")


# ============================================================================
# COMPLEX CONTINUUM PAGE
# ============================================================================

def plot_compass_with_ribbon(df: pd.DataFrame, year: int, outdir: Path, top_n: int = 12) -> None:
    """
    Panel A: Compass with weighted median ribbon and IQR band.
    """
    df_metrics = compute_agency_metrics(df)
    
    if df_metrics.empty:
        print(f"Warning: No agency metrics for ribbon compass ({year})")
        return
    
    # Compute ribbon
    ribbon = compute_ribbon_band(df_metrics, n_bins=30)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Draw IQR band
    if not ribbon.empty and len(ribbon) > 2:
        ax.fill_between(
            ribbon["x_center"],
            ribbon["y_q25"],
            ribbon["y_q75"],
            alpha=0.2,
            color="#2E6F88",
            label="IQR band"
        )
        ax.plot(
            ribbon["x_center"],
            ribbon["y_median"],
            color="#2E6F88",
            linewidth=3,
            alpha=0.8,
            label="Weighted median"
        )
    
    # Scatter all agencies (faint)
    ax.scatter(
        df_metrics["X"],
        df_metrics["Y"],
        s=df_metrics["total"] / df_metrics["total"].max() * 150 + 15,
        alpha=0.15,
        color="#5C6670",
        edgecolors="none"
    )
    
    # Highlight top agencies and extremes
    top_agencies = df_metrics.nlargest(top_n, "total")
    extremes_x = pd.concat([
        df_metrics.nsmallest(2, "X"),
        df_metrics.nlargest(2, "X")
    ])
    extremes_y = pd.concat([
        df_metrics.nsmallest(2, "Y"),
        df_metrics.nlargest(2, "Y")
    ])
    highlights = pd.concat([top_agencies, extremes_x, extremes_y]).drop_duplicates(subset=["agency"])
    
    # Plot highlights
    ax.scatter(
        highlights["X"],
        highlights["Y"],
        s=highlights["total"] / highlights["total"].max() * 300 + 50,
        alpha=0.7,
        color="#9A8153",
        edgecolors="white",
        linewidths=2,
        zorder=10
    )
    
    # Label highlights
    for _, row in highlights.iterrows():
        label = str(row["agency"])[:35]
        ax.annotate(
            label,
            (row["X"], row["Y"]),
            fontsize=9,
            alpha=0.9,
            ha="center",
            path_effects=[withStroke(linewidth=4, foreground="white")],
            zorder=11
        )
    
    # Quadrant shading
    ax.axhspan(0.5, 1.05, -1.05, 0, alpha=0.03, color="#12161A", zorder=0)
    ax.axhspan(0.5, 1.05, 0, 1.05, alpha=0.03, color="#6B7D6D", zorder=0)
    ax.axhspan(-0.05, 0.5, -1.05, 0, alpha=0.03, color="#12161A", zorder=0)
    ax.axhspan(-0.05, 0.5, 0, 1.05, alpha=0.03, color="#6B7D6D", zorder=0)
    
    # Quadrant grid
    ax.axhline(0.5, color=GRID_COLOR, linewidth=1.5, alpha=0.5, zorder=1)
    ax.axvline(0, color=GRID_COLOR, linewidth=1.5, alpha=0.5, zorder=1)
    
    # Quadrant labels
    label_props = dict(fontsize=12, alpha=0.5, style="italic", weight="bold")
    ax.text(0.5, 0.85, "High Citizen\nHigh Professional", ha="center", va="center", **label_props)
    ax.text(-0.5, 0.85, "High Corporate\nHigh Professional", ha="center", va="center", **label_props)
    ax.text(0.5, 0.15, "High Citizen\nLow Professional", ha="center", va="center", **label_props)
    ax.text(-0.5, 0.15, "High Corporate\nLow Professional", ha="center", va="center", **label_props)
    
    # Arrow annotation
    ax.annotate(
        "",
        xy=(0.8, -0.02),
        xytext=(-0.8, -0.02),
        arrowprops=dict(arrowstyle="->", lw=2, color="#5C6670", alpha=0.4)
    )
    ax.text(0, -0.02, "Corporate → Citizen", ha="center", va="top", fontsize=10, alpha=0.5)
    
    # Styling
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Corporate ← → Citizen", fontsize=FONT_LABEL, fontweight="bold")
    ax.set_ylabel("Low ← → High Professional Input", fontsize=FONT_LABEL, fontweight="bold")
    ax.set_title(
        f"Agency Continuum with Weighted Ribbon ({year})",
        fontsize=FONT_TITLE,
        fontweight="bold",
        pad=20
    )
    
    ax.legend(fontsize=FONT_LEGEND, frameon=True, loc="upper right", framealpha=0.9)
    apply_clean_style(ax)
    
    plt.tight_layout()
    fig.savefig(outdir / f"agency_compass_ribbon_{year}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: agency_compass_ribbon_{year}.png")


def plot_ranked_stream(df: pd.DataFrame, year: int, outdir: Path) -> None:
    """
    Panel B: Ranked stream graph showing composition across corporate→grassroots continuum.
    """
    df_metrics = compute_agency_metrics(df)
    
    if df_metrics.empty:
        print(f"Warning: No agency metrics for stream graph ({year})")
        return
    
    # Sort by X (corporate → grassroots)
    df_sorted = df_metrics.sort_values("X").reset_index(drop=True)
    
    # Compute cumulative volume
    df_sorted["cum_volume"] = df_sorted["total"].cumsum() / df_sorted["total"].sum()
    df_sorted["cum_volume_prev"] = df_sorted["cum_volume"].shift(1, fill_value=0)
    df_sorted["cum_volume_center"] = (df_sorted["cum_volume"] + df_sorted["cum_volume_prev"]) / 2
    
    # Prepare shares for each category
    shares_data = []
    for cat in CATEGORY_ORDER:
        share_col = f"share_{cat}"
        if share_col in df_sorted.columns:
            shares_data.append(df_sorted[share_col].values)
        else:
            shares_data.append(np.zeros(len(df_sorted)))
    
    shares = np.array(shares_data)
    
    # Apply moving average smoothing (window = 5-10% of agencies)
    window_size = max(3, int(len(df_sorted) * 0.07))
    if window_size % 2 == 0:
        window_size += 1
    
    from scipy.ndimage import uniform_filter1d
    shares_smooth = np.array([uniform_filter1d(row, size=window_size, mode="nearest") for row in shares])
    
    # Normalize to 100% at each position
    shares_smooth_sum = shares_smooth.sum(axis=0, keepdims=True)
    shares_smooth_sum[shares_smooth_sum == 0] = 1
    shares_norm = shares_smooth / shares_smooth_sum
    
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Stacked area plot
    colors = get_category_colors()
    ax.stackplot(
        df_sorted["cum_volume_center"],
        *shares_norm,
        labels=CATEGORY_ORDER,
        colors=colors,
        alpha=0.85
    )
    
    # Mark key agencies
    top_3_indices = df_sorted.nlargest(3, "total").index.values
    for idx in top_3_indices:
        if idx < len(df_sorted):
            x_pos = df_sorted.loc[idx, "cum_volume_center"]
            ax.axvline(x_pos, color="#12161A", linewidth=1, alpha=0.3, linestyle="--")
            ax.text(
                x_pos,
                1.02,
                str(df_sorted.loc[idx, "agency"])[:20],
                rotation=45,
                fontsize=9,
                alpha=0.7,
                ha="left"
            )
    
    # Styling
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Cumulative % of Total Comments (Corporate → Citizen agencies)", fontsize=FONT_LABEL, fontweight="bold")
    ax.set_ylabel("Comment Composition", fontsize=FONT_LABEL, fontweight="bold")
    ax.set_title(
        f"How Comment Makeup Changes Across Agencies ({year})",
        fontsize=FONT_TITLE,
        fontweight="bold",
        pad=20
    )
    
    # Add subtitle explanation
    ax.text(
        0.5, 1.08,
        "Agencies sorted from most corporate-dominated (left) to most citizen-dominated (right)",
        transform=ax.transAxes,
        ha="center",
        fontsize=10,
        alpha=0.6,
        style="italic"
    )
    
    ax.legend(fontsize=FONT_LEGEND - 1, frameon=True, loc="upper left", bbox_to_anchor=(1, 1), framealpha=0.9)
    apply_clean_style(ax)
    
    plt.tight_layout()
    fig.savefig(outdir / f"ranked_stream_composition_{year}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: ranked_stream_composition_{year}.png")


def plot_spotlight_strip(df: pd.DataFrame, year: int, outdir: Path) -> None:
    """
    Panel C: Spotlight strip showing agencies with extreme values.
    """
    df_metrics = compute_agency_metrics(df)
    
    if df_metrics.empty:
        print(f"Warning: No agency metrics for spotlight strip ({year})")
        return
    
    # Find extremes
    max_oc_count = df_metrics.loc[df_metrics[f"n_Ordinary Citizen"].idxmax()]
    max_exp_pct = df_metrics.loc[df_metrics[f"share_Academic/Industry/Expert (incl. small/local business)"].idxmax()]
    max_lob_pct = df_metrics.loc[df_metrics[f"share_Political Consultant/Lobbyist"].idxmax()]
    max_org_pct = df_metrics.loc[df_metrics[f"share_Large Organization/Corporation"].idxmax()]
    
    extremes = [
        ("Most OC (count)", max_oc_count, f"{int(max_oc_count['n_Ordinary Citizen']):,}"),
        ("Highest % Expert", max_exp_pct, f"{100*max_exp_pct['share_Academic/Industry/Expert (incl. small/local business)']:.1f}%"),
        ("Highest % Lobbyist", max_lob_pct, f"{100*max_lob_pct['share_Political Consultant/Lobbyist']:.1f}%"),
        ("Highest % Corporate", max_org_pct, f"{100*max_org_pct['share_Large Organization/Corporation']:.1f}%"),
    ]
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 6))
    
    for ax, (label, row, metric) in zip(axes, extremes):
        # Get shares for this agency
        shares = [row[f"share_{cat}"] for cat in CATEGORY_ORDER]
        colors = get_category_colors()
        
        # Stacked bar
        left = 0
        for share, color in zip(shares, colors):
            ax.barh(0, share, left=left, color=color, height=0.6, edgecolor="white", linewidth=2)
            left += share
        
        # Labels
        agency_name = str(row["agency"])[:40]
        ax.text(-0.02, 0, f"{label}\n{agency_name}", ha="right", va="center", fontsize=11, fontweight="bold")
        ax.text(1.02, 0, metric, ha="left", va="center", fontsize=11, fontweight="bold", color="#2E6F88")
        
        # Styling
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, 0.5)
        ax.axis("off")
    
    fig.suptitle(f"Spotlight: Extreme Agencies ({year})", fontsize=FONT_TITLE, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outdir / f"spotlight_extremes_{year}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: spotlight_extremes_{year}.png")


def plot_continuum_page(df: pd.DataFrame, year: int, outdir: Path) -> None:
    """
    Combined continuum page with all three panels.
    """
    df_metrics = compute_agency_metrics(df)
    
    if df_metrics.empty:
        print(f"Warning: No agency metrics for continuum page ({year})")
        return
    
    # Create figure with custom gridspec
    fig = plt.figure(figsize=(16, 20))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.7, 0.25, 0.05], hspace=0.3)
    
    # Panel A: Compass with ribbon
    ax1 = fig.add_subplot(gs[0])
    
    ribbon = compute_ribbon_band(df_metrics, n_bins=30)
    
    if not ribbon.empty and len(ribbon) > 2:
        ax1.fill_between(ribbon["x_center"], ribbon["y_q25"], ribbon["y_q75"], alpha=0.2, color="#2E6F88")
        ax1.plot(ribbon["x_center"], ribbon["y_median"], color="#2E6F88", linewidth=3, alpha=0.8)
    
    ax1.scatter(df_metrics["X"], df_metrics["Y"], s=50, alpha=0.15, color="#5C6670")
    
    top_12 = df_metrics.nlargest(12, "total")
    ax1.scatter(top_12["X"], top_12["Y"], s=150, alpha=0.7, color="#9A8153", edgecolors="white", linewidths=2)
    
    for _, row in top_12.iterrows():
        ax1.annotate(
            str(row["agency"])[:30],
            (row["X"], row["Y"]),
            fontsize=8,
            ha="center",
            path_effects=[withStroke(linewidth=3, foreground="white")]
        )
    
    ax1.axhline(0.5, color=GRID_COLOR, linewidth=1.5, alpha=0.5)
    ax1.axvline(0, color=GRID_COLOR, linewidth=1.5, alpha=0.5)
    ax1.set_xlim(-1.05, 1.05)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xlabel("Corporate ← → Citizen", fontsize=FONT_LABEL, fontweight="bold")
    ax1.set_ylabel("Low ← → High Professional", fontsize=FONT_LABEL, fontweight="bold")
    ax1.set_title("Agency Compass", fontsize=FONT_TITLE, fontweight="bold")
    apply_clean_style(ax1)
    
    # Panel B: Stream
    ax2 = fig.add_subplot(gs[1])
    
    df_sorted = df_metrics.sort_values("X").reset_index(drop=True)
    df_sorted["cum_volume"] = df_sorted["total"].cumsum() / df_sorted["total"].sum()
    df_sorted["cum_volume_prev"] = df_sorted["cum_volume"].shift(1, fill_value=0)
    df_sorted["cum_volume_center"] = (df_sorted["cum_volume"] + df_sorted["cum_volume_prev"]) / 2
    
    shares_data = []
    for cat in CATEGORY_ORDER:
        share_col = f"share_{cat}"
        if share_col in df_sorted.columns:
            shares_data.append(df_sorted[share_col].values)
        else:
            shares_data.append(np.zeros(len(df_sorted)))
    
    shares = np.array(shares_data)
    window_size = max(3, int(len(df_sorted) * 0.07))
    if window_size % 2 == 0:
        window_size += 1
    
    from scipy.ndimage import uniform_filter1d
    shares_smooth = np.array([uniform_filter1d(row, size=window_size, mode="nearest") for row in shares])
    shares_smooth_sum = shares_smooth.sum(axis=0, keepdims=True)
    shares_smooth_sum[shares_smooth_sum == 0] = 1
    shares_norm = shares_smooth / shares_smooth_sum
    
    colors = get_category_colors()
    ax2.stackplot(df_sorted["cum_volume_center"], *shares_norm, colors=colors, alpha=0.85)
    
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("Cumulative % of Comments (Corporate → Citizen)", fontsize=FONT_LABEL, fontweight="bold")
    ax2.set_ylabel("Comment Composition", fontsize=FONT_LABEL, fontweight="bold")
    ax2.set_title("Ranked Stream", fontsize=FONT_TITLE, fontweight="bold")
    apply_clean_style(ax2)
    
    # Panel C: Spotlight (simplified for combined view)
    ax3 = fig.add_subplot(gs[2])
    ax3.text(
        0.5,
        0.5,
        "See spotlight_extremes_{}.png for agency highlights".format(year),
        ha="center",
        va="center",
        fontsize=12,
        style="italic",
        alpha=0.6
    )
    ax3.axis("off")
    
    fig.suptitle(f"Participation Continuum ({year})", fontsize=FONT_TITLE + 2, fontweight="bold", y=0.995)
    
    plt.savefig(outdir / f"participation_continuum_{year}.png", dpi=300, bbox_inches="tight")
    plt.savefig(outdir / f"participation_continuum_{year}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: participation_continuum_{year}.png/.pdf")


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate policy-brief visualizations for Federal Register comment makeup"
    )
    parser.add_argument("--year", type=int, default=2024, help="Year to analyze")
    parser.add_argument(
        "--makeup",
        type=str,
        default=None,
        help="Path to makeup CSV (auto-detects if not provided)"
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (auto-creates year subdir)"
    )
    parser.add_argument(
        "--simple-only",
        action="store_true",
        help="Only generate simple 3-chart set"
    )
    parser.add_argument(
        "--continuum-only",
        action="store_true",
        help="Only generate complex continuum page"
    )
    
    args = parser.parse_args()
    
    # Auto-detect makeup file
    if args.makeup is None:
        base_path = Path(__file__).parent
        candidates = [
            base_path / f"makeup_data_{args.year}.csv",
            base_path / "makeup_data.csv",
            base_path / ".." / "makeup" / "data" / f"makeup_results_{args.year}.csv",
        ]
        for path in candidates:
            if path.exists():
                args.makeup = str(path)
                break
        
        if args.makeup is None:
            print("ERROR: Could not find makeup CSV. Please specify --makeup")
            return

    makeup_path = Path(args.makeup)
    if not makeup_path.exists():
        print(f"ERROR: {makeup_path} not found")
        return

    # Auto-create output directory
    if args.out_dir is None:
        base_outdir = Path(__file__).parent.parent / "makeup" / "data" / "plots"
        outdir = base_outdir / str(args.year)
    else:
        outdir = Path(args.out_dir)
    
    outdir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Federal Register Comment Visualization Suite")
    print(f"{'='*60}")
    print(f"Year: {args.year}")
    print(f"Input: {makeup_path}")
    print(f"Output: {outdir}")
    print(f"{'='*60}\n")
    
    # Load and preprocess data
    print("Loading data...")
    df = pd.read_csv(makeup_path)
    
    required_cols = ["comment_id", "category"]
    for col in required_cols:
        if col not in df.columns:
            print(f"ERROR: Missing required column: {col}")
            return

    print(f"Loaded {len(df):,} rows")
    
    print("Preprocessing...")
    df = normalize_categories(df)
    df = deduplicate_comments(df)
    df = explode_agencies(df)
    
    print(f"After preprocessing: {len(df):,} rows, {df['agency'].nunique()} agencies")
    
    # Generate visualizations
    generate_simple = not args.continuum_only
    generate_continuum = not args.simple_only
    
    if generate_simple:
        print("\n" + "="*60)
        print("SIMPLE 3-CHART SET")
        print("="*60)
        plot_composition_donut(df, args.year, outdir)
        plot_agency_compass(df, args.year, outdir)
        plot_workload_vs_makeup(df, args.year, outdir)
    
    if generate_continuum:
        print("\n" + "="*60)
        print("COMPLEX CONTINUUM PAGE")
        print("="*60)
        plot_compass_with_ribbon(df, args.year, outdir)
        plot_ranked_stream(df, args.year, outdir)
        plot_spotlight_strip(df, args.year, outdir)
        plot_continuum_page(df, args.year, outdir)
    
    print("\n" + "="*60)
    print(f"[COMPLETE] All visualizations saved to: {outdir}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()


# learnings from building the plotting suite:
#
# - always write plots into stratification_scripts/makeup/data/plots/<year> so they travel with
#   the data and don't clutter the output/ directory
# - auto-create the directory to avoid runtime errors when running on clean machines
# - agency names often contain commas so explode_agencies must split and trim safely
# - category labels must match CATEGORY_ORDER exactly or the donut chart outputs zeros
# - log scales on workload plots need care (use symlog or clip to avoid log(0) issues)
# - keep matplotlib styling consistent (apply_clean_style) so visuals look like the brief
# - deduplicate comment_id before plotting to avoid double-counting from joins
# - annotate top agencies sparingly or the compass becomes unreadable
