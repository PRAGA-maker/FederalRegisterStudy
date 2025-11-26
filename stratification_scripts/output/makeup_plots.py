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

def plot_composition_donut(df: pd.DataFrame, year: int, outdir: Path, fr_csv_path: Optional[Path] = None) -> None:
    """
    Chart 1: Clean donut chart of comment composition.
    Now includes side-by-side comparison excluding top 10 documents.
    """
    counts_all = df["category"].value_counts().reindex(CATEGORY_ORDER, fill_value=0)
    if counts_all.sum() == 0:
        print(f"Warning: No data for donut chart ({year})")
        return
    
    percentages_all = 100 * counts_all / counts_all.sum()
    colors = get_category_colors()
    
    # Try to exclude top 10 documents if document_number is available
    df_excluded = None
    if fr_csv_path and fr_csv_path.exists() and "document_number" in df.columns:
        try:
            # Load FR data to get document comment counts
            fr_df = pd.read_csv(fr_csv_path)
            if "document_number" in fr_df.columns and "comment_count" in fr_df.columns:
                # Get top 10 documents by comment count
                top_docs = fr_df.nlargest(10, "comment_count")["document_number"].tolist()
                # Exclude comments from those documents
                df_excluded = df[~df["document_number"].isin(top_docs)].copy()
        except Exception as e:
            print(f"Warning: Could not load FR data for exclusion: {e}")
    
    # Create figure with side-by-side if we have exclusion data
    if df_excluded is not None and len(df_excluded) > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # Left: All comments
        wedges1, texts1 = ax1.pie(
            counts_all.values,
            colors=colors,
            startangle=90,
            wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2)
        )
        ax1.set_title(f"All Comments ({year})", fontsize=FONT_TITLE, fontweight="bold", pad=20)
        
        # Right: Excluding top 10 documents
        counts_excluded = df_excluded["category"].value_counts().reindex(CATEGORY_ORDER, fill_value=0)
        percentages_excluded = 100 * counts_excluded / counts_excluded.sum() if counts_excluded.sum() > 0 else percentages_all * 0
        
        wedges2, texts2 = ax2.pie(
            counts_excluded.values,
            colors=colors,
            startangle=90,
            wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2)
        )
        ax2.set_title(f"Excluding Top 10 Documents ({year})", fontsize=FONT_TITLE, fontweight="bold", pad=20)
        
        # Combined legend
        legend_labels = []
        for cat in CATEGORY_ORDER:
            pct_all = percentages_all.get(cat, 0)
            pct_excl = percentages_excluded.get(cat, 0)
            diff = pct_excl - pct_all
            diff_str = f" ({diff:+.1f}pp)" if abs(diff) > 0.1 else ""
            legend_labels.append(f"{cat}: {pct_all:.1f}% → {pct_excl:.1f}%{diff_str}")
        
        fig.legend(
            legend_labels,
            loc="center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=2,
            fontsize=FONT_LEGEND - 1,
            frameon=False
        )
        
        plt.tight_layout(rect=[0, 0.08, 1, 1])
    else:
        # Single donut if no exclusion data
        fig, ax = plt.subplots(figsize=(10, 8))
        
        wedges, texts = ax.pie(
            counts_all.values,
            colors=colors,
            startangle=90,
            wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2)
        )
        
        ax.set_title(f"Who commented in {year}?", fontsize=FONT_TITLE, fontweight="bold", pad=20)
        
        legend_labels = [f"{cat}: {pct:.1f}%" for cat, pct in zip(CATEGORY_ORDER, percentages_all)]
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
    Improved: Distribute labels across all quadrants
    """
    df_metrics = compute_agency_metrics(df)
    
    if df_metrics.empty:
        print(f"Warning: No agency metrics for compass ({year})")
        return
    
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
    
    # Define quadrants
    def get_quadrant(x, y):
        if x >= 0 and y >= 0.5:
            return 1  # High Citizen High Professional
        elif x < 0 and y >= 0.5:
            return 2  # High Corporate High Professional
        elif x >= 0 and y < 0.5:
            return 3  # High Citizen Low Professional
        else:
            return 4  # High Corporate Low Professional
    
    df_metrics["quadrant"] = df_metrics.apply(lambda r: get_quadrant(r["X"], r["Y"]), axis=1)
    
    # Select top agencies per quadrant (at least 2-3 per quadrant)
    labels_per_quadrant = max(2, top_n // 4)
    agencies_to_label = []
    
    for q in [1, 2, 3, 4]:
        quadrant_agencies = df_metrics[df_metrics["quadrant"] == q].nlargest(labels_per_quadrant, "total")
        agencies_to_label.append(quadrant_agencies)
    
    agencies_to_label = pd.concat(agencies_to_label)
    
    # Also add overall top agencies if not already included
    top_overall = df_metrics.nlargest(top_n, "total")
    agencies_to_label = pd.concat([agencies_to_label, top_overall]).drop_duplicates(subset=["agency"])
    
    # Prepare labels
    try:
        from adjustText import adjust_text
        texts = []
        for _, row in agencies_to_label.iterrows():
            label = str(row["agency"])[:35]
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
        # Fallback: manual positioning per quadrant
        for _, row in agencies_to_label.iterrows():
            label = str(row["agency"])[:35]
            # Offset labels slightly based on quadrant
            offset_x = 0.05 if row["X"] >= 0 else -0.05
            offset_y = 0.05 if row["Y"] >= 0.5 else -0.05
            ax.annotate(
                label,
                (row["X"], row["Y"]),
                xytext=(offset_x * 20, offset_y * 20),
                textcoords="offset points",
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
    
    # Add quadrant statistics
    for q, label_pos in [(1, (0.5, 0.9)), (2, (-0.5, 0.9)), (3, (0.5, 0.1)), (4, (-0.5, 0.1))]:
        q_count = len(df_metrics[df_metrics["quadrant"] == q])
        q_vol = df_metrics[df_metrics["quadrant"] == q]["total"].sum()
        ax.text(
            label_pos[0], label_pos[1],
            f"{q_count} agencies\n{q_vol:,.0f} comments",
            ha="center", va="center" if q <= 2 else "center",
            fontsize=8, alpha=0.5, style="italic"
        )
    
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
    ax.set_yscale("log")
    ax.set_xlabel("Total Comments (log scale)", fontsize=FONT_LABEL, fontweight="bold")
    ax.set_ylabel("Count by Category (log scale)", fontsize=FONT_LABEL, fontweight="bold")
    ax.set_title(f"Workload vs. Who Shows Up ({year})", fontsize=FONT_TITLE, fontweight="bold", pad=20)
    
    ax.legend(fontsize=FONT_LEGEND - 1, frameon=True, loc="upper left", framealpha=0.9)
    apply_clean_style(ax)
    
    plt.tight_layout()
    fig.savefig(outdir / f"workload_vs_makeup_{year}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: workload_vs_makeup_{year}.png")


def plot_workload_vs_citizen_by_agency(df: pd.DataFrame, year: int, outdir: Path) -> None:
    """
    Workload vs citizen input by agency (like Gini frontier).
    Shows relationship between total comments and citizen participation.
    """
    df_metrics = compute_agency_metrics(df)
    
    if df_metrics.empty:
        print(f"Warning: No agency metrics for workload vs citizen chart ({year})")
        return
    
    oc_count_col = f"n_Ordinary Citizen"
    if oc_count_col not in df_metrics.columns:
        print(f"Warning: Missing {oc_count_col} column")
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Scatter plot
    ax.scatter(
        df_metrics["total"],
        df_metrics[oc_count_col],
        s=60,
        alpha=0.5,
        color=COLOR_PALETTE["Ordinary Citizen"],
        edgecolors="white",
        linewidths=0.5
    )
    
    # Add diagonal reference line (perfect citizen participation)
    max_total = df_metrics["total"].max()
    max_citizen = df_metrics[oc_count_col].max()
    max_val = max(max_total, max_citizen)
    ax.plot([0, max_val], [0, max_val], '--', color="#5C6670", alpha=0.3, linewidth=1, label="Perfect citizen participation")
    
    # Styling
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Total Comments by Agency (log scale)", fontsize=FONT_LABEL, fontweight="bold")
    ax.set_ylabel("Ordinary Citizen Comments (log scale)", fontsize=FONT_LABEL, fontweight="bold")
    ax.set_title(f"Workload vs Citizen Input (agency, {year})", fontsize=FONT_TITLE, fontweight="bold", pad=20)
    
    ax.legend(fontsize=FONT_LEGEND - 1, frameon=True, loc="upper left", framealpha=0.9)
    apply_clean_style(ax)
    
    plt.tight_layout()
    fig.savefig(outdir / f"workload_vs_citizen_by_agency_{year}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: workload_vs_citizen_by_agency_{year}.png")


def plot_workload_vs_citizen_by_document(df: pd.DataFrame, year: int, outdir: Path) -> None:
    """
    Workload vs citizen input by document_number (like Gini frontier).
    Shows relationship between total comments per document and citizen participation.
    """
    if "document_number" not in df.columns:
        print(f"Warning: document_number not available for workload vs citizen by document ({year})")
        return
    
    # Group by document_number
    doc_grp = df.groupby("document_number").agg({
        "comment_id": "count",
        "category": lambda x: (x == "Ordinary Citizen").sum()
    }).rename(columns={"comment_id": "total", "category": "citizen_count"})
    
    if doc_grp.empty:
        print(f"Warning: No document-level data ({year})")
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Scatter plot
    ax.scatter(
        doc_grp["total"],
        doc_grp["citizen_count"],
        s=40,
        alpha=0.4,
        color=COLOR_PALETTE["Ordinary Citizen"],
        edgecolors="white",
        linewidths=0.3
    )
    
    # Add diagonal reference line
    max_total = doc_grp["total"].max()
    max_citizen = doc_grp["citizen_count"].max()
    max_val = max(max_total, max_citizen)
    ax.plot([0, max_val], [0, max_val], '--', color="#5C6670", alpha=0.3, linewidth=1, label="Perfect citizen participation")
    
    # Styling
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Total Comments by Document (log scale)", fontsize=FONT_LABEL, fontweight="bold")
    ax.set_ylabel("Ordinary Citizen Comments (log scale)", fontsize=FONT_LABEL, fontweight="bold")
    ax.set_title(f"Workload vs Citizen Input (document_number, {year})", fontsize=FONT_TITLE, fontweight="bold", pad=20)
    
    ax.legend(fontsize=FONT_LEGEND - 1, frameon=True, loc="upper left", framealpha=0.9)
    apply_clean_style(ax)
    
    plt.tight_layout()
    fig.savefig(outdir / f"workload_vs_citizen_by_document_number_{year}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: workload_vs_citizen_by_document_number_{year}.png")


def plot_comment_distribution_roi(df: pd.DataFrame, year: int, outdir: Path, fr_csv_path: Optional[Path] = None) -> None:
    """
    High-ROI visualization: Shows that most documents get few/no comments and few citizens participate.
    This is the key story visualization.
    """
    # Get document-level stats
    if "document_number" in df.columns:
        doc_stats = df.groupby("document_number").agg({
            "comment_id": "count",
            "category": lambda x: (x == "Ordinary Citizen").sum()
        }).rename(columns={"comment_id": "total_comments", "category": "citizen_comments"})
        doc_stats["has_citizens"] = (doc_stats["citizen_comments"] > 0).astype(int)
    else:
        print(f"Warning: document_number not available for ROI chart ({year})")
        return
    
    # Load FR data to get documents with 0 comments
    zero_comment_docs = 0
    if fr_csv_path and fr_csv_path.exists():
        try:
            fr_df = pd.read_csv(fr_csv_path)
            if "document_number" in fr_df.columns and "comment_count" in fr_df.columns:
                zero_comment_docs = (fr_df["comment_count"] == 0).sum()
        except Exception as e:
            print(f"Warning: Could not load FR data: {e}")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Panel 1: Distribution of comment counts per document
    ax1 = axes[0, 0]
    comment_counts = doc_stats["total_comments"].values
    bins = np.logspace(0, np.log10(max(comment_counts.max(), 1)), 30)
    ax1.hist(comment_counts, bins=bins, color="#5C6670", alpha=0.7, edgecolor="white", linewidth=0.5)
    ax1.set_xscale("log")
    ax1.set_xlabel("Comments per Document (log scale)", fontsize=FONT_LABEL, fontweight="bold")
    ax1.set_ylabel("Number of Documents", fontsize=FONT_LABEL, fontweight="bold")
    ax1.set_title("Distribution of Comment Volume", fontsize=FONT_TITLE - 2, fontweight="bold")
    apply_clean_style(ax1)
    
    # Add statistics
    median_comments = doc_stats["total_comments"].median()
    pct_low = (doc_stats["total_comments"] <= 5).sum() / len(doc_stats) * 100
    ax1.axvline(median_comments, color="#9A8153", linestyle="--", linewidth=2, alpha=0.7, label=f"Median: {median_comments:.0f}")
    ax1.text(0.7, 0.95, f"{pct_low:.1f}% have ≤5 comments", transform=ax1.transAxes, 
             fontsize=10, alpha=0.7, va="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax1.legend()
    
    # Panel 2: Documents with zero comments (if available)
    ax2 = axes[0, 1]
    if zero_comment_docs > 0:
        total_docs = len(fr_df) if fr_csv_path and fr_csv_path.exists() else len(doc_stats)
        docs_with_comments = total_docs - zero_comment_docs
        categories = ["Documents\nwith Comments", "Documents\nwith Zero Comments"]
        counts = [docs_with_comments, zero_comment_docs]
        colors_bar = [COLOR_PALETTE["Ordinary Citizen"], "#DDE3EA"]
        
        bars = ax2.bar(categories, counts, color=colors_bar, alpha=0.8, edgecolor="white", linewidth=2)
        ax2.set_ylabel("Number of Documents", fontsize=FONT_LABEL, fontweight="bold")
        ax2.set_title("Comment Participation Rate", fontsize=FONT_TITLE - 2, fontweight="bold")
        
        # Add percentages
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            pct = count / total_docs * 100
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count:,}\n({pct:.1f}%)',
                    ha='center', va='bottom', fontsize=11, fontweight="bold")
    else:
        ax2.text(0.5, 0.5, "Zero-comment data\nnot available", 
                ha="center", va="center", transform=ax2.transAxes, fontsize=12, alpha=0.5)
        ax2.set_title("Comment Participation Rate", fontsize=FONT_TITLE - 2, fontweight="bold")
    apply_clean_style(ax2)
    
    # Panel 3: Citizen participation rate by comment volume
    ax3 = axes[1, 0]
    # Bin documents by comment volume
    doc_stats["comment_bin"] = pd.cut(doc_stats["total_comments"], 
                                       bins=[0, 1, 5, 10, 50, 100, float('inf')],
                                       labels=["1", "2-5", "6-10", "11-50", "51-100", "100+"])
    bin_stats = doc_stats.groupby("comment_bin").agg({
        "has_citizens": "mean",
        "citizen_comments": "mean",
        "total_comments": "count"
    }).reset_index()
    
    x_pos = np.arange(len(bin_stats))
    bars = ax3.bar(x_pos, bin_stats["has_citizens"] * 100, 
                   color=COLOR_PALETTE["Ordinary Citizen"], alpha=0.7, edgecolor="white", linewidth=1.5)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(bin_stats["comment_bin"].astype(str), rotation=0)
    ax3.set_ylabel("% Documents with Citizen Comments", fontsize=FONT_LABEL, fontweight="bold")
    ax3.set_xlabel("Comments per Document", fontsize=FONT_LABEL, fontweight="bold")
    ax3.set_title("Citizen Participation by Document Volume", fontsize=FONT_TITLE - 2, fontweight="bold")
    ax3.set_ylim(0, 100)
    
    # Add value labels
    for bar, val in zip(bars, bin_stats["has_citizens"] * 100):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%',
                ha='center', va='bottom', fontsize=9, fontweight="bold")
    apply_clean_style(ax3)
    
    # Panel 4: Cumulative distribution showing concentration
    ax4 = axes[1, 1]
    sorted_docs = doc_stats.sort_values("total_comments", ascending=False)
    sorted_docs["cum_pct_docs"] = np.arange(1, len(sorted_docs) + 1) / len(sorted_docs) * 100
    sorted_docs["cum_pct_comments"] = sorted_docs["total_comments"].cumsum() / sorted_docs["total_comments"].sum() * 100
    
    ax4.plot(sorted_docs["cum_pct_docs"], sorted_docs["cum_pct_comments"], 
            color="#12161A", linewidth=3, alpha=0.8)
    ax4.fill_between(sorted_docs["cum_pct_docs"], 0, sorted_docs["cum_pct_comments"], 
                     alpha=0.2, color="#12161A")
    
    # Add reference line (perfect equality)
    ax4.plot([0, 100], [0, 100], '--', color="#5C6670", alpha=0.4, linewidth=1.5, label="Perfect equality")
    
    # Mark key points
    top10_pct_docs = sorted_docs.iloc[len(sorted_docs)//10]["cum_pct_docs"]
    top10_pct_comments = sorted_docs.iloc[len(sorted_docs)//10]["cum_pct_comments"]
    ax4.plot([top10_pct_docs, top10_pct_docs], [0, top10_pct_comments], 
            '--', color="#9A8153", alpha=0.6, linewidth=1.5)
    ax4.plot([0, top10_pct_docs], [top10_pct_comments, top10_pct_comments], 
            '--', color="#9A8153", alpha=0.6, linewidth=1.5)
    ax4.text(top10_pct_docs, top10_pct_comments, 
            f'Top 10%: {top10_pct_comments:.1f}% of comments',
            fontsize=10, alpha=0.8, ha="left", va="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    
    ax4.set_xlabel("% of Documents (sorted by volume)", fontsize=FONT_LABEL, fontweight="bold")
    ax4.set_ylabel("% of Total Comments", fontsize=FONT_LABEL, fontweight="bold")
    ax4.set_title("Comment Concentration (Lorenz Curve)", fontsize=FONT_TITLE - 2, fontweight="bold")
    ax4.set_xlim(0, 100)
    ax4.set_ylim(0, 100)
    ax4.legend()
    apply_clean_style(ax4)
    
    fig.suptitle(f"The Participation Gap ({year})", fontsize=FONT_TITLE + 2, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(outdir / f"participation_gap_{year}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: participation_gap_{year}.png")


# ============================================================================
# COMPLEX CONTINUUM PAGE
# ============================================================================

def plot_compass_with_ribbon(df: pd.DataFrame, year: int, outdir: Path, top_n: int = 12) -> None:
    """
    Panel A: Compass with weighted median ribbon and IQR band.
    Improved: Add more context about what the ribbon shows, highlight key insights
    """
    df_metrics = compute_agency_metrics(df)
    
    if df_metrics.empty:
        print(f"Warning: No agency metrics for ribbon compass ({year})")
        return
    
    # Compute ribbon
    ribbon = compute_ribbon_band(df_metrics, n_bins=30)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Draw IQR band with more informative styling
    if not ribbon.empty and len(ribbon) > 2:
        ax.fill_between(
            ribbon["x_center"],
            ribbon["y_q25"],
            ribbon["y_q75"],
            alpha=0.25,
            color="#2E6F88",
            label="IQR: 50% of agencies fall within this band"
        )
        ax.plot(
            ribbon["x_center"],
            ribbon["y_median"],
            color="#2E6F88",
            linewidth=3.5,
            alpha=0.9,
            label="Weighted median professional input",
            zorder=5
        )
        
        # Highlight key insights: where does professional input peak?
        max_prof_idx = ribbon["y_median"].idxmax()
        if max_prof_idx is not None and not pd.isna(max_prof_idx):
            max_x = ribbon.loc[max_prof_idx, "x_center"]
            max_y = ribbon.loc[max_prof_idx, "y_median"]
            ax.plot([max_x, max_x], [0, max_y], '--', color="#9A8153", alpha=0.6, linewidth=2, zorder=4)
            ax.scatter([max_x], [max_y], s=200, color="#9A8153", edgecolors="white", 
                      linewidths=2, zorder=6, label=f"Peak: {max_y:.2f} at X={max_x:.2f}")
    
    # Scatter all agencies (faint)
    ax.scatter(
        df_metrics["X"],
        df_metrics["Y"],
        s=df_metrics["total"] / df_metrics["total"].max() * 150 + 15,
        alpha=0.15,
        color="#5C6670",
        edgecolors="none",
        zorder=1
    )
    
    # Highlight top agencies and extremes (distributed across quadrants)
    top_agencies = df_metrics.nlargest(top_n, "total")
    
    # Get representatives from each quadrant
    def get_quadrant(x, y):
        if x >= 0 and y >= 0.5: return 1
        elif x < 0 and y >= 0.5: return 2
        elif x >= 0 and y < 0.5: return 3
        else: return 4
    
    df_metrics["quadrant"] = df_metrics.apply(lambda r: get_quadrant(r["X"], r["Y"]), axis=1)
    highlights_list = [top_agencies]
    
    for q in [1, 2, 3, 4]:
        q_agencies = df_metrics[df_metrics["quadrant"] == q]
        if len(q_agencies) > 0:
            highlights_list.append(q_agencies.nlargest(2, "total"))
    
    highlights = pd.concat(highlights_list).drop_duplicates(subset=["agency"])
    
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
    
    # Label highlights (fewer labels, better distributed)
    labels_per_quadrant = 2
    for q in [1, 2, 3, 4]:
        q_highlights = highlights[highlights["quadrant"] == q].head(labels_per_quadrant)
        for _, row in q_highlights.iterrows():
            label = str(row["agency"])[:30]
            ax.annotate(
                label,
                (row["X"], row["Y"]),
                fontsize=8,
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
    label_props = dict(fontsize=11, alpha=0.5, style="italic", weight="bold")
    ax.text(0.5, 0.85, "High Citizen\nHigh Professional", ha="center", va="center", **label_props)
    ax.text(-0.5, 0.85, "High Corporate\nHigh Professional", ha="center", va="center", **label_props)
    ax.text(0.5, 0.15, "High Citizen\nLow Professional", ha="center", va="center", **label_props)
    ax.text(-0.5, 0.15, "High Corporate\nLow Professional", ha="center", va="center", **label_props)
    
    # Add interpretation text
    if not ribbon.empty:
        left_prof = ribbon[ribbon["x_center"] < -0.3]["y_median"].mean()
        right_prof = ribbon[ribbon["x_center"] > 0.3]["y_median"].mean()
        if not pd.isna(left_prof) and not pd.isna(right_prof):
            insight = "Corporate agencies have higher\nprofessional input" if left_prof > right_prof else "Citizen agencies have higher\nprofessional input"
            ax.text(0.02, 0.02, insight, transform=ax.transAxes, fontsize=9, 
                   alpha=0.6, style="italic", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    
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
        f"Professional Input Across Agency Spectrum ({year})",
        fontsize=FONT_TITLE,
        fontweight="bold",
        pad=20
    )
    
    ax.legend(fontsize=FONT_LEGEND - 1, frameon=True, loc="upper right", framealpha=0.9)
    apply_clean_style(ax)
    
    plt.tight_layout()
    fig.savefig(outdir / f"agency_compass_ribbon_{year}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: agency_compass_ribbon_{year}.png")


def plot_ranked_stream(df: pd.DataFrame, year: int, outdir: Path) -> None:
    """
    Panel B: Ranked stream graph showing composition across corporate→grassroots continuum.
    Improved: Show absolute counts instead of percentages, fix axis overlap
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
    
    # Prepare absolute counts for each category (more informative than shares)
    counts_data = []
    for cat in CATEGORY_ORDER:
        count_col = f"n_{cat}"
        if count_col in df_sorted.columns:
            counts_data.append(df_sorted[count_col].values)
        else:
            counts_data.append(np.zeros(len(df_sorted)))
    
    counts = np.array(counts_data)
    
    # Apply moving average smoothing
    window_size = max(3, int(len(df_sorted) * 0.07))
    if window_size % 2 == 0:
        window_size += 1
    
    from scipy.ndimage import uniform_filter1d
    counts_smooth = np.array([uniform_filter1d(row, size=window_size, mode="nearest") for row in counts])
    
    fig, ax = plt.subplots(figsize=(16, 7))
    
    # Stacked area plot with absolute counts
    colors = get_category_colors()
    ax.stackplot(
        df_sorted["cum_volume_center"],
        *counts_smooth,
        labels=CATEGORY_ORDER,
        colors=colors,
        alpha=0.85
    )
    
    # Mark key transition points (where composition changes significantly)
    # Find where citizen share crosses 50%
    citizen_share_col = f"share_Ordinary Citizen"
    if citizen_share_col in df_sorted.columns:
        citizen_shares = df_sorted[citizen_share_col]
    else:
        citizen_shares = pd.Series([0] * len(df_sorted))
    
    if len(citizen_shares) > 0:
        # Find median point where citizen share becomes dominant
        median_idx = len(df_sorted) // 2
        if median_idx < len(df_sorted):
            x_pos = df_sorted.loc[median_idx, "cum_volume_center"]
            ax.axvline(x_pos, color="#12161A", linewidth=2, alpha=0.4, linestyle="--", zorder=10)
            ax.text(
                x_pos, ax.get_ylim()[1] * 0.95,
                "Midpoint",
                rotation=90,
                fontsize=9,
                alpha=0.7,
                ha="right",
                va="top"
            )
    
    # Mark top agencies
    top_3_indices = df_sorted.nlargest(3, "total").index.values
    for idx in top_3_indices:
        if idx < len(df_sorted):
            x_pos = df_sorted.loc[idx, "cum_volume_center"]
            ax.axvline(x_pos, color="#12161A", linewidth=1, alpha=0.2, linestyle=":")
            agency_name = str(df_sorted.loc[idx, "agency"])[:18]
            ax.text(
                x_pos,
                ax.get_ylim()[1] * 0.98,
                agency_name,
                rotation=45,
                fontsize=8,
                alpha=0.6,
                ha="left",
                va="bottom"
            )
    
    # Styling with better spacing
    ax.set_xlim(0, 1)
    ax.set_ylim(0, None)  # Auto-scale y-axis
    ax.set_xlabel("Cumulative % of Total Comments (Corporate → Citizen agencies)", 
                   fontsize=FONT_LABEL, fontweight="bold", labelpad=10)
    ax.set_ylabel("Comment Count", fontsize=FONT_LABEL, fontweight="bold", labelpad=10)
    ax.set_title(
        f"Comment Volume Across Agency Spectrum ({year})",
        fontsize=FONT_TITLE,
        fontweight="bold",
        pad=25
    )
    
    # Add subtitle explanation
    ax.text(
        0.5, 1.02,
        "Agencies sorted from corporate-dominated (left) to citizen-dominated (right)",
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
    
    # Try to find FR CSV for additional context
    fr_csv_path = None
    base_path = Path(__file__).parent.parent.parent
    fr_candidates = [
        base_path / "stratification_scripts" / "output" / f"federal_register_{args.year}_comments.csv",
        base_path / "stratification_scripts" / "output" / "federal_register_2015_comments.csv",
    ]
    for path in fr_candidates:
        if path.exists():
            fr_csv_path = path
            print(f"Found FR CSV: {fr_csv_path}")
            break
    
    # Try to join document_number if available in FR CSV
    if fr_csv_path and fr_csv_path.exists() and "document_number" not in df.columns:
        try:
            fr_df = pd.read_csv(fr_csv_path, usecols=["document_number", "comment_id"], nrows=1000)
            if "document_number" in fr_df.columns and "comment_id" in fr_df.columns:
                # Try to join on comment_id
                df = df.merge(fr_df[["comment_id", "document_number"]], on="comment_id", how="left")
                print(f"Joined document_number from FR CSV")
        except Exception as e:
            print(f"Warning: Could not join document_number: {e}")
    
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
        plot_composition_donut(df, args.year, outdir, fr_csv_path)
        plot_agency_compass(df, args.year, outdir)
        plot_workload_vs_makeup(df, args.year, outdir)
        plot_workload_vs_citizen_by_agency(df, args.year, outdir)
        if "document_number" in df.columns:
            plot_workload_vs_citizen_by_document(df, args.year, outdir)
        plot_comment_distribution_roi(df, args.year, outdir, fr_csv_path)
    
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
