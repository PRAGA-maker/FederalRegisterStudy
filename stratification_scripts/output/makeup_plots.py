"""
Federal Register Comment Makeup Visualization Suite

Generates policy-brief visualizations showing who participates in federal rulemaking.
Produces both simple 3-chart sets and complex continuum pages with ribbon analysis.

Optional dependency: adjustText (pip install adjusttext) for better label positioning

Example:
    # CLI usage
    $ python -m stratification_scripts.output.makeup_plots --year 2024
    
    # Programmatic usage
    >>> from stratification_scripts.output.makeup_plots import generate_plots_for_year
    >>> from stratification_scripts.config import PipelineConfig
    >>> 
    >>> config = PipelineConfig(year=2024)
    >>> generate_plots_for_year(config)
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patheffects import withStroke
import numpy as np
import pandas as pd

from stratification_scripts.config import (
    PipelineConfig,
    get_makeup_data_path,
    get_fr_csv_path,
    get_plots_dir,
)
from stratification_scripts.logging_utils import (
    get_logger,
    log_banner,
    setup_logging,
)

logger = get_logger(__name__)
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
# DATA PREPROCESSING & WEIGHTING
# ============================================================================

def normalize_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Map categories to standardized names."""
    df = df.copy()
    if "category" in df.columns:
        df["category"] = df["category"].map(CATEGORY_MAPPING).fillna(df["category"])
        # Filter to only known categories
        df = df[df["category"].isin(CATEGORY_ORDER)]
    return df


def parse_agency_hierarchy(agency_str: str) -> Tuple[str, Optional[str]]:
    """Parse agency string to extract main agency and sub-agency.
    
    Examples:
        "Defense Department, Army Department" -> ("Defense Department", "Army Department")
        "Nuclear Regulatory Commission" -> ("Nuclear Regulatory Commission", None)
        "Transportation Department, National Highway Traffic Safety Administration" -> ("Transportation Department", "National Highway Traffic Safety Administration")
    """
    if pd.isna(agency_str) or agency_str == "":
        return ("Unknown", None)
    
    agency_str = str(agency_str).strip()
    parts = [p.strip() for p in agency_str.split(",")]
    
    if len(parts) == 1:
        return (parts[0], None)
    elif len(parts) >= 2:
        # First part is main agency, second is sub-agency
        # If there are more parts, join them as sub-agency
        main_agency = parts[0]
        sub_agency = ", ".join(parts[1:])
        return (main_agency, sub_agency)
    else:
        return ("Unknown", None)


def explode_agencies(df: pd.DataFrame) -> pd.DataFrame:
    """Split multi-agency entries and parse hierarchy."""
    df = df.copy()
    if "agency" not in df.columns or df["agency"].isna().all():
        df["agency"] = "Unknown"
        df["main_agency"] = "Unknown"
        df["sub_agency"] = None
        return df
    
    # Parse agency hierarchy
    df["agency"] = df["agency"].fillna("Unknown")
    parsed = df["agency"].apply(parse_agency_hierarchy)
    df["main_agency"] = [p[0] for p in parsed]
    df["sub_agency"] = [p[1] for p in parsed]
    
    # For Sankey/dendrogram, we want to explode so each comment can link to both main and sub-agency
    # But keep the original agency column for compatibility
    return df


def deduplicate_comments(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate comments by comment_id."""
    if "comment_id" in df.columns:
        df = df.drop_duplicates(subset=["comment_id"])
    return df


def calculate_weights(df: pd.DataFrame, fr_csv_path: Optional[Path]) -> pd.DataFrame:
    """
    Calculate sampling weights to reconstruct the full population.
    
    Implements TWO types of weights:
    
    1. `weight` (Stratum Weight): 
       - Scales sampled comments to represent the FULL POPULATION of the stratum (Agency x Bin).
       - Includes expansion for UNSAMPLED documents in the same stratum.
       - Use for: Agency-level metrics, Totals, Donuts, Sankey.
       - Formula: N_pop_stratum / N_sample_stratum
       
    2. `weight_doc` (Document Weight):
       - Scales sampled comments to represent ONLY THE DOCUMENT they belong to.
       - Does NOT expand to unsampled documents.
       - Use for: Document-level scatterplots (Workload vs Citizen).
       - Formula: N_true_doc / N_sample_doc
    """
    df = df.copy()
    
    # Default weights = 1.0 if no FR data available
    if not fr_csv_path or not fr_csv_path.exists():
        print("Warning: FR metadata not found. Using unweighted data (weight=1.0)")
        df["weight"] = 1.0
        df["weight_doc"] = 1.0
        return df
    
    try:
        # Load FR Population Data
        fr_df = pd.read_csv(fr_csv_path)
        
        # Ensure critical columns exist
        if "agency" not in fr_df.columns or "comment_count" not in fr_df.columns:
            print("Warning: FR metadata missing 'agency' or 'comment_count'. Using unweighted data.")
            df["weight"] = 1.0
            df["weight_doc"] = 1.0
            return df
            
        # Filter to documents with comments
        fr_df = fr_df[fr_df["comment_count"] > 0].copy()
        
        # Assign Comment Bins (matching mine_comments.py logic)
        def get_bin(n):
            if n <= 10: return "0-10"
            elif n <= 100: return "11-100"
            elif n <= 1000: return "101-1000"
            elif n <= 10000: return "1001-10000"
            else: return "10000+"
            
        fr_df["comment_bin"] = fr_df["comment_count"].apply(get_bin)
        
        # ---------------------------------------------------------
        # 1. Calculate Stratum Weight (Population Expansion)
        # ---------------------------------------------------------
        
        # Calculate Population Totals (N_pop) per Stratum (Agency x Bin)
        pop_strata = fr_df.groupby(["agency", "comment_bin"])["comment_count"].sum().rename("N_pop").reset_index()
        
        # Join document metadata to sample data to get comment counts per document for binning
        if "document_number" in df.columns and "document_number" in fr_df.columns:
            df = df.merge(fr_df[["document_number", "comment_count"]], on="document_number", how="left")
            # Fill missing comment counts with 1 (assume singleton) to avoid errors
            df["comment_count"] = df["comment_count"].fillna(1)
            df["comment_bin"] = df["comment_count"].apply(get_bin)
        else:
            print("Warning: Could not link documents to FR metadata. Using unweighted data.")
            df["weight"] = 1.0
            df["weight_doc"] = 1.0
            return df
            
        # Calculate Sample Totals (N_sample) per Stratum
        # NOTE: This counts ALL comments including duplicates (each duplicate is a separate submission)
        sample_strata = df.groupby(["agency", "comment_bin"]).size().rename("N_sample").reset_index()
        
        # Merge Population and Sample Stratum Totals
        strata_weights = pd.merge(pop_strata, sample_strata, on=["agency", "comment_bin"], how="inner")
        
        # Calculate Weight
        strata_weights["weight"] = strata_weights["N_pop"] / strata_weights["N_sample"]
        
        # Merge stratum weights back to main dataframe
        df = df.merge(strata_weights[["agency", "comment_bin", "weight"]], on=["agency", "comment_bin"], how="left")
        
        # ---------------------------------------------------------
        # 2. Calculate Document Weight (Document Reconstruction)
        # ---------------------------------------------------------
        
        # Count sampled comments per document
        # NOTE: This counts ALL comments including duplicates (each duplicate is a separate submission)
        doc_sample_counts = df.groupby("document_number").size().rename("n_sample_doc")
        
        # Join back to DF
        df = df.merge(doc_sample_counts, on="document_number", how="left")
        
        # Calculate weight_doc = True Total / Sampled Total
        # This reconstructs the document exactly, without inflating for missing neighbor documents
        # NOTE: Duplicates are counted separately - each duplicate comment gets the same weight as if unique
        df["weight_doc"] = df["comment_count"] / df["n_sample_doc"]
        
        # Fill missing weights and provide diagnostics
        missing_weights = df["weight"].isna().sum()
        if missing_weights > 0:
            print(f"Warning: {missing_weights} comments could not be weighted (metadata mismatch). Defaulting to 1.0")
            
            # Diagnose why weights are missing
            unweighted = df[df["weight"].isna()]
            if "agency" in unweighted.columns:
                unmatched_agencies = unweighted["agency"].value_counts().head(10)
                print(f"  Top unmatched agencies (will use weight=1.0):")
                for agency, count in unmatched_agencies.items():
                    print(f"    {agency}: {count} comments")
            
            df["weight"] = df["weight"].fillna(1.0)
            df["weight_doc"] = df["weight_doc"].fillna(1.0)
        
        # Count orphaned comments (weight == 1.0 after merge)
        orphaned = (df["weight"] == 1.0).sum()
        total = len(df)
        pct_orphaned = orphaned / total * 100 if total > 0 else 0
        
        print(f"Reweighting complete.")
        print(f"  Stratum Weight (Population): Mean={df['weight'].mean():.2f}, Max={df['weight'].max():.2f}")
        print(f"  Document Weight (Specific):  Mean={df['weight_doc'].mean():.2f}, Max={df['weight_doc'].max():.2f}")
        print(f"  Comments with weight=1.0 (orphaned): {orphaned:,} ({pct_orphaned:.1f}%)")
        
        if orphaned > 0 and orphaned < total * 0.05:  # Less than 5% orphaned is acceptable
            print(f"  [OK] Weight calculation success rate: {100-pct_orphaned:.1f}%")
        elif orphaned >= total * 0.05:
            print(f"  [WARNING] High rate of orphaned comments ({pct_orphaned:.1f}%) - check agency matching")
        
        return df
        
    except Exception as e:
        print(f"Error calculating weights: {e}")
        import traceback
        traceback.print_exc()
        df["weight"] = 1.0
        df["weight_doc"] = 1.0
        return df


def compute_agency_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate agency-level metrics for compass positioning using WEIGHTED sums.
    Returns DataFrame with columns: agency, total, X, Y, plus share/count columns.
    """
    # Ensure weight column exists
    if "weight" not in df.columns:
        df["weight"] = 1.0
        
    # Group by agency and category, summing weights instead of counting rows
    grp = df.groupby(["agency", "category"])["weight"].sum().rename("n").reset_index()
    
    # Calculate totals
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
    # Use direct column access since we've ensured columns exist above
    oc_share_col = f"share_Ordinary Citizen"
    org_share_col = f"share_Large Organization/Corporation"
    exp_share_col = f"share_Academic/Industry/Expert (incl. small/local business)"
    lob_share_col = f"share_Political Consultant/Lobbyist"
    
    wide["X"] = (wide[oc_share_col] - wide[org_share_col]).clip(-1, 1)
    wide["Y"] = (wide[exp_share_col] + wide[lob_share_col]).clip(0, 1)
    
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


def plot_weight_distribution(df: pd.DataFrame, outdir: Path) -> None:
    """
    Plot distribution of weights to verify reweighting logic.
    """
    if "weight" not in df.columns:
        print("Warning: No weights to plot distribution for")
        return
        
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Panel 1: Stratum Weights
    weights = df["weight"].values
    ax1 = axes[0]
    
    # Handle edge case where all weights are the same
    if weights.min() == weights.max():
        # If all weights are 1.0, show a simple bar
        ax1.bar([0], [len(weights)], color="#5C6670", alpha=0.7, width=0.5)
        ax1.set_xlim(-1, 1)
        ax1.set_xlabel("Weight", fontsize=FONT_LABEL, fontweight="bold")
        ax1.text(0, len(weights)/2, f"All weights = {weights[0]:.1f}\n(n={len(weights):,})", 
                ha="center", va="center", fontsize=11, fontweight="bold")
    else:
        ax1.hist(weights, bins=np.logspace(np.log10(max(1, weights.min())), np.log10(weights.max()), 50), 
                color="#5C6670", alpha=0.7, edgecolor="white")
        ax1.set_xscale("log")
        ax1.set_xlabel("Weight (log scale)", fontsize=FONT_LABEL, fontweight="bold")
    
    ax1.set_ylabel("Number of Comments", fontsize=FONT_LABEL, fontweight="bold")
    ax1.set_title("Stratum Weights (Population Expansion)", fontsize=FONT_TITLE-2, fontweight="bold")
    
    # Panel 2: Document Weights
    if "weight_doc" in df.columns:
        weights_doc = df["weight_doc"].values
        ax2 = axes[1]
        
        # Handle edge case where all weights are the same
        if weights_doc.min() == weights_doc.max():
            ax2.bar([0], [len(weights_doc)], color="#9A8153", alpha=0.7, width=0.5)
            ax2.set_xlim(-1, 1)
            ax2.set_xlabel("Weight", fontsize=FONT_LABEL, fontweight="bold")
            ax2.text(0, len(weights_doc)/2, f"All weights = {weights_doc[0]:.1f}\n(n={len(weights_doc):,})", 
                    ha="center", va="center", fontsize=11, fontweight="bold")
        else:
            ax2.hist(weights_doc, bins=np.logspace(np.log10(max(1, weights_doc.min())), np.log10(weights_doc.max()), 50), 
                    color="#9A8153", alpha=0.7, edgecolor="white")
            ax2.set_xscale("log")
            ax2.set_xlabel("Weight (log scale)", fontsize=FONT_LABEL, fontweight="bold")
        
        ax2.set_ylabel("Number of Comments", fontsize=FONT_LABEL, fontweight="bold")
        ax2.set_title("Document Weights (Reconstruction)", fontsize=FONT_TITLE-2, fontweight="bold")
    
    apply_clean_style(ax1)
    if "weight_doc" in df.columns:
        apply_clean_style(ax2)
    plt.tight_layout()
    fig.savefig(outdir / "weight_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[OK] Saved: weight_distribution.png")


# ============================================================================
# SIMPLE CHART SET
# ============================================================================

def plot_composition_donut(df: pd.DataFrame, year: int, outdir: Path, fr_csv_path: Optional[Path] = None) -> None:
    """
    Chart 1: Clean donut chart of comment composition.
    Now includes side-by-side comparison excluding top 10 documents.
    Updated to use weighted sums.
    """
    if "weight" not in df.columns:
        df["weight"] = 1.0
        
    # Calculate weighted counts
    counts_all = df.groupby("category")["weight"].sum().reindex(CATEGORY_ORDER, fill_value=0)
    
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
                # Get top 1% documents by comment count
                total_docs = len(fr_df)
                top_n = max(10, int(np.ceil(total_docs * 0.01)))
                top_docs = fr_df.nlargest(top_n, "comment_count")["document_number"].tolist()
                # Exclude comments from those documents
                df_excluded = df[~df["document_number"].isin(top_docs)].copy()
                print(f"Excluding {top_n} top documents (top 1%) from donut chart")
        except Exception as e:
            print(f"Warning: Could not load FR data for exclusion: {e}")
    
    # Create figure with side-by-side if we have exclusion data
    if df_excluded is not None and len(df_excluded) > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # Left: All comments (weighted)
        wedges1, texts1 = ax1.pie(
            counts_all.values,
            colors=colors,
            startangle=90,
            wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2)
        )
        ax1.set_title(f"All Comments ({year})", fontsize=FONT_TITLE, fontweight="bold", pad=20)
        
        # Right: Excluding top 1% of documents (weighted)
        counts_excluded = df_excluded.groupby("category")["weight"].sum().reindex(CATEGORY_ORDER, fill_value=0)
        percentages_excluded = 100 * counts_excluded / counts_excluded.sum() if counts_excluded.sum() > 0 else percentages_all * 0
        
        wedges2, texts2 = ax2.pie(
            counts_excluded.values,
            colors=colors,
            startangle=90,
            wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2)
        )
        ax2.set_title(f"Excluding Top 1% Documents ({year})", fontsize=FONT_TITLE, fontweight="bold", pad=20)
        
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


def plot_agency_compass(df: pd.DataFrame, year: int, outdir: Path, top_n: Optional[int] = None) -> None:
    """
    Chart 2: Agency compass with quadrants.
    X = OC - ORG, Y = EXP + LOB
    Improved: Distribute labels across all quadrants
    Uses weighted metrics.
    All agencies are plotted; labeling scales with total number of agencies.
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
    
    # Select agencies per quadrant for labeling (scale with total number of agencies)
    # If top_n not specified, use ~20% of agencies per quadrant (minimum 2, maximum 15 per quadrant)
    total_agencies = len(df_metrics)
    if top_n is None:
        labels_per_quadrant = max(2, min(15, int(total_agencies * 0.05)))
    else:
        labels_per_quadrant = max(2, top_n // 4)
    
    agencies_to_label = []
    
    for q in [1, 2, 3, 4]:
        quadrant_agencies = df_metrics[df_metrics["quadrant"] == q].nlargest(labels_per_quadrant, "total")
        agencies_to_label.append(quadrant_agencies)
    
    agencies_to_label = pd.concat(agencies_to_label)
    
    # Also add overall top agencies if not already included (scale with total)
    if top_n is None:
        top_n_effective = min(30, max(10, int(total_agencies * 0.25)))
    else:
        top_n_effective = top_n
    top_overall = df_metrics.nlargest(top_n_effective, "total")
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
        "Bubble size = comment volume (weighted)",
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
    Uses weighted metrics.
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
    ax.set_xscale("symlog", linthresh=1)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xlabel("Total Comments (log scale, weighted)", fontsize=FONT_LABEL, fontweight="bold")
    ax.set_ylabel("Count by Category (log scale, weighted)", fontsize=FONT_LABEL, fontweight="bold")
    ax.set_title(f"Workload vs. Who Shows Up ({year})", fontsize=FONT_TITLE, fontweight="bold", pad=20)
    
    ax.legend(fontsize=FONT_LEGEND - 1, frameon=True, loc="upper left", framealpha=0.9)
    apply_clean_style(ax)
    
    plt.tight_layout()
    fig.savefig(outdir / f"workload_vs_makeup_{year}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: workload_vs_makeup_{year}.png")


def fit_exponential_trend(x_data: np.ndarray, y_data: np.ndarray, n_bins: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit exponential trend with tighter quantile-based bands (p10-p90) and smooth median curve.
    Median follows exponential growth at low x (~10^1), then smoothly transitions to approach y=x.
    Returns: (x_fit, y_median, y_p90, y_p10)
    """
    # Filter out zeros and negatives (allow y=0 for better near-zero handling)
    mask = (x_data > 0) & (y_data >= 0)
    x_clean = x_data[mask]
    y_clean = y_data[mask]
    
    if len(x_clean) < 10:
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    try:
        from scipy.interpolate import interp1d, UnivariateSpline
        
        # Use log-spaced bins for better coverage across orders of magnitude
        log_x_min = np.log10(x_clean.min())
        log_x_max = np.log10(x_clean.max())
        
        if log_x_max <= log_x_min:
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        # Create more bins for better resolution
        log_bins = np.linspace(log_x_min, log_x_max, n_bins + 1)
        bins = 10 ** log_bins
        
        # Calculate quantiles within each bin (p10, p50, p90) - tighter bands
        x_binned = []
        y_median = []
        y_p10 = []
        y_p90 = []
        
        for i in range(len(bins) - 1):
            bin_mask = (x_clean >= bins[i]) & (x_clean < bins[i + 1])
            if bin_mask.sum() >= 10:  # Need at least 10 points for p10/p90
                x_bin_center = np.sqrt(bins[i] * bins[i + 1])  # Geometric mean
                y_bin_values = y_clean[bin_mask]
                
                x_binned.append(x_bin_center)
                y_median.append(np.median(y_bin_values))
                y_p10.append(np.percentile(y_bin_values, 10))
                y_p90.append(np.percentile(y_bin_values, 90))
            elif bin_mask.sum() >= 5:  # 5-9 points: use p25/p75 as proxy
                x_bin_center = np.sqrt(bins[i] * bins[i + 1])
                y_bin_values = y_clean[bin_mask]
                x_binned.append(x_bin_center)
                y_median.append(np.median(y_bin_values))
                y_p10.append(np.percentile(y_bin_values, 25))  # Wider proxy
                y_p90.append(np.percentile(y_bin_values, 75))
            elif bin_mask.sum() >= 3:  # 3-4 points: use min/median/max
                x_bin_center = np.sqrt(bins[i] * bins[i + 1])
                y_bin_values = y_clean[bin_mask]
                x_binned.append(x_bin_center)
                y_median.append(np.median(y_bin_values))
                y_p10.append(np.min(y_bin_values))
                y_p90.append(np.max(y_bin_values))
        
        if len(x_binned) < 3:
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        x_binned = np.array(x_binned)
        y_median = np.array(y_median)
        y_p10 = np.array(y_p10)
        y_p90 = np.array(y_p90)
        
        # Sort by x for proper interpolation
        sort_idx = np.argsort(x_binned)
        x_binned = x_binned[sort_idx]
        y_median = y_median[sort_idx]
        y_p10 = y_p10[sort_idx]
        y_p90 = y_p90[sort_idx]
        
        # Use log-space for interpolation
        log_x_binned = np.log10(x_binned)
        y_min_positive = max(y_median[y_median > 0].min() if (y_median > 0).any() else 1, 1e-10)
        
        # Generate smooth curve for plotting
        log_x_fit = np.linspace(log_x_min, log_x_max, 500)
        x_fit = 10 ** log_x_fit
        
        # ===== MEDIAN TREND: Exponential growth at low x, then transition to y=x =====
        # Fit power law to low values (x < 10^2)
        low_x_mask = x_binned < 100
        if low_x_mask.sum() >= 3:
            log_x_low = np.log10(x_binned[low_x_mask])
            log_y_low = np.log10(np.maximum(y_median[low_x_mask], y_min_positive))
            coeffs = np.polyfit(log_x_low, log_y_low, 1)
            a_low = 10 ** coeffs[1]
            b_low = coeffs[0]
        else:
            # Fallback: use simple power law
            log_x_all = np.log10(x_binned)
            log_y_all = np.log10(np.maximum(y_median, y_min_positive))
            coeffs = np.polyfit(log_x_all, log_y_all, 1)
            a_low = 10 ** coeffs[1]
            b_low = coeffs[0]
        
        # Power law function
        def power_law(x, a, b):
            return a * (x ** b)
        
        # Transition point: where we start blending toward y=x
        # Use a smooth transition function: sigmoid-like blend
        transition_x = 10  # Start transition around x=10
        transition_width = 2.0  # Width of transition in log space
        
        # Calculate power law prediction
        y_power = power_law(x_fit, a_low, b_low)
        
        # Calculate y=x line
        y_equals_x = x_fit
        
        # Smooth transition: blend power law with y=x
        # Transition weight: 0 at low x, 1 at high x
        log_x_fit_vals = np.log10(x_fit)
        transition_weight = 1 / (1 + np.exp(-(log_x_fit_vals - np.log10(transition_x)) / transition_width))
        
        # Blend: at low x use power law, at high x use y=x, smooth transition in between
        y_median_fit = y_power * (1 - transition_weight) + y_equals_x * transition_weight
        
        # Ensure median doesn't exceed y=x (should approach from below)
        y_median_fit = np.minimum(y_median_fit, y_equals_x)
        
        # ===== CONFIDENCE BANDS: Smooth p10-p90 =====
        # Use cubic spline interpolation for smoother bands
        log_y_p10 = np.log10(np.maximum(y_p10, y_min_positive))
        log_y_p90 = np.log10(np.maximum(y_p90, y_min_positive))
        
        # Use UnivariateSpline for very smooth curves
        try:
            spline_p10 = UnivariateSpline(log_x_binned, log_y_p10, s=len(log_x_binned) * 0.1, k=3)
            spline_p90 = UnivariateSpline(log_x_binned, log_y_p90, s=len(log_x_binned) * 0.1, k=3)
            y_p10_fit = 10 ** spline_p10(log_x_fit)
            y_p90_fit = 10 ** spline_p90(log_x_fit)
        except:
            # Fallback to cubic interpolation if spline fails
            interp_p10 = interp1d(log_x_binned, log_y_p10, kind='cubic', 
                                   fill_value='extrapolate', bounds_error=False)
            interp_p90 = interp1d(log_x_binned, log_y_p90, kind='cubic', 
                                   fill_value='extrapolate', bounds_error=False)
            y_p10_fit = 10 ** interp_p10(log_x_fit)
            y_p90_fit = 10 ** interp_p90(log_x_fit)
        
        # Ensure proper ordering: p10 <= median <= p90
        y_p10_fit = np.maximum(np.minimum(y_p10_fit, y_median_fit), 0)
        y_p90_fit = np.maximum(y_p90_fit, y_median_fit)
        
        return x_fit, y_median_fit, y_p90_fit, y_p10_fit
        
    except Exception as e:
        # Fallback: simple binned approach with quantiles
        log_x_min = np.log10(x_clean.min())
        log_x_max = np.log10(x_clean.max())
        log_bins = np.linspace(log_x_min, log_x_max, n_bins + 1)
        bins = 10 ** log_bins
        
        x_binned = []
        y_median = []
        y_p10 = []
        y_p90 = []
        
        for i in range(len(bins) - 1):
            bin_mask = (x_clean >= bins[i]) & (x_clean < bins[i + 1])
            if bin_mask.sum() >= 10:
                x_bin_center = np.sqrt(bins[i] * bins[i + 1])
                y_bin_values = y_clean[bin_mask]
                x_binned.append(x_bin_center)
                y_median.append(np.median(y_bin_values))
                y_p10.append(np.percentile(y_bin_values, 10))
                y_p90.append(np.percentile(y_bin_values, 90))
            elif bin_mask.sum() >= 3:
                x_bin_center = np.sqrt(bins[i] * bins[i + 1])
                y_bin_values = y_clean[bin_mask]
                x_binned.append(x_bin_center)
                y_median.append(np.median(y_bin_values))
                y_p10.append(np.min(y_bin_values))
                y_p90.append(np.max(y_bin_values))
        
        if len(x_binned) < 3:
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        return np.array(x_binned), np.array(y_median), np.array(y_p90), np.array(y_p10)


def plot_workload_vs_citizen_by_agency(df: pd.DataFrame, year: int, outdir: Path) -> None:
    """
    Workload vs citizen input by agency (like Gini frontier).
    Shows relationship between total comments and citizen participation.
    Uses weighted metrics.
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
    
    # Fit and plot exponential trend with quantile-based bands (p10-p90)
    x_data = df_metrics["total"].values
    y_data = df_metrics[oc_count_col].values
    x_fit, y_median, y_p90, y_p10 = fit_exponential_trend(x_data, y_data)
    
    if len(x_fit) > 0:
        # Plot p10-p90 bands - tighter, cleaner bands
        ax.fill_between(
            x_fit,
            y_p10,
            y_p90,
            alpha=0.2,
            color="#9A8153",
            label="10th-90th percentile"
        )
        # Plot median trend line (exponential growth transitioning to y=x)
        ax.plot(
            x_fit,
            y_median,
            color="#9A8153",
            linewidth=2.5,
            alpha=0.9,
            label="Median trend"
        )
    
    # Add diagonal reference line (perfect citizen participation)
    max_total = df_metrics["total"].max()
    max_citizen = df_metrics[oc_count_col].max()
    max_val = max(max_total, max_citizen)
    ax.plot([0, max_val], [0, max_val], '--', color="#5C6670", alpha=0.3, linewidth=1, label="Perfect citizen participation")
    
    # Styling
    ax.set_xscale("symlog", linthresh=1)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xlabel("Total Comments by Agency (log scale, weighted)", fontsize=FONT_LABEL, fontweight="bold")
    ax.set_ylabel("Ordinary Citizen Comments (log scale, weighted)", fontsize=FONT_LABEL, fontweight="bold")
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
    
    CRITICAL FIX: Uses `weight_doc` (Document Specific Weight) instead of Stratum Weight.
    X-axis: True FR Total Volume
    Y-axis: Sum of `weight_doc` for Citizen comments.
    
    This ensures that for any single document, Y <= X.
    """
    if "document_number" not in df.columns:
        print(f"Warning: document_number not available for workload vs citizen by document ({year})")
        return
    
    # --- FIX: Deduplicate to ensure 1 row per comment, ignoring agency split ---
    df_doc_view = df.drop_duplicates(subset=["comment_id"])
    # --------------------------------------------------------------------------
        
    # Check if we have weights
    if "weight_doc" not in df_doc_view.columns:
        print("Warning: weight_doc not available. Using standard weight.")
        df_doc_view["weight_doc"] = df_doc_view.get("weight", 1.0)
        
    if "comment_count" not in df_doc_view.columns:
        print("Warning: True comment_count not available.")
        return
    
    # Group by document_number
    # For Total (X): We can take the first value of 'comment_count' (True Total from FR)
    # For Citizen (Y): We sum 'weight_doc' for citizens.
    
    doc_grp = df_doc_view.groupby("document_number").agg({
        "comment_count": "first"
    }).rename(columns={"comment_count": "true_total"})
    
    # Calculate weighted citizen count using DOCUMENT WEIGHT
    citizen_df = df_doc_view[df_doc_view["category"] == "Ordinary Citizen"]
    citizen_counts = citizen_df.groupby("document_number")["weight_doc"].sum().rename("estimated_citizen_total")
    
    # Join
    doc_grp = doc_grp.join(citizen_counts, how="left").fillna(0)
    
    if doc_grp.empty:
        print(f"Warning: No document-level data ({year})")
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Scatter plot
    ax.scatter(
        doc_grp["true_total"],
        doc_grp["estimated_citizen_total"],
        s=40,
        alpha=0.4,
        color=COLOR_PALETTE["Ordinary Citizen"],
        edgecolors="white",
        linewidths=0.3
    )
    
    # Fit and plot exponential trend with quantile-based bands (p10-p90)
    x_data = doc_grp["true_total"].values
    y_data = doc_grp["estimated_citizen_total"].values
    x_fit, y_median, y_p90, y_p10 = fit_exponential_trend(x_data, y_data)
    
    if len(x_fit) > 0:
        # Plot p10-p90 bands - tighter, cleaner bands
        ax.fill_between(
            x_fit,
            y_p10,
            y_p90,
            alpha=0.2,
            color="#9A8153",
            label="10th-90th percentile"
        )
        # Plot median trend line (exponential growth transitioning to y=x)
        ax.plot(
            x_fit,
            y_median,
            color="#9A8153",
            linewidth=2.5,
            alpha=0.9,
            label="Median trend"
        )
    
    # Add diagonal reference line
    max_total = doc_grp["true_total"].max()
    ax.plot([0, max_total], [0, max_total], '--', color="#5C6670", alpha=0.3, linewidth=1, label="Perfect citizen participation")
    
    # Styling
    ax.set_xscale("symlog", linthresh=1)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xlabel("Total Comments by Document (True FR Volume)", fontsize=FONT_LABEL, fontweight="bold")
    ax.set_ylabel("Ordinary Citizen Comments (Estimated)", fontsize=FONT_LABEL, fontweight="bold")
    ax.set_title(f"Workload vs Citizen Input (document_number, {year})", fontsize=FONT_TITLE, fontweight="bold", pad=20)
    
    ax.legend(fontsize=FONT_LEGEND - 1, frameon=True, loc="upper left", framealpha=0.9)
    apply_clean_style(ax)
    
    plt.tight_layout()
    fig.savefig(outdir / f"workload_vs_citizen_by_document_number_{year}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: workload_vs_citizen_by_document_number_{year}.png")


def compute_lorenz_and_gini(counts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute Lorenz curve and Gini coefficient following efficiency_frontier.py pattern.
    Returns: (lorenz_x, lorenz_y, gini)
    Filters out NaN values before processing.
    """
    # Filter out NaN values
    counts_clean = counts[~np.isnan(counts)]
    if len(counts_clean) == 0:
        # Degenerate case; return equality line
        x = np.linspace(0, 1, 2)
        y = x.copy()
        return x, y, 0.0
    
    asc = np.sort(np.maximum(counts_clean, 0))
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


def prepare_fr_stats(df: pd.DataFrame, fr_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare FR statistics with citizen estimates.
    Returns fr_stats DataFrame with estimated_citizen_count, has_data, known_count columns.
    
    FIX: Now correctly identifies ALL sampled documents, not just those with citizen comments.
    """
    # Deduplicate to ensure 1 row per comment
    df_doc_view = df.drop_duplicates(subset=["comment_id"])
    
    # Identify ALL sampled documents (not just those with citizen comments)
    sampled_documents = pd.Series(df_doc_view["document_number"].unique(), name="document_number")
    sampled_documents = pd.DataFrame({"document_number": sampled_documents, "is_sampled": True})
    
    # Calculate citizen estimates for sampled documents using DOCUMENT WEIGHTS
    if "weight_doc" not in df_doc_view.columns:
        df_doc_view["weight_doc"] = 1.0
        
    citizen_df = df_doc_view[df_doc_view["category"] == "Ordinary Citizen"]
    citizen_counts = citizen_df.groupby("document_number")["weight_doc"].sum().rename("estimated_citizen_count")
    
    # Merge estimates back to FULL FR dataframe
    cols_to_load = ["document_number", "comment_count"]
    if "count_source" in fr_df.columns:
        cols_to_load.append("count_source")
    if "publication_date" in fr_df.columns:
        cols_to_load.append("publication_date")
    if "comments_close_on" in fr_df.columns:
        cols_to_load.append("comments_close_on")
        
    fr_stats = fr_df[cols_to_load].copy()
    
    # Mark which documents were sampled
    fr_stats = fr_stats.merge(sampled_documents, on="document_number", how="left")
    fr_stats["is_sampled"] = fr_stats["is_sampled"].fillna(False)
    
    # Merge citizen counts
    fr_stats = fr_stats.merge(citizen_counts, on="document_number", how="left")
    
    # Fill NaNs: 
    # - Documents with 0 comments have 0 citizen comments
    # - Sampled documents with no citizen comments have 0 citizen comments (not NaN)
    fr_stats.loc[fr_stats["comment_count"] == 0, "estimated_citizen_count"] = 0
    fr_stats.loc[fr_stats["is_sampled"] & fr_stats["estimated_citizen_count"].isna(), "estimated_citizen_count"] = 0
    
    # Define "has_data" as documents where we either have a sample OR the total count is 0
    fr_stats["has_data"] = (fr_stats["comment_count"] == 0) | (fr_stats["is_sampled"])
    
    # For documents with data, "has_citizens" is true if estimate > 0.5
    fr_stats["has_citizens"] = (fr_stats["estimated_citizen_count"] > 0.5).astype(int)
    
    # Identify documents with known vs unknown comment counts
    if "count_source" in fr_stats.columns:
        fr_stats["known_count"] = fr_stats["count_source"] != "unknown"
    else:
        fr_stats["known_count"] = True
    
    return fr_stats


def plot_comment_distribution_and_participation(fr_stats: pd.DataFrame, year: int, outdir: Path) -> None:
    """
    Figure 1: Comment volume distribution (histogram) + Participation rate (bar chart).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel 1: Distribution of comment counts per document (Use FULL UNIVERSE)
    ax1 = axes[0]
    comment_counts = fr_stats["comment_count"].values
    
    # Plot only docs with >= 1 comment in the log histogram
    counts_nonzero = comment_counts[comment_counts > 0]
    
    if len(counts_nonzero) > 0:
        # Define bins - ensure we have valid range
        max_comments = max(counts_nonzero.max(), 1)
        min_comments = max(counts_nonzero.min(), 1)
        
        if max_comments > min_comments:
            # Use log bins
            bins = np.logspace(np.log10(min_comments), np.log10(max_comments), 30)
        else:
            # Fallback for edge case
            bins = np.linspace(min_comments, max_comments + 1, 30)
        
        n, bins_edges, patches = ax1.hist(counts_nonzero, bins=bins, color="#5C6670", alpha=0.7, edgecolor="white", linewidth=0.5)
        ax1.set_xscale("log")
        
        # Add statistics
        pct_low = (counts_nonzero <= 5).sum() / len(counts_nonzero) * 100
        ax1.text(0.7, 0.95, f"{pct_low:.1f}% of active docs\nhave ≤5 comments", transform=ax1.transAxes, 
                 fontsize=10, alpha=0.7, va="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    else:
        ax1.text(0.5, 0.5, "No documents with comments", ha="center", va="center", transform=ax1.transAxes)
    
    ax1.set_xlabel("Comments per Document (log scale)", fontsize=FONT_LABEL, fontweight="bold")
    ax1.set_ylabel("Number of Documents", fontsize=FONT_LABEL, fontweight="bold")
    ax1.set_title("Distribution of Comment Volume (Non-Zero)", fontsize=FONT_TITLE - 2, fontweight="bold")
    apply_clean_style(ax1)
    
    # Panel 2: Documents with zero comments (ONLY KNOWN COUNTS)
    ax2 = axes[1]
    # Filter to only documents where we know the comment count (exclude unknown-count docs)
    fr_stats_known = fr_stats[fr_stats["known_count"]].copy()
    total_docs_known = len(fr_stats_known)
    zero_docs = (fr_stats_known["comment_count"] == 0).sum()
    docs_with_comments = total_docs_known - zero_docs
    
    # Count unknown-count docs for annotation
    unknown_count_docs = (~fr_stats["known_count"]).sum() if "known_count" in fr_stats.columns else 0
    
    categories = ["Documents\nwith Comments", "Documents\nwith Zero Comments"]
    counts = [docs_with_comments, zero_docs]
    colors_bar = [COLOR_PALETTE["Ordinary Citizen"], "#DDE3EA"]
    
    bars = ax2.bar(categories, counts, color=colors_bar, alpha=0.8, edgecolor="white", linewidth=2)
    ax2.set_ylabel("Number of Documents", fontsize=FONT_LABEL, fontweight="bold")
    ax2.set_title("Comment Participation Rate", fontsize=FONT_TITLE - 2, fontweight="bold")
    
    # Add percentages on top of bars in black font
    max_height = max(counts)
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        pct = count / total_docs_known * 100 if total_docs_known > 0 else 0
        # Position text on top of bar
        y_pos = height + max_height * 0.02
        ax2.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{count:,}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight="bold", color='black')
    
    # Add padding at top to prevent overlap
    ax2.set_ylim(0, max_height * 1.15)
    apply_clean_style(ax2)
    
    fig.suptitle(f"The Participation Gap ({year})", fontsize=FONT_TITLE + 2, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outdir / f"participation_gap_{year}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: participation_gap_{year}.png")


def plot_citizen_participation_by_volume(fr_stats: pd.DataFrame, year: int, outdir: Path) -> None:
    """
    Figure 2: Citizen participation rate by comment volume (bar chart).
    Fixed logic: Calculate percentage correctly for each bin.
    Only uses documents where we have data (sampled documents).
    """
    # Only use documents where we HAVE DATA (sampled or zero comments)
    # For zero-comment docs, we know has_citizens = 0
    # For sampled docs with comments, we use has_citizens
    valid_stats = fr_stats[fr_stats["has_data"]].copy()
    
    # Bin documents by comment volume (only non-zero for this chart)
    valid_stats_nonzero = valid_stats[valid_stats["comment_count"] > 0].copy()
    
    if len(valid_stats_nonzero) == 0:
        print(f"Warning: No documents with comments for citizen participation chart ({year})")
        return
    
    # Bin documents
    valid_stats_nonzero["comment_bin"] = pd.cut(
        valid_stats_nonzero["comment_count"], 
        bins=[0, 1, 5, 10, 50, 100, float('inf')],
        labels=["1", "2-5", "6-10", "11-50", "51-100", "100+"],
        include_lowest=False
    )
    
    # Calculate percentage of documents with citizen comments in each bin
    # has_citizens is already calculated (1 if estimated_citizen_count > 0.5, else 0)
    bin_stats = valid_stats_nonzero.groupby("comment_bin").agg({
        "has_citizens": "mean",  # Mean gives percentage
        "comment_count": "count"  # Total docs in bin
    }).reset_index()
    
    bin_stats = bin_stats[bin_stats["comment_count"] > 0]  # Only bins with data
    
    if len(bin_stats) == 0:
        print(f"Warning: No valid bins for citizen participation chart ({year})")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_pos = np.arange(len(bin_stats))
    bars = ax.bar(x_pos, bin_stats["has_citizens"] * 100, 
                   color=COLOR_PALETTE["Ordinary Citizen"], alpha=0.7, edgecolor="white", linewidth=1.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(bin_stats["comment_bin"].astype(str), rotation=0)
    ax.set_ylabel("% Documents with Citizen Comments", fontsize=FONT_LABEL, fontweight="bold")
    ax.set_xlabel("Comments per Document", fontsize=FONT_LABEL, fontweight="bold")
    ax.set_title(f"Citizen Participation by Document Volume ({year})", fontsize=FONT_TITLE - 2, fontweight="bold")
    ax.set_ylim(0, 100)
    
    # Add value labels
    for bar, val, count in zip(bars, bin_stats["has_citizens"] * 100, bin_stats["comment_count"]):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%\n(n={count})',
                ha='center', va='bottom', fontsize=9, fontweight="bold")
    apply_clean_style(ax)
    
    plt.tight_layout()
    fig.savefig(outdir / f"citizen_participation_by_volume_{year}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: citizen_participation_by_volume_{year}.png")


def plot_lorenz_concentration(fr_stats: pd.DataFrame, year: int, outdir: Path) -> None:
    """
    Figure 3: Lorenz curve with Gini coefficient, following efficiency_frontier.py pattern.
    Filters out NaN values (documents with unknown comment counts) before processing.
    """
    # Use FULL UNIVERSE (zeros included, though they don't add volume)
    # Filter out NaN values (documents with unknown comment counts)
    comment_counts = fr_stats["comment_count"].values
    comment_counts_clean = comment_counts[~np.isnan(comment_counts)]
    
    if len(comment_counts_clean) == 0:
        print(f"Warning: No valid comment counts for Lorenz curve ({year})")
        return
    
    # Compute Lorenz curve and Gini
    lorenz_x, lorenz_y, gini = compute_lorenz_and_gini(comment_counts_clean)
    
    # Also compute efficiency frontier (coverage at top x% of docs)
    ordered = np.sort(comment_counts_clean)[::-1]  # Descending
    total = ordered.sum()
    if total <= 0:
        print(f"Warning: No comments for Lorenz curve ({year})")
        return
    
    cum = np.cumsum(ordered)
    n = ordered.size
    frontier_x = np.arange(1, n + 1) / n * 100
    frontier_y = cum / total * 100
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: Efficiency Frontier (coverage curve)
    ax1 = axes[0]
    ax1.plot(frontier_x, frontier_y, label="Coverage frontier", color="#1f77b4", lw=2)
    ax1.plot([0, 100], [0, 100], ls="--", color="#777777", label="Uniform attention (y = x)")
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 100)
    ax1.set_xlabel("Share of documents selected (best-first)", fontsize=FONT_LABEL, fontweight="bold")
    ax1.set_ylabel("Share of total comments captured", fontsize=FONT_LABEL, fontweight="bold")
    ax1.set_title(f"Public attention is concentrated ({year})", fontsize=FONT_TITLE - 2, fontweight="bold")
    ax1.grid(True, alpha=0.2)
    
    # Simplified: Only 3 points - 0.05%, 0.1%, and 0.2% of documents
    doc_shares = [0.05, 0.1, 0.2]  # Percentage of documents
    colors = ["#2ca02c", "#ff7f0e", "#d62728"]  # Green, Orange, Red
    
    # Calculate comment percentages captured at each document share
    for i, doc_share in enumerate(doc_shares):
        # Find the index corresponding to this document share
        idx = max(0, int(np.ceil(doc_share / 100 * len(frontier_x))) - 1)
        idx = min(idx, len(frontier_x) - 1)
        
        x_pct = frontier_x[idx]
        y_pct = frontier_y[idx]
        color = colors[i]
        
        # Draw point on curve
        ax1.scatter([x_pct], [y_pct], color=color, s=60, zorder=3, edgecolors="white", linewidths=1.5)
        
        # Format label
        if doc_share < 0.1:
            label = f"{doc_share:.2f}% docs\n→ {y_pct:.1f}% cmt"
        else:
            label = f"{doc_share:.1f}% docs\n→ {y_pct:.1f}% cmt"
        
        # Position labels on the right side, aligned with their y-coordinates to avoid arrow tangling
        # Use different x positions to prevent arrows from crossing
        text_x = 25 + (i * 8)  # Stagger x positions: 25, 33, 41
        text_y = y_pct  # Align with curve point's y-coordinate
        
        ax1.annotate(
            label,
            xy=(x_pct, y_pct),
            xytext=(text_x, text_y),
            fontsize=9,
            color=color,
            fontweight="bold",
            ha="left",
            va="center",
            arrowprops=dict(
                arrowstyle='->',
                color=color,
                lw=1.5,
                alpha=0.7,
                connectionstyle="arc3,rad=0.1"
            ),
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, edgecolor=color, linewidth=1.2)
        )
    
    ax1.legend(loc="lower right", fontsize=9)
    
    # Right: Lorenz Curve
    ax2 = axes[1]
    ax2.plot(lorenz_x * 100, lorenz_y * 100, color="#1f77b4", lw=2)
    ax2.plot([0, 100], [0, 100], ls="--", color="#777777", label="Perfect equality")
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 100)
    ax2.set_xlabel("Cumulative share of documents (worst → best)", fontsize=FONT_LABEL, fontweight="bold")
    ax2.set_ylabel("Cumulative share of comments", fontsize=FONT_LABEL, fontweight="bold")
    ax2.set_title(f"Lorenz curve (Gini = {gini:.3f})", fontsize=FONT_TITLE - 2, fontweight="bold")
    ax2.grid(True, alpha=0.2)
    ax2.legend()
    
    fig.suptitle(f"Comment Concentration Analysis ({year})", fontsize=FONT_TITLE + 2, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outdir / f"comment_concentration_{year}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: comment_concentration_{year}.png")


def verify_zero_comment_dates(fr_stats: pd.DataFrame, year: int) -> Optional[str]:
    """
    Verify if zero-comment docs are clustered near the end of the year (still open for comment).
    Returns annotation text if significant clustering found, None otherwise.
    """
    if "publication_date" not in fr_stats.columns:
        return None
    
    try:
        fr_stats["pub_date"] = pd.to_datetime(fr_stats["publication_date"], errors="coerce")
        zero_docs = fr_stats[fr_stats["comment_count"] == 0].copy()
        
        if len(zero_docs) == 0:
            return None
        
        zero_docs["month"] = zero_docs["pub_date"].dt.to_period("M")
        monthly_counts = zero_docs["month"].value_counts().sort_index()
        
        # Check last 3 months
        if len(monthly_counts) >= 3:
            last_3_months = monthly_counts.tail(3)
            last_3_total = last_3_months.sum()
            total_zero = len(zero_docs)
            pct_last_3 = last_3_total / total_zero * 100 if total_zero > 0 else 0
            
            if pct_last_3 > 30:  # More than 30% in last 3 months
                return f"Note: {pct_last_3:.1f}% of zero-comment docs published in last 3 months (may still be open for comment)"
        
        return None
    except Exception as e:
        print(f"Warning: Could not analyze zero-comment dates: {e}")
        return None


def plot_comment_distribution_roi(df: pd.DataFrame, year: int, outdir: Path, fr_csv_path: Optional[Path] = None) -> None:
    """
    High-ROI visualization: Shows that most documents get few/no comments and few citizens participate.
    This is the key story visualization.
    
    Now split into separate clean charts:
    1. Comment distribution + Participation rate
    2. Citizen participation by volume
    3. Lorenz curve with Gini coefficient
    
    Uses full FR metadata (fr_df) as the primary source for document-level histograms.
    """
    # Load FR data to get complete document universe
    fr_df = None
    if fr_csv_path and fr_csv_path.exists():
        try:
            fr_df = pd.read_csv(fr_csv_path)
        except Exception as e:
            print(f"Warning: Could not load FR data: {e}")
            
    if fr_df is None:
        print(f"Error: FR metadata required for ROI chart ({year}) to show true distribution.")
        return
    
    # Prepare FR statistics
    fr_stats = prepare_fr_stats(df, fr_df)
    
    # Verify zero-comment rate
    zero_comment_note = verify_zero_comment_dates(fr_stats, year)
    if zero_comment_note:
        print(f"Zero-comment analysis: {zero_comment_note}")
    
    # Generate separate charts
    plot_comment_distribution_and_participation(fr_stats, year, outdir)
    plot_citizen_participation_by_volume(fr_stats, year, outdir)
    plot_lorenz_concentration(fr_stats, year, outdir)


# ============================================================================
# COMPLEX CONTINUUM PAGE
# ============================================================================

def plot_compass_with_ribbon(df: pd.DataFrame, year: int, outdir: Path, top_n: Optional[int] = None) -> None:
    """
    Panel A: Compass with weighted median ribbon and IQR band.
    Improved: Add more context about what the ribbon shows, highlight key insights
    Uses weighted metrics.
    All agencies are plotted; highlighting scales with total number of agencies.
    """
    df_metrics = compute_agency_metrics(df)
    
    if df_metrics.empty:
        print(f"Warning: No agency metrics for ribbon compass ({year})")
        return
    
    # Compute ribbon (uses all agencies)
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
    # Scale highlighting with total number of agencies
    total_agencies = len(df_metrics)
    if top_n is None:
        top_n_effective = min(20, max(8, int(total_agencies * 0.15)))
    else:
        top_n_effective = top_n
    top_agencies = df_metrics.nlargest(top_n_effective, "total")
    
    # Get representatives from each quadrant
    def get_quadrant(x, y):
        if x >= 0 and y >= 0.5: return 1
        elif x < 0 and y >= 0.5: return 2
        elif x >= 0 and y < 0.5: return 3
        else: return 4
    
    df_metrics["quadrant"] = df_metrics.apply(lambda r: get_quadrant(r["X"], r["Y"]), axis=1)
    highlights_list = [top_agencies]
    
    # Scale per-quadrant highlights with total agencies
    per_quadrant_n = max(2, min(5, int(total_agencies * 0.04)))
    for q in [1, 2, 3, 4]:
        q_agencies = df_metrics[df_metrics["quadrant"] == q]
        if len(q_agencies) > 0:
            highlights_list.append(q_agencies.nlargest(per_quadrant_n, "total"))
    
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
    Panel B: Ranked stream graph showing composition across documents sorted by comment volume.
    
    X-axis: Document rank (1, 2, 3, ...) - each document gets equal visual weight.
    Documents are sorted by comment count (ascending), filtering out documents with 0 comments.
    Shows how comment makeup changes as documents receive more comments.
    Secondary x-axis shows comment count reference points.
    """
    # Check if document_number is available
    if "document_number" not in df.columns:
        print(f"Warning: document_number not available for stream graph ({year})")
        return
    
    # --- FIX: Deduplicate to ensure 1 row per comment ---
    df_doc_view = df.drop_duplicates(subset=["comment_id"])
    # ----------------------------------------------------
    
    # Ensure weight_doc exists for document-level reconstruction
    if "weight_doc" not in df_doc_view.columns:
        if "weight" in df_doc_view.columns:
            df_doc_view["weight_doc"] = df_doc_view["weight"]
        else:
            df_doc_view["weight_doc"] = 1.0
    
    # Get true comment count per document
    if "comment_count" not in df_doc_view.columns:
        # Calculate from sampled data using weight_doc
        doc_counts = df_doc_view.groupby("document_number")["weight_doc"].sum().rename("comment_count")
        df_doc_view = df_doc_view.merge(doc_counts, on="document_number", how="left")
    
    # Filter out documents with 0 comments
    df_doc_view = df_doc_view[df_doc_view["comment_count"] > 0].copy()
    
    if len(df_doc_view) == 0:
        print(f"Warning: No documents with comments for stream graph ({year})")
        return
    
    # Calculate document-level metrics
    # Group by document_number and category, sum weights
    doc_cat_counts = df_doc_view.groupby(["document_number", "category"])["weight_doc"].sum().rename("n").reset_index()
    
    # Calculate total comments per document
    doc_totals = doc_cat_counts.groupby("document_number")["n"].sum().rename("total").reset_index()
    
    # Merge to get shares
    doc_metrics = doc_cat_counts.merge(doc_totals, on="document_number", how="left")
    doc_metrics["share"] = doc_metrics["n"] / doc_metrics["total"].where(doc_metrics["total"] > 0, 1)
    
    # Pivot to wide format
    doc_wide = doc_metrics.pivot(index="document_number", columns="category", values=["share", "n"]).fillna(0.0)
    doc_wide.columns = ["_".join(col).strip() for col in doc_wide.columns.values]
    doc_wide = doc_wide.reset_index()
    
    # Merge back total and comment_count
    doc_wide = doc_wide.merge(doc_totals, on="document_number", how="left")
    doc_wide = doc_wide.merge(df_doc_view[["document_number", "comment_count"]].drop_duplicates(), on="document_number", how="left")
    
    # Ensure all category share columns exist
    for cat in CATEGORY_ORDER:
        share_col = f"share_{cat}"
        if share_col not in doc_wide.columns:
            doc_wide[share_col] = 0.0
    
    # Sort documents by comment count (ascending)
    doc_sorted = doc_wide.sort_values("comment_count").reset_index(drop=True)
    
    # Add document rank (1-based index) for x-axis - gives each document equal visual weight
    doc_sorted["doc_rank"] = np.arange(1, len(doc_sorted) + 1)
    
    # Prepare SHARES (percentages) for each category
    shares_data = []
    for cat in CATEGORY_ORDER:
        share_col = f"share_{cat}"
        if share_col in doc_sorted.columns:
            shares_data.append(doc_sorted[share_col].values * 100)  # Convert to %
        else:
            shares_data.append(np.zeros(len(doc_sorted)))
    
    shares = np.array(shares_data)
    
    # Apply moving average smoothing
    # Use a window size relative to number of documents (0.08 = heavier smoothing)
    window_size = max(5, int(len(doc_sorted) * 0.08))
    if window_size % 2 == 0:
        window_size += 1
    
    from scipy.ndimage import uniform_filter1d
    shares_smooth = np.array([uniform_filter1d(row, size=window_size, mode="nearest") for row in shares])
    
    # Normalize again after smoothing to ensure it sums to 100% exactly
    shares_smooth_sum = shares_smooth.sum(axis=0, keepdims=True)
    shares_smooth_sum[shares_smooth_sum == 0] = 1
    shares_norm = shares_smooth / shares_smooth_sum * 100
    
    # Create figure with gridspec: main stream on top, slice bars below
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.25)
    
    ax = fig.add_subplot(gs[0])
    ax_slices = fig.add_subplot(gs[1])
    
    # Stacked area plot with PERCENTAGES
    colors = get_category_colors()
    ax.stackplot(
        doc_sorted["doc_rank"],
        *shares_norm,
        labels=CATEGORY_ORDER,
        colors=colors,
        alpha=0.85
    )
    
    # Calculate p10, p50, p90 positions for slice bars below (don't draw on main chart)
    percentiles = [10, 50, 90]
    percentile_indices = {}
    for p in percentiles:
        idx = int(len(doc_sorted) * p / 100)
        idx = max(0, min(idx, len(doc_sorted) - 1))
        percentile_indices[p] = idx
    
    # Styling for main stream chart
    ax.set_xlim(0, len(doc_sorted) + 1)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Makeup of Comments (%)", fontsize=FONT_LABEL, fontweight="bold", labelpad=10)
    ax.set_title(
        f"Comment Composition by Document Volume ({year})",
        fontsize=FONT_TITLE,
        fontweight="bold",
        pad=15
    )
    
    # Create 6 tick intervals based on comment count milestones
    tick_positions = [1]
    tick_labels_main = ["1"]
    
    min_comments = int(doc_sorted["comment_count"].min())
    max_comments = int(doc_sorted["comment_count"].max())
    
    milestones = [10, 50, 100, 500, 1000, 10000, 100000, 1000000]
    milestones = [m for m in milestones if m <= max_comments and m > min_comments]
    milestones = milestones[:5]
    
    for milestone in milestones:
        matching_docs = doc_sorted[doc_sorted["comment_count"] >= milestone]
        if len(matching_docs) > 0:
            rank_at_milestone = matching_docs.iloc[0]["doc_rank"]
            if rank_at_milestone not in tick_positions:
                tick_positions.append(rank_at_milestone)
                tick_labels_main.append(f"{milestone:,}")
    
    if len(doc_sorted) > 0:
        top_rank = len(doc_sorted)
        if top_rank not in tick_positions:
            tick_positions.append(top_rank)
            tick_labels_main.append(f"{max_comments:,}")
    
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels_main, fontsize=10, rotation=0)
    ax.set_xlabel("Documents, ordered by number comments", fontsize=FONT_LABEL, fontweight="bold", labelpad=10)
    
    ax.legend(title="Legend", fontsize=FONT_LEGEND - 1, title_fontsize=FONT_LEGEND, 
              frameon=True, loc="upper left", bbox_to_anchor=(1, 1), framealpha=0.9)
    apply_clean_style(ax)
    
    # ===== Draw p10, p50, p90 slice bars below =====
    slice_y_positions = [0.7, 0.4, 0.1]  # y positions for p10, p50, p90 bars
    bar_height = 0.2
    
    for i, p in enumerate(percentiles):
        idx = percentile_indices[p]
        y_pos = slice_y_positions[i]
        
        # Get composition at this percentile (use smoothed values for consistency)
        shares_at_p = [shares_norm[cat_idx, idx] for cat_idx in range(len(CATEGORY_ORDER))]
        
        # Draw stacked horizontal bar
        left = 0
        for share, color in zip(shares_at_p, colors):
            ax_slices.barh(y_pos, share, left=left, height=bar_height, color=color, edgecolor="white", linewidth=1)
            left += share
        
        # Label for percentile
        comment_count_at_p = int(doc_sorted.iloc[idx]["comment_count"])
        ax_slices.text(-2, y_pos, f"p{p}\n({comment_count_at_p:,} comments)", 
                       ha="right", va="center", fontsize=10, fontweight="bold")
    
    ax_slices.set_xlim(0, 100)
    ax_slices.set_ylim(0, 1)
    ax_slices.axis("off")
    ax_slices.set_title("Composition at Percentiles", fontsize=FONT_LABEL, fontweight="bold", loc="left")
    
    plt.tight_layout()
    fig.savefig(outdir / f"ranked_stream_composition_{year}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: ranked_stream_composition_{year}.png")


def plot_ranked_stream_experiments(df: pd.DataFrame, year: int, outdir: Path) -> None:
    """
    Experimental alternative projections for the ranked stream composition.
    Creates multiple visualizations exploring different ways to show the interaction
    between # comments, # documents, and document density.
    """
    if "document_number" not in df.columns:
        print(f"Warning: document_number not available for stream experiments ({year})")
        return
    
    df_doc_view = df.drop_duplicates(subset=["comment_id"])
    
    if "weight_doc" not in df_doc_view.columns:
        df_doc_view["weight_doc"] = df_doc_view.get("weight", 1.0)
    
    if "comment_count" not in df_doc_view.columns:
        doc_counts = df_doc_view.groupby("document_number")["weight_doc"].sum().rename("comment_count")
        df_doc_view = df_doc_view.merge(doc_counts, on="document_number", how="left")
    
    df_doc_view = df_doc_view[df_doc_view["comment_count"] > 0].copy()
    
    if len(df_doc_view) == 0:
        return
    
    # Calculate document-level metrics
    doc_cat_counts = df_doc_view.groupby(["document_number", "category"])["weight_doc"].sum().rename("n").reset_index()
    doc_totals = doc_cat_counts.groupby("document_number")["n"].sum().rename("total").reset_index()
    doc_metrics = doc_cat_counts.merge(doc_totals, on="document_number", how="left")
    doc_metrics["share"] = doc_metrics["n"] / doc_metrics["total"].where(doc_metrics["total"] > 0, 1)
    
    doc_wide = doc_metrics.pivot(index="document_number", columns="category", values="share").fillna(0.0)
    doc_wide = doc_wide.reset_index()
    doc_wide = doc_wide.merge(doc_totals, on="document_number", how="left")
    doc_wide = doc_wide.merge(df_doc_view[["document_number", "comment_count"]].drop_duplicates(), on="document_number", how="left")
    
    for cat in CATEGORY_ORDER:
        if cat not in doc_wide.columns:
            doc_wide[cat] = 0.0
    
    doc_sorted = doc_wide.sort_values("comment_count").reset_index(drop=True)
    colors = get_category_colors()
    
    # ===== EXPERIMENT 1: Binned Composition Bars =====
    # Shows average composition per comment-count bin with # documents labeled
    fig1, ax1 = plt.subplots(figsize=(14, 8))
    
    # Define comment count bins
    bins = [0, 5, 10, 50, 100, 500, 1000, 10000, float('inf')]
    bin_labels = ["1-5", "6-10", "11-50", "51-100", "101-500", "501-1K", "1K-10K", "10K+"]
    
    doc_sorted["comment_bin"] = pd.cut(doc_sorted["comment_count"], bins=bins, labels=bin_labels, include_lowest=True)
    
    # Calculate mean composition and document count per bin
    bin_stats = []
    for bin_label in bin_labels:
        bin_data = doc_sorted[doc_sorted["comment_bin"] == bin_label]
        if len(bin_data) > 0:
            stats = {"bin": bin_label, "n_docs": len(bin_data)}
            for cat in CATEGORY_ORDER:
                stats[cat] = bin_data[cat].mean() * 100
            bin_stats.append(stats)
    
    if len(bin_stats) > 0:
        bin_df = pd.DataFrame(bin_stats)
        x_pos = np.arange(len(bin_df))
        
        # Stacked bar chart
        bottom = np.zeros(len(bin_df))
        for cat, color in zip(CATEGORY_ORDER, colors):
            values = bin_df[cat].values
            ax1.bar(x_pos, values, bottom=bottom, color=color, label=cat, edgecolor="white", linewidth=0.5)
            bottom += values
        
        # Add document count labels on top
        for i, (_, row) in enumerate(bin_df.iterrows()):
            ax1.text(i, 102, f"n={int(row['n_docs'])}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(bin_df["bin"], fontsize=10)
        ax1.set_xlabel("Comments per Document (bin)", fontsize=FONT_LABEL, fontweight="bold")
        ax1.set_ylabel("Average Composition (%)", fontsize=FONT_LABEL, fontweight="bold")
        ax1.set_ylim(0, 115)
        ax1.set_title(f"Composition by Comment Volume Bin ({year})\nShowing # documents per bin", 
                      fontsize=FONT_TITLE, fontweight="bold")
        ax1.legend(title="Legend", fontsize=FONT_LEGEND - 1, frameon=True, loc="upper left", bbox_to_anchor=(1, 1))
        apply_clean_style(ax1)
    
    plt.tight_layout()
    fig1.savefig(outdir / f"stream_experiment_binned_{year}.png", dpi=300, bbox_inches="tight")
    plt.close(fig1)
    print(f"[OK] Saved: stream_experiment_binned_{year}.png")
    
    # ===== EXPERIMENT 2: Document-Weighted Area with Density Markers =====
    # X-axis shows document rank (equal visual weight per doc), with markers showing comment density
    fig2, (ax2_main, ax2_density) = plt.subplots(2, 1, figsize=(16, 9), 
                                                   gridspec_kw={'height_ratios': [4, 1]}, sharex=True)
    
    doc_sorted["doc_rank"] = np.arange(1, len(doc_sorted) + 1)
    
    # Prepare shares
    shares_data = np.array([[doc_sorted[cat].values[i] * 100 for i in range(len(doc_sorted))] for cat in CATEGORY_ORDER])
    
    # Smooth
    from scipy.ndimage import uniform_filter1d
    window_size = max(5, int(len(doc_sorted) * 0.08))
    if window_size % 2 == 0:
        window_size += 1
    shares_smooth = np.array([uniform_filter1d(row, size=window_size, mode="nearest") for row in shares_data])
    shares_smooth_sum = shares_smooth.sum(axis=0, keepdims=True)
    shares_smooth_sum[shares_smooth_sum == 0] = 1
    shares_norm = shares_smooth / shares_smooth_sum * 100
    
    ax2_main.stackplot(doc_sorted["doc_rank"], *shares_norm, labels=CATEGORY_ORDER, colors=colors, alpha=0.85)
    ax2_main.set_ylabel("Composition (%)", fontsize=FONT_LABEL, fontweight="bold")
    ax2_main.set_ylim(0, 100)
    ax2_main.set_title(f"Equal Document Weight with Comment Density ({year})", fontsize=FONT_TITLE, fontweight="bold")
    ax2_main.legend(title="Legend", fontsize=FONT_LEGEND - 1, frameon=True, loc="upper right")
    apply_clean_style(ax2_main)
    
    # Density subplot: show comment count as area/line
    ax2_density.fill_between(doc_sorted["doc_rank"], 0, doc_sorted["comment_count"], 
                             color="#5C6670", alpha=0.5, label="Comment count")
    ax2_density.set_yscale("log")
    ax2_density.set_ylabel("# Comments\n(log)", fontsize=FONT_LABEL - 2, fontweight="bold")
    ax2_density.set_xlabel("Document Rank (sorted by comment count)", fontsize=FONT_LABEL, fontweight="bold")
    ax2_density.set_xlim(0, len(doc_sorted) + 1)
    apply_clean_style(ax2_density)
    
    plt.tight_layout()
    fig2.savefig(outdir / f"stream_experiment_density_{year}.png", dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"[OK] Saved: stream_experiment_density_{year}.png")
    
    # ===== EXPERIMENT 3: Heatmap View =====
    # Shows composition as a heatmap with log-scaled x-axis for comment counts
    fig3, axes3 = plt.subplots(len(CATEGORY_ORDER), 1, figsize=(14, 10), sharex=True)
    
    # Create log-spaced bins for comment counts
    log_bins = np.logspace(0, np.log10(max(doc_sorted["comment_count"].max(), 10)), 50)
    
    for ax, cat, color in zip(axes3, CATEGORY_ORDER, colors):
        # Calculate mean share in each log bin
        doc_sorted["log_bin"] = pd.cut(doc_sorted["comment_count"], bins=log_bins, include_lowest=True)
        bin_means = doc_sorted.groupby("log_bin", observed=True)[cat].mean() * 100
        
        if len(bin_means) > 0:
            x_centers = [(interval.left + interval.right) / 2 for interval in bin_means.index]
            ax.fill_between(x_centers, 0, bin_means.values, color=color, alpha=0.7)
            ax.plot(x_centers, bin_means.values, color=color, linewidth=2)
        
        ax.set_ylabel(cat[:15] + "...\n(%)" if len(cat) > 15 else cat + "\n(%)", fontsize=8)
        ax.set_ylim(0, 100)
        ax.set_xscale("log")
        apply_clean_style(ax)
    
    axes3[-1].set_xlabel("Comments per Document (log scale)", fontsize=FONT_LABEL, fontweight="bold")
    fig3.suptitle(f"Category Composition by Comment Volume ({year})", fontsize=FONT_TITLE, fontweight="bold", y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig3.savefig(outdir / f"stream_experiment_heatmap_{year}.png", dpi=300, bbox_inches="tight")
    plt.close(fig3)
    print(f"[OK] Saved: stream_experiment_heatmap_{year}.png")


def plot_spotlight_strip(df: pd.DataFrame, year: int, outdir: Path) -> None:
    """
    Panel C: Spotlight strip showing agencies with extreme values.
    Uses weighted metrics.
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
    
    fig, axes = plt.subplots(4, 1, figsize=(20, 6))
    
    # Define bar region to leave space for labels
    bar_x_start = 0.30
    bar_x_end = 1.0
    bar_width = bar_x_end - bar_x_start
    
    for ax, (label, row, metric) in zip(axes, extremes):
        # Get shares for this agency
        shares = [row[f"share_{cat}"] for cat in CATEGORY_ORDER]
        colors = get_category_colors()
        
        # Stacked bar (shifted right to leave room for labels)
        left = bar_x_start
        for share, color in zip(shares, colors):
            ax.barh(0, share * bar_width, left=left, color=color, height=0.6, edgecolor="white", linewidth=2)
            left += share * bar_width
        
        # Labels - full agency name with more space
        agency_name = str(row["agency"])
        ax.text(bar_x_start - 0.02, 0, f"{label}\n{agency_name}", ha="right", va="center", fontsize=12, fontweight="bold")
        ax.text(bar_x_end + 0.02, 0, metric, ha="left", va="center", fontsize=12, fontweight="bold", color="#2E6F88")
        
        # Styling
        ax.set_xlim(0, 1.15)
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
    Uses weighted metrics.
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
    
    # Highlight top agencies (scale with total number of agencies)
    total_agencies = len(df_metrics)
    top_n_effective = min(20, max(8, int(total_agencies * 0.15)))
    top_highlighted = df_metrics.nlargest(top_n_effective, "total")
    ax1.scatter(top_highlighted["X"], top_highlighted["Y"], s=150, alpha=0.7, color="#9A8153", edgecolors="white", linewidths=2)
    
    for _, row in top_highlighted.iterrows():
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
            shares_data.append(df_sorted[share_col].values * 100)
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
    shares_norm = shares_smooth / shares_smooth_sum * 100
    
    colors = get_category_colors()
    ax2.stackplot(df_sorted["cum_volume_center"], *shares_norm, colors=colors, alpha=0.85)
    
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 100)
    ax2.set_xlabel("Cumulative % of Comments (Corporate → Citizen)", fontsize=FONT_LABEL, fontweight="bold")
    ax2.set_ylabel("Makeup (%)", fontsize=FONT_LABEL, fontweight="bold")
    ax2.set_title("Ranked Stream (Composition)", fontsize=FONT_TITLE, fontweight="bold")
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



def plot_agency_clustering(df: pd.DataFrame, year: int, outdir: Path, min_comments: int = 50) -> None:
    """
    Hierarchical tree showing actual agency structure: Main Agency → Sub-Agency
    Color-coded by dominant commenter category at each level.
    Uses weighted metrics.
    """
    try:
        # Use hierarchy metrics (same as Sankey)
        merged, main_totals, sub_totals = compute_agency_hierarchy_metrics(df)
        
        # Filter sub-agencies with enough comments
        sub_totals_filtered = sub_totals[sub_totals["sub_total"] >= min_comments].copy()
        if len(sub_totals_filtered) == 0:
            print(f"Warning: No sub-agencies with >={min_comments} comments for dendrogram ({year})")
            return
        
        # Include all sub-agencies meeting the minimum threshold (no limit)
        
        # Merge to get category breakdowns
        merged_filtered = merged.merge(sub_totals_filtered[["main_agency", "sub_agency"]], on=["main_agency", "sub_agency"], how="inner")
        
        # Calculate dominant category for each sub-agency
        def get_dominant_category(group_data):
            """Get dominant category and its share."""
            cat_totals = {}
            total = group_data["n"].sum()
            if total == 0:
                return None, 0.0
            
            for cat in CATEGORY_ORDER:
                cat_total = group_data[group_data["category"] == cat]["n"].sum()
                cat_totals[cat] = cat_total / total
            
            dominant = max(cat_totals.items(), key=lambda x: x[1])
            return dominant[0], dominant[1]
        
        # Sub-agency level colors
        sub_colors = {}
        for _, row in sub_totals_filtered.iterrows():
            main = row["main_agency"]
            sub_raw = row["sub_agency"]
            
            # Handle None/empty sub-agency
            if pd.notna(sub_raw) and str(sub_raw).strip() != "" and sub_raw != main:
                sub_data = merged_filtered[
                    (merged_filtered["main_agency"] == main) & 
                    (merged_filtered["sub_agency"] == sub_raw)
                ]
                sub_key = (main, sub_raw)
            else:
                # No sub-agency, use main agency data where sub_agency is None/empty
                sub_data = merged_filtered[
                    (merged_filtered["main_agency"] == main) & 
                    (merged_filtered["sub_agency"].isna() | (merged_filtered["sub_agency"] == "") | (merged_filtered["sub_agency"] == main))
                ]
                sub_key = (main, main)
            
            dom_cat, dom_share = get_dominant_category(sub_data)
            sub_colors[sub_key] = (
                COLOR_PALETTE.get(dom_cat, "#5C6670") if dom_cat else "#5C6670"
            )
        
        # Main agency level colors (aggregate of their sub-agencies)
        main_colors = {}
        for main_agency in sub_totals_filtered["main_agency"].unique():
            main_data = merged_filtered[merged_filtered["main_agency"] == main_agency]
            dom_cat, dom_share = get_dominant_category(main_data)
            main_colors[main_agency] = (
                COLOR_PALETTE.get(dom_cat, "#5C6670") if dom_cat else "#5C6670"
            )
        
        # Build tree structure - store (display_name, original_sub) tuples
        tree = {}
        for _, row in sub_totals_filtered.iterrows():
            main = row["main_agency"]
            sub_raw = row["sub_agency"]
            # Use sub_agency if available, otherwise use main_agency as the "sub" (single-level)
            if pd.notna(sub_raw) and str(sub_raw).strip() != "" and sub_raw != main:
                sub_display = sub_raw
                sub_original = sub_raw
            else:
                sub_display = main  # No sub-agency, treat main as both
                sub_original = main
            if main not in tree:
                tree[main] = []
            if (sub_display, sub_original) not in tree[main]:  # Avoid duplicates
                tree[main].append((sub_display, sub_original))
        
        # Sort main agencies by total volume
        main_volumes = sub_totals_filtered.groupby("main_agency")["sub_total"].sum().sort_values(ascending=False)
        main_agencies = main_volumes.index.tolist()
        
        # Calculate positions
        fig, ax = plt.subplots(figsize=(20, max(12, len(sub_totals_filtered) * 0.3)))
        
        x_root = 0.05
        x_main = 0.25
        x_sub = 0.55
        x_end = 0.95
        
        y_positions = {}
        y_main_positions = {}
        
        # First, collect all positions in a list to normalize later
        raw_y_positions = []
        raw_main_positions = {}
        
        # Position sub-agencies (calculate raw positions first)
        y = 0.0  # Start at 0 in raw space
        spacing_between_subs = 1.0  # Base spacing between sub-agencies
        spacing_between_mains = 2.0  # Extra spacing between main agencies
        
        for main_agency in main_agencies:
            if main_agency not in tree:
                continue
            
            main_y_start = y
            sub_agencies = tree[main_agency]
            
            for sub_display, sub_original in sub_agencies:
                raw_y_positions.append((main_agency, sub_display, y))
                y_positions[(main_agency, sub_display)] = y
                y += spacing_between_subs
            
            main_y_end = y - spacing_between_subs
            raw_main_positions[main_agency] = (main_y_start + main_y_end) / 2
            y_main_positions[main_agency] = (main_y_start + main_y_end) / 2
            
            y += spacing_between_mains  # Extra spacing between main agencies
        
        # Normalize all positions to fit within [0.05, 0.95] range
        if len(raw_y_positions) > 0:
            min_y = min(pos[2] for pos in raw_y_positions)
            max_y = max(pos[2] for pos in raw_y_positions)
            if max_y > min_y:
                y_range = max_y - min_y
                y_scale = 0.9 / y_range  # Scale to use 90% of vertical space
                y_offset = 0.05  # Start at 5% from bottom
                
                # Normalize sub-agency positions
                for main_agency, sub_display, raw_y in raw_y_positions:
                    y_positions[(main_agency, sub_display)] = y_offset + (raw_y - min_y) * y_scale
                
                # Normalize main agency positions
                for main_agency in y_main_positions:
                    raw_main_y = raw_main_positions[main_agency]
                    y_main_positions[main_agency] = y_offset + (raw_main_y - min_y) * y_scale
            else:
                # All positions are the same, center them
                center_y = 0.5
                for main_agency, sub_display, _ in raw_y_positions:
                    y_positions[(main_agency, sub_display)] = center_y
                for main_agency in y_main_positions:
                    y_main_positions[main_agency] = center_y
        
        # Draw connections and nodes
        # Root node
        ax.scatter([x_root], [0.5], s=500, color="#12161A", zorder=10, edgecolors="white", linewidths=2)
        ax.text(x_root, 0.5, "All\nComments", ha="center", va="center", fontsize=10, fontweight="bold", color="white", zorder=11)
        
        # Main agency nodes and connections
        for main_agency in main_agencies:
            if main_agency not in tree:
                continue
            
            main_y = y_main_positions[main_agency]
            color = main_colors.get(main_agency, "#5C6670")
            
            # Connection from root
            ax.plot([x_root, x_main], [0.5, main_y], color="#5C6670", linewidth=2, alpha=0.3, zorder=1)
            
            # Main agency node
            ax.scatter([x_main], [main_y], s=300, color=color, zorder=10, edgecolors="white", linewidths=2)
            ax.text(x_main, main_y, main_agency[:30], ha="center", va="center", fontsize=8, fontweight="bold", 
                   color="white", zorder=11, bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8))
            
            # Sub-agency nodes and connections
            for sub_display, sub_original in tree[main_agency]:
                sub_y = y_positions[(main_agency, sub_display)]
                # Use original sub_agency for color lookup
                sub_key = (main_agency, sub_original)
                sub_color = sub_colors.get(sub_key, "#5C6670")
                
                # Connection from main to sub
                ax.plot([x_main, x_sub], [main_y, sub_y], color=sub_color, linewidth=1.5, alpha=0.5, zorder=2)
                
                # Sub-agency node
                sub_label = sub_display if sub_display != main_agency else f"{main_agency} (no sub)"
                ax.scatter([x_sub], [sub_y], s=200, color=sub_color, zorder=9, edgecolors="white", linewidths=1.5)
                ax.text(x_sub, sub_y, sub_label[:25], ha="left", va="center", fontsize=7,
                       bbox=dict(boxstyle="round,pad=0.2", facecolor=sub_color, alpha=0.7, edgecolor="white", linewidth=0.5))
        
        # Legend for colors
        legend_elements = [mpatches.Patch(facecolor=COLOR_PALETTE[cat], label=cat[:30]) for cat in CATEGORY_ORDER]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=8, framealpha=0.9)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title(f"Agency Hierarchy: Main → Sub-Agency ({year})\nColor-coded by dominant commenter category", 
                    fontsize=FONT_TITLE, fontweight="bold", pad=20)
        
        plt.tight_layout()
        fig.savefig(outdir / f"agency_clustering_dendrogram_{year}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Saved: agency_clustering_dendrogram_{year}.png")
        
        # Generate text report
        write_agency_clustering_report(merged_filtered, sub_totals_filtered, year, outdir)
        
    except Exception as e:
        print(f"Error generating dendrogram ({year}): {e}")
        import traceback
        traceback.print_exc()


def write_agency_clustering_report(merged: pd.DataFrame, sub_totals: pd.DataFrame, year: int, outdir: Path) -> None:
    """
    Write a text report summarizing agency clustering by dominant commenter category.
    
    Outputs two tables:
    1. Summary: # agencies per dominant category
    2. Detailed: Each agency with dominant category, % share, and delta to second-biggest
    """
    try:
        # Calculate dominant category for each sub-agency
        agency_stats = []
        
        for _, row in sub_totals.iterrows():
            main = row["main_agency"]
            sub_raw = row["sub_agency"]
            
            # Get full agency name for display
            if pd.notna(sub_raw) and str(sub_raw).strip() != "" and sub_raw != main:
                agency_name = f"{main}, {sub_raw}"
                sub_data = merged[
                    (merged["main_agency"] == main) & 
                    (merged["sub_agency"] == sub_raw)
                ]
            else:
                agency_name = main
                sub_data = merged[
                    (merged["main_agency"] == main) & 
                    (merged["sub_agency"].isna() | (merged["sub_agency"] == "") | (merged["sub_agency"] == main))
                ]
            
            if len(sub_data) == 0:
                continue
            
            # Calculate category shares
            total = sub_data["n"].sum()
            if total == 0:
                continue
            
            cat_shares = {}
            for cat in CATEGORY_ORDER:
                cat_total = sub_data[sub_data["category"] == cat]["n"].sum()
                cat_shares[cat] = (cat_total / total) * 100
            
            # Find dominant and second-biggest
            sorted_cats = sorted(cat_shares.items(), key=lambda x: x[1], reverse=True)
            dominant_cat, dominant_share = sorted_cats[0]
            second_cat, second_share = sorted_cats[1] if len(sorted_cats) > 1 else (None, 0)
            delta = dominant_share - second_share
            
            agency_stats.append({
                "agency": agency_name,
                "dominant_category": dominant_cat,
                "share_pct": dominant_share,
                "delta_to_second": delta,
                "second_category": second_cat,
                "second_share": second_share,
                "total_comments": int(total)
            })
        
        if len(agency_stats) == 0:
            print(f"Warning: No agency stats to write for report ({year})")
            return
        
        # Create report
        report_lines = []
        report_lines.append(f"Agency Clustering Report ({year})")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Table 1: Summary by dominant category
        report_lines.append("TABLE 1: AGENCIES BY DOMINANT COMMENTER CATEGORY")
        report_lines.append("-" * 50)
        report_lines.append(f"{'Dominant Category':<45} | {'# Agencies':>10}")
        report_lines.append("-" * 50)
        
        cat_counts = {}
        for stat in agency_stats:
            cat = stat["dominant_category"]
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
        
        for cat in CATEGORY_ORDER:
            count = cat_counts.get(cat, 0)
            report_lines.append(f"{cat:<45} | {count:>10}")
        
        report_lines.append("-" * 50)
        report_lines.append(f"{'TOTAL':<45} | {len(agency_stats):>10}")
        report_lines.append("")
        report_lines.append("")
        
        # Table 2: Detailed breakdown
        report_lines.append("TABLE 2: DETAILED AGENCY BREAKDOWN")
        report_lines.append("-" * 120)
        header = f"{'Agency':<50} | {'Dominant Category':<25} | {'Share':>7} | {'Delta':>8} | {'Comments':>10}"
        report_lines.append(header)
        report_lines.append("-" * 120)
        
        # Sort by dominant category, then by share
        sorted_stats = sorted(agency_stats, key=lambda x: (CATEGORY_ORDER.index(x["dominant_category"]), -x["share_pct"]))
        
        current_cat = None
        for stat in sorted_stats:
            # Add separator between categories
            if stat["dominant_category"] != current_cat:
                if current_cat is not None:
                    report_lines.append("")
                current_cat = stat["dominant_category"]
                report_lines.append(f"--- {current_cat} ---")
            
            agency_display = stat["agency"][:48]
            cat_display = stat["dominant_category"][:23]
            line = f"{agency_display:<50} | {cat_display:<25} | {stat['share_pct']:>6.1f}% | {stat['delta_to_second']:>+7.1f}pp | {stat['total_comments']:>10,}"
            report_lines.append(line)
        
        report_lines.append("-" * 120)
        report_lines.append("")
        report_lines.append("Notes:")
        report_lines.append("- Share: Percentage of comments from the dominant category")
        report_lines.append("- Delta: Difference in percentage points between dominant and second-largest category")
        report_lines.append("- pp = percentage points")
        
        # Write to file
        report_path = outdir / f"agency_clustering_report_{year}.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))
        
        print(f"[OK] Saved: agency_clustering_report_{year}.txt")
        
    except Exception as e:
        print(f"Error writing clustering report ({year}): {e}")
        import traceback
        traceback.print_exc()


def compute_agency_hierarchy_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate metrics with main_agency and sub_agency hierarchy.
    Uses weighted sums.
    """
    # Ensure hierarchy columns exist
    if "main_agency" not in df.columns or "sub_agency" not in df.columns:
        # Parse if not already done
        parsed = df["agency"].apply(parse_agency_hierarchy)
        df["main_agency"] = [p[0] for p in parsed]
        df["sub_agency"] = [p[1] for p in parsed]
    
    if "weight" not in df.columns:
        df["weight"] = 1.0
        
    # --- CRITICAL FIX 2: Fill NaN sub-agencies to prevent groupby dropping them ---
    # Create a temporary view for calculation
    df_calc = df.copy()
    
    # If sub_agency is missing, treat the main agency as the sub-agency (self-reference)
    # or use a placeholder like "General"
    df_calc["sub_agency"] = df_calc["sub_agency"].fillna(df_calc["main_agency"])
    # ------------------------------------------------------------------------------
        
    # Group by main_agency, sub_agency, and category
    grp = df_calc.groupby(["main_agency", "sub_agency", "category"])["weight"].sum().rename("n").reset_index()
    
    # Calculate totals at sub-agency level
    sub_totals = grp.groupby(["main_agency", "sub_agency"])["n"].sum().rename("sub_total").reset_index()
    merged = grp.merge(sub_totals, on=["main_agency", "sub_agency"], how="left")
    merged["share"] = merged["n"] / merged["sub_total"].where(merged["sub_total"] > 0, 1)
    
    # Calculate main agency totals
    main_totals = merged.groupby("main_agency")["n"].sum().rename("main_total").reset_index()
    
    return merged, main_totals, sub_totals


def plot_sankey_agency_category(df: pd.DataFrame, year: int, outdir: Path, min_agency_comments: int = 100) -> None:
    """
    Sankey diagram showing flow: Comments -> Main Agency -> Sub-Agency -> Category
    Width of flows proportional to comment volume.
    Creates interactive HTML version only.
    Uses weighted volume.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("Warning: plotly not installed")
        return
    
    # Compute hierarchy metrics
    merged, main_totals, sub_totals = compute_agency_hierarchy_metrics(df)
    
    # Filter by minimum comments at sub-agency level
    sub_totals_filtered = sub_totals[sub_totals["sub_total"] >= min_agency_comments].copy()
    if len(sub_totals_filtered) == 0:
        return
    
    # Include all sub-agencies meeting the minimum threshold (no limit)
    
    # Merge back to get category breakdowns
    merged_filtered = merged.merge(sub_totals_filtered[["main_agency", "sub_agency"]], on=["main_agency", "sub_agency"], how="inner")
    
    # Build node structure: [All Comments, Main Agencies..., Sub-Agencies..., Categories...]
    main_agencies = sorted(sub_totals_filtered["main_agency"].unique())
    sub_agencies_list = []
    for _, row in sub_totals_filtered.iterrows():
        # Use sub_agency if available, otherwise use main_agency
        sub_name = row["sub_agency"] if pd.notna(row["sub_agency"]) and str(row["sub_agency"]).strip() != "" else row["main_agency"]
        sub_agencies_list.append((row["main_agency"], sub_name))
    
    node_labels = ["All Comments"]
    node_labels.extend(main_agencies)
    node_labels.extend([sub[1] for sub in sub_agencies_list])
    node_labels.extend(CATEGORY_ORDER)
    
    # Node indices
    source_idx = 0
    main_start = 1
    sub_start = main_start + len(main_agencies)
    cat_start = sub_start + len(sub_agencies_list)
    
    # Build links
    links_source = []
    links_target = []
    links_value = []
    links_color = []
    
    cat_colors = get_category_colors()
    cat_color_map = {cat: color for cat, color in zip(CATEGORY_ORDER, cat_colors)}
    
    def hex_to_rgba(hex_color, alpha=0.6):
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"
    
    # Flow: All Comments -> Main Agencies
    main_totals_dict = sub_totals_filtered.groupby("main_agency")["sub_total"].sum().to_dict()
    for main_agency in main_agencies:
        main_idx = main_start + main_agencies.index(main_agency)
        links_source.append(source_idx)
        links_target.append(main_idx)
        links_value.append(int(main_totals_dict.get(main_agency, 0)))
        links_color.append("rgba(92, 102, 112, 0.4)")
    
    # Flow: Main Agencies -> Sub-Agencies
    for idx, (_, row) in enumerate(sub_totals_filtered.iterrows()):
        main_idx = main_start + main_agencies.index(row["main_agency"])
        sub_idx = sub_start + idx
        links_source.append(main_idx)
        links_target.append(sub_idx)
        links_value.append(int(row["sub_total"]))
        links_color.append("rgba(92, 102, 112, 0.5)")
    
    # Flow: Sub-Agencies -> Categories
    for idx, (_, row) in enumerate(sub_totals_filtered.iterrows()):
        sub_idx = sub_start + idx
        # Handle None sub_agency properly
        if pd.notna(row["sub_agency"]) and str(row["sub_agency"]).strip() != "":
            sub_data = merged_filtered[
                (merged_filtered["main_agency"] == row["main_agency"]) & 
                (merged_filtered["sub_agency"] == row["sub_agency"])
            ]
        else:
            # No sub-agency, match by main_agency only where sub_agency is also None/empty
            sub_data = merged_filtered[
                (merged_filtered["main_agency"] == row["main_agency"]) & 
                (merged_filtered["sub_agency"].isna() | (merged_filtered["sub_agency"] == ""))
            ]
        for _, cat_row in sub_data.iterrows():
            cat = cat_row["category"]
            cat_idx = cat_start + CATEGORY_ORDER.index(cat)
            links_source.append(sub_idx)
            links_target.append(cat_idx)
            links_value.append(int(cat_row["n"]))
            links_color.append(hex_to_rgba(cat_color_map[cat], alpha=0.7))
    
    # Node colors
    source_color = "#12161A"
    main_colors = ["#5C6670"] * len(main_agencies)
    sub_colors = ["#8A9399"] * len(sub_agencies_list)
    node_colors = [source_color] + main_colors + sub_colors + cat_colors
    
    # Create Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        arrangement='snap',
        node=dict(
            pad=30,
            thickness=35,
            line=dict(color="white", width=3),
            label=node_labels,
            color=node_colors,
        ),
        link=dict(
            source=links_source,
            target=links_target,
            value=links_value,
            color=links_color,
            line=dict(color="rgba(0,0,0,0.2)", width=1)
        )
    )])
    
    fig.update_layout(
        title=dict(
            text=f"Comment Flow: Main Agency → Sub-Agency → Categories ({year})",
            font=dict(size=22, family="Arial, sans-serif", color="#12161A"),
            x=0.5,
            xanchor='center',
            pad=dict(t=25, b=15)
        ),
        font=dict(family="Arial, sans-serif", size=12, color="#5C6670"),
        height=max(1000, len(sub_agencies_list) * 40),
        paper_bgcolor="#FAFAFA",
        plot_bgcolor="#FAFAFA",
        margin=dict(l=30, r=30, t=120, b=30)
    )
    
    # Save as HTML only
    html_path = outdir / f"sankey_agency_category_{year}.html"
    fig.write_html(str(html_path))
    print(f"[OK] Saved: sankey_agency_category_{year}.html")


def plot_sankey_agency_category_matplotlib(df: pd.DataFrame, year: int, outdir: Path, min_agency_comments: int = 100) -> None:
    """
    Fallback matplotlib-based Sankey using custom drawing.
    Shows Comments -> Agency -> Category with width proportional to volume.
    Uses weighted volume.
    """
    df_metrics = compute_agency_metrics(df)
    
    # Filter agencies with sufficient volume
    df_filtered = df_metrics[df_metrics["total"] >= min_agency_comments].copy()
    if len(df_filtered) == 0:
        print(f"Warning: No agencies with >={min_agency_comments} comments for Sankey")
        return
    
    # Include all agencies meeting the minimum threshold (no limit)
    df_filtered = df_filtered.sort_values("total", ascending=True).copy()
    
    # Calculate positions
    n_agencies = len(df_filtered)
    n_categories = len(CATEGORY_ORDER)
    
    fig, ax = plt.subplots(figsize=(16, max(10, n_agencies * 0.4)))
    
    # Column positions
    x_left = 0.1   # All Comments
    x_mid = 0.4    # Agencies
    x_right = 0.7  # Categories
    
    # Calculate cumulative positions for agencies (sorted by volume, bottom to top)
    agency_y_positions = {}
    agency_heights = {}
    total_height = 0
    for idx, (_, row) in enumerate(df_filtered.iterrows()):
        height = row["total"] / df_filtered["total"].sum()
        agency_y_positions[idx] = total_height + height / 2
        agency_heights[idx] = height
        total_height += height
    
    # Scale to fit plot
    y_scale = 0.8 / total_height if total_height > 0 else 1
    for idx in agency_y_positions:
        agency_y_positions[idx] *= y_scale
        agency_heights[idx] *= y_scale
    
    # Calculate cumulative positions for categories
    cat_totals = {}
    for cat in CATEGORY_ORDER:
        count_col = f"n_{cat}"
        cat_totals[cat] = df_filtered[count_col].sum() if count_col in df_filtered.columns else 0
    
    total_cat_volume = sum(cat_totals.values())
    cat_y_positions = {}
    cat_heights = {}
    y_pos = 0
    for cat in CATEGORY_ORDER:
        height = cat_totals[cat] / total_cat_volume if total_cat_volume > 0 else 0
        cat_y_positions[cat] = y_pos + height / 2
        cat_heights[cat] = height
        y_pos += height
    
    # Scale categories
    cat_y_scale = 0.8 / y_pos if y_pos > 0 else 1
    for cat in cat_y_positions:
        cat_y_positions[cat] *= cat_y_scale
        cat_heights[cat] *= cat_y_scale
    
    # Draw flows from All Comments to Agencies
    colors = get_category_colors()
    for idx, (_, row) in enumerate(df_filtered.iterrows()):
        y_center = agency_y_positions[idx]
        height = agency_heights[idx]
        
        # Draw rectangle for agency
        rect = mpatches.FancyBboxPatch(
            (x_mid - 0.02, y_center - height/2),
            0.04, height,
            boxstyle="round,pad=0.001",
            facecolor="#5C6670",
            edgecolor="white",
            linewidth=0.5
        )
        ax.add_patch(rect)
        
        # Label agency
        ax.text(x_mid, y_center, str(row["agency"])[:25], 
               ha="center", va="center", fontsize=7, rotation=0,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="none"))
        
        # Flow from All Comments to Agency
        ax.plot([x_left + 0.05, x_mid - 0.02], [y_center, y_center], 
               color="#5C6670", linewidth=max(1, height * 200), alpha=0.3, zorder=0)
    
    # Color mapping for categories
    cat_color_map = {cat: color for cat, color in zip(CATEGORY_ORDER, colors)}
    
    # Draw flows from Agencies to Categories
    for idx, (_, row) in enumerate(df_filtered.iterrows()):
        agency_y = agency_y_positions[idx]
        
        # Calculate cumulative flow for this agency
        y_offset = 0
        for cat in CATEGORY_ORDER:
            count_col = f"n_{cat}"
            if count_col in row and row[count_col] > 0:
                cat_y = cat_y_positions[cat]
                flow_width = row[count_col] / row["total"] if row["total"] > 0 else 0
                flow_height = agency_heights[idx] * flow_width
                
                # Draw curved flow
                # Simple bezier-like curve
                t = np.linspace(0, 1, 50)
                x_curve = x_mid + 0.02 + (x_right - x_mid - 0.04) * t
                y_curve = agency_y + (cat_y - agency_y) * (3*t**2 - 2*t**3)
                
                ax.plot(x_curve, y_curve, 
                       color=cat_color_map[cat], 
                       linewidth=max(0.5, flow_height * 300), 
                       alpha=0.6, zorder=1)
    
    # Draw category rectangles
    for cat in CATEGORY_ORDER:
        y_center = cat_y_positions[cat]
        height = cat_heights[cat]
        if height > 0:
            rect = mpatches.FancyBboxPatch(
                (x_right - 0.02, y_center - height/2),
                0.04, height,
                boxstyle="round,pad=0.001",
                facecolor=cat_color_map[cat],
                edgecolor="white",
                linewidth=0.5
            )
            ax.add_patch(rect)
            ax.text(x_right, y_center, cat[:20], 
                   ha="center", va="center", fontsize=8, rotation=0,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="none"))
    
    # Draw "All Comments" source
    ax.add_patch(mpatches.FancyBboxPatch(
        (x_left - 0.02, 0.1),
        0.04, 0.8,
        boxstyle="round,pad=0.001",
        facecolor="#12161A",
        edgecolor="white",
        linewidth=1
    ))
    ax.text(x_left, 0.5, "All\nComments", ha="center", va="center", 
           fontsize=12, fontweight="bold", color="white")
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(f"Comment Flow: Agencies → Categories ({year})\nAgencies with >{min_agency_comments} comments", 
                fontsize=FONT_TITLE, fontweight="bold", pad=20)
    
    plt.tight_layout()
    fig.savefig(outdir / f"sankey_agency_category_{year}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: sankey_agency_category_{year}.png (matplotlib)")


def generate_narrative_summary(df: pd.DataFrame, fr_csv_path: Optional[Path], year: int) -> None:
    """
    Print narrative stats for the policy brief.
    Updated to report weighted statistics.
    """
    print("\n" + "="*60)
    print(f"NARRATIVE SUMMARY ({year})")
    print("="*60)
    
    # 1. Participation gap
    # Raw counts
    n_comments_raw = len(df)
    
    # Weighted counts
    if "weight" not in df.columns:
        df["weight"] = 1.0
        
    n_comments_weighted = df["weight"].sum()
    
    n_citizens_weighted = df[df["category"] == "Ordinary Citizen"]["weight"].sum()
    pct_citizen = n_citizens_weighted / n_comments_weighted * 100 if n_comments_weighted > 0 else 0
    
    print(f"Total Comments Analyzed (sample size): {n_comments_raw:,}")
    print(f"Estimated Population Volume (weighted): {n_comments_weighted:,.0f}")
    print(f"Ordinary Citizen Comments (weighted): {n_citizens_weighted:,.0f} ({pct_citizen:.1f}%)")
    
    # 2. Zero comment docs and Concentration
    if fr_csv_path and fr_csv_path.exists():
        try:
            fr_df = pd.read_csv(fr_csv_path)
            total_docs = len(fr_df)
            if "comment_count" in fr_df.columns:
                # Filter to only documents where we know the comment count
                # (exclude non-regs.gov docs with unknown counts)
                if "count_source" in fr_df.columns:
                    fr_df_known = fr_df[fr_df["count_source"] != "unknown"].copy()
                    unknown_count_docs = (fr_df["count_source"] == "unknown").sum()
                else:
                    fr_df_known = fr_df.copy()
                    unknown_count_docs = 0
                
                total_docs_known = len(fr_df_known)
                zero_docs = (fr_df_known["comment_count"] == 0).sum()
                pct_zero = zero_docs / total_docs_known * 100 if total_docs_known > 0 else 0
                
                print(f"Total Documents (Federal Register): {total_docs:,}")
                if unknown_count_docs > 0:
                    print(f"  ({unknown_count_docs:,} docs with unknown counts excluded from stats)")
                print(f"Documents with 0 comments (known counts only): {zero_docs:,} ({pct_zero:.1f}%)")
                
                # Urgency / Attention
                # Top 1% of documents capture X% of comments
                top_1_pct_n = max(1, int(np.ceil(total_docs * 0.01)))
                top_docs = fr_df.nlargest(top_1_pct_n, "comment_count")
                top_vol = top_docs["comment_count"].sum()
                total_vol = fr_df["comment_count"].sum()
                pct_captured = top_vol / total_vol * 100 if total_vol > 0 else 0
                
                print(f"\nConcentration of Attention:")
                print(f"- Top 1% of documents ({top_1_pct_n:,}) capture {pct_captured:.1f}% of all comments.")
                print(f"- {pct_zero:.1f}% of documents received zero public attention.")
                
                # Agency scrutiny
                if "agency_acronym" in fr_df.columns:
                    agency_grp = fr_df.groupby("agency_acronym")["comment_count"].sum().sort_values()
                    zero_comment_agencies = (agency_grp == 0).sum()
                    print(f"- {zero_comment_agencies} agencies received 0 comments total across all their documents.")
            
        except Exception as e:
            print(f"Error calculating document stats: {e}")
    else:
        print("(Federal Register metadata CSV not found; cannot calculate zero-comment stats)")
        
    print("="*60 + "\n")


# ============================================================================
# MAIN
# ============================================================================

def generate_plots_for_year(
    config: PipelineConfig,
    simple_only: bool = False,
    continuum_only: bool = False,
) -> None:
    """
    Generate visualizations for a single year.
    
    This is the main entry point for programmatic use.
    
    Args:
        config: Pipeline configuration
        simple_only: Only generate simple 3-chart set
        continuum_only: Only generate complex continuum page
    
    Side Effects:
        Writes plot files to plots directory.
        Logs progress to configured logger.
    """
    year = config.year
    makeup_path = get_makeup_data_path(year)
    fr_csv_path = get_fr_csv_path(year)
    outdir = get_plots_dir(year)
    
    if not makeup_path.exists():
        logger.error(f"{makeup_path} not found. Run classify_makeup.py first.")
        return
    
    log_banner(logger, "Federal Register Comment Visualization Suite")
    logger.info(f"Year: {year}")
    logger.info(f"Input: {makeup_path}")
    logger.info(f"Output: {outdir}")
    
    # Load and preprocess data
    logger.info("Loading data...")
    df = pd.read_csv(makeup_path)
    
    required_cols = ["comment_id", "category"]
    for col in required_cols:
        if col not in df.columns:
            logger.error(f"Missing required column: {col}")
            return

    logger.info(f"Loaded {len(df):,} rows")
    
    # Check for FR CSV
    if fr_csv_path.exists():
        logger.info(f"Found FR CSV: {fr_csv_path}")
    else:
        fr_csv_path = None
    
    # Try to join document_number if available
    if fr_csv_path and "document_number" not in df.columns:
        try:
            fr_df = pd.read_csv(fr_csv_path, usecols=["document_number", "comment_id"])
            if "document_number" in fr_df.columns and "comment_id" in fr_df.columns:
                before_merge = len(df)
                df = df.merge(fr_df[["comment_id", "document_number"]], on="comment_id", how="left")
                merged_count = df["document_number"].notna().sum()
                merge_rate = merged_count / before_merge * 100 if before_merge > 0 else 0
                logger.info(f"Joined document_number: {merged_count:,}/{before_merge:,} ({merge_rate:.1f}%)")
                if merge_rate < 90:
                    logger.warning("Low merge rate - some plots may be incomplete")
        except Exception as e:
            logger.debug(f"Could not join document_number: {e}")
    
    logger.info("Preprocessing...")
    df = normalize_categories(df)
    df = deduplicate_comments(df)
    
    logger.info("Calculating weights...")
    df = calculate_weights(df, fr_csv_path)
    df = explode_agencies(df)
    
    logger.info(f"After preprocessing: {len(df):,} rows, {df['agency'].nunique()} agencies")
    
    # Generate narrative summary
    generate_narrative_summary(df, fr_csv_path, year)
    
    # Generate visualizations
    generate_simple = not continuum_only
    generate_continuum = not simple_only
    
    plot_weight_distribution(df, outdir)
    
    if generate_simple:
        log_banner(logger, "SIMPLE 3-CHART SET")
        plot_composition_donut(df, year, outdir, fr_csv_path)
        plot_agency_compass(df, year, outdir)
        plot_workload_vs_makeup(df, year, outdir)
        plot_workload_vs_citizen_by_agency(df, year, outdir)
        if "document_number" in df.columns:
            plot_workload_vs_citizen_by_document(df, year, outdir)
        plot_comment_distribution_roi(df, year, outdir, fr_csv_path)
    
    if generate_continuum:
        log_banner(logger, "COMPLEX CONTINUUM PAGE")
        plot_compass_with_ribbon(df, year, outdir)
        plot_ranked_stream(df, year, outdir)
        plot_ranked_stream_experiments(df, year, outdir)
        plot_spotlight_strip(df, year, outdir)
        plot_agency_clustering(df, year, outdir)
        plot_sankey_agency_category(df, year, outdir)
        plot_continuum_page(df, year, outdir)
    
    log_banner(logger, f"COMPLETE - All visualizations saved to: {outdir}")


def main() -> int:
    """CLI entry point."""
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
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    
    args = parser.parse_args()
    
    setup_logging(verbose=args.verbose, quiet=args.quiet, year=args.year)
    
    config = PipelineConfig(
        year=args.year,
        verbose=args.verbose,
        quiet=args.quiet,
    )
    
    # Handle custom paths if provided
    if args.makeup:
        makeup_path = Path(args.makeup)
        if not makeup_path.exists():
            logger.error(f"{makeup_path} not found")
            return 1
    
    if args.out_dir:
        outdir = Path(args.out_dir)
        outdir.mkdir(parents=True, exist_ok=True)
    
    try:
        generate_plots_for_year(config, args.simple_only, args.continuum_only)
        return 0
    except Exception as e:
        logger.error(f"Plot generation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())


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
