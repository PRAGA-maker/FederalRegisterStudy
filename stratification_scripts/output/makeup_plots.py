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

# ---------------------------------------------------------------------------
# TODO: New analysis capabilities (Batch 5, Agent E — 2026-02-17)
# Reference: plans/eager-waddling-squirrel.md
#
# Agent E added 6 new FR API fields to the FR CSV and fixed doc_type /
# eligibility_reason propagation so they flow through all pipeline steps.
# This unlocks several analyses that were previously impossible.
#
# NEW DATA AVAILABLE (FR CSV, 48 columns — was 42):
#   - abstract        : document summary text (~80% populated)
#   - topics          : policy domain tags, e.g. "Air pollution control,
#                       Environmental protection" (Rules ~50%, Notices ~0%)
#   - action          : granular action type, e.g. "Proposed rule; 12-month
#                       finding.", "Advance notice of proposed rulemaking."
#                       Best proxy for notice subtype (~90% populated)
#   - significant     : EO 12866 significance flag (True/False for rules,
#                       None for notices)
#   - cfr_titles      : affected CFR title numbers, e.g. "40, 50"
#                       (40=EPA, 29=Labor, 26=Tax — rules/prorules only)
#   - cfr_parts_count : number of CFR parts affected (regulatory scope proxy)
#
# FIXED PROPAGATION (comments_raw 26 cols — was 24, makeup_data 7 — was 5):
#   - doc_type            : "Proposed Rule", "Rule", "Notice" — now in
#                           comments_raw AND makeup_data (was lost after Step 1)
#   - eligibility_reason  : "prorule" vs "notice-with-comment-period" — same
#
# ANALYSES NOW POSSIBLE (TODO — implement as needed):
#   1. Commenter composition by doc_type — split donut/bar by Proposed Rule
#      vs Notice vs Rule. Uses doc_type in makeup_data.
#   2. Policy domain stratification — group by cfr_titles or topics to answer
#      "Do environmental rules attract different commenters than labor rules?"
#   3. Significance analysis — compare commenter mix for EO 12866 "significant"
#      rules vs routine rules. Uses significant field in FR CSV.
#   4. Notice subtype breakdown — use action field to classify NO_RIN notices
#      into PRA/info collection, meetings, petitions, EPA SIPs, etc.
#   5. Regulatory scope vs participation — correlate cfr_parts_count with
#      comment volume and commenter diversity.
#
# RIN-BASED CHAIN-LINKING (Batch 5, Agent F audit — 2026-02-18)
# Reference: plans/encapsulated-wandering-hollerith.md
#
# The pipeline already captures full rulemaking chains via rin_timetable JSON.
# No additional API calls needed — all data is in the FR CSV.
#
# DATA AVAILABLE PER RIN (from FR CSV):
#   - rin_timetable         : JSON array of ALL actions, dates, FR citations
#                             (e.g., 20 entries for CFTC RIN 3038-AD82)
#   - timetable_sequence    : condensed chain string, e.g.
#                             "NPRM->NPRM_CLOSED->FINAL->FINAL_EFFECTIVE"
#   - nprm_document_number  : FR doc number of the NPRM
#   - final_rule_document_number : FR doc number of the Final Rule
#   - nprm_date, final_action_date, effective_date : milestone dates
#   - nprm_citation, final_action_citation : FR volume/page citations
#
# CHAIN RECONSTRUCTION (zero API calls):
#   1. Group FR CSV by rin — multiple docs per RIN exist in the same year
#      (e.g., RIN 3038-AD82 has 5 docs in 2014: initial reopening,
#      correction, 2nd reopening, extension, 3rd reopening)
#   2. Parse rin_timetable JSON for the full action sequence
#   3. Match timetable entries to CSV rows by rin + publication_date
#      — each match = a comment-eligible step with comment_count
#   4. Non-comment-eligible docs (final rules without comment periods)
#      are identified via final_rule_document_number + timetable
#      date/citation. They have no row but their existence is known.
#   5. Join comments_raw on document_number to get actual comment text
#      per chain step
#
# KEY INSIGHT: lifecycle_stage reflects the RIN's CURRENT state (as of
# pipeline run), not the document's role. A 2014 "Proposed Rule" gets
# FINAL_EFFECTIVE if the RIN progressed to final by 2026. The document's
# actual role in the chain is determined by matching its publication_date
# to a timetable entry.
#
# CROSS-YEAR JOINING:
#   Load federal_register_{year}_comments.csv for multiple years, concat,
#   group by rin. The NPRM (e.g., 2012), supplemental NPRMs (2015),
#   and final rule (2016) all join into one chain with comments at each
#   eligible step. This is the primary remaining analytical unlock.
#
# EXAMPLE ANALYSES ENABLED:
#   6. Chain-level comment evolution — for RINs with multiple comment
#      periods (reopenings, supplemental NPRMs), track how comment
#      volume and commenter composition change across steps.
#   7. NPRM-to-Final lifecycle storytelling — for a given RIN, show
#      the full timeline with comment counts at each eligible step,
#      plus the final rule date/citation as the outcome.
#   8. Reopening impact — compare comment volume on initial NPRM vs
#      reopened comment periods for the same rulemaking.
#   9. Duration vs participation — correlate nprm_to_final_days with
#      total comment volume across all chain steps.
#
# STILL BLOCKED:
#   - Cross-year lifecycle comparison (proposedplan.md Section 20) — needs
#     multi-year CSVs loaded and joined by rin. The data infrastructure
#     exists (rin_timetable, document linking); the join script does not.
# ---------------------------------------------------------------------------
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
    get_agency_responses_path,
)
from stratification_scripts.logging_utils import (
    get_logger,
    log_banner,
    setup_logging,
)
from stratification_scripts.output.lifecycle_plots import generate_lifecycle_plots

logger = get_logger(__name__)

# Scope warning suppression to known noisy sources instead of global ignore.
# matplotlib emits FutureWarnings on layout/tick changes; pandas/numpy emit
# DeprecationWarnings during groupby/plot operations.
warnings.filterwarnings("ignore", category=FutureWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
warnings.filterwarnings("ignore", category=FutureWarning, module="numpy")

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

# Display names for plots (removes "incl. small/local business" phrasing)
CATEGORY_DISPLAY_NAMES = {
    "Undecided/Anonymous": "Undecided/Anonymous",
    "Ordinary Citizen": "Ordinary Citizen",
    "Large Organization/Corporation": "Large Organization/Corporation",
    "Academic/Industry/Expert (incl. small/local business)": "Academic/Industry/Expert",
    "Political Consultant/Lobbyist": "Political Consultant/Lobbyist",
}

COLOR_PALETTE = {
    "Undecided/Anonymous": "#DDE3EA",  # paper
    "Ordinary Citizen": "#6B7D6D",      # sage
    "Large Organization/Corporation": "#12161A",  # ink
    "Academic/Industry/Expert (incl. small/local business)": "#9A8153",  # ochre
    "Political Consultant/Lobbyist": "#6D7581",  # slate
}

# One-pager palette — used only by plot_composition_dumbbell and
# plot_voice_vs_outcome to match the Google Doc brand (burnt orange
# headings, deep navy, warm tan, ink). Leaving COLOR_PALETTE alone so
# the other ~15 plots in this suite don't accidentally re-color.
ONE_PAGER_PALETTE = {
    "Ordinary Citizen": "#D76F2C",                                              # burnt orange (doc accent)
    "Large Organization/Corporation": "#1E3A5F",                                # deep navy
    "Academic/Industry/Expert (incl. small/local business)": "#C9B078",         # warm tan
    "Political Consultant/Lobbyist": "#12161A",                                 # ink
    "Undecided/Anonymous": "#BFC6CC",                                           # paper
}
# Colors dark enough to take white overlay text.
ONE_PAGER_DARK_ON = {"#D76F2C", "#1E3A5F", "#12161A"}

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
        logger.warning("FR metadata not found. Using unweighted data (weight=1.0)")
        df["weight"] = 1.0
        df["weight_doc"] = 1.0
        return df
    
    try:
        # Load FR Population Data
        fr_df = pd.read_csv(fr_csv_path)
        
        # Ensure critical columns exist
        if "agency" not in fr_df.columns or "comment_count" not in fr_df.columns:
            logger.warning("FR metadata missing 'agency' or 'comment_count'. Using unweighted data.")
            df["weight"] = 1.0
            df["weight_doc"] = 1.0
            return df
            
        # Filter to documents with comments
        fr_df = fr_df[fr_df["comment_count"] > 0].copy()
        
        # Assign Comment Bins (matching mine_comments.py logic)
        # NaN/None → "UNKNOWN" bin (docket-fallback docs with no known count)
        def get_bin(n):
            if pd.isna(n): return "UNKNOWN"
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
            # NaN comment_count = doc not in FR population (docket-fallback with unknown count)
            # Leave NaN — get_bin() puts them in "UNKNOWN" bin, no population expansion
            n_unmatched = df["comment_count"].isna().sum()
            if n_unmatched > 0:
                n_unmatched_docs = df.loc[df["comment_count"].isna(), "document_number"].nunique()
                logger.warning(
                    f"{n_unmatched} comments ({n_unmatched_docs} docs) have unknown comment_count "
                    f"(not in FR population) -- binned as UNKNOWN, weight=1.0"
                )
            df["comment_bin"] = df["comment_count"].apply(get_bin)
        else:
            logger.warning("Could not link documents to FR metadata. Using unweighted data.")
            df["weight"] = 1.0
            df["weight_doc"] = 1.0
            return df
            
        # Calculate Sample Totals (N_sample) per Stratum
        # NOTE: This counts ALL comments including duplicates (each duplicate is a separate submission)
        sample_strata = df.groupby(["agency", "comment_bin"]).size().rename("N_sample").reset_index()
        
        # Merge Population and Sample Stratum Totals
        strata_weights = pd.merge(pop_strata, sample_strata, on=["agency", "comment_bin"], how="inner")
        
        # Calculate Weight — guard against N_sample=0 (would produce inf)
        zero_sample = strata_weights[strata_weights["N_sample"] == 0]
        if len(zero_sample) > 0:
            logger.warning(
                f"{len(zero_sample)} strata have N_sample=0 (will be excluded from weighting): "
                f"{zero_sample[['agency', 'comment_bin']].to_dict('records')}"
            )
            strata_weights = strata_weights[strata_weights["N_sample"] > 0]
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
        # Guard: n_sample_doc should never be 0 (would mean a doc with no comments in the sample),
        # but protect against inf just in case
        df["weight_doc"] = df["comment_count"] / df["n_sample_doc"].replace(0, np.nan)

        # Detect if all documents are present (no doc-level sampling)
        # fr_df is already filtered to comment_count > 0, so this excludes
        # comment-eligible docs that received 0 comments (no sample rows either)
        fr_doc_count = fr_df["document_number"].nunique()
        sample_doc_count = df["document_number"].nunique()

        if sample_doc_count >= fr_doc_count * 0.95:
            logger.info(
                f"All-docs mode detected ({sample_doc_count}/{fr_doc_count} docs present). "
                f"Using document-level weights only (no cross-doc expansion)."
            )
            df["weight"] = df["weight_doc"]
        else:
            logger.info(
                f"Doc-sampling mode detected ({sample_doc_count}/{fr_doc_count} docs). "
                f"Using stratum weights for cross-doc expansion."
            )

        # Diagnostics: distinguish actual mismatches (NaN) from natural weight=1.0
        was_nan = df["weight"].isna()
        n_fallback = was_nan.sum()

        df["weight"] = df["weight"].fillna(1.0)
        df["weight_doc"] = df["weight_doc"].fillna(1.0)

        n_natural_one = ((df["weight"] == 1.0) & ~was_nan).sum()
        total = len(df)

        logger.info(f"Reweighting complete.")
        logger.info(f"  Stratum Weight (Population): Mean={df['weight'].mean():.2f}, Max={df['weight'].max():.2f}")
        logger.info(f"  Document Weight (Specific):  Mean={df['weight_doc'].mean():.2f}, Max={df['weight_doc'].max():.2f}")
        if n_natural_one > 0:
            logger.info(f"  Fully-sampled docs (weight=1.0, correct): {n_natural_one:,}")

        if n_fallback > 0:
            pct = n_fallback / total * 100 if total > 0 else 0
            logger.warning(f"{n_fallback} comments could not be weighted (metadata mismatch, defaulted to 1.0)")
            unweighted = df[was_nan]
            if "agency" in unweighted.columns:
                for agency, count in unweighted["agency"].value_counts().head(10).items():
                    logger.warning(f"    {agency}: {count} comments")
            if pct >= 5:
                logger.warning(f"High unweightable rate ({pct:.1f}%) - check agency/bin matching")
        else:
            logger.info(f"  [OK] All comments successfully weighted.")
        
        return df

    except Exception as e:
        logger.error(f"Weight calculation failed: {e}", exc_info=True)
        raise RuntimeError(
            f"Weight calculation failed — refusing to silently default to 1.0. "
            f"Fix the underlying issue before generating plots. Error: {e}"
        ) from e


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


# ============================================================================
# ONE-PAGER REDESIGNED CHARTS (added 2026-04-09)
# ----------------------------------------------------------------------------
# These two functions replace, in the project one-pager only, the roles of:
#   - plot_composition_donut     -> plot_composition_dumbbell   (shift story)
#   - plot_decision_by_type      -> plot_voice_vs_outcome       (disconnect)
# The originals remain in place and are still generated by
# generate_plots_for_year(). The dumbbell + disconnect charts are additive
# outputs designed to read at ~500px wide and deliver their narrative at a
# glance for a funder/partner audience.
# ============================================================================


def plot_voice_vs_outcome(df: pd.DataFrame, year: int, outdir: Path) -> None:
    """
    One-pager chart: attention multiplier (option C from design review).

    For each commenter type, computes the ratio of their share of accepted
    comments to their share of voice. A value of 1.0x means a group is
    accepted at exactly the rate it participates. Above 1.0x = they punch
    above their weight. Below 1.0x = they are underheard.

    Single horizontal bar per category, sorted descending by multiplier.
    A dashed vertical reference line at 1.0x marks parity. One number per
    group, directly comparable, no geometry decoding.

    Same data conventions as the original ``plot_decision_by_type`` /
    earlier voice-vs-outcome prototype:
      - voice share uses ``weight``
      - outcome share filters to response_found == "yes" AND
        agency_decision == "accept", weighted by ``weight_doc``
    """
    if "weight" not in df.columns:
        df["weight"] = 1.0
    if "weight_doc" not in df.columns:
        df["weight_doc"] = df["weight"]

    # --- Voice share ------------------------------------------------------
    voice_weights = (
        df.groupby("category")["weight"].sum().reindex(CATEGORY_ORDER, fill_value=0.0)
    )
    voice_total = voice_weights.sum()
    if voice_total <= 0:
        logger.warning(f"[attention_multiplier] no voice weight for {year}; skipping")
        return
    voice_pct = 100 * voice_weights / voice_total

    # --- Outcome share ----------------------------------------------------
    if "response_found" not in df.columns or "agency_decision" not in df.columns:
        logger.warning(
            f"[attention_multiplier] response tracking columns missing for {year}; skipping"
        )
        return
    df_accepted = df[
        (df["response_found"] == "yes")
        & (df["category"].notna())
        & (df["category"].isin(CATEGORY_ORDER))
        & (df["agency_decision"] == "accept")
    ]
    if len(df_accepted) == 0:
        logger.warning(f"[attention_multiplier] no accepted comments for {year}; skipping")
        return
    outcome_weights = (
        df_accepted.groupby("category")["weight_doc"]
        .sum()
        .reindex(CATEGORY_ORDER, fill_value=0.0)
    )
    outcome_total = outcome_weights.sum()
    if outcome_total <= 0:
        logger.warning(f"[attention_multiplier] outcome weight zero for {year}; skipping")
        return
    outcome_pct = 100 * outcome_weights / outcome_total

    # --- Ratio per category (skip division-by-zero) ----------------------
    multipliers: Dict[str, float] = {}
    for cat in CATEGORY_ORDER:
        v = float(voice_pct[cat])
        o = float(outcome_pct[cat])
        if v > 0:
            multipliers[cat] = o / v
    if not multipliers:
        logger.warning(f"[attention_multiplier] no computable multipliers for {year}; skipping")
        return

    SHORT_NAMES = {
        "Undecided/Anonymous": "Anonymous",
        "Ordinary Citizen": "Ordinary Citizen",
        "Large Organization/Corporation": "Large Org.",
        "Academic/Industry/Expert (incl. small/local business)": "Expert",
        "Political Consultant/Lobbyist": "Lobbyist",
    }

    # Sort descending so the loudest disconnect sits at the top.
    sorted_cats = sorted(multipliers.keys(), key=lambda c: multipliers[c], reverse=True)
    values = [multipliers[c] for c in sorted_cats]
    colors = [ONE_PAGER_PALETTE[c] for c in sorted_cats]
    labels = [SHORT_NAMES[c] for c in sorted_cats]

    fig, ax = plt.subplots(figsize=(9.5, 4.2), constrained_layout=True)

    max_val = max(max(values), 1.0) * 1.22
    y_pos = np.arange(len(sorted_cats))

    # Bars
    ax.barh(
        y_pos, values,
        color=colors, edgecolor="white", linewidth=1.5,
        height=0.65, zorder=2,
    )
    ax.invert_yaxis()

    # (parity reference line and label removed per design review)

    # Inline ratio labels at the end of each bar.
    for i, (v, c) in enumerate(zip(values, colors)):
        ax.text(
            v + max_val * 0.012, i, f"{v:.1f}×",
            va="center", ha="left",
            fontsize=FONT_LABEL + 1, fontweight="bold", color=c,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=FONT_LABEL, fontweight="bold")
    ax.set_xlim(0, max_val)
    ax.set_xlabel(
        "Share of accepted comments ÷ share of all comments",
        fontsize=FONT_LABEL - 1,
    )

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID_COLOR)
    ax.tick_params(labelsize=FONT_LABEL - 2, colors="#5C6670")
    ax.grid(True, axis="x", alpha=GRID_ALPHA, color=GRID_COLOR, linestyle="-", linewidth=0.6)
    ax.set_axisbelow(True)

    ax.set_title(
        "Who has the most influence?",
        fontsize=FONT_TITLE - 2, fontweight="bold", loc="left", pad=10,
    )

    out_path = outdir / f"voice_vs_outcome_{year}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] Saved: {out_path.name}")


def plot_composition_dumbbell(
    df: pd.DataFrame,
    year: int,
    outdir: Path,
    fr_csv_path: Optional[Path] = None,
) -> None:
    """
    One-pager chart: clean donut pair — "All Comments" vs "Without Mass
    Filings".

    Polished variant of ``plot_composition_donut`` for the project one-pager.
    Uses ``ONE_PAGER_PALETTE`` colors, drops the year from titles, reframes
    "Excluding Top 1% Documents" as "Without Mass Filings" for instant
    comprehension, and replaces the verbose delta-legend with a compact
    color-swatch row. Large slices get inline % labels; small slices are
    identified via the legend only.

    Function name kept as ``plot_composition_dumbbell`` to avoid re-wiring.
    """
    if "weight" not in df.columns:
        df["weight"] = 1.0

    # --- Weighted shares: all comments ------------------------------------
    counts_all = (
        df.groupby("category")["weight"].sum().reindex(CATEGORY_ORDER, fill_value=0.0)
    )
    if counts_all.sum() <= 0:
        logger.warning(f"[composition_donut_clean] no data for {year}; skipping")
        return
    pct_all = 100 * counts_all / counts_all.sum()

    # --- Weighted shares: excluding top 1% of docs by comment_count -------
    df_excluded = None
    if (
        fr_csv_path
        and Path(fr_csv_path).exists()
        and "document_number" in df.columns
    ):
        try:
            fr_df = pd.read_csv(
                fr_csv_path, usecols=["document_number", "comment_count"]
            )
            total_docs = len(fr_df)
            top_n = max(10, int(np.ceil(total_docs * 0.01)))
            top_docs = set(
                fr_df.nlargest(top_n, "comment_count")["document_number"].tolist()
            )
            df_excluded = df[~df["document_number"].isin(top_docs)]
        except Exception as e:
            logger.warning(
                f"[composition_donut_clean] exclusion failed for {year}: {e}"
            )

    if df_excluded is None or len(df_excluded) == 0:
        logger.warning(
            f"[composition_donut_clean] no exclusion data for {year}; skipping"
        )
        return

    counts_excl = (
        df_excluded.groupby("category")["weight"]
        .sum()
        .reindex(CATEGORY_ORDER, fill_value=0.0)
    )
    pct_excl = (
        100 * counts_excl / counts_excl.sum()
        if counts_excl.sum() > 0
        else pct_all * 0
    )

    CALLOUT_NAMES = {
        "Undecided/Anonymous": "Anonymous",
        "Ordinary Citizen": "Citizen",
        "Large Organization/Corporation": "Large Org.",
        "Academic/Industry/Expert (incl. small/local business)": "Expert",
        "Political Consultant/Lobbyist": "Lobbyist",
    }
    LEGEND_NAMES = {
        "Undecided/Anonymous": "Anonymous",
        "Ordinary Citizen": "Ordinary Citizen",
        "Large Organization/Corporation": "Large Org.",
        "Academic/Industry/Expert (incl. small/local business)": "Expert",
        "Political Consultant/Lobbyist": "Lobbyist",
    }

    # Only annotate the two key categories with external callouts.
    HIGHLIGHT = {
        "Ordinary Citizen",
        "Large Organization/Corporation",
    }

    colors = [ONE_PAGER_PALETTE[c] for c in CATEGORY_ORDER]

    # --- Figure: 3-column gridspec — donut | comparison | donut -----------
    fig = plt.figure(figsize=(16, 6))
    gs = fig.add_gridspec(
        1, 3, width_ratios=[1, 0.38, 1], wspace=0.02,
    )
    ax1 = fig.add_subplot(gs[0, 0])
    ax_mid = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    donut_kw = dict(
        startangle=90,
        wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2.5),
    )

    # Left donut — no external labels, the center column handles it.
    wedges1, _ = ax1.pie(counts_all.values, colors=colors, **donut_kw)
    ax1.set_title(
        "All Comments", fontsize=FONT_TITLE - 1, fontweight="bold", pad=16,
    )

    # Right donut
    wedges2, _ = ax2.pie(counts_excl.values, colors=colors, **donut_kw)
    ax2.set_title(
        "Without Campaigns",
        fontsize=FONT_TITLE - 1, fontweight="bold", pad=16,
    )

    # --- Center comparison column -----------------------------------------
    ax_mid.axis("off")

    citizen_all = float(pct_all.get("Ordinary Citizen", 0))
    citizen_excl = float(pct_excl.get("Ordinary Citizen", 0))
    org_all = float(pct_all.get("Large Organization/Corporation", 0))
    org_excl = float(pct_excl.get("Large Organization/Corporation", 0))

    c_citizen = ONE_PAGER_PALETTE["Ordinary Citizen"]
    c_org = ONE_PAGER_PALETTE["Large Organization/Corporation"]

    # Citizen comparison — upper half
    ax_mid.text(
        0.5, 0.68, "Citizen",
        ha="center", va="bottom",
        fontsize=FONT_LABEL + 1, fontweight="bold", color=c_citizen,
    )
    ax_mid.text(
        0.5, 0.62,
        f"{citizen_all:.0f}%   \u2192   {citizen_excl:.0f}%",
        ha="center", va="top",
        fontsize=FONT_TITLE + 4, fontweight="bold", color=c_citizen,
    )

    # Large Org comparison — lower half
    ax_mid.text(
        0.5, 0.38, "Large Org.",
        ha="center", va="bottom",
        fontsize=FONT_LABEL + 1, fontweight="bold", color=c_org,
    )
    ax_mid.text(
        0.5, 0.32,
        f"{org_all:.0f}%   \u2192   {org_excl:.0f}%",
        ha="center", va="top",
        fontsize=FONT_TITLE + 4, fontweight="bold", color=c_org,
    )

    # Compact legend — color swatches + short names, single row.
    from matplotlib.patches import Patch

    legend_handles = [
        Patch(
            facecolor=ONE_PAGER_PALETTE[c], edgecolor="white",
            label=LEGEND_NAMES[c],
        )
        for c in CATEGORY_ORDER
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center", bbox_to_anchor=(0.5, 0.01),
        ncol=len(CATEGORY_ORDER), frameon=False,
        fontsize=FONT_LABEL - 1,
        handlelength=1.2, handletextpad=0.5, columnspacing=1.5,
    )
    out_path = outdir / f"composition_shift_{year}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] Saved: {out_path.name}")


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
        
    # Check if we have weights — weight_doc is REQUIRED for document-level plots
    if "weight_doc" not in df_doc_view.columns:
        logger.error(
            "weight_doc not available for document-level plot (year=%s). "
            "Cannot use stratum weight as substitute — would violate Y<=X invariant. Skipping.",
            year,
        )
        return
        
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
    # Use trapezoidal integration (trapz equivalent)
    try:
        gini = float(1 - 2 * np.trapz(lorenz_y, lorenz_x))
    except AttributeError:
        # Fallback: manual trapezoidal integration
        dx = lorenz_x[1:] - lorenz_x[:-1]
        y_avg = (lorenz_y[1:] + lorenz_y[:-1]) / 2
        area = np.sum(dx * y_avg)
        gini = float(1 - 2 * area)
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

        if len(doc_sorted) == 0 or idx >= len(doc_sorted):
            continue

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
                
                # Urgency / Attention — use known-count docs only (consistent with zero-doc stats)
                # Top 1% of documents capture X% of comments
                top_1_pct_n = max(1, int(np.ceil(total_docs_known * 0.01)))
                top_docs = fr_df_known.nlargest(top_1_pct_n, "comment_count")
                top_vol = top_docs["comment_count"].sum()
                total_vol = fr_df_known["comment_count"].sum()
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

# ============================================================================
# RESPONSE TRACKING PLOTS
# ============================================================================

def load_response_data(year: int) -> Optional[pd.DataFrame]:
    """Load agency_responses_{year}.csv if exists"""
    responses_path = get_agency_responses_path(year)
    if not responses_path.exists():
        return None
    return pd.read_csv(responses_path)


def generate_response_plots(
    df_responses: pd.DataFrame,
    df_makeup: pd.DataFrame,
    plots_dir: Path,
    year: int,
) -> None:
    """
    Generate all response tracking plots.
    
    Args:
        df_responses: DataFrame with response tracking data
        df_makeup: DataFrame with makeup data (for joining commenter types and weights)
        plots_dir: Directory to save plots
        year: Year being analyzed
    """
    logger.info("Generating response tracking plots...")
    
    # Deduplicate responses by comment_id to avoid cross-products if a comment
    # was sent to multiple agencies (we want to count each unique comment once
    # for these global statistics). 
    # Keep the first response found for each comment.
    df_responses = df_responses.drop_duplicates(subset=["comment_id"])
    
    # Join responses with makeup data on comment_id
    # Preserve weights and other necessary columns from makeup data
    # Note: df_responses already has "agency" column (from track_responses.py line 362)
    # We don't merge agency from makeup - keep the one from df_responses
    merge_cols = ["comment_id", "category"]
    if "weight" in df_makeup.columns:
        merge_cols.append("weight")
    if "weight_doc" in df_makeup.columns:
        merge_cols.append("weight_doc")
    if "document_number" in df_makeup.columns:
        merge_cols.append("document_number")
    
    # Merge without agency - df_responses already has it and we want to keep that one
    # Use inner join to ensure we only keep comments that have both response data AND makeup classification
    # This prevents counting comments without category assignments
    df = df_responses.merge(df_makeup[merge_cols], on="comment_id", how="inner")
    
    # Log join statistics for debugging
    logger.info(f"Joined {len(df_responses):,} responses with {len(df_makeup):,} makeup records -> {len(df):,} matched rows")
    
    # Ensure agency exists (should already be there from df_responses)
    # Fill any missing values with "Unknown"
    if "agency" not in df.columns:
        df["agency"] = "Unknown"
    else:
        df["agency"] = df["agency"].fillna("Unknown")
    
    # Ensure weights exist (default to 1.0 if missing)
    if "weight" not in df.columns:
        df["weight"] = 1.0
    if "weight_doc" not in df.columns:
        df["weight_doc"] = df.get("weight", 1.0)

    # Apply response sampling weight if present (Horvitz-Thompson: multiply
    # doc-level weight by the inverse sampling probability from response tracking)
    if "response_sample_weight" in df.columns:
        rsw = pd.to_numeric(df["response_sample_weight"], errors="coerce").fillna(1.0)
        df["weight"] = df["weight"] * rsw
        df["weight_doc"] = df["weight_doc"] * rsw
        n_sampled = (rsw > 1.0).sum()
        if n_sampled > 0:
            logger.info(f"Applied response_sample_weight to {n_sampled:,} sampled rows (max weight={rsw.max():.1f})")
    
    # Normalize categories (this filters to only known categories)
    df_before_norm = len(df)
    df = normalize_categories(df)
    df_after_norm = len(df)
    if df_before_norm != df_after_norm:
        logger.info(f"Category normalization: {df_before_norm:,} -> {df_after_norm:,} rows (filtered {df_before_norm - df_after_norm:,} unknown categories)")
    
    # Validate that all rows have valid categories
    invalid_cats = df[~df["category"].isin(CATEGORY_ORDER)]
    if len(invalid_cats) > 0:
        logger.warning(f"Found {len(invalid_cats):,} rows with invalid categories after normalization - these will be excluded")
        df = df[df["category"].isin(CATEGORY_ORDER)]
    
    # 1. Response Rate by Agency
    plot_response_rate_by_agency(df, year, plots_dir)
    
    # 2. Accept/Reject Distribution (replaced with text report)
    write_decision_distribution_report(df, year, plots_dir)
    
    # 4. Accept/Reject by Commenter Type
    plot_decision_by_type(df, year, plots_dir)

    # 4b. Voice-vs-outcome disconnect chart (one-pager redesign of 4)
    plot_voice_vs_outcome(df, year, plots_dir)
    
    # 5. Comment Length vs Rejection (two plots)
    plot_length_vs_rejection_binned(df, year, plots_dir)
    plot_length_vs_rejection_logistic(df, year, plots_dir)
    
    # 7. Agency Response Heatmap
    plot_response_heatmap(df, year, plots_dir)
    
    # 8. Agency Response vs Acceptance Rate
    plot_agency_response_acceptance(df, year, plots_dir)
    
    logger.info("Response tracking plots complete")


def plot_response_rate_by_agency(df: pd.DataFrame, year: int, outdir: Path) -> None:
    """Bar chart showing % of comments responded to per agency (weighted)"""
    if len(df) == 0 or "agency" not in df.columns:
        logger.warning("No data for response rate by agency plot")
        return
    
    # Ensure weights exist
    if "weight" not in df.columns:
        df["weight"] = 1.0
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Calculate response rates by agency using DOCUMENT WEIGHTED statistics
    df["responded"] = (df["response_found"] == "yes").astype(int)
    
    # Calculate weighted response rates
    agency_stats = []
    for agency, group in df.groupby("agency"):
        total_weight = group["weight_doc"].sum()
        if total_weight > 0:
            responded_weight = group[group["responded"] == 1]["weight_doc"].sum()
            response_rate = (responded_weight / total_weight * 100) if total_weight > 0 else 0
            agency_stats.append({
                "agency": agency,
                "response_rate": response_rate
            })
    agency_stats = pd.DataFrame(agency_stats)
    
    if len(agency_stats) == 0:
        logger.warning("No agency data for response rate plot")
        plt.close()
        return
    
    # Sort by response rate
    agency_stats = agency_stats.sort_values("response_rate", ascending=True)
    
    # Calculate dynamic height
    fig_height = max(8, len(agency_stats) * 0.25)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    
    # Plot
    ax.barh(agency_stats["agency"], agency_stats["response_rate"], color="#6B7D6D")
    ax.set_xlabel("Response Rate (%)", fontsize=FONT_LABEL)
    ax.set_title(f"Agency Response Rate - {year}", fontsize=FONT_TITLE, fontweight="bold")
    ax.grid(axis="x", alpha=GRID_ALPHA, color=GRID_COLOR)
    
    plt.tight_layout()
    plt.savefig(outdir / f"response_rate_by_agency_{year}.png", dpi=300, bbox_inches="tight")
    plt.close()


def write_decision_distribution_report(df: pd.DataFrame, year: int, outdir: Path) -> None:
    """
    Write a text report summarizing agency decision distribution.
    Uses weighted statistics to account for sampling.
    
    Outputs tables:
    1. Decision counts and proportions (weighted)
    2. Conditional probabilities (P(accept|response), P(reject|response))
    3. Breakdown by commenter type (weighted)
    4. Notable asymmetries (agencies with extreme acceptance/rejection rates, weighted)
    """
    try:
        # Ensure weights exist
        if "weight" not in df.columns:
            df["weight"] = 1.0
        
        # Filter to responses that were found AND have valid category
        # This ensures we only count comments with both response data and makeup classification
        df_found = df[
            (df["response_found"] == "yes") & 
            (df["category"].notna()) & 
            (df["category"].isin(CATEGORY_ORDER))
        ].copy()
        
        if len(df_found) == 0:
            logger.warning("No responses found with valid categories for decision distribution report")
            return
        
        # Validate agency_decision values
        valid_decisions = ["accept", "reject", "partial", "uncertain"]
        invalid_decisions = df_found[~df_found["agency_decision"].isin(valid_decisions)]
        if len(invalid_decisions) > 0:
            logger.warning(f"Found {len(invalid_decisions):,} rows with invalid agency_decision values - excluding from report")
            df_found = df_found[df_found["agency_decision"].isin(valid_decisions)]

        # Count decisions using DOCUMENT WEIGHTED sums
        decision_weighted = df_found.groupby("agency_decision")["weight_doc"].sum()
        total_responses_weighted = df_found["weight_doc"].sum()

        # Calculate proportions (weighted)
        decision_props = (decision_weighted / total_responses_weighted * 100) if total_responses_weighted > 0 else pd.Series()

        # Calculate conditional probabilities (weighted)
        accept_weighted = decision_weighted.get("accept", 0)
        reject_weighted = decision_weighted.get("reject", 0)
        partial_weighted = decision_weighted.get("partial", 0)
        uncertain_weighted = decision_weighted.get("uncertain", 0)

        p_accept_given_response = (accept_weighted / total_responses_weighted * 100) if total_responses_weighted > 0 else 0
        p_reject_given_response = (reject_weighted / total_responses_weighted * 100) if total_responses_weighted > 0 else 0
        p_partial_given_response = (partial_weighted / total_responses_weighted * 100) if total_responses_weighted > 0 else 0
        p_uncertain_given_response = (uncertain_weighted / total_responses_weighted * 100) if total_responses_weighted > 0 else 0
        
        # Breakdown by commenter type (weighted)
        # Calculate weighted proportions for each category × decision combination
        decision_by_type_weighted = {}
        for cat in CATEGORY_ORDER:
            cat_data = df_found[df_found["category"] == cat]
            if len(cat_data) > 0:
                cat_total_weighted = cat_data["weight_doc"].sum()
                if cat_total_weighted > 0:
                    cat_decisions = {}
                    for decision in ["accept", "reject", "partial", "uncertain"]:
                        decision_data = cat_data[cat_data["agency_decision"] == decision]
                        decision_weight = decision_data["weight_doc"].sum()
                        cat_decisions[decision] = (decision_weight / cat_total_weighted * 100) if cat_total_weighted > 0 else 0
                    decision_by_type_weighted[cat] = cat_decisions

        # Agency-level asymmetries (weighted)
        # Calculate weighted acceptance/rejection rates by agency
        agency_stats = []
        for agency, agency_data in df_found.groupby("agency"):
            agency_total_weighted = agency_data["weight_doc"].sum()
            if agency_total_weighted > 0:
                accept_weight = agency_data[agency_data["agency_decision"] == "accept"]["weight_doc"].sum()
                reject_weight = agency_data[agency_data["agency_decision"] == "reject"]["weight_doc"].sum()
                partial_weight = agency_data[agency_data["agency_decision"] == "partial"]["weight_doc"].sum()
                uncertain_weight = agency_data[agency_data["agency_decision"] == "uncertain"]["weight_doc"].sum()
                
                accept_pct = (accept_weight / agency_total_weighted * 100) if agency_total_weighted > 0 else 0
                reject_pct = (reject_weight / agency_total_weighted * 100) if agency_total_weighted > 0 else 0
                
                agency_stats.append({
                    "agency": agency,
                    "accept_pct": accept_pct,
                    "reject_pct": reject_pct,
                    "total_weighted": agency_total_weighted
                })

        agency_df = pd.DataFrame(agency_stats)
        agency_totals = agency_df.set_index("agency")["total_weighted"]
        
        # Find agencies with >50% acceptance or rejection (clear threshold, not percentile)
        if len(agency_df) > 0:
            high_acceptance = agency_df[agency_df["accept_pct"] > 50].sort_values("accept_pct", ascending=False)
            high_rejection = agency_df[agency_df["reject_pct"] > 50].sort_values("reject_pct", ascending=False)
        else:
            high_acceptance = pd.DataFrame()
            high_rejection = pd.DataFrame()
        
        # Create report
        report_lines = []
        report_lines.append(f"Agency Decision Distribution Report ({year})")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Table 1: Decision counts and proportions
        report_lines.append("TABLE 1: DECISION COUNTS AND PROPORTIONS")
        report_lines.append("-" * 50)
        report_lines.append(f"{'Decision':<20} | {'Count':>10} | {'Proportion':>12}")
        report_lines.append("-" * 50)
        
        for decision in ["accept", "reject", "partial", "uncertain"]:
            count_weighted = decision_weighted.get(decision, 0)
            prop = decision_props.get(decision, 0)
            report_lines.append(f"{decision.capitalize():<20} | {count_weighted:>10,.0f} | {prop:>11.1f}%")
        
        report_lines.append("-" * 50)
        report_lines.append(f"{'TOTAL':<20} | {total_responses_weighted:>10,.0f} | {100.0:>11.1f}%")
        report_lines.append("")
        report_lines.append("")
        
        # Table 2: Conditional probabilities
        report_lines.append("TABLE 2: CONDITIONAL PROBABILITIES")
        report_lines.append("-" * 50)
        report_lines.append("Given that a response was found:")
        report_lines.append(f"  P(accept | response) = {p_accept_given_response:.1f}%")
        report_lines.append(f"  P(reject | response) = {p_reject_given_response:.1f}%")
        report_lines.append(f"  P(partial | response) = {p_partial_given_response:.1f}%")
        report_lines.append(f"  P(uncertain | response) = {p_uncertain_given_response:.1f}%")
        report_lines.append("")
        report_lines.append("")
        
        # Table 3: Breakdown by commenter type
        report_lines.append("TABLE 3: DECISION RATES BY COMMENTER TYPE")
        report_lines.append("-" * 80)
        header = f"{'Commenter Type':<45} | {'Accept %':>10} | {'Reject %':>10} | {'Partial %':>10} | {'Uncertain %':>12}"
        report_lines.append(header)
        report_lines.append("-" * 95)

        for cat in CATEGORY_ORDER:
            if cat in decision_by_type_weighted:
                cat_decisions = decision_by_type_weighted[cat]
                accept_pct = cat_decisions.get("accept", 0)
                reject_pct = cat_decisions.get("reject", 0)
                partial_pct = cat_decisions.get("partial", 0)
                uncertain_pct = cat_decisions.get("uncertain", 0)
                cat_display = CATEGORY_DISPLAY_NAMES.get(cat, cat)
                report_lines.append(f"{cat_display:<45} | {accept_pct:>9.1f}% | {reject_pct:>9.1f}% | {partial_pct:>9.1f}% | {uncertain_pct:>11.1f}%")
        
        report_lines.append("-" * 80)
        report_lines.append("")
        report_lines.append("")
        
        # Table 4: Agencies with >50% acceptance or rejection
        report_lines.append("TABLE 4: AGENCIES BY DECISION RATE")
        report_lines.append("-" * 100)
        
        if len(high_acceptance) > 0:
            report_lines.append("Agencies with >50% Acceptance Rate:")
            report_lines.append("-" * 100)
            header = f"{'Agency':<50} | {'Accept %':>10} | {'Total Responses':>15}"
            report_lines.append(header)
            report_lines.append("-" * 100)
            for _, row in high_acceptance.iterrows():
                accept_pct = row["accept_pct"]
                total = row["total_weighted"]
                agency_display = str(row["agency"])[:48]
                report_lines.append(f"{agency_display:<50} | {accept_pct:>9.1f}% | {total:>15,.0f}")
            report_lines.append("")
        
        if len(high_rejection) > 0:
            report_lines.append("Agencies with >50% Rejection Rate:")
            report_lines.append("-" * 100)
            header = f"{'Agency':<50} | {'Reject %':>10} | {'Total Responses':>15}"
            report_lines.append(header)
            report_lines.append("-" * 100)
            for _, row in high_rejection.iterrows():
                reject_pct = row["reject_pct"]
                total = row["total_weighted"]
                agency_display = str(row["agency"])[:48]
                report_lines.append(f"{agency_display:<50} | {reject_pct:>9.1f}% | {total:>15,.0f}")
        
        report_lines.append("-" * 100)
        report_lines.append("")
        report_lines.append("Notes:")
        report_lines.append("- All percentages are conditional on response being found")
        report_lines.append("- All counts and percentages use WEIGHTED statistics to account for sampling")
        report_lines.append("- Agencies shown have >50% acceptance or rejection rate")
        report_lines.append("- Only agencies with at least 1 response are included")
        
        # Write to file
        report_path = outdir / f"decision_distribution_report_{year}.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))
        
        print(f"[OK] Saved: decision_distribution_report_{year}.txt")
        
    except Exception as e:
        logger.error(f"Error writing decision distribution report ({year}): {e}")
        import traceback
        traceback.print_exc()


def plot_decision_by_type(df: pd.DataFrame, year: int, outdir: Path) -> None:
    """Grouped bar chart of accept/reject by commenter type (weighted)"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Ensure weights exist
    if "weight" not in df.columns:
        df["weight"] = 1.0
    
    # CRITICAL: Filter to responses that were found AND have valid category
    # Must filter on both conditions to ensure we're only counting comments with:
    # 1. A response was found (response_found == "yes")
    # 2. A valid category assignment (category is not null and is in CATEGORY_ORDER)
    df_found = df[
        (df["response_found"] == "yes") & 
        (df["category"].notna()) & 
        (df["category"].isin(CATEGORY_ORDER))
    ].copy()
    
    if len(df_found) == 0:
        logger.warning("No responses found with valid categories for decision by type plot")
        return
    
    # Additional validation: ensure agency_decision is valid
    valid_decisions = ["accept", "reject", "partial", "uncertain"]
    df_found = df_found[df_found["agency_decision"].isin(valid_decisions)].copy()
    
    if len(df_found) == 0:
        logger.warning("No responses with valid decisions for decision by type plot")
        return
    
    # Log diagnostic info
    logger.info(f"Plot data: {len(df_found):,} rows with response_found=='yes' and valid category")
    logger.info(f"Categories in data: {sorted(df_found['category'].unique().tolist())}")
    
    # Calculate decision rates by category using DOCUMENT WEIGHTED statistics
    decision_by_type = {}
    for cat in CATEGORY_ORDER:
        # Explicit filtering: must match category exactly (after normalization)
        cat_data = df_found[df_found["category"] == cat].copy()
        if len(cat_data) > 0:
            cat_total_weighted = cat_data["weight_doc"].sum()
            if cat_total_weighted > 0:
                cat_decisions = {}
                for decision in valid_decisions:
                    decision_data = cat_data[cat_data["agency_decision"] == decision].copy()
                    decision_weight = decision_data["weight_doc"].sum()
                    pct = (decision_weight / cat_total_weighted * 100) if cat_total_weighted > 0 else 0
                    cat_decisions[decision] = pct
                    
                    # Log consultant stats for debugging
                    if cat == "Political Consultant/Lobbyist":
                        logger.debug(f"  Consultants - {decision}: {len(decision_data):,} rows, {decision_weight:.2f} weight, {pct:.1f}%")
                
                decision_by_type[cat] = cat_decisions
                
                # Log total for each category
                if cat == "Political Consultant/Lobbyist":
                    logger.info(f"Consultants total: {len(cat_data):,} rows, {cat_total_weighted:.2f} weight")
                    logger.info(f"Consultants breakdown: accept={cat_decisions.get('accept', 0):.1f}%, reject={cat_decisions.get('reject', 0):.1f}%, partial={cat_decisions.get('partial', 0):.1f}%, uncertain={cat_decisions.get('uncertain', 0):.1f}%")
    
    # Convert to DataFrame for plotting
    decision_df = pd.DataFrame(decision_by_type).T.fillna(0)
    decision_df = decision_df.reindex(CATEGORY_ORDER, fill_value=0)
    
    # Rename index for display using global mapping
    decision_by_type_display = decision_df.rename(index=CATEGORY_DISPLAY_NAMES)
    
    # Ensure all decision columns exist
    for decision in valid_decisions:
        if decision not in decision_by_type_display.columns:
            decision_by_type_display[decision] = 0
    
    # Validate that percentages sum to ~100% for each category (allowing for rounding)
    # This is a sanity check to catch any logic errors
    for cat in decision_by_type_display.index:
        row_sum = decision_by_type_display.loc[cat, valid_decisions].sum()
        if row_sum > 0 and abs(row_sum - 100.0) > 1.0:  # Allow 1% tolerance for rounding
            logger.warning(f"Category {cat} percentages sum to {row_sum:.1f}% (expected ~100%)")
    
    # Plot grouped bar
    decision_by_type_display[valid_decisions].plot(kind="bar", ax=ax, color=["#6B7D6D", "#DDE3EA", "#B8860B", "#9A8153"])
    ax.set_ylabel("Percentage", fontsize=FONT_LABEL)
    ax.set_xlabel("Commenter Type", fontsize=FONT_LABEL)
    ax.set_title(f"Agency Decision by Commenter Type - {year}", fontsize=FONT_TITLE, fontweight="bold")
    ax.legend(title="Decision", fontsize=FONT_LEGEND, bbox_to_anchor=(1.15, 1), loc="upper left")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.grid(axis="y", alpha=GRID_ALPHA, color=GRID_COLOR)
    
    # Adjust layout to ensure legend stays within bounds
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(outdir / f"decision_by_type_{year}.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_length_vs_rejection_binned(df: pd.DataFrame, year: int, outdir: Path) -> None:
    """
    Plot 1: Binned rejection probability by comment length (weighted).
    Shows P(reject | length bin) with smoothed trend line.
    """
    if len(df) == 0 or "response_found" not in df.columns:
        logger.warning("No data for length vs rejection binned plot")
        return
    
    # Ensure weights exist
    if "weight" not in df.columns:
        df["weight"] = 1.0
        
    # Filter to responses that were found and have valid decisions
    df_found = df[(df["response_found"] == "yes") & (df["agency_decision"].notna())].copy()
    
    if len(df_found) == 0:
        logger.warning("No responses found for length vs rejection binned plot")
        return
        
    if "comment_text_length" not in df_found.columns:
        logger.warning("No comment_text_length column for plot")
        return
        
    # Create binary rejection variable
    df_found["reject"] = (df_found["agency_decision"] == "reject").astype(int)
    
    # Filter out invalid lengths and cap at 95th percentile
    df_valid = df_found[df_found["comment_text_length"].notna() & (df_found["comment_text_length"] > 0)].copy()
    
    if len(df_valid) == 0:
        logger.warning("No valid comment lengths for plot")
        return
        
    # Cap at 95th percentile to handle long tail (use weighted quantile)
    max_length = weighted_quantile(df_valid["comment_text_length"].values, df_valid["weight_doc"].values, 0.95)
    df_valid = df_valid[df_valid["comment_text_length"] <= max_length]
    
    if len(df_valid) < 20:
        logger.warning("Insufficient data for binned plot")
        return
    
    # Use quantile-based binning (equal-count bins)
    n_bins = min(20, max(5, int(len(df_valid) / 10)))  # Adaptive bin count
    try:
        df_valid["length_bin"] = pd.qcut(df_valid["comment_text_length"], q=n_bins, duplicates="drop")
    except ValueError:
        # Fallback to regular cut if qcut fails
        df_valid["length_bin"] = pd.cut(df_valid["comment_text_length"], bins=n_bins)
    
    # Calculate rejection probability by bin using DOCUMENT WEIGHTED statistics
    bin_stats = []
    for bin_label, group in df_valid.groupby("length_bin", observed=True):
        total_weight = group["weight_doc"].sum()
        reject_weight = group[group["reject"] == 1]["weight_doc"].sum()
        reject_prob = (reject_weight / total_weight) if total_weight > 0 else 0
        # Ensure probability stays within [0, 1] bounds
        reject_prob = np.clip(reject_prob, 0.0, 1.0)
        bin_stats.append({
            "length_bin": bin_label,
            "reject_prob": reject_prob,
            "count": len(group),
            "bin_center": group["comment_text_length"].mean()
        })
    bin_stats = pd.DataFrame(bin_stats)
    
    # Filter bins with minimum size (at least 5 comments)
    min_bin_size = 5
    bin_stats = bin_stats[bin_stats["count"] >= min_bin_size]
    
    if len(bin_stats) == 0:
        logger.warning("No bins with sufficient data")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot binned probabilities (convert to percentage)
    reject_pct = bin_stats["reject_prob"].values * 100
    ax.scatter(bin_stats["bin_center"], reject_pct, 
              s=30, alpha=0.6, color="#6B7D6D", label="Binned data")
    
    # Add smoothed trend line using UnivariateSpline
    try:
        from scipy.interpolate import UnivariateSpline
        # Sort by bin_center for spline
        bin_stats_sorted = bin_stats.sort_values("bin_center")
        x_smooth = bin_stats_sorted["bin_center"].values
        y_smooth = bin_stats_sorted["reject_prob"].values * 100
        
        if len(x_smooth) >= 3:
            # Use tighter smoothing parameter to reduce overshooting
            # Increase smoothing factor to reduce oscillations
            smoothing_factor = len(x_smooth) * 0.2  # Increased from 0.1 to 0.2
            spline = UnivariateSpline(x_smooth, y_smooth, s=smoothing_factor, k=min(3, len(x_smooth) - 1))
            x_fit = np.linspace(x_smooth.min(), x_smooth.max(), 200)
            y_fit = spline(x_fit)
            # CRITICAL FIX: Clip smoothed values to valid probability range [0, 100]
            y_fit = np.clip(y_fit, 0.0, 100.0)
            ax.plot(x_fit, y_fit, color="#9A8153", linewidth=2.5, alpha=0.9, label="Smoothed trend")
    except Exception as e:
        logger.debug(f"Could not fit spline: {e}")
    
    ax.set_xlabel("Comment Length (characters)", fontsize=FONT_LABEL, fontweight="bold")
    ax.set_ylabel("P(Reject | Length Bin) (%)", fontsize=FONT_LABEL, fontweight="bold")
    ax.set_title(f"Rejection Probability by Comment Length ({year})", fontsize=FONT_TITLE, fontweight="bold")
    # Ensure y-axis stays within valid probability range
    ax.set_ylim(-2, 102)
    ax.grid(alpha=GRID_ALPHA, color=GRID_COLOR)
    ax.legend(fontsize=FONT_LEGEND - 1, frameon=True, loc="upper right", framealpha=0.9)
    apply_clean_style(ax)
    
    plt.tight_layout()
    plt.savefig(outdir / f"length_vs_rejection_binned_{year}.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: length_vs_rejection_binned_{year}.png")


def plot_length_vs_rejection_logistic(df: pd.DataFrame, year: int, outdir: Path) -> None:
    """
    Statistical plot: Association between comment length and rejection probability (weighted).
    
    Shows:
    - Binned empirical rejection rates with confidence intervals (weighted)
    - Logistic regression curve with 95% CI (weighted)
    - Rug plot showing data distribution
    - Uses log(length) for proper modeling
    """
    if len(df) == 0 or "response_found" not in df.columns:
        logger.warning("No data for length vs rejection logistic plot")
        return
    
    # Ensure weights exist
    if "weight" not in df.columns:
        df["weight"] = 1.0
    
    # Filter to responses that were found and have valid decisions
    df_found = df[(df["response_found"] == "yes") & (df["agency_decision"].notna())].copy()
    
    if len(df_found) == 0:
        logger.warning("No responses found for length vs rejection logistic plot")
        return
    
    if "comment_text_length" not in df_found.columns:
        logger.warning("No comment_text_length column for plot")
        return
    
    # Create binary rejection variable
    df_found["reject"] = (df_found["agency_decision"] == "reject").astype(int)
    
    # Filter out invalid lengths
    df_valid = df_found[df_found["comment_text_length"].notna() & (df_found["comment_text_length"] > 0)].copy()
    
    if len(df_valid) < 20:
        logger.warning("Insufficient data for logistic regression")
        return
    
    # Use log1p(length) for modeling (handles zeros naturally)
    df_valid["log_length"] = np.log1p(df_valid["comment_text_length"])
    
    # Check for statsmodels availability
    try:
        import statsmodels.api as sm
        from scipy import stats
    except ImportError:
        logger.warning("statsmodels not available, skipping logistic regression plot")
        return
    
    # Fit weighted logistic regression with proper uncertainty estimation
    try:
        X = df_valid[["log_length"]].values
        y = df_valid["reject"].values
        weights = df_valid["weight_doc"].values
        
        # Add intercept for statsmodels
        X_with_const = sm.add_constant(X)
        
        # Fit weighted logistic regression
        model = sm.Logit(y, X_with_const, weights=weights).fit(disp=0)
        
        # Generate prediction range (in log space)
        log_length_min = df_valid["log_length"].min()
        log_length_max = df_valid["log_length"].max()
        x_range_log = np.linspace(log_length_min, log_length_max, 200)
        x_range_log_const = sm.add_constant(x_range_log)
        
        # Get predicted probabilities and confidence intervals
        y_pred = model.predict(x_range_log_const) * 100
        
        # Get 95% confidence intervals
        pred_ci = model.get_prediction(x_range_log_const).conf_int(alpha=0.05)
        y_pred_lower = pred_ci[:, 0] * 100
        y_pred_upper = pred_ci[:, 1] * 100
        
        # Convert back to original scale for plotting
        x_range_original = np.expm1(x_range_log)
        
        # Calculate binned empirical rates for comparison (weighted)
        # Use equal-count bins (quantile-based)
        n_bins = min(20, max(5, int(len(df_valid) / 50)))
        try:
            df_valid["length_bin"] = pd.qcut(df_valid["log_length"], q=n_bins, duplicates="drop")
        except ValueError:
            df_valid["length_bin"] = pd.cut(df_valid["log_length"], bins=n_bins)
        
        # Calculate weighted empirical rejection rates with Wilson confidence intervals
        bin_stats = []
        for bin_label, group in df_valid.groupby("length_bin", observed=True):
            total_weight = group["weight_doc"].sum()
            if total_weight > 0 and len(group) >= 10:  # Minimum bin size
                reject_weight = group[group["reject"] == 1]["weight_doc"].sum()
                p_weighted = reject_weight / total_weight
                
                # Effective sample size for CI calculation (using Kish's effective sample size)
                n_eff = (total_weight ** 2) / (group["weight_doc"] ** 2).sum() if len(group) > 0 else len(group)
                n_eff = max(1, int(n_eff))
                
                # Wilson confidence interval (using effective sample size)
                z = 1.96  # 95% CI
                denominator = 1 + (z**2 / n_eff)
                center = (p_weighted + (z**2 / (2 * n_eff))) / denominator
                margin = (z / denominator) * np.sqrt((p_weighted * (1 - p_weighted) / n_eff) + (z**2 / (4 * n_eff**2)))
                
                bin_stats.append({
                    "log_length": group["log_length"].mean(),
                    "length": group["comment_text_length"].mean(),
                    "reject_rate": p_weighted * 100,
                    "ci_lower": max(0, (center - margin) * 100),
                    "ci_upper": min(100, (center + margin) * 100),
                    "n": len(group),
                    "n_eff": n_eff
                })
        
        bin_df = pd.DataFrame(bin_stats)
        
        # Create figure with two panels: main plot and rug
        fig = plt.figure(figsize=(12, 8))
        gs = fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.1)
        ax_main = fig.add_subplot(gs[0])
        ax_rug = fig.add_subplot(gs[1])
        
        # Main plot: binned empirical rates with CI
        if len(bin_df) > 0:
            # Scale point size by effective sample size (or bin count if n_eff not available)
            size_col = "n_eff" if "n_eff" in bin_df.columns else "n"
            sizes = np.clip(bin_df[size_col] / bin_df[size_col].max() * 200, 20, 200)
            
            ax_main.scatter(bin_df["length"], bin_df["reject_rate"],
                          s=sizes, alpha=0.7, color="#6B7D6D", 
                          edgecolors="white", linewidths=1.5, zorder=3,
                          label="Binned empirical rates")
            
            # Add error bars for confidence intervals
            ax_main.errorbar(bin_df["length"], bin_df["reject_rate"],
                           yerr=[bin_df["reject_rate"] - bin_df["ci_lower"],
                                 bin_df["ci_upper"] - bin_df["reject_rate"]],
                           fmt='none', color="#6B7D6D", alpha=0.4, linewidth=1, zorder=2)
        
        # Plot logistic regression curve with CI band
        ax_main.plot(x_range_original, y_pred, color="#9A8153", linewidth=2.5,
                    alpha=0.9, label="Logistic regression", zorder=4)
        ax_main.fill_between(x_range_original, y_pred_lower, y_pred_upper,
                            color="#9A8153", alpha=0.2, zorder=1,
                            label="95% confidence interval")
        
        # Add model statistics annotation
        coef = model.params[1]  # Coefficient for log_length
        pvalue = model.pvalues[1]
        sign = "decreases" if coef < 0 else "increases"
        sig_level = "***" if pvalue < 0.001 else "**" if pvalue < 0.01 else "*" if pvalue < 0.05 else "ns"
        
        stats_text = f"β = {coef:.3f} ({sign} rejection with length)\np = {pvalue:.4f} {sig_level}"
        ax_main.text(0.02, 0.98, stats_text, transform=ax_main.transAxes,
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
        
        # Styling
        ax_main.set_xscale("log")
        ax_main.set_xlabel("Comment Length (characters, log scale)", fontsize=FONT_LABEL, fontweight="bold")
        ax_main.set_ylabel("P(Reject) (%)", fontsize=FONT_LABEL, fontweight="bold")
        ax_main.set_title(f"Association between Comment Length and Rejection Probability ({year})\nWeighted logistic model", 
                    fontsize=FONT_TITLE, fontweight="bold")
        ax_main.set_ylim(-2, 102)
        ax_main.grid(alpha=GRID_ALPHA, color=GRID_COLOR)
        ax_main.legend(fontsize=FONT_LEGEND - 1, frameon=True, loc="upper right", framealpha=0.9)
        apply_clean_style(ax_main)
        
        # Rug plot: show data distribution
        sample_size = min(5000, len(df_valid))
        df_rug = df_valid.sample(sample_size) if len(df_valid) > sample_size else df_valid
        ax_rug.scatter(df_rug["comment_text_length"], np.ones(len(df_rug)),
                      alpha=0.1, s=1, color="#5C6670", marker="|")
        ax_rug.set_xscale("log")
        ax_rug.set_xlabel("Comment Length Distribution", fontsize=FONT_LABEL - 2, fontweight="bold")
        ax_rug.set_ylabel("", fontsize=1)
        ax_rug.set_yticks([])
        ax_rug.spines["top"].set_visible(False)
        ax_rug.spines["right"].set_visible(False)
        ax_rug.spines["left"].set_visible(False)
        ax_rug.spines["bottom"].set_color(GRID_COLOR)
        ax_rug.tick_params(labelsize=FONT_LABEL - 2, colors="#5C6670")
        
        plt.tight_layout()
        plt.savefig(outdir / f"length_vs_rejection_logistic_{year}.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[OK] Saved: length_vs_rejection_logistic_{year}.png")
        
    except Exception as e:
        logger.warning(f"Error fitting logistic regression: {e}")
        import traceback
        traceback.print_exc()


def plot_response_heatmap(df: pd.DataFrame, year: int, outdir: Path) -> None:
    """Heatmap of response rates by agency × commenter type (weighted)"""
    if len(df) == 0 or "agency" not in df.columns or "category" not in df.columns:
        logger.warning("No data for response heatmap plot")
        return
    
    # Ensure weights exist
    if "weight" not in df.columns:
        df["weight"] = 1.0
    
    # Calculate response rates using DOCUMENT WEIGHTED statistics
    df["responded"] = (df["response_found"] == "yes").astype(int)
    
    # Create pivot table with weighted means
    try:
        # Calculate weighted response rates for each agency × category combination
        heatmap_dict = {}
        agency_totals = {}
        for (agency, category), group in df.groupby(["agency", "category"]):
            total_weight = group["weight_doc"].sum()
            responded_weight = group[group["responded"] == 1]["weight_doc"].sum()
            response_rate = (responded_weight / total_weight * 100) if total_weight > 0 else np.nan
            if agency not in heatmap_dict:
                heatmap_dict[agency] = {}
                agency_totals[agency] = 0
            heatmap_dict[agency][category] = response_rate
            agency_totals[agency] += total_weight
        
        # Convert to DataFrame
        heatmap_data = pd.DataFrame(heatmap_dict).T
        # Ensure all categories are present
        for cat in CATEGORY_ORDER:
            if cat not in heatmap_data.columns:
                heatmap_data[cat] = np.nan
        heatmap_data = heatmap_data[CATEGORY_ORDER]
        
        # Filter to agencies with at least 10 comments total (to avoid sparse data)
        agency_totals_series = pd.Series(agency_totals)
        agencies_with_data = agency_totals_series[agency_totals_series >= 10].index
        heatmap_data = heatmap_data.loc[heatmap_data.index.intersection(agencies_with_data)]
        
    except Exception as e:
        logger.warning(f"Could not create heatmap pivot table: {e}")
        return
    
    if len(heatmap_data) == 0 or len(heatmap_data.columns) == 0:
        logger.warning("No data for heatmap after pivot")
        return
    
    # Sort all agencies by total weighted comments (high volume at top)
    try:
        agency_volumes = df.groupby("agency")["weight_doc"].sum().sort_values(ascending=False)
        # Only keep agencies that are in both
        common_agencies = heatmap_data.index.intersection(agency_volumes.index)
        agency_volumes_filtered = agency_volumes.loc[common_agencies]
        heatmap_data = heatmap_data.loc[agency_volumes_filtered.index]
    except Exception as e:
        logger.warning(f"Could not sort heatmap: {e}")
    
    if len(heatmap_data) == 0:
        logger.warning("No data for heatmap after filtering")
        return
    
    # Calculate dynamic dimensions - better proportions
    n_agencies = len(heatmap_data)
    n_categories = len(heatmap_data.columns)
    
    # More generous spacing: 0.5 inches per agency row, minimum 12 inches height
    fig_height = max(12, n_agencies * 0.5)
    # Width: enough for categories + colorbar + margins
    fig_width = max(14, n_categories * 2.5 + 2)
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Configure colormap to show NaN as grey (0 volume)
    cmap = plt.get_cmap("YlGn").copy()
    cmap.set_bad("#DDE3EA")  # Use "Paper" light grey for zero-volume cells
    
    # Plot heatmap with better colormap range
    im = ax.imshow(heatmap_data.values, cmap=cmap, aspect="auto", vmin=0, vmax=100)
    
    # Apply display name mapping to category labels
    heatmap_data_display = heatmap_data.copy()
    heatmap_data_display.index.name = None
    heatmap_data_display.columns = [CATEGORY_DISPLAY_NAMES.get(col, col) for col in heatmap_data_display.columns]
    
    # Truncate long agency names for better readability
    def truncate_agency_name(name, max_length=50):
        name_str = str(name)
        if len(name_str) > max_length:
            return name_str[:max_length-3] + "..."
        return name_str
    
    truncated_agency_names = [truncate_agency_name(agency) for agency in heatmap_data_display.index]
    
    # Set ticks with better spacing
    ax.set_xticks(range(len(heatmap_data_display.columns)))
    ax.set_yticks(range(len(heatmap_data_display.index)))
    ax.set_xticklabels(heatmap_data_display.columns, rotation=45, ha="right", fontsize=FONT_LABEL - 1)
    ax.set_yticklabels(truncated_agency_names, fontsize=FONT_LABEL - 2)
    
    # Add colorbar with better positioning
    cbar = plt.colorbar(im, ax=ax, pad=0.02, shrink=0.8)
    cbar.set_label("Response Rate (%)", fontsize=FONT_LABEL, fontweight="bold")
    cbar.ax.tick_params(labelsize=FONT_LABEL - 1)
    
    # Add values to cells with dynamic text color for better contrast
    for i in range(len(heatmap_data_display.index)):
        for j in range(len(heatmap_data_display.columns)):
            value = heatmap_data_display.values[i, j]
            if np.isnan(value):
                continue
            # Use dark text on light backgrounds (low values), light text on dark backgrounds (high values)
            # Threshold at 50% for color switching
            text_color = "black" if value < 50 else "white"
            # Add text outline for better readability on all backgrounds
            text = ax.text(
                j, i, f"{value:.0f}",
                ha="center", va="center",
                color=text_color,
                fontsize=FONT_LABEL - 1,
                fontweight="bold",
                path_effects=[withStroke(linewidth=3, foreground="white" if text_color == "black" else "black")]
            )
    
    ax.set_title(
        f"Response Rate Heatmap: Agency × Commenter Type - {year}",
        fontsize=FONT_TITLE,
        fontweight="bold",
        pad=20
    )
    
    # Add grid lines for better readability
    ax.set_xticks(np.arange(len(heatmap_data_display.columns)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(heatmap_data_display.index)) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.5, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(outdir / f"response_heatmap_{year}.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: response_heatmap_{year}.png")


def plot_agency_response_acceptance(df: pd.DataFrame, year: int, outdir: Path) -> None:
    """
    Single scatterplot: Agency-level acceptance rates vs comment volume (weighted).
    
    Shows relationship between agency workload (total weighted comments) and acceptance rate,
    with color encoding response rate bucket.
    """
    if len(df) == 0 or "agency" not in df.columns or "response_found" not in df.columns:
        logger.warning("No data for agency response vs acceptance plot")
        return
    
    # Ensure weights exist
    if "weight" not in df.columns:
        df["weight"] = 1.0
    
    # Calculate response rate by agency using DOCUMENT WEIGHTED statistics
    df["responded"] = (df["response_found"] == "yes").astype(int)
    agency_stats = df.groupby("agency").agg({
        "responded": lambda x: (df.loc[x.index, "responded"] * df.loc[x.index, "weight_doc"]).sum() / df.loc[x.index, "weight_doc"].sum() * 100 if df.loc[x.index, "weight_doc"].sum() > 0 else 0,
        "weight_doc": "sum"  # Total weighted comments per agency
    }).reset_index()
    agency_stats.columns = ["agency", "response_rate", "total_comments"]
    
    # Filter to agencies with responses found AND valid categories
    # Note: For agency-level stats, we still want valid categories to ensure proper weighting
    df_found = df[
        (df["response_found"] == "yes") & 
        (df["category"].notna()) & 
        (df["category"].isin(CATEGORY_ORDER))
    ].copy()
    
    if len(df_found) == 0:
        logger.warning("No responses found with valid categories for acceptance rate calculation")
        return
    
    # Calculate acceptance rate by agency using DOCUMENT WEIGHTED statistics
    acceptance_stats = []
    for agency, agency_data in df_found.groupby("agency"):
        total_weight = agency_data["weight_doc"].sum()
        if total_weight > 0:
            accept_weight = agency_data[agency_data["agency_decision"] == "accept"]["weight_doc"].sum()
            acceptance_rate = (accept_weight / total_weight * 100) if total_weight > 0 else 0
            acceptance_stats.append({
                "agency": agency,
                "acceptance_rate": acceptance_rate
            })
    acceptance_stats = pd.DataFrame(acceptance_stats)
    
    # Merge stats
    agency_stats = agency_stats.merge(acceptance_stats, on="agency", how="left")
    agency_stats["acceptance_rate"] = agency_stats["acceptance_rate"].fillna(0)
    
    # Filter to agencies with at least 10 comments and at least 1 response
    agency_stats = agency_stats[
        (agency_stats["total_comments"] >= 10) & 
        (agency_stats["response_rate"] > 0)
    ]
    
    if len(agency_stats) == 0:
        logger.warning("No agencies with sufficient data for plot")
        return
    
    # Calculate response rate tertiles for coloring
    response_rate_tertiles = agency_stats["response_rate"].quantile([0.33, 0.67])
    low_threshold = response_rate_tertiles.iloc[0]
    high_threshold = response_rate_tertiles.iloc[1]
    
    def get_response_bucket(rate):
        if rate <= low_threshold:
            return "Low"
        elif rate <= high_threshold:
            return "Medium"
        else:
            return "High"
    
    agency_stats["response_bucket"] = agency_stats["response_rate"].apply(get_response_bucket)
    
    # Color mapping for response buckets
    bucket_colors = {
        "Low": "#DDE3EA",
        "Medium": "#9A8153",
        "High": "#6B7D6D"
    }
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot all agencies as scatter points
    for bucket in ["Low", "Medium", "High"]:
        bucket_data = agency_stats[agency_stats["response_bucket"] == bucket]
        if len(bucket_data) > 0:
            ax.scatter(bucket_data["total_comments"], bucket_data["acceptance_rate"],
                      s=80, alpha=0.6, color=bucket_colors[bucket],
                      edgecolors="white", linewidths=1, zorder=3,
                      label=f"{bucket} Response Rate")
    
    # Label top agencies by comment volume (top 10-15)
    top_agencies = agency_stats.nlargest(15, "total_comments")
    
    try:
        from adjustText import adjust_text
        texts = []
        for _, row in top_agencies.iterrows():
            agency_name = str(row["agency"])
            # Shorten agency names
            if "," in agency_name:
                agency_name = agency_name.split(",")[0]
            agency_name = agency_name[:30]
            
            text = ax.text(row["total_comments"], row["acceptance_rate"],
                          agency_name, fontsize=8, alpha=0.8,
                          ha="left", va="bottom",
                          path_effects=[withStroke(linewidth=2, foreground="white")])
            texts.append(text)
        
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.3),
                   expand_points=(1.2, 1.3), force_text=(0.3, 0.5))
    except ImportError:
        # Fallback: simple labels without adjustment
        for _, row in top_agencies.head(10).iterrows():
            agency_name = str(row["agency"])
            if "," in agency_name:
                agency_name = agency_name.split(",")[0]
            agency_name = agency_name[:25]
            ax.annotate(agency_name, 
                       (row["total_comments"], row["acceptance_rate"]),
                       xytext=(5, 5), textcoords="offset points",
                       fontsize=8, alpha=0.7,
                       path_effects=[withStroke(linewidth=2, foreground="white")])
    
    # Add reference lines (weighted mean)
    if len(agency_stats) > 0:
        # Calculate weighted mean acceptance rate
        total_weight_all = agency_stats.merge(df_found.groupby("agency")["weight_doc"].sum().reset_index(), on="agency", how="left")["weight_doc"].sum()
        accept_weight_all = df_found[df_found["agency_decision"] == "accept"]["weight_doc"].sum()
        overall_acceptance = (accept_weight_all / total_weight_all * 100) if total_weight_all > 0 else 0
    else:
        overall_acceptance = 0
    ax.axhline(overall_acceptance, color="#5C6670", linestyle="--", 
              linewidth=1, alpha=0.5, zorder=1, label=f"Overall mean ({overall_acceptance:.1f}%)")
    
    # Styling
    ax.set_xscale("log")
    ax.set_xlabel("Total Comments by Agency (log scale)", fontsize=FONT_LABEL, fontweight="bold")
    ax.set_ylabel("Acceptance Rate (%)", fontsize=FONT_LABEL, fontweight="bold")
    ax.set_title(f"Agency Acceptance Rates by Comment Volume ({year})", 
                fontsize=FONT_TITLE, fontweight="bold")
    ax.set_ylim(-2, 102)
    ax.grid(alpha=GRID_ALPHA, color=GRID_COLOR)
    ax.legend(fontsize=FONT_LEGEND - 1, frameon=True, loc="upper right", framealpha=0.9)
    apply_clean_style(ax)
    
    plt.tight_layout()
    plt.savefig(outdir / f"agency_response_acceptance_{year}.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: agency_response_acceptance_{year}.png")


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
        plot_composition_dumbbell(df, year, outdir, fr_csv_path)
        plot_agency_compass(df, year, outdir)
        plot_workload_vs_makeup(df, year, outdir)
        plot_workload_vs_citizen_by_agency(df, year, outdir)
        if "document_number" in df.columns:
            plot_workload_vs_citizen_by_document(df, year, outdir)
        plot_comment_distribution_roi(df, year, outdir, fr_csv_path)
    
    if generate_continuum:
        log_banner(logger, "COMPLEX CONTINUUM PAGE")
        plot_ranked_stream(df, year, outdir)
        plot_ranked_stream_experiments(df, year, outdir)
        plot_spotlight_strip(df, year, outdir)
        plot_agency_clustering(df, year, outdir)
    
    # Generate response tracking plots if data available
    df_responses = load_response_data(year)
    if df_responses is not None:
        log_banner(logger, "RESPONSE TRACKING PLOTS")
        generate_response_plots(df_responses, df, outdir, year)
    else:
        logger.info("No response tracking data found, skipping response plots")

    # Generate lifecycle-specific plots (Phase 4: items 4I, 4J, 4K)
    log_banner(logger, "LIFECYCLE PLOTS")
    generate_lifecycle_plots(df, fr_csv_path, year, outdir)

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
