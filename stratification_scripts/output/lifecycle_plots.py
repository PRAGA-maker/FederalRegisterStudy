"""
Lifecycle-specific visualizations for the Federal Register Study.

Generates plots that analyze rulemaking lifecycle stages, timeline durations,
commenter types by stage, response rates by stage, and RIN coverage.

These are Phase 4 (items 4I, 4J, 4K) visualizations that depend on lifecycle
columns propagated through the pipeline.

Example:
    >>> from stratification_scripts.output.lifecycle_plots import generate_lifecycle_plots
    >>> generate_lifecycle_plots(df_makeup, fr_csv_path, year, outdir)

# ---------------------------------------------------------------------------
# TODO: Batch 5 Agent E (2026-02-17) added doc_type, eligibility_reason,
# topics, cfr_titles, action, significant to the data pipeline.
# See makeup_plots.py header and plans/eager-waddling-squirrel.md for the
# full list of new analyses now possible. Key addition for lifecycle plots:
# can now split lifecycle heatmaps by doc_type (Proposed Rule vs Notice).
# ---------------------------------------------------------------------------
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from stratification_scripts.config import (
    get_fr_csv_path,
    get_agency_responses_path,
    get_lifecycle_csv_path,
)
from stratification_scripts.logging_utils import get_logger

logger = get_logger(__name__)

# Scope warning suppression to known noisy sources instead of global ignore.
# matplotlib emits FutureWarnings on layout/tick changes; pandas/numpy emit
# DeprecationWarnings during groupby/plot operations.
warnings.filterwarnings("ignore", category=FutureWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
warnings.filterwarnings("ignore", category=FutureWarning, module="numpy")

# ============================================================================
# CONSTANTS — match makeup_plots.py style tokens
# ============================================================================

FONT_TITLE = 18
FONT_LABEL = 13
FONT_LEGEND = 12
GRID_COLOR = "#CCCCCC"
GRID_ALPHA = 0.3

# Lifecycle stage display order — matches reginfo/client.py LIFECYCLE_STAGES
# Ordered by analytical importance: active rulemaking first, then terminal, then missing
LIFECYCLE_STAGE_ORDER = [
    # Pre-rule stages
    "PRERULE_ANPRM",
    "PRERULE_RFI",
    # Proposed rule stages
    "NPRM_ACTIVE",
    "NPRM_CLOSED",
    # Final rule stages
    "INTERIM_FINAL",
    "FINAL_PENDING",
    "FINAL_EFFECTIVE",
    # Terminal stages
    "WITHDRAWN",
    "LONG_TERM_STALLED",
    # Unknown/missing data
    "NO_RIN",
    "AGENDA_NOT_FOUND",
    "UNKNOWN",
]

# Muted but distinct colors per lifecycle stage — all 12 official stages
LIFECYCLE_COLORS = {
    "PRERULE_ANPRM":     "#B07AA1",  # mauve
    "PRERULE_RFI":       "#D4A6C8",  # light mauve
    "NPRM_ACTIVE":       "#F28E2B",  # warm orange
    "NPRM_CLOSED":       "#FFBE7D",  # light orange
    "INTERIM_FINAL":     "#76B7B2",  # teal
    "FINAL_PENDING":     "#E15759",  # muted red
    "FINAL_EFFECTIVE":   "#4E79A7",  # steel blue
    "WITHDRAWN":         "#EDC948",  # gold
    "LONG_TERM_STALLED": "#59A14F",  # green
    "NO_RIN":            "#9C9C9C",  # gray
    "AGENDA_NOT_FOUND":  "#BAB0AC",  # warm gray
    "UNKNOWN":           "#AAAAAA",  # neutral gray
}

# Commenter categories (must match makeup_plots.py CATEGORY_ORDER)
CATEGORY_ORDER = [
    "Undecided/Anonymous",
    "Ordinary Citizen",
    "Large Organization/Corporation",
    "Academic/Industry/Expert (incl. small/local business)",
    "Political Consultant/Lobbyist",
]

CATEGORY_DISPLAY = {
    "Undecided/Anonymous": "Undecided",
    "Ordinary Citizen": "Ordinary Citizen",
    "Large Organization/Corporation": "Organization/Corp",
    "Academic/Industry/Expert (incl. small/local business)": "Expert/Academic",
    "Political Consultant/Lobbyist": "Lobbyist/Consultant",
}

# Category mapping (same as makeup_plots.py)
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


def _apply_clean_style(ax: plt.Axes) -> None:
    """Apply consistent clean styling to axis (matches makeup_plots.py)."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID_COLOR)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.tick_params(labelsize=FONT_LABEL, colors="#5C6670")
    ax.grid(True, alpha=GRID_ALPHA, color=GRID_COLOR, linestyle="-", linewidth=0.8)


def _get_stage_color(stage: str) -> str:
    """Return color for a lifecycle stage, with fallback for unknown stages."""
    return LIFECYCLE_COLORS.get(stage, "#AAAAAA")


def _filter_stages_present(stages: list, data_stages: set) -> list:
    """Return only stages that exist in data, preserving order. Append any unknown stages."""
    ordered = [s for s in stages if s in data_stages]
    extras = sorted(data_stages - set(ordered))
    return ordered + extras


# ============================================================================
# PLOT 1: Comment Volume by Lifecycle Stage
# ============================================================================

def plot_comment_volume_by_stage(
    df: pd.DataFrame,
    year: int,
    outdir: Path,
) -> None:
    """
    Bar chart showing weighted comment counts per lifecycle_stage.

    Uses the weight column (stratum weight) from the makeup data to
    reconstruct population-level comment volumes at each lifecycle stage.

    Args:
        df: Preprocessed makeup DataFrame with 'lifecycle_stage' and 'weight'.
        year: Analysis year.
        outdir: Directory to save the plot.
    """
    if "lifecycle_stage" not in df.columns:
        logger.warning("lifecycle_stage column missing -- skipping comment volume by stage plot")
        return
    if "weight" not in df.columns:
        logger.warning("weight column missing -- skipping comment volume by stage plot")
        return

    # Weighted counts per stage
    stage_counts = df.groupby("lifecycle_stage")["weight"].sum().sort_values(ascending=False)
    stages_present = list(stage_counts.index)
    ordered = _filter_stages_present(LIFECYCLE_STAGE_ORDER, set(stages_present))
    stage_counts = stage_counts.reindex(ordered).dropna()

    if stage_counts.empty:
        logger.warning("No lifecycle stage data for comment volume plot")
        return

    logger.info(
        f"Comment volume by stage: {len(stage_counts)} stages, "
        f"{stage_counts.sum():.0f} total weighted comments"
    )

    fig, ax = plt.subplots(figsize=(12, 7))

    colors = [_get_stage_color(s) for s in stage_counts.index]
    bars = ax.bar(range(len(stage_counts)), stage_counts.values, color=colors, edgecolor="white", linewidth=1.2)

    # Annotate each bar with count and raw N
    raw_counts = df.groupby("lifecycle_stage").size().reindex(stage_counts.index).fillna(0).astype(int)
    for i, (bar_obj, stage) in enumerate(zip(bars, stage_counts.index)):
        weighted = stage_counts[stage]
        raw = raw_counts[stage]
        ax.text(
            bar_obj.get_x() + bar_obj.get_width() / 2,
            bar_obj.get_height() + stage_counts.max() * 0.01,
            f"{weighted:,.0f}\n(n={raw})",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    ax.set_xticks(range(len(stage_counts)))
    ax.set_xticklabels(stage_counts.index, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Weighted Comment Count", fontsize=FONT_LABEL, fontweight="bold")
    ax.set_title(f"Comment Volume by Lifecycle Stage ({year})", fontsize=FONT_TITLE, fontweight="bold", pad=15)
    ax.set_ylim(0, stage_counts.max() * 1.18)

    _apply_clean_style(ax)
    plt.tight_layout()
    fig.savefig(outdir / f"lifecycle_comment_volume_{year}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[OK] Saved: lifecycle_comment_volume_{year}.png")


# ============================================================================
# PLOT 2: Commenter Type by Stage Heatmap
# ============================================================================

def plot_commenter_type_by_stage_heatmap(
    df: pd.DataFrame,
    year: int,
    outdir: Path,
) -> None:
    """
    Heatmap matrix with commenter category on one axis and lifecycle_stage on
    the other.  Color intensity = weighted count.

    Args:
        df: Preprocessed makeup DataFrame with 'lifecycle_stage', 'category', 'weight'.
        year: Analysis year.
        outdir: Directory to save the plot.
    """
    if "lifecycle_stage" not in df.columns or "category" not in df.columns:
        logger.warning("Missing lifecycle_stage or category -- skipping heatmap")
        return
    if "weight" not in df.columns:
        logger.warning("Missing weight column -- skipping heatmap")
        return

    # Normalize categories (same logic as makeup_plots.py)
    df_work = df.copy()
    df_work["category"] = df_work["category"].map(CATEGORY_MAPPING).fillna(df_work["category"])
    df_work = df_work[df_work["category"].isin(CATEGORY_ORDER)]

    # Pivot: rows=category, cols=lifecycle_stage, values=weighted count
    pivot = df_work.groupby(["category", "lifecycle_stage"])["weight"].sum().reset_index()
    pivot_wide = pivot.pivot(index="category", columns="lifecycle_stage", values="weight").fillna(0)

    # Order rows and columns
    stages_present = _filter_stages_present(LIFECYCLE_STAGE_ORDER, set(pivot_wide.columns))
    cats_present = [c for c in CATEGORY_ORDER if c in pivot_wide.index]
    pivot_wide = pivot_wide.reindex(index=cats_present, columns=stages_present).fillna(0)

    if pivot_wide.empty:
        logger.warning("Empty pivot table for heatmap -- skipping")
        return

    logger.info(f"Heatmap: {len(cats_present)} categories x {len(stages_present)} stages")

    # Use short display names for categories
    display_labels = [CATEGORY_DISPLAY.get(c, c) for c in cats_present]

    fig, ax = plt.subplots(figsize=(max(10, len(stages_present) * 1.5), max(5, len(cats_present) * 0.9)))

    # Use a sequential colormap
    cmap = plt.cm.YlOrBr
    im = ax.imshow(pivot_wide.values, cmap=cmap, aspect="auto")

    # Annotate cells with values
    for i in range(len(cats_present)):
        for j in range(len(stages_present)):
            val = pivot_wide.values[i, j]
            # Choose text color based on background intensity
            text_color = "white" if val > pivot_wide.values.max() * 0.6 else "black"
            ax.text(j, i, f"{val:,.0f}", ha="center", va="center", fontsize=10, fontweight="bold", color=text_color)

    ax.set_xticks(range(len(stages_present)))
    ax.set_xticklabels(stages_present, rotation=40, ha="right", fontsize=10)
    ax.set_yticks(range(len(cats_present)))
    ax.set_yticklabels(display_labels, fontsize=11)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Weighted Comment Count", fontsize=FONT_LABEL)

    ax.set_title(
        f"Commenter Type by Lifecycle Stage ({year})",
        fontsize=FONT_TITLE, fontweight="bold", pad=15,
    )

    plt.tight_layout()
    fig.savefig(outdir / f"lifecycle_heatmap_{year}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[OK] Saved: lifecycle_heatmap_{year}.png")


# ============================================================================
# PLOT 3: Response Rate by Lifecycle Stage
# ============================================================================

def plot_response_rate_by_stage(
    df_makeup: pd.DataFrame,
    year: int,
    outdir: Path,
) -> None:
    """
    Bar chart showing % of comments with response_found='yes' at each
    lifecycle_stage.  Joins makeup_data with agency_responses by comment_id.
    Annotates each bar with sample N.

    Args:
        df_makeup: Preprocessed makeup DataFrame (needs comment_id, lifecycle_stage).
        year: Analysis year.
        outdir: Directory to save the plot.
    """
    responses_path = get_agency_responses_path(year)
    if not responses_path.exists():
        logger.info(f"No agency_responses_{year}.csv found -- skipping response rate by stage plot")
        return

    df_resp = pd.read_csv(responses_path)
    logger.info(f"Loaded {len(df_resp):,} response records from {responses_path.name}")

    # Deduplicate responses
    df_resp = df_resp.drop_duplicates(subset=["comment_id"])

    # Merge with makeup to get weights and categories
    merge_cols = ["comment_id"]
    if "lifecycle_stage" in df_makeup.columns:
        # Use lifecycle_stage from makeup (already preprocessed)
        merge_cols.append("lifecycle_stage")
    if "weight" in df_makeup.columns:
        merge_cols.append("weight")

    df = df_resp.merge(df_makeup[merge_cols].drop_duplicates(subset=["comment_id"]), on="comment_id", how="inner")

    # If lifecycle_stage came from both sides, prefer makeup version
    if "lifecycle_stage_x" in df.columns and "lifecycle_stage_y" in df.columns:
        df["lifecycle_stage"] = df["lifecycle_stage_y"].fillna(df["lifecycle_stage_x"])
        df.drop(columns=["lifecycle_stage_x", "lifecycle_stage_y"], inplace=True)

    if "lifecycle_stage" not in df.columns:
        logger.warning("No lifecycle_stage available after merge -- skipping response rate by stage plot")
        return

    if "weight" not in df.columns:
        df["weight"] = 1.0

    # Calculate response rate per stage
    df["responded"] = (df["response_found"] == "yes").astype(int)

    stage_stats = []
    for stage, group in df.groupby("lifecycle_stage"):
        n = len(group)
        total_w = group["weight"].sum()
        responded_w = group.loc[group["responded"] == 1, "weight"].sum()
        rate = (responded_w / total_w * 100) if total_w > 0 else 0
        stage_stats.append({"stage": stage, "response_rate": rate, "n": n})

    stage_df = pd.DataFrame(stage_stats)
    if stage_df.empty:
        logger.warning("No data for response rate by stage")
        return

    # Order by LIFECYCLE_STAGE_ORDER
    ordered_stages = _filter_stages_present(LIFECYCLE_STAGE_ORDER, set(stage_df["stage"]))
    stage_df["sort_key"] = stage_df["stage"].apply(lambda s: ordered_stages.index(s) if s in ordered_stages else 999)
    stage_df = stage_df.sort_values("sort_key").reset_index(drop=True)

    logger.info(f"Response rate by stage: {len(stage_df)} stages, {df['responded'].sum()} responses out of {len(df)} comments")

    fig, ax = plt.subplots(figsize=(12, 7))

    colors = [_get_stage_color(s) for s in stage_df["stage"]]
    bars = ax.bar(range(len(stage_df)), stage_df["response_rate"], color=colors, edgecolor="white", linewidth=1.2)

    # Annotate bars
    for i, (bar_obj, row) in enumerate(zip(bars, stage_df.itertuples())):
        ax.text(
            bar_obj.get_x() + bar_obj.get_width() / 2,
            bar_obj.get_height() + 1.5,
            f"{row.response_rate:.1f}%\n(n={row.n})",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    ax.set_xticks(range(len(stage_df)))
    ax.set_xticklabels(stage_df["stage"], rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Response Rate (%)", fontsize=FONT_LABEL, fontweight="bold")
    ax.set_ylim(0, min(stage_df["response_rate"].max() * 1.25, 110))
    ax.set_title(f"Agency Response Rate by Lifecycle Stage ({year})", fontsize=FONT_TITLE, fontweight="bold", pad=15)

    _apply_clean_style(ax)
    plt.tight_layout()
    fig.savefig(outdir / f"lifecycle_response_rate_{year}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[OK] Saved: lifecycle_response_rate_{year}.png")


# ============================================================================
# PLOT 4: NPRM-to-Final Duration Distribution
# ============================================================================

def plot_nprm_to_final_duration(
    fr_csv_path: Path,
    year: int,
    outdir: Path,
) -> None:
    """
    Histogram of nprm_to_final_days from the FR CSV (document-level metric).
    Annotates mean/median and flags outliers (> 3 standard deviations).

    Args:
        fr_csv_path: Path to the Federal Register CSV.
        year: Analysis year.
        outdir: Directory to save the plot.
    """
    if not fr_csv_path.exists():
        logger.warning(f"FR CSV not found at {fr_csv_path} -- skipping NPRM-to-Final duration plot")
        return

    fr_df = pd.read_csv(fr_csv_path)
    if "nprm_to_final_days" not in fr_df.columns:
        logger.warning("nprm_to_final_days column missing from FR CSV -- skipping duration plot")
        return

    durations = fr_df["nprm_to_final_days"].dropna()
    if len(durations) == 0:
        logger.info("No NPRM-to-Final durations available (0 non-null values) -- skipping duration plot")
        return

    logger.info(f"NPRM-to-Final durations: {len(durations)} documents, range [{durations.min():.0f}, {durations.max():.0f}] days")

    mean_val = durations.mean()
    median_val = durations.median()
    std_val = durations.std() if len(durations) > 1 else 0

    # Flag outliers (> 3 SD above mean)
    outlier_threshold = mean_val + 3 * std_val if std_val > 0 else durations.max() + 1
    n_outliers = (durations > outlier_threshold).sum()

    fig, ax = plt.subplots(figsize=(12, 7))

    # Choose binning based on data range
    dur_range = durations.max() - durations.min()
    if dur_range <= 0:
        n_bins = 5
    else:
        n_bins = min(40, max(10, int(np.sqrt(len(durations)) * 2)))

    ax.hist(durations.values, bins=n_bins, color="#4E79A7", alpha=0.8, edgecolor="white", linewidth=1)

    # Draw mean and median lines
    ax.axvline(mean_val, color="#E15759", linestyle="--", linewidth=2, label=f"Mean: {mean_val:,.0f} days")
    ax.axvline(median_val, color="#F28E2B", linestyle="-", linewidth=2, label=f"Median: {median_val:,.0f} days")

    # Annotation box
    stats_text = (
        f"N = {len(durations)}\n"
        f"Mean = {mean_val:,.0f} days ({mean_val/365:.1f} yr)\n"
        f"Median = {median_val:,.0f} days ({median_val/365:.1f} yr)\n"
        f"Std = {std_val:,.0f} days"
    )
    if n_outliers > 0:
        stats_text += f"\nOutliers (>3 SD): {n_outliers}"

    ax.text(
        0.97, 0.95, stats_text,
        transform=ax.transAxes, fontsize=10,
        va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor=GRID_COLOR),
    )

    ax.set_xlabel("NPRM to Final Action (days)", fontsize=FONT_LABEL, fontweight="bold")
    ax.set_ylabel("Number of Documents", fontsize=FONT_LABEL, fontweight="bold")
    ax.set_title(f"NPRM-to-Final Duration Distribution ({year})", fontsize=FONT_TITLE, fontweight="bold", pad=15)
    ax.legend(fontsize=FONT_LEGEND, frameon=True, loc="upper right", bbox_to_anchor=(0.97, 0.72))

    _apply_clean_style(ax)
    plt.tight_layout()
    fig.savefig(outdir / f"lifecycle_nprm_to_final_{year}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[OK] Saved: lifecycle_nprm_to_final_{year}.png")


# ============================================================================
# PLOT 5: RIN Coverage / Timeline Confidence Summary
# ============================================================================

def plot_rin_coverage(
    fr_csv_path: Path,
    year: int,
    outdir: Path,
) -> None:
    """
    Donut chart showing timeline_confidence distribution (full/partial/
    label_only/none) from the FR CSV.

    Args:
        fr_csv_path: Path to the Federal Register CSV.
        year: Analysis year.
        outdir: Directory to save the plot.
    """
    if not fr_csv_path.exists():
        logger.warning(f"FR CSV not found at {fr_csv_path} -- skipping RIN coverage plot")
        return

    fr_df = pd.read_csv(fr_csv_path)
    if "timeline_confidence" not in fr_df.columns:
        logger.warning("timeline_confidence column missing from FR CSV -- skipping RIN coverage plot")
        return

    # Fill missing as 'none'
    confidence = fr_df["timeline_confidence"].fillna("none")
    counts = confidence.value_counts()

    # Define display order and colors
    conf_order = ["full", "partial", "label_only", "none"]
    conf_colors = {
        "full": "#4E79A7",
        "partial": "#F28E2B",
        "label_only": "#EDC948",
        "none": "#9C9C9C",
    }

    # Keep only present categories in order, then append any extras
    present = [c for c in conf_order if c in counts.index]
    extras = sorted(set(counts.index) - set(conf_order))
    ordered = present + extras

    counts = counts.reindex(ordered).dropna()
    colors = [conf_colors.get(c, "#AAAAAA") for c in counts.index]

    if counts.empty:
        logger.warning("No timeline_confidence data for RIN coverage plot")
        return

    logger.info(f"RIN coverage: {dict(counts)}")

    fig, ax = plt.subplots(figsize=(10, 8))

    total = counts.sum()
    labels_with_pct = [f"{name}\n{count:,} ({count/total*100:.1f}%)" for name, count in counts.items()]

    wedges, texts = ax.pie(
        counts.values,
        colors=colors,
        startangle=90,
        wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2),
    )

    # Center text
    ax.text(0, 0, f"N={total}", ha="center", va="center", fontsize=16, fontweight="bold", color="#333333")

    ax.legend(
        labels_with_pct,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        fontsize=FONT_LEGEND,
        frameon=False,
        title="Timeline Confidence",
        title_fontsize=FONT_LEGEND + 1,
    )

    ax.set_title(
        f"RIN Coverage Summary ({year})",
        fontsize=FONT_TITLE, fontweight="bold", pad=20,
    )

    plt.tight_layout()
    fig.savefig(outdir / f"lifecycle_rin_coverage_{year}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[OK] Saved: lifecycle_rin_coverage_{year}.png")


# ============================================================================
# ITEM 4J: Lifecycle Summary CSV
# ============================================================================

def generate_lifecycle_summary_csv(
    fr_csv_path: Path,
    year: int,
    outdir: Path,
) -> Optional[Path]:
    """
    Generate one-row-per-RIN lifecycle summary CSV.

    Columns: rin, lifecycle_stage, timetable_action_count, nprm_date,
    final_action_date, nprm_to_final_days, nprm_document_number,
    final_rule_document_number, num_fr_documents, timeline_confidence,
    timetable_sequence.

    Reads FR CSV directly; does not depend on makeup_data.
    Saves to the output directory alongside existing lifecycle data.

    Args:
        fr_csv_path: Path to the Federal Register CSV.
        year: Analysis year.
        outdir: Directory to save the CSV.

    Returns:
        Path to the saved CSV, or None if generation failed.
    """
    if not fr_csv_path.exists():
        logger.warning(f"FR CSV not found at {fr_csv_path} -- skipping lifecycle summary CSV")
        return None

    fr_df = pd.read_csv(fr_csv_path)

    if "rin" not in fr_df.columns:
        logger.warning("rin column missing from FR CSV -- skipping lifecycle summary CSV")
        return None

    # Work only with docs that have a RIN
    rin_df = fr_df[fr_df["rin"].notna() & (fr_df["rin"] != "")].copy()

    if len(rin_df) == 0:
        logger.info("No RIN-bearing documents found -- skipping lifecycle summary CSV")
        return None

    logger.info(f"Building lifecycle summary for {rin_df['rin'].nunique()} unique RINs from {len(rin_df)} documents")

    # Aggregate per RIN
    # For columns where docs sharing a RIN should have the same value, take first.
    # For num_fr_documents, count rows per RIN.
    agg_dict = {}

    # Required columns — use first non-null
    for col in [
        "lifecycle_stage", "timeline_confidence", "timetable_sequence",
        "nprm_date", "final_action_date", "nprm_to_final_days",
        "nprm_document_number", "final_rule_document_number",
    ]:
        if col in rin_df.columns:
            agg_dict[col] = "first"

    # timetable_action_count: take max (most complete timetable)
    if "timetable_action_count" in rin_df.columns:
        agg_dict["timetable_action_count"] = "max"

    # document_number: count unique
    agg_dict["document_number"] = "nunique"

    summary = rin_df.groupby("rin").agg(agg_dict).reset_index()
    summary.rename(columns={"document_number": "num_fr_documents"}, inplace=True)

    # Define output column order
    output_cols = [
        "rin", "lifecycle_stage", "timetable_action_count",
        "nprm_date", "final_action_date", "nprm_to_final_days",
        "nprm_document_number", "final_rule_document_number",
        "num_fr_documents", "timeline_confidence", "timetable_sequence",
    ]
    # Only include columns that exist
    output_cols = [c for c in output_cols if c in summary.columns]
    summary = summary[output_cols]

    # Save to the lifecycle CSV path (config-managed) AND to the plots dir
    lifecycle_path = get_lifecycle_csv_path(year)
    summary.to_csv(lifecycle_path, index=False)
    logger.info(f"[OK] Saved lifecycle summary CSV: {lifecycle_path} ({len(summary)} RINs)")

    # Also save a copy alongside plots for convenience
    plots_copy = outdir / f"rin_lifecycle_{year}.csv"
    summary.to_csv(plots_copy, index=False)
    logger.info(f"[OK] Copy saved: {plots_copy}")

    return lifecycle_path


# ============================================================================
# PLOT 6: Citizen Participation vs Withdrawal/Stall Rate
# ============================================================================

def plot_citizen_vs_withdrawal(
    df: pd.DataFrame,
    year: int,
    outdir: Path,
) -> None:
    """
    Grouped bar chart showing how citizen comment share correlates with
    documents ending up WITHDRAWN or LONG_TERM_STALLED.

    Documents are binned by their citizen commenter share (% of sampled
    comments classified as Ordinary Citizen). For each bin, we show the
    % of documents that reached a terminal-negative lifecycle stage
    (WITHDRAWN or LONG_TERM_STALLED) vs completing normally (FINAL_EFFECTIVE).

    Only RIN-bearing documents with known lifecycle outcomes are included
    (NO_RIN and AGENDA_NOT_FOUND are excluded since they lack lifecycle data).

    Args:
        df: Preprocessed makeup DataFrame with lifecycle_stage, category,
            document_number, and weight columns.
        year: Analysis year.
        outdir: Directory to save the plot.
    """
    required = {"lifecycle_stage", "category", "document_number"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        logger.warning(f"Missing columns for citizen vs withdrawal plot: {missing}")
        return

    # Filter to documents with meaningful lifecycle outcomes
    outcome_stages = {"FINAL_EFFECTIVE", "WITHDRAWN", "LONG_TERM_STALLED",
                      "NPRM_CLOSED", "INTERIM_FINAL", "FINAL_PENDING"}
    df_outcome = df[df["lifecycle_stage"].isin(outcome_stages)].copy()

    if len(df_outcome) == 0:
        logger.warning("No documents with lifecycle outcomes -- skipping citizen vs withdrawal plot")
        return

    # Compute per-document citizen share (unweighted count ratio within sample)
    doc_total = df_outcome.groupby("document_number").size().rename("n_total")
    doc_citizen = (
        df_outcome[df_outcome["category"] == "Ordinary Citizen"]
        .groupby("document_number").size().rename("n_citizen")
    )
    doc_stage = df_outcome.groupby("document_number")["lifecycle_stage"].first()

    doc_stats = pd.DataFrame({"n_total": doc_total, "n_citizen": doc_citizen,
                               "lifecycle_stage": doc_stage}).fillna({"n_citizen": 0})
    doc_stats["citizen_pct"] = doc_stats["n_citizen"] / doc_stats["n_total"] * 100
    doc_stats["is_terminal_negative"] = doc_stats["lifecycle_stage"].isin(
        {"WITHDRAWN", "LONG_TERM_STALLED"}
    )

    n_docs = len(doc_stats)
    n_terminal = doc_stats["is_terminal_negative"].sum()
    if n_terminal == 0:
        logger.info("No WITHDRAWN/LONG_TERM_STALLED documents in outcome set -- skipping plot")
        return

    # Bin documents by citizen share
    bins = [0, 1, 25, 50, 75, 100.01]
    labels = ["0%\n(no citizens)", "1-25%", "25-50%", "50-75%", "75-100%"]
    doc_stats["citizen_bin"] = pd.cut(doc_stats["citizen_pct"], bins=bins,
                                       labels=labels, right=False)

    # Compute rates per bin
    bin_stats = doc_stats.groupby("citizen_bin", observed=False).agg(
        n_docs=("is_terminal_negative", "size"),
        n_withdrawn=("is_terminal_negative", lambda x: ((doc_stats.loc[x.index, "lifecycle_stage"] == "WITHDRAWN").sum())),
        n_stalled=("is_terminal_negative", lambda x: ((doc_stats.loc[x.index, "lifecycle_stage"] == "LONG_TERM_STALLED").sum())),
        n_terminal=("is_terminal_negative", "sum"),
    )
    bin_stats["withdrawn_pct"] = bin_stats["n_withdrawn"] / bin_stats["n_docs"] * 100
    bin_stats["stalled_pct"] = bin_stats["n_stalled"] / bin_stats["n_docs"] * 100

    # Drop empty bins
    bin_stats = bin_stats[bin_stats["n_docs"] > 0]

    if len(bin_stats) < 2:
        logger.warning("Not enough populated bins for citizen vs withdrawal plot")
        return

    logger.info(
        f"Citizen vs withdrawal: {n_docs} docs with outcomes, "
        f"{n_terminal} terminal-negative ({n_terminal/n_docs*100:.1f}%)"
    )

    # Plot grouped bars
    fig, ax = plt.subplots(figsize=(10, 7))

    x = np.arange(len(bin_stats))
    width = 0.35

    bars_w = ax.bar(x - width / 2, bin_stats["withdrawn_pct"], width,
                    color=LIFECYCLE_COLORS["WITHDRAWN"], edgecolor="white",
                    linewidth=1.2, label="Withdrawn")
    bars_s = ax.bar(x + width / 2, bin_stats["stalled_pct"], width,
                    color=LIFECYCLE_COLORS["LONG_TERM_STALLED"], edgecolor="white",
                    linewidth=1.2, label="Long-Term Stalled")

    # Annotate bars with counts
    for bars, col in [(bars_w, "n_withdrawn"), (bars_s, "n_stalled")]:
        for bar_obj, (_, row) in zip(bars, bin_stats.iterrows()):
            count = int(row[col])
            if count > 0:
                ax.text(
                    bar_obj.get_x() + bar_obj.get_width() / 2,
                    bar_obj.get_height() + 0.5,
                    f"n={count}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold",
                )

    # Add doc count below each group
    for i, (_, row) in enumerate(bin_stats.iterrows()):
        ax.text(
            i, -2.5,
            f"({int(row['n_docs'])} docs)",
            ha="center", va="top", fontsize=8, color="#888888",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(bin_stats.index, fontsize=11)
    ax.set_xlabel("Citizen Share of Commenters per Document", fontsize=FONT_LABEL, fontweight="bold")
    ax.set_ylabel("% of Documents", fontsize=FONT_LABEL, fontweight="bold")
    ax.set_title(
        f"Citizen Participation vs Rule Withdrawal & Stall ({year})",
        fontsize=FONT_TITLE, fontweight="bold", pad=15,
    )

    y_max = max(bin_stats["withdrawn_pct"].max(), bin_stats["stalled_pct"].max())
    ax.set_ylim(-4, max(y_max * 1.4, 10))

    ax.legend(fontsize=FONT_LEGEND, loc="upper left", framealpha=0.9)
    _apply_clean_style(ax)
    plt.tight_layout()
    fig.savefig(outdir / f"citizen_vs_withdrawal_{year}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[OK] Saved: citizen_vs_withdrawal_{year}.png")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def generate_lifecycle_plots(
    df_makeup: pd.DataFrame,
    fr_csv_path: Optional[Path],
    year: int,
    outdir: Path,
) -> None:
    """
    Generate all lifecycle-specific visualizations.

    This is called from generate_plots_for_year() in makeup_plots.py.

    Args:
        df_makeup: Preprocessed makeup DataFrame (deduplicated, weighted, categories normalized).
        fr_csv_path: Path to FR CSV (may be None if not available).
        year: Analysis year.
        outdir: Directory to save plots.
    """
    logger.info(f"=== LIFECYCLE PLOTS ({year}) ===")

    # Validate data availability
    has_lifecycle = "lifecycle_stage" in df_makeup.columns
    has_weight = "weight" in df_makeup.columns
    has_fr = fr_csv_path is not None and fr_csv_path.exists()

    if not has_lifecycle:
        logger.warning(
            "lifecycle_stage column not in makeup data. "
            "Lifecycle plots require pipeline Steps 1-4 with lifecycle enrichment. Skipping."
        )
        return

    if not has_weight:
        logger.warning("weight column not in makeup data. Lifecycle plots will use unweighted counts.")
        df_makeup = df_makeup.copy()
        df_makeup["weight"] = 1.0

    # Log data summary for transparency
    stage_dist = df_makeup["lifecycle_stage"].value_counts()
    logger.info(f"Lifecycle stage distribution in makeup data:")
    for stage, count in stage_dist.items():
        logger.info(f"  {stage}: {count} comments")

    # --- Plot 1: Comment Volume by Lifecycle Stage ---
    plot_comment_volume_by_stage(df_makeup, year, outdir)

    # --- Plot 2: Commenter Type by Stage Heatmap ---
    plot_commenter_type_by_stage_heatmap(df_makeup, year, outdir)

    # --- Plot 3: Response Rate by Lifecycle Stage ---
    plot_response_rate_by_stage(df_makeup, year, outdir)

    # --- Plot 4: NPRM-to-Final Duration Distribution ---
    if has_fr:
        plot_nprm_to_final_duration(fr_csv_path, year, outdir)
    else:
        logger.info("No FR CSV available -- skipping NPRM-to-Final duration plot")

    # --- Plot 5: RIN Coverage Summary ---
    if has_fr:
        plot_rin_coverage(fr_csv_path, year, outdir)
    else:
        logger.info("No FR CSV available -- skipping RIN coverage plot")

    # --- Plot 6: Citizen Participation vs Withdrawal/Stall ---
    if "document_number" in df_makeup.columns:
        plot_citizen_vs_withdrawal(df_makeup, year, outdir)
    else:
        logger.info("document_number not in makeup data -- skipping citizen vs withdrawal plot")

    # --- Item 4J: Lifecycle Summary CSV ---
    if has_fr:
        generate_lifecycle_summary_csv(fr_csv_path, year, outdir)
    else:
        logger.info("No FR CSV available -- skipping lifecycle summary CSV")

    logger.info(f"=== LIFECYCLE PLOTS COMPLETE ({year}) ===")
