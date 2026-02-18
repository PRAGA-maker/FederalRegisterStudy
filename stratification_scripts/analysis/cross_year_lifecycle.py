"""
Cross-year lifecycle analysis for the Federal Register Study.

Joins documents across years by RIN to track how rules progress from
NPRM to Final Rule across calendar years, and how commenter composition
changes at each stage.

Runs AFTER the pipeline — reads FR CSVs and makeup_data CSVs produced
by Steps 1-6. Does NOT modify pipeline code or business logic.

Feasibility (2014 VERIFIED, 10,406 docs):
  - 2,032 docs with RIN (19.5%)
  - 843 with both nprm_document_number AND final_rule_document_number
  - 183 same-year pairs (21.7%), 660 cross-year (78.3%)
  - Median NPRM-to-Final: 375 days (most pairs span 2+ calendar years)
  - Top cross-year pattern: 2014->2015 (321 docs)

Usage:
    python -m stratification_scripts.analysis.cross_year_lifecycle --years 2014 --verbose
    python -m stratification_scripts.analysis.cross_year_lifecycle --years 2013,2014
    python -m stratification_scripts.analysis.cross_year_lifecycle --years 2013-2015
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from stratification_scripts.config import (
    get_fr_csv_path,
    get_makeup_data_path,
    get_data_dir,
    get_output_dir,
)
from stratification_scripts.logging_utils import (
    get_logger,
    setup_logging,
    log_banner,
)

logger = get_logger(__name__)

# Suppress matplotlib/pandas FutureWarnings (match lifecycle_plots.py)
warnings.filterwarnings("ignore", category=FutureWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

# ============================================================================
# CONSTANTS
# ============================================================================

FONT_TITLE = 18
FONT_LABEL = 13
FONT_LEGEND = 12
GRID_COLOR = "#CCCCCC"
GRID_ALPHA = 0.3

CATEGORY_ORDER = [
    "Undecided/Anonymous",
    "Ordinary Citizen",
    "Large Organization/Corporation",
    "Academic/Industry/Expert (incl. small/local business)",
    "Political Consultant/Lobbyist",
]

CATEGORY_SHORT = {
    "Undecided/Anonymous": "Undecided",
    "Ordinary Citizen": "Citizen",
    "Large Organization/Corporation": "Organization",
    "Academic/Industry/Expert (incl. small/local business)": "Expert",
    "Political Consultant/Lobbyist": "Lobbyist",
}

LIFECYCLE_COLORS = {
    "FINAL_EFFECTIVE": "#4E79A7",
    "WITHDRAWN": "#EDC948",
    "LONG_TERM_STALLED": "#59A14F",
    "NPRM_ACTIVE": "#F28E2B",
    "NPRM_CLOSED": "#FFBE7D",
    "NO_RIN": "#9C9C9C",
    "AGENDA_NOT_FOUND": "#BAB0AC",
    "INTERIM_FINAL": "#76B7B2",
}

# FR CSV required columns for cross-year analysis
REQUIRED_FR_COLS = [
    "document_number", "agency", "doc_type", "rin",
    "lifecycle_stage", "nprm_document_number", "final_rule_document_number",
]


def apply_clean_style(ax: plt.Axes) -> None:
    """Apply consistent clean styling (match lifecycle_plots.py)."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID_COLOR)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.tick_params(labelsize=FONT_LABEL, colors="#5C6670")
    ax.grid(True, alpha=GRID_ALPHA, color=GRID_COLOR, linestyle="-", linewidth=0.8)


# ============================================================================
# DATA LOADING
# ============================================================================


def parse_years(years_str: str) -> List[int]:
    """Parse year string: '2013,2014' or '2013-2016' or mixed.

    >>> parse_years("2013,2014")
    [2013, 2014]
    >>> parse_years("2013-2015")
    [2013, 2014, 2015]
    """
    years: set[int] = set()
    for part in years_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            years.update(range(int(start), int(end) + 1))
        else:
            years.add(int(part))
    return sorted(years)


def _extract_year_from_doc_number(doc_num: str) -> Optional[int]:
    """Extract publication year from FR document number (format: YYYY-NNNNN).

    >>> _extract_year_from_doc_number("2014-00409")
    2014
    >>> _extract_year_from_doc_number("2015-03733")
    2015
    """
    if not isinstance(doc_num, str) or not doc_num.strip():
        return None
    match = re.match(r"^(\d{4})-", doc_num.strip())
    if match:
        return int(match.group(1))
    return None


def load_multi_year_fr(years: List[int]) -> pd.DataFrame:
    """Load and concatenate FR CSVs for multiple years.

    Adds a 'source_year' column. Warns and skips years where file is missing.
    Also tries _VERIFIED.csv suffix for 2014.
    """
    frames: List[pd.DataFrame] = []

    for year in years:
        path = get_fr_csv_path(year)

        # For 2014, prefer the VERIFIED file
        verified_path = path.parent / f"federal_register_{year}_comments_VERIFIED.csv"
        if verified_path.exists():
            path = verified_path
            logger.info(f"Using VERIFIED CSV for {year}: {path.name}")

        if not path.exists():
            logger.warning(f"FR CSV not found for {year}: {path} -- skipping")
            continue

        df = pd.read_csv(path, low_memory=False)

        # Validate required columns
        missing = [c for c in REQUIRED_FR_COLS if c not in df.columns]
        if missing:
            logger.error(f"{year}: FR CSV missing columns: {missing} -- skipping")
            continue

        df["source_year"] = year
        logger.info(f"Loaded {year}: {len(df):,} docs, "
                     f"{df['rin'].notna().sum():,} with RIN")
        frames.append(df)

    if not frames:
        logger.error("No FR CSVs loaded. Cannot proceed.")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    logger.info(f"Combined: {len(combined):,} docs across {len(frames)} year(s)")
    return combined


def load_multi_year_makeup(years: List[int]) -> Optional[pd.DataFrame]:
    """Load and concatenate makeup_data CSVs for multiple years.

    Returns None if no makeup_data is available for any requested year.
    """
    frames: List[pd.DataFrame] = []

    for year in years:
        path = get_makeup_data_path(year)
        if not path.exists():
            logger.info(f"No makeup_data for {year}: {path} -- skipping")
            continue

        df = pd.read_csv(path, low_memory=False)
        required = ["document_number", "comment_id", "category"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            logger.warning(f"{year}: makeup_data missing columns: {missing} -- skipping")
            continue

        df["source_year"] = year
        logger.info(f"Loaded makeup_data {year}: {len(df):,} classified comments")
        frames.append(df)

    if not frames:
        logger.info("No makeup_data available for any requested year.")
        return None

    combined = pd.concat(frames, ignore_index=True)
    logger.info(f"Combined makeup_data: {len(combined):,} comments "
                f"across {len(frames)} year(s)")
    return combined


# ============================================================================
# RIN REGISTRY
# ============================================================================


def build_rin_registry(fr_df: pd.DataFrame) -> pd.DataFrame:
    """Build a registry of RINs with their NPRM and Final Rule documents.

    For each RIN, identifies:
    - NPRM doc(s): doc_type == "Proposed Rule"
    - Final Rule doc(s): doc_type == "Rule"
    - Which years each appears in
    - Whether the pair is "resolvable" (both NPRM and Final in loaded data)

    Returns DataFrame with one row per unique RIN.
    """
    # Filter to RIN-bearing docs only
    rin_df = fr_df[fr_df["rin"].notna() & (fr_df["rin"] != "")].copy()

    if rin_df.empty:
        logger.warning("No RIN-bearing documents found.")
        return pd.DataFrame()

    logger.info(f"Building RIN registry from {len(rin_df):,} RIN-bearing docs "
                f"({rin_df['rin'].nunique():,} unique RINs)")

    # Global set of all years present in the loaded data
    global_loaded_years = set(fr_df["source_year"].unique())

    rows = []
    for rin, group in rin_df.groupby("rin"):
        # Identify NPRM and Final Rule docs
        nprm_docs = group[group["doc_type"] == "Proposed Rule"]
        final_docs = group[group["doc_type"] == "Rule"]
        all_years = sorted(group["source_year"].unique())

        # Pick canonical NPRM doc (highest timetable_action_count, or first)
        nprm_doc = None
        if not nprm_docs.empty:
            if "timetable_action_count" in nprm_docs.columns:
                nprm_doc = nprm_docs.sort_values(
                    "timetable_action_count", ascending=False
                ).iloc[0]
            else:
                nprm_doc = nprm_docs.iloc[0]

        # Pick canonical Final Rule doc
        final_doc = None
        if not final_docs.empty:
            if "timetable_action_count" in final_docs.columns:
                final_doc = final_docs.sort_values(
                    "timetable_action_count", ascending=False
                ).iloc[0]
            else:
                final_doc = final_docs.iloc[0]

        # Get the primary record (any doc for this RIN, prefer one with most info)
        primary = group.sort_values(
            "timetable_action_count" if "timetable_action_count" in group.columns
            else "document_number",
            ascending=False
        ).iloc[0]

        # Extract linked document years from doc number references
        nprm_doc_num = primary.get("nprm_document_number", "")
        final_doc_num = primary.get("final_rule_document_number", "")
        nprm_linked_year = _extract_year_from_doc_number(
            str(nprm_doc_num) if pd.notna(nprm_doc_num) else ""
        )
        final_linked_year = _extract_year_from_doc_number(
            str(final_doc_num) if pd.notna(final_doc_num) else ""
        )

        # Determine if resolvable: have actual docs (not just references) for both stages
        has_nprm_in_data = not nprm_docs.empty
        has_final_in_data = not final_docs.empty
        resolvable = has_nprm_in_data and has_final_in_data

        # Determine outcome from lifecycle_stage
        stage = primary.get("lifecycle_stage", "UNKNOWN")
        if stage == "FINAL_EFFECTIVE":
            outcome = "finalized"
        elif stage == "WITHDRAWN":
            outcome = "withdrawn"
        elif stage in ("LONG_TERM_STALLED", "NPRM_ACTIVE", "NPRM_CLOSED"):
            outcome = "stalled"
        else:
            outcome = "unknown"

        # Safe numeric extraction
        nprm_to_final = primary.get("nprm_to_final_days", np.nan)
        if pd.notna(nprm_to_final) and float(nprm_to_final) <= 0:
            nprm_to_final = np.nan  # Skip negative/zero durations

        action_count = primary.get("timetable_action_count", 0)
        comment_count = primary.get("comment_count", np.nan)

        # NPRM and Final comment counts (from actual docs in data if available)
        nprm_comment_count = (
            float(nprm_doc["comment_count"])
            if nprm_doc is not None and pd.notna(nprm_doc.get("comment_count"))
            else np.nan
        )
        final_comment_count = (
            float(final_doc["comment_count"])
            if final_doc is not None and pd.notna(final_doc.get("comment_count"))
            else np.nan
        )

        # Determine the years we need but don't have (check global loaded data)
        missing_years = set()
        if nprm_linked_year and nprm_linked_year not in global_loaded_years:
            missing_years.add(nprm_linked_year)
        if final_linked_year and final_linked_year not in global_loaded_years:
            missing_years.add(final_linked_year)

        rows.append({
            "rin": rin,
            "agency": primary["agency"],
            "doc_count_in_data": len(group),
            "years_in_data": ",".join(str(y) for y in all_years),
            "nprm_year": int(nprm_doc["source_year"]) if nprm_doc is not None else (
                nprm_linked_year
            ),
            "final_year": int(final_doc["source_year"]) if final_doc is not None else (
                final_linked_year
            ),
            "nprm_document_number": (
                nprm_doc["document_number"] if nprm_doc is not None else (
                    str(nprm_doc_num) if pd.notna(nprm_doc_num) and nprm_doc_num else ""
                )
            ),
            "final_rule_document_number": (
                final_doc["document_number"] if final_doc is not None else (
                    str(final_doc_num) if pd.notna(final_doc_num) and final_doc_num else ""
                )
            ),
            "nprm_to_final_days": nprm_to_final,
            "lifecycle_stage": stage,
            "timetable_action_count": action_count,
            "nprm_comment_count": nprm_comment_count,
            "final_comment_count": final_comment_count,
            "outcome": outcome,
            "resolvable": resolvable,
            "has_nprm_in_data": has_nprm_in_data,
            "has_final_in_data": has_final_in_data,
            "nprm_linked_year": nprm_linked_year,
            "final_linked_year": final_linked_year,
            "missing_years": ",".join(str(y) for y in sorted(missing_years)) if missing_years else "",
        })

    registry = pd.DataFrame(rows)
    resolvable_count = registry["resolvable"].sum()
    logger.info(f"RIN registry: {len(registry):,} unique RINs, "
                f"{resolvable_count:,} resolvable ({resolvable_count/len(registry)*100:.1f}%)")
    return registry


# ============================================================================
# COMMENTER COMPOSITION
# ============================================================================


def compare_commenter_composition(
    registry: pd.DataFrame,
    makeup_df: Optional[pd.DataFrame],
) -> Optional[pd.DataFrame]:
    """Compare commenter composition between NPRM and Final Rule stages.

    For each resolvable RIN where makeup_data exists for both NPRM and Final docs,
    computes category distribution at each stage.

    Returns None if makeup_data unavailable.
    """
    if makeup_df is None:
        logger.info("No makeup_data available -- skipping composition comparison.")
        return None

    resolvable = registry[registry["resolvable"]].copy()
    if resolvable.empty:
        logger.info("No resolvable RINs -- skipping composition comparison.")
        return None

    # Get document numbers for NPRM and Final from resolvable RINs
    nprm_docs = set(resolvable["nprm_document_number"].dropna())
    final_docs = set(resolvable["final_rule_document_number"].dropna())

    # Filter makeup_data to those documents
    nprm_makeup = makeup_df[makeup_df["document_number"].isin(nprm_docs)]
    final_makeup = makeup_df[makeup_df["document_number"].isin(final_docs)]

    if nprm_makeup.empty and final_makeup.empty:
        logger.info("No makeup_data for any resolvable RIN documents -- "
                     "skipping composition comparison.")
        return None

    logger.info(f"Composition data: {len(nprm_makeup):,} NPRM comments, "
                f"{len(final_makeup):,} Final comments")

    # Compute category percentages per document
    def _category_pct(df: pd.DataFrame, doc_col: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        cats = df.groupby([doc_col, "category"]).size().unstack(fill_value=0)
        total = cats.sum(axis=1)
        pct = cats.div(total, axis=0) * 100
        return pct

    nprm_pct = _category_pct(nprm_makeup, "document_number")
    final_pct = _category_pct(final_makeup, "document_number")

    # Merge with registry by document_number
    comp_rows = []
    for _, row in resolvable.iterrows():
        nprm_dn = row["nprm_document_number"]
        final_dn = row["final_rule_document_number"]

        nprm_data = nprm_pct.loc[nprm_dn] if nprm_dn in nprm_pct.index else None
        final_data = final_pct.loc[final_dn] if final_dn in final_pct.index else None

        if nprm_data is None and final_data is None:
            continue

        entry = {"rin": row["rin"], "agency": row["agency"]}
        for cat in CATEGORY_ORDER:
            short = CATEGORY_SHORT.get(cat, cat)
            entry[f"nprm_{short}_pct"] = (
                float(nprm_data.get(cat, 0)) if nprm_data is not None else np.nan
            )
            entry[f"final_{short}_pct"] = (
                float(final_data.get(cat, 0)) if final_data is not None else np.nan
            )
        comp_rows.append(entry)

    if not comp_rows:
        logger.info("No RINs with makeup_data on both sides.")
        return None

    comp_df = pd.DataFrame(comp_rows)
    logger.info(f"Composition comparison: {len(comp_df):,} RINs with data")
    return comp_df


# ============================================================================
# COVERAGE REPORT
# ============================================================================


def generate_coverage_report(
    registry: pd.DataFrame,
    years: List[int],
) -> str:
    """Generate a human-readable coverage report.

    Reports what fraction of cross-year pairs can be resolved with the
    loaded years, and which years are needed to close the gap.
    """
    if registry.empty:
        return "No RINs in registry."

    total = len(registry)
    resolvable = int(registry["resolvable"].sum())
    has_nprm = int(registry["has_nprm_in_data"].sum())
    has_final = int(registry["has_final_in_data"].sum())

    # Count RINs with any linked doc references
    has_nprm_ref = registry["nprm_linked_year"].notna().sum()
    has_final_ref = registry["final_linked_year"].notna().sum()
    has_both_refs = (
        registry["nprm_linked_year"].notna() & registry["final_linked_year"].notna()
    ).sum()

    # Build missing-year histogram
    all_missing = []
    for val in registry["missing_years"]:
        if isinstance(val, str) and val:
            all_missing.extend(int(y) for y in val.split(","))

    missing_hist: Dict[int, int] = {}
    for y in all_missing:
        missing_hist[y] = missing_hist.get(y, 0) + 1

    lines = [
        f"=== CROSS-YEAR COVERAGE REPORT ===",
        f"",
        f"Loaded years: {', '.join(str(y) for y in years)}",
        f"Total unique RINs: {total:,}",
        f"  With NPRM doc in loaded data:  {has_nprm:,} ({has_nprm/total*100:.1f}%)",
        f"  With Final doc in loaded data: {has_final:,} ({has_final/total*100:.1f}%)",
        f"  Resolvable (both in data):     {resolvable:,} ({resolvable/total*100:.1f}%)",
        f"",
        f"Document number references (from timetable linkages):",
        f"  RINs with NPRM doc number ref:  {int(has_nprm_ref):,}",
        f"  RINs with Final doc number ref: {int(has_final_ref):,}",
        f"  RINs with both refs:            {int(has_both_refs):,}",
        f"",
    ]

    if missing_hist:
        lines.append("Years needed to close the gap (RIN count per year):")
        for y in sorted(missing_hist.keys()):
            lines.append(f"  {y}: {missing_hist[y]:,} RINs")
        lines.append("")

        # Top 3 most-needed years
        top3 = sorted(missing_hist.items(), key=lambda x: x[1], reverse=True)[:3]
        lines.append(
            f"Highest-impact years to add: "
            f"{', '.join(f'{y} ({n} RINs)' for y, n in top3)}"
        )
    else:
        lines.append("No missing years -- all linked documents are in loaded data.")

    lines.append("")
    lines.append(
        f"VERDICT: With years {'+'.join(str(y) for y in years)}, "
        f"you can resolve {resolvable:,}/{total:,} ({resolvable/total*100:.1f}%) "
        f"of cross-year RIN pairs."
    )
    gap_pct = (total - resolvable) / total * 100 if total > 0 else 0
    lines.append(f"The gap is {gap_pct:.1f}%.")

    return "\n".join(lines)


# ============================================================================
# PLOTS
# ============================================================================


def plot_cross_year_timeline(
    registry: pd.DataFrame,
    outdir: Path,
) -> None:
    """Scatter plot: NPRM year vs Final year, colored by outcome."""
    df = registry[
        registry["nprm_year"].notna() & registry["final_year"].notna()
    ].copy()

    if df.empty:
        logger.warning("No RINs with both NPRM and Final years -- skipping timeline plot.")
        return

    df["nprm_year"] = df["nprm_year"].astype(int)
    df["final_year"] = df["final_year"].astype(int)

    outcome_colors = {
        "finalized": "#4E79A7",
        "withdrawn": "#EDC948",
        "stalled": "#E15759",
        "unknown": "#9C9C9C",
    }

    fig, ax = plt.subplots(figsize=(10, 8))

    for outcome, color in outcome_colors.items():
        subset = df[df["outcome"] == outcome]
        if subset.empty:
            continue
        # Jitter to avoid overplotting
        jitter_x = np.random.normal(0, 0.12, len(subset))
        jitter_y = np.random.normal(0, 0.12, len(subset))
        ax.scatter(
            subset["nprm_year"] + jitter_x,
            subset["final_year"] + jitter_y,
            c=color, alpha=0.5, s=20, label=f"{outcome} (n={len(subset)})",
            edgecolors="none",
        )

    # Diagonal reference line (same-year)
    all_years = sorted(set(df["nprm_year"]) | set(df["final_year"]))
    if all_years:
        ax.plot(
            [min(all_years), max(all_years)],
            [min(all_years), max(all_years)],
            "k--", alpha=0.3, linewidth=1, label="Same year"
        )

    ax.set_xlabel("NPRM Year", fontsize=FONT_LABEL)
    ax.set_ylabel("Final Rule Year", fontsize=FONT_LABEL)
    ax.set_title("Cross-Year Rulemaking Timeline", fontsize=FONT_TITLE,
                 fontweight="bold", pad=15)
    ax.legend(fontsize=FONT_LEGEND, loc="upper left")
    apply_clean_style(ax)

    plt.tight_layout()
    path = outdir / "cross_year_timeline.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[OK] Saved: {path.name}")


def plot_coverage_gap(
    registry: pd.DataFrame,
    outdir: Path,
) -> None:
    """Bar chart: year histogram of unresolvable linked documents."""
    all_missing = []
    for val in registry["missing_years"]:
        if isinstance(val, str) and val:
            all_missing.extend(int(y) for y in val.split(","))

    if not all_missing:
        logger.info("No missing years -- skipping coverage gap plot.")
        return

    missing_series = pd.Series(all_missing)
    counts = missing_series.value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(12, 6))
    x_pos = np.arange(len(counts))
    ax.bar(x_pos, counts.values, color="#E15759", alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(y) for y in counts.index], rotation=45, ha="right")

    ax.set_xlabel("Year", fontsize=FONT_LABEL)
    ax.set_ylabel("Number of RINs Needing This Year", fontsize=FONT_LABEL)
    ax.set_title("Coverage Gap: Which Years Are Missing?", fontsize=FONT_TITLE,
                 fontweight="bold", pad=15)

    # Annotate top bars
    for x, y in zip(x_pos, counts.values):
        ax.text(x, y + 1, str(y), ha="center", va="bottom", fontsize=10)

    apply_clean_style(ax)
    plt.tight_layout()
    path = outdir / "coverage_gap_years.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[OK] Saved: {path.name}")


def plot_composition_shift(
    comp_df: Optional[pd.DataFrame],
    outdir: Path,
) -> None:
    """Paired bar chart: NPRM composition vs Final composition per category."""
    if comp_df is None or comp_df.empty:
        logger.info("No composition data -- skipping composition shift plot.")
        return

    # Compute mean percentages across all RINs
    nprm_means = []
    final_means = []
    labels = []
    for cat in CATEGORY_ORDER:
        short = CATEGORY_SHORT.get(cat, cat)
        nprm_col = f"nprm_{short}_pct"
        final_col = f"final_{short}_pct"
        if nprm_col in comp_df.columns and final_col in comp_df.columns:
            nprm_means.append(comp_df[nprm_col].mean())
            final_means.append(comp_df[final_col].mean())
            labels.append(short)

    if not labels:
        logger.warning("No composition columns found -- skipping plot.")
        return

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, nprm_means, width, label="NPRM Stage",
           color="#F28E2B", alpha=0.85)
    ax.bar(x + width / 2, final_means, width, label="Final Rule Stage",
           color="#4E79A7", alpha=0.85)

    ax.set_xlabel("Commenter Category", fontsize=FONT_LABEL)
    ax.set_ylabel("Mean Percentage (%)", fontsize=FONT_LABEL)
    ax.set_title("Commenter Composition: NPRM vs Final Rule",
                 fontsize=FONT_TITLE, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FONT_LABEL)
    ax.legend(fontsize=FONT_LEGEND)
    apply_clean_style(ax)

    plt.tight_layout()
    path = outdir / "composition_shift_nprm_vs_final.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[OK] Saved: {path.name}")


# ============================================================================
# HYPOTHESIS TESTING
# ============================================================================


def run_hypotheses(fr_df: pd.DataFrame, outdir: Path) -> None:
    """Run hypothesis tests on single-year data.

    H1: WITHDRAWN vs FINAL_EFFECTIVE comment_count comparison
    H2: nprm_to_final_days vs timetable_action_count correlation
    H3: ANPRM docs take longer to finalize
    H4: Coverage gap affects median-duration rules most
    """
    log_banner(logger, "HYPOTHESIS TESTING")

    _test_h1_withdrawn_vs_finalized(fr_df)
    _test_h2_duration_vs_complexity(fr_df, outdir)
    _test_h3_anprm_prolongs(fr_df)
    _test_h4_coverage_gap_by_duration(fr_df)


def _test_h1_withdrawn_vs_finalized(fr_df: pd.DataFrame) -> None:
    """H1: Withdrawn rules have different comment volume than finalized ones."""
    logger.info("--- H1: WITHDRAWN vs FINAL_EFFECTIVE comment volume ---")

    rin_df = fr_df[
        fr_df["rin"].notna()
        & (fr_df["rin"] != "")
        & fr_df["comment_count"].notna()
        & (fr_df["comment_count"] > 0)
    ].copy()

    withdrawn = rin_df[rin_df["lifecycle_stage"] == "WITHDRAWN"]["comment_count"]
    finalized = rin_df[rin_df["lifecycle_stage"] == "FINAL_EFFECTIVE"]["comment_count"]

    logger.info(f"  WITHDRAWN:      N={len(withdrawn):,}, "
                f"median={withdrawn.median():.0f}, mean={withdrawn.mean():.1f}"
                if len(withdrawn) > 0 else "  WITHDRAWN: N=0")
    logger.info(f"  FINAL_EFFECTIVE: N={len(finalized):,}, "
                f"median={finalized.median():.0f}, mean={finalized.mean():.1f}"
                if len(finalized) > 0 else "  FINAL_EFFECTIVE: N=0")

    if len(withdrawn) >= 5 and len(finalized) >= 5:
        stat, pval = stats.mannwhitneyu(
            withdrawn, finalized, alternative="two-sided"
        )
        logger.info(f"  Mann-Whitney U: stat={stat:.1f}, p={pval:.4f}")
        if pval < 0.05:
            logger.info("  RESULT: Statistically significant difference (p < 0.05)")
        else:
            logger.info("  RESULT: No significant difference (p >= 0.05)")
    else:
        logger.info("  RESULT: Insufficient data for statistical test (need N >= 5 per group)")


def _test_h2_duration_vs_complexity(fr_df: pd.DataFrame, outdir: Path) -> None:
    """H2: Longer NPRM-to-Final correlates with more complex timetable sequences."""
    logger.info("--- H2: Duration vs timetable complexity ---")

    df = fr_df[
        fr_df["nprm_to_final_days"].notna()
        & (fr_df["nprm_to_final_days"] > 0)
        & fr_df["timetable_action_count"].notna()
        & (fr_df["timetable_action_count"] > 0)
    ].copy()

    if len(df) < 10:
        logger.info(f"  Insufficient data: {len(df)} rows (need >= 10)")
        return

    r, pval = stats.spearmanr(df["nprm_to_final_days"], df["timetable_action_count"])
    logger.info(f"  N={len(df):,}")
    logger.info(f"  Spearman r={r:.3f}, p={pval:.4f}")

    if pval < 0.05:
        direction = "positive" if r > 0 else "negative"
        logger.info(f"  RESULT: Significant {direction} correlation")
    else:
        logger.info("  RESULT: No significant correlation")

    # Scatter plot
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(
        df["timetable_action_count"],
        df["nprm_to_final_days"],
        alpha=0.3, s=15, c="#4E79A7", edgecolors="none",
    )
    ax.set_xlabel("Timetable Action Count", fontsize=FONT_LABEL)
    ax.set_ylabel("NPRM-to-Final (days)", fontsize=FONT_LABEL)
    ax.set_title(
        f"H2: Duration vs Complexity (r={r:.3f}, p={pval:.4f})",
        fontsize=FONT_TITLE, fontweight="bold", pad=15,
    )
    apply_clean_style(ax)
    plt.tight_layout()
    path = outdir / "h2_duration_vs_complexity.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  [OK] Saved: {path.name}")


def _test_h3_anprm_prolongs(fr_df: pd.DataFrame) -> None:
    """H3: Rules with ANPRM take longer to finalize."""
    logger.info("--- H3: ANPRM prolongs finalization ---")

    df = fr_df[
        fr_df["nprm_to_final_days"].notna()
        & (fr_df["nprm_to_final_days"] > 0)
    ].copy()

    if "anprm_date" not in df.columns:
        logger.info("  anprm_date column not available -- skipping H3")
        return

    with_anprm = df[df["anprm_date"].notna() & (df["anprm_date"] != "")][
        "nprm_to_final_days"
    ]
    without_anprm = df[df["anprm_date"].isna() | (df["anprm_date"] == "")][
        "nprm_to_final_days"
    ]

    logger.info(f"  With ANPRM:    N={len(with_anprm):,}, "
                f"median={with_anprm.median():.0f}, mean={with_anprm.mean():.1f}"
                if len(with_anprm) > 0 else "  With ANPRM: N=0")
    logger.info(f"  Without ANPRM: N={len(without_anprm):,}, "
                f"median={without_anprm.median():.0f}, mean={without_anprm.mean():.1f}"
                if len(without_anprm) > 0 else "  Without ANPRM: N=0")

    if len(with_anprm) >= 5 and len(without_anprm) >= 5:
        stat, pval = stats.mannwhitneyu(
            with_anprm, without_anprm, alternative="greater"
        )
        logger.info(f"  Mann-Whitney U (one-sided, ANPRM > no-ANPRM): "
                     f"stat={stat:.1f}, p={pval:.4f}")
        if pval < 0.05:
            logger.info("  RESULT: ANPRM significantly prolongs finalization (p < 0.05)")
        else:
            logger.info("  RESULT: No significant evidence ANPRM prolongs (p >= 0.05)")
    else:
        logger.info("  RESULT: Insufficient data for statistical test")


def _test_h4_coverage_gap_by_duration(fr_df: pd.DataFrame) -> None:
    """H4: Coverage gap primarily affects median-duration rules (1-3 years)."""
    logger.info("--- H4: Coverage gap by duration range ---")

    df = fr_df[
        fr_df["nprm_to_final_days"].notna()
        & (fr_df["nprm_to_final_days"] > 0)
        & fr_df["nprm_document_number"].notna()
        & fr_df["final_rule_document_number"].notna()
    ].copy()

    if df.empty:
        logger.info("  No docs with both nprm/final doc numbers and duration -- skipping H4")
        return

    # Extract years from linked doc numbers
    df["nprm_ref_year"] = df["nprm_document_number"].apply(_extract_year_from_doc_number)
    df["final_ref_year"] = df["final_rule_document_number"].apply(_extract_year_from_doc_number)

    df = df[df["nprm_ref_year"].notna() & df["final_ref_year"].notna()].copy()
    df["same_year"] = df["nprm_ref_year"] == df["final_ref_year"]

    # Bin by duration
    bins = [0, 365, 730, 1095, 1825, float("inf")]
    labels = ["<1yr", "1-2yr", "2-3yr", "3-5yr", "5yr+"]
    df["duration_bin"] = pd.cut(
        df["nprm_to_final_days"], bins=bins, labels=labels, right=True
    )

    logger.info(f"  Total docs with both refs and duration: {len(df):,}")
    logger.info(f"  Same-year pairs: {df['same_year'].sum():,} ({df['same_year'].mean()*100:.1f}%)")
    logger.info("")
    logger.info("  Duration bin | Total | Same-year | Same-year %")
    logger.info("  " + "-" * 55)

    for label in labels:
        subset = df[df["duration_bin"] == label]
        n_total = len(subset)
        n_same = subset["same_year"].sum()
        pct = (n_same / n_total * 100) if n_total > 0 else 0
        logger.info(f"  {label:>12s} | {n_total:5d} | {n_same:9d} | {pct:10.1f}%")

    logger.info("")
    logger.info("  INTERPRETATION: Higher same-year % = better coverage with single-year data.")
    logger.info("  Lower same-year % = more cross-year pairs needed (gap).")


# ============================================================================
# MAIN
# ============================================================================


def run_cross_year_analysis(
    years: List[int],
    outdir: Optional[Path] = None,
    verbose: bool = False,
    quiet: bool = False,
) -> int:
    """Main analysis entry point.

    Returns 0 on success, 1 on failure.
    """
    setup_logging(verbose=verbose, quiet=quiet)
    log_banner(logger, f"CROSS-YEAR LIFECYCLE ANALYSIS (years: {', '.join(str(y) for y in years)})")

    # Output directory
    if outdir is None:
        outdir = get_data_dir() / "plots" / "cross_year"
    outdir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {outdir}")

    # ---- Step 1: Load data ----
    log_banner(logger, "STEP 1: LOAD DATA", char="-")
    fr_df = load_multi_year_fr(years)
    if fr_df.empty:
        return 1

    makeup_df = load_multi_year_makeup(years)

    # ---- Step 2: Build RIN registry ----
    log_banner(logger, "STEP 2: BUILD RIN REGISTRY", char="-")
    registry = build_rin_registry(fr_df)
    if registry.empty:
        logger.error("Empty RIN registry. No RIN-bearing documents found.")
        return 1

    # ---- Step 3: Coverage report ----
    log_banner(logger, "STEP 3: COVERAGE REPORT", char="-")
    report = generate_coverage_report(registry, years)
    for line in report.split("\n"):
        logger.info(line)

    # ---- Step 4: Commenter composition ----
    log_banner(logger, "STEP 4: COMMENTER COMPOSITION", char="-")
    comp_df = compare_commenter_composition(registry, makeup_df)

    # ---- Step 5: Plots ----
    log_banner(logger, "STEP 5: VISUALIZATIONS", char="-")
    plot_cross_year_timeline(registry, outdir)
    plot_coverage_gap(registry, outdir)
    plot_composition_shift(comp_df, outdir)

    # ---- Step 6: Hypothesis testing ----
    run_hypotheses(fr_df, outdir)

    # ---- Step 7: Save summary CSV ----
    log_banner(logger, "STEP 7: SAVE SUMMARY CSV", char="-")
    csv_path = get_output_dir() / "cross_year_summary.csv"
    registry.to_csv(csv_path, index=False)
    logger.info(f"[OK] Saved summary CSV: {csv_path} ({len(registry):,} RINs)")

    # Save composition CSV if available
    if comp_df is not None:
        comp_path = get_output_dir() / "cross_year_composition.csv"
        comp_df.to_csv(comp_path, index=False)
        logger.info(f"[OK] Saved composition CSV: {comp_path} ({len(comp_df):,} RINs)")

    log_banner(logger, "CROSS-YEAR ANALYSIS COMPLETE")
    return 0


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Cross-year lifecycle analysis: join documents across years by RIN, "
            "track NPRM-to-Final progression, compare commenter composition."
        )
    )
    parser.add_argument(
        "--years", required=True,
        help="Years to analyze: '2013,2014' or '2013-2016' or mixed"
    )
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Custom output directory for plots/CSVs")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable debug logging")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress non-error output")

    args = parser.parse_args()
    years = parse_years(args.years)

    if not years:
        print("ERROR: No valid years provided.", file=sys.stderr)
        return 1

    return run_cross_year_analysis(
        years=years,
        outdir=args.output_dir,
        verbose=args.verbose,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    sys.exit(main())
