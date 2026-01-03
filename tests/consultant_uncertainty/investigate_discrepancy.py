"""
Investigate the discrepancy between the plot (low uncertainty) and our analysis (high uncertainty).

The plot only shows comments where response_found == "yes", but we're looking at all comments.
Let's investigate:
1. What happens when we filter to only response_found == "yes"?
2. Are there weird cases where consultants have no response but uncertain decision?
3. Are there cases where others have no response but accept decision?
4. What about weighting - does that change things?
"""

import polars as pl
from pathlib import Path

# Paths to data files
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "stratification_scripts" / "makeup" / "data"
OUTPUT_DIR = PROJECT_ROOT / "stratification_scripts" / "output"

COMMENTS_RAW = DATA_DIR / "comments_raw_2025.csv"
MAKEUP_DATA = OUTPUT_DIR / "makeup_data_2025.csv"
AGENCY_RESPONSES = DATA_DIR / "agency_responses_2025.csv"
OUTPUT_TXT = Path(__file__).parent / "discrepancy_investigation.txt"


def load_and_join_data() -> pl.DataFrame:
    """Load and join all relevant data files."""
    print("Loading data files...")
    
    df_comments = pl.read_csv(COMMENTS_RAW)
    df_makeup = pl.read_csv(MAKEUP_DATA)
    df_responses = pl.read_csv(AGENCY_RESPONSES)
    
    # Join: comments -> makeup -> responses
    df = df_comments.join(df_makeup, on="comment_id", how="inner")
    
    # Handle duplicate columns in responses
    response_cols = [col for col in df_responses.columns if col not in ["document_number", "agency", "comment_id"]]
    df_responses_subset = df_responses.select(["comment_id"] + response_cols)
    df = df.join(df_responses_subset, on="comment_id", how="left")
    
    # Add weights if they exist in makeup data
    if "weight" in df_makeup.columns:
        df = df.join(df_makeup.select(["comment_id", "weight"]), on="comment_id", how="left", suffix="_makeup")
        # Use weight from makeup if available, otherwise 1.0
        if "weight_makeup" in df.columns:
            df = df.with_columns(
                pl.coalesce([pl.col("weight_makeup"), pl.lit(1.0)]).alias("weight")
            )
    else:
        df = df.with_columns(pl.lit(1.0).alias("weight"))
    
    return df


def analyze_discrepancy(df: pl.DataFrame) -> str:
    """Analyze the discrepancy between plot and our analysis."""
    lines = []
    lines.append("=" * 80)
    lines.append("DISCREPANCY INVESTIGATION: Plot vs Analysis")
    lines.append("=" * 80)
    lines.append("")
    lines.append("The plot only shows comments where response_found == 'yes'")
    lines.append("Our analysis shows ALL comments (including no response cases)")
    lines.append("")
    
    # Split consultants and others
    consultants = df.filter(pl.col("category") == "Political Consultant/Lobbyist")
    others = df.filter(pl.col("category") != "Political Consultant/Lobbyist")
    
    lines.append("=" * 80)
    lines.append("1. ALL COMMENTS (what our analysis shows)")
    lines.append("=" * 80)
    lines.append("")
    
    # Overall decision rates for all comments
    consultants_total = consultants.height
    others_total = others.height
    
    consultants_accept_all = consultants.filter(pl.col("agency_decision") == "accept").height
    consultants_reject_all = consultants.filter(pl.col("agency_decision") == "reject").height
    consultants_uncertain_all = consultants.filter(pl.col("agency_decision") == "uncertain").height
    
    others_accept_all = others.filter(pl.col("agency_decision") == "accept").height
    others_reject_all = others.filter(pl.col("agency_decision") == "reject").height
    others_uncertain_all = others.filter(pl.col("agency_decision") == "uncertain").height
    
    lines.append("CONSULTANTS (all comments):")
    lines.append(f"  Accept: {consultants_accept_all:,} ({100*consultants_accept_all/consultants_total:.1f}%)")
    lines.append(f"  Reject: {consultants_reject_all:,} ({100*consultants_reject_all/consultants_total:.1f}%)")
    lines.append(f"  Uncertain: {consultants_uncertain_all:,} ({100*consultants_uncertain_all/consultants_total:.1f}%)")
    lines.append("")
    
    lines.append("OTHERS (all comments):")
    lines.append(f"  Accept: {others_accept_all:,} ({100*others_accept_all/others_total:.1f}%)")
    lines.append(f"  Reject: {others_reject_all:,} ({100*others_reject_all/others_total:.1f}%)")
    lines.append(f"  Uncertain: {others_uncertain_all:,} ({100*others_uncertain_all/others_total:.1f}%)")
    lines.append("")
    
    lines.append("=" * 80)
    lines.append("2. ONLY COMMENTS WITH RESPONSES FOUND (what the plot shows)")
    lines.append("=" * 80)
    lines.append("")
    
    # Filter to only response_found == "yes"
    consultants_with_response = consultants.filter(pl.col("response_found") == "yes")
    others_with_response = others.filter(pl.col("response_found") == "yes")
    
    consultants_resp_total = consultants_with_response.height
    others_resp_total = others_with_response.height
    
    consultants_accept_resp = consultants_with_response.filter(pl.col("agency_decision") == "accept").height
    consultants_reject_resp = consultants_with_response.filter(pl.col("agency_decision") == "reject").height
    consultants_uncertain_resp = consultants_with_response.filter(pl.col("agency_decision") == "uncertain").height
    
    others_accept_resp = others_with_response.filter(pl.col("agency_decision") == "accept").height
    others_reject_resp = others_with_response.filter(pl.col("agency_decision") == "reject").height
    others_uncertain_resp = others_with_response.filter(pl.col("agency_decision") == "uncertain").height
    
    lines.append("CONSULTANTS (with responses found):")
    if consultants_resp_total > 0:
        lines.append(f"  Accept: {consultants_accept_resp:,} ({100*consultants_accept_resp/consultants_resp_total:.1f}%)")
        lines.append(f"  Reject: {consultants_reject_resp:,} ({100*consultants_reject_resp/consultants_resp_total:.1f}%)")
        lines.append(f"  Uncertain: {consultants_uncertain_resp:,} ({100*consultants_uncertain_resp/consultants_resp_total:.1f}%)")
    else:
        lines.append("  No responses found")
    lines.append("")
    
    lines.append("OTHERS (with responses found):")
    if others_resp_total > 0:
        lines.append(f"  Accept: {others_accept_resp:,} ({100*others_accept_resp/others_resp_total:.1f}%)")
        lines.append(f"  Reject: {others_reject_resp:,} ({100*others_reject_resp/others_resp_total:.1f}%)")
        lines.append(f"  Uncertain: {others_uncertain_resp:,} ({100*others_uncertain_resp/others_resp_total:.1f}%)")
    else:
        lines.append("  No responses found")
    lines.append("")
    
    lines.append("=" * 80)
    lines.append("3. WEIRD CASES INVESTIGATION")
    lines.append("=" * 80)
    lines.append("")
    
    # Case 1: Consultants with no response but uncertain decision
    consultants_no_resp_uncertain = consultants.filter(
        (pl.col("response_found") == "no") & 
        (pl.col("agency_decision") == "uncertain")
    )
    lines.append(f"Consultants: No response + Uncertain decision: {consultants_no_resp_uncertain.height:,}")
    
    # Case 2: Others with no response but accept decision
    others_no_resp_accept = others.filter(
        (pl.col("response_found") == "no") & 
        (pl.col("agency_decision") == "accept")
    )
    lines.append(f"Others: No response + Accept decision: {others_no_resp_accept.height:,}")
    
    # Case 3: Consultants with no response but accept decision
    consultants_no_resp_accept = consultants.filter(
        (pl.col("response_found") == "no") & 
        (pl.col("agency_decision") == "accept")
    )
    lines.append(f"Consultants: No response + Accept decision: {consultants_no_resp_accept.height:,}")
    
    # Case 4: Others with no response but uncertain decision
    others_no_resp_uncertain = others.filter(
        (pl.col("response_found") == "no") & 
        (pl.col("agency_decision") == "uncertain")
    )
    lines.append(f"Others: No response + Uncertain decision: {others_no_resp_uncertain.height:,}")
    lines.append("")
    
    # Case 5: Consultants with response but uncertain decision
    consultants_resp_uncertain = consultants.filter(
        (pl.col("response_found") == "yes") & 
        (pl.col("agency_decision") == "uncertain")
    )
    lines.append(f"Consultants: Response found + Uncertain decision: {consultants_resp_uncertain.height:,}")
    
    # Case 6: Others with response but uncertain decision
    others_resp_uncertain = others.filter(
        (pl.col("response_found") == "yes") & 
        (pl.col("agency_decision") == "uncertain")
    )
    lines.append(f"Others: Response found + Uncertain decision: {others_resp_uncertain.height:,}")
    lines.append("")
    
    lines.append("=" * 80)
    lines.append("4. WEIGHTED ANALYSIS (like the plot)")
    lines.append("=" * 80)
    lines.append("")
    
    # Calculate weighted percentages for response_found == "yes" only
    consultants_with_response_weighted = consultants_with_response["weight"].sum()
    others_with_response_weighted = others_with_response["weight"].sum()
    
    if consultants_with_response_weighted > 0:
        consultants_accept_weighted = consultants_with_response.filter(pl.col("agency_decision") == "accept")["weight"].sum()
        consultants_reject_weighted = consultants_with_response.filter(pl.col("agency_decision") == "reject")["weight"].sum()
        consultants_uncertain_weighted = consultants_with_response.filter(pl.col("agency_decision") == "uncertain")["weight"].sum()
        
        lines.append("CONSULTANTS (weighted, with responses found):")
        lines.append(f"  Accept: {100*consultants_accept_weighted/consultants_with_response_weighted:.1f}%")
        lines.append(f"  Reject: {100*consultants_reject_weighted/consultants_with_response_weighted:.1f}%")
        lines.append(f"  Uncertain: {100*consultants_uncertain_weighted/consultants_with_response_weighted:.1f}%")
        lines.append("")
    
    if others_with_response_weighted > 0:
        others_accept_weighted = others_with_response.filter(pl.col("agency_decision") == "accept")["weight"].sum()
        others_reject_weighted = others_with_response.filter(pl.col("agency_decision") == "reject")["weight"].sum()
        others_uncertain_weighted = others_with_response.filter(pl.col("agency_decision") == "uncertain")["weight"].sum()
        
        lines.append("OTHERS (weighted, with responses found):")
        lines.append(f"  Accept: {100*others_accept_weighted/others_with_response_weighted:.1f}%")
        lines.append(f"  Reject: {100*others_reject_weighted/others_with_response_weighted:.1f}%")
        lines.append(f"  Uncertain: {100*others_uncertain_weighted/others_with_response_weighted:.1f}%")
        lines.append("")
    
    lines.append("=" * 80)
    lines.append("5. KEY INSIGHT")
    lines.append("=" * 80)
    lines.append("")
    lines.append("The plot shows ONLY comments where response_found == 'yes'")
    lines.append("Among those comments:")
    lines.append(f"  - Consultants have {100*consultants_uncertain_resp/consultants_resp_total:.1f}% uncertain (if weighted: {100*consultants_uncertain_weighted/consultants_with_response_weighted:.1f}%)")
    lines.append(f"  - Others have {100*others_uncertain_resp/others_resp_total:.1f}% uncertain (if weighted: {100*others_uncertain_weighted/others_with_response_weighted:.1f}%)")
    lines.append("")
    lines.append("But consultants have:")
    lines.append(f"  - {100*consultants_resp_total/consultants_total:.1f}% response rate")
    lines.append(f"  - {100*consultants_no_resp_uncertain.height/consultants_total:.1f}% no response + uncertain")
    lines.append("")
    lines.append("While others have:")
    lines.append(f"  - {100*others_resp_total/others_total:.1f}% response rate")
    lines.append(f"  - {100*others_no_resp_uncertain.height/others_total:.1f}% no response + uncertain")
    lines.append("")
    
    return "\n".join(lines)


def main():
    print("Investigating discrepancy...")
    df = load_and_join_data()
    report = analyze_discrepancy(df)
    
    OUTPUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"Investigation complete! Output saved to: {OUTPUT_TXT}")


if __name__ == "__main__":
    main()

