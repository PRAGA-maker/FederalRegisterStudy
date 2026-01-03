"""
Debug what data would actually be passed to the plot function.
Check if the join is causing issues with response_found values.
"""

import polars as pl
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "stratification_scripts" / "makeup" / "data"
OUTPUT_DIR = PROJECT_ROOT / "stratification_scripts" / "output"

COMMENTS_RAW = DATA_DIR / "comments_raw_2025.csv"
MAKEUP_DATA = OUTPUT_DIR / "makeup_data_2025.csv"
AGENCY_RESPONSES = DATA_DIR / "agency_responses_2025.csv"
OUTPUT_TXT = Path(__file__).parent / "plot_data_debug.txt"


def simulate_plot_data():
    """Simulate exactly what the plot function receives."""
    lines = []
    lines.append("Loading data...")
    
    # Load as polars
    df_responses = pl.read_csv(AGENCY_RESPONSES)
    df_makeup = pl.read_csv(MAKEUP_DATA)
    
    lines.append(f"Responses: {df_responses.height:,}")
    lines.append(f"Makeup: {df_makeup.height:,}")
    
    # Join exactly like the plot code does (line 2471)
    merge_cols = ["comment_id", "category"]
    if "weight" in df_makeup.columns:
        merge_cols.append("weight")
    if "weight_doc" in df_makeup.columns:
        merge_cols.append("weight_doc")
    if "document_number" in df_makeup.columns:
        merge_cols.append("document_number")
    
    df = df_responses.join(df_makeup.select(merge_cols), on="comment_id", how="left")
    
    lines.append(f"After join: {df.height:,}")
    lines.append("response_found value counts:")
    vc = df["response_found"].value_counts().sort("response_found")
    for row in vc.iter_rows(named=True):
        lines.append(f"  {row['response_found']}: {row['count']:,}")
    lines.append("")
    
    # Check for nulls
    lines.append(f"response_found nulls: {df.filter(pl.col('response_found').is_null()).height}")
    lines.append(f"agency_decision nulls: {df.filter(pl.col('agency_decision').is_null()).height}")
    lines.append("")
    
    # Filter like plot does (line 2762)
    df_found = df.filter(pl.col("response_found") == "yes")
    lines.append(f"After filtering to response_found == 'yes': {df_found.height:,}")
    lines.append("")
    
    # Check consultants
    consultants = df_found.filter(pl.col("category") == "Political Consultant/Lobbyist")
    lines.append(f"Consultants with response_found == 'yes': {consultants.height:,}")
    
    if consultants.height > 0:
        lines.append("Consultant decision breakdown:")
        vc = consultants["agency_decision"].value_counts().sort("agency_decision")
        for row in vc.iter_rows(named=True):
            lines.append(f"  {row['agency_decision']}: {row['count']:,}")
        lines.append("")
        
        # Weighted percentages
        if "weight" in consultants.columns:
            total_weight = consultants["weight"].sum()
            lines.append(f"Total consultant weight: {total_weight:.2f}")
            
            for decision in ["accept", "reject", "uncertain"]:
                decision_data = consultants.filter(pl.col("agency_decision") == decision)
                decision_weight = decision_data["weight"].sum()
                pct = (decision_weight / total_weight * 100) if total_weight > 0 else 0
                lines.append(f"  {decision}: {decision_weight:.2f} ({pct:.1f}%)")
    
    # Check what happens if we DON'T filter
    lines.append("\n" + "="*80)
    lines.append("IF WE DON'T FILTER (all comments):")
    lines.append("="*80)
    consultants_all = df.filter(pl.col("category") == "Political Consultant/Lobbyist")
    lines.append(f"All consultants: {consultants_all.height:,}")
    
    if consultants_all.height > 0:
        lines.append("Consultant decision breakdown (all):")
        vc = consultants_all["agency_decision"].value_counts().sort("agency_decision")
        for row in vc.iter_rows(named=True):
            lines.append(f"  {row['agency_decision']}: {row['count']:,}")
        lines.append("")
        
        # Check response_found distribution
        lines.append("Consultant response_found distribution:")
        vc = consultants_all["response_found"].value_counts().sort("response_found")
        for row in vc.iter_rows(named=True):
            lines.append(f"  {row['response_found']}: {row['count']:,}")
        lines.append("")
        
        # Weighted percentages for all
        if "weight" in consultants_all.columns:
            total_weight = consultants_all["weight"].sum()
            lines.append(f"Total consultant weight (all): {total_weight:.2f}")
            
            for decision in ["accept", "reject", "uncertain"]:
                decision_data = consultants_all.filter(pl.col("agency_decision") == decision)
                decision_weight = decision_data["weight"].sum()
                pct = (decision_weight / total_weight * 100) if total_weight > 0 else 0
                lines.append(f"  {decision}: {decision_weight:.2f} ({pct:.1f}%)")
    
    # Check if there are consultants with null response_found
    consultants_null = df.filter(
        (pl.col("category") == "Political Consultant/Lobbyist") & 
        pl.col("response_found").is_null()
    )
    lines.append(f"\nConsultants with NULL response_found: {consultants_null.height:,}")
    if consultants_null.height > 0:
        lines.append("Their agency_decision values:")
        vc = consultants_null["agency_decision"].value_counts().sort("agency_decision")
        for row in vc.iter_rows(named=True):
            lines.append(f"  {row['agency_decision']}: {row['count']:,}")
    
    # Write to file
    OUTPUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    print(f"Debug output saved to: {OUTPUT_TXT}")


if __name__ == "__main__":
    simulate_plot_data()

