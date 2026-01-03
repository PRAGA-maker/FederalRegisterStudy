"""
Test script to verify the plot logic is working correctly.
This simulates what the plot function should do.
"""

import polars as pl
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "stratification_scripts" / "makeup" / "data"
OUTPUT_DIR = PROJECT_ROOT / "stratification_scripts" / "output"

COMMENTS_RAW = DATA_DIR / "comments_raw_2025.csv"
MAKEUP_DATA = OUTPUT_DIR / "makeup_data_2025.csv"
AGENCY_RESPONSES = DATA_DIR / "agency_responses_2025.csv"

CATEGORY_ORDER = [
    "Undecided/Anonymous",
    "Ordinary Citizen",
    "Large Organization/Corporation",
    "Academic/Industry/Expert (incl. small/local business)",
    "Political Consultant/Lobbyist",
]

def test_plot_logic():
    """Test the exact logic that plot_decision_by_type should use."""
    print("Loading data...")
    
    # Load data
    df_responses = pl.read_csv(AGENCY_RESPONSES)
    df_makeup = pl.read_csv(MAKEUP_DATA)
    
    print(f"Responses: {df_responses.height:,}")
    print(f"Makeup: {df_makeup.height:,}")
    
    # Join with INNER join (like the fixed code)
    merge_cols = ["comment_id", "category"]
    if "weight" in df_makeup.columns:
        merge_cols.append("weight")
    
    df = df_responses.join(df_makeup.select(merge_cols), on="comment_id", how="inner")
    print(f"After inner join: {df.height:,}")
    
    # Add default weight if missing
    if "weight" not in df.columns:
        df = df.with_columns(pl.lit(1.0).alias("weight"))
    
    # Filter to responses found AND valid category (like the fixed code)
    df_found = df.filter(
        (pl.col("response_found") == "yes") & 
        (pl.col("category").is_not_null()) & 
        (pl.col("category").is_in(CATEGORY_ORDER))
    )
    print(f"After filtering to response_found=='yes' AND valid category: {df_found.height:,}")
    
    # Check consultants
    consultants = df_found.filter(pl.col("category") == "Political Consultant/Lobbyist")
    print(f"\nConsultants with response_found=='yes' AND valid category: {consultants.height:,}")
    
    if consultants.height > 0:
        consultants_total_weight = consultants["weight"].sum()
        print(f"Total consultant weight: {consultants_total_weight:.2f}")
        
        for decision in ["accept", "reject", "uncertain"]:
            decision_data = consultants.filter(pl.col("agency_decision") == decision)
            decision_weight = decision_data["weight"].sum()
            pct = (decision_weight / consultants_total_weight * 100) if consultants_total_weight > 0 else 0
            count = decision_data.height
            print(f"  {decision}: {count:,} ({decision_weight:.2f} weight) = {pct:.1f}%")
        
        # Verify sum
        total_pct = sum([
            (consultants.filter(pl.col("agency_decision") == d)["weight"].sum() / consultants_total_weight * 100) 
            if consultants_total_weight > 0 else 0
            for d in ["accept", "reject", "uncertain"]
        ])
        print(f"\nTotal percentage: {total_pct:.1f}% (should be ~100%)")
    
    # Check others for comparison
    others = df_found.filter(pl.col("category") != "Political Consultant/Lobbyist")
    print(f"\nOthers with response_found=='yes' AND valid category: {others.height:,}")
    
    if others.height > 0:
        others_total_weight = others["weight"].sum()
        print(f"Total others weight: {others_total_weight:.2f}")
        
        for decision in ["accept", "reject", "uncertain"]:
            decision_data = others.filter(pl.col("agency_decision") == decision)
            decision_weight = decision_data["weight"].sum()
            pct = (decision_weight / others_total_weight * 100) if others_total_weight > 0 else 0
            count = decision_data.height
            print(f"  {decision}: {count:,} ({decision_weight:.2f} weight) = {pct:.1f}%")

if __name__ == "__main__":
    test_plot_logic()

