"""
Analyze why political consultant comments have uncertain agency decisions.

This script compares political consultants/lobbyists with other commenter types
to identify features that may explain the high uncertainty rate in agency decisions.
"""

import polars as pl
import random
from pathlib import Path
from typing import Dict, List, Tuple

# Paths to data files (relative to project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "stratification_scripts" / "makeup" / "data"
OUTPUT_DIR = PROJECT_ROOT / "stratification_scripts" / "output"

COMMENTS_RAW = DATA_DIR / "comments_raw_2025.csv"
MAKEUP_DATA = OUTPUT_DIR / "makeup_data_2025.csv"
AGENCY_RESPONSES = DATA_DIR / "agency_responses_2025.csv"
OUTPUT_TXT = Path(__file__).parent / "consultant_uncertainty_analysis.txt"


def load_and_join_data() -> pl.DataFrame:
    """Load and join all relevant data files."""
    print("Loading data files...")
    
    # Load comments with text
    df_comments = pl.read_csv(COMMENTS_RAW)
    print(f"  Loaded {len(df_comments):,} comments")
    
    # Load makeup classifications
    df_makeup = pl.read_csv(MAKEUP_DATA)
    print(f"  Loaded {len(df_makeup):,} makeup classifications")
    
    # Load agency responses
    df_responses = pl.read_csv(AGENCY_RESPONSES)
    print(f"  Loaded {len(df_responses):,} agency responses")
    
    # Join: comments -> makeup -> responses
    # First join: comments with makeup
    df = df_comments.join(df_makeup, on="comment_id", how="inner")
    
    # Second join: with responses (handle duplicate columns)
    # Select only unique columns from responses (exclude document_number and agency if they exist in df)
    response_cols = [col for col in df_responses.columns if col not in ["document_number", "agency", "comment_id"]]
    response_cols.insert(0, "comment_id")  # Ensure comment_id is first for join
    df_responses_subset = df_responses.select(["comment_id"] + response_cols[1:])
    df = df.join(df_responses_subset, on="comment_id", how="left")
    
    print(f"  Joined dataset: {len(df):,} rows")
    
    return df


def calculate_stats(df: pl.DataFrame) -> Dict:
    """Calculate statistics comparing consultants to other commenters."""
    stats = {}
    
    # Identify consultants
    consultants = df.filter(pl.col("category") == "Political Consultant/Lobbyist")
    others = df.filter(pl.col("category") != "Political Consultant/Lobbyist")
    
    stats["total_consultants"] = consultants.height
    stats["total_others"] = others.height
    
    # Response found rates
    stats["consultants_response_yes"] = consultants.filter(pl.col("response_found") == "yes").height
    stats["consultants_response_no"] = consultants.filter(pl.col("response_found") == "no").height
    stats["consultants_response_uncertain"] = consultants.filter(pl.col("response_found") == "uncertain").height
    
    stats["others_response_yes"] = others.filter(pl.col("response_found") == "yes").height
    stats["others_response_no"] = others.filter(pl.col("response_found") == "no").height
    stats["others_response_uncertain"] = others.filter(pl.col("response_found") == "uncertain").height
    
    # Agency decision rates (for comments with responses found)
    consultants_with_response = consultants.filter(pl.col("response_found") == "yes")
    others_with_response = others.filter(pl.col("response_found") == "yes")
    
    stats["consultants_accept"] = consultants_with_response.filter(pl.col("agency_decision") == "accept").height
    stats["consultants_reject"] = consultants_with_response.filter(pl.col("agency_decision") == "reject").height
    stats["consultants_decision_uncertain"] = consultants_with_response.filter(pl.col("agency_decision") == "uncertain").height
    
    stats["others_accept"] = others_with_response.filter(pl.col("agency_decision") == "accept").height
    stats["others_reject"] = others_with_response.filter(pl.col("agency_decision") == "reject").height
    stats["others_decision_uncertain"] = others_with_response.filter(pl.col("agency_decision") == "uncertain").height
    
    # Overall decision rates (including no response as uncertain)
    stats["consultants_overall_accept"] = consultants.filter(pl.col("agency_decision") == "accept").height
    stats["consultants_overall_reject"] = consultants.filter(pl.col("agency_decision") == "reject").height
    stats["consultants_overall_uncertain"] = consultants.filter(pl.col("agency_decision") == "uncertain").height
    
    stats["others_overall_accept"] = others.filter(pl.col("agency_decision") == "accept").height
    stats["others_overall_reject"] = others.filter(pl.col("agency_decision") == "reject").height
    stats["others_overall_uncertain"] = others.filter(pl.col("agency_decision") == "uncertain").height
    
    # Comment length statistics
    consultants = consultants.with_columns(
        pl.col("comment_text").fill_null("").str.len_chars().alias("comment_length")
    )
    others = others.with_columns(
        pl.col("comment_text").fill_null("").str.len_chars().alias("comment_length")
    )
    
    if "comment_length" in consultants.columns:
        stats["consultants_avg_length"] = consultants["comment_length"].mean()
        stats["consultants_median_length"] = consultants["comment_length"].median()
    else:
        stats["consultants_avg_length"] = 0
        stats["consultants_median_length"] = 0
    
    if "comment_length" in others.columns:
        stats["others_avg_length"] = others["comment_length"].mean()
        stats["others_median_length"] = others["comment_length"].median()
    else:
        stats["others_avg_length"] = 0
        stats["others_median_length"] = 0
    
    # Attachment rates
    stats["consultants_with_attachments"] = consultants.filter(pl.col("has_attachments") == True).height
    stats["others_with_attachments"] = others.filter(pl.col("has_attachments") == True).height
    
    # Agency distribution
    if "agency" in consultants.columns:
        stats["consultants_unique_agencies"] = consultants["agency"].n_unique()
    else:
        stats["consultants_unique_agencies"] = 0
    
    if "agency" in others.columns:
        stats["others_unique_agencies"] = others["agency"].n_unique()
    else:
        stats["others_unique_agencies"] = 0
    
    # Document distribution
    if "document_number" in consultants.columns:
        stats["consultants_unique_docs"] = consultants["document_number"].n_unique()
    else:
        stats["consultants_unique_docs"] = 0
    
    if "document_number" in others.columns:
        stats["others_unique_docs"] = others["document_number"].n_unique()
    else:
        stats["others_unique_docs"] = 0
    
    return stats


def format_stats_report(stats: Dict) -> str:
    """Format statistics into a readable report."""
    lines = []
    lines.append("=" * 80)
    lines.append("POLITICAL CONSULTANT/LOBBYIST UNCERTAINTY ANALYSIS")
    lines.append("=" * 80)
    lines.append("")
    
    # Basic counts
    lines.append("BASIC COUNTS")
    lines.append("-" * 80)
    lines.append(f"Total Consultants/Lobbyists: {stats['total_consultants']:,}")
    lines.append(f"Total Other Commenters: {stats['total_others']:,}")
    lines.append("")
    
    # Response found rates
    lines.append("RESPONSE FOUND RATES")
    lines.append("-" * 80)
    consultants_total = stats["total_consultants"]
    others_total = stats["total_others"]
    
    if consultants_total > 0:
        lines.append("Consultants:")
        lines.append(f"  Yes: {stats['consultants_response_yes']:,} ({100*stats['consultants_response_yes']/consultants_total:.1f}%)")
        lines.append(f"  No: {stats['consultants_response_no']:,} ({100*stats['consultants_response_no']/consultants_total:.1f}%)")
        lines.append(f"  Uncertain: {stats['consultants_response_uncertain']:,} ({100*stats['consultants_response_uncertain']/consultants_total:.1f}%)")
    
    if others_total > 0:
        lines.append("Others:")
        lines.append(f"  Yes: {stats['others_response_yes']:,} ({100*stats['others_response_yes']/others_total:.1f}%)")
        lines.append(f"  No: {stats['others_response_no']:,} ({100*stats['others_response_no']/others_total:.1f}%)")
        lines.append(f"  Uncertain: {stats['others_response_uncertain']:,} ({100*stats['others_response_uncertain']/others_total:.1f}%)")
    lines.append("")
    
    # Agency decision rates (for responses found)
    consultants_with_response = stats["consultants_response_yes"]
    others_with_response = stats["others_response_yes"]
    
    lines.append("AGENCY DECISION RATES (for comments with responses found)")
    lines.append("-" * 80)
    if consultants_with_response > 0:
        lines.append("Consultants:")
        lines.append(f"  Accept: {stats['consultants_accept']:,} ({100*stats['consultants_accept']/consultants_with_response:.1f}%)")
        lines.append(f"  Reject: {stats['consultants_reject']:,} ({100*stats['consultants_reject']/consultants_with_response:.1f}%)")
        lines.append(f"  Uncertain: {stats['consultants_decision_uncertain']:,} ({100*stats['consultants_decision_uncertain']/consultants_with_response:.1f}%)")
    
    if others_with_response > 0:
        lines.append("Others:")
        lines.append(f"  Accept: {stats['others_accept']:,} ({100*stats['others_accept']/others_with_response:.1f}%)")
        lines.append(f"  Reject: {stats['others_reject']:,} ({100*stats['others_reject']/others_with_response:.1f}%)")
        lines.append(f"  Uncertain: {stats['others_decision_uncertain']:,} ({100*stats['others_decision_uncertain']/others_with_response:.1f}%)")
    lines.append("")
    
    # Overall decision rates
    lines.append("OVERALL DECISION RATES (all comments)")
    lines.append("-" * 80)
    if consultants_total > 0:
        lines.append("Consultants:")
        lines.append(f"  Accept: {stats['consultants_overall_accept']:,} ({100*stats['consultants_overall_accept']/consultants_total:.1f}%)")
        lines.append(f"  Reject: {stats['consultants_overall_reject']:,} ({100*stats['consultants_overall_reject']/consultants_total:.1f}%)")
        lines.append(f"  Uncertain: {stats['consultants_overall_uncertain']:,} ({100*stats['consultants_overall_uncertain']/consultants_total:.1f}%)")
    
    if others_total > 0:
        lines.append("Others:")
        lines.append(f"  Accept: {stats['others_overall_accept']:,} ({100*stats['others_overall_accept']/others_total:.1f}%)")
        lines.append(f"  Reject: {stats['others_overall_reject']:,} ({100*stats['others_overall_reject']/others_total:.1f}%)")
        lines.append(f"  Uncertain: {stats['others_overall_uncertain']:,} ({100*stats['others_overall_uncertain']/others_total:.1f}%)")
    lines.append("")
    
    # Comment characteristics
    lines.append("COMMENT CHARACTERISTICS")
    lines.append("-" * 80)
    lines.append(f"Consultants - Avg Length: {stats['consultants_avg_length']:.0f} chars")
    lines.append(f"Consultants - Median Length: {stats['consultants_median_length']:.0f} chars")
    lines.append(f"Others - Avg Length: {stats['others_avg_length']:.0f} chars")
    lines.append(f"Others - Median Length: {stats['others_median_length']:.0f} chars")
    lines.append("")
    
    if consultants_total > 0:
        lines.append(f"Consultants - With Attachments: {stats['consultants_with_attachments']:,} ({100*stats['consultants_with_attachments']/consultants_total:.1f}%)")
    if others_total > 0:
        lines.append(f"Others - With Attachments: {stats['others_with_attachments']:,} ({100*stats['others_with_attachments']/others_total:.1f}%)")
    lines.append("")
    
    # Distribution
    lines.append("DISTRIBUTION")
    lines.append("-" * 80)
    lines.append(f"Consultants - Unique Agencies: {stats['consultants_unique_agencies']}")
    lines.append(f"Others - Unique Agencies: {stats['others_unique_agencies']}")
    lines.append(f"Consultants - Unique Documents: {stats['consultants_unique_docs']}")
    lines.append(f"Others - Unique Documents: {stats['others_unique_docs']}")
    lines.append("")
    
    return "\n".join(lines)


def sample_comments(df: pl.DataFrame, n_samples: int = 10) -> List[Dict]:
    """Sample comments from consultants in different categories."""
    consultants = df.filter(pl.col("category") == "Political Consultant/Lobbyist")
    
    samples = []
    
    # Category 1: Yes response + Yes accepted
    cat1 = consultants.filter(
        (pl.col("response_found") == "yes") & 
        (pl.col("agency_decision") == "accept")
    )
    if cat1.height > 0:
        sample_size = min(n_samples, cat1.height)
        sampled = cat1.sample(sample_size, seed=42)
        samples.append({
            "category": "Yes Response + Accept",
            "comments": sampled.to_dicts()
        })
    
    # Category 2: Yes response + No accepted (reject)
    cat2 = consultants.filter(
        (pl.col("response_found") == "yes") & 
        (pl.col("agency_decision") == "reject")
    )
    if cat2.height > 0:
        sample_size = min(n_samples, cat2.height)
        sampled = cat2.sample(sample_size, seed=42)
        samples.append({
            "category": "Yes Response + Reject",
            "comments": sampled.to_dicts()
        })
    
    # Category 3: No response + Yes accepted (may not exist)
    cat3 = consultants.filter(
        (pl.col("response_found") == "no") & 
        (pl.col("agency_decision") == "accept")
    )
    if cat3.height > 0:
        sample_size = min(n_samples, cat3.height)
        sampled = cat3.sample(sample_size, seed=42)
        samples.append({
            "category": "No Response + Accept",
            "comments": sampled.to_dicts()
        })
    
    # Category 4: No response + Uncertain decision
    cat4 = consultants.filter(
        (pl.col("response_found") == "no") & 
        (pl.col("agency_decision") == "uncertain")
    )
    if cat4.height > 0:
        sample_size = min(n_samples, cat4.height)
        sampled = cat4.sample(sample_size, seed=42)
        samples.append({
            "category": "No Response + Uncertain",
            "comments": sampled.to_dicts()
        })
    
    # Category 5: Uncertain decision (response_found == "uncertain")
    cat5 = consultants.filter(pl.col("response_found") == "uncertain")
    if cat5.height > 0:
        sample_size = min(n_samples, cat5.height)
        sampled = cat5.sample(sample_size, seed=42)
        samples.append({
            "category": "Uncertain Response",
            "comments": sampled.to_dicts()
        })
    
    return samples


def format_sample_comments(samples: List[Dict]) -> str:
    """Format sampled comments into readable text."""
    lines = []
    lines.append("=" * 80)
    lines.append("SAMPLE COMMENTS BY CATEGORY")
    lines.append("=" * 80)
    lines.append("")
    
    for sample_group in samples:
        lines.append("-" * 80)
        lines.append(f"CATEGORY: {sample_group['category']}")
        lines.append(f"Total in category: {len(sample_group['comments'])}")
        lines.append("-" * 80)
        lines.append("")
        
        for i, comment in enumerate(sample_group["comments"], 1):
            lines.append(f"Sample {i}:")
            lines.append(f"  Comment ID: {comment.get('comment_id') or 'N/A'}")
            lines.append(f"  Document: {comment.get('document_number') or 'N/A'}")
            lines.append(f"  Agency: {comment.get('agency') or 'N/A'}")
            lines.append(f"  Response Found: {comment.get('response_found') or 'N/A'}")
            lines.append(f"  Agency Decision: {comment.get('agency_decision') or 'N/A'}")
            
            comment_text_val = comment.get("comment_text")
            if comment_text_val:
                comment_text = str(comment_text_val)[:500]
                if len(str(comment_text_val)) > 500:
                    comment_text += "... [truncated]"
                lines.append(f"  Comment Text: {comment_text}")
            else:
                lines.append(f"  Comment Text: N/A")
            
            response_text_val = comment.get("response_text")
            if response_text_val and str(response_text_val) != "N/A":
                response_text = str(response_text_val)[:500]
                if len(str(response_text_val)) > 500:
                    response_text += "... [truncated]"
                lines.append(f"  Response Text: {response_text}")
            
            lines.append("")
    
    return "\n".join(lines)


def main():
    """Main analysis function."""
    print("Starting consultant uncertainty analysis...")
    
    # Load data
    df = load_and_join_data()
    
    # Calculate statistics
    print("\nCalculating statistics...")
    stats = calculate_stats(df)
    
    # Generate report
    print("\nGenerating report...")
    report = format_stats_report(stats)
    
    # Sample comments
    print("\nSampling comments...")
    random.seed(42)  # For reproducibility
    samples = sample_comments(df, n_samples=10)
    samples_text = format_sample_comments(samples)
    
    # Combine and write output
    full_report = report + "\n\n" + samples_text
    
    OUTPUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write(full_report)
    
    print(f"\nAnalysis complete! Output saved to: {OUTPUT_TXT}")
    print(f"\nSummary:")
    print(f"  Total Consultants: {stats['total_consultants']:,}")
    print(f"  Consultants with Uncertain Decisions: {stats['consultants_overall_uncertain']:,} ({100*stats['consultants_overall_uncertain']/stats['total_consultants']:.1f}%)")
    print(f"  Others with Uncertain Decisions: {stats['others_overall_uncertain']:,} ({100*stats['others_overall_uncertain']/stats['total_others']:.1f}%)")


if __name__ == "__main__":
    main()

