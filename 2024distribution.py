"""
2024 Federal Register Document Comment Distribution Analysis

This script analyzes Federal Register documents from 2024 that are available for 
public inspection and examines the distribution of comments received on each document.

APIs Used:
- Federal Register API: For fetching document metadata
- Regulations.gov API: For fetching comment counts
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import time
from plotnine import (
    ggplot, aes, geom_histogram, geom_bar, theme_minimal, 
    labs, theme, element_rect, element_text, scale_fill_manual,
    geom_text, scale_y_continuous
)
import warnings
warnings.filterwarnings('ignore')


# API Configuration
REGULATIONS_API_KEY = "lFUuSOntbPfvhw5fzYfhk9Y8m8rTnVwcb4LOc7G3"
FEDERAL_REGISTER_BASE_URL = "https://www.federalregister.gov/api/v1"
REGULATIONS_GOV_BASE_URL = "https://api.regulations.gov/v4"

# Color scheme
PRIMARY_COLOR = "#2596be"
BACKGROUND_COLOR = "#808080"  # Metal grey


def fetch_federal_register_documents(year=2024):
    """
    Fetch all Federal Register documents from a given year that are open for comment.
    """
    print(f"\n📋 Fetching Federal Register documents from {year}...")
    
    all_documents = []
    page = 1
    per_page = 1000  # Max per page
    
    # First, get a count
    params = {
        'conditions[publication_date][year]': year,
        'conditions[type][]': ['RULE', 'PRORULE', 'NOTICE'],  # Documents typically open for comment
        'per_page': 1,
        'fields[]': ['document_number']
    }
    
    response = requests.get(f"{FEDERAL_REGISTER_BASE_URL}/documents.json", params=params)
    
    if response.status_code != 200:
        print(f"❌ Error fetching documents: {response.status_code}")
        return []
    
    data = response.json()
    total_count = data.get('count', 0)
    print(f"📊 Total documents found: {total_count}")
    
    # Now fetch all documents with proper pagination
    total_pages = (total_count // per_page) + 1
    
    with tqdm(total=total_count, desc="Fetching FR documents") as pbar:
        for page in range(1, total_pages + 1):
            params = {
                'conditions[publication_date][year]': year,
                'conditions[type][]': ['RULE', 'PRORULE', 'NOTICE'],
                'per_page': per_page,
                'page': page,
                'fields[]': ['document_number', 'title', 'type', 'publication_date', 
                           'agencies', 'docket_ids', 'comment_url', 'comments_close_on']
            }
            
            try:
                response = requests.get(f"{FEDERAL_REGISTER_BASE_URL}/documents.json", 
                                      params=params, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    documents = data.get('results', [])
                    all_documents.extend(documents)
                    pbar.update(len(documents))
                else:
                    print(f"⚠️  Warning: Page {page} returned status {response.status_code}")
                
                # Be respectful to the API
                time.sleep(0.5)
                
            except Exception as e:
                print(f"⚠️  Error on page {page}: {str(e)}")
                continue
    
    print(f"✅ Successfully fetched {len(all_documents)} documents")
    return all_documents


def get_comment_count_from_regulations_gov(document_number=None, docket_id=None):
    """
    Fetch comment count for a document from Regulations.gov API.
    """
    headers = {
        'X-Api-Key': REGULATIONS_API_KEY
    }
    
    # Try to find the document and get comment count
    if docket_id:
        # Clean up docket ID
        docket_id = docket_id.strip() if isinstance(docket_id, str) else None
        
        if docket_id:
            try:
                # Get docket information which includes comment counts
                response = requests.get(
                    f"{REGULATIONS_GOV_BASE_URL}/dockets/{docket_id}",
                    headers=headers,
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    # The numberOfComments field in docket data
                    comment_count = data.get('data', {}).get('attributes', {}).get('numberOfComments', 0)
                    return comment_count
                    
            except Exception as e:
                pass
    
    # If docket lookup failed, try searching by document number
    if document_number:
        try:
            params = {
                'filter[searchTerm]': document_number,
                'page[size]': 1
            }
            response = requests.get(
                f"{REGULATIONS_GOV_BASE_URL}/documents",
                headers=headers,
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                documents = data.get('data', [])
                if documents:
                    comment_count = documents[0].get('attributes', {}).get('numberOfCommentsReceived', 0)
                    return comment_count
                    
        except Exception as e:
            pass
    
    return 0


def analyze_comment_distribution(documents):
    """
    Analyze the distribution of comments across Federal Register documents.
    """
    print("\n💬 Fetching comment counts from Regulations.gov...")
    
    document_data = []
    
    with tqdm(total=len(documents), desc="Getting comment counts") as pbar:
        for doc in documents:
            doc_number = doc.get('document_number')
            docket_ids = doc.get('docket_ids', [])
            docket_id = docket_ids[0] if docket_ids else None
            
            # Get comment count
            comment_count = get_comment_count_from_regulations_gov(
                document_number=doc_number,
                docket_id=docket_id
            )
            
            document_data.append({
                'document_number': doc_number,
                'title': doc.get('title', '')[:100],  # Truncate long titles
                'type': doc.get('type', ''),
                'publication_date': doc.get('publication_date', ''),
                'docket_id': docket_id,
                'comment_count': comment_count,
                'has_comments': comment_count > 0
            })
            
            pbar.update(1)
            
            # Be very respectful to the API - rate limiting
            time.sleep(0.2)
    
    df = pd.DataFrame(document_data)
    return df


def generate_analysis(df):
    """
    Generate statistical analysis of the comment distribution.
    """
    print("\n" + "="*80)
    print("📊 FEDERAL REGISTER 2024 COMMENT DISTRIBUTION ANALYSIS")
    print("="*80)
    
    total_docs = len(df)
    docs_with_comments = df['has_comments'].sum()
    docs_without_comments = total_docs - docs_with_comments
    
    print(f"\n📄 Total Documents Analyzed: {total_docs:,}")
    print(f"💬 Documents with Comments: {docs_with_comments:,} ({docs_with_comments/total_docs*100:.1f}%)")
    print(f"🚫 Documents without Comments: {docs_without_comments:,} ({docs_without_comments/total_docs*100:.1f}%)")
    
    print("\n" + "-"*80)
    print("📈 COMMENT STATISTICS")
    print("-"*80)
    
    if docs_with_comments > 0:
        df_with_comments = df[df['comment_count'] > 0]
        
        print(f"Total Comments: {df['comment_count'].sum():,}")
        print(f"Average Comments per Document (all): {df['comment_count'].mean():.2f}")
        print(f"Average Comments per Document (with comments): {df_with_comments['comment_count'].mean():.2f}")
        print(f"Median Comments: {df['comment_count'].median():.0f}")
        print(f"Max Comments on Single Document: {df['comment_count'].max():,}")
        print(f"Min Comments (excluding 0): {df_with_comments['comment_count'].min():,}")
        
        print("\n📊 DISTRIBUTION BREAKDOWN")
        print("-"*80)
        
        # Create bins for analysis
        bins = [0, 1, 10, 50, 100, 500, 1000, 5000, float('inf')]
        labels = ['0', '1-10', '11-50', '51-100', '101-500', '501-1000', '1001-5000', '5000+']
        df['comment_range'] = pd.cut(df['comment_count'], bins=bins, labels=labels, right=False)
        
        range_counts = df['comment_range'].value_counts().sort_index()
        for range_label, count in range_counts.items():
            percentage = (count / total_docs) * 100
            print(f"{range_label:>12} comments: {count:>6,} documents ({percentage:>5.1f}%)")
        
        print("\n🏆 TOP 10 MOST COMMENTED DOCUMENTS")
        print("-"*80)
        top_10 = df.nlargest(10, 'comment_count')[['document_number', 'title', 'comment_count']]
        for idx, row in top_10.iterrows():
            print(f"{row['comment_count']:>8,} | {row['document_number']} | {row['title'][:60]}")
    
    print("\n" + "="*80 + "\n")
    
    return df


def create_visualization(df):
    """
    Create a visualization of the comment distribution using plotnine.
    """
    print("🎨 Creating visualization...")
    
    # Prepare data for visualization
    # Create meaningful bins
    df_plot = df.copy()
    
    # For better visualization, create log-scale bins
    bins = [0, 1, 10, 50, 100, 500, 1000, 5000, float('inf')]
    labels = ['0', '1-10', '11-50', '51-100', '101-500', '501-1K', '1K-5K', '5K+']
    df_plot['comment_range'] = pd.cut(df_plot['comment_count'], bins=bins, labels=labels, right=False)
    
    # Count documents in each range
    range_counts = df_plot['comment_range'].value_counts().reset_index()
    range_counts.columns = ['comment_range', 'count']
    range_counts = range_counts.sort_values('comment_range')
    
    # Create the plot
    plot = (
        ggplot(range_counts, aes(x='comment_range', y='count')) +
        geom_bar(stat='identity', fill=PRIMARY_COLOR, alpha=0.9) +
        geom_text(aes(label='count'), va='bottom', size=10, color='white', 
                 format_string='{:.0f}', nudge_y=len(df)*0.01) +
        labs(
            title='Distribution of Comments on 2024 Federal Register Documents',
            subtitle='Analysis of documents available for public inspection',
            x='Number of Comments',
            y='Number of Documents'
        ) +
        theme_minimal() +
        theme(
            plot_background=element_rect(fill=BACKGROUND_COLOR),
            panel_background=element_rect(fill=BACKGROUND_COLOR),
            figure_size=(12, 7),
            plot_title=element_text(size=16, weight='bold', color='white'),
            plot_subtitle=element_text(size=11, color='white'),
            axis_title=element_text(size=12, weight='bold', color='white'),
            axis_text=element_text(size=10, color='white'),
            axis_text_x=element_text(rotation=45, ha='right'),
            panel_grid_major=element_rect(color='#a0a0a0', alpha=0.3),
            panel_grid_minor=element_rect(color='#a0a0a0', alpha=0.1)
        )
    )
    
    # Save the plot
    output_file = '2024_federal_register_comment_distribution.png'
    plot.save(output_file, dpi=300, verbose=False)
    print(f"✅ Visualization saved to: {output_file}")
    
    return plot


def main():
    """
    Main execution function.
    """
    print("=" * 80)
    print("🇺🇸 FEDERAL REGISTER 2024 COMMENT DISTRIBUTION ANALYZER")
    print("=" * 80)
    print("📅 Analysis Year: 2024")
    print("🔑 Using Regulations.gov API Key")
    print("=" * 80)
    
    # Step 1: Fetch Federal Register documents
    documents = fetch_federal_register_documents(year=2024)
    
    if not documents:
        print("❌ No documents found. Exiting.")
        return
    
    # Step 2: Analyze comment distribution
    df = analyze_comment_distribution(documents)
    
    # Step 3: Generate analysis
    df = generate_analysis(df)
    
    # Step 4: Create visualization
    create_visualization(df)
    
    # Step 5: Save data
    output_csv = '2024_federal_register_analysis.csv'
    df.to_csv(output_csv, index=False)
    print(f"💾 Data saved to: {output_csv}")
    
    print("\n✨ Analysis complete!")


if __name__ == "__main__":
    main()
