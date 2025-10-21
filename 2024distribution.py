"""
Federal Register 2024 Comment Distribution Analysis

This script analyzes the distribution of comments on documents available for 
inspection in the Federal Register during 2024.

APIs Used:
- Federal Register API: To fetch documents from 2024
- Regulations.gov API: To get comment counts for each document
"""

import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
from plotnine import (
    ggplot, aes, geom_histogram, geom_bar, theme, element_rect, 
    element_text, labs, scale_fill_manual, theme_minimal, geom_vline,
    element_blank, stat_bin
)
import warnings
warnings.filterwarnings('ignore')


# API Configuration
FEDERAL_REGISTER_BASE_URL = "https://www.federalregister.gov/api/v1"
REGULATIONS_GOV_BASE_URL = "https://api.regulations.gov/v4"
REGULATIONS_GOV_API_KEY = "lFUuSOntbPfvhw5fzYfhk9Y8m8rTnVwcb4LOc7G3"


def fetch_federal_register_documents(year=2024):
    """
    Fetch all documents from the Federal Register for a given year.
    
    The Federal Register API allows filtering by publication date and 
    provides document metadata including document numbers which can be 
    used to query Regulations.gov.
    """
    print(f"\n{'='*60}")
    print(f"Fetching Federal Register documents from {year}")
    print(f"{'='*60}\n")
    
    documents = []
    page = 1
    per_page = 1000  # Maximum allowed by API
    
    # First, get the total count
    params = {
        'conditions[publication_date][year]': year,
        'per_page': 1,
        'page': 1
    }
    
    response = requests.get(f"{FEDERAL_REGISTER_BASE_URL}/documents.json", params=params)
    
    if response.status_code != 200:
        print(f"Error fetching documents: {response.status_code}")
        return []
    
    data = response.json()
    total_count = data.get('count', 0)
    print(f"Total documents found: {total_count}")
    
    # Now fetch all documents with pagination
    total_pages = (total_count // per_page) + (1 if total_count % per_page > 0 else 0)
    
    with tqdm(total=total_pages, desc="Fetching FR documents", ncols=80) as pbar:
        while True:
            params = {
                'conditions[publication_date][year]': year,
                'per_page': per_page,
                'page': page,
                'fields[]': ['document_number', 'title', 'publication_date', 
                            'type', 'agencies', 'docket_id']
            }
            
            response = requests.get(f"{FEDERAL_REGISTER_BASE_URL}/documents.json", params=params)
            
            if response.status_code != 200:
                print(f"Error on page {page}: {response.status_code}")
                break
            
            data = response.json()
            results = data.get('results', [])
            
            if not results:
                break
            
            documents.extend(results)
            pbar.update(1)
            
            # Check if we've reached the last page
            if len(results) < per_page:
                break
            
            page += 1
            time.sleep(0.1)  # Be respectful to the API
    
    print(f"\nSuccessfully fetched {len(documents)} documents\n")
    return documents


def get_comment_count(document_number, docket_id=None):
    """
    Get the comment count for a document using Regulations.gov API.
    
    The Regulations.gov API can search for documents by document number
    or docket ID and return comment counts.
    """
    headers = {
        'X-Api-Key': REGULATIONS_GOV_API_KEY
    }
    
    # Try to find the document on Regulations.gov
    # First try by document number
    if document_number:
        try:
            # Search for the document
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
                if data.get('data') and len(data['data']) > 0:
                    doc = data['data'][0]
                    # Get comment count from attributes
                    attributes = doc.get('attributes', {})
                    comment_count = attributes.get('numberOfCommentsReceived', 0)
                    if comment_count is None:
                        comment_count = 0
                    return comment_count
            
            time.sleep(0.25)  # Rate limiting
            
        except Exception as e:
            pass
    
    # If document number didn't work, try docket ID
    if docket_id:
        try:
            params = {
                'filter[searchTerm]': docket_id,
                'page[size]': 1
            }
            
            response = requests.get(
                f"{REGULATIONS_GOV_BASE_URL}/dockets",
                headers=headers,
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('data') and len(data['data']) > 0:
                    docket = data['data'][0]
                    # Some dockets have comment counts
                    attributes = docket.get('attributes', {})
                    return attributes.get('numberOfComments', 0) or 0
            
            time.sleep(0.25)
            
        except Exception as e:
            pass
    
    return 0


def analyze_comment_distribution(documents):
    """
    Analyze the distribution of comments across Federal Register documents.
    """
    print(f"\n{'='*60}")
    print("Fetching comment counts from Regulations.gov")
    print(f"{'='*60}\n")
    
    data_list = []
    
    # Get comment counts for each document
    for doc in tqdm(documents, desc="Getting comment counts", ncols=80):
        document_number = doc.get('document_number')
        docket_id = doc.get('docket_id')
        
        comment_count = get_comment_count(document_number, docket_id)
        
        data_list.append({
            'document_number': document_number,
            'title': doc.get('title', 'N/A'),
            'publication_date': doc.get('publication_date'),
            'type': doc.get('type'),
            'docket_id': docket_id,
            'comment_count': comment_count
        })
    
    df = pd.DataFrame(data_list)
    
    # Print analysis
    print(f"\n{'='*60}")
    print("ANALYSIS: Federal Register 2024 Comment Distribution")
    print(f"{'='*60}\n")
    
    print(f"Total documents analyzed: {len(df)}")
    print(f"Documents with comments: {len(df[df['comment_count'] > 0])}")
    print(f"Documents without comments: {len(df[df['comment_count'] == 0])}")
    print(f"\nComment Statistics:")
    print(f"  Mean comments per document: {df['comment_count'].mean():.2f}")
    print(f"  Median comments per document: {df['comment_count'].median():.0f}")
    print(f"  Max comments on a document: {df['comment_count'].max():.0f}")
    print(f"  Total comments across all documents: {df['comment_count'].sum():.0f}")
    
    # Documents with most comments
    print(f"\n{'='*60}")
    print("Top 10 Documents by Comment Count:")
    print(f"{'='*60}\n")
    top_docs = df.nlargest(10, 'comment_count')[['document_number', 'comment_count', 'title']]
    for idx, row in top_docs.iterrows():
        print(f"{row['comment_count']:>6.0f} comments - {row['document_number']}")
        print(f"       {row['title'][:80]}...")
        print()
    
    # Distribution breakdown
    print(f"{'='*60}")
    print("Comment Count Distribution:")
    print(f"{'='*60}\n")
    
    bins = [0, 1, 10, 50, 100, 500, 1000, float('inf')]
    labels = ['0', '1-9', '10-49', '50-99', '100-499', '500-999', '1000+']
    df['comment_bin'] = pd.cut(df['comment_count'], bins=bins, labels=labels, right=False)
    
    distribution = df['comment_bin'].value_counts().sort_index()
    for label, count in distribution.items():
        percentage = (count / len(df)) * 100
        print(f"  {label:>10} comments: {count:>6} documents ({percentage:>5.2f}%)")
    
    print(f"\n{'='*60}\n")
    
    # Save data
    df.to_csv('federal_register_2024_comments.csv', index=False)
    print("Data saved to: federal_register_2024_comments.csv\n")
    
    return df


def create_visualization(df):
    """
    Create a visualization of the comment distribution using plotnine.
    """
    print("Creating visualization...")
    
    # Filter out zero comments for better visualization
    df_with_comments = df[df['comment_count'] > 0].copy()
    
    if len(df_with_comments) == 0:
        print("No documents with comments found. Creating visualization with all data...")
        df_with_comments = df.copy()
    
    # Create bins for the histogram
    bins = [0, 1, 5, 10, 25, 50, 100, 250, 500, 1000, 5000, 10000, 50000]
    df_with_comments['comment_bin'] = pd.cut(
        df_with_comments['comment_count'], 
        bins=bins,
        include_lowest=True
    )
    
    # Count documents in each bin
    bin_counts = df_with_comments['comment_bin'].value_counts().sort_index()
    bin_df = pd.DataFrame({
        'bin': [str(b) for b in bin_counts.index],
        'count': bin_counts.values
    })
    
    # Create the plot
    plot = (
        ggplot(df_with_comments, aes(x='comment_count')) +
        geom_histogram(fill='#2596be', color='#1a7a9e', bins=50, alpha=0.9) +
        labs(
            title='Distribution of Comments on Federal Register Documents (2024)',
            subtitle=f'Total Documents: {len(df):,} | Documents with Comments: {len(df_with_comments):,}',
            x='Number of Comments',
            y='Number of Documents'
        ) +
        theme_minimal() +
        theme(
            plot_background=element_rect(fill='#8c8c8c'),
            panel_background=element_rect(fill='#8c8c8c'),
            panel_grid_major=element_rect(color='#a0a0a0', size=0.3),
            panel_grid_minor=element_blank(),
            plot_title=element_text(size=14, weight='bold', color='#ffffff'),
            plot_subtitle=element_text(size=10, color='#f0f0f0'),
            axis_title=element_text(size=11, color='#ffffff'),
            axis_text=element_text(size=9, color='#f0f0f0'),
            figure_size=(12, 6)
        )
    )
    
    # Save the plot
    plot.save('federal_register_2024_comment_distribution.png', dpi=300)
    print("Visualization saved to: federal_register_2024_comment_distribution.png")
    
    # Create a second plot with log scale for better visibility
    plot_log = (
        ggplot(df_with_comments, aes(x='comment_count')) +
        geom_histogram(fill='#2596be', color='#1a7a9e', bins=50, alpha=0.9) +
        labs(
            title='Distribution of Comments on Federal Register Documents (2024) - Log Scale',
            subtitle=f'Total Documents: {len(df):,} | Documents with Comments: {len(df_with_comments):,}',
            x='Number of Comments (log scale)',
            y='Number of Documents (log scale)'
        ) +
        theme_minimal() +
        theme(
            plot_background=element_rect(fill='#8c8c8c'),
            panel_background=element_rect(fill='#8c8c8c'),
            panel_grid_major=element_rect(color='#a0a0a0', size=0.3),
            panel_grid_minor=element_blank(),
            plot_title=element_text(size=14, weight='bold', color='#ffffff'),
            plot_subtitle=element_text(size=10, color='#f0f0f0'),
            axis_title=element_text(size=11, color='#ffffff'),
            axis_text=element_text(size=9, color='#f0f0f0'),
            figure_size=(12, 6)
        )
    )
    
    plot_log.save('federal_register_2024_comment_distribution_log.png', dpi=300)
    print("Log scale visualization saved to: federal_register_2024_comment_distribution_log.png")


def main():
    """
    Main execution function.
    """
    print("\n" + "="*60)
    print("Federal Register 2024 Comment Distribution Analysis")
    print("="*60)
    
    # Fetch documents from Federal Register
    documents = fetch_federal_register_documents(year=2024)
    
    if not documents:
        print("No documents fetched. Exiting.")
        return
    
    # Analyze comment distribution
    df = analyze_comment_distribution(documents)
    
    # Create visualization
    create_visualization(df)
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
