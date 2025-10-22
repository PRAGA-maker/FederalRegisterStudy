#!/usr/bin/env python3
"""
Simple script to plot the distribution of comment_count values
in the federal_register_2024_comments.csv file
"""

import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Read the CSV file
    df = pd.read_csv('federal_register_2024_comments.csv')
    comment_counts = df['comment_count']
    
    # Get value counts
    value_counts = comment_counts.value_counts().sort_index()
    
    # Create two plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Regular scale plot
    ax1.plot(value_counts.index, value_counts.values, 'k-', linewidth=1)
    ax1.set_xlabel('Number of Comments')
    ax1.set_ylabel('Number of Records')
    ax1.set_title('Comment Count Distribution (Linear Scale)')
    ax1.grid(True, alpha=0.3)
    
    # Log scale plot
    ax2.plot(value_counts.index, value_counts.values, 'k-', linewidth=1)
    ax2.set_xlabel('Number of Comments')
    ax2.set_ylabel('Number of Records (Log Scale)')
    ax2.set_title('Comment Count Distribution (Log Scale)')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comment_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
