#!/usr/bin/env python3
"""
BPE Visualization Dashboard

This script creates a comprehensive dashboard that combines all BPE visualizations
into a single, polished display for presentation purposes.
"""

import os
import argparse
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import seaborn as sns
from BPEexample import BPETokenizer
from language_modeling_viz_fixed import visualize_language_modeling
from compare_tokenization import (
    analyze_tokenization, 
    plot_token_distribution, 
    plot_token_counts,
    visualize_tokenization_comparison
)
from visualize_bpe import visualize_tokenization

def create_dashboard(tokenizer, text, output_file=None):
    """
    Create a comprehensive dashboard with all BPE visualizations.
    
    Args:
        tokenizer: Trained BPETokenizer instance
        text: Sample text for analysis
        output_file: Path to save the dashboard (if None, display instead)
    """
    # Set the style for all plots
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_style("whitegrid")
    
    # Create a large figure for the dashboard
    fig = plt.figure(figsize=(24, 34), facecolor='white')
    
    # Create a grid layout for the dashboard
    gs = GridSpec(5, 2, height_ratios=[1, 1.5, 1.5, 1, 1.5], figure=fig, hspace=0.4, wspace=0.3)
    
    # Add a title to the dashboard
    fig.suptitle("Byte Pair Encoding (BPE) Visualization Dashboard", 
                fontsize=28, fontweight='bold', y=0.98)
    
    # Add subtitle with explanation
    fig.text(0.5, 0.955, 
             "A comprehensive view of BPE tokenization and its advantages for language modeling",
             ha='center', fontsize=16, style='italic', color='#555555')
    
    # Section 1: Original Text Sample (top left)
    ax_text = fig.add_subplot(gs[0, 0])
    ax_text.axis('off')
    
    # Format the text sample with a nice box
    text_sample = text[:500] + "..." if len(text) > 500 else text
    ax_text.text(0.5, 0.5, f'Sample Text:\n\n"{text_sample}"', 
               ha='center', va='center', fontsize=12, 
               color='#2C3E50',
               bbox=dict(facecolor='#f8f9fa', alpha=0.8, edgecolor='lightgray',
                         boxstyle='round,pad=1.0'),
               wrap=True)
    
    # Add a section title
    ax_text.text(0.5, 1.05, "Text Sample", 
               ha='center', va='bottom', fontsize=18, fontweight='bold',
               transform=ax_text.transAxes)
    
    # Section 2: BPE Vocabulary Stats (top right)
    ax_stats = fig.add_subplot(gs[0, 1])
    ax_stats.axis('off')
    
    # Get vocabulary statistics
    vocab_size = len(tokenizer.vocab)
    char_vocab_size = len(set(''.join(tokenizer.vocab.keys())))
    avg_token_len = np.mean([len(token) for token in tokenizer.vocab.keys()])
    
    # Format the statistics with a nice layout
    stats_text = (
        f"BPE Vocabulary Statistics:\n\n"
        f"• Vocabulary Size: {vocab_size} tokens\n"
        f"• Character Vocabulary: {char_vocab_size} unique characters\n"
        f"• Average Token Length: {avg_token_len:.2f} characters\n"
        f"• Most Common Tokens: {', '.join(list(tokenizer.vocab.keys())[:5])}\n"
    )
    
    ax_stats.text(0.5, 0.5, stats_text, 
                ha='center', va='center', fontsize=14, 
                color='#2C3E50',
                bbox=dict(facecolor='#f8f9fa', alpha=0.8, edgecolor='lightgray',
                          boxstyle='round,pad=1.0'))
    
    # Add a section title
    ax_stats.text(0.5, 1.05, "Vocabulary Statistics", 
                ha='center', va='bottom', fontsize=18, fontweight='bold',
                transform=ax_stats.transAxes)
    
    # Section 3: Tokenization Comparison (middle left)
    # We'll create this in a separate function and add it to the subplot
    ax_tokens = fig.add_subplot(gs[1, :])
    ax_tokens.axis('off')
    ax_tokens.text(0.5, 1.05, "Tokenization Approaches Comparison", 
                 ha='center', va='bottom', fontsize=18, fontweight='bold',
                 transform=ax_tokens.transAxes)
    
    # Create a temporary file for the tokenization comparison
    temp_token_file = "_temp_tokenization.png"
    visualize_tokenization_comparison(tokenizer, temp_token_file)
    
    # Load the image and display it in the subplot
    token_img = plt.imread(temp_token_file)
    ax_tokens.imshow(token_img)
    ax_tokens.axis('off')
    
    # Section 4: Language Modeling Visualization (bottom left)
    ax_lm = fig.add_subplot(gs[2, 0])
    ax_lm.axis('off')
    ax_lm.text(0.5, 1.05, "Language Modeling with BPE", 
             ha='center', va='bottom', fontsize=18, fontweight='bold',
             transform=ax_lm.transAxes)
    
    # Create a temporary file for the language modeling visualization
    temp_lm_file = "_temp_lm.png"
    visualize_language_modeling(temp_lm_file)
    
    # Load the image and display it in the subplot
    lm_img = plt.imread(temp_lm_file)
    ax_lm.imshow(lm_img)
    ax_lm.axis('off')
    
    # Section 5: Token Distribution (bottom right)
    ax_dist = fig.add_subplot(gs[2, 1])
    ax_dist.axis('off')
    ax_dist.text(0.5, 1.05, "Token Distribution", 
               ha='center', va='bottom', fontsize=18, fontweight='bold',
               transform=ax_dist.transAxes)
    
    # Analyze tokenization
    analysis = analyze_tokenization(text, tokenizer)
    
    # Create a temporary file for the token distribution
    temp_dist_file = "_temp_distribution.png"
    plot_token_distribution(analysis, temp_dist_file)
    
    # Load the image and display it in the subplot
    dist_img = plt.imread(temp_dist_file)
    ax_dist.imshow(dist_img)
    ax_dist.axis('off')
    
    # Section 6: Token Counts Comparison (bottom)
    ax_counts = fig.add_subplot(gs[3, :])
    ax_counts.axis('off')
    ax_counts.text(0.5, 1.05, "Token Counts Comparison", 
                 ha='center', va='bottom', fontsize=18, fontweight='bold',
                 transform=ax_counts.transAxes)
    
    # Create a temporary file for the token counts
    temp_counts_file = "_temp_counts.png"
    plot_token_counts(analysis, temp_counts_file)
    
    # Load the image and display it in the subplot
    counts_img = plt.imread(temp_counts_file)
    ax_counts.imshow(counts_img)
    ax_counts.axis('off')
    
    # Add a footer with explanation
    footer_text = (
        "This dashboard visualizes how Byte Pair Encoding (BPE) tokenization works and its advantages for language modeling. "
        "BPE creates a vocabulary of subword units by iteratively merging the most frequent pairs of characters or character sequences. "
        "This approach balances the trade-offs between character-level tokenization (small vocabulary, long sequences) and "
        "word-level tokenization (large vocabulary, out-of-vocabulary issues)."
    )
    
    fig.text(0.5, 0.01, footer_text, ha='center', va='bottom', 
            fontsize=12, style='italic', color='#666666', 
            bbox=dict(facecolor='white', alpha=0.8, edgecolor=None),
            wrap=True)
    
    # Add a border around the entire figure
    fig.patch.set_linewidth(2)
    fig.patch.set_edgecolor('lightgray')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.02, 1, 0.94])  # Leave room for title and footer
    
    # Save or display the figure
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.5, facecolor='white')
        print(f"Dashboard saved to {output_file}")
    else:
        plt.show()
    
    # Section 7: BPE Process Visualization
    ax_process = fig.add_subplot(gs[4, :])
    ax_process.axis('off')
    ax_process.text(0.5, 1.05, "BPE Tokenization Process", 
                 ha='center', va='bottom', fontsize=18, fontweight='bold',
                 transform=ax_process.transAxes)
    
    # Create a temporary file for the BPE process visualization
    temp_process_file = "_temp_process.png"
    sample_text = "language model" if len(text) > 30 else text[:30]
    visualize_tokenization(sample_text, tokenizer, temp_process_file)
    
    # Load the image and display it in the subplot
    process_img = plt.imread(temp_process_file)
    ax_process.imshow(process_img)
    ax_process.axis('off')
    
    # Clean up temporary files
    for temp_file in [temp_token_file, temp_lm_file, temp_dist_file, temp_counts_file, temp_process_file]:
        if os.path.exists(temp_file):
            os.remove(temp_file)


def main():
    parser = argparse.ArgumentParser(description="Create a BPE visualization dashboard")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to the trained BPE model")
    parser.add_argument("--text-file", type=str, required=True,
                        help="Path to the text file to analyze")
    parser.add_argument("--output", type=str, default="bpe_dashboard.png",
                        help="Output file for the dashboard")
    args = parser.parse_args()
    
    # Load the tokenizer
    tokenizer = BPETokenizer.load(args.model)
    
    # Load the text
    with open(args.text_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Create the dashboard
    create_dashboard(tokenizer, text, args.output)


if __name__ == "__main__":
    main()
