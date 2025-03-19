#!/usr/bin/env python3
"""
Compare different tokenization approaches: character-level, word-level, and BPE.
This script demonstrates the differences between these tokenization methods.
"""

import argparse
import json
import re
from collections import Counter
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from BPEexample import BPETokenizer, load_corpus_from_file


def character_tokenize(text: str) -> List[str]:
    """Tokenize text at the character level."""
    return list(text)


def word_tokenize(text: str) -> List[str]:
    """Simple word-level tokenization."""
    # Split on whitespace and keep punctuation separate
    return re.findall(r'\b\w+\b|[^\w\s]', text.lower())


def bpe_tokenize(text: str, tokenizer: BPETokenizer) -> List[str]:
    """Tokenize text using BPE."""
    token_ids = tokenizer.encode(text)
    return [tokenizer.vocab_reversed.get(idx, "<UNK>") for idx in token_ids]


def analyze_tokenization(text: str, tokenizer: BPETokenizer) -> Dict:
    """Analyze different tokenization approaches."""
    char_tokens = character_tokenize(text)
    word_tokens = word_tokenize(text)
    bpe_tokens = bpe_tokenize(text, tokenizer)
    
    return {
        "text_length": len(text),
        "char_tokens": {
            "count": len(char_tokens),
            "unique": len(set(char_tokens)),
            "tokens": char_tokens[:20],  # First 20 tokens
            "most_common": Counter(char_tokens).most_common(10)
        },
        "word_tokens": {
            "count": len(word_tokens),
            "unique": len(set(word_tokens)),
            "tokens": word_tokens[:20],
            "most_common": Counter(word_tokens).most_common(10)
        },
        "bpe_tokens": {
            "count": len(bpe_tokens),
            "unique": len(set(bpe_tokens)),
            "tokens": bpe_tokens[:20],
            "most_common": Counter(bpe_tokens).most_common(10)
        }
    }


def plot_token_distribution(analysis: Dict, output_file: str = None):
    """Plot token distribution for different tokenization methods."""
    # Use a clean, modern style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Use fixed dimensions for consistent alignment with a clean white background
    fig, axs = plt.subplots(3, 1, figsize=(16, 18), facecolor='white')
    
    # Add a main title to the figure
    fig.suptitle("Token Distribution Comparison", 
                fontsize=20, fontweight='bold', y=0.98, color='#2C3E50')
    
    # Character tokens
    char_tokens = [token for token, _ in analysis["char_tokens"]["most_common"]]
    char_counts = [count for _, count in analysis["char_tokens"]["most_common"]]
    
    # Create bar chart with improved styling
    bars = axs[0].bar(char_tokens, char_counts, 
                     color='#5DA5DA', alpha=0.9, 
                     edgecolor='#2C3E50', linewidth=1.0)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        axs[0].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{height:.0f}', ha='center', va='bottom', 
                   fontsize=10, fontweight='bold', color='#2C3E50')
    
    axs[0].set_title('Character Token Distribution', 
                    fontsize=16, fontweight='bold', color='#2C3E50', pad=15)
    axs[0].set_ylabel('Frequency', fontsize=14, fontweight='bold', color='#2C3E50')
    axs[0].tick_params(axis='both', which='major', labelsize=12)
    axs[0].set_ylim(0, max(char_counts) * 1.15)  # Add space for labels
    axs[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Word tokens
    word_tokens = [token for token, _ in analysis["word_tokens"]["most_common"]]
    word_counts = [count for _, count in analysis["word_tokens"]["most_common"]]
    
    # Create bar chart with improved styling
    bars = axs[1].bar(word_tokens, word_counts, 
                     color='#60BD68', alpha=0.9, 
                     edgecolor='#2C3E50', linewidth=1.0)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        axs[1].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{height:.0f}', ha='center', va='bottom', 
                   fontsize=10, fontweight='bold', color='#2C3E50')
    
    axs[1].set_title('Word Token Distribution', 
                    fontsize=16, fontweight='bold', color='#2C3E50', pad=15)
    axs[1].set_ylabel('Frequency', fontsize=14, fontweight='bold', color='#2C3E50')
    axs[1].tick_params(axis='both', which='major', labelsize=12)
    axs[1].set_ylim(0, max(word_counts) * 1.15)  # Add space for labels
    axs[1].grid(axis='y', linestyle='--', alpha=0.7)
    
    # BPE tokens
    bpe_tokens = [token for token, _ in analysis["bpe_tokens"]["most_common"]]
    bpe_counts = [count for _, count in analysis["bpe_tokens"]["most_common"]]
    
    # Create bar chart with improved styling
    bars = axs[2].bar(bpe_tokens, bpe_counts, 
                     color='#F15854', alpha=0.9, 
                     edgecolor='#2C3E50', linewidth=1.0)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        axs[2].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{height:.0f}', ha='center', va='bottom', 
                   fontsize=10, fontweight='bold', color='#2C3E50')
    
    axs[2].set_title('BPE Token Distribution', 
                    fontsize=16, fontweight='bold', color='#2C3E50', pad=15)
    axs[2].set_ylabel('Frequency', fontsize=14, fontweight='bold', color='#2C3E50')
    axs[2].tick_params(axis='both', which='major', labelsize=12)
    axs[2].set_ylim(0, max(bpe_counts) * 1.15)  # Add space for labels
    axs[2].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add explanatory text at the bottom
    explanation = (
        "This visualization shows the distribution of the most frequent tokens for each tokenization method. "
        "Note how character-level tokenization has fewer unique tokens with higher frequencies, "
        "while word-level has more diverse tokens with lower frequencies. "
        "BPE provides a balance between token diversity and frequency."
    )
    fig.text(0.5, 0.01, explanation, ha='center', va='bottom', 
            fontsize=12, style='italic', color='#666666')
    
    # Add a border around the entire figure
    fig.patch.set_linewidth(2)
    fig.patch.set_edgecolor('lightgray')
    
    # Use tight_layout with specific padding for consistent alignment
    plt.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=4.0)  # Leave room for title and footer
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.5, facecolor='white')
        print(f"Token distribution plot saved to {output_file}")
    else:
        plt.show()


def plot_token_counts(analysis: Dict, output_file: str = None):
    """Plot token counts for different tokenization methods."""
    # Use a clean, modern style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    methods = ['Character', 'Word', 'BPE']
    total_counts = [
        analysis["char_tokens"]["count"],
        analysis["word_tokens"]["count"],
        analysis["bpe_tokens"]["count"]
    ]
    unique_counts = [
        analysis["char_tokens"]["unique"],
        analysis["word_tokens"]["unique"],
        analysis["bpe_tokens"]["unique"]
    ]
    
    # Use fixed dimensions for consistent alignment with a clean white background
    fig, ax = plt.subplots(figsize=(16, 10), facecolor='white')
    
    # Add a title with improved styling
    ax.set_title('Token Counts by Tokenization Method', 
                fontsize=18, fontweight='bold', color='#2C3E50', pad=20)
    
    x = range(len(methods))
    width = 0.35
    
    # Use more vibrant colors with better opacity
    ax.bar([i - width/2 for i in x], total_counts, width, 
          label='Total Tokens', color='#5DA5DA', alpha=0.9, 
          edgecolor='#2C3E50', linewidth=1.0)
    ax.bar([i + width/2 for i in x], unique_counts, width, 
          label='Unique Tokens', color='#F15854', alpha=0.9, 
          edgecolor='#2C3E50', linewidth=1.0)
    
    ax.set_ylabel('Count', fontsize=16, fontweight='bold', color='#2C3E50')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=14, fontweight='bold')
    
    # Improve legend styling
    ax.legend(fontsize=14, frameon=True, 
             framealpha=0.9, edgecolor='lightgray',
             loc='upper right', bbox_to_anchor=(0.95, 0.95))
    ax.tick_params(axis='y', which='major', labelsize=12)
    
    # Add count labels on top of bars with improved styling
    for i, v in enumerate(total_counts):
        ax.text(i - width/2, v + max(total_counts) * 0.02, str(v), 
                ha='center', fontsize=14, fontweight='bold', color='#2C3E50')
    
    for i, v in enumerate(unique_counts):
        ax.text(i + width/2, v + max(total_counts) * 0.02, str(v), 
                ha='center', fontsize=14, fontweight='bold', color='#2C3E50')
    
    # Add explanatory text below the chart
    explanation = (
        "Character tokenization has few unique tokens but produces long sequences.\n"
        "Word tokenization creates shorter sequences but has vocabulary growth issues.\n"
        "BPE tokenization provides a balance between sequence length and vocabulary size."
    )
    fig.text(0.5, 0.01, explanation, ha='center', va='bottom', 
            fontsize=12, style='italic', color='#666666')
    
    # Add a border around the entire figure
    fig.patch.set_linewidth(2)
    fig.patch.set_edgecolor('lightgray')
    
    # Use tight_layout with specific padding for consistent alignment
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Leave room for title and footer
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.5, facecolor='white')
        print(f"Token counts plot saved to {output_file}")
    else:
        plt.show()


def visualize_tokenization_comparison(tokenizer, output_file=None):
    """
    Create a visualization comparing character-level, word-level, and BPE tokenization.
    
    Args:
        tokenizer: Trained BPETokenizer instance
        output_file: Path to save the visualization (if None, display instead)
    """
    # Use a clean, modern style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Example text
    text = "The transformer architecture revolutionized natural language processing."
    
    # Character-level tokenization
    char_tokens = list(text)
    
    # Word-level tokenization
    word_tokens = text.split()
    
    # BPE tokenization
    bpe_token_ids = tokenizer.encode(text)
    bpe_tokens = [tokenizer.vocab_reversed.get(token_id, "<UNK>") for token_id in bpe_token_ids]
    
    # Create a figure with a clean white background
    fig = plt.figure(figsize=(16, 12), facecolor='white')
    gs = GridSpec(4, 1, height_ratios=[0.5, 1, 1, 1], figure=fig, hspace=0.4)
    
    # Add the original text at the top
    ax_text = fig.add_subplot(gs[0])
    ax_text.axis('off')
    ax_text.text(0.5, 0.5, f'"{text}"', 
               ha='center', va='center', fontsize=16, 
               fontweight='bold', color='#2C3E50',
               bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray',
                         boxstyle='round,pad=0.5'))
    
    # Create the tokenization visualizations
    axs = [fig.add_subplot(gs[i+1]) for i in range(3)]
    
    # Add a main title to the figure
    fig.suptitle("Tokenization Approaches Comparison", 
                fontsize=20, fontweight='bold', y=0.98)
    
    # Plot character-level tokenization
    ax = axs[0]
    token_width = 0.6  # Fixed width for each token
    
    # Add a subtitle for this section
    ax.text(0.5, 1.15, "Character-level Tokenization", 
           ha='center', va='center', fontsize=16, fontweight='bold',
           transform=ax.transAxes)
    
    for i, token in enumerate(char_tokens):
        if token == " ":
            token = "□"  # Represent space with a visible character
        x_pos = i * token_width
        # Create rectangle with modern styling
        rect = plt.Rectangle((x_pos, 0), token_width * 0.9, 0.8, 
                          facecolor='#5DA5DA', alpha=0.9,  # Vibrant blue
                          edgecolor='#2C3E50', linewidth=1.0,
                          zorder=2, clip_on=False)
        ax.add_patch(rect)
        ax.text(x_pos + token_width * 0.45, 0.4, token, 
              ha='center', va='center', fontsize=11, fontweight='bold',
              color='white')  # White text for better contrast
    
    # Add a label showing the number of tokens
    ax.text(0.98, 0.1, f"Total tokens: {len(char_tokens)}", 
           ha='right', va='center', fontsize=12, fontweight='bold',
           color='#2C3E50', transform=ax.transAxes,
           bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray',
                    boxstyle='round,pad=0.3'))
    
    ax.set_xlim(-1, len(char_tokens) * token_width + 1)
    ax.set_ylim(-0.2, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Plot word-level tokenization
    ax = axs[1]
    token_width = 3.5  # Wider for words
    
    # Add a subtitle for this section
    ax.text(0.5, 1.15, "Word-level Tokenization", 
           ha='center', va='center', fontsize=16, fontweight='bold',
           transform=ax.transAxes)
    
    for i, token in enumerate(word_tokens):
        x_pos = i * token_width
        # Create rectangle with modern styling
        rect = plt.Rectangle((x_pos, 0), token_width * 0.9, 0.8, 
                          facecolor='#60BD68', alpha=0.9,  # Vibrant green
                          edgecolor='#2C3E50', linewidth=1.0,
                          zorder=2, clip_on=False)
        ax.add_patch(rect)
        ax.text(x_pos + token_width * 0.45, 0.4, token, 
              ha='center', va='center', fontsize=12, fontweight='bold',
              color='white')  # White text for better contrast
    
    # Add a label showing the number of tokens
    ax.text(0.98, 0.1, f"Total tokens: {len(word_tokens)}", 
           ha='right', va='center', fontsize=12, fontweight='bold',
           color='#2C3E50', transform=ax.transAxes,
           bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray',
                    boxstyle='round,pad=0.3'))
    
    ax.set_xlim(-1, len(word_tokens) * token_width + 1)
    ax.set_ylim(-0.2, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Plot BPE tokenization
    ax = axs[2]
    token_width = 2.2  # Width for BPE tokens (between char and word)
    
    # Add a subtitle for this section
    ax.text(0.5, 1.15, "BPE Tokenization", 
           ha='center', va='center', fontsize=16, fontweight='bold',
           transform=ax.transAxes)
    
    for i, token in enumerate(bpe_tokens):
        x_pos = i * token_width
        # Create rectangle with modern styling
        rect = plt.Rectangle((x_pos, 0), token_width * 0.9, 0.8, 
                          facecolor='#F15854', alpha=0.9,  # Vibrant red
                          edgecolor='#2C3E50', linewidth=1.0,
                          zorder=2, clip_on=False)
        ax.add_patch(rect)
        ax.text(x_pos + token_width * 0.45, 0.4, token, 
              ha='center', va='center', fontsize=12, fontweight='bold',
              color='white')  # White text for better contrast
    
    # Add a label showing the number of tokens
    ax.text(0.98, 0.1, f"Total tokens: {len(bpe_tokens)}", 
           ha='right', va='center', fontsize=12, fontweight='bold',
           color='#2C3E50', transform=ax.transAxes,
           bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray',
                    boxstyle='round,pad=0.3'))
    
    ax.set_xlim(-1, len(bpe_tokens) * token_width + 1)
    ax.set_ylim(-0.2, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Add a footer with explanation
    footer_text = (
        "Character-level tokenization splits text into individual characters, resulting in a small vocabulary but long sequences. "
        "Word-level tokenization treats each word as a token, creating shorter sequences but a large vocabulary with OOV issues. "
        "BPE tokenization finds a balance by creating subword units that efficiently represent the text."
    )
    fig.text(0.5, 0.01, footer_text, ha='center', va='bottom', 
            fontsize=10, style='italic', color='#666666', 
            bbox=dict(facecolor='white', alpha=0.8, edgecolor=None))
    
    # Add a border around the entire figure
    fig.patch.set_linewidth(2)
    fig.patch.set_edgecolor('lightgray')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leave room for title and footer
    
    # Save or display the figure
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.5, facecolor='white')
        print(f"Tokenization comparison saved to {output_file}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Compare tokenization approaches")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to the trained BPE model")
    parser.add_argument("--text-file", type=str, required=True,
                        help="Path to the text file to analyze")
    parser.add_argument("--output-prefix", type=str, default="tokenization_comparison",
                        help="Prefix for output files")
    parser.add_argument("--visualize-only", action="store_true",
                        help="Only generate the visual comparison without analysis")
    args = parser.parse_args()
    
    # Load the tokenizer
    tokenizer = BPETokenizer.load(args.model)
    
    if args.visualize_only:
        # Generate only the visual comparison
        visualize_tokenization_comparison(tokenizer, f"{args.output_prefix}_visual.png")
        print(f"Visual tokenization comparison saved to {args.output_prefix}_visual.png")
        return
    
    # Load the text
    with open(args.text_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Analyze tokenization
    analysis = analyze_tokenization(text, tokenizer)
    
    # Print summary
    print(f"Text length: {analysis['text_length']} characters")
    print("\nCharacter-level tokenization:")
    print(f"  Total tokens: {analysis['char_tokens']['count']}")
    print(f"  Unique tokens: {analysis['char_tokens']['unique']}")
    print(f"  First few tokens: {' '.join(analysis['char_tokens']['tokens'])}")
    
    print("\nWord-level tokenization:")
    print(f"  Total tokens: {analysis['word_tokens']['count']}")
    print(f"  Unique tokens: {analysis['word_tokens']['unique']}")
    print(f"  First few tokens: {' '.join(analysis['word_tokens']['tokens'])}")
    
    print("\nBPE tokenization:")
    print(f"  Total tokens: {analysis['bpe_tokens']['count']}")
    print(f"  Unique tokens: {analysis['bpe_tokens']['unique']}")
    print(f"  First few tokens: {' '.join(analysis['bpe_tokens']['tokens'])}")
    
    # Plot token distributions
    plot_token_distribution(analysis, f"{args.output_prefix}_distribution.png")
    
    # Plot token counts
    plot_token_counts(analysis, f"{args.output_prefix}_counts.png")
    
    # Create the visual tokenization comparison
    visualize_tokenization_comparison(tokenizer, f"{args.output_prefix}_visual.png")
    
    # Save analysis to JSON
    with open(f"{args.output_prefix}_analysis.json", 'w', encoding='utf-8') as f:
        # Convert Counter objects to lists for JSON serialization
        serializable_analysis = {
            "text_length": analysis["text_length"],
            "char_tokens": {
                "count": analysis["char_tokens"]["count"],
                "unique": analysis["char_tokens"]["unique"],
                "tokens": analysis["char_tokens"]["tokens"],
                "most_common": analysis["char_tokens"]["most_common"]
            },
            "word_tokens": {
                "count": analysis["word_tokens"]["count"],
                "unique": analysis["word_tokens"]["unique"],
                "tokens": analysis["word_tokens"]["tokens"],
                "most_common": analysis["word_tokens"]["most_common"]
            },
            "bpe_tokens": {
                "count": analysis["bpe_tokens"]["count"],
                "unique": analysis["bpe_tokens"]["unique"],
                "tokens": analysis["bpe_tokens"]["tokens"],
                "most_common": analysis["bpe_tokens"]["most_common"]
            }
        }
        json.dump(serializable_analysis, f, ensure_ascii=False, indent=2)
    
    print(f"\nAll visualizations and analysis saved with prefix: {args.output_prefix}")


if __name__ == "__main__":
    main()
