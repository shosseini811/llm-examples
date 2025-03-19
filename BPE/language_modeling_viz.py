#!/usr/bin/env python3
"""
Create a visualization showing how BPE tokenization helps with language modeling.
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from BPEexample import BPETokenizer


def visualize_language_modeling(output_file=None, tokenizer=None):
    """
    Create a visualization showing how BPE tokenization helps with language modeling.
    
    Args:
        tokenizer: Trained BPETokenizer instance
        output_file: Path to save the visualization (if None, display instead)
    """
    # Use a clean, modern style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create a figure with a fixed size for consistent alignment
    fig = plt.figure(figsize=(16, 12), facecolor='white')
    gs = GridSpec(3, 1, height_ratios=[1, 1, 1.5], figure=fig, hspace=0.4)
    
    # Add a main title to the figure
    fig.suptitle("Language Modeling with BPE Tokenization", 
                fontsize=20, fontweight='bold', y=0.98)
    
    # Example sentences
    example_sentences = [
        "I love natural language processing",
        "I love natural language understanding",
        "I love computational linguistics"
    ]
    
    # Tokenize the sentences
    tokenized_sentences = []
    if tokenizer:
        # Use the provided tokenizer
        for sentence in example_sentences:
            tokens = []
            token_ids = tokenizer.encode(sentence)
            for token_id in token_ids:
                token = tokenizer.vocab_reversed.get(token_id, "<UNK>")
                tokens.append(token)
            tokenized_sentences.append(tokens)
    else:
        # Use predefined tokenization for demonstration
        tokenized_sentences = [
            ["I", "love", "natural", "language", "process", "ing"],
            ["I", "love", "natural", "language", "understand", "ing"],
            ["I", "love", "computation", "al", "linguistic", "s"]
        ]
    
    # Plot 1: Word-level tokenization (conceptual)
    ax1 = fig.add_subplot(gs[0])
    word_tokens = [sentence.split() for sentence in example_sentences]
    
    # Calculate token width for consistent spacing
    max_tokens = max(len(tokens) for tokens in word_tokens)
    token_width = 1.2  # Fixed width for each token
    
    # Add a subtitle for this section
    ax1.text(0.5, 1.15, "Word-level Tokenization", 
            ha='center', va='center', fontsize=16, fontweight='bold',
            transform=ax1.transAxes)
    
    for i, tokens in enumerate(word_tokens):
        y_pos = i
        for j, token in enumerate(tokens):
            x_pos = j * token_width
            # Create rectangle with shadow effect for a modern look
            rect = plt.Rectangle((x_pos, y_pos - 0.4), token_width * 0.9, 0.8, 
                                facecolor='#5DA5DA', alpha=0.9,  # More vibrant blue
                                edgecolor='#2C3E50', linewidth=1.0,
                                zorder=2, clip_on=False)
            ax1.add_patch(rect)
            ax1.text(x_pos + token_width * 0.45, y_pos, token, 
                    ha='center', va='center', fontsize=12, fontweight='bold',
                    color='white')  # White text for better contrast
    
    # Highlight the differences
    for i, tokens in enumerate(word_tokens):
        if i > 0:  # Compare with the first sentence
            for j, (token, first_token) in enumerate(zip(tokens, word_tokens[0])):
                if j < len(tokens) and j < len(word_tokens[0]) and token != first_token:
                    x_pos = j * token_width
                    # Use a different color for different words
                    rect = plt.Rectangle((x_pos, i - 0.4), token_width * 0.9, 0.8, 
                                        facecolor='#F15854', alpha=0.9,  # Vibrant red
                                        edgecolor='#2C3E50', linewidth=1.0,
                                        zorder=3, clip_on=False)
                    ax1.add_patch(rect)
                    ax1.text(x_pos + token_width * 0.45, i, token, 
                            ha='center', va='center', fontsize=12, fontweight='bold',
                            color='white')
    
    # Add sentence labels with better styling
    for i, sentence in enumerate(example_sentences):
        ax1.text(-1.0, i, f"Sentence {i+1}:", ha='right', va='center', 
                fontsize=12, fontweight='bold', color='#2C3E50')
    
    ax1.set_xlim(-1.5, max_tokens * token_width + 1.5)
    ax1.set_ylim(-0.8, len(example_sentences) - 0.2)
    ax1.set_yticks([])
    ax1.set_xticks([])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    
    # Plot 2: BPE tokenization
    ax2 = fig.add_subplot(gs[1])
    
    # Add a subtitle for this section
    ax2.text(0.5, 1.15, "BPE Tokenization", 
            ha='center', va='center', fontsize=16, fontweight='bold',
            transform=ax2.transAxes)
    
    # Calculate token width for BPE tokens
    max_bpe_tokens = max(len(tokens) for tokens in tokenized_sentences)
    bpe_token_width = 0.8  # Adjusted width for BPE tokens
    
    for i, tokens in enumerate(tokenized_sentences):
        y_pos = i
        for j, token in enumerate(tokens):
            x_pos = j * bpe_token_width
            # Create rectangle for a modern look
            rect = plt.Rectangle((x_pos, y_pos - 0.4), bpe_token_width * 0.9, 0.8, 
                                facecolor='#60BD68', alpha=0.8,  # Vibrant green
                                edgecolor='#2C3E50', linewidth=1.0,
                                zorder=2, clip_on=False)
            ax2.add_patch(rect)
            ax2.text(x_pos + bpe_token_width * 0.45, y_pos, token, 
                    ha='center', va='center', fontsize=11, fontweight='bold',
                    color='white')  # White text for better contrast
    
    # Highlight shared subword units
    shared_subwords = {}
    for i, tokens in enumerate(tokenized_sentences):
        for j, token in enumerate(tokens):
            if token not in shared_subwords:
                shared_subwords[token] = []
            shared_subwords[token].append((i, j))
    
    # Color the tokens that appear in multiple sentences
    for token, positions in shared_subwords.items():
        if len(positions) > 1 and token not in [" ", "<UNK>"]:
            for i, j in positions:
                x_pos = j * bpe_token_width
                # Use a special styling for shared tokens
                rect = plt.Rectangle((x_pos, i - 0.4), bpe_token_width * 0.9, 0.8, 
                                    facecolor='#60BD68', alpha=0.9,  # Same green but more opaque
                                    edgecolor='#4062BB', linewidth=2.0,  # Blue border
                                    zorder=3, clip_on=False)
                ax2.add_patch(rect)
                ax2.text(x_pos + bpe_token_width * 0.45, i, token, 
                        ha='center', va='center', fontsize=11, fontweight='bold',
                        color='white')
    
    # Add sentence labels with better styling
    for i, sentence in enumerate(example_sentences):
        ax2.text(-1.0, i, f"Sentence {i+1}:", ha='right', va='center', 
                fontsize=12, fontweight='bold', color='#2C3E50')
    
    ax2.set_xlim(-1.5, max_bpe_tokens * bpe_token_width + 1.5)
    ax2.set_ylim(-0.8, len(example_sentences) - 0.2)
    ax2.set_yticks([])
    ax2.set_xticks([])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    
    # Plot 3: Language modeling explanation
    ax3 = fig.add_subplot(gs[2])
    ax3.axis('off')
    
    # Create a background panel for the explanation
    explanation_panel = plt.Rectangle((0.05, 0.05), 0.9, 0.9, 
                                    facecolor='#F5F5F5', alpha=0.7,
                                    edgecolor='#CCCCCC', linewidth=1.0,
                                    transform=ax3.transAxes,
                                    zorder=1)
    ax3.add_patch(explanation_panel)
    
    # Add text explanation with improved styling
    explanation = [
        "Key Differences and Benefits",
        "",
        "Word-level tokenization treats each word as a separate token:",
        "- Vocabulary grows with each new word form",
        "- No parameter sharing between similar words",
        "- Out-of-vocabulary words are a significant problem",
        "",
        "BPE tokenization breaks words into subword units:",
        "- Vocabulary size is controlled and fixed",
        "- Common subword units are shared across words",
        "- Rare words can be represented as sequences of subword units",
        "- Better handling of morphology and compound words",
        "",
        "Benefits for language modeling:",
        "- More efficient use of the vocabulary",
        "- Better generalization to unseen words",
        "- Improved parameter sharing across related words",
        "- Reduced perplexity and improved model performance"
    ]
    
    y_pos = 0.9
    for i, line in enumerate(explanation):
        if i == 0:  # Title
            ax3.text(0.5, y_pos, line, ha='center', va='top', 
                    fontsize=16, fontweight='bold', color='#2C3E50',
                    transform=ax3.transAxes, zorder=2)
        elif line == "":  # Empty line
            pass
        elif line.startswith("-"):  # Bullet point
            ax3.text(0.12, y_pos, "•", ha='left', va='top', 
                    fontsize=14, color='#E15759', fontweight='bold',
                    transform=ax3.transAxes, zorder=2)
            ax3.text(0.15, y_pos, line[2:], ha='left', va='top', 
                    fontsize=13, color='#333333',
                    transform=ax3.transAxes, zorder=2)
        else:  # Section header
            ax3.text(0.08, y_pos, line, ha='left', va='top', 
                    fontsize=14, fontweight='bold', color='#2C3E50',
                    transform=ax3.transAxes, zorder=2)
        
        y_pos -= 0.05
    
    # Add a legend with improved styling
    legend_x = 0.7
    legend_y = 0.85
    legend_title = "Legend"
    
    # Legend title
    ax3.text(legend_x, legend_y + 0.05, legend_title, 
            ha='left', va='center', fontsize=14, 
            fontweight='bold', color='#2C3E50',
            transform=ax3.transAxes, zorder=2)
    
    # Create a background for the legend
    legend_panel = plt.Rectangle((legend_x - 0.02, legend_y - 0.25), 0.3, 0.35, 
                                facecolor='white', alpha=0.8,
                                edgecolor='#CCCCCC', linewidth=1.0,
                                transform=ax3.transAxes,
                                zorder=1)
    ax3.add_patch(legend_panel)
    
    # Word-level legend
    ax3.add_patch(plt.Rectangle((legend_x, legend_y), 0.05, 0.03, 
                              facecolor='#5DA5DA', alpha=0.9, edgecolor='#2C3E50',
                              transform=ax3.transAxes, zorder=2,
                              boxstyle='round,pad=0.01'))
    ax3.text(legend_x + 0.07, legend_y + 0.015, "Word token", 
            ha='left', va='center', fontsize=12, color='#333333',
            transform=ax3.transAxes, zorder=2)
    
    # Different word legend
    ax3.add_patch(plt.Rectangle((legend_x, legend_y - 0.06), 0.05, 0.03, 
                              facecolor='#F15854', alpha=0.9, edgecolor='#2C3E50',
                              transform=ax3.transAxes, zorder=2,
                              boxstyle='round,pad=0.01'))
    ax3.text(legend_x + 0.07, legend_y - 0.06 + 0.015, "Different word", 
            ha='left', va='center', fontsize=12, color='#333333',
            transform=ax3.transAxes, zorder=2)
    
    # BPE token legend
    ax3.add_patch(plt.Rectangle((legend_x, legend_y - 0.12), 0.05, 0.03, 
                              facecolor='#60BD68', alpha=0.8, edgecolor='#2C3E50',
                              transform=ax3.transAxes, zorder=2,
                              boxstyle='round,pad=0.01'))
    ax3.text(legend_x + 0.07, legend_y - 0.12 + 0.015, "BPE token", 
            ha='left', va='center', fontsize=12, color='#333333',
            transform=ax3.transAxes, zorder=2)
    
    # Shared BPE token legend
    ax3.add_patch(plt.Rectangle((legend_x, legend_y - 0.18), 0.05, 0.03, 
                              facecolor='#60BD68', alpha=0.9, edgecolor='#4062BB',
                              linewidth=2.0, transform=ax3.transAxes, zorder=2,
                              boxstyle='round,pad=0.01'))
    ax3.text(legend_x + 0.07, legend_y - 0.18 + 0.015, "Shared BPE token", 
            ha='left', va='center', fontsize=12, color='#333333',
            transform=ax3.transAxes, zorder=2)
    
    # Add a footer with citation
    footer_text = (
        "BPE tokenization enables more efficient language modeling by breaking words into subword units, "
        "allowing for better parameter sharing and handling of rare words."
    )
    fig.text(0.5, 0.01, footer_text, ha='center', va='bottom', 
            fontsize=10, style='italic', color='#666666')
    
    # Add a border around the entire figure
    fig.patch.set_linewidth(2)
    fig.patch.set_edgecolor('lightgray')
    
    # Adjust layout with specific padding
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leave room for title and footer
    
    # Save or display the figure
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.5, facecolor='white')
        print(f"Language modeling visualization saved to {output_file}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Create a language modeling visualization")
    parser.add_argument("--model", type=str, required=False,
                        help="Path to the trained BPE model")
    parser.add_argument("--output", type=str, default="language_modeling_viz.png",
                        help="Path to save the visualization")
    parser.add_argument("--demo", action="store_true",
                        help="Run in demo mode without a tokenizer")
    args = parser.parse_args()
    
    if args.demo:
        # Create the visualization without a tokenizer
        visualize_language_modeling(args.output)
    else:
        # Load the BPE tokenizer
        tokenizer = BPETokenizer.load(args.model)
        
        # Create the visualization
        visualize_language_modeling(args.output, tokenizer)


if __name__ == "__main__":
    main()
