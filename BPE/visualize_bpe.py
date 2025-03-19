#!/usr/bin/env python3
"""
Visualize BPE tokenization process on a given text.
This script demonstrates how BPE tokenization works step by step.
"""

import json
import argparse
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from BPEexample import BPETokenizer


def visualize_tokenization(text, tokenizer, output_file=None):
    """
    Visualize the BPE tokenization process on a given text.
    
    Args:
        text: Input text to tokenize
        tokenizer: Trained BPETokenizer instance
        output_file: Path to save the visualization (if None, display instead)
    """
    # Start with character-level tokenization
    chars = list(text.lower())
    
    # Track the merges step by step
    steps = [chars.copy()]
    tokens_at_step = [chars.copy()]
    
    # Apply merges iteratively
    current_tokens = chars.copy()
    while True:
        # Find all pairs that can be merged
        pairs = [(current_tokens[i], current_tokens[i+1]) 
                for i in range(len(current_tokens)-1)]
        
        # Find the first pair that can be merged according to our rules
        merged = False
        for pair in pairs:
            if pair in tokenizer.merges:
                new_token = tokenizer.merges[pair]
                # Apply the merge
                i = 0
                new_tokens = []
                while i < len(current_tokens):
                    if (i < len(current_tokens) - 1 and 
                        current_tokens[i] == pair[0] and 
                        current_tokens[i + 1] == pair[1]):
                        new_tokens.append(new_token)
                        i += 2
                    else:
                        new_tokens.append(current_tokens[i])
                        i += 1
                
                current_tokens = new_tokens
                steps.append(f"Merge: '{pair[0]}' + '{pair[1]}' → '{new_token}'")
                tokens_at_step.append(current_tokens.copy())
                merged = True
                break
        
        if not merged:
            # No more merges possible
            break
    
    # Calculate dimensions for the visualization
    max_tokens_width = max(len(tokens) for tokens in tokens_at_step)
    num_steps = len(steps)
    
    # Create the visualization with fixed width regardless of token count
    fig_width = 16  # Wider for better spacing
    fig_height = max(8, num_steps * 0.7 + 2)  # Taller for better spacing
    
    # Use a clean, modern style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor='white')
    
    # Define a colormap for the tokens - use a more visually appealing palette
    colors = plt.cm.viridis(np.linspace(0, 0.8, 20))  # Using viridis colormap
    
    # Create a mapping of tokens to colors
    unique_tokens = set()
    for tokens in tokens_at_step:
        unique_tokens.update(tokens)
    token_to_color = {token: i % 20 for i, token in enumerate(sorted(unique_tokens))}
    
    # Calculate token width based on available space
    token_width = min(1.0, (fig_width - 6) / max_tokens_width)
    
    # Add a title and subtitle
    ax.text(max_tokens_width * token_width / 2, 1.0, 
            "Byte Pair Encoding (BPE) Tokenization Process", 
            fontsize=18, fontweight='bold', ha='center')
    ax.text(max_tokens_width * token_width / 2, 0.5, 
            f"Input text: '{text}'", 
            fontsize=14, ha='center', style='italic')
    
    # Plot each step
    for i, (step_label, tokens) in enumerate(zip(['Initial tokenization'] + steps[1:], tokens_at_step)):
        # Create a colored representation of the tokens
        token_colors = [token_to_color[token] for token in tokens]
        
        # Plot colored rectangles for each token
        for j, (token, color_idx) in enumerate(zip(tokens, token_colors)):
            rect = plt.Rectangle((j * token_width, -i - 1), token_width * 0.9, 0.7, 
                                facecolor=colors[color_idx], alpha=0.8,
                                edgecolor='black', linewidth=1.0,
                                zorder=2, # Ensure rectangles are above grid lines
                                clip_on=False)
            ax.add_patch(rect)
            plt.text(j * token_width + token_width/2, -i - 1 + 0.35, token, 
                    ha='center', va='center', fontsize=12, fontweight='bold',
                    color='white', zorder=3)  # White text for better contrast
        
        # Add step label with a clearer format
        if i == 0:
            plt.text(-token_width * 1.5, -i - 1 + 0.35, "Initial:", 
                    ha='right', va='center', fontsize=12, fontweight='bold')
        else:
            plt.text(-token_width * 1.5, -i - 1 + 0.35, f"Step {i}:", 
                    ha='right', va='center', fontsize=12, fontweight='bold')
            
            # Add merge description in a box
            merge_box = plt.Rectangle((len(tokens) * token_width + token_width, -i - 1), 
                                    token_width * 8, 0.7, 
                                    facecolor='lavender', alpha=0.7,
                                    edgecolor='navy', linewidth=1.0,
                                    zorder=2, clip_on=False)
            ax.add_patch(merge_box)
            plt.text(len(tokens) * token_width + token_width * 5, -i - 1 + 0.35, 
                    step_label, ha='center', va='center', 
                    fontsize=11, color='navy', fontweight='bold', zorder=3)
    
    # Set plot limits and labels
    ax.set_xlim(-token_width * 3, max_tokens_width * token_width + token_width * 10)
    ax.set_ylim(-num_steps - 1.5, 1.5)
    
    # Remove axes and grid for a cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    
    # Add a footer with explanation
    footer_text = (
        "BPE tokenization iteratively merges the most frequent adjacent token pairs to form new tokens. "
        "This visualization shows each merge step in the process."
    )
    ax.text(max_tokens_width * token_width / 2, -num_steps - 1, 
            footer_text, ha='center', va='center', 
            fontsize=10, style='italic', wrap=True)
    
    # Add a border around the entire figure
    fig.patch.set_linewidth(2)
    fig.patch.set_edgecolor('lightgray')
    
    # Adjust layout with specific padding
    plt.tight_layout(pad=2.0, rect=[0, 0, 1, 0.98])  # Leave room for title
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.5, facecolor='white')
        print(f"Visualization saved to {output_file}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize BPE tokenization process")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to the trained BPE model")
    parser.add_argument("--text", type=str, required=True,
                        help="Text to tokenize and visualize")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save the visualization (if not provided, display instead)")
    args = parser.parse_args()
    
    # Load the tokenizer
    tokenizer = BPETokenizer.load(args.model)
    
    # Visualize the tokenization process
    visualize_tokenization(args.text, tokenizer, args.output)


if __name__ == "__main__":
    main()
