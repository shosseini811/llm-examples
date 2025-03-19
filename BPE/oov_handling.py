#!/usr/bin/env python3
"""
Demonstrate how BPE handles out-of-vocabulary (OOV) words.
This script shows the advantage of BPE over word-level tokenization for unseen words.
"""

import argparse
import json
from typing import List, Dict, Set
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from BPEexample import BPETokenizer


def word_tokenize(text: str) -> List[str]:
    """Simple word-level tokenization."""
    return text.lower().split()


def train_word_vocab(corpus: List[str], vocab_size: int) -> Dict[str, int]:
    """Train a word-level vocabulary."""
    # Count word frequencies
    word_counts = {}
    for text in corpus:
        for word in word_tokenize(text):
            word_counts[word] = word_counts.get(word, 0) + 1
    
    # Sort by frequency
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Create vocabulary (limit to vocab_size)
    vocab = {"<UNK>": 0}  # Unknown token
    for i, (word, _) in enumerate(sorted_words[:vocab_size-1]):
        vocab[word] = i + 1
    
    return vocab


def analyze_oov_handling(train_corpus: List[str], test_corpus: List[str], 
                         bpe_tokenizer: BPETokenizer, word_vocab_size: int) -> Dict:
    """Analyze how BPE and word-level tokenization handle OOV words."""
    # Train word-level vocabulary
    word_vocab = train_word_vocab(train_corpus, word_vocab_size)
    
    # Analyze test corpus
    results = {
        "train_corpus_size": len(train_corpus),
        "test_corpus_size": len(test_corpus),
        "word_vocab_size": len(word_vocab),
        "bpe_vocab_size": len(bpe_tokenizer.vocab),
        "test_examples": []
    }
    
    # Process each test sentence
    for text in test_corpus:
        # Word-level tokenization
        words = word_tokenize(text)
        word_tokens = []
        word_oov = []
        
        for word in words:
            if word in word_vocab:
                word_tokens.append(word)
                word_oov.append(False)
            else:
                word_tokens.append("<UNK>")
                word_oov.append(True)
        
        # BPE tokenization
        bpe_token_ids = bpe_tokenizer.encode(text)
        bpe_tokens = [bpe_tokenizer.vocab_reversed.get(idx, "<UNK>") for idx in bpe_token_ids]
        
        # Count OOV tokens
        word_oov_count = sum(word_oov)
        bpe_oov_count = bpe_tokens.count("<UNK>")
        
        # Add example to results
        results["test_examples"].append({
            "text": text,
            "word_tokens": word_tokens,
            "word_oov": word_oov,
            "word_oov_count": word_oov_count,
            "word_oov_percentage": word_oov_count / len(words) * 100 if words else 0,
            "bpe_tokens": bpe_tokens,
            "bpe_oov_count": bpe_oov_count,
            "bpe_oov_percentage": bpe_oov_count / len(bpe_tokens) * 100 if bpe_tokens else 0
        })
    
    # Calculate overall statistics
    total_words = sum(len(ex["word_tokens"]) for ex in results["test_examples"])
    total_word_oov = sum(ex["word_oov_count"] for ex in results["test_examples"])
    total_bpe_tokens = sum(len(ex["bpe_tokens"]) for ex in results["test_examples"])
    total_bpe_oov = sum(ex["bpe_oov_count"] for ex in results["test_examples"])
    
    results["overall"] = {
        "total_words": total_words,
        "total_word_oov": total_word_oov,
        "word_oov_percentage": total_word_oov / total_words * 100 if total_words else 0,
        "total_bpe_tokens": total_bpe_tokens,
        "total_bpe_oov": total_bpe_oov,
        "bpe_oov_percentage": total_bpe_oov / total_bpe_tokens * 100 if total_bpe_tokens else 0
    }
    
    return results


def visualize_oov_comparison(analysis: Dict, output_file: str = None):
    """Visualize OOV handling comparison between word-level and BPE tokenization."""
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # Plot 1: Overall OOV percentage
    ax1 = fig.add_subplot(gs[0, 0])
    methods = ['Word-level', 'BPE']
    oov_percentages = [
        analysis["overall"]["word_oov_percentage"],
        analysis["overall"]["bpe_oov_percentage"]
    ]
    
    bars = ax1.bar(methods, oov_percentages, color=['skyblue', 'salmon'])
    ax1.set_ylabel('OOV Percentage (%)')
    ax1.set_title('Overall OOV Token Percentage')
    
    # Add percentage labels
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom')
    
    # Plot 2: OOV percentage per example
    ax2 = fig.add_subplot(gs[0, 1])
    example_indices = range(len(analysis["test_examples"]))
    word_oov_percentages = [ex["word_oov_percentage"] for ex in analysis["test_examples"]]
    bpe_oov_percentages = [ex["bpe_oov_percentage"] for ex in analysis["test_examples"]]
    
    ax2.plot(example_indices, word_oov_percentages, 'o-', color='skyblue', label='Word-level')
    ax2.plot(example_indices, bpe_oov_percentages, 'o-', color='salmon', label='BPE')
    ax2.set_xlabel('Example Index')
    ax2.set_ylabel('OOV Percentage (%)')
    ax2.set_title('OOV Percentage per Example')
    ax2.legend()
    
    # Plot 3: Detailed example visualization
    ax3 = fig.add_subplot(gs[1, :])
    
    # Choose an example with OOV words (preferably the one with the biggest difference)
    diffs = [ex["word_oov_percentage"] - ex["bpe_oov_percentage"] 
             for ex in analysis["test_examples"]]
    if diffs:
        example_idx = diffs.index(max(diffs))
        example = analysis["test_examples"][example_idx]
        
        # Visualize word-level tokenization
        word_colors = ['red' if oov else 'green' for oov in example["word_oov"]]
        
        # Create a text representation with colored words
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 2)
        ax3.axis('off')
        
        # Title
        ax3.text(0.5, 1.9, f'Example {example_idx}: OOV Word Handling', 
                ha='center', va='center', fontsize=14, fontweight='bold')
        
        # Original text
        ax3.text(0.5, 1.7, f'Original: "{example["text"]}"', 
                ha='center', va='center', fontsize=12)
        
        # Word-level tokenization
        ax3.text(0.0, 1.4, 'Word-level:', va='center', fontsize=12, fontweight='bold')
        x_pos = 0.15
        for i, (token, is_oov) in enumerate(zip(example["word_tokens"], example["word_oov"])):
            color = 'red' if is_oov else 'green'
            ax3.text(x_pos, 1.4, token, va='center', fontsize=12, color=color)
            x_pos += len(token) * 0.015 + 0.02
        
        # BPE tokenization
        ax3.text(0.0, 1.1, 'BPE:', va='center', fontsize=12, fontweight='bold')
        x_pos = 0.15
        for token in example["bpe_tokens"]:
            color = 'red' if token == "<UNK>" else 'green'
            ax3.text(x_pos, 1.1, token, va='center', fontsize=12, color=color)
            x_pos += len(token) * 0.015 + 0.02
        
        # Legend
        ax3.text(0.2, 0.7, '■', color='green', fontsize=16)
        ax3.text(0.23, 0.7, 'Known token', fontsize=12)
        ax3.text(0.5, 0.7, '■', color='red', fontsize=16)
        ax3.text(0.53, 0.7, 'Unknown token (OOV)', fontsize=12)
        
        # Summary
        ax3.text(0.5, 0.4, 
                f'Word-level OOV: {example["word_oov_count"]}/{len(example["word_tokens"])} tokens ({example["word_oov_percentage"]:.1f}%)', 
                ha='center', va='center', fontsize=12)
        ax3.text(0.5, 0.3, 
                f'BPE OOV: {example["bpe_oov_count"]}/{len(example["bpe_tokens"])} tokens ({example["bpe_oov_percentage"]:.1f}%)', 
                ha='center', va='center', fontsize=12)
        ax3.text(0.5, 0.2, 
                f'Improvement: {example["word_oov_percentage"] - example["bpe_oov_percentage"]:.1f} percentage points', 
                ha='center', va='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"OOV comparison visualization saved to {output_file}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Demonstrate BPE handling of OOV words")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to the trained BPE model")
    parser.add_argument("--train-file", type=str, required=True,
                        help="Path to the training corpus file")
    parser.add_argument("--test-file", type=str, required=True,
                        help="Path to the test corpus file")
    parser.add_argument("--word-vocab-size", type=int, default=100,
                        help="Size of the word-level vocabulary")
    parser.add_argument("--output-prefix", type=str, default="oov_comparison",
                        help="Prefix for output files")
    args = parser.parse_args()
    
    # Load the BPE tokenizer
    tokenizer = BPETokenizer.load(args.model)
    
    # Load the training and test corpora
    with open(args.train_file, 'r', encoding='utf-8') as f:
        train_corpus = f.read().split('\n\n')
        train_corpus = [p.strip() for p in train_corpus if p.strip()]
    
    with open(args.test_file, 'r', encoding='utf-8') as f:
        test_corpus = f.read().split('\n\n')
        test_corpus = [p.strip() for p in test_corpus if p.strip()]
    
    # Analyze OOV handling
    analysis = analyze_oov_handling(train_corpus, test_corpus, tokenizer, args.word_vocab_size)
    
    # Print summary
    print(f"Training corpus: {analysis['train_corpus_size']} documents")
    print(f"Test corpus: {analysis['test_corpus_size']} documents")
    print(f"Word vocabulary size: {analysis['word_vocab_size']}")
    print(f"BPE vocabulary size: {analysis['bpe_vocab_size']}")
    
    print("\nOverall OOV statistics:")
    print(f"  Word-level: {analysis['overall']['total_word_oov']}/{analysis['overall']['total_words']} tokens "
          f"({analysis['overall']['word_oov_percentage']:.2f}%)")
    print(f"  BPE: {analysis['overall']['total_bpe_oov']}/{analysis['overall']['total_bpe_tokens']} tokens "
          f"({analysis['overall']['bpe_oov_percentage']:.2f}%)")
    
    # Visualize OOV comparison
    visualize_oov_comparison(analysis, f"{args.output_prefix}.png")
    
    # Save analysis to JSON
    with open(f"{args.output_prefix}.json", 'w', encoding='utf-8') as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    
    print(f"Analysis saved to {args.output_prefix}.json")


if __name__ == "__main__":
    main()
