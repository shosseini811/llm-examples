#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SubwordTokenizer: A Byte Pair Encoding (BPE) Implementation

This module provides an implementation of the BPE algorithm for subword tokenization,
which is commonly used in modern NLP models.
"""

import re
import json
import argparse
import os
from typing import Dict, List, Tuple, Set
from collections import defaultdict


class BPETokenizer:
    """
    Subword Tokenizer using Byte Pair Encoding (BPE) algorithm.
    
    This tokenizer starts with character-level tokens and iteratively merges
    the most frequent adjacent token pairs to create a vocabulary of subword units.
    """
    
    def __init__(self, vocab_size: int = 50):
        """
        Initialize a BPE tokenizer.
        
        Args:
            vocab_size: Maximum vocabulary size (number of tokens)
        """
        self.vocab_size = vocab_size
        self.special_tokens = {
            "<PAD>": 0,   # Padding token
            "<UNK>": 1,   # Unknown token
            "<BOS>": 2,   # Beginning of sequence
            "<EOS>": 3    # End of sequence
        }
        
        # Initialize with empty vocabulary and merges
        self.vocab = {token: idx for token, idx in self.special_tokens.items()}
        self.merges = {}  # Dictionary to store merge rules
        self.vocab_reversed = {idx: token for token, idx in self.vocab.items()}
    
    def _get_stats(self, sequences: List[List[str]]) -> CounterType:
        """
        Count frequencies of adjacent token pairs in all sequences.
        
        Args:
            sequences: List of tokenized sequences
            
        Returns:
            Counter of adjacent token pairs
        """
        pairs = Counter()
        for seq in sequences:
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i+1])
                pairs[pair] += 1
        return pairs
    
    def _merge_pair(self, sequences: List[List[str]], pair: Tuple[str, str], 
                   new_token: str) -> List[List[str]]:
        """
        Apply a merge operation to all sequences.
        
        Args:
            sequences: List of tokenized sequences
            pair: Pair of tokens to merge
            new_token: New token to replace the pair with
            
        Returns:
            Updated sequences with the merge applied
        """
        new_sequences = []
        for seq in sequences:
            new_seq = []
            i = 0
            while i < len(seq):
                if i < len(seq) - 1 and seq[i] == pair[0] and seq[i+1] == pair[1]:
                    new_seq.append(new_token)
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1
            new_sequences.append(new_seq)
            
        return new_sequences
    
    def train(self, corpus: List[str], min_frequency: int = 2, 
              verbose: bool = False) -> None:
        """
        Train the BPE model on a corpus of text.
        
        Args:
            corpus: List of text documents/sentences
            min_frequency: Minimum frequency for a pair to be considered for merging
            verbose: Whether to print progress information
        """
        if verbose:
            print(f"Training BPE with target vocab size: {self.vocab_size}")
            print(f"Corpus size: {len(corpus)} documents")
        
        # Initialize with character-level tokenization
        # Split each text into characters (preserving case)
        sequences = [[c for c in text] for text in corpus]
        
        # Add all unique characters to the vocabulary
        all_chars = set()
        for text in corpus:
            all_chars.update(text)
        
        # Add characters to vocabulary
        for char in sorted(all_chars):
            if char not in self.vocab:
                self.vocab[char] = len(self.vocab)
                self.vocab_reversed[len(self.vocab_reversed)] = char
        
        num_merges = 0
        vocab_size = len(self.vocab)
        
        # Main BPE training loop
        while vocab_size < self.vocab_size:
            # Get statistics for pairs
            stats = self._get_stats(sequences)
            if not stats:
                break
                
            # Find the most frequent pair
            most_frequent = max(stats.items(), key=lambda x: x[1])
            pair, freq = most_frequent
            
            if freq < min_frequency:
                if verbose:
                    print(f"Stopping: most frequent pair occurs only {freq} times")
                break
                
            # Create a new token by joining the pair
            new_token = pair[0] + pair[1]
            
            # Add the new token to the vocabulary
            self.vocab[new_token] = len(self.vocab)
            self.vocab_reversed[len(self.vocab_reversed)] = new_token
            
            # Add the merge rule
            self.merges[pair] = new_token
            
            # Apply the merge to all sequences
            sequences = self._merge_pair(sequences, pair, new_token)
            
            num_merges += 1
            vocab_size = len(self.vocab)
            
            if verbose and num_merges % 100 == 0:
                print(f"Merge #{num_merges}: {pair} -> {new_token} (freq: {freq})")
                print(f"Current vocab size: {vocab_size}")
        
        if verbose:
            print(f"Final vocabulary size: {len(self.vocab)}")
            print(f"Total merges performed: {num_merges}")
    
    def _tokenize_word(self, word: str) -> List[str]:
        """
        Tokenize a single word using the learned BPE merges.
        
        Args:
            word: The word to tokenize
            
        Returns:
            List of subword tokens
        """
        # Start with character-level tokenization
        tokens = list(word)
        
        # Apply merges in the order they were learned
        while True:
            # Find all pairs that can be merged
            pairs = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]
            # Find the first pair that can be merged according to our rules
            for pair in pairs:
                if pair in self.merges:
                    new_token = self.merges[pair]
                    # Apply the merge
                    tokens = self._merge_pair([tokens], pair, new_token)[0]
                    break
            else:
                # No more merges possible
                break
                
        return tokens
    
    def encode(self, text: str) -> List[int]:
        """
        Encode a text string into a sequence of token IDs.
        
        Args:
            text: The text to encode
            
        Returns:
            List of token IDs
        """
        # Split text by whitespace (preserving case)
        words = text.split()
        
        # Tokenize each word and flatten the result
        tokens = []
        for word in words:
            word_tokens = self._tokenize_word(word)
            tokens.extend(word_tokens)
            # Add space token between words (except after the last word)
            if word != words[-1]:
                tokens.append(" ")
        
        # Convert tokens to IDs, using <UNK> for unknown tokens
        token_ids = []
        for token in tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                token_ids.append(self.vocab["<UNK>"])
                
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode a sequence of token IDs back to text.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Reconstructed text
        """
        tokens = [self.vocab_reversed.get(idx, "<UNK>") for idx in token_ids]
        text = "".join(tokens)
        return text
    
    def save(self, path: str) -> None:
        """
        Save the BPE model to a file.
        
        Args:
            path: Path to save the model
        """
        # Ensure both parts of the merge key are non-empty strings
        valid_merges = {}
        for k, v in self.merges.items():
            # Convert tuple keys to string representation for JSON serialization
            valid_merges[f"{k[0]} {k[1]}"] = v
            
        model = {
            "vocab_size": self.vocab_size,
            "vocab": self.vocab,
            "merges": valid_merges,
            "special_tokens": self.special_tokens
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(model, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'BPETokenizer':
        """
        Load a BPE model from a file.
        
        Args:
            path: Path to the saved model
            
        Returns:
            Loaded BPETokenizer instance
        """
        with open(path, 'r', encoding='utf-8') as f:
            model = json.load(f)
            
        tokenizer = cls(vocab_size=model["vocab_size"])
        tokenizer.vocab = model["vocab"]
        tokenizer.special_tokens = model["special_tokens"]
        
        # Convert merges back to the right format
        tokenizer.merges = {}
        for k, v in model["merges"].items():
            try:
                first, second = k.split()
                tokenizer.merges[(first, second)] = v
            except ValueError:
                # Handle the case where the key doesn't contain a space
                print(f"Warning: Skipping invalid merge rule key: {k}")
        
        # Rebuild the reversed vocabulary
        tokenizer.vocab_reversed = {idx: token for token, idx in tokenizer.vocab.items()}
        
        return tokenizer


def generate_sample_corpus() -> List[str]:
    """Generate a sample corpus for demonstration purposes."""
    return [
        "Natural language processing helps computers understand human language.",
        "Deep learning has revolutionized artificial intelligence research.",
        "Transformers have become the dominant architecture for language models.",
        "GPT stands for Generative Pre-trained Transformer.",
        "BERT is a bidirectional encoder representation from transformers.",
        "Tokenization is the process of breaking text into smaller units.",
        "Large language models are trained on vast amounts of text data.",
        "The transformer architecture uses self-attention mechanisms.",
        "Word embeddings represent words as dense vectors in a continuous space.",
        "Neural networks learn representations from data.",
        "Recurrent neural networks process sequential data.",
        "Convolutional neural networks are primarily used for image processing.",
        "Transfer learning leverages knowledge from one task to improve another.",
        "Fine-tuning adapts pre-trained models to specific downstream tasks.",
        "Attention mechanisms help models focus on relevant parts of the input.",
        "Encoder-decoder architectures are common in sequence-to-sequence tasks.",
        "Backpropagation is used to train neural networks by computing gradients.",
        "The quick brown fox jumps over the lazy dog again and again.",
        "Language modeling predicts the next word given previous context.",
        "Byte pair encoding iteratively merges the most frequent pairs of tokens.",
        "Subword tokenization balances vocabulary size and handling of rare words.",
        "Word tokenization splits text at word boundaries.",
        "Character tokenization represents each character as a separate token.",
        "Vocabulary size affects model performance and training efficiency.",
        "Out-of-vocabulary words pose challenges for language models.",
        "Regularization techniques prevent overfitting in machine learning models.",
        "Dropout randomly deactivates neurons during training."
    ]


def load_corpus_from_file(file_path: str) -> List[str]:
    """
    Load a corpus from a text file.
    
    Args:
        file_path: Path to the text file
        
    Returns:
        List of sentences/paragraphs from the file
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        # Split by paragraphs (empty lines)
        paragraphs = f.read().split('\n\n')
        # Filter out empty paragraphs
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        # Further split paragraphs into sentences for more granular training
        sentences = []
        for paragraph in paragraphs:
            # Simple sentence splitting by punctuation
            para_sentences = re.split(r'(?<=[.!?])\s+', paragraph)
            sentences.extend([s.strip() for s in para_sentences if s.strip()])
            
        return sentences


def main():
    """Main function to demonstrate BPE tokenization."""
    parser = argparse.ArgumentParser(description="Train and test a BPE tokenizer")
    parser.add_argument("--vocab-size", type=int, default=500,
                        help="Target vocabulary size")
    parser.add_argument("--save-path", type=str, default="bpe_model.json",
                        help="Path to save the trained model")
    parser.add_argument("--load-path", type=str, default=None,
                        help="Path to load a pre-trained model")
    parser.add_argument("--corpus-file", type=str, default=None,
                        help="Path to a text file to use as corpus")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed progress information")
    parser.add_argument("--min-frequency", type=int, default=2,
                        help="Minimum frequency for a pair to be considered for merging")
    args = parser.parse_args()
    
    # Either load a pre-trained model or train a new one
    if args.load_path:
        print(f"Loading BPE model from {args.load_path}")
        tokenizer = BPETokenizer.load(args.load_path)
    else:
        print(f"Training new BPE model with vocab size {args.vocab_size}")
        
        # Load corpus from file if specified, otherwise use the built-in sample corpus
        if args.corpus_file:
            print(f"Loading corpus from {args.corpus_file}")
            corpus = load_corpus_from_file(args.corpus_file)
            print(f"Loaded {len(corpus)} sentences/paragraphs")
        else:
            print("Using built-in sample corpus")
            corpus = generate_sample_corpus()
            
        tokenizer = BPETokenizer(vocab_size=args.vocab_size)
        tokenizer.train(corpus, min_frequency=args.min_frequency, verbose=args.verbose)
        
        # Save the model
        tokenizer.save(args.save_path)
        print(f"Saved model to {args.save_path}")
    
    # Demonstrate encoding and decoding
    # test_sentences = [
    #     "The quick brown fox jumps over the lazy dog.",
    #     "BPE tokenization is useful for language models.",
    #     "This is an example of encoding and decoding with BPE."
    # ]

    test_sentences = [
    "This is the Hugging Face Course.",
    "This chapter is about tokenization.",
    "This section shows several tokenizer algorithms.",
    "Hopefully, you will be able to understand how they are trained and generate tokens.",
]
    
    print("\nDemonstration of encoding and decoding:")
    for sentence in test_sentences:
        print(f"\nOriginal: {sentence}")
        
        # Encode
        token_ids = tokenizer.encode(sentence)
        tokens = [tokenizer.vocab_reversed.get(idx, "<UNK>") for idx in token_ids]
        
        print(f"Encoded (token IDs): {token_ids[:10]}{'...' if len(token_ids) > 10 else ''}")
        print(f"Tokens: {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
        
        # Decode
        decoded = tokenizer.decode(token_ids)
        print(f"Decoded: {decoded}")
    
    # Print some statistics about the vocabulary
    print(f"\nVocabulary size: {len(tokenizer.vocab)}")
    print(f"Number of merges: {len(tokenizer.merges)}")
    
    # Show some example merges
    print("\nExample merge rules:")
    for i, (pair, merged) in enumerate(list(tokenizer.merges.items())[:10]):
        print(f"  {pair[0]} + {pair[1]} -> {merged}")
    
    # Show some example vocabulary items
    print("\nExample vocabulary items:")
    vocab_items = list(tokenizer.vocab.items())
    vocab_items.sort(key=lambda x: x[1])  # Sort by token ID
    for token, idx in vocab_items[:20]:  # Show first 20 items
        if token in tokenizer.special_tokens:
            print(f"  {idx}: {token} (special token)")
        else:
            print(f"  {idx}: '{token}'")


if __name__ == "__main__":
    main()
