#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Byte-Pair Encoding (BPE) Implementation

This module implements BPE tokenization from scratch, inspired by but different
from the Hugging Face implementation.
"""

import re
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


class WordTokenizer:
    """Simple word-level tokenizer for pre-tokenization."""
    
    @staticmethod
    def tokenize(text: str) -> List[Tuple[str, int]]:
        """Split text into words and return with offsets."""
        words = []
        offset = 0
        # Add spaces before punctuation for better subword tokenization
        text = re.sub(r'([.,!?])', r' \1', text)
        # Split on whitespace
        for match in re.finditer(r'\S+', text):
            word = match.group()
            words.append((word, offset))
            offset = match.end()
        return words


class BPETokenizer:
    """
    Basic BPE tokenizer implementation.
    
    This implementation follows a similar approach to standard BPE but with
    some modifications for simplicity and educational purposes.
    """
    
    def __init__(self, target_vocab_size: int = 50):
        """
        Initialize a new BPE tokenizer.
        
        Args:
            target_vocab_size: Target size for final vocabulary
        """
        self.target_vocab_size = target_vocab_size
        self.word_tokenizer = WordTokenizer()
        
        # Initialize basic vocabulary
        self.vocab = ["<|eos|>"]  # End of sequence token
        self.char_to_id = {}
        self.id_to_char = {}
        self.splits = {}
        self.merges = {}
        
        # Track word frequencies for training
        self.word_freqs = defaultdict(int)
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text before tokenization.
        
        Args:
            text: Input text to preprocess
            
        Returns:
            Preprocessed text
        """
        # Add space before punctuation and special characters
        text = re.sub(r'([.,!?()])', r' \1 ', text)
        # Standardize whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def train(self, corpus: List[str]) -> None:
        """
        Train the BPE tokenizer on a text corpus.
        
        Args:
            corpus: List of training texts
        """
        # Process and count word frequencies
        for text in corpus:
            text = self._preprocess_text(text)
            # Pre-tokenize text into words
            words_with_offsets = self.word_tokenizer.tokenize(text)
            for word, _ in words_with_offsets:
                self.word_freqs[word] += 1
        
        # Build initial character vocabulary
        alphabet = set()
        for word in self.word_freqs:
            for char in word:
                alphabet.add(char)
        
        # Add characters to vocabulary
        for char in sorted(alphabet):
            self.vocab.append(char)
            self.char_to_id[char] = len(self.vocab) - 1
            self.id_to_char[len(self.vocab) - 1] = char
        
        # Initialize splits for each word
        self.splits = {word: [c for c in word] for word in self.word_freqs}
        
        # Main training loop
        while len(self.vocab) < self.target_vocab_size:
            # Count pair frequencies
            pair_freqs = self._compute_pair_frequencies()
            if not pair_freqs:
                break
            
            # Find best pair to merge
            best_pair = max(pair_freqs.items(), key=lambda x: x[1])[0]
            
            # Create new token and add to vocabulary
            new_token = best_pair[0] + best_pair[1]
            self.vocab.append(new_token)
            idx = len(self.vocab) - 1
            
            # Update mappings
            self.merges[best_pair] = new_token
            
            # Apply the merge to all splits
            self._apply_merge(best_pair)
    
    def _compute_pair_frequencies(self) -> Dict[Tuple[str, str], int]:
        """
        Count frequencies of adjacent token pairs in current splits.
        
        Returns:
            Dictionary mapping token pairs to their frequencies
        """
        pair_freqs = defaultdict(int)
        
        for word, freq in self.word_freqs.items():
            split = self.splits[word]
            if len(split) < 2:
                continue
                
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pair_freqs[pair] += freq
        
        return pair_freqs
    
    def _apply_merge(self, pair: Tuple[str, str]) -> None:
        """
        Apply a merge operation to all splits.
        
        Args:
            pair: Pair of tokens to merge
        """
        for word in self.word_freqs:
            split = self.splits[word]
            if len(split) < 2:
                continue
            
            i = 0
            new_split = []
            while i < len(split):
                if i < len(split) - 1 and split[i] == pair[0] and split[i + 1] == pair[1]:
                    new_split.append(pair[0] + pair[1])
                    i += 2
                else:
                    new_split.append(split[i])
                    i += 1
            
            self.splits[word] = new_split
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text into token IDs.
        
        Args:
            text: Text to encode
            
        Returns:
            List of token IDs
        """
        # Pre-tokenize into words
        words_with_offsets = self.word_tokenizer.tokenize(text)
        
        # Tokenize each word
        token_ids = []
        for word, _ in words_with_offsets:
            # Start with character-level split
            split = [c for c in word]
            
            # Apply merges in order
            for pair, merged in self.merges.items():
                i = 0
                while i < len(split) - 1:
                    if split[i] == pair[0] and split[i + 1] == pair[1]:
                        split = split[:i] + [merged] + split[i + 2:]
                    else:
                        i += 1
            
            # Convert tokens to IDs
            for token in split:
                if token in self.vocab:
                    token_ids.append(self.vocab.index(token))
                else:
                    # Unknown token - use individual characters
                    for char in token:
                        if char in self.char_to_id:
                            token_ids.append(self.char_to_id[char])
                        else:
                            # Skip unknown characters
                            continue
        
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Decoded text
        """
        tokens = []
        for idx in token_ids:
            if idx < len(self.vocab):
                tokens.append(self.vocab[idx])
            else:
                tokens.append("<|unk|>")
        
        # Join tokens and clean up any artifacts
        text = "".join(tokens)
        text = re.sub(r' ([.,!?]) ', r'\1 ', text)  # Fix punctuation spacing
        return text.strip()
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text into token IDs.
        
        Args:
            text: Text to encode
            
        Returns:
            List of token IDs
        """
        # Pre-tokenize into words
        text = self._preprocess_text(text)
        words_with_offsets = self.word_tokenizer.tokenize(text)
        
        # Tokenize each word
        token_ids = []
        for word, _ in words_with_offsets:
            # Start with character-level split
            split = [c for c in word]
            
            # Apply merges in order
            for pair, merged in self.merges.items():
                i = 0
                while i < len(split) - 1:
                    if split[i] == pair[0] and split[i + 1] == pair[1]:
                        split = split[:i] + [merged] + split[i + 2:]
                    else:
                        i += 1
            
            # Convert tokens to IDs
            for token in split:
                if token in self.vocab:
                    token_ids.append(self.vocab.index(token))
                else:
                    # Unknown token - use individual characters
                    for char in token:
                        if char in self.char_to_id:
                            token_ids.append(self.char_to_id[char])
                        else:
                            # Skip unknown characters
                            continue
        
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Decoded text
        """
        tokens = []
        for idx in token_ids:
            if idx < len(self.vocab):
                tokens.append(self.vocab[idx])
            else:
                tokens.append("<|unk|>")
        
        # Join tokens and clean up any artifacts
        text = "".join(tokens)
        text = re.sub(r' ([.,!?]) ', r'\1 ', text)  # Fix punctuation spacing
        return text.strip()
    
    def save(self, path: str) -> None:
        """
        Save the tokenizer to a file.
        
        Args:
            path: Path to save the model
        """
        model_data = {
            "target_vocab_size": self.target_vocab_size,
            "vocab": self.vocab,
            "char_to_id": self.char_to_id,
            "id_to_char": {str(k): v for k, v in self.id_to_char.items()},  # Convert int keys to str for JSON
            "merges": {f"{k[0]} {k[1]}": v for k, v in self.merges.items()}
        }
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(model_data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """
        Load a tokenizer from a file.
        
        Args:
            path: Path to load the model from
            
        Returns:
            Loaded tokenizer instance
        """
        with open(path, "r", encoding="utf-8") as f:
            model_data = json.load(f)
        
        tokenizer = cls(target_vocab_size=model_data["target_vocab_size"])
        tokenizer.vocab = model_data["vocab"]
        tokenizer.char_to_id = model_data["char_to_id"]
        tokenizer.id_to_char = {int(k): v for k, v in model_data["id_to_char"].items()}
        
        # Convert merge rules back to tuples
        tokenizer.merges = {}
        for k, v in model_data["merges"].items():
            first, second = k.split()
            tokenizer.merges[(first, second)] = v
        
        return tokenizer


def main():
    """Main function to demonstrate BPE tokenization."""
    # Sample training corpus
    corpus = [
        "This is the Hugging Face Course.",
        "This chapter is about tokenization.",
        "This section shows several tokenizer algorithms.",
        "Hopefully, you will be able to understand how they are trained and generate tokens.",
        "The quick brown fox jumps over the lazy dog.",
        "BPE tokenization works by merging frequent pairs of characters.",
        "Machine learning models need good tokenization."
    ]
    
    # Create and train tokenizer
    print("Training BPE tokenizer...")
    tokenizer = BPETokenizer(target_vocab_size=100)
    tokenizer.train(corpus)
    
    # Show vocabulary
    print("\nVocabulary:")
    for i, token in enumerate(tokenizer.vocab):
        if i < 20:  # Show first 20 tokens
            print(f"{i}: {repr(token)}")
        elif i == 20:
            print("...")
    
    # Test tokenization
    test_text = "Hello world! This is a test of BPE."
    print(f"\nTest text: {test_text}")
    
    # Encode
    token_ids = tokenizer.encode(test_text)
    print(f"\nToken IDs: {token_ids}")
    
    # Show individual tokens
    tokens = [tokenizer.vocab[i] for i in token_ids]
    print(f"Tokens: {[repr(t) for t in tokens]}")
    
    # Decode
    decoded = tokenizer.decode(token_ids)
    print(f"Decoded: {decoded}")
    
    # Show some merge rules
    print("\nSome merge rules:")
    for i, (pair, merged) in enumerate(tokenizer.merges.items()):
        if i >= 5:  # Show first 5 merge rules
            break
        print(f"{repr(pair[0])} + {repr(pair[1])} -> {repr(merged)}")


if __name__ == "__main__":
    main()
