# Byte Pair Encoding (BPE) Core Implementation

This is a clean, focused implementation of the Byte Pair Encoding (BPE) algorithm from scratch, without any visualization components. It provides the core functionality needed to train a BPE tokenizer, encode text to token IDs, and decode token IDs back to text.

## What is BPE?

Byte Pair Encoding (BPE) is a subword tokenization algorithm used in many modern language models. It starts with character-level tokenization and iteratively merges the most frequent pairs of adjacent tokens until a desired vocabulary size is reached.

BPE provides a good balance between character-level and word-level tokenization, allowing models to handle both common words and rare or out-of-vocabulary words effectively.

## Features

- Complete BPE training from a text corpus
- Encoding text to token IDs
- Decoding token IDs back to text
- Saving and loading trained models
- Special token handling (PAD, UNK, BOS, EOS)
- Command-line interface for training and testing
- Case-preserving tokenization

## Usage

### Basic Usage

```python
from bpe import BPETokenizer

# Create a new tokenizer with a target vocabulary size
tokenizer = BPETokenizer(vocab_size=1000)

# Train on a corpus of text
corpus = [
    "Natural language processing helps computers understand human language.",
    "Deep learning has revolutionized artificial intelligence research.",
    # ... more training sentences
]
tokenizer.train(corpus, verbose=True)

# Encode text to token IDs
text = "This is an example sentence."
token_ids = tokenizer.encode(text)
print(f"Encoded: {token_ids}")

# Decode token IDs back to text
decoded_text = tokenizer.decode(token_ids)
print(f"Decoded: {decoded_text}")

# Save the trained model
tokenizer.save("bpe_model.json")

# Load a saved model
loaded_tokenizer = BPETokenizer.load("bpe_model.json")
```

### Command-Line Interface

The module can also be run as a script:

```bash
# Train a new BPE model with default settings
python bpe.py --verbose

# Train with a specific vocabulary size
python bpe.py --vocab-size 1000 --verbose

# Train using a custom corpus file
python bpe.py --corpus-file my_corpus.txt --vocab-size 1000 --verbose

# Load a pre-trained model
python bpe.py --load-path bpe_model.json
```

## Implementation Details

The implementation follows these key steps:
1. Character-level tokenization of the input corpus
2. Counting frequencies of adjacent token pairs
3. Iteratively merging the most frequent pairs
4. Applying the learned merges to tokenize new text

The code is thoroughly commented and includes type hints for better readability and maintainability.

## Requirements

- Python 3.6+
- No external dependencies beyond the Python standard library

## License

MIT
