# Byte Pair Encoding (BPE) Implementation and Visualization

This project provides a complete implementation of the Byte Pair Encoding (BPE) algorithm used in modern large language models (LLMs) for subword tokenization, along with high-quality visualizations for educational and presentation purposes.

## What is BPE?

Byte Pair Encoding is a data compression and tokenization algorithm that iteratively merges the most frequent pairs of adjacent tokens (starting with characters) to form new tokens. It was adapted for NLP by [Sennrich et al. (2016)](https://arxiv.org/abs/1508.07909) and has become a fundamental component in modern language models like GPT and BERT.

BPE offers several advantages:
- Handles out-of-vocabulary words by breaking them into subword units
- Balances vocabulary size and token length
- Effectively represents both common words and rare words/morphemes

## Features

This implementation includes:
- Complete BPE training from a text corpus
- Encoding text to token IDs
- Decoding token IDs back to text
- Saving and loading trained models
- Special token handling (PAD, UNK, BOS, EOS)
- Command-line interface for training and testing
- High-quality visualizations of the BPE process
- Comparative analysis of tokenization methods
- Language modeling visualization
- Comprehensive dashboard for presentations

## How BPE Works

1. **Initialization**: Start with a vocabulary of individual characters
2. **Counting**: Count frequencies of adjacent token pairs in the corpus
3. **Merging**: Iteratively merge the most frequent pair to create a new token
4. **Vocabulary Building**: Add the new merged token to the vocabulary
5. **Repeat**: Continue until reaching the desired vocabulary size or minimum frequency threshold

## Usage

### Running the Example

```bash
# Train a new BPE model with default settings
python BPEexample.py

# Train with a specific vocabulary size
python BPEexample.py --vocab-size 1000

# Train with verbose output
python BPEexample.py --verbose

# Save to a specific path
python BPEexample.py --save-path my_bpe_model.json

# Load a previously trained model
python BPEexample.py --load-path my_bpe_model.json
```

### Using the BPETokenizer in Your Code

```python
from BPEexample import BPETokenizer, generate_sample_corpus

# Train a new tokenizer
corpus = generate_sample_corpus()  # Or your own list of texts
tokenizer = BPETokenizer(vocab_size=500)
tokenizer.train(corpus, verbose=True)

# Save the trained model
tokenizer.save("my_bpe_model.json")

# Load a pre-trained model
tokenizer = BPETokenizer.load("my_bpe_model.json")

# Encode text to token IDs
text = "Hello, this is an example."
token_ids = tokenizer.encode(text)
print(f"Encoded: {token_ids}")

# Decode token IDs back to text
decoded_text = tokenizer.decode(token_ids)
print(f"Decoded: {decoded_text}")
```

## Implementation Details

The implementation follows these key steps:
1. Character-level tokenization of the input corpus
2. Counting frequencies of adjacent token pairs
3. Iteratively merging the most frequent pairs
4. Applying the learned merges to tokenize new text

The code is thoroughly commented and includes type hints for better readability and maintainability.

## Visualizations

This project includes several high-quality visualizations to help understand BPE tokenization:

### BPE Dashboard

A comprehensive dashboard that combines multiple visualizations into a single presentation-quality display. The dashboard includes:
- Text sample display
- BPE vocabulary statistics
- Tokenization comparison (character, word, and BPE)
- Language modeling visualization
- Token distribution analysis
- Token counts comparison
- BPE process visualization

```bash
# Generate the dashboard
python bpe_dashboard.py --model model.json --text-file sample.txt --output visualizations/bpe_dashboard.png
```

### BPE Process Visualization

A step-by-step visualization of how BPE tokenization works, showing each merge operation in the process:

```bash
# Visualize the BPE process
python visualize_bpe.py --model model.json --text "language model" --output visualizations/bpe_process.png
```

### Tokenization Comparison

Compare character-level, word-level, and BPE tokenization approaches with color-coded tokens:

```bash
# Compare tokenization methods
python compare_tokenization.py --model model.json --text "This is an example of tokenization." --output visualizations/tokenization_comparison.png
```

### Language Modeling Visualization

Visualize how BPE tokenization helps with language modeling by showing parameter sharing across similar words:

```bash
# Visualize language modeling with BPE
python language_modeling_viz_fixed.py --output visualizations/language_modeling.png
```

### Token Distribution and Counts

Analyze the distribution and counts of tokens across different tokenization methods:

```bash
# Generate token distribution visualization
python compare_tokenization.py --model model.json --text-file sample.txt --distribution-output visualizations/token_distribution.png

# Generate token counts visualization
python compare_tokenization.py --model model.json --text-file sample.txt --counts-output visualizations/token_counts.png
```

All visualizations are saved in the `visualizations/` directory.

## Applications in LLMs

BPE tokenization is a critical component in modern language models:
- Reduces vocabulary size while maintaining expressiveness
- Handles rare words and morphological variations
- Enables efficient training and inference
- Provides a balance between character and word tokenization

## Project Structure

The project is organized as follows:

```
.
├── BPEexample.py           # Core BPE implementation
├── bpe_dashboard.py        # Comprehensive visualization dashboard
├── cleanup.py              # Utility to clean up temporary files
├── compare_tokenization.py # Tokenization comparison visualizations
├── language_modeling_viz.py # Language modeling visualization
├── language_modeling_viz_fixed.py # Fixed version without boxstyle parameter
├── model.json              # Trained BPE model
├── README.md               # This file
├── requirements.txt        # Project dependencies
├── sample.txt              # Sample text for visualization
├── test_corpus.txt         # Sample corpus for training
├── visualize_bpe.py        # BPE process visualization
└── visualizations/         # Directory for visualization outputs
```

## Installation

To set up the project, clone the repository and install the dependencies:

```bash
# Clone the repository
git clone <repository-url>
cd <repository-directory>

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Dependencies

This project requires the following Python packages:

```
# Core dependencies
matplotlib>=3.5.0
numpy>=1.20.0

# Visualization dependencies
seaborn>=0.11.0
pandas>=1.3.0

# Image processing
Pillow>=8.0.0

# Utilities
tqdm>=4.60.0
argparse>=1.4.0
```

## References

- Sennrich, R., Haddow, B., & Birch, A. (2016). [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)
- Gage, P. (1994). A New Algorithm for Data Compression
- OpenAI's GPT models use a variant called Byte-Level BPE
- Radford, A., et al. (2019). [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- Google's BERT uses WordPiece, a similar subword tokenization method
