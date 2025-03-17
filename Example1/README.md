# Word2Vec Implementation and Custom Similarity Functions

This project demonstrates the implementation of Word2Vec word embeddings using both the Gensim library and custom implementations. It includes examples of computing word similarities and finding similar words using cosine similarity.

## Project Structure

- `test.py`: Basic Word2Vec implementation using Gensim
- `custom_similarity.py`: Custom implementation of similarity functions
- `notes.md`: Mathematical explanations and theory
- `requirements.txt`: Project dependencies

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Features

### 1. Basic Word2Vec Usage (`test.py`)
- Training a simple Word2Vec model
- Accessing word vectors
- Finding similar words
- Computing word similarities

### 2. Custom Implementations (`custom_similarity.py`)
Two main functions are implemented:

#### `calculate_cosine_similarity(vec1, vec2)`
- Computes cosine similarity between two word vectors
- Implements the formula: cos(θ) = (A·B)/(||A||×||B||)
- Returns a similarity score between -1 and 1

#### `find_most_similar(target_vector, word_vectors_dict, topn=3)`
- Finds the most similar words to a target word
- Uses cosine similarity to compare with all words in vocabulary
- Returns top N similar words with their similarity scores

## Mathematical Foundation

The project implements two key similarity metrics:

1. **Cosine Similarity**:
```
similarity(vₐ, vᵦ) = cos(θ) = (vₐ · vᵦ)/(||vₐ|| ||vᵦ||)
```
Where:
- vₐ · vᵦ is the dot product
- ||v|| is the magnitude of vector v

2. **Most Similar Words**:
- Computes cosine similarity with all words in vocabulary
- Sorts results by similarity score
- Returns top N most similar words

## Example Usage

```python
# Using Gensim's implementation
from gensim.models import Word2Vec

# Train model
sentences = [['the', 'cat', 'sat'], ['the', 'dog', 'ran']]
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# Find similar words
similar_words = model.wv.most_similar('cat', topn=3)

# Using custom implementation
from custom_similarity import calculate_cosine_similarity, find_most_similar

# Create word vectors dictionary
word_vectors = {word: model.wv[word] for word in model.wv.index_to_key}

# Find similar words using custom implementation
custom_similar = find_most_similar(model.wv['cat'], word_vectors, topn=3)
```

## Dependencies

- gensim==4.3.3
- numpy==1.24.4
- scipy==1.10.1
- smart-open==7.1.0
- wrapt==1.17.2

## Further Reading

For detailed mathematical explanations and theory behind Word2Vec and cosine similarity, refer to `notes.md` in the repository.

## Contributing

Feel free to submit issues and enhancement requests!
