import numpy as np
from gensim.models import Word2Vec

def calculate_cosine_similarity(vec1, vec2):
    """
    Calculate cosine similarity between two vectors.
    Cosine similarity = dot product of vectors / (magnitude of vec1 * magnitude of vec2)
    """
    # Calculate dot product
    dot_product = np.dot(vec1, vec2)
    
    # Calculate magnitudes
    magnitude1 = np.sqrt(np.sum(vec1 ** 2))
    magnitude2 = np.sqrt(np.sum(vec2 ** 2))
    
    # Calculate cosine similarity
    similarity = dot_product / (magnitude1 * magnitude2)
    
    return similarity

def find_most_similar(target_vector, word_vectors_dict, topn=3):
    """
    Find the most similar words to a target word by calculating cosine similarity
    with all words in the vocabulary.
    
    Parameters:
    - target_vector: numpy array, the vector of the target word
    - word_vectors_dict: dictionary mapping words to their vectors
    - topn: int, number of most similar words to return
    
    Returns:
    - List of tuples (word, similarity_score) sorted by similarity
    """
    similarities = []
    
    # Calculate similarity with all words in vocabulary
    for word, vector in word_vectors_dict.items():
        similarity = calculate_cosine_similarity(target_vector, vector)
        similarities.append((word, similarity))
    
    # Sort by similarity score in descending order
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Return top N similar words (excluding the target word itself if present)
    result = []
    for word, similarity in similarities:
        # Skip if the vectors are identical (same word)
        if np.array_equal(word_vectors_dict[word], target_vector):
            continue
        result.append((word, similarity))
        if len(result) >= topn:
            break
            
    return result

# Example usage with your Word2Vec model:
sentences = [
    ['the', 'cat', 'sat', 'on', 'the', 'mat'],
    ['the', 'dog', 'ran', 'in', 'the', 'park'],
    ['cat', 'and', 'dog', 'are', 'friends']
]

# Train model
model = Word2Vec(sentences, vector_size=128, window=5, min_count=1, workers=4)

# Create a dictionary of word vectors for our custom function
word_vectors = {word: model.wv[word] for word in model.wv.index_to_key}

print("\nTesting similarity between 'cat' and 'dog':")
# Get word vectors
cat_vector = model.wv['cat']
dog_vector = model.wv['dog']

# Calculate similarity using our custom function
custom_similarity = calculate_cosine_similarity(cat_vector, dog_vector)
print("Custom similarity between 'cat' and 'dog':", custom_similarity)

# Compare with gensim's built-in similarity
gensim_similarity = model.wv.similarity('cat', 'dog')
print("Gensim similarity between 'cat' and 'dog':", gensim_similarity)

print("\nTesting most_similar for 'cat':")
# Find similar words using our custom function
custom_similar = find_most_similar(cat_vector, word_vectors, topn=3)
print("Custom most similar to 'cat':", custom_similar)

# Compare with gensim's built-in most_similar
gensim_similar = model.wv.most_similar('cat', topn=3)
print("Gensim most similar to 'cat':", gensim_similar)
