# First, install gensim if you haven't already
# pip install gensim

import numpy
from gensim.models import Word2Vec

# Sample sentences (tokenized into words)
sentences = [
    ['the', 'cat', 'sat', 'on', 'the', 'mat'],
    ['the', 'dog', 'ran', 'in', 'the', 'park'],
    ['cat', 'and', 'dog', 'are', 'friends']
]

# Train Word2Vec model
# Parameters:
# - vector_size: dimension of the word vectors (e.g., 100)
# - window: max distance between current and predicted word
# - min_count: ignores words with frequency lower than this
# - workers: number of worker threads to train the model
model = Word2Vec(sentences, vector_size=128, window=5, min_count=1, workers=4)

# Save the model (optional)
model.save("simple_word2vec.model")

# Access the word vector for a specific word
cat_vector = model.wv['cat']
print("Vector for 'cat':", cat_vector[:10], "...")  # Showing first 10 dimensions

# Find similar words
similar_words = model.wv.most_similar('cat', topn=3)
print("Words similar to 'cat':", similar_words)

# Calculate and print dot products for similar words
cat_vector = model.wv['cat']
print("\nDot products with 'cat':")
for word, score in similar_words:
    similar_vector = model.wv[word]
    # Calculate dot product using numpy's dot product function
    dot_product = numpy.dot(cat_vector, similar_vector)
    print(f"Dot product between 'cat' and '{word}': {dot_product}")

# Calculate similarity between two words
similarity = model.wv.similarity('cat', 'dog')
print("\nSimilarity between 'cat' and 'dog':", similarity)

