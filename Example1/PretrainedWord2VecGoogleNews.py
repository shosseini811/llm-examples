from gensim.models import KeyedVectors

# Load the pretrained Word2Vec model (Google News vectors)
model_path = "./models/GoogleNews-vectors-negative300.bin.gz"
word_vectors = KeyedVectors.load_word2vec_format(model_path, binary=True)

# Access the word vector for a specific word (e.g., 'cat')
cat_vector = word_vectors['cat']
print("Vector for 'cat' (first 10 dimensions):", cat_vector[:10], "...")

# Find similar words to 'cat'
similar_words = word_vectors.most_similar('cat', topn=5)
print("Words similar to 'cat':", similar_words)

# Calculate and print similarity between 'cat' and 'dog'
dog_vector = word_vectors['dog']
similarity_cat_dog = word_vectors.similarity('cat', 'dog')
print(f"\nSimilarity between 'cat' and 'dog': {similarity_cat_dog}")

# Calculate and print similarity between 'cat' and 'sat'
sat_vector = word_vectors['sat']
similarity_cat_sat = word_vectors.similarity('cat', 'sat')
print(f"\nSimilarity between 'cat' and 'sat': {similarity_cat_sat}")

# calculate and print similarity between 'male' and 'king'
male_vector = word_vectors['male']
king_vector = word_vectors['female']
similarity_male_king = word_vectors.similarity('male', 'female')
print(f"\nSimilarity between 'male' and 'female': {similarity_male_king}")

