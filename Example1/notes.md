The `similarity` method in Word2Vec (from `gensim.models.Word2Vec`) calculates the **cosine similarity** between two word vectors. Here's the mathematical equation for cosine similarity, which is what `model.wv.similarity('cat', 'dog')` computes:

Given two word vectors:
- \( \mathbf{v_{cat}} \) = vector for "cat" (e.g., a 100-dimensional vector),
- \( \mathbf{v_{dog}} \) = vector for "dog" (e.g., a 100-dimensional vector),

The cosine similarity is defined as:

\[
\text{similarity}(\mathbf{v_{cat}}, \mathbf{v_{dog}}) = \cos(\theta) = \frac{\mathbf{v_{cat}} \cdot \mathbf{v_{dog}}}{\|\mathbf{v_{cat}}\| \|\mathbf{v_{dog}}\|}
\]

Where:
- \( \mathbf{v_{cat}} \cdot \mathbf{v_{dog}} \) is the **dot product** of the two vectors:
  \[
  \mathbf{v_{cat}} \cdot \mathbf{v_{dog}} = \sum_{i=1}^{n} v_{cat,i} \cdot v_{dog,i}
  \]
  (where \( n \) is the dimensionality of the vectors, e.g., 100, and \( v_{cat,i} \) is the \( i \)-th component of the "cat" vector).

- \( \|\mathbf{v_{cat}}\| \) is the **magnitude (norm)** of the "cat" vector:
  \[
  \|\mathbf{v_{cat}}\| = \sqrt{\sum_{i=1}^{n} v_{cat,i}^2}
  \]

- \( \|\mathbf{v_{dog}}\| \) is the **magnitude (norm)** of the "dog" vector:
  \[
  \|\mathbf{v_{dog}}\| = \sqrt{\sum_{i=1}^{n} v_{dog,i}^2}
  \]

### In Context:
For `similarity = model.wv.similarity('cat', 'dog')`, the vectors \( \mathbf{v_{cat}} \) and \( \mathbf{v_{dog}} \) are retrieved from the trained Word2Vec model (`model.wv['cat']` and `model.wv['dog']`), and the cosine similarity is computed using the formula above.

### Example with Small Vectors:
If:
- \( \mathbf{v_{cat}} = [1, 2] \),
- \( \mathbf{v_{dog}} = [2, 1] \),

Then:
1. Dot product: \( \mathbf{v_{cat}} \cdot \mathbf{v_{dog}} = (1 \cdot 2) + (2 \cdot 1) = 2 + 2 = 4 \),
2. Magnitude of \( \mathbf{v_{cat}} \): \( \|\mathbf{v_{cat}}\| = \sqrt{1^2 + 2^2} = \sqrt{5} \),
3. Magnitude of \( \mathbf{v_{dog}} \): \( \|\mathbf{v_{dog}}\| = \sqrt{2^2 + 1^2} = \sqrt{5} \),
4. Cosine similarity: \( \frac{4}{\sqrt{5} \cdot \sqrt{5}} = \frac{4}{5} = 0.8 \).

In Word2Vec, this process is applied to higher-dimensional vectors (e.g., 100 dimensions), but the math remains the same!

---------

The `most_similar` method in Word2Vec (from `gensim.models.Word2Vec`) finds the top \( n \) words whose vectors have the highest cosine similarity to a given word's vector. Here's the mathematical foundation for `similar_words = model.wv.most_similar('cat', topn=3)`:

### Cosine Similarity Recap
For any two vectors \( \mathbf{v_a} \) (e.g., the vector for "cat") and \( \mathbf{v_b} \) (vector for another word in the vocabulary), the cosine similarity is:

\[
\text{similarity}(\mathbf{v_a}, \mathbf{v_b}) = \cos(\theta) = \frac{\mathbf{v_a} \cdot \mathbf{v_b}}{\|\mathbf{v_a}\| \|\mathbf{v_b}\|}
\]

Where:
- \( \mathbf{v_a} \cdot \mathbf{v_b} = \sum_{i=1}^{n} v_{a,i} \cdot v_{b,i} \) (dot product),
- \( \|\mathbf{v_a}\| = \sqrt{\sum_{i=1}^{n} v_{a,i}^2} \) (magnitude of \( \mathbf{v_a} \)),
- \( \|\mathbf{v_b}\| = \sqrt{\sum_{i=1}^{n} v_{b,i}^2} \) (magnitude of \( \mathbf{v_b} \)),
- \( n \) is the dimensionality of the vectors (e.g., 100 in the earlier example).

### The `most_similar` Process
For `model.wv.most_similar('cat', topn=3)`:
1. **Input Vector**: Let \( \mathbf{v_{cat}} \) be the vector for "cat" (e.g., `model.wv['cat']`).
2. **Vocabulary**: Let \( V \) be the set of all word vectors in the model's vocabulary (excluding "cat" itself, typically).
3. **Compute Similarities**: For each word \( w \) in \( V \) with vector \( \mathbf{v_w} \), calculate:
   \[
   s_w = \frac{\mathbf{v_{cat}} \cdot \mathbf{v_w}}{\|\mathbf{v_{cat}}\| \|\mathbf{v_w}\|}
   \]
4. **Sort and Select**: Sort all \( s_w \) values in descending order and select the top 3 (since `topn=3`).

### Mathematical Representation
Define the similarity score for each word \( w \) in the vocabulary:
\[
s_w = \text{cosine_similarity}(\mathbf{v_{cat}}, \mathbf{v_w}) = \frac{\sum_{i=1}^{n} v_{cat,i} \cdot v_{w,i}}{\sqrt{\sum_{i=1}^{n} v_{cat,i}^2} \cdot \sqrt{\sum_{i=1}^{n} v_{w,i}^2}}
\]

Then, `most_similar('cat', topn=3)` returns:
\[
\text{top 3 words} = \arg\text{top}_3 \{ s_w \mid w \in V \}
\]
Where \( \arg\text{top}_3 \) denotes the 3 words \( w \) with the highest \( s_w \) values, typically returned as a list of tuples \( (w, s_w) \).

### In Practice
- \( \mathbf{v_{cat}} \) is fixed (e.g., a 100-dimensional vector from the model).
- For every other word \( w \) in the vocabulary (e.g., "dog", "mat", "park"), compute \( s_w \).
- Sort all \( s_w \) and pick the top 3, e.g., \( [("dog", 0.95), ("mat", 0.87), ("park", 0.76)] \).

### Simplified Example
If the vocabulary is small, say \( V = \{\text{"dog"}, \text{"mat"}, \text{"park"}\} \), and vectors are 2D:
- \( \mathbf{v_{cat}} = [1, 2] \),
- \( \mathbf{v_{dog}} = [2, 1] \), \( s_{dog} = 0.8 \),
- \( \mathbf{v_{mat}} = [1, 0] \), \( s_{mat} = 0.447 \),
- \( \mathbf{v_{park}} = [0, 1] \), \( s_{park} = 0.894 \),

Sort \( [0.8, 0.447, 0.894] \) → top 3: \( [("park", 0.894), ("dog", 0.8), ("mat", 0.447)] \). (Here, all are returned since there are only 3 words, but with `topn=3`, it limits to 3 even in a larger vocab.)

This is exactly what `most_similar` does, just scaled to the full vocabulary and higher dimensions!