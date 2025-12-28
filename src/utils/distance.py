"""
Optimized distance metric implementations for vector similarity.

Supports:
- Cosine similarity
- Euclidean distance
- Dot product

Uses NumPy and Numba for performance optimization.
"""

import numpy as np
from typing import Union
from numba import jit


@jit(nopython=True)
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Formula: (A·B)/(||A||·||B||)

    Args:
        a: First vector (numpy array)
        b: Second vector (numpy array)

    Returns:
        Cosine similarity (0 = orthogonal, 1 = identical, -1 = opposite)
    """
    # Compute dot product
    dot = 0.0
    for i in range(len(a)):
        dot += a[i] * b[i]

    # Compute magnitudes (L2 norms)
    norm_a = 0.0
    norm_b = 0.0
    for i in range(len(a)):
        norm_a += a[i] * a[i]
        norm_b += b[i] * b[i]

    norm_a = np.sqrt(norm_a)
    norm_b = np.sqrt(norm_b)

    # Avoid division by zero
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    # Return cosine similarity
    return dot / (norm_a * norm_b)


@jit(nopython=True)
def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Euclidean (L2) distance between two vectors.

    Formula: sqrt(Σ(A_i - B_i)²)

    Args:
        a: First vector (numpy array)
        b: Second vector (numpy array)

    Returns:
        Euclidean distance (0 = identical, larger = more different)
    """
    # Compute squared differences element-wise
    diff = a - b

    # Sum of squares
    squared_sum = 0.0
    for i in range(len(diff)):
        squared_sum += diff[i] * diff[i]

    # Return square root
    return np.sqrt(squared_sum)


@jit(nopython=True)
def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute dot product between two vectors.

    Formula: A·B = Σ(a[i] × b[i])

    Args:
        a: First vector (numpy array)
        b: Second vector (numpy array)

    Returns:
        Dot product (higher = more similar for normalized vectors)
    """
    result = 0.0
    for i in range(len(a)):
        result += a[i] * b[i]
    return result


def batch_cosine_similarity(queries: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between query vectors and a batch of vectors.

    Args:
        queries: Query vectors (n_queries x dimension)
        vectors: Database vectors (n_vectors x dimension)

    Returns:
        Similarity matrix (n_queries x n_vectors)
    """
    # TODO: Implement batch cosine similarity
    pass


def batch_euclidean_distance(queries: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """
    Compute Euclidean distance between query vectors and a batch of vectors.

    Args:
        queries: Query vectors (n_queries x dimension)
        vectors: Database vectors (n_vectors x dimension)

    Returns:
        Distance matrix (n_queries x n_vectors)
    """
    # TODO: Implement batch Euclidean distance
    pass
