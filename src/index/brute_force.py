"""
Brute-force nearest neighbor search (baseline implementation).

This serves as:
1. Baseline for benchmarking HNSW performance
2. Ground truth for recall calculation
3. Fallback for small datasets
"""

import numpy as np
from typing import List, Tuple
from .base import VectorIndex
from ..utils.distance import euclidean_distance, cosine_similarity


class BruteForceIndex(VectorIndex):
    """
    Simple brute-force nearest neighbor search.

    Compares query against all vectors in the dataset.
    O(n) search time, O(n) space complexity.
    """

    def __init__(self, metric: str = "euclidean"):
        """
        Initialize brute-force index.

        Args:
            metric: Distance metric ("euclidean" or "cosine")
        """
        if metric not in ["euclidean", "cosine"]:
            raise ValueError(f"Unsupported metric: {metric}. Use 'euclidean' or 'cosine'")

        self.metric = metric
        self.vectors: np.ndarray = None
        self.vector_ids: List[int] = []

        # Select distance function
        if metric == "euclidean":
            self.distance_fn = euclidean_distance
        else:  # cosine
            self.distance_fn = cosine_similarity

    def build(self, vectors: np.ndarray) -> None:
        """Build the index from vectors."""
        # Store all vectors
        self.vectors = np.array(vectors)

        # Auto-assign vector IDs (0, 1, 2, ...)
        self.vector_ids = list(range(len(vectors)))

    def insert(self, vector: np.ndarray, vector_id: int) -> None:
        """Insert a vector."""
        # Add to vectors array
        if self.vectors is None:
            self.vectors = np.array([vector])
        else:
            self.vectors = np.vstack([self.vectors, vector])

        # Add ID
        self.vector_ids.append(vector_id)

    def search(self, query: np.ndarray, k: int) -> List[Tuple[int, float]]:
        """
        Search for k nearest neighbors using brute force.

        Computes distance to all vectors and returns top-k.
        """
        if self.vectors is None or len(self.vectors) == 0:
            return []

        # Ensure k doesn't exceed number of vectors
        k = min(k, len(self.vectors))

        # Compute distance to all vectors
        distances = []
        for i, vec in enumerate(self.vectors):
            dist = self.distance_fn(query, vec)
            distances.append((self.vector_ids[i], dist))

        # Sort by distance
        # For cosine similarity, higher is better, so negate for sorting
        if self.metric == "cosine":
            # Sort descending (higher similarity = better)
            distances.sort(key=lambda x: -x[1])
        else:
            # Sort ascending (lower distance = better)
            distances.sort(key=lambda x: x[1])

        # Return top-k
        return distances[:k]

    def delete(self, vector_id: int) -> None:
        """Delete a vector."""
        # Find index of vector_id
        if vector_id not in self.vector_ids:
            raise ValueError(f"Vector ID {vector_id} not found")

        idx = self.vector_ids.index(vector_id)

        # Remove from vectors and IDs
        self.vectors = np.delete(self.vectors, idx, axis=0)
        self.vector_ids.pop(idx)

    def save(self, filepath: str) -> None:
        """Save index to disk."""
        np.savez(
            filepath,
            vectors=self.vectors,
            vector_ids=np.array(self.vector_ids),
            metric=self.metric,
        )

    def load(self, filepath: str) -> None:
        """Load index from disk."""
        data = np.load(filepath, allow_pickle=True)
        self.vectors = data['vectors']
        self.vector_ids = data['vector_ids'].tolist()
        self.metric = str(data['metric'])

        # Re-initialize distance function
        if self.metric == "euclidean":
            self.distance_fn = euclidean_distance
        else:
            self.distance_fn = cosine_similarity
