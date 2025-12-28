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
        self.metric = metric
        self.vectors: np.ndarray = None
        self.vector_ids: List[int] = []

    def build(self, vectors: np.ndarray) -> None:
        """Build the index from vectors."""
        # TODO: Implement build
        pass

    def insert(self, vector: np.ndarray, vector_id: int) -> None:
        """Insert a vector."""
        # TODO: Implement insert
        pass

    def search(self, query: np.ndarray, k: int) -> List[Tuple[int, float]]:
        """
        Search for k nearest neighbors using brute force.

        Computes distance to all vectors and returns top-k.
        """
        # TODO: Implement search
        pass

    def delete(self, vector_id: int) -> None:
        """Delete a vector."""
        # TODO: Implement delete
        pass

    def save(self, filepath: str) -> None:
        """Save index to disk."""
        # TODO: Implement save
        pass

    def load(self, filepath: str) -> None:
        """Load index from disk."""
        # TODO: Implement load
        pass
