"""
Abstract base class for vector indices.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple


class VectorIndex(ABC):
    """Abstract base class for all vector index implementations."""

    @abstractmethod
    def build(self, vectors: np.ndarray) -> None:
        """
        Build the index from a set of vectors.

        Args:
            vectors: Array of vectors (n_vectors x dimension)
        """
        pass

    @abstractmethod
    def insert(self, vector: np.ndarray, vector_id: int) -> None:
        """
        Insert a single vector into the index.

        Args:
            vector: Vector to insert (dimension,)
            vector_id: ID of the vector
        """
        pass

    @abstractmethod
    def search(self, query: np.ndarray, k: int) -> List[Tuple[int, float]]:
        """
        Search for k nearest neighbors.

        Args:
            query: Query vector (dimension,)
            k: Number of neighbors to return

        Returns:
            List of (vector_id, distance) tuples sorted by distance
        """
        pass

    @abstractmethod
    def delete(self, vector_id: int) -> None:
        """
        Remove a vector from the index.

        Args:
            vector_id: ID of the vector to remove
        """
        pass

    @abstractmethod
    def save(self, filepath: str) -> None:
        """
        Save the index to disk.

        Args:
            filepath: Path to save the index
        """
        pass

    @abstractmethod
    def load(self, filepath: str) -> None:
        """
        Load the index from disk.

        Args:
            filepath: Path to load the index from
        """
        pass
