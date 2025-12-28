"""
Vector storage using memory-mapped files for efficient random access.

Features:
- Memory-mapped file I/O
- Zero-copy vector access
- Pre-allocated storage
- Efficient insert/delete operations
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple


class VectorStore:
    """
    Memory-mapped vector storage for efficient random access to large vector datasets.

    Uses numpy.memmap for zero-copy access to vector data on disk.
    """

    def __init__(
        self,
        dimension: int,
        max_vectors: int,
        data_dir: str = "./data/vectors",
        dtype: np.dtype = np.float32,
    ):
        """
        Initialize vector store.

        Args:
            dimension: Vector dimensionality
            max_vectors: Maximum number of vectors to store (pre-allocated)
            data_dir: Directory to store vector files
            dtype: Data type for vectors (default: float32)
        """
        self.dimension = dimension
        self.max_vectors = max_vectors
        self.data_dir = Path(data_dir)
        self.dtype = dtype

        # Create data directory if it doesn't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Paths for storage files
        self.vectors_path = self.data_dir / "vectors.mmap"
        self.metadata_path = self.data_dir / "metadata.npz"

        # Track current vector count
        self.count = 0

        # Initialize or load memory-mapped file
        if self.vectors_path.exists():
            # Load existing store
            self._load()
        else:
            # Create new memory-mapped file
            # Shape: (max_vectors, dimension)
            self.vectors = np.memmap(
                self.vectors_path,
                dtype=self.dtype,
                mode='w+',
                shape=(self.max_vectors, self.dimension)
            )

    def insert(self, vector: np.ndarray, vector_id: Optional[int] = None) -> int:
        """
        Insert a vector into the store.

        Args:
            vector: Vector to insert (dimension,)
            vector_id: Optional ID for the vector (auto-assigned if None)

        Returns:
            ID of the inserted vector
        """
        # Validate vector dimension
        if len(vector) != self.dimension:
            raise ValueError(f"Vector dimension {len(vector)} does not match store dimension {self.dimension}")

        # Check capacity
        if self.count >= self.max_vectors:
            raise RuntimeError(f"Vector store is full (max: {self.max_vectors})")

        # Assign ID (auto-increment for now)
        if vector_id is None:
            vector_id = self.count

        # Store the vector
        self.vectors[vector_id] = vector

        # Increment count
        self.count += 1

        # Flush to disk
        self.vectors.flush()

        return vector_id

    def get(self, vector_id: int) -> np.ndarray:
        """
        Retrieve a vector by ID.

        Args:
            vector_id: ID of the vector to retrieve

        Returns:
            Vector data (dimension,)
        """
        # Validate ID
        if vector_id < 0 or vector_id >= self.count:
            raise ValueError(f"Invalid vector_id {vector_id}. Valid range: [0, {self.count})")

        # Return the vector (creates a copy to avoid mutation)
        return np.array(self.vectors[vector_id])

    def delete(self, vector_id: int) -> None:
        """
        Delete a vector (lazy deletion with tombstone).

        Args:
            vector_id: ID of the vector to delete
        """
        # TODO: Implement lazy deletion
        pass

    def get_all_vectors(self) -> np.ndarray:
        """
        Get all non-deleted vectors.

        Returns:
            Array of vectors (n_vectors x dimension)
        """
        # Return only the vectors we've inserted (0 to count)
        return np.array(self.vectors[:self.count])

    def close(self) -> None:
        """Close the vector store and flush to disk."""
        # Save metadata (vector count)
        np.savez(self.metadata_path, count=self.count)

        # Flush memory-mapped file
        if hasattr(self, 'vectors'):
            self.vectors.flush()

    def _load(self) -> None:
        """Load existing vector store from disk."""
        # Load metadata
        if self.metadata_path.exists():
            metadata = np.load(self.metadata_path)
            self.count = int(metadata['count'])
        else:
            self.count = 0

        # Load memory-mapped file
        self.vectors = np.memmap(
            self.vectors_path,
            dtype=self.dtype,
            mode='r+',
            shape=(self.max_vectors, self.dimension)
        )
