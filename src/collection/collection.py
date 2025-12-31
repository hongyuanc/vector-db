"""
Collection class that wraps VectorStore and HNSW index together.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

from ..storage.vector_store import VectorStore
from ..index.hnsw import HNSWIndex


class Collection:
    """
    A collection manages a vector store and its HNSW index.

    Provides a unified interface for:
    - Inserting vectors
    - Searching for nearest neighbors
    - Persistence (save/load)
    - Configuration management
    """

    def __init__(
        self,
        name: str,
        dimension: int,
        data_dir: str = "./data",
        max_vectors: int = 1_000_000,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        metric: str = "euclidean",
    ):
        """
        Initialize a collection.

        Args:
            name: Collection name
            dimension: Vector dimension
            data_dir: Base directory for data storage
            max_vectors: Maximum number of vectors
            M: HNSW M parameter (max connections per node)
            ef_construction: HNSW ef_construction parameter
            ef_search: HNSW ef_search parameter (default for searches)
            metric: Distance metric ("euclidean", "cosine", "dot_product")
        """
        self.name = name
        self.dimension = dimension
        self.data_dir = Path(data_dir) / name
        self.max_vectors = max_vectors
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.metric = metric

        # Create data directory
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize vector store
        self.store = VectorStore(
            dimension=dimension,
            max_vectors=max_vectors,
            data_dir=str(self.data_dir)
        )

        # Initialize HNSW index
        self.index = HNSWIndex(
            M=M,
            ef_construction=ef_construction,
            ef_search=ef_search,
            metric=metric
        )

        # Track whether index is built
        self._index_built = False

        # Try to load existing data
        self._try_load()

    def _try_load(self):
        """Try to load existing collection data."""
        index_path = self.data_dir / "index.npz"
        config_path = self.data_dir / "config.json"

        if index_path.exists() and config_path.exists():
            # Load index
            self.index.load(str(index_path))
            self._index_built = True
            print(f"Loaded existing index from {index_path}")

            # Load config
            with open(config_path, "r") as f:
                config = json.load(f)
            print(f"Loaded collection config: {config}")

    def insert(self, vector: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Insert a single vector.

        Args:
            vector: Vector to insert (1D array)
            metadata: Optional metadata (currently not used)

        Returns:
            Vector ID
        """
        # Validate dimension
        if len(vector) != self.dimension:
            raise ValueError(f"Vector dimension {len(vector)} != {self.dimension}")

        # Insert into store
        vector_id = self.store.insert(vector)

        # Add to index
        if not self._index_built:
            # Build index from scratch when first vector added
            # For now, just track that we need to rebuild
            pass
        else:
            # Incremental insert (not yet implemented in HNSW)
            # For now, mark index as needing rebuild
            self._index_built = False

        return vector_id

    def batch_insert(self, vectors: np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None) -> List[int]:
        """
        Insert multiple vectors in batch.

        Args:
            vectors: Vectors to insert (2D array, shape: [n_vectors, dimension])
            metadata: Optional metadata for each vector

        Returns:
            List of vector IDs
        """
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Vector dimension {vectors.shape[1]} != {self.dimension}")

        vector_ids = []
        for vector in vectors:
            vector_id = self.store.insert(vector)
            vector_ids.append(vector_id)

        # Mark index as needing rebuild
        self._index_built = False

        return vector_ids

    def build_index(self):
        """Build or rebuild the HNSW index from all vectors in the store."""
        if self.store.count == 0:
            raise ValueError("Cannot build index on empty collection")

        # Get all vectors from store
        vectors = self.store.vectors[:self.store.count]

        # Build HNSW index
        print(f"Building index for {self.store.count} vectors...")
        self.index.build(vectors)
        self._index_built = True
        print("Index built successfully")

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        ef: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[int, float]]:
        """
        Search for k nearest neighbors.

        Args:
            query: Query vector (1D array)
            k: Number of neighbors to return
            ef: Search width (overrides default ef_search)
            filter: Metadata filter (not yet implemented)

        Returns:
            List of (vector_id, distance) tuples
        """
        if len(query) != self.dimension:
            raise ValueError(f"Query dimension {len(query)} != {self.dimension}")

        if not self._index_built:
            # Auto-build index if needed
            if self.store.count > 0:
                self.build_index()
            else:
                return []

        # Search using HNSW
        results = self.index.search(query, k=k, ef=ef)

        # TODO: Apply metadata filter if provided

        return results

    def save(self):
        """Save collection state to disk."""
        # Save HNSW index
        index_path = self.data_dir / "index.npz"
        if self._index_built:
            self.index.save(str(index_path))
            print(f"Saved index to {index_path}")

        # Save config
        config = {
            "name": self.name,
            "dimension": self.dimension,
            "max_vectors": self.max_vectors,
            "M": self.M,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
            "metric": self.metric,
            "vector_count": self.store.count,
            "index_built": self._index_built,
        }
        config_path = self.data_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Saved config to {config_path}")

        # VectorStore auto-saves on insert, but let's close it properly
        self.store.close()

    def close(self):
        """Close the collection and save state."""
        self.save()

    @property
    def count(self) -> int:
        """Get number of vectors in collection."""
        return self.store.count

    @property
    def is_index_built(self) -> bool:
        """Check if index is built."""
        return self._index_built
