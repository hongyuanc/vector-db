"""
Hierarchical Navigable Small World (HNSW) index implementation.

Based on the paper:
"Efficient and robust approximate nearest neighbor search using
Hierarchical Navigable Small World graphs" by Malkov & Yashunin (2018)

Key concepts:
- Multi-layer graph structure
- Greedy search in upper layers
- Beam search in layer 0
- Probabilistic layer assignment
"""

import numpy as np
from typing import List, Tuple, Dict, Set
from dataclasses import dataclass, field
from .base import VectorIndex
from ..utils.distance import euclidean_distance, cosine_similarity


@dataclass
class HNSWNode:
    """Represents a node in the HNSW graph."""
    vector_id: int
    layer: int
    # connections[layer] = set of neighbor IDs
    connections: Dict[int, Set[int]] = field(default_factory=lambda: {})


class HNSWIndex(VectorIndex):
    """
    HNSW index for approximate nearest neighbor search.

    Parameters:
        M: Maximum number of connections per node per layer
        ef_construction: Size of dynamic candidate list during construction
        ef_search: Size of dynamic candidate list during search
        ml: Layer assignment multiplier (default: 1/ln(2))
    """

    def __init__(
        self,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        ml: float = 1.0 / np.log(2.0),
        metric: str = "euclidean",
    ):
        """
        Initialize HNSW index.

        Args:
            M: Maximum connections per node per layer
            ef_construction: Search width during index building
            ef_search: Search width during querying
            ml: Layer assignment multiplier
            metric: Distance metric
        """
        # Validate metric
        if metric not in ["euclidean", "cosine"]:
            raise ValueError(f"Unsupported metric: {metric}. Use 'euclidean' or 'cosine'")

        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.ml = ml
        self.metric = metric

        # Select distance function
        if metric == "euclidean":
            self.distance_fn = euclidean_distance
        else:  # cosine
            self.distance_fn = cosine_similarity

        # Graph structure
        self.nodes: Dict[int, HNSWNode] = {}
        self.entry_point: int = None
        self.max_layer: int = 0

        # Vector storage reference
        self.vectors: np.ndarray = None

    def _assign_layer(self) -> int:
        """
        Assign a random layer to a new node.

        Uses exponential decay: l = floor(-ln(uniform(0,1)) * ml)
        """
        # Generate random float in (0, 1) - avoid exactly 0 which would cause ln(0) = -inf
        uniform_random = np.random.uniform(0, 1)

        # Avoid edge case of exactly 0
        if uniform_random < 1e-9:
            uniform_random = 1e-9

        # Apply exponential decay formula
        layer = int(-np.log(uniform_random) * self.ml)

        return layer

    def build(self, vectors: np.ndarray) -> None:
        """
        Build HNSW index from scratch.

        Args:
            vectors: Array of vectors (n_vectors x dimension)
        """
        # TODO: Implement index construction
        pass

    def insert(self, vector: np.ndarray, vector_id: int) -> None:
        """
        Insert a vector into the HNSW graph.

        Steps:
        1. Assign layer to new node
        2. Find nearest neighbors at each layer
        3. Add bidirectional edges
        4. Prune connections if needed
        """
        # TODO: Implement insertion
        pass

    def search(self, query: np.ndarray, k: int, ef: int = None) -> List[Tuple[int, float]]:
        """
        Search for k nearest neighbors.

        Args:
            query: Query vector
            k: Number of neighbors to return
            ef: Search width (defaults to self.ef_search)

        Returns:
            List of (vector_id, distance) tuples
        """
        # TODO: Implement search
        pass

    def _search_layer(
        self, query: np.ndarray, entry_points: List[int], num_closest: int, layer: int
    ) -> List[Tuple[int, float]]:
        """
        Search for nearest neighbors at a specific layer.

        Args:
            query: Query vector
            entry_points: Starting points for search
            num_closest: Number of closest points to return
            layer: Layer to search

        Returns:
            List of (vector_id, distance) tuples
        """
        # TODO: Implement layer search (greedy or beam search)
        pass

    def _select_neighbors(
        self, candidates: List[Tuple[int, float]], M: int
    ) -> List[int]:
        """
        Select M neighbors from candidates using simple heuristic.

        Simple heuristic:
        - Sort candidates by distance
        - Take M nearest neighbors
        - Fast and effective (used in production systems)

        Args:
            candidates: List of (vector_id, distance) tuples
            M: Maximum number of neighbors to select

        Returns:
            List of selected vector IDs
        """
        # Handle edge case: fewer candidates than M
        if len(candidates) <= M:
            return [vid for vid, _ in candidates]

        # Sort by distance
        # For cosine similarity (higher = better), negate for sorting
        # For euclidean (lower = better), sort ascending
        if self.metric == "cosine":
            # Higher similarity is better, so negate
            sorted_candidates = sorted(candidates, key=lambda x: -x[1])
        else:
            # Lower distance is better
            sorted_candidates = sorted(candidates, key=lambda x: x[1])

        # Select top M
        selected = sorted_candidates[:M]

        # Return just the vector IDs
        return [vid for vid, _ in selected]

    def delete(self, vector_id: int) -> None:
        """Delete a vector from the index."""
        # TODO: Implement deletion
        pass

    def save(self, filepath: str) -> None:
        """Save index to disk."""
        # TODO: Implement save
        pass

    def load(self, filepath: str) -> None:
        """Load index from disk."""
        # TODO: Implement load
        pass
