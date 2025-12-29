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
import heapq
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

        This is the core construction algorithm. For each new node:
        1. Assign a random layer
        2. Search from top layer down (greedy, then beam search)
        3. Add bidirectional connections
        4. Prune neighbors to maintain max M connections

        Args:
            vector: Vector to insert (dimension,)
            vector_id: ID for the vector
        """
        # Assign layer for new node using exponential decay
        node_layer = self._assign_layer()

        # Create new node
        new_node = HNSWNode(vector_id=vector_id, layer=node_layer)
        self.nodes[vector_id] = new_node

        # Handle first node - it becomes the entry point
        if self.entry_point is None:
            self.entry_point = vector_id
            self.max_layer = node_layer
            return

        # Search from top layer down to node_layer+1 (greedy search, beam=1)
        # This finds the region where the new node should be inserted
        nearest = [self.entry_point]

        for lc in range(self.max_layer, node_layer, -1):
            # Greedy search at upper layers (num_closest=1)
            candidates = self._search_layer(vector, nearest, num_closest=1, layer=lc)
            # Extract just the IDs for next layer
            nearest = [c[0] for c in candidates]

        # Insert node at layers [node_layer, node_layer-1, ..., 0]
        # At each layer, find neighbors and create bidirectional connections
        for lc in range(node_layer, -1, -1):
            # Beam search to find ef_construction candidates at this layer
            candidates = self._search_layer(
                vector, nearest, num_closest=self.ef_construction, layer=lc
            )

            # Select M neighbors (M*2 for layer 0, M for higher layers)
            # Layer 0 gets more connections for better recall
            M = self.M * 2 if lc == 0 else self.M
            neighbors = self._select_neighbors(candidates, M=M)

            # Add bidirectional edges between new node and selected neighbors
            new_node.connections[lc] = set(neighbors)

            for neighbor_id in neighbors:
                neighbor_node = self.nodes[neighbor_id]

                # Initialize layer connections if needed
                if lc not in neighbor_node.connections:
                    neighbor_node.connections[lc] = set()

                # Add edge from neighbor to new node
                neighbor_node.connections[lc].add(vector_id)

                # Prune neighbor's connections if it exceeds M
                if len(neighbor_node.connections[lc]) > M:
                    # Recompute best M neighbors for this neighbor
                    neighbor_candidates = []
                    for conn_id in neighbor_node.connections[lc]:
                        dist = self.distance_fn(
                            self.vectors[neighbor_id], self.vectors[conn_id]
                        )
                        neighbor_candidates.append((conn_id, dist))

                    # Select best M and update connections
                    pruned = self._select_neighbors(neighbor_candidates, M=M)
                    neighbor_node.connections[lc] = set(pruned)

            # Update nearest for next layer (use neighbors we just connected to)
            nearest = neighbors

        # Update global entry point if new node has highest layer
        if node_layer > self.max_layer:
            self.max_layer = node_layer
            self.entry_point = vector_id

    def search(self, query: np.ndarray, k: int, ef: int = None) -> List[Tuple[int, float]]:
        """
        Search for k nearest neighbors.

        Algorithm:
        1. Start from entry point at top layer
        2. Greedy search (beam=1) down to layer 1
        3. Beam search (beam=ef) at layer 0
        4. Return top-k results

        Args:
            query: Query vector
            k: Number of neighbors to return
            ef: Search width at layer 0 (defaults to self.ef_search)
                Higher ef = better recall but slower search

        Returns:
            List of (vector_id, distance) tuples sorted by distance
        """
        # Handle empty index
        if self.entry_point is None:
            return []

        # Use default ef_search if not provided
        if ef is None:
            ef = self.ef_search

        # Ensure ef is at least k (can't return k results with ef < k)
        ef = max(ef, k)

        # Start from entry point
        nearest = [self.entry_point]

        # Greedy search from top layer down to layer 1
        # This quickly navigates to the right region
        for lc in range(self.max_layer, 0, -1):
            candidates = self._search_layer(query, nearest, num_closest=1, layer=lc)
            # Extract IDs for next layer
            nearest = [c[0] for c in candidates]

        # Beam search at layer 0 for precision
        # This explores multiple paths to find the best k neighbors
        candidates = self._search_layer(query, nearest, num_closest=ef, layer=0)

        # Return top-k results
        return candidates[:k]

    def _search_layer(
        self, query: np.ndarray, entry_points: List[int], num_closest: int, layer: int
    ) -> List[Tuple[int, float]]:
        """
        Search for nearest neighbors at a specific layer.

        This is the core navigation algorithm that makes HNSW fast.

        Uses beam search at layer 0 (explores multiple paths with width=num_closest)
        and greedy search at upper layers (follows single best path with width=1).

        Algorithm:
        1. Start from entry_points
        2. Maintain candidates heap (nodes to explore) and visited set
        3. Explore neighbors, adding promising ones to candidates
        4. Stop when no better candidates found
        5. Return top num_closest results

        Args:
            query: Query vector
            entry_points: Starting points for search
            num_closest: Number of closest points to return (beam width)
            layer: Layer to search

        Returns:
            List of (vector_id, distance) tuples sorted by distance
        """
        # Track visited nodes to avoid cycles
        visited = set(entry_points)

        # Candidates heap: (distance, vector_id)
        # Min-heap for euclidean (lower is better)
        # For cosine (higher is better), we negate distances
        candidates = []

        # Results heap: (-distance, vector_id) - max-heap to keep best
        # We negate distance to make it a max-heap (Python only has min-heap)
        results = []

        # Initialize with entry points
        for ep_id in entry_points:
            dist = self.distance_fn(query, self.vectors[ep_id])

            # For cosine similarity, negate to make lower=better (for min-heap)
            if self.metric == "cosine":
                heap_dist = -dist  # Higher similarity -> lower heap value
            else:
                heap_dist = dist   # Lower distance -> lower heap value

            heapq.heappush(candidates, (heap_dist, ep_id))
            heapq.heappush(results, (-heap_dist, ep_id))  # Negate for max-heap

        # Beam search loop
        while candidates:
            # Pop closest candidate
            current_dist, current_id = heapq.heappop(candidates)

            # Optimization: if current is farther than worst result, and we have enough results, stop
            if len(results) >= num_closest:
                # Get the farthest point in results (top of max-heap)
                worst_dist = -results[0][0]  # Negate back
                if current_dist > worst_dist:
                    break

            # Explore neighbors at this layer
            if current_id not in self.nodes:
                continue

            node = self.nodes[current_id]

            # Check if this node has connections at this layer
            if layer not in node.connections:
                continue

            # Examine all neighbors
            for neighbor_id in node.connections[layer]:
                # Skip if already visited
                if neighbor_id in visited:
                    continue

                visited.add(neighbor_id)

                # Compute distance to neighbor
                neighbor_dist = self.distance_fn(query, self.vectors[neighbor_id])

                # Convert to heap distance
                if self.metric == "cosine":
                    heap_dist = -neighbor_dist
                else:
                    heap_dist = neighbor_dist

                # Add to candidates for further exploration
                heapq.heappush(candidates, (heap_dist, neighbor_id))

                # Add to results
                heapq.heappush(results, (-heap_dist, neighbor_id))

                # Prune results if too many (keep only num_closest best)
                if len(results) > num_closest:
                    heapq.heappop(results)  # Remove worst (largest in max-heap)

        # Convert results back to list of (vector_id, distance)
        result_list = []
        while results:
            neg_dist, vid = heapq.heappop(results)
            actual_dist = -neg_dist  # Convert back from heap representation

            # For cosine, convert back to similarity (we negated it for heap)
            if self.metric == "cosine":
                actual_dist = -actual_dist

            result_list.append((vid, actual_dist))

        # Reverse to get closest first (we popped from max-heap)
        result_list.reverse()

        return result_list

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
