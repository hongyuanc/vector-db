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

import heapq
import warnings
from dataclasses import dataclass, field

import numpy as np

from .base import VectorIndex

# Import Cython-optimized distance functions for performance
try:
    from ..utils.distance_cy import cosine_similarity, euclidean_distance
    from . import hnsw_core
    CYTHON_AVAILABLE = True
except ImportError:
    # Fallback to Numba if Cython not compiled
    from ..utils.distance import cosine_similarity, euclidean_distance
    hnsw_core = None
    CYTHON_AVAILABLE = False
    import warnings
    warnings.warn(
        "Cython extensions not available, falling back to Numba. "
        "Run 'python setup.py build_ext --inplace' to build Cython extensions.",
        stacklevel=2,
    )

try:
    from . import hnsw_cpp
    CPP_AVAILABLE = True
except ImportError:
    hnsw_cpp = None
    CPP_AVAILABLE = False


@dataclass
class HNSWNode:
    """Represents a node in the HNSW graph."""
    vector_id: int
    layer: int
    # connections[layer] = set of neighbor IDs
    connections: dict[int, set[int]] = field(default_factory=lambda: {})


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
        self.nodes: dict[int, HNSWNode] = {}
        self.entry_point: int = None
        self.max_layer: int = 0

        # Vector storage reference
        self.vectors: np.ndarray = None

        # Cached graph connections for Cython (built on-demand)
        self._graph_connections_cache: dict = None
        self._vectors_f64_cache: np.ndarray = None

        # Cached CSR graph connections for C++ search (built after index construction)
        self._cpp_graph_cache: dict = None
        self._vectors_f32_cache: np.ndarray = None
        self._python_graph_materialized = True

    @property
    def graph_storage_mode(self) -> str:
        """
        Describe the current HNSW graph ownership shape.

        Returns:
            "compact_csr" when graph edges live only in C++ CSR arrays.
            "materialized_python" when Python node connection sets are present.
            "empty" when the index has no nodes.
        """
        if not self.nodes:
            return "empty"
        if self._cpp_graph_cache is not None and not self._python_graph_materialized:
            return "compact_csr"
        if self._python_graph_materialized:
            return "materialized_python"
        return "unknown"

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
        Build HNSW index.

        This is a convenience method that stores the vectors and inserts
        them one by one using the insert() operation. This is the most
        common way to construct an HNSW index.

        Args:
            vectors: Array of vectors (n_vectors x dimension)
                    Shape: (n_vector111s, dimension)

        Example:
            >>> index = HNSWIndex(M=16, ef_construction=200)
            >>> vectors = np.random.randn(1000, 128).astype(np.float32)
            >>> index.build(vectors)
            >>> results = index.search(query_vector, k=10)
        """
        # Store vector data
        self.vectors = vectors
        self.nodes = {}
        self.entry_point = None
        self.max_layer = 0
        self._graph_connections_cache = None
        self._vectors_f64_cache = None
        self._cpp_graph_cache = None
        self._vectors_f32_cache = None
        self._python_graph_materialized = True

        # Insert each vector into the graph
        total = len(vectors)
        if total == 0:
            return

        if CPP_AVAILABLE and hnsw_cpp is not None and hasattr(hnsw_cpp, "build_graph"):
            levels = np.asarray([self._assign_layer() for _ in range(total)], dtype=np.int32)
            graph = hnsw_cpp.build_graph(
                self._get_vectors_f32(),
                levels,
                self.M,
                self.ef_construction,
                self.metric,
                include_connections=False,
            )
            self._load_cpp_build_result(graph)
            print(f"Progress: {total:,}/{total:,} vectors (100.0%)", flush=True)
        else:
            self._build_with_python_insert(vectors)

        # Pre-build cache for Cython if available
        if (
            self.vectors is not None
            and self.nodes
            and self._python_graph_materialized
            and CYTHON_AVAILABLE
            and hnsw_core is not None
        ):
            self._build_cython_cache()
        if (
            self.vectors is not None
            and self.nodes
            and CPP_AVAILABLE
            and hnsw_cpp is not None
            and self._cpp_graph_cache is None
        ):
            self._build_cpp_cache()

    def _build_with_python_insert(self, vectors: np.ndarray) -> None:
        """Build the graph through the incremental Python/Cython insert path."""
        total = len(vectors)
        log_interval = max(1, total // 20)  # Log 20 times during build

        for i in range(total):
            self.insert(vectors[i], vector_id=i)

            # Progress logging
            if (i + 1) % log_interval == 0 or (i + 1) == total:
                progress = (i + 1) / total * 100
                print(f"Progress: {i+1:,}/{total:,} vectors ({progress:.1f}%)", flush=True)

    def _load_cpp_build_result(self, graph: dict) -> None:
        """Load a C++ batch build result without eagerly copying Python edges."""
        levels = graph["levels"]
        self.nodes = {
            vector_id: HNSWNode(vector_id=vector_id, layer=int(layer))
            for vector_id, layer in enumerate(levels)
        }

        connections = graph.get("connections", [])
        for node_id, layer, neighbors in connections:
            node = self.nodes[int(node_id)]
            node.connections[int(layer)] = {int(neighbor_id) for neighbor_id in neighbors}

        self._load_cpp_layers(graph.get("layers", {}))
        self._python_graph_materialized = bool(connections) or not self._cpp_graph_cache

        entry_point = int(graph["entry_point"])
        self.entry_point = entry_point if entry_point >= 0 else None
        self.max_layer = int(graph["max_layer"])

    def materialize_python_graph(self) -> None:
        """Rebuild Python connection sets from the C++ CSR cache when needed."""
        self._ensure_python_graph_materialized()

    def _ensure_python_graph_materialized(self) -> None:
        """Lazy compatibility path for inspection, persistence, and mutation."""
        if self._python_graph_materialized:
            return

        if self._cpp_graph_cache is None:
            self._python_graph_materialized = True
            return

        for node in self.nodes.values():
            node.connections = {}

        active_node_ids = set(self.nodes)
        for layer, (offsets, neighbors) in self._cpp_graph_cache.items():
            node_count = max(0, len(offsets) - 1)
            for node_id in range(node_count):
                node = self.nodes.get(node_id)
                if node is None:
                    continue

                begin = int(offsets[node_id])
                end = int(offsets[node_id + 1])
                begin = max(0, min(begin, len(neighbors)))
                end = max(begin, min(end, len(neighbors)))

                neighbor_set = {
                    int(neighbor_id)
                    for neighbor_id in neighbors[begin:end]
                    if int(neighbor_id) in active_node_ids
                }
                if neighbor_set:
                    node.connections[int(layer)] = neighbor_set

        self._python_graph_materialized = True
        self._graph_connections_cache = None

    def _ensure_mutable_python_graph(self, operation: str) -> None:
        """Materialize compact CSR graph before a mutating operation."""
        if not self._python_graph_materialized and self._cpp_graph_cache is not None:
            warnings.warn(
                f"{operation} materializes the compact CSR graph into Python "
                "connection sets; this keeps mutation behavior available but "
                "increases graph memory until the index is rebuilt compactly.",
                RuntimeWarning,
                stacklevel=3,
            )
        self._ensure_python_graph_materialized()

    def _load_cpp_layers(self, layers: dict) -> None:
        """Load C++ builder CSR layers into the post-build search cache."""
        if not layers:
            self._cpp_graph_cache = None
            return

        self._cpp_graph_cache = {
            int(layer): (
                np.ascontiguousarray(layer_data["offsets"], dtype=np.int32),
                np.ascontiguousarray(layer_data["neighbors"], dtype=np.int32),
            )
            for layer, layer_data in layers.items()
        }

    def _collect_cpp_layer_arrays_for_save(self) -> dict[str, np.ndarray]:
        """Return CSR layer arrays in a direct npz-friendly shape."""
        if not self._cpp_graph_cache:
            return {"csr_layers": np.array([], dtype=np.int32)}

        csr_arrays = {}
        layers = sorted(int(layer) for layer in self._cpp_graph_cache)
        csr_arrays["csr_layers"] = np.asarray(layers, dtype=np.int32)

        for layer in layers:
            offsets, neighbors = self._cpp_graph_cache[layer]
            csr_arrays[f"csr_offsets_{layer}"] = np.ascontiguousarray(
                offsets, dtype=np.int32
            )
            csr_arrays[f"csr_neighbors_{layer}"] = np.ascontiguousarray(
                neighbors, dtype=np.int32
            )

        return csr_arrays

    def _load_cpp_layers_from_npz(self, data) -> None:
        """Restore CSR layer arrays saved directly in the index npz."""
        if "csr_layers" not in data.files:
            self._cpp_graph_cache = None
            return

        graph_cache = {}
        for layer_value in data["csr_layers"]:
            layer = int(layer_value)
            offsets_key = f"csr_offsets_{layer}"
            neighbors_key = f"csr_neighbors_{layer}"
            if offsets_key not in data.files or neighbors_key not in data.files:
                raise ValueError(f"Missing CSR arrays for HNSW layer {layer}")

            graph_cache[layer] = (
                np.ascontiguousarray(data[offsets_key], dtype=np.int32),
                np.ascontiguousarray(data[neighbors_key], dtype=np.int32),
            )

        self._cpp_graph_cache = graph_cache if graph_cache else None

    def _build_cython_cache(self):
        """Build cached data structures for Cython-optimized search."""
        self._ensure_python_graph_materialized()

        # Ensure vectors are cached as float64
        self._get_vectors_f64()

        # Build graph connections dict once
        self._graph_connections_cache = {}
        for node_id, node in self.nodes.items():
            for lc, neighbors in node.connections.items():
                self._graph_connections_cache[(node_id, lc)] = neighbors

    def _get_vectors_f64(self) -> np.ndarray:
        """Get or create cached float64 vectors for Cython operations."""
        if self._vectors_f64_cache is None:
            if self.vectors.dtype != np.float64:
                self._vectors_f64_cache = self.vectors.astype(np.float64)
            else:
                self._vectors_f64_cache = self.vectors
        return self._vectors_f64_cache

    def _get_vectors_f32(self) -> np.ndarray:
        """Get or create cached contiguous float32 vectors for C++ operations."""
        if self._vectors_f32_cache is None:
            self._vectors_f32_cache = np.ascontiguousarray(self.vectors, dtype=np.float32)
        return self._vectors_f32_cache

    def _build_cpp_cache(self):
        """Build per-layer CSR adjacency arrays for C++-optimized search."""
        self._ensure_python_graph_materialized()

        if not self.nodes:
            self._cpp_graph_cache = {}
            return

        self._get_vectors_f32()

        max_node_id = max(self.nodes)
        node_count = max_node_id + 1
        layers = set()
        for node in self.nodes.values():
            layers.update(node.connections)

        graph_cache = {}
        for layer in layers:
            offsets = np.zeros(node_count + 1, dtype=np.int32)
            adjacency = []

            for node_id in range(node_count):
                offsets[node_id] = len(adjacency)
                node = self.nodes.get(node_id)
                if node is None:
                    continue

                neighbors = node.connections.get(layer, set())
                adjacency.extend(
                    sorted(
                        int(neighbor_id)
                        for neighbor_id in neighbors
                        if 0 <= neighbor_id < node_count and neighbor_id in self.nodes
                    )
                )

            offsets[node_count] = len(adjacency)
            graph_cache[layer] = (
                offsets,
                np.asarray(adjacency, dtype=np.int32),
            )

        self._cpp_graph_cache = graph_cache

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
        # Note: We incrementally update graph cache during build for performance
        # Vectors_f64_cache remains valid during build
        self._ensure_mutable_python_graph("insert()")
        had_existing_nodes = bool(self.nodes)
        self._cpp_graph_cache = None
        self._vectors_f32_cache = None

        # Assign layer for new node using exponential decay
        node_layer = self._assign_layer()

        # Create new node
        new_node = HNSWNode(vector_id=vector_id, layer=node_layer)
        self.nodes[vector_id] = new_node

        # Initialize Cython caches if using Cython (for fast search during build)
        if CYTHON_AVAILABLE and hnsw_core is not None:
            if self._graph_connections_cache is None:
                if had_existing_nodes:
                    self._build_cython_cache()
                else:
                    self._graph_connections_cache = {}
            if self._vectors_f64_cache is None:
                self._get_vectors_f64()

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

            # Update Cython cache for new node's connections
            if self._graph_connections_cache is not None:
                self._graph_connections_cache[(vector_id, lc)] = new_node.connections[lc]

            for neighbor_id in neighbors:
                neighbor_node = self.nodes[neighbor_id]

                # Initialize layer connections if needed
                if lc not in neighbor_node.connections:
                    neighbor_node.connections[lc] = set()

                # Add edge from neighbor to new node
                neighbor_node.connections[lc].add(vector_id)

                # Update Cython cache for neighbor's connections
                if self._graph_connections_cache is not None:
                    self._graph_connections_cache[(neighbor_id, lc)] = neighbor_node.connections[lc]

                # Prune neighbor's connections if it exceeds M
                if len(neighbor_node.connections[lc]) > M:
                    # Use native pruning when available to reduce build-time Python work.
                    if CPP_AVAILABLE and hnsw_cpp is not None:
                        pruned = hnsw_cpp.prune_connections(
                            self._get_vectors_f32(),
                            neighbor_id,
                            list(neighbor_node.connections[lc]),
                            M,
                            self.metric,
                        )
                        neighbor_node.connections[lc] = set(pruned)
                    elif CYTHON_AVAILABLE and hnsw_core is not None:
                        # Use cached float64 vectors (avoid repeated conversions)
                        pruned = hnsw_core.prune_connections(
                            self._get_vectors_f64(),
                            neighbor_id,
                            neighbor_node.connections[lc],
                            M,
                            self.metric
                        )
                        neighbor_node.connections[lc] = set(pruned)
                    else:
                        # Fallback to Python implementation
                        neighbor_candidates = []
                        for conn_id in neighbor_node.connections[lc]:
                            dist = self.distance_fn(
                                self.vectors[neighbor_id], self.vectors[conn_id]
                            )
                            neighbor_candidates.append((conn_id, dist))

                        # Select best M and update connections
                        pruned = self._select_neighbors(neighbor_candidates, M=M)
                        neighbor_node.connections[lc] = set(pruned)

                    # Update Cython cache after pruning
                    if self._graph_connections_cache is not None:
                        self._graph_connections_cache[(neighbor_id, lc)] = neighbor_node.connections[lc]

            # Update nearest for next layer (use neighbors we just connected to)
            nearest = neighbors

        # Update global entry point if new node has highest layer
        if node_layer > self.max_layer:
            self.max_layer = node_layer
            self.entry_point = vector_id

    def search(self, query: np.ndarray, k: int, ef: int = None) -> list[tuple[int, float]]:
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

    def search_batch(
        self,
        queries: np.ndarray,
        k: int,
        ef: int = None,
    ) -> list[list[tuple[int, float]]]:
        """
        Search for k nearest neighbors for each query vector.

        This keeps batch-query call sites explicit and gives benchmarks one API
        to optimize further without changing search semantics.

        Args:
            queries: Query vectors (n_queries x dimension)
            k: Number of neighbors to return per query
            ef: Search width at layer 0 (defaults to self.ef_search)

        Returns:
            One search result list per query, preserving query order.
        """
        query_array = np.asarray(queries)
        if query_array.ndim != 2:
            raise ValueError("queries must be a 2D array shaped (n_queries, dimension)")
        if query_array.shape[0] == 0:
            return []
        if k <= 0:
            return [[] for _query in range(query_array.shape[0])]
        if self.entry_point is None:
            return [[] for _query in range(query_array.shape[0])]

        if ef is None:
            ef = self.ef_search
        ef = max(ef, k)

        if (
            CPP_AVAILABLE
            and hnsw_cpp is not None
            and hasattr(hnsw_cpp, "search_batch")
            and self._cpp_graph_cache is not None
            and self._vectors_f32_cache is not None
            and all(layer in self._cpp_graph_cache for layer in range(self.max_layer + 1))
        ):
            query_array = np.ascontiguousarray(query_array, dtype=np.float32)
            return hnsw_cpp.search_batch(
                queries=query_array,
                vectors=self._vectors_f32_cache,
                layers=self._cpp_graph_cache,
                entry_point=self.entry_point,
                max_layer=self.max_layer,
                k=k,
                ef=ef,
                metric=self.metric,
            )

        return [self.search(query, k=k, ef=ef) for query in query_array]

    def _search_layer(
        self,
        query: np.ndarray,
        entry_points: list[int],
        num_closest: int,
        layer: int,
        use_cython: bool = True,
        use_cpp: bool = True,
    ) -> list[tuple[int, float]]:
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
            use_cython: Whether to use Cython optimization (False during build)
            use_cpp: Whether to use C++ CSR search when the cache is available

        Returns:
            List of (vector_id, distance) tuples sorted by distance
        """
        if (
            use_cpp
            and CPP_AVAILABLE
            and hnsw_cpp is not None
            and self._cpp_graph_cache is not None
            and self._vectors_f32_cache is not None
            and layer in self._cpp_graph_cache
        ):
            offsets, neighbors = self._cpp_graph_cache[layer]
            query_f32 = np.ascontiguousarray(query, dtype=np.float32)
            return hnsw_cpp.search_layer(
                query=query_f32,
                vectors=self._vectors_f32_cache,
                offsets=offsets,
                neighbors=neighbors,
                entry_points=entry_points,
                num_closest=num_closest,
                metric=self.metric,
            )

        if not self._python_graph_materialized:
            self._ensure_python_graph_materialized()
            if use_cython and CYTHON_AVAILABLE and hnsw_core is not None:
                self._build_cython_cache()

        # Use Cython-optimized search if available and cache is ready
        if use_cython and CYTHON_AVAILABLE and hnsw_core is not None and self._graph_connections_cache is not None and self._vectors_f64_cache is not None:

            # Convert query to float64 for Cython
            query_f64 = query.astype(np.float64) if query.dtype != np.float64 else query

            return hnsw_core.search_layer(
                query_f64,
                self._vectors_f64_cache,
                entry_points,
                num_closest,
                layer,
                self._graph_connections_cache,
                self.metric
            )

        # Fallback to Python implementation
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
        self, candidates: list[tuple[int, float]], M: int
    ) -> list[int]:
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
        # Use Cython-optimized version if available
        if CYTHON_AVAILABLE and hnsw_core is not None:
            return hnsw_core.select_neighbors(candidates, M, self.metric)

        # Fallback to Python implementation
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
        """
        Delete a vector from the HNSW index.

        This removes the node and all its connections from the graph.
        Algorithm:
        1. Remove all edges pointing TO this node from neighbors
        2. Remove the node itself
        3. Handle entry point update if needed

        Args:
            vector_id: ID of vector to delete

        Note:
            - Does not modify self.vectors array (keeps indexing consistent)
            - If deleting entry point, finds new entry point from remaining nodes
            - Graph remains navigable after deletion
        """
        # Check if vector exists
        if vector_id not in self.nodes:
            raise ValueError(f"Vector ID {vector_id} not found in index")

        self._ensure_mutable_python_graph("delete()")
        self._cpp_graph_cache = None
        self._vectors_f32_cache = None

        # Remove all edges pointing TO this node from ALL nodes
        # Note: We must check all nodes, not just this node's neighbors,
        # because HNSW connections might not be perfectly symmetric due to M-limit pruning
        for other_node in self.nodes.values():
            for layer in other_node.connections:
                other_node.connections[layer].discard(vector_id)

        # Remove the node itself
        del self.nodes[vector_id]

        # Handle entry point update if we deleted it
        if self.entry_point == vector_id:
            if len(self.nodes) == 0:
                # Index is now empty
                self.entry_point = None
                self.max_layer = 0
            else:
                # Find new entry point - pick node with highest layer
                new_entry = max(self.nodes.values(), key=lambda n: n.layer)
                self.entry_point = new_entry.vector_id
                self.max_layer = new_entry.layer

    def save(self, filepath: str) -> None:
        """
        Save HNSW index to disk.

        Serializes the entire index state including:
        - Graph structure (nodes and connections)
        - Index parameters (M, ef_construction, ef_search, ml, metric)
        - Entry point and layer information
        - Vector data

        Args:
            filepath: Path to save the index (.npz file)

        Example:
            >>> index.save("my_index.npz")
            >>> new_index = HNSWIndex()
            >>> new_index.load("my_index.npz")
        """
        import pickle

        has_compact_cpp_graph = (
            self._cpp_graph_cache is not None
            and not self._python_graph_materialized
        )
        if not has_compact_cpp_graph:
            self._ensure_python_graph_materialized()

        node_ids = np.asarray(sorted(self.nodes), dtype=np.int64)
        node_layers = np.asarray(
            [self.nodes[int(vid)].layer for vid in node_ids], dtype=np.int32
        )

        # Serialize graph structure using pickle
        # We need to convert the nodes dict to a serializable format
        nodes_data = {}
        for vid, node in self.nodes.items():
            # Convert sets to lists for serialization
            connections_serializable = {
                layer: list(neighbors) for layer, neighbors in node.connections.items()
            }
            nodes_data[vid] = {
                'vector_id': node.vector_id,
                'layer': node.layer,
                'connections': connections_serializable
            }

        # Save everything to .npz file
        save_data = {
            "format_version": np.array(2, dtype=np.int16),
            # Graph structure
            "nodes": pickle.dumps(nodes_data),
            "node_ids": node_ids,
            "node_layers": node_layers,
            "python_graph_materialized": np.array(
                self._python_graph_materialized, dtype=np.bool_
            ),
            "entry_point": self.entry_point if self.entry_point is not None else -1,
            "max_layer": self.max_layer,
            # Parameters
            "M": self.M,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
            "ml": self.ml,
            "metric": self.metric,
            # Vector data
            "vectors": self.vectors if self.vectors is not None else np.array([]),
        }
        save_data.update(self._collect_cpp_layer_arrays_for_save())
        np.savez(filepath, **save_data)

    def load(self, filepath: str) -> None:
        """
        Load HNSW index from disk.

        Restores the complete index state from a saved file.
        This completely replaces the current index state.

        Args:
            filepath: Path to the saved index (.npz file)

        Example:
            >>> index = HNSWIndex()
            >>> index.load("my_index.npz")
            >>> results = index.search(query, k=10)
        """
        import pickle

        # Load data from .npz file
        data = np.load(filepath, allow_pickle=True)

        # Restore parameters
        self.M = int(data['M'])
        self.ef_construction = int(data['ef_construction'])
        self.ef_search = int(data['ef_search'])
        self.ml = float(data['ml'])
        self.metric = str(data['metric'])

        # Restore distance function based on metric
        if self.metric == "euclidean":
            self.distance_fn = euclidean_distance
        else:  # cosine
            self.distance_fn = cosine_similarity

        # Restore entry point and max layer
        entry_point_val = int(data['entry_point'])
        self.entry_point = entry_point_val if entry_point_val != -1 else None
        self.max_layer = int(data['max_layer'])

        # Restore vectors
        vectors_data = data['vectors']
        self.vectors = vectors_data if len(vectors_data) > 0 else None
        self._graph_connections_cache = None
        self._vectors_f64_cache = None
        self._cpp_graph_cache = None
        self._vectors_f32_cache = None

        self._load_cpp_layers_from_npz(data)

        # Restore graph structure. New files store node IDs/layers directly so
        # compact CSR loads do not need Python edge sets at all.
        saved_python_graph_materialized = bool(
            data["python_graph_materialized"].item()
        ) if "python_graph_materialized" in data.files else True

        self.nodes = {}
        if (
            "node_ids" in data.files
            and "node_layers" in data.files
            and not saved_python_graph_materialized
        ):
            for vid, layer in zip(data["node_ids"], data["node_layers"], strict=True):
                vector_id = int(vid)
                self.nodes[vector_id] = HNSWNode(
                    vector_id=vector_id,
                    layer=int(layer),
                )
        else:
            nodes_data = pickle.loads(data['nodes'].tobytes())
            for vid, node_dict in nodes_data.items():
                # Convert lists back to sets
                connections = {
                    layer: set(neighbors)
                    for layer, neighbors in node_dict['connections'].items()
                }
                node = HNSWNode(
                    vector_id=node_dict['vector_id'],
                    layer=node_dict['layer'],
                    connections=connections
                )
                self.nodes[vid] = node

        self._python_graph_materialized = (
            saved_python_graph_materialized or self._cpp_graph_cache is None
        )

        if self.vectors is not None and self._cpp_graph_cache is not None:
            self._get_vectors_f32()
        if (
            self._python_graph_materialized
            and self.vectors is not None
            and self.nodes
            and CYTHON_AVAILABLE
            and hnsw_core is not None
        ):
            self._build_cython_cache()
        if (
            self.vectors is not None
            and self.nodes
            and CPP_AVAILABLE
            and hnsw_cpp is not None
            and self._cpp_graph_cache is None
        ):
            self._build_cpp_cache()
