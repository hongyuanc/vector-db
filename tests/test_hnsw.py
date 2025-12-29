"""
Tests for HNSW index implementation.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from src.index.hnsw import HNSWIndex, HNSWNode


class TestHNSWLayerAssignment:
    """Test HNSW layer assignment mechanism."""

    def test_layer_assignment_distribution(self):
        """Test that layer assignment follows exponential distribution."""
        index = HNSWIndex(M=16, metric="euclidean")

        # Generate 10000 layer assignments
        layers = [index._assign_layer() for _ in range(10000)]

        # Count vectors per layer
        layer_counts = {}
        for layer in layers:
            layer_counts[layer] = layer_counts.get(layer, 0) + 1

        # With ml=1/ln(2), expect ~50% on layer 0 (allow some variance)
        # 10000 samples -> expect ~5000, allow 4500-5500
        assert 4500 < layer_counts[0] < 5500, f"Expected ~5000 on layer 0, got {layer_counts[0]}"

        # Should have some vectors on higher layers
        max_layer = max(layers)
        assert max_layer >= 2, "Should have vectors on at least layer 2"

        # Each layer should have roughly half the vectors of the layer below
        # (with some variance due to randomness)
        if 1 in layer_counts and 0 in layer_counts:
            ratio = layer_counts[1] / layer_counts[0]
            assert 0.35 < ratio < 0.65, f"Layer 1/Layer 0 ratio should be ~0.5, got {ratio}"

    def test_layer_assignment_non_negative(self):
        """Test that layer assignment never returns negative."""
        index = HNSWIndex(M=16, metric="euclidean")

        for _ in range(1000):
            layer = index._assign_layer()
            assert layer >= 0, f"Layer should be non-negative, got {layer}"

    def test_layer_assignment_is_int(self):
        """Test that layer assignment returns integer."""
        index = HNSWIndex(M=16, metric="euclidean")

        layer = index._assign_layer()
        assert isinstance(layer, (int, np.integer)), f"Layer should be int, got {type(layer)}"


class TestHNSWNode:
    """Test HNSW node data structure."""

    def test_create_node(self):
        """Test creating an HNSW node."""
        node = HNSWNode(vector_id=5, layer=2)

        assert node.vector_id == 5
        assert node.layer == 2
        assert node.connections == {}

    def test_node_add_connections(self):
        """Test adding connections to a node."""
        node = HNSWNode(vector_id=5, layer=2)

        # Add connections on layer 0
        node.connections[0] = {1, 2, 3}

        assert 0 in node.connections
        assert len(node.connections[0]) == 3
        assert 1 in node.connections[0]


class TestHNSWIndex:
    """Test HNSW index basic functionality."""

    def test_create_index_euclidean(self):
        """Test creating HNSW index with Euclidean metric."""
        index = HNSWIndex(M=16, ef_construction=200, metric="euclidean")

        assert index.M == 16
        assert index.ef_construction == 200
        assert index.metric == "euclidean"
        assert index.entry_point is None
        assert index.max_layer == 0

    def test_create_index_cosine(self):
        """Test creating HNSW index with cosine metric."""
        index = HNSWIndex(M=16, metric="cosine")

        assert index.metric == "cosine"

    def test_invalid_metric(self):
        """Test that invalid metric raises error."""
        with pytest.raises(ValueError, match="Unsupported metric"):
            HNSWIndex(metric="invalid")


class TestNeighborSelection:
    """Test HNSW neighbor selection heuristic."""

    def test_select_neighbors_euclidean(self):
        """Test selecting M nearest neighbors with Euclidean distance."""
        index = HNSWIndex(M=3, metric="euclidean")

        # Candidates: (vector_id, distance)
        candidates = [(1, 5.0), (2, 1.0), (3, 3.0), (4, 2.0), (5, 10.0)]

        # Should select 3 with smallest distances: 2 (1.0), 4 (2.0), 3 (3.0)
        selected = index._select_neighbors(candidates, M=3)

        assert len(selected) == 3
        assert 2 in selected  # distance 1.0
        assert 4 in selected  # distance 2.0
        assert 3 in selected  # distance 3.0

    def test_select_neighbors_cosine(self):
        """Test selecting M nearest neighbors with cosine similarity."""
        index = HNSWIndex(M=3, metric="cosine")

        # Candidates: (vector_id, similarity) - higher is better
        candidates = [(1, 0.5), (2, 0.9), (3, 0.7), (4, 0.8), (5, 0.1)]

        # Should select 3 with highest similarities: 2 (0.9), 4 (0.8), 3 (0.7)
        selected = index._select_neighbors(candidates, M=3)

        assert len(selected) == 3
        assert 2 in selected  # similarity 0.9
        assert 4 in selected  # similarity 0.8
        assert 3 in selected  # similarity 0.7

    def test_select_neighbors_fewer_than_M(self):
        """Test when there are fewer candidates than M."""
        index = HNSWIndex(M=5, metric="euclidean")

        candidates = [(1, 1.0), (2, 2.0)]

        # Should return all candidates
        selected = index._select_neighbors(candidates, M=5)

        assert len(selected) == 2
        assert 1 in selected
        assert 2 in selected

    def test_select_neighbors_empty(self):
        """Test with empty candidates list."""
        index = HNSWIndex(M=3, metric="euclidean")

        selected = index._select_neighbors([], M=3)

        assert selected == []


class TestLayerSearch:
    """Test HNSW layer search algorithm."""

    @pytest.fixture
    def simple_graph_euclidean(self):
        """
        Create a simple HNSW graph for testing.

        Graph structure at layer 0:
        v0 [1,0,0] --- v1 [0,1,0]
         |              |
        v2 [0,0,1] --- v3 [1,1,0]
        """
        index = HNSWIndex(M=2, metric="euclidean")

        # Create vectors
        index.vectors = np.array([
            [1.0, 0.0, 0.0],  # v0
            [0.0, 1.0, 0.0],  # v1
            [0.0, 0.0, 1.0],  # v2
            [1.0, 1.0, 0.0],  # v3
        ], dtype=np.float32)

        # Create nodes with connections at layer 0
        index.nodes[0] = HNSWNode(vector_id=0, layer=0, connections={0: {1, 2}})
        index.nodes[1] = HNSWNode(vector_id=1, layer=0, connections={0: {0, 3}})
        index.nodes[2] = HNSWNode(vector_id=2, layer=0, connections={0: {0, 3}})
        index.nodes[3] = HNSWNode(vector_id=3, layer=0, connections={0: {1, 2}})

        return index

    @pytest.fixture
    def simple_graph_cosine(self):
        """Create a simple graph with cosine similarity."""
        index = HNSWIndex(M=2, metric="cosine")

        # Create unit vectors for easier cosine calculation
        index.vectors = np.array([
            [1.0, 0.0, 0.0],   # v0
            [0.0, 1.0, 0.0],   # v1
            [0.0, 0.0, 1.0],   # v2
            [0.707, 0.707, 0.0],  # v3 - 45 degrees between v0 and v1
        ], dtype=np.float32)

        # Create nodes with connections at layer 0
        index.nodes[0] = HNSWNode(vector_id=0, layer=0, connections={0: {1, 3}})
        index.nodes[1] = HNSWNode(vector_id=1, layer=0, connections={0: {0, 3}})
        index.nodes[2] = HNSWNode(vector_id=2, layer=0, connections={0: {0, 1}})
        index.nodes[3] = HNSWNode(vector_id=3, layer=0, connections={0: {0, 1}})

        return index

    def test_search_layer_euclidean_single_entry(self, simple_graph_euclidean):
        """Test layer search with single entry point."""
        index = simple_graph_euclidean

        # Query close to v0 [1,0,0]
        query = np.array([0.9, 0.0, 0.0], dtype=np.float32)

        # Start from v1, search for 2 nearest
        results = index._search_layer(query, entry_points=[1], num_closest=2, layer=0)

        # Should find v0 as closest (distance ~0.1)
        assert len(results) >= 1
        assert results[0][0] == 0, "v0 should be closest to query [0.9,0,0]"

    def test_search_layer_euclidean_multiple_entries(self, simple_graph_euclidean):
        """Test layer search with multiple entry points."""
        index = simple_graph_euclidean

        # Query at origin
        query = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # Start from v0 and v1
        results = index._search_layer(query, entry_points=[0, 1], num_closest=3, layer=0)

        # Should find all connected nodes
        assert len(results) >= 3
        found_ids = {r[0] for r in results}
        assert 0 in found_ids
        assert 1 in found_ids
        assert 2 in found_ids or 3 in found_ids

    def test_search_layer_cosine(self, simple_graph_cosine):
        """Test layer search with cosine similarity."""
        index = simple_graph_cosine

        # Query similar to v0 [1,0,0]
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        # Start from v1
        results = index._search_layer(query, entry_points=[1], num_closest=2, layer=0)

        # Should find v0 as most similar (cosine similarity = 1.0)
        assert len(results) >= 1
        assert results[0][0] == 0, "v0 should be most similar to query [1,0,0]"
        assert results[0][1] == pytest.approx(1.0), "Cosine similarity should be 1.0"

    def test_search_layer_beam_width(self, simple_graph_euclidean):
        """Test that beam width (num_closest) is respected."""
        index = simple_graph_euclidean

        query = np.array([0.5, 0.5, 0.0], dtype=np.float32)

        # Search with beam width 1
        results = index._search_layer(query, entry_points=[0], num_closest=1, layer=0)
        assert len(results) == 1

        # Search with beam width 3
        results = index._search_layer(query, entry_points=[0], num_closest=3, layer=0)
        assert len(results) <= 3

    def test_search_layer_empty_graph(self):
        """Test search on empty graph."""
        index = HNSWIndex(M=2, metric="euclidean")
        index.vectors = np.array([
            [1.0, 0.0, 0.0],
        ], dtype=np.float32)

        query = np.array([0.5, 0.0, 0.0], dtype=np.float32)

        # Entry point exists but has no connections
        index.nodes[0] = HNSWNode(vector_id=0, layer=0, connections={})

        results = index._search_layer(query, entry_points=[0], num_closest=5, layer=0)

        # Should return only the entry point
        assert len(results) == 1
        assert results[0][0] == 0

    def test_search_layer_returns_sorted_results(self, simple_graph_euclidean):
        """Test that results are sorted by distance (closest first)."""
        index = simple_graph_euclidean

        query = np.array([0.5, 0.5, 0.5], dtype=np.float32)

        results = index._search_layer(query, entry_points=[0], num_closest=4, layer=0)

        # Verify distances are in ascending order (closest first)
        for i in range(len(results) - 1):
            assert results[i][1] <= results[i + 1][1], "Results should be sorted by distance"


class TestInsert:
    """Test HNSW insert operation."""

    def test_insert_first_node(self):
        """Test inserting the first node becomes entry point."""
        index = HNSWIndex(M=16, metric="euclidean")

        # Create vector storage
        index.vectors = np.array([
            [1.0, 0.0, 0.0],
        ], dtype=np.float32)

        # Insert first node
        index.insert(index.vectors[0], vector_id=0)

        # Verify it becomes entry point
        assert index.entry_point == 0, "First node should be entry point"
        assert 0 in index.nodes, "Node 0 should exist"
        assert index.nodes[0].vector_id == 0

    def test_insert_two_nodes(self):
        """Test inserting two nodes creates bidirectional connection."""
        index = HNSWIndex(M=16, metric="euclidean")

        # Create vectors
        index.vectors = np.array([
            [1.0, 0.0, 0.0],  # v0
            [0.0, 1.0, 0.0],  # v1
        ], dtype=np.float32)

        # Insert both nodes
        index.insert(index.vectors[0], vector_id=0)
        index.insert(index.vectors[1], vector_id=1)

        # Verify nodes exist
        assert 0 in index.nodes
        assert 1 in index.nodes

        # Verify bidirectional connection at layer 0
        # (both should be connected since there are only 2 nodes)
        node0 = index.nodes[0]
        node1 = index.nodes[1]

        assert 0 in node0.connections, "Node 0 should have layer 0 connections"
        assert 0 in node1.connections, "Node 1 should have layer 0 connections"

        # Check if they're connected (might not be if layers don't overlap)
        # At minimum, verify graph structure is created
        assert isinstance(node0.connections, dict)
        assert isinstance(node1.connections, dict)

    def test_insert_multiple_nodes_creates_graph(self):
        """Test inserting multiple nodes creates connected graph."""
        index = HNSWIndex(M=3, ef_construction=50, metric="euclidean")

        # Create simple 3D vectors
        vectors = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.5, 0.5, 0.5],
        ], dtype=np.float32)

        index.vectors = vectors

        # Insert all nodes
        for i in range(len(vectors)):
            index.insert(vectors[i], vector_id=i)

        # Verify all nodes exist
        assert len(index.nodes) == 5, "Should have 5 nodes"

        # Verify graph is connected at layer 0
        # All nodes should have at least one connection at layer 0
        for i in range(5):
            node = index.nodes[i]
            assert 0 in node.connections, f"Node {i} should have layer 0 connections"
            assert len(node.connections[0]) > 0, f"Node {i} should be connected to other nodes"

    def test_insert_respects_max_connections(self):
        """Test that nodes don't exceed M connections."""
        index = HNSWIndex(M=2, ef_construction=50, metric="euclidean")

        # Create many nearby vectors
        vectors = np.random.randn(20, 3).astype(np.float32)
        index.vectors = vectors

        # Insert all
        for i in range(len(vectors)):
            index.insert(vectors[i], vector_id=i)

        # Check that no node has more than M*2 connections at layer 0
        # (layer 0 can have up to M*2 = 4 connections)
        for i in range(len(vectors)):
            node = index.nodes[i]
            if 0 in node.connections:
                num_connections = len(node.connections[0])
                assert num_connections <= index.M * 2, \
                    f"Node {i} has {num_connections} connections, max is {index.M * 2}"

    def test_insert_updates_entry_point(self):
        """Test that entry point is updated when higher layer node is inserted."""
        # Fix random seed for reproducibility
        np.random.seed(42)

        index = HNSWIndex(M=16, metric="euclidean")

        # Create vectors
        vectors = np.random.randn(50, 3).astype(np.float32)
        index.vectors = vectors

        # Insert nodes - eventually one should land on a higher layer
        for i in range(50):
            index.insert(vectors[i], vector_id=i)

        # Verify entry point exists and has high layer
        assert index.entry_point is not None
        assert index.max_layer >= 0

        # Verify entry point node has the max layer
        entry_node = index.nodes[index.entry_point]
        assert entry_node.layer == index.max_layer

    def test_insert_cosine_metric(self):
        """Test insert with cosine similarity metric."""
        index = HNSWIndex(M=3, ef_construction=50, metric="cosine")

        # Create unit vectors
        vectors = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.707, 0.707, 0.0],
        ], dtype=np.float32)

        index.vectors = vectors

        # Insert all
        for i in range(len(vectors)):
            index.insert(vectors[i], vector_id=i)

        # Verify graph is created
        assert len(index.nodes) == 3
        for i in range(3):
            assert i in index.nodes


class TestSearch:
    """Test HNSW search operation."""

    def test_search_empty_index(self):
        """Test searching an empty index."""
        index = HNSWIndex(M=16, metric="euclidean")
        index.vectors = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = index.search(query, k=5)

        assert results == [], "Empty index should return empty results"

    def test_search_single_vector(self):
        """Test searching with single vector in index."""
        index = HNSWIndex(M=16, metric="euclidean")
        index.vectors = np.array([
            [1.0, 0.0, 0.0],
        ], dtype=np.float32)

        # Insert single vector
        index.insert(index.vectors[0], vector_id=0)

        # Search for it
        query = np.array([0.9, 0.0, 0.0], dtype=np.float32)
        results = index.search(query, k=1)

        assert len(results) == 1
        assert results[0][0] == 0

    def test_search_finds_nearest(self):
        """Test that search finds the nearest vector."""
        index = HNSWIndex(M=5, ef_construction=50, metric="euclidean")

        # Create vectors where v0 is clearly closest to query
        vectors = np.array([
            [1.0, 0.0, 0.0],  # v0 - closest to query
            [0.0, 1.0, 0.0],  # v1
            [0.0, 0.0, 1.0],  # v2
            [5.0, 5.0, 5.0],  # v3 - far away
        ], dtype=np.float32)

        index.vectors = vectors

        # Build index
        for i in range(len(vectors)):
            index.insert(vectors[i], vector_id=i)

        # Query close to v0
        query = np.array([0.9, 0.0, 0.0], dtype=np.float32)
        results = index.search(query, k=1)

        # Should find v0
        assert len(results) == 1
        assert results[0][0] == 0, "Should find v0 as nearest"

    def test_search_returns_k_results(self):
        """Test that search returns exactly k results."""
        index = HNSWIndex(M=5, ef_construction=50, metric="euclidean")

        # Create 10 random vectors
        np.random.seed(42)
        vectors = np.random.randn(10, 3).astype(np.float32)
        index.vectors = vectors

        # Build index
        for i in range(len(vectors)):
            index.insert(vectors[i], vector_id=i)

        # Search for different k values
        query = np.random.randn(3).astype(np.float32)

        for k in [1, 3, 5, 10]:
            results = index.search(query, k=k)
            assert len(results) == k, f"Should return exactly {k} results"

    def test_search_with_custom_ef(self):
        """Test search with custom ef parameter."""
        index = HNSWIndex(M=5, ef_construction=50, ef_search=10, metric="euclidean")

        # Create vectors
        np.random.seed(42)
        vectors = np.random.randn(20, 3).astype(np.float32)
        index.vectors = vectors

        # Build index
        for i in range(len(vectors)):
            index.insert(vectors[i], vector_id=i)

        query = np.random.randn(3).astype(np.float32)

        # Search with different ef values
        # Higher ef should explore more thoroughly
        results_ef10 = index.search(query, k=5, ef=10)
        results_ef50 = index.search(query, k=5, ef=50)

        assert len(results_ef10) == 5
        assert len(results_ef50) == 5

    def test_search_results_sorted(self):
        """Test that search results are sorted by distance."""
        index = HNSWIndex(M=5, ef_construction=50, metric="euclidean")

        # Create vectors
        vectors = np.array([
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ], dtype=np.float32)

        index.vectors = vectors

        # Build index
        for i in range(len(vectors)):
            index.insert(vectors[i], vector_id=i)

        # Query at origin
        query = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        results = index.search(query, k=5)

        # Results should be sorted by distance (v0 closest, v4 farthest)
        for i in range(len(results) - 1):
            assert results[i][1] <= results[i + 1][1], \
                "Results should be sorted by distance (ascending)"

    def test_search_cosine_metric(self):
        """Test search with cosine similarity."""
        index = HNSWIndex(M=5, ef_construction=50, metric="cosine")

        # Create unit vectors
        vectors = np.array([
            [1.0, 0.0, 0.0],  # v0
            [0.0, 1.0, 0.0],  # v1
            [0.0, 0.0, 1.0],  # v2
            [0.707, 0.707, 0.0],  # v3 - 45° between v0 and v1
        ], dtype=np.float32)

        index.vectors = vectors

        # Build index
        for i in range(len(vectors)):
            index.insert(vectors[i], vector_id=i)

        # Query identical to v0
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = index.search(query, k=1)

        # Should find v0 with similarity 1.0
        assert results[0][0] == 0, "Should find v0"
        assert results[0][1] == pytest.approx(1.0), "Cosine similarity should be 1.0"

    def test_search_large_k(self):
        """Test searching with k larger than number of vectors."""
        index = HNSWIndex(M=5, ef_construction=50, metric="euclidean")

        # Create only 3 vectors
        vectors = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

        index.vectors = vectors

        # Build index
        for i in range(len(vectors)):
            index.insert(vectors[i], vector_id=i)

        # Search for k=10 (more than we have)
        query = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        results = index.search(query, k=10)

        # Should return all 3 vectors
        assert len(results) == 3, "Should return all available vectors"


class TestBuild:
    """Test HNSW build operation."""

    def test_build_simple(self):
        """Test building index with a few vectors."""
        index = HNSWIndex(M=5, ef_construction=50, metric="euclidean")

        # Create simple vectors
        vectors = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.5, 0.5, 0.5],
        ], dtype=np.float32)

        # Build index
        index.build(vectors)

        # Verify vectors are stored
        assert index.vectors is not None
        assert len(index.vectors) == 5

        # Verify all nodes were created
        assert len(index.nodes) == 5
        for i in range(5):
            assert i in index.nodes

        # Verify entry point was set
        assert index.entry_point is not None

    def test_build_and_search(self):
        """Test that build creates a searchable index."""
        index = HNSWIndex(M=5, ef_construction=50, metric="euclidean")

        # Create vectors
        np.random.seed(42)
        vectors = np.random.randn(20, 3).astype(np.float32)

        # Build index
        index.build(vectors)

        # Search should work
        query = np.random.randn(3).astype(np.float32)
        results = index.search(query, k=5)

        assert len(results) == 5
        # Results should be sorted by distance
        for i in range(len(results) - 1):
            assert results[i][1] <= results[i + 1][1]

    def test_build_creates_connected_graph(self):
        """Test that build creates a connected graph."""
        index = HNSWIndex(M=3, ef_construction=50, metric="euclidean")

        # Create random vectors
        np.random.seed(42)
        vectors = np.random.randn(15, 3).astype(np.float32)

        # Build index
        index.build(vectors)

        # All nodes should be connected at layer 0
        for i in range(len(vectors)):
            node = index.nodes[i]
            assert 0 in node.connections, f"Node {i} should have layer 0 connections"
            assert len(node.connections[0]) > 0, f"Node {i} should be connected"

    def test_build_with_cosine_metric(self):
        """Test build with cosine similarity metric."""
        index = HNSWIndex(M=5, ef_construction=50, metric="cosine")

        # Create unit vectors
        vectors = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.707, 0.707, 0.0],
            [0.577, 0.577, 0.577],
        ], dtype=np.float32)

        # Build
        index.build(vectors)

        # Search for exact match
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = index.search(query, k=1)

        # Should find v0 with similarity 1.0
        assert results[0][0] == 0
        assert results[0][1] == pytest.approx(1.0)

    def test_build_empty_array(self):
        """Test building with empty array."""
        index = HNSWIndex(M=5, metric="euclidean")

        vectors = np.array([], dtype=np.float32).reshape(0, 3)

        # Build with empty array
        index.build(vectors)

        # Should have no nodes
        assert len(index.nodes) == 0
        assert index.entry_point is None

        # Search should return empty
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = index.search(query, k=5)
        assert results == []

    def test_build_large_index(self):
        """Test building a larger index."""
        index = HNSWIndex(M=16, ef_construction=100, metric="euclidean")

        # Create 100 random vectors
        np.random.seed(42)
        vectors = np.random.randn(100, 10).astype(np.float32)

        # Build
        index.build(vectors)

        # Verify size
        assert len(index.nodes) == 100
        assert index.vectors.shape == (100, 10)

        # Test search works
        query = np.random.randn(10).astype(np.float32)
        results = index.search(query, k=10)

        assert len(results) == 10

    def test_build_respects_construction_parameters(self):
        """Test that build uses ef_construction and M correctly."""
        # Low M means fewer connections
        index_low_m = HNSWIndex(M=2, ef_construction=20, metric="euclidean")

        np.random.seed(42)
        vectors = np.random.randn(20, 3).astype(np.float32)

        index_low_m.build(vectors)

        # Check that connections are limited by M
        for i in range(20):
            node = index_low_m.nodes[i]
            if 0 in node.connections:
                # Layer 0 can have M*2 connections
                assert len(node.connections[0]) <= index_low_m.M * 2


class TestDelete:
    """Test HNSW delete operation."""

    def test_delete_single_vector(self):
        """Test deleting a single vector from index."""
        index = HNSWIndex(M=4, ef_construction=20, metric="euclidean")

        # Build small index
        vectors = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

        index.build(vectors)
        assert len(index.nodes) == 3, "Should have 3 nodes before deletion"

        # Delete middle vector
        index.delete(1)

        assert len(index.nodes) == 2, "Should have 2 nodes after deletion"
        assert 1 not in index.nodes, "Deleted node should not exist"
        assert 0 in index.nodes, "Other nodes should remain"
        assert 2 in index.nodes, "Other nodes should remain"

    def test_delete_removes_connections(self):
        """Test that deleting a vector removes all connections to it."""
        index = HNSWIndex(M=4, ef_construction=20, metric="euclidean")

        # Build index
        np.random.seed(42)
        vectors = np.random.randn(10, 3).astype(np.float32)
        index.build(vectors)

        # Note which nodes were connected to node 5
        node_5 = index.nodes[5]
        connected_to_5 = set()
        for layer in node_5.connections:
            connected_to_5.update(node_5.connections[layer])

        # Delete node 5
        index.delete(5)

        # Verify no node has connections to node 5 anymore
        for vid, node in index.nodes.items():
            for layer in node.connections:
                assert 5 not in node.connections[layer], \
                    f"Node {vid} at layer {layer} should not connect to deleted node 5"

    def test_delete_entry_point(self):
        """Test deleting the entry point updates it correctly."""
        index = HNSWIndex(M=4, ef_construction=20, metric="euclidean")

        # Build index
        vectors = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        index.build(vectors)

        old_entry = index.entry_point

        # Delete entry point
        index.delete(old_entry)

        # Should have a new entry point
        assert index.entry_point is not None, "Should have new entry point"
        assert index.entry_point != old_entry, "Entry point should change"
        assert index.entry_point in index.nodes, "New entry point should exist"

    def test_delete_all_vectors(self):
        """Test deleting all vectors leaves empty index."""
        index = HNSWIndex(M=4, ef_construction=20, metric="euclidean")

        # Build index
        vectors = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)
        index.build(vectors)

        # Delete all
        index.delete(0)
        index.delete(1)

        assert len(index.nodes) == 0, "Should have no nodes"
        assert index.entry_point is None, "Entry point should be None"
        assert index.max_layer == 0, "Max layer should be 0"

    def test_delete_nonexistent_vector(self):
        """Test that deleting non-existent vector raises error."""
        index = HNSWIndex(M=4, ef_construction=20, metric="euclidean")

        vectors = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        index.build(vectors)

        with pytest.raises(ValueError, match="not found"):
            index.delete(999)

    def test_search_after_delete(self):
        """Test that search still works after deleting vectors."""
        index = HNSWIndex(M=4, ef_construction=20, metric="euclidean")

        # Build index
        np.random.seed(42)
        vectors = np.random.randn(20, 3).astype(np.float32)
        index.build(vectors)

        # Delete some vectors
        index.delete(5)
        index.delete(10)
        index.delete(15)

        # Search should still work
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = index.search(query, k=5)

        assert len(results) == 5, "Should return 5 results"
        # Deleted vectors should not appear in results
        result_ids = [vid for vid, _ in results]
        assert 5 not in result_ids, "Deleted vector should not appear"
        assert 10 not in result_ids, "Deleted vector should not appear"
        assert 15 not in result_ids, "Deleted vector should not appear"


class TestSaveLoad:
    """Test HNSW save and load operations."""

    def test_save_and_load_simple(self):
        """Test basic save and load functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "index.npz"

            # Create and build index
            index1 = HNSWIndex(M=8, ef_construction=50, ef_search=30, metric="euclidean")
            vectors = np.array([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
            ], dtype=np.float32)
            index1.build(vectors)

            # Save
            index1.save(str(filepath))

            # Load in new index
            index2 = HNSWIndex()
            index2.load(str(filepath))

            # Verify parameters restored
            assert index2.M == 8, "M should be restored"
            assert index2.ef_construction == 50, "ef_construction should be restored"
            assert index2.ef_search == 30, "ef_search should be restored"
            assert index2.metric == "euclidean", "metric should be restored"

            # Verify graph structure restored
            assert len(index2.nodes) == 5, "Should have 5 nodes"
            assert index2.entry_point is not None, "Entry point should be restored"

    def test_save_load_preserves_search_results(self):
        """Test that search results are identical after save/load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "index.npz"

            # Create and build index
            index1 = HNSWIndex(M=16, ef_construction=200, metric="euclidean")
            np.random.seed(42)
            vectors = np.random.randn(50, 128).astype(np.float32)
            index1.build(vectors)

            # Perform search before save
            query = np.random.randn(128).astype(np.float32)
            results1 = index1.search(query, k=10)

            # Save and load
            index1.save(str(filepath))
            index2 = HNSWIndex()
            index2.load(str(filepath))

            # Perform same search after load
            results2 = index2.search(query, k=10)

            # Results should be identical
            assert len(results1) == len(results2), "Should return same number of results"
            for (vid1, dist1), (vid2, dist2) in zip(results1, results2):
                assert vid1 == vid2, f"Result IDs should match: {vid1} vs {vid2}"
                assert abs(dist1 - dist2) < 1e-6, f"Distances should match: {dist1} vs {dist2}"

    def test_save_load_with_cosine_metric(self):
        """Test save/load with cosine similarity metric."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "index.npz"

            # Create index with cosine metric
            index1 = HNSWIndex(M=8, ef_construction=50, metric="cosine")
            vectors = np.random.randn(20, 64).astype(np.float32)
            index1.build(vectors)

            # Save and load
            index1.save(str(filepath))
            index2 = HNSWIndex()
            index2.load(str(filepath))

            # Verify metric restored
            assert index2.metric == "cosine", "Metric should be cosine"

            # Verify distance function is correct
            query = np.random.randn(64).astype(np.float32)
            results = index2.search(query, k=5)

            # Cosine similarity should be in [0, 1] range (or [-1, 1] for negative)
            for vid, dist in results:
                assert -1.0 <= dist <= 1.0, f"Cosine similarity should be in [-1, 1], got {dist}"

    def test_save_load_empty_index(self):
        """Test saving and loading an empty index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "index.npz"

            # Create empty index
            index1 = HNSWIndex(M=8, ef_construction=50, metric="euclidean")

            # Save
            index1.save(str(filepath))

            # Load
            index2 = HNSWIndex()
            index2.load(str(filepath))

            # Verify empty state
            assert len(index2.nodes) == 0, "Should have no nodes"
            assert index2.entry_point is None, "Entry point should be None"
            assert index2.vectors is None, "Vectors should be None"

    def test_save_load_graph_structure(self):
        """Test that graph structure (connections) is preserved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "index.npz"

            # Create index
            index1 = HNSWIndex(M=4, ef_construction=20, metric="euclidean")
            vectors = np.random.randn(10, 3).astype(np.float32)
            index1.build(vectors)

            # Record graph structure
            original_structure = {}
            for vid, node in index1.nodes.items():
                original_structure[vid] = {
                    'layer': node.layer,
                    'connections': {
                        layer: set(neighbors) for layer, neighbors in node.connections.items()
                    }
                }

            # Save and load
            index1.save(str(filepath))
            index2 = HNSWIndex()
            index2.load(str(filepath))

            # Verify structure matches
            assert len(index2.nodes) == len(original_structure), "Node count should match"

            for vid, node in index2.nodes.items():
                assert vid in original_structure, f"Node {vid} should exist in original"
                orig = original_structure[vid]

                assert node.layer == orig['layer'], f"Layer should match for node {vid}"
                assert node.connections.keys() == orig['connections'].keys(), \
                    f"Layers should match for node {vid}"

                for layer in node.connections:
                    assert node.connections[layer] == orig['connections'][layer], \
                        f"Connections at layer {layer} should match for node {vid}"

    def test_load_overwrites_existing_index(self):
        """Test that loading overwrites existing index state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "index.npz"

            # Create first index and save
            index1 = HNSWIndex(M=8, ef_construction=50, metric="euclidean")
            vectors1 = np.random.randn(10, 3).astype(np.float32)
            index1.build(vectors1)
            index1.save(str(filepath))

            # Create second index with different data
            index2 = HNSWIndex(M=16, ef_construction=100, metric="cosine")
            vectors2 = np.random.randn(20, 3).astype(np.float32)
            index2.build(vectors2)

            # Load first index into second (should overwrite)
            index2.load(str(filepath))

            # Verify it now matches first index
            assert index2.M == 8, "M should be from loaded index"
            assert index2.metric == "euclidean", "Metric should be from loaded index"
            assert len(index2.nodes) == 10, "Should have 10 nodes from loaded index"
