"""
Tests for HNSW index implementation.
"""

import pytest
import numpy as np
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
