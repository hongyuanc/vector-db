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
