"""
Tests for brute-force index implementation.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
from src.index.brute_force import BruteForceIndex


class TestBruteForceIndex:
    """Test suite for BruteForceIndex."""

    @pytest.fixture
    def sample_vectors(self):
        """Create sample vectors for testing."""
        # Create 10 simple 3D vectors
        return np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
        ], dtype=np.float32)

    def test_build_index(self, sample_vectors):
        """Test building an index from vectors."""
        index = BruteForceIndex(metric="euclidean")
        index.build(sample_vectors)

        assert len(index.vectors) == 10, "Should have 10 vectors"
        assert len(index.vector_ids) == 10, "Should have 10 IDs"

    def test_search_euclidean(self, sample_vectors):
        """Test search with Euclidean distance."""
        index = BruteForceIndex(metric="euclidean")
        index.build(sample_vectors)

        # Query with [1, 0, 0] - should find vector 0 as closest
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = index.search(query, k=3)

        assert len(results) == 3, "Should return 3 results"

        # First result should be exact match (ID 0, distance 0)
        assert results[0][0] == 0, "First result should be vector 0"
        assert results[0][1] == pytest.approx(0.0), "Distance should be 0"

    def test_search_cosine(self, sample_vectors):
        """Test search with cosine similarity."""
        index = BruteForceIndex(metric="cosine")
        index.build(sample_vectors)

        # Query with [1, 0, 0] - should find vector 0 and 7 as most similar
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = index.search(query, k=3)

        assert len(results) == 3, "Should return 3 results"

        # First result should be exact match (ID 0 or 7, similarity 1.0)
        assert results[0][0] in [0, 7], "First result should be vector 0 or 7"
        assert results[0][1] == pytest.approx(1.0), "Cosine similarity should be 1.0"

    def test_insert(self):
        """Test inserting vectors one by one."""
        index = BruteForceIndex(metric="euclidean")

        # Insert vectors
        index.insert(np.array([1.0, 0.0, 0.0], dtype=np.float32), vector_id=0)
        index.insert(np.array([0.0, 1.0, 0.0], dtype=np.float32), vector_id=1)
        index.insert(np.array([0.0, 0.0, 1.0], dtype=np.float32), vector_id=2)

        assert len(index.vectors) == 3, "Should have 3 vectors"

        # Search
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = index.search(query, k=1)

        assert results[0][0] == 0, "Should find vector 0"

    def test_delete(self, sample_vectors):
        """Test deleting a vector."""
        index = BruteForceIndex(metric="euclidean")
        index.build(sample_vectors)

        # Delete vector 0
        index.delete(0)

        assert len(index.vectors) == 9, "Should have 9 vectors after deletion"
        assert 0 not in index.vector_ids, "Vector 0 should be removed"

    def test_save_load(self, sample_vectors):
        """Test saving and loading index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "index.npz"

            # Create and save index
            index1 = BruteForceIndex(metric="euclidean")
            index1.build(sample_vectors)
            index1.save(str(filepath))

            # Load in new index
            index2 = BruteForceIndex(metric="euclidean")
            index2.load(str(filepath))

            assert len(index2.vectors) == 10, "Loaded index should have 10 vectors"
            assert index2.metric == "euclidean", "Metric should be euclidean"

            # Verify search works the same
            query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            results1 = index1.search(query, k=3)
            results2 = index2.search(query, k=3)

            assert results1 == results2, "Search results should match"

    def test_empty_index_search(self):
        """Test searching an empty index."""
        index = BruteForceIndex(metric="euclidean")
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = index.search(query, k=5)

        assert results == [], "Empty index should return empty results"

    def test_k_larger_than_dataset(self, sample_vectors):
        """Test when k is larger than number of vectors."""
        index = BruteForceIndex(metric="euclidean")
        index.build(sample_vectors[:5])  # Only 5 vectors

        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = index.search(query, k=10)  # Ask for 10

        assert len(results) == 5, "Should return only 5 results"

    def test_invalid_metric(self):
        """Test that invalid metric raises error."""
        with pytest.raises(ValueError, match="Unsupported metric"):
            BruteForceIndex(metric="invalid")
