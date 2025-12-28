"""
Tests for vector storage implementation.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
from src.storage.vector_store import VectorStore


class TestVectorStore:
    """Test suite for VectorStore."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def vector_store(self, temp_dir):
        """Create a vector store for testing."""
        return VectorStore(dimension=128, max_vectors=1000, data_dir=temp_dir)

    def test_insert_vector(self, vector_store):
        """Test inserting a single vector."""
        vector = np.random.randn(128).astype(np.float32)
        vector_id = vector_store.insert(vector)

        assert vector_id == 0, "First vector should have ID 0"
        assert vector_store.count == 1, "Count should be 1"

    def test_get_vector(self, vector_store):
        """Test retrieving a vector by ID."""
        # Insert a vector
        original = np.array([1.0, 2.0, 3.0] + [0.0] * 125, dtype=np.float32)
        vector_id = vector_store.insert(original)

        # Retrieve it
        retrieved = vector_store.get(vector_id)

        # Verify it matches
        assert np.allclose(retrieved, original), "Retrieved vector should match original"

    def test_insert_multiple_vectors(self, vector_store):
        """Test inserting multiple vectors."""
        vectors = [np.random.randn(128).astype(np.float32) for _ in range(10)]
        ids = [vector_store.insert(v) for v in vectors]

        assert len(ids) == 10, "Should have 10 IDs"
        assert ids == list(range(10)), "IDs should be 0-9"
        assert vector_store.count == 10, "Count should be 10"

    def test_get_all_vectors(self, vector_store):
        """Test retrieving all vectors."""
        # Insert 5 vectors
        vectors = [np.random.randn(128).astype(np.float32) for _ in range(5)]
        for v in vectors:
            vector_store.insert(v)

        # Get all
        all_vectors = vector_store.get_all_vectors()

        assert all_vectors.shape == (5, 128), "Shape should be (5, 128)"

    def test_persistence(self, temp_dir):
        """Test that vectors persist across store instances."""
        # Create store and insert vectors
        store1 = VectorStore(dimension=128, max_vectors=1000, data_dir=temp_dir)
        vector1 = np.array([1.0, 2.0, 3.0] + [0.0] * 125, dtype=np.float32)
        vector2 = np.array([4.0, 5.0, 6.0] + [0.0] * 125, dtype=np.float32)

        store1.insert(vector1)
        store1.insert(vector2)
        store1.close()

        # Load store in new instance
        store2 = VectorStore(dimension=128, max_vectors=1000, data_dir=temp_dir)

        assert store2.count == 2, "Count should be 2 after reload"

        # Verify vectors match
        retrieved1 = store2.get(0)
        retrieved2 = store2.get(1)

        assert np.allclose(retrieved1, vector1), "First vector should persist"
        assert np.allclose(retrieved2, vector2), "Second vector should persist"

        store2.close()

    def test_invalid_dimension(self, vector_store):
        """Test that inserting wrong dimension raises error."""
        wrong_vector = np.random.randn(64).astype(np.float32)  # Wrong size

        with pytest.raises(ValueError, match="dimension"):
            vector_store.insert(wrong_vector)


@pytest.mark.slow
class TestVectorStorePerformance:
    """Performance tests for VectorStore."""

    def test_insert_throughput(self):
        """Test insert throughput."""
        # TODO: Implement benchmark
        pass

    def test_retrieval_latency(self):
        """Test retrieval latency."""
        # TODO: Implement benchmark
        pass
