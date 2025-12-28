"""
Tests for distance metric implementations.
"""

import pytest
import numpy as np
from src.utils.distance import (
    cosine_similarity,
    euclidean_distance,
    dot_product,
    batch_cosine_similarity,
    batch_euclidean_distance,
)


class TestDistanceMetrics:
    """Test suite for distance metrics."""

    def test_cosine_similarity_identical_vectors(self):
        """Test that cosine similarity of identical vectors is 1."""
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([1.0, 2.0, 3.0, 4.0])
        similarity = cosine_similarity(a, b)
        assert similarity == pytest.approx(1.0), f"Expected 1.0, got {similarity}"

    def test_cosine_similarity_orthogonal_vectors(self):
        """Test that cosine similarity of orthogonal vectors is 0."""
        # In 2D: (1,0) and (0,1) are perpendicular (90 degrees)
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        similarity = cosine_similarity(a, b)
        assert similarity == pytest.approx(0.0), f"Expected 0.0, got {similarity}"

    def test_cosine_similarity_opposite_vectors(self):
        """Test that cosine similarity of opposite vectors is -1."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([-1.0, -2.0, -3.0])
        similarity = cosine_similarity(a, b)
        assert similarity == pytest.approx(-1.0), f"Expected -1.0, got {similarity}"

    def test_cosine_similarity_magnitude_independent(self):
        """Test that cosine similarity is independent of magnitude."""
        # Same direction, different lengths -> should have similarity of 1
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([2.0, 4.0, 6.0])  # 2x of a
        similarity = cosine_similarity(a, b)
        assert similarity == pytest.approx(1.0), f"Expected 1.0, got {similarity}"

    def test_euclidean_distance_identical_vectors(self):
        """Test that Euclidean distance of identical vectors is 0."""
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([1.0, 2.0, 3.0, 4.0])
        distance = euclidean_distance(a, b)
        assert distance == pytest.approx(0.0), f"Expected 0.0, got {distance}"

    def test_euclidean_distance_known_value(self):
        """Test Euclidean distance with known values."""
        # Simple 2D case: (0,0) to (3,4) should be 5 (classic 3-4-5 triangle)
        a = np.array([0.0, 0.0])
        b = np.array([3.0, 4.0])
        distance = euclidean_distance(a, b)
        assert distance == pytest.approx(5.0), f"Expected 5.0, got {distance}"

        # Test with higher dimensions
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 6.0, 8.0])
        # sqrt((4-1)² + (6-2)² + (8-3)²) = sqrt(9 + 16 + 25) = sqrt(50)
        expected = np.sqrt(50.0)
        distance = euclidean_distance(a, b)
        assert distance == pytest.approx(expected), f"Expected {expected}, got {distance}"

    def test_dot_product(self):
        """Test dot product calculation."""
        # Simple test: [1,2,3] · [4,5,6] = 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        result = dot_product(a, b)
        assert result == pytest.approx(32.0), f"Expected 32.0, got {result}"

    def test_dot_product_orthogonal(self):
        """Test that orthogonal vectors have dot product of 0."""
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        result = dot_product(a, b)
        assert result == pytest.approx(0.0), f"Expected 0.0, got {result}"

    def test_dot_product_equals_cosine_for_normalized(self):
        """For normalized vectors, dot product should equal cosine similarity."""
        # Create a normalized vector (unit length)
        a = np.array([0.6, 0.8])  # magnitude = sqrt(0.36 + 0.64) = 1.0
        b = np.array([0.8, 0.6])  # magnitude = sqrt(0.64 + 0.36) = 1.0

        dot = dot_product(a, b)
        cos = cosine_similarity(a, b)

        assert dot == pytest.approx(cos), f"Dot={dot}, Cosine={cos} should be equal"

    def test_batch_operations(self):
        """Test batch distance calculations."""
        # TODO: Implement test
        pass


@pytest.mark.benchmark
class TestDistancePerformance:
    """Performance benchmarks for distance metrics."""

    def test_single_distance_performance(self):
        """Benchmark single distance calculation."""
        # TODO: Implement benchmark
        pass

    def test_batch_distance_performance(self):
        """Benchmark batch distance calculations."""
        # TODO: Implement benchmark
        pass
