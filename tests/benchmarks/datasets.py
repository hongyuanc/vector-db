"""
Dataset utilities for benchmarking vector indices.

Provides synthetic dataset generation for benchmarking without external dependencies.
Can be extended later to support real-world datasets like SIFT1M.
"""

import numpy as np
from typing import Tuple, Optional


def generate_random_dataset(
    n_vectors: int,
    dimension: int,
    n_queries: int = 100,
    seed: int = 42,
    distribution: str = "normal"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic random dataset for benchmarking.

    This creates random vectors and queries with controlled randomness
    for reproducible benchmarks. While not as realistic as real-world
    datasets, it's sufficient for demonstrating algorithm performance.

    Args:
        n_vectors: Number of vectors in the database
        dimension: Dimensionality of vectors
        n_queries: Number of query vectors to generate
        seed: Random seed for reproducibility
        distribution: "normal" (Gaussian) or "uniform"

    Returns:
        Tuple of (database_vectors, query_vectors)

    Example:
        >>> vectors, queries = generate_random_dataset(10000, 128)
        >>> vectors.shape
        (10000, 128)
        >>> queries.shape
        (100, 128)
    """
    np.random.seed(seed)

    if distribution == "normal":
        # Gaussian distribution (most common for vector embeddings)
        vectors = np.random.randn(n_vectors, dimension).astype(np.float32)
        queries = np.random.randn(n_queries, dimension).astype(np.float32)
    elif distribution == "uniform":
        # Uniform distribution in [0, 1]
        vectors = np.random.rand(n_vectors, dimension).astype(np.float32)
        queries = np.random.rand(n_queries, dimension).astype(np.float32)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    return vectors, queries


def generate_clustered_dataset(
    n_vectors: int,
    dimension: int,
    n_clusters: int = 10,
    n_queries: int = 100,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate clustered dataset (more realistic than pure random).

    Creates vectors grouped around cluster centers, similar to how
    real embeddings often have semantic clusters. This tests whether
    HNSW can effectively navigate clustered data.

    Args:
        n_vectors: Number of vectors in the database
        dimension: Dimensionality of vectors
        n_clusters: Number of clusters
        n_queries: Number of query vectors
        seed: Random seed for reproducibility

    Returns:
        Tuple of (database_vectors, query_vectors)
    """
    np.random.seed(seed)

    # Generate cluster centers
    centers = np.random.randn(n_clusters, dimension).astype(np.float32) * 3.0

    # Assign each vector to a random cluster
    cluster_assignments = np.random.randint(0, n_clusters, size=n_vectors)

    # Generate vectors around cluster centers with some noise
    vectors = np.zeros((n_vectors, dimension), dtype=np.float32)
    for i in range(n_vectors):
        cluster_idx = cluster_assignments[i]
        # Vector = cluster center + small noise
        vectors[i] = centers[cluster_idx] + np.random.randn(dimension).astype(np.float32) * 0.5

    # Generate queries (also clustered, so they have nearby neighbors)
    query_clusters = np.random.randint(0, n_clusters, size=n_queries)
    queries = np.zeros((n_queries, dimension), dtype=np.float32)
    for i in range(n_queries):
        cluster_idx = query_clusters[i]
        queries[i] = centers[cluster_idx] + np.random.randn(dimension).astype(np.float32) * 0.5

    return vectors, queries


def compute_ground_truth(
    vectors: np.ndarray,
    queries: np.ndarray,
    k: int,
    metric: str = "euclidean"
) -> np.ndarray:
    """
    Compute ground truth nearest neighbors using brute force.

    This is the "correct" answer that we compare approximate methods against
    to calculate recall. Uses exhaustive search so it's slow but accurate.

    Args:
        vectors: Database vectors (n_vectors x dimension)
        queries: Query vectors (n_queries x dimension)
        k: Number of neighbors to find
        metric: "euclidean" or "cosine"

    Returns:
        Ground truth neighbor IDs (n_queries x k)
        Each row contains k nearest neighbor IDs for that query

    Example:
        >>> vectors, queries = generate_random_dataset(1000, 128, n_queries=10)
        >>> gt = compute_ground_truth(vectors, queries, k=10)
        >>> gt.shape
        (10, 10)
    """
    from src.index.brute_force import BruteForceIndex

    # Use brute force to get exact results
    bf_index = BruteForceIndex(metric=metric)
    bf_index.build(vectors)

    # Compute exact neighbors for each query
    ground_truth = np.zeros((len(queries), k), dtype=np.int32)

    for i, query in enumerate(queries):
        results = bf_index.search(query, k=k)
        # Extract just the IDs
        neighbor_ids = [vid for vid, _ in results]
        ground_truth[i] = neighbor_ids[:k]  # Ensure exactly k neighbors

    return ground_truth


def get_benchmark_dataset(
    name: str = "random",
    size: str = "small",
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get a benchmark dataset by name and size.

    Provides a convenient interface for getting datasets of different sizes.
    Defaults to synthetic data that works without downloads.

    Args:
        name: Dataset type - "random" or "clustered"
        size: "small" (10k), "medium" (100k), "large" (1M)
        **kwargs: Additional arguments passed to dataset generator

    Returns:
        Tuple of (database_vectors, query_vectors)

    Example:
        >>> # Quick test with small dataset
        >>> vectors, queries = get_benchmark_dataset("random", "small")
        >>> # More realistic test with clustered data
        >>> vectors, queries = get_benchmark_dataset("clustered", "medium")
    """
    # Define sizes
    sizes = {
        "tiny": 1_000,
        "small": 10_000,
        "medium": 100_000,
        "large": 1_000_000,
    }

    n_vectors = sizes.get(size, 10_000)
    dimension = kwargs.get("dimension", 128)
    n_queries = kwargs.get("n_queries", 100)
    seed = kwargs.get("seed", 42)

    if name == "random":
        return generate_random_dataset(
            n_vectors=n_vectors,
            dimension=dimension,
            n_queries=n_queries,
            seed=seed,
            distribution=kwargs.get("distribution", "normal")
        )
    elif name == "clustered":
        return generate_clustered_dataset(
            n_vectors=n_vectors,
            dimension=dimension,
            n_clusters=kwargs.get("n_clusters", 10),
            n_queries=n_queries,
            seed=seed
        )
    else:
        raise ValueError(f"Unknown dataset: {name}. Use 'random' or 'clustered'")
