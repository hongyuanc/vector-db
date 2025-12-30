"""
Recall accuracy benchmarks.

Measures the quality of approximate search results by comparing against
ground truth from brute force search.

Recall@k = (# correct neighbors found) / k

Perfect recall = 1.0 (found all correct neighbors)
Lower recall = missed some correct neighbors
"""

import pytest
import numpy as np
from src.index.hnsw import HNSWIndex
from .datasets import get_benchmark_dataset, compute_ground_truth


def calculate_recall(predictions: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Calculate recall@k for nearest neighbor predictions.

    Args:
        predictions: Predicted neighbor IDs (n_queries x k)
        ground_truth: True neighbor IDs (n_queries x k)

    Returns:
        Average recall across all queries
    """
    n_queries = len(predictions)
    total_recall = 0.0

    for i in range(n_queries):
        # Convert to sets for intersection
        pred_set = set(predictions[i])
        truth_set = set(ground_truth[i])

        # How many correct neighbors did we find?
        correct = len(pred_set.intersection(truth_set))
        k = len(ground_truth[i])

        recall = correct / k
        total_recall += recall

    return total_recall / n_queries


class TestRecallAccuracy:
    """Measure recall accuracy of HNSW vs ground truth."""

    @pytest.mark.benchmark
    def test_recall_with_default_params(self):
        """
        Measure HNSW recall with high-quality parameters.

        With ef_search=200, should achieve ~95%+ recall.
        This shows that HNSW is both fast AND accurate.
        """
        # Generate dataset
        vectors, queries = get_benchmark_dataset("random", "small", dimension=128, n_queries=100)
        k = 10

        print(f"\n{'='*60}")
        print(f"Recall Accuracy Benchmark")
        print(f"{'='*60}")
        print(f"Dataset: {len(vectors):,} vectors x {vectors.shape[1]} dimensions")
        print(f"Queries: {len(queries):,}")
        print(f"k: {k}")
        print()

        # Compute ground truth (exact neighbors)
        print("Computing ground truth (brute force)...")
        ground_truth = compute_ground_truth(vectors, queries, k=k, metric="euclidean")

        # Build HNSW index
        print("Building HNSW index...")
        hnsw_index = HNSWIndex(M=16, ef_construction=200, ef_search=200, metric="euclidean")
        hnsw_index.build(vectors)

        # Get HNSW predictions
        print("Searching with HNSW...")
        predictions = np.zeros((len(queries), k), dtype=np.int32)
        for i, query in enumerate(queries):
            results = hnsw_index.search(query, k=k)
            neighbor_ids = [vid for vid, _ in results]
            predictions[i] = neighbor_ids[:k]

        # Calculate recall
        recall = calculate_recall(predictions, ground_truth)

        print()
        print(f"Results:")
        print(f"  Recall@{k}: {recall:.2%}")
        print(f"  Parameters: M=16, ef_construction=200, ef_search=200")
        print(f"{'='*60}\n")

        # Assert good recall
        assert recall >= 0.90, f"Recall should be at least 90%, got {recall:.2%}"

    @pytest.mark.benchmark
    def test_recall_vs_ef_search(self):
        """
        Show recall/speed tradeoff by varying ef_search.

        Lower ef_search = faster but lower recall
        Higher ef_search = slower but higher recall

        This is the key tunable parameter for production systems.
        """
        vectors, queries = get_benchmark_dataset("random", "small", dimension=128, n_queries=50)
        k = 10

        print(f"\n{'='*60}")
        print(f"Recall vs ef_search Parameter")
        print(f"{'='*60}")
        print(f"Dataset: {len(vectors):,} vectors")
        print(f"Testing ef_search values: [10, 20, 50, 100, 200]")
        print()

        # Compute ground truth
        print("Computing ground truth...")
        ground_truth = compute_ground_truth(vectors, queries, k=k, metric="euclidean")

        # Build HNSW index
        print("Building HNSW index...")
        hnsw_index = HNSWIndex(M=16, ef_construction=200, metric="euclidean")
        hnsw_index.build(vectors)

        # Test different ef_search values
        ef_values = [10, 20, 50, 100, 200]

        print()
        print(f"{'ef_search':<12} {'Recall@10':<12}")
        print(f"{'-'*25}")

        for ef in ef_values:
            predictions = np.zeros((len(queries), k), dtype=np.int32)

            for i, query in enumerate(queries):
                results = hnsw_index.search(query, k=k, ef=ef)
                neighbor_ids = [vid for vid, _ in results]
                predictions[i] = neighbor_ids[:k]

            recall = calculate_recall(predictions, ground_truth)
            print(f"{ef:<12} {recall:<12.2%}")

        print(f"{'='*60}\n")

    @pytest.mark.benchmark
    def test_recall_vs_M_parameter(self):
        """
        Show how M parameter affects recall.

        M controls graph connectivity:
        - Higher M = more connections = better recall but slower build
        - Lower M = fewer connections = faster build but lower recall

        This is set at build time and can't be changed after.
        """
        vectors, queries = get_benchmark_dataset("random", "small", dimension=128, n_queries=50)
        k = 10

        print(f"\n{'='*60}")
        print(f"Recall vs M Parameter (Graph Connectivity)")
        print(f"{'='*60}")
        print(f"Dataset: {len(vectors):,} vectors")
        print(f"Testing M values: [4, 8, 16, 32]")
        print()

        # Compute ground truth
        print("Computing ground truth...")
        ground_truth = compute_ground_truth(vectors, queries, k=k, metric="euclidean")

        M_values = [4, 8, 16, 32]

        print()
        print(f"{'M':<8} {'Recall@10':<12} {'Note'}")
        print(f"{'-'*50}")

        for M in M_values:
            # Build index with this M
            hnsw_index = HNSWIndex(M=M, ef_construction=200, ef_search=50, metric="euclidean")
            hnsw_index.build(vectors)

            # Get predictions
            predictions = np.zeros((len(queries), k), dtype=np.int32)
            for i, query in enumerate(queries):
                results = hnsw_index.search(query, k=k)
                neighbor_ids = [vid for vid, _ in results]
                predictions[i] = neighbor_ids[:k]

            recall = calculate_recall(predictions, ground_truth)

            note = "Low connectivity" if M <= 8 else "High connectivity" if M >= 32 else "Balanced"
            print(f"{M:<8} {recall:<12.2%} {note}")

        print(f"{'='*60}\n")

    @pytest.mark.benchmark
    def test_recall_on_clustered_data(self):
        """
        Test recall on clustered dataset.

        Clustered data is more realistic - real embeddings often have
        semantic clusters. HNSW should handle this well.
        """
        vectors, queries = get_benchmark_dataset(
            "clustered", "small", dimension=128, n_queries=50, n_clusters=20
        )
        k = 10

        print(f"\n{'='*60}")
        print(f"Recall on Clustered Dataset")
        print(f"{'='*60}")
        print(f"Dataset: {len(vectors):,} vectors in ~20 clusters")
        print(f"Queries: {len(queries):,}")
        print()

        # Compute ground truth
        print("Computing ground truth...")
        ground_truth = compute_ground_truth(vectors, queries, k=k, metric="euclidean")

        # Build HNSW
        print("Building HNSW index...")
        hnsw_index = HNSWIndex(M=16, ef_construction=200, ef_search=50, metric="euclidean")
        hnsw_index.build(vectors)

        # Get predictions
        predictions = np.zeros((len(queries), k), dtype=np.int32)
        for i, query in enumerate(queries):
            results = hnsw_index.search(query, k=k)
            neighbor_ids = [vid for vid, _ in results]
            predictions[i] = neighbor_ids[:k]

        recall = calculate_recall(predictions, ground_truth)

        print()
        print(f"Results:")
        print(f"  Recall@{k}: {recall:.2%}")
        print(f"  Note: Clustered data often has higher recall")
        print(f"        because neighbors are naturally grouped")
        print(f"{'='*60}\n")

        assert recall >= 0.92, f"Recall on clustered data should be high, got {recall:.2%}"
