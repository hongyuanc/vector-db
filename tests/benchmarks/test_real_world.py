"""
Real-world dataset benchmarks.

Tests HNSW performance on standard benchmark datasets used in academic papers:
- SIFT1M: Image descriptors (128D)
- GIST1M: Image descriptors (960D)
- GloVe: Text embeddings (50-300D)

These provide realistic performance metrics comparable to published research.
"""

import pytest
import numpy as np
import time
from src.index.hnsw import HNSWIndex
from src.index.brute_force import BruteForceIndex
from .real_datasets import load_dataset, list_available_datasets, download_dataset
from .test_recall_accuracy import calculate_recall


class TestRealWorldDatasets:
    """Benchmark HNSW on standard real-world datasets."""

    @pytest.mark.benchmark
    @pytest.mark.real_data
    def test_sift1m_small(self):
        """
        SIFT1M benchmark (10k subset).

        SIFT1M is the standard benchmark for vector search.
        Used in almost every ANN paper since 2011.

        Expected performance (with M=16, ef_construction=200):
        - Recall@10 > 90% (ef_search=100-200)
        - QPS: 500-2000 queries/sec
        """
        try:
            vectors, queries, ground_truth = load_dataset(
                "sift1m", subset="small", download_if_missing=False
            )
        except FileNotFoundError:
            pytest.skip("SIFT1M not downloaded. Run: download_dataset('sift1m')")

        k = 10

        print(f"\n{'='*60}")
        print(f"SIFT1M Benchmark (Small)")
        print(f"{'='*60}")
        print(f"Dataset: {len(vectors):,} vectors x {vectors.shape[1]}D")
        print(f"Queries: {len(queries):,}")
        print(f"k: {k}")
        print()

        # Build HNSW index
        print("Building HNSW index...")
        hnsw = HNSWIndex(M=16, ef_construction=200, ef_search=100, metric="euclidean")
        build_start = time.time()
        hnsw.build(vectors)
        build_time = time.time() - build_start

        print(f"Build time: {build_time:.2f}s")
        print()

        # Search benchmark
        print("Running search benchmark...")
        predictions = np.zeros((len(queries), k), dtype=np.int32)

        search_start = time.time()
        for i, query in enumerate(queries):
            results = hnsw.search(query, k=k)
            predictions[i] = [vid for vid, _ in results][:k]
        search_time = time.time() - search_start

        qps = len(queries) / search_time
        avg_latency = search_time / len(queries) * 1000  # ms

        # Calculate recall
        # Note: Pre-computed ground truth is for full dataset, not subset
        # So we recompute ground truth for our subset using brute force
        print("Computing ground truth for subset...")
        from .datasets import compute_ground_truth
        ground_truth_subset = compute_ground_truth(vectors, queries, k=k, metric="euclidean")

        recall = calculate_recall(predictions, ground_truth_subset)
        print(f"Recall@{k}: {recall:.2%}")

        print(f"QPS: {qps:.1f}")
        print(f"Avg latency: {avg_latency:.2f}ms")
        print(f"{'='*60}\n")

        # Assert reasonable performance
        assert qps > 100, f"QPS too low: {qps:.1f}"
        assert recall > 0.85, f"Recall too low: {recall:.2%}"

    @pytest.mark.benchmark
    @pytest.mark.real_data
    @pytest.mark.slow
    def test_sift1m_medium(self):
        """
        SIFT1M benchmark (100k subset).

        Tests performance at larger scale.
        This is closer to real production workloads.
        """
        try:
            vectors, queries, ground_truth = load_dataset(
                "sift1m", subset="medium", download_if_missing=False
            )
        except FileNotFoundError:
            pytest.skip("SIFT1M not downloaded. Run: download_dataset('sift1m')")

        k = 10

        print(f"\n{'='*60}")
        print(f"SIFT1M Benchmark (Medium - 100k vectors)")
        print(f"{'='*60}")
        print(f"Dataset: {len(vectors):,} vectors")
        print(f"Queries: {len(queries):,}")
        print()

        # Build HNSW
        print("Building HNSW index...")
        hnsw = HNSWIndex(M=16, ef_construction=200, ef_search=100, metric="euclidean")
        build_start = time.time()
        hnsw.build(vectors)
        build_time = time.time() - build_start

        print(f"Build time: {build_time:.2f}s ({build_time/60:.1f} min)")
        print(f"Build rate: {len(vectors)/build_time:.0f} vectors/sec")
        print()

        # Search benchmark (use subset of queries for speed)
        n_test_queries = min(100, len(queries))
        test_queries = queries[:n_test_queries]
        predictions = np.zeros((n_test_queries, k), dtype=np.int32)

        print(f"Running search benchmark ({n_test_queries} queries)...")
        search_start = time.time()
        for i, query in enumerate(test_queries):
            results = hnsw.search(query, k=k)
            predictions[i] = [vid for vid, _ in results][:k]
        search_time = time.time() - search_start

        qps = n_test_queries / search_time
        avg_latency = search_time / n_test_queries * 1000

        # Calculate recall (recompute ground truth for subset)
        print("Computing ground truth for subset...")
        from .datasets import compute_ground_truth
        ground_truth_subset = compute_ground_truth(vectors, test_queries, k=k, metric="euclidean")
        recall = calculate_recall(predictions, ground_truth_subset)
        print(f"Recall@{k}: {recall:.2%}")

        print(f"QPS: {qps:.1f}")
        print(f"Avg latency: {avg_latency:.2f}ms")
        print(f"{'='*60}\n")

    @pytest.mark.benchmark
    @pytest.mark.real_data
    @pytest.mark.slow
    def test_sift1m_full(self):
        """
        SIFT1M benchmark (FULL 1M vectors).

        This is the real deal - full SIFT1M dataset.
        Shows production-scale performance.
        Build time: ~5-15 minutes depending on CPU.
        """
        try:
            vectors, queries, ground_truth = load_dataset(
                "sift1m", subset="full", download_if_missing=False
            )
        except FileNotFoundError:
            pytest.skip("SIFT1M not downloaded. Run: download_dataset('sift1m')")

        k = 10

        print(f"\n{'='*60}")
        print(f"SIFT1M Benchmark (FULL - 1M vectors)")
        print(f"{'='*60}")
        print(f"Dataset: {len(vectors):,} vectors x {vectors.shape[1]}D")
        print(f"Queries: {len(queries):,}")
        print(f"k: {k}")
        print()

        # Build HNSW (use optimized build method)
        print("Building HNSW index (this will take ~8-12 minutes on M4 Pro)...")
        hnsw = HNSWIndex(M=16, ef_construction=200, ef_search=100, metric="euclidean")
        build_start = time.time()

        # Use the optimized build() method
        hnsw.build(vectors)

        build_time = time.time() - build_start

        print(f"Build time: {build_time:.2f}s ({build_time/60:.1f} min)")
        print(f"Build rate: {len(vectors)/build_time:.0f} vectors/sec")
        print()

        # Search benchmark (use subset of queries for reasonable test time)
        n_test_queries = 100
        test_queries = queries[:n_test_queries]
        predictions = np.zeros((n_test_queries, k), dtype=np.int32)

        print(f"Running search benchmark ({n_test_queries} queries)...")
        search_start = time.time()
        for i, query in enumerate(test_queries):
            results = hnsw.search(query, k=k)
            predictions[i] = [vid for vid, _ in results][:k]
        search_time = time.time() - search_start

        qps = n_test_queries / search_time
        avg_latency = search_time / n_test_queries * 1000

        # Calculate recall using pre-computed ground truth
        # (This time it's correct because we used the FULL dataset!)
        if ground_truth is not None:
            recall = calculate_recall(predictions, ground_truth[:n_test_queries, :k])
            print(f"Recall@{k}: {recall:.2%}")
        else:
            print("Ground truth not available")

        print(f"QPS: {qps:.1f}")
        print(f"Avg latency: {avg_latency:.2f}ms")
        print()
        print(f"Index Statistics:")
        print(f"  Total vectors: {len(vectors):,}")
        print(f"  Memory (approx): {len(vectors) * vectors.shape[1] * 4 / 1024 / 1024:.1f} MB")
        print(f"{'='*60}\n")

    @pytest.mark.benchmark
    @pytest.mark.real_data
    def test_sift1m_parameter_sweep(self):
        """
        Sweep ef_search parameter on SIFT1M.

        Shows recall vs speed tradeoff on real data.
        These results can be compared to published papers.
        """
        try:
            vectors, queries, ground_truth = load_dataset(
                "sift1m", subset="small", download_if_missing=False
            )
        except FileNotFoundError:
            pytest.skip("SIFT1M not downloaded")

        k = 10

        print(f"\n{'='*60}")
        print(f"SIFT1M Parameter Sweep (ef_search)")
        print(f"{'='*60}")
        print(f"Dataset: {len(vectors):,} vectors")
        print()

        # Build index once
        print("Building HNSW index...")
        hnsw = HNSWIndex(M=16, ef_construction=200, metric="euclidean")
        hnsw.build(vectors)

        # Compute ground truth for subset
        n_test_queries = min(100, len(queries))
        test_queries = queries[:n_test_queries]

        print("Computing ground truth for subset...")
        from .datasets import compute_ground_truth
        ground_truth_subset = compute_ground_truth(vectors, test_queries, k=k, metric="euclidean")

        # Test different ef_search values
        ef_values = [10, 20, 50, 100, 200, 400]

        print()
        print(f"{'ef_search':<12} {'Recall@10':<12} {'QPS':<10} {'Latency (ms)':<15}")
        print(f"{'-'*55}")

        for ef in ef_values:
            predictions = np.zeros((n_test_queries, k), dtype=np.int32)

            start = time.time()
            for i, query in enumerate(test_queries):
                results = hnsw.search(query, k=k, ef=ef)
                predictions[i] = [vid for vid, _ in results][:k]
            elapsed = time.time() - start

            recall = calculate_recall(predictions, ground_truth_subset)
            qps = n_test_queries / elapsed
            latency_ms = (elapsed / n_test_queries) * 1000

            print(f"{ef:<12} {recall:<12.2%} {qps:<10.1f} {latency_ms:<15.2f}")

        print(f"{'='*60}\n")

    @pytest.mark.benchmark
    @pytest.mark.real_data
    def test_dimensions_comparison(self):
        """
        Compare performance across different vector dimensions.

        Tests how HNSW scales with dimensionality.
        Higher dimensions typically mean slower search.
        """
        # We'll use synthetic data with different dimensions
        # since not all real datasets may be downloaded
        from .datasets import get_benchmark_dataset

        dimensions = [64, 128, 256, 512, 768, 1536]
        n_vectors = 10_000
        n_queries = 50
        k = 10

        print(f"\n{'='*60}")
        print(f"Dimensionality Impact on Performance")
        print(f"{'='*60}")
        print(f"Dataset: {n_vectors:,} vectors")
        print(f"Queries: {n_queries}")
        print()

        print(f"{'Dimension':<12} {'Build (s)':<12} {'QPS':<10} {'Latency (ms)':<15}")
        print(f"{'-'*55}")

        for dim in dimensions:
            vectors, queries = get_benchmark_dataset(
                "random", "small", dimension=dim, n_queries=n_queries
            )
            vectors = vectors[:n_vectors]

            # Build index
            hnsw = HNSWIndex(M=16, ef_construction=200, ef_search=50, metric="euclidean")
            build_start = time.time()
            hnsw.build(vectors)
            build_time = time.time() - build_start

            # Search
            search_start = time.time()
            for query in queries:
                hnsw.search(query, k=k)
            search_time = time.time() - search_start

            qps = n_queries / search_time
            latency_ms = (search_time / n_queries) * 1000

            print(f"{dim:<12} {build_time:<12.2f} {qps:<10.1f} {latency_ms:<15.2f}")

        print(f"{'='*60}\n")


def test_list_datasets():
    """Helper test to show available datasets."""
    list_available_datasets()


if __name__ == "__main__":
    # Quick test
    list_available_datasets()
