"""
ChromaDB vs Our HNSW Implementation - Head-to-Head Comparison.

Benchmarks both systems on identical SIFT1M datasets to provide
fair performance comparison.

FAIRNESS CRITERIA:
1. In-Memory Mode: ChromaDB uses EphemeralClient (no disk I/O) to match our in-memory implementation
2. Batched Queries: All queries sent in batch to remove Python loop overhead
3. Identical Parameters: Both use M=16, ef_construction=200, ef_search=100
4. Same Dataset & Metric: SIFT1M with L2 (Euclidean) distance

This compares HNSW algorithm performance, not database features.
"""

import pytest
import numpy as np
import time

try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

from src.index.hnsw import HNSWIndex
from .real_datasets import load_dataset
from .test_recall_accuracy import calculate_recall
from .datasets import compute_ground_truth


@pytest.mark.skipif(not CHROMADB_AVAILABLE, reason="ChromaDB not installed")
class TestChromaDBComparison:
    """Compare our HNSW implementation with ChromaDB."""

    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_chromadb_vs_ours_10k(self):
        """
        Head-to-head comparison on SIFT1M 10k subset.

        Tests both systems with identical data and parameters
        to provide fair performance comparison.
        """
        try:
            vectors, queries, ground_truth = load_dataset(
                "sift1m", subset="small", download_if_missing=False
            )
        except FileNotFoundError:
            pytest.skip("SIFT1M not downloaded")

        k = 10
        n_test_queries = min(100, len(queries))
        test_queries = queries[:n_test_queries]

        print(f"\n{'='*70}")
        print(f"ChromaDB vs Our HNSW - Head-to-Head Comparison (10k vectors)")
        print(f"{'='*70}")
        print(f"Dataset: {len(vectors):,} vectors × {vectors.shape[1]}D")
        print(f"Test queries: {n_test_queries}")
        print(f"k: {k}")
        print()

        # Compute ground truth for subset
        print("Computing ground truth...")
        ground_truth_subset = compute_ground_truth(vectors, test_queries, k=k, metric="euclidean")
        print()

        # ========== Our HNSW Implementation ==========
        print(f"{'='*70}")
        print("Testing: Our HNSW Implementation")
        print(f"{'='*70}")

        hnsw = HNSWIndex(M=16, ef_construction=200, ef_search=100, metric="euclidean")

        build_start = time.time()
        hnsw.build(vectors)
        our_build_time = time.time() - build_start

        print(f"Build time: {our_build_time:.2f}s")

        # Search benchmark
        our_predictions = np.zeros((n_test_queries, k), dtype=np.int32)
        search_start = time.time()
        for i, query in enumerate(test_queries):
            results = hnsw.search(query, k=k)
            our_predictions[i] = [vid for vid, _ in results][:k]
        our_search_time = time.time() - search_start

        our_qps = n_test_queries / our_search_time
        our_latency = our_search_time / n_test_queries * 1000
        our_recall = calculate_recall(our_predictions, ground_truth_subset)

        print(f"Recall@{k}: {our_recall:.2%}")
        print(f"QPS: {our_qps:.1f}")
        print(f"Avg latency: {our_latency:.2f}ms")
        print()

        # ========== ChromaDB ==========
        print(f"{'='*70}")
        print("Testing: ChromaDB (In-Memory Ephemeral Mode)")
        print(f"{'='*70}")

        # Use ephemeral client (in-memory, no disk I/O) for fair comparison
        client = chromadb.EphemeralClient()

        collection = client.create_collection(
            name="sift1m_benchmark",
            metadata={
                "hnsw:space": "l2",  # L2 distance (Euclidean)
                "hnsw:construction_ef": 200,
                "hnsw:search_ef": 100,
                "hnsw:M": 16,
            }
        )

        # Build index
        build_start = time.time()

        # ChromaDB requires string IDs
        ids = [str(i) for i in range(len(vectors))]
        embeddings = vectors.tolist()

        # Batch insert for efficiency
        batch_size = 5000
        for i in range(0, len(vectors), batch_size):
            end_idx = min(i + batch_size, len(vectors))
            collection.add(
                ids=ids[i:end_idx],
                embeddings=embeddings[i:end_idx]
            )

        chroma_build_time = time.time() - build_start
        print(f"Build time: {chroma_build_time:.2f}s")

        # Search benchmark - BATCHED to remove Python loop overhead
        search_start = time.time()

        # Batch all queries at once (fair comparison)
        results = collection.query(
            query_embeddings=test_queries.tolist(),
            n_results=k
        )

        chroma_search_time = time.time() - search_start

        # Convert results to predictions array
        chroma_predictions = np.zeros((n_test_queries, k), dtype=np.int32)
        for i in range(n_test_queries):
            chroma_predictions[i] = [int(id_str) for id_str in results['ids'][i]]

        chroma_qps = n_test_queries / chroma_search_time
        chroma_latency = chroma_search_time / n_test_queries * 1000
        chroma_recall = calculate_recall(chroma_predictions, ground_truth_subset)

        print(f"Recall@{k}: {chroma_recall:.2%}")
        print(f"QPS: {chroma_qps:.1f}")
        print(f"Avg latency: {chroma_latency:.2f}ms")
        print()

        # Also test single-query latency for fairness
        print("Single-query latency test (10 queries):")
        single_query_times = []
        for i in range(10):
            start = time.time()
            collection.query(
                query_embeddings=[test_queries[i].tolist()],
                n_results=k
            )
            single_query_times.append((time.time() - start) * 1000)

        avg_single = np.mean(single_query_times)
        print(f"  Avg single-query latency: {avg_single:.2f}ms")
        print()

        # ========== Comparison Summary ==========
        print(f"{'='*70}")
        print("Head-to-Head Comparison Summary")
        print(f"{'='*70}")
        print()
        print(f"{'Metric':<20} {'Our HNSW':<15} {'ChromaDB':<15} {'Winner':<15}")
        print(f"{'-'*70}")
        print(f"{'Build Time (s)':<20} {our_build_time:<15.2f} {chroma_build_time:<15.2f} {'Ours' if our_build_time < chroma_build_time else 'ChromaDB':<15}")
        print(f"{'Recall@10':<20} {our_recall:<15.2%} {chroma_recall:<15.2%} {'Ours' if our_recall > chroma_recall else 'ChromaDB':<15}")
        print(f"{'QPS':<20} {our_qps:<15.1f} {chroma_qps:<15.1f} {'Ours' if our_qps > chroma_qps else 'ChromaDB':<15}")
        print(f"{'Latency (ms)':<20} {our_latency:<15.2f} {chroma_latency:<15.2f} {'Ours' if our_latency < chroma_latency else 'ChromaDB':<15}")
        print()

        # Performance ratios
        build_ratio = our_build_time / chroma_build_time if chroma_build_time > 0 else 0
        qps_ratio = our_qps / chroma_qps if chroma_qps > 0 else 0
        latency_ratio = our_latency / chroma_latency if chroma_latency > 0 else 0

        print("Performance Ratios (Ours / ChromaDB):")
        print(f"  Build time: {build_ratio:.2f}x {'faster' if build_ratio < 1 else 'slower'}")
        print(f"  QPS: {qps_ratio:.2f}x {'faster' if qps_ratio > 1 else 'slower'}")
        print(f"  Latency: {latency_ratio:.2f}x {'faster' if latency_ratio < 1 else 'slower'}")
        print(f"{'='*70}\n")

    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_chromadb_vs_ours_100k(self):
        """
        Head-to-head comparison on SIFT1M 100k subset.

        Tests scalability comparison at medium scale.
        """
        try:
            vectors, queries, ground_truth = load_dataset(
                "sift1m", subset="medium", download_if_missing=False
            )
        except FileNotFoundError:
            pytest.skip("SIFT1M not downloaded")

        k = 10
        n_test_queries = 100
        test_queries = queries[:n_test_queries]

        print(f"\n{'='*70}")
        print(f"ChromaDB vs Our HNSW - Head-to-Head Comparison (100k vectors)")
        print(f"{'='*70}")
        print(f"Dataset: {len(vectors):,} vectors × {vectors.shape[1]}D")
        print(f"Test queries: {n_test_queries}")
        print()

        # Compute ground truth
        print("Computing ground truth...")
        ground_truth_subset = compute_ground_truth(vectors, test_queries, k=k, metric="euclidean")
        print()

        # ========== Our HNSW Implementation ==========
        print("Testing: Our HNSW Implementation")
        hnsw = HNSWIndex(M=16, ef_construction=200, ef_search=100, metric="euclidean")

        build_start = time.time()
        hnsw.build(vectors)
        our_build_time = time.time() - build_start

        print(f"Build time: {our_build_time:.2f}s ({our_build_time/60:.1f} min)")

        our_predictions = np.zeros((n_test_queries, k), dtype=np.int32)
        search_start = time.time()
        for i, query in enumerate(test_queries):
            results = hnsw.search(query, k=k)
            our_predictions[i] = [vid for vid, _ in results][:k]
        our_search_time = time.time() - search_start

        our_qps = n_test_queries / our_search_time
        our_latency = our_search_time / n_test_queries * 1000
        our_recall = calculate_recall(our_predictions, ground_truth_subset)

        print(f"Recall@{k}: {our_recall:.2%}")
        print(f"QPS: {our_qps:.1f}")
        print(f"Avg latency: {our_latency:.2f}ms")
        print()

        # ========== ChromaDB ==========
        print("Testing: ChromaDB (In-Memory Ephemeral Mode)")

        # Use ephemeral client (in-memory, no disk I/O) for fair comparison
        client = chromadb.EphemeralClient()

        collection = client.create_collection(
            name="sift1m_benchmark",
            metadata={
                "hnsw:space": "l2",
                "hnsw:construction_ef": 200,
                "hnsw:search_ef": 100,
                "hnsw:M": 16,
            }
        )

        build_start = time.time()
        ids = [str(i) for i in range(len(vectors))]
        embeddings = vectors.tolist()

        batch_size = 5000
        for i in range(0, len(vectors), batch_size):
            end_idx = min(i + batch_size, len(vectors))
            collection.add(
                ids=ids[i:end_idx],
                embeddings=embeddings[i:end_idx]
            )

        chroma_build_time = time.time() - build_start
        print(f"Build time: {chroma_build_time:.2f}s ({chroma_build_time/60:.1f} min)")

        # Search benchmark - BATCHED to remove Python loop overhead
        search_start = time.time()

        results = collection.query(
            query_embeddings=test_queries.tolist(),
            n_results=k
        )

        chroma_search_time = time.time() - search_start

        chroma_predictions = np.zeros((n_test_queries, k), dtype=np.int32)
        for i in range(n_test_queries):
            chroma_predictions[i] = [int(id_str) for id_str in results['ids'][i]]

        chroma_qps = n_test_queries / chroma_search_time
        chroma_latency = chroma_search_time / n_test_queries * 1000
        chroma_recall = calculate_recall(chroma_predictions, ground_truth_subset)

        print(f"Recall@{k}: {chroma_recall:.2%}")
        print(f"QPS: {chroma_qps:.1f}")
        print(f"Avg latency: {chroma_latency:.2f}ms")
        print()

        # Summary
        print(f"{'='*70}")
        print("Summary (100k vectors)")
        print(f"{'='*70}")
        print(f"{'Metric':<20} {'Our HNSW':<15} {'ChromaDB':<15} {'Ratio':<15}")
        print(f"{'-'*70}")
        print(f"{'Build Time (s)':<20} {our_build_time:<15.2f} {chroma_build_time:<15.2f} {our_build_time/chroma_build_time:<15.2f}x")
        print(f"{'Recall@10':<20} {our_recall:<15.2%} {chroma_recall:<15.2%} {our_recall/chroma_recall:<15.2f}x")
        print(f"{'QPS':<20} {our_qps:<15.1f} {chroma_qps:<15.1f} {our_qps/chroma_qps:<15.2f}x")
        print(f"{'Latency (ms)':<20} {our_latency:<15.2f} {chroma_latency:<15.2f} {our_latency/chroma_latency:<15.2f}x")
        print(f"{'='*70}\n")


if __name__ == "__main__":
    # Quick test
    pytest.main([__file__, "-v", "-s"])
