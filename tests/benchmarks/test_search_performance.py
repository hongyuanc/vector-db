"""
Search performance benchmarks.

Compares search latency and throughput between different index types.
Demonstrates the performance advantage of HNSW over brute-force search.
"""

import pytest
import numpy as np
import time
from src.index.hnsw import HNSWIndex
from src.index.brute_force import BruteForceIndex
from .datasets import get_benchmark_dataset


class TestSearchPerformance:
    """Benchmark search performance across different indices."""

    @pytest.mark.benchmark
    def test_search_latency_small(self):
        """
        Compare search latency on small dataset (10k vectors).

        Measures average query time per search operation.
        HNSW should be significantly faster than brute force.
        """
        # Generate dataset
        vectors, queries = get_benchmark_dataset("random", "small", dimension=128)
        k = 10

        print(f"\n{'='*60}")
        print(f"Search Latency Benchmark (Small Dataset)")
        print(f"{'='*60}")
        print(f"Dataset: {len(vectors):,} vectors x {vectors.shape[1]} dimensions")
        print(f"Queries: {len(queries):,}")
        print(f"k: {k}")
        print()

        # Build brute force index
        bf_index = BruteForceIndex(metric="euclidean")
        bf_start = time.time()
        bf_index.build(vectors)
        bf_build_time = time.time() - bf_start

        # Build HNSW index
        hnsw_index = HNSWIndex(M=16, ef_construction=200, ef_search=50, metric="euclidean")
        hnsw_start = time.time()
        hnsw_index.build(vectors)
        hnsw_build_time = time.time() - hnsw_start

        print(f"Build Times:")
        print(f"  Brute Force: {bf_build_time:.3f}s")
        print(f"  HNSW:        {hnsw_build_time:.3f}s")
        print()

        # Benchmark brute force search
        bf_start = time.time()
        for query in queries:
            bf_index.search(query, k=k)
        bf_total_time = time.time() - bf_start
        bf_avg_time = bf_total_time / len(queries)
        bf_qps = len(queries) / bf_total_time

        # Benchmark HNSW search
        hnsw_start = time.time()
        for query in queries:
            hnsw_index.search(query, k=k)
        hnsw_total_time = time.time() - hnsw_start
        hnsw_avg_time = hnsw_total_time / len(queries)
        hnsw_qps = len(queries) / hnsw_total_time

        # Calculate speedup
        speedup = bf_total_time / hnsw_total_time

        print(f"Search Performance:")
        print(f"  Brute Force:")
        print(f"    Total:    {bf_total_time:.3f}s")
        print(f"    Avg/query: {bf_avg_time*1000:.2f}ms")
        print(f"    QPS:      {bf_qps:.1f}")
        print()
        print(f"  HNSW:")
        print(f"    Total:    {hnsw_total_time:.3f}s")
        print(f"    Avg/query: {hnsw_avg_time*1000:.2f}ms")
        print(f"    QPS:      {hnsw_qps:.1f}")
        print()
        print(f"  Speedup:   {speedup:.1f}x faster")
        print(f"{'='*60}\n")

        # Assert HNSW is faster
        assert speedup > 3.0, f"HNSW should be at least 3x faster, got {speedup:.1f}x"

    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_search_latency_medium(self):
        """
        Compare search latency on medium dataset (100k vectors).

        At larger scales, HNSW's advantage becomes even more pronounced.
        Brute force is O(n), HNSW is O(log n).
        """
        # Generate dataset
        vectors, queries = get_benchmark_dataset("random", "medium", dimension=128, n_queries=50)
        k = 10

        print(f"\n{'='*60}")
        print(f"Search Latency Benchmark (Medium Dataset)")
        print(f"{'='*60}")
        print(f"Dataset: {len(vectors):,} vectors x {vectors.shape[1]} dimensions")
        print(f"Queries: {len(queries):,}")
        print(f"k: {k}")
        print()

        # Build indices
        bf_index = BruteForceIndex(metric="euclidean")
        bf_start = time.time()
        bf_index.build(vectors)
        bf_build_time = time.time() - bf_start

        hnsw_index = HNSWIndex(M=16, ef_construction=200, ef_search=50, metric="euclidean")
        hnsw_start = time.time()
        hnsw_index.build(vectors)
        hnsw_build_time = time.time() - hnsw_start

        print(f"Build Times:")
        print(f"  Brute Force: {bf_build_time:.3f}s")
        print(f"  HNSW:        {hnsw_build_time:.3f}s")
        print()

        # Benchmark searches
        bf_start = time.time()
        for query in queries:
            bf_index.search(query, k=k)
        bf_total_time = time.time() - bf_start
        bf_avg_time = bf_total_time / len(queries)
        bf_qps = len(queries) / bf_total_time

        hnsw_start = time.time()
        for query in queries:
            hnsw_index.search(query, k=k)
        hnsw_total_time = time.time() - hnsw_start
        hnsw_avg_time = hnsw_total_time / len(queries)
        hnsw_qps = len(queries) / hnsw_total_time

        speedup = bf_total_time / hnsw_total_time

        print(f"Search Performance:")
        print(f"  Brute Force:")
        print(f"    Total:     {bf_total_time:.3f}s")
        print(f"    Avg/query: {bf_avg_time*1000:.2f}ms")
        print(f"    QPS:       {bf_qps:.1f}")
        print()
        print(f"  HNSW:")
        print(f"    Total:     {hnsw_total_time:.3f}s")
        print(f"    Avg/query: {hnsw_avg_time*1000:.2f}ms")
        print(f"    QPS:       {hnsw_qps:.1f}")
        print()
        print(f"  Speedup:    {speedup:.1f}x faster")
        print(f"{'='*60}\n")

        # At 100k scale, speedup should be even larger
        assert speedup > 10.0, f"HNSW should be at least 10x faster at 100k scale, got {speedup:.1f}x"

    @pytest.mark.benchmark
    def test_throughput_comparison(self):
        """
        Compare query throughput (QPS) between indices.

        Measures how many queries per second each index can handle.
        Important metric for production systems.
        """
        vectors, queries = get_benchmark_dataset("random", "small", dimension=128)
        k = 10

        print(f"\n{'='*60}")
        print(f"Throughput Benchmark (Queries Per Second)")
        print(f"{'='*60}")
        print(f"Dataset: {len(vectors):,} vectors x {vectors.shape[1]} dimensions")
        print(f"Queries: {len(queries):,}")
        print()

        # Build indices
        bf_index = BruteForceIndex(metric="euclidean")
        bf_index.build(vectors)

        hnsw_index = HNSWIndex(M=16, ef_construction=200, ef_search=50, metric="euclidean")
        hnsw_index.build(vectors)

        # Measure throughput
        bf_start = time.time()
        for query in queries:
            bf_index.search(query, k=k)
        bf_time = time.time() - bf_start
        bf_qps = len(queries) / bf_time

        hnsw_start = time.time()
        for query in queries:
            hnsw_index.search(query, k=k)
        hnsw_time = time.time() - hnsw_start
        hnsw_qps = len(queries) / hnsw_time

        print(f"Throughput (QPS):")
        print(f"  Brute Force: {bf_qps:>8.1f} queries/sec")
        print(f"  HNSW:        {hnsw_qps:>8.1f} queries/sec")
        print(f"  Improvement: {hnsw_qps/bf_qps:>8.1f}x")
        print(f"{'='*60}\n")

        assert hnsw_qps > bf_qps, "HNSW should have higher throughput"

    @pytest.mark.benchmark
    def test_ef_search_impact(self):
        """
        Measure how ef_search parameter affects search speed.

        Lower ef_search = faster but lower recall
        Higher ef_search = slower but higher recall

        This demonstrates the speed/accuracy tradeoff.
        """
        vectors, queries = get_benchmark_dataset("random", "small", dimension=128, n_queries=50)
        k = 10

        print(f"\n{'='*60}")
        print(f"ef_search Parameter Impact on Speed")
        print(f"{'='*60}")
        print(f"Dataset: {len(vectors):,} vectors")
        print(f"Testing ef_search values: [10, 20, 50, 100, 200]")
        print()

        # Build index once
        hnsw_index = HNSWIndex(M=16, ef_construction=200, metric="euclidean")
        hnsw_index.build(vectors)

        ef_values = [10, 20, 50, 100, 200]
        results = []

        print(f"{'ef_search':<12} {'Avg Time (ms)':<15} {'QPS':<10}")
        print(f"{'-'*40}")

        for ef in ef_values:
            start = time.time()
            for query in queries:
                hnsw_index.search(query, k=k, ef=ef)
            total_time = time.time() - start

            avg_time_ms = (total_time / len(queries)) * 1000
            qps = len(queries) / total_time

            print(f"{ef:<12} {avg_time_ms:<15.2f} {qps:<10.1f}")
            results.append((ef, avg_time_ms, qps))

        print(f"{'='*60}\n")

        # Verify that higher ef_search is slower
        assert results[0][1] < results[-1][1], "Higher ef_search should be slower"
