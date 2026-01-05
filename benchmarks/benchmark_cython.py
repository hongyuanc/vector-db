"""
Cython Performance Benchmark Script

Benchmarks the Cython-optimized HNSW implementation at different scales
to demonstrate performance improvements over pure Python.

Usage:
    python benchmarks/benchmark_cython.py

Requirements:
    - Cython extensions must be built: python setup.py build_ext --inplace
    - numpy, cython installed

Methodology Notes:
    - Queries are generated separately from indexed vectors (avoids self-query bias)
    - Garbage collection forced before timing measurements
    - Warmup phase to stabilize cache/JIT
    - 1M vector test requires Cython (would take hours in pure Python)
"""

import numpy as np
import time
import sys
import gc
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.index.hnsw import HNSWIndex, CYTHON_AVAILABLE


def run_benchmark():
    """Run comprehensive benchmark across multiple scales."""

    print('=' * 80)
    print('VECTOR DATABASE PERFORMANCE BENCHMARK')
    print('=' * 80)
    print(f'Cython optimizations: {"ENABLED ✓" if CYTHON_AVAILABLE else "DISABLED ✗"}')
    if not CYTHON_AVAILABLE:
        print('WARNING: Cython not available. Run: python setup.py build_ext --inplace')
    print()

    # Test configurations
    test_configs = [
        {'n_vectors': 10_000, 'name': '10k', 'n_queries': 1000},
        {'n_vectors': 100_000, 'name': '100k', 'n_queries': 1000},
        {'n_vectors': 1_000_000, 'name': '1M', 'n_queries': 1000},
    ]

    # HNSW parameters
    dimension = 128
    M = 16
    ef_construction = 200
    ef_search = 100

    print(f'Configuration:')
    print(f'  Dimension: {dimension}')
    print(f'  M: {M}')
    print(f'  ef_construction: {ef_construction}')
    print(f'  ef_search: {ef_search}')
    print(f'  Metric: Euclidean (L2)')
    print()
    print('=' * 80)
    print()

    results = []

    for config in test_configs:
        n_vectors = config['n_vectors']
        name = config['name']
        n_queries = config['n_queries']

        # Safety check: Skip 1M test without Cython (would take hours)
        if n_vectors >= 1_000_000 and not CYTHON_AVAILABLE:
            print(f'Skipping {name} test (Cython required for reasonable build time)')
            print(f'  Without Cython: ~2-10 hours build time')
            print(f'  With Cython: ~2-5 minutes build time')
            print()
            continue

        print(f'Testing {name} vectors ({n_vectors:,} x {dimension}D)')
        print('-' * 80)

        # Generate dataset
        np.random.seed(42)
        vectors = np.random.randn(n_vectors, dimension).astype(np.float32)

        # Generate SEPARATE query vectors (avoids self-query bias)
        np.random.seed(123)  # Different seed for queries
        queries = np.random.randn(n_queries, dimension).astype(np.float32)

        # Build index
        print(f'Building index...')
        hnsw = HNSWIndex(
            M=M,
            ef_construction=ef_construction,
            ef_search=ef_search,
            metric='euclidean'
        )

        build_start = time.time()
        hnsw.build(vectors)
        build_time = time.time() - build_start

        vectors_per_sec = n_vectors / build_time
        build_time_str = f'{build_time:.2f}s'
        if build_time > 60:
            build_time_str = f'{build_time/60:.1f}min ({build_time:.1f}s)'

        print(f'Build completed: {build_time_str}')
        print(f'Throughput: {vectors_per_sec:.0f} vectors/sec')

        # Search benchmark with proper methodology
        print(f'\nPreparing search benchmark...')

        # Force garbage collection before timing
        gc.collect()

        # Warmup phase: stabilize cache/JIT (100 queries)
        print(f'Warming up (100 queries)...')
        warmup_queries = min(100, n_queries)
        for i in range(warmup_queries):
            _ = hnsw.search(queries[i], k=10)

        # Force GC again after warmup
        gc.collect()

        # Timed benchmark
        print(f'Searching {n_queries} queries (k=10)...')
        search_start = time.time()

        for query in queries:
            _ = hnsw.search(query, k=10)

        search_time = time.time() - search_start

        qps = n_queries / search_time
        latency_ms = (search_time / n_queries) * 1000

        print(f'QPS: {qps:,.1f}')
        print(f'Avg latency: {latency_ms:.2f}ms')

        # Calculate recall using ground truth (proper methodology)
        print(f'\nCalculating recall@10 (ground truth comparison)...')
        recall_queries = min(100, n_queries)

        # Compute ground truth with brute force
        print(f'  Computing ground truth for {recall_queries} queries...')
        ground_truth_results = []
        for i in range(recall_queries):
            # Brute force: compute distance to all vectors
            distances = np.linalg.norm(vectors - queries[i], axis=1)
            # Get indices of 10 nearest
            nearest_indices = np.argsort(distances)[:10]
            ground_truth_results.append(set(nearest_indices))

        # Compare HNSW results to ground truth
        print(f'  Comparing HNSW results to ground truth...')
        total_correct = 0
        total_possible = recall_queries * 10

        for i in range(recall_queries):
            search_results = hnsw.search(queries[i], k=10)
            hnsw_ids = set([vid for vid, _ in search_results])
            correct = len(hnsw_ids & ground_truth_results[i])
            total_correct += correct

        recall = total_correct / total_possible
        print(f'Recall@10: {recall:.2%} (true nearest neighbors)')

        # Memory estimate
        memory_mb = (n_vectors * dimension * 4) / (1024 * 1024)  # float32 = 4 bytes
        print(f'\nEstimated memory: ~{memory_mb:.0f}MB (vectors only)')

        results.append({
            'name': name,
            'n_vectors': n_vectors,
            'build_time': build_time,
            'vectors_per_sec': vectors_per_sec,
            'qps': qps,
            'latency_ms': latency_ms,
            'recall': recall,
            'memory_mb': memory_mb,
        })

        print()
        print('=' * 80)
        print()

    # Summary table
    print()
    print('PERFORMANCE SUMMARY')
    print('=' * 80)
    print(f"{'Scale':<10} {'Build Time':<15} {'Throughput':<18} {'QPS':<15} {'Latency':<12} {'Recall':<10}")
    print('-' * 80)

    for r in results:
        build_str = f"{r['build_time']:.1f}s"
        if r['build_time'] > 60:
            build_str = f"{r['build_time']/60:.1f}min"

        throughput_str = f"{r['vectors_per_sec']:.0f} vec/s"
        qps_str = f"{r['qps']:,.0f}"
        latency_str = f"{r['latency_ms']:.2f}ms"
        recall_str = f"{r['recall']:.1%}"

        print(f"{r['name']:<10} {build_str:<15} {throughput_str:<18} {qps_str:<15} {latency_str:<12} {recall_str:<10}")

    print('=' * 80)
    print()

    # Scalability analysis
    print('SCALABILITY ANALYSIS')
    print('=' * 80)

    if len(results) >= 2:
        # Compare 10k vs 100k
        ratio_10k_100k = results[0]['latency_ms'] / results[1]['latency_ms'] if results[1]['latency_ms'] > 0 else 0
        print(f"10k → 100k (10x vectors):")
        print(f"  Latency increase: {results[1]['latency_ms'] - results[0]['latency_ms']:.2f}ms")
        print(f"  ({results[1]['latency_ms'] / results[0]['latency_ms']:.2f}x slower)")
        print()

    if len(results) >= 3:
        # Compare 100k vs 1M
        print(f"100k → 1M (10x vectors):")
        print(f"  Latency increase: {results[2]['latency_ms'] - results[1]['latency_ms']:.2f}ms")
        print(f"  ({results[2]['latency_ms'] / results[1]['latency_ms']:.2f}x slower)")
        print()

        # Compare 10k vs 1M
        print(f"10k → 1M (100x vectors):")
        print(f"  Latency increase: {results[2]['latency_ms'] - results[0]['latency_ms']:.2f}ms")
        print(f"  ({results[2]['latency_ms'] / results[0]['latency_ms']:.2f}x slower)")
        print()

    return results


if __name__ == '__main__':
    try:
        results = run_benchmark()
        print('\nBenchmark completed successfully!')
        sys.exit(0)
    except KeyboardInterrupt:
        print('\n\nBenchmark interrupted by user.')
        sys.exit(1)
    except Exception as e:
        print(f'\n\nBenchmark failed with error: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
