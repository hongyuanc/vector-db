"""
Reproducible benchmark runner for the vector database.

This script consolidates the ad hoc benchmark work into one command that records:
- dataset and HNSW configuration
- git commit and runtime environment
- build time and insert throughput
- search QPS and p50/p95/p99 latency
- recall@k against brute-force ground truth
- memory estimates
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_hnsw_module = importlib.import_module("src.index.hnsw")
CYTHON_AVAILABLE = _hnsw_module.CYTHON_AVAILABLE
CPP_AVAILABLE = _hnsw_module.CPP_AVAILABLE
HNSWIndex = _hnsw_module.HNSWIndex


BYTES_PER_MIB = 1024 * 1024


def _generate_random_dataset(
    n_vectors: int,
    dimension: int,
    n_queries: int,
    seed: int,
    distribution: str,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    if distribution == "normal":
        vectors = rng.standard_normal((n_vectors, dimension), dtype=np.float32)
        queries = rng.standard_normal((n_queries, dimension), dtype=np.float32)
    elif distribution == "uniform":
        vectors = rng.random((n_vectors, dimension), dtype=np.float32)
        queries = rng.random((n_queries, dimension), dtype=np.float32)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    return vectors, queries


def _generate_clustered_dataset(
    n_vectors: int,
    dimension: int,
    n_queries: int,
    seed: int,
    n_clusters: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_clusters, dimension), dtype=np.float32) * 3.0

    vector_clusters = rng.integers(0, n_clusters, size=n_vectors)
    vectors = centers[vector_clusters] + (
        rng.standard_normal((n_vectors, dimension), dtype=np.float32) * 0.5
    )

    query_clusters = rng.integers(0, n_clusters, size=n_queries)
    queries = centers[query_clusters] + (
        rng.standard_normal((n_queries, dimension), dtype=np.float32) * 0.5
    )

    return vectors.astype(np.float32), queries.astype(np.float32)


def _subset_for_size(size: int) -> str:
    if size <= 1_000:
        return "tiny"
    if size <= 10_000:
        return "small"
    if size <= 100_000:
        return "medium"
    return "full"


def load_dataset(
    name: str,
    size: int,
    dimension: int = 128,
    n_queries: int = 100,
    seed: int = 42,
    distribution: str = "normal",
    subset: str | None = None,
    download_if_missing: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load or generate benchmark data.

    Synthetic datasets are the default because they are deterministic and require
    no downloads. Real datasets reuse tests/benchmarks/real_datasets.py when the
    data is already present.
    """
    if name == "random":
        return _generate_random_dataset(size, dimension, n_queries, seed, distribution)

    if name == "clustered":
        return _generate_clustered_dataset(size, dimension, n_queries, seed)

    try:
        from tests.benchmarks.real_datasets import load_dataset as load_real_dataset
    except ImportError as exc:
        raise ValueError(
            f"Unknown dataset '{name}'. Use random, clustered, or a configured real dataset."
        ) from exc

    real_subset = subset or _subset_for_size(size)
    vectors, queries, _ground_truth = load_real_dataset(
        name,
        subset=real_subset,
        download_if_missing=download_if_missing,
    )

    return vectors[:size].astype(np.float32), queries[:n_queries].astype(np.float32)


def _compute_ground_truth(
    vectors: np.ndarray,
    queries: np.ndarray,
    k: int,
    metric: str,
) -> list[set[int]]:
    k = min(k, len(vectors))
    ground_truth: list[set[int]] = []

    if metric == "euclidean":
        for query in queries:
            distances = np.linalg.norm(vectors - query, axis=1)
            nearest = np.argpartition(distances, k - 1)[:k]
            ground_truth.append({int(idx) for idx in nearest})
        return ground_truth

    if metric == "cosine":
        vector_norms = np.linalg.norm(vectors, axis=1)
        vector_norms[vector_norms == 0.0] = 1.0

        for query in queries:
            query_norm = np.linalg.norm(query)
            if query_norm == 0.0:
                scores = np.zeros(len(vectors), dtype=np.float32)
            else:
                scores = vectors @ query / (vector_norms * query_norm)
            nearest = np.argpartition(-scores, k - 1)[:k]
            ground_truth.append({int(idx) for idx in nearest})
        return ground_truth

    raise ValueError(f"Unsupported metric: {metric}")


def _calculate_recall(predictions: list[list[int]], ground_truth: list[set[int]], k: int) -> float:
    if not predictions:
        return 0.0

    total_correct = 0
    total_possible = len(predictions) * min(k, len(ground_truth[0]))

    for predicted_ids, true_ids in zip(predictions, ground_truth, strict=True):
        total_correct += len(set(predicted_ids) & true_ids)

    return total_correct / total_possible if total_possible else 0.0


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "average": 0.0}

    samples = np.array(values, dtype=np.float64)
    return {
        "p50": round(float(np.percentile(samples, 50)), 4),
        "p95": round(float(np.percentile(samples, 95)), 4),
        "p99": round(float(np.percentile(samples, 99)), 4),
        "average": round(float(np.mean(samples)), 4),
    }


def _git_output(args: list[str]) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"

    if result.returncode != 0:
        return "unknown"

    return result.stdout.strip() or "unknown"


def _peak_rss_mb() -> float | None:
    try:
        import resource
    except ImportError:
        return None

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        bytes_used = usage
    else:
        bytes_used = usage * 1024
    return round(bytes_used / BYTES_PER_MIB, 2)


def _environment() -> dict[str, Any]:
    status = _git_output(["status", "--porcelain"])
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_output(["rev-parse", "--short", "HEAD"]),
        "git_dirty": status != "unknown" and bool(status),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "cython_available": CYTHON_AVAILABLE,
        "cpp_available": CPP_AVAILABLE,
    }


def run_benchmark_suite(
    dataset: str = "random",
    size: int = 10_000,
    dimension: int = 128,
    n_queries: int = 100,
    k: int = 10,
    metric: str = "euclidean",
    M: int = 16,
    ef_construction: int = 200,
    ef_search: int = 50,
    seed: int = 42,
    distribution: str = "normal",
    subset: str | None = None,
    warmup_queries: int = 10,
    download_if_missing: bool = False,
) -> dict[str, Any]:
    """Run one benchmark configuration and return structured results."""
    if metric not in {"euclidean", "cosine"}:
        raise ValueError("metric must be 'euclidean' or 'cosine'")
    if size <= 0:
        raise ValueError("size must be positive")
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    if n_queries <= 0:
        raise ValueError("n_queries must be positive")
    if k <= 0:
        raise ValueError("k must be positive")

    vectors, queries = load_dataset(
        dataset,
        size=size,
        dimension=dimension,
        n_queries=n_queries,
        seed=seed,
        distribution=distribution,
        subset=subset,
        download_if_missing=download_if_missing,
    )

    effective_k = min(k, len(vectors))
    np.random.seed(seed)

    rss_before = _peak_rss_mb()
    index = HNSWIndex(
        M=M,
        ef_construction=ef_construction,
        ef_search=ef_search,
        metric=metric,
    )

    build_start = time.perf_counter()
    index.build(vectors)
    build_time = time.perf_counter() - build_start

    for query in queries[:warmup_queries]:
        index.search(query, k=effective_k, ef=ef_search)

    latencies_ms: list[float] = []
    predictions: list[list[int]] = []

    search_start = time.perf_counter()
    for query in queries:
        query_start = time.perf_counter()
        results = index.search(query, k=effective_k, ef=ef_search)
        latencies_ms.append((time.perf_counter() - query_start) * 1000)
        predictions.append([int(vector_id) for vector_id, _distance in results[:effective_k]])
    search_time = time.perf_counter() - search_start

    ground_truth = _compute_ground_truth(vectors, queries, effective_k, metric)
    recall = _calculate_recall(predictions, ground_truth, effective_k)

    rss_after = _peak_rss_mb()
    memory: dict[str, float | None] = {
        "vector_data_mb": round(float(vectors.nbytes) / BYTES_PER_MIB, 2),
        "query_data_mb": round(float(queries.nbytes) / BYTES_PER_MIB, 2),
        "process_peak_rss_mb": rss_after,
    }
    if rss_before is not None and rss_after is not None:
        memory["process_peak_rss_delta_mb"] = round(max(0.0, rss_after - rss_before), 2)

    return {
        "schema_version": "benchmark.v1",
        "environment": _environment(),
        "dataset": {
            "name": dataset,
            "size": int(len(vectors)),
            "dimension": int(vectors.shape[1]),
            "n_queries": int(len(queries)),
            "seed": seed,
        },
        "config": {
            "hnsw": {
                "M": M,
                "ef_construction": ef_construction,
                "ef_search": ef_search,
                "metric": metric,
            },
            "k": effective_k,
            "warmup_queries": min(warmup_queries, len(queries)),
            "distribution": distribution if dataset == "random" else None,
            "subset": subset,
        },
        "metrics": {
            "build_time_seconds": round(build_time, 6),
            "vectors_per_second": round(len(vectors) / build_time, 2) if build_time else 0.0,
            "search_time_seconds": round(search_time, 6),
            "qps": round(len(queries) / search_time, 2) if search_time else 0.0,
            "latency_ms": _percentiles(latencies_ms),
            "recall_at_k": round(recall, 6),
            "memory": memory,
        },
    }


def format_markdown_report(result: dict[str, Any]) -> str:
    """Render a benchmark result as a compact Markdown report."""
    dataset = result["dataset"]
    config = result["config"]["hnsw"]
    metrics = result["metrics"]
    latency = metrics["latency_ms"]
    memory = metrics["memory"]
    k = result["config"]["k"]

    return "\n".join(
        [
            "# Benchmark Results",
            "",
            "## Run Context",
            "",
            f"- Commit: `{result['environment']['git_commit']}`",
            f"- Dirty worktree: `{result['environment']['git_dirty']}`",
            f"- Python: `{result['environment']['python']}`",
            f"- NumPy: `{result['environment']['numpy']}`",
            f"- Cython extensions available: `{result['environment']['cython_available']}`",
            f"- C++ HNSW extension available: `{result['environment']['cpp_available']}`",
            "",
            "## Dataset",
            "",
            f"- Name: `{dataset['name']}`",
            f"- Vectors: `{dataset['size']}`",
            f"- Dimension: `{dataset['dimension']}`",
            f"- Queries: `{dataset['n_queries']}`",
            f"- Seed: `{dataset['seed']}`",
            "",
            "## HNSW Configuration",
            "",
            f"- M: `{config['M']}`",
            f"- ef_construction: `{config['ef_construction']}`",
            f"- ef_search: `{config['ef_search']}`",
            f"- Metric: `{config['metric']}`",
            "",
            "## Metrics",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Build Time | {metrics['build_time_seconds']:.4f}s |",
            f"| Build Throughput | {metrics['vectors_per_second']:.2f} vectors/sec |",
            f"| QPS | {metrics['qps']:.2f} |",
            f"| Average Latency | {latency['average']:.4f} ms |",
            f"| p50 Latency | {latency['p50']:.4f} ms |",
            f"| p95 Latency | {latency['p95']:.4f} ms |",
            f"| p99 Latency | {latency['p99']:.4f} ms |",
            f"| Recall@{k} | {metrics['recall_at_k']:.4f} |",
            f"| Vector Data | {memory['vector_data_mb']:.2f} MiB |",
            f"| Process Peak RSS | {memory['process_peak_rss_mb']} MiB |",
            "",
        ]
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a reproducible vector DB benchmark")
    parser.add_argument("--dataset", default="random", help="Dataset: random, clustered, sift1m, etc.")
    parser.add_argument("--size", type=int, default=10_000, help="Number of vectors")
    parser.add_argument("--subset", default=None, help="Real dataset subset: tiny, small, medium, full")
    parser.add_argument("--dimension", type=int, default=128, help="Synthetic vector dimension")
    parser.add_argument("--queries", type=int, default=100, help="Number of query vectors")
    parser.add_argument("--k", type=int, default=10, help="Number of nearest neighbors")
    parser.add_argument("--metric", choices=["euclidean", "cosine"], default="euclidean")
    parser.add_argument("--M", type=int, default=16, help="HNSW max connections")
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--ef-search", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--distribution", choices=["normal", "uniform"], default="normal")
    parser.add_argument("--warmup-queries", type=int, default=10)
    parser.add_argument("--download-if-missing", action="store_true")
    parser.add_argument("--output", default="results.json", help="JSON output path")
    parser.add_argument("--markdown-output", default=None, help="Optional Markdown output path")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    result = run_benchmark_suite(
        dataset=args.dataset,
        size=args.size,
        dimension=args.dimension,
        n_queries=args.queries,
        k=args.k,
        metric=args.metric,
        M=args.M,
        ef_construction=args.ef_construction,
        ef_search=args.ef_search,
        seed=args.seed,
        distribution=args.distribution,
        subset=args.subset,
        warmup_queries=args.warmup_queries,
        download_if_missing=args.download_if_missing,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")

    if args.markdown_output:
        markdown_path = Path(args.markdown_output)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(format_markdown_report(result))

    metrics = result["metrics"]
    print(f"Results written to {output_path}")
    if args.markdown_output:
        print(f"Markdown report written to {args.markdown_output}")
    print(
        "Summary: "
        f"recall@{result['config']['k']}={metrics['recall_at_k']:.4f}, "
        f"qps={metrics['qps']:.2f}, "
        f"p99={metrics['latency_ms']['p99']:.4f}ms"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
