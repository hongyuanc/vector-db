import json
from importlib import import_module


def _result(metrics):
    return {
        "schema_version": "benchmark.v1",
        "environment": {"git_commit": "abc123"},
        "dataset": {"name": "random", "size": 100, "dimension": 8, "n_queries": 10, "seed": 42},
        "config": {
            "hnsw": {"M": 4, "ef_construction": 20, "ef_search": 10, "metric": "euclidean"},
            "k": 3,
        },
        "metrics": metrics,
    }


def test_compare_results_calculates_deltas_and_statuses():
    compare = import_module("benchmarks.compare_results")
    baseline = _result(
        {
            "build_time_seconds": 10.0,
            "vectors_per_second": 100.0,
            "qps": 1000.0,
            "latency_ms": {"average": 1.0, "p50": 0.8, "p95": 1.5, "p99": 2.0},
            "recall_at_k": 0.9,
            "memory": {
                "vector_data_mb": 1.0,
                "process_peak_rss_mb": 100.0,
                "graph": {"total_graph_mb": 3.0},
            },
        }
    )
    candidate = _result(
        {
            "build_time_seconds": 8.0,
            "vectors_per_second": 125.0,
            "qps": 900.0,
            "latency_ms": {"average": 0.9, "p50": 0.7, "p95": 1.6, "p99": 1.8},
            "recall_at_k": 0.92,
            "memory": {
                "vector_data_mb": 1.0,
                "process_peak_rss_mb": 120.0,
                "graph": {"total_graph_mb": 2.0},
            },
        }
    )

    comparison = compare.compare_results(baseline, candidate)

    assert comparison["baseline"]["commit"] == "abc123"
    assert comparison["candidate"]["commit"] == "abc123"

    metrics = comparison["metrics"]
    assert metrics["build_time_seconds"]["delta"] == -2.0
    assert metrics["build_time_seconds"]["percent_delta"] == -20.0
    assert metrics["build_time_seconds"]["status"] == "improved"

    assert metrics["qps"]["delta"] == -100.0
    assert metrics["qps"]["percent_delta"] == -10.0
    assert metrics["qps"]["status"] == "regressed"

    assert metrics["recall_at_k"]["delta"] == 0.02
    assert metrics["recall_at_k"]["status"] == "improved"

    assert metrics["latency_ms.p95"]["delta"] == 0.1
    assert metrics["latency_ms.p95"]["status"] == "regressed"
    assert metrics["memory.graph.total_graph_mb"]["delta"] == -1.0
    assert metrics["memory.graph.total_graph_mb"]["status"] == "improved"


def test_main_writes_markdown_comparison(tmp_path):
    compare = import_module("benchmarks.compare_results")
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    output_path = tmp_path / "comparison.md"

    baseline_path.write_text(
        json.dumps(
            _result(
                {
                    "build_time_seconds": 10.0,
                    "vectors_per_second": 100.0,
                    "qps": 1000.0,
                    "latency_ms": {"average": 1.0, "p50": 0.8, "p95": 1.5, "p99": 2.0},
                    "recall_at_k": 0.9,
                    "memory": {
                        "vector_data_mb": 1.0,
                        "process_peak_rss_mb": 100.0,
                        "graph": {"total_graph_mb": 3.0},
                    },
                }
            )
        )
    )
    candidate_path.write_text(
        json.dumps(
            _result(
                {
                    "build_time_seconds": 8.0,
                    "vectors_per_second": 125.0,
                    "qps": 900.0,
                    "latency_ms": {"average": 0.9, "p50": 0.7, "p95": 1.6, "p99": 1.8},
                    "recall_at_k": 0.92,
                    "memory": {
                        "vector_data_mb": 1.0,
                        "process_peak_rss_mb": 120.0,
                        "graph": {"total_graph_mb": 2.0},
                    },
                }
            )
        )
    )

    exit_code = compare.main([str(baseline_path), str(candidate_path), "--output", str(output_path)])

    assert exit_code == 0
    markdown = output_path.read_text()
    assert "# Benchmark Comparison" in markdown
    assert "build_time_seconds" in markdown
    assert "improved" in markdown
    assert "regressed" in markdown
