import json
from importlib import import_module


def test_run_benchmark_suite_returns_structured_metrics():
    benchmark = import_module("benchmarks.benchmark")

    result = benchmark.run_benchmark_suite(
        dataset="random",
        size=80,
        dimension=8,
        n_queries=6,
        k=3,
        metric="euclidean",
        M=4,
        ef_construction=20,
        ef_search=10,
        seed=123,
        warmup_queries=1,
    )

    assert result["schema_version"] == "benchmark.v1"
    assert result["dataset"] == {
        "name": "random",
        "size": 80,
        "dimension": 8,
        "n_queries": 6,
        "seed": 123,
    }
    assert result["config"]["hnsw"] == {
        "M": 4,
        "ef_construction": 20,
        "ef_search": 10,
        "metric": "euclidean",
    }

    metrics = result["metrics"]
    assert metrics["build_time_seconds"] >= 0
    assert metrics["vectors_per_second"] > 0
    assert metrics["qps"] > 0
    assert 0.0 <= metrics["recall_at_k"] <= 1.0
    assert set(metrics["latency_ms"]) == {"p50", "p95", "p99", "average"}
    assert metrics["latency_ms"]["p99"] >= metrics["latency_ms"]["p50"]
    assert "vector_data_mb" in metrics["memory"]
    assert "graph" in metrics["memory"]
    assert metrics["memory"]["graph"]["python_nodes"] > 0
    assert "python_graph_materialized" in metrics["memory"]["graph"]
    assert metrics["memory"]["graph"]["total_graph_mb"] > 0

    assert result["environment"]["git_commit"]
    assert "python" in result["environment"]
    assert "numpy" in result["environment"]
    assert "cpp_available" in result["environment"]
    if result["environment"]["cpp_available"]:
        assert metrics["memory"]["graph"]["python_edges"] == 0
        assert metrics["memory"]["graph"]["cpp_edges"] > 0
    else:
        assert metrics["memory"]["graph"]["python_edges"] > 0
        assert metrics["memory"]["graph"]["cpp_edges"] == 0


def test_main_writes_json_and_markdown_reports(tmp_path):
    benchmark = import_module("benchmarks.benchmark")
    json_path = tmp_path / "results.json"
    markdown_path = tmp_path / "results.md"

    exit_code = benchmark.main(
        [
            "--dataset",
            "random",
            "--size",
            "60",
            "--dimension",
            "8",
            "--queries",
            "5",
            "--k",
            "3",
            "--M",
            "4",
            "--ef-construction",
            "20",
            "--ef-search",
            "10",
            "--warmup-queries",
            "1",
            "--output",
            str(json_path),
            "--markdown-output",
            str(markdown_path),
        ]
    )

    assert exit_code == 0

    data = json.loads(json_path.read_text())
    assert data["schema_version"] == "benchmark.v1"
    assert data["metrics"]["qps"] > 0

    markdown = markdown_path.read_text()
    assert "# Benchmark Results" in markdown
    assert "Recall@3" in markdown
    assert "p99 Latency" in markdown
    assert "Graph Total" in markdown
