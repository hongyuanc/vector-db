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
    assert set(metrics["cpp_build_stats"]) == {
        "vectors",
        "dimensions",
        "max_layer",
        "directed_edges",
        "total_seconds",
        "construction_seconds",
        "search_seconds",
        "greedy_search_seconds",
        "candidate_search_seconds",
        "prune_seconds",
        "csr_export_seconds",
        "uses_squared_l2",
        "uses_float_l2_accumulation",
        "search_calls",
        "greedy_search_calls",
        "candidate_search_calls",
        "visited_resizes",
        "uses_reusable_search_heaps",
        "search_heap_resizes",
        "uses_bounded_adjacency",
        "uses_heuristic_neighbors",
        "uses_heuristic_reverse_pruning",
        "adjacency_layers_allocated",
        "max_observed_degree",
        "distance_evaluations",
        "search_distance_evaluations",
        "neighbor_selection_distance_evaluations",
        "prune_distance_evaluations",
        "visited_nodes",
        "max_visited_nodes_per_search",
        "candidate_heap_pushes",
        "result_heap_pushes",
        "neighbor_selection_calls",
        "selected_degree_total",
        "average_selected_degree",
        "max_selected_degree",
        "prune_calls",
        "prune_input_total",
        "average_prune_input_size",
        "max_prune_input_size",
    }
    assert metrics["cpp_build_stats"]["vectors"] == 80
    assert metrics["cpp_build_stats"]["dimensions"] == 8
    assert metrics["qps"] > 0
    assert 0.0 <= metrics["recall_at_k"] <= 1.0
    assert set(metrics["latency_ms"]) == {"p50", "p95", "p99", "average"}
    assert metrics["latency_ms"]["p99"] >= metrics["latency_ms"]["p50"]
    assert "vector_data_mb" in metrics["memory"]
    assert "graph" in metrics["memory"]
    assert metrics["memory"]["graph"]["python_nodes"] > 0
    assert "python_graph_materialized" in metrics["memory"]["graph"]
    assert metrics["memory"]["graph"]["total_graph_mb"] > 0
    assert set(metrics["persistence"]) == {"compact", "materialized"}
    assert set(metrics["persistence"]["compact"]) == {
        "available",
        "save_time_seconds",
        "load_time_seconds",
        "file_size_mb",
        "process_peak_rss_mb",
        "loaded_graph",
    }
    assert set(metrics["persistence"]["materialized"]) == {
        "available",
        "save_time_seconds",
        "load_time_seconds",
        "file_size_mb",
        "process_peak_rss_mb",
        "loaded_graph",
    }
    assert metrics["persistence"]["materialized"]["available"] is True
    assert metrics["persistence"]["materialized"]["save_time_seconds"] >= 0
    assert metrics["persistence"]["materialized"]["load_time_seconds"] >= 0
    assert metrics["persistence"]["materialized"]["loaded_graph"]["python_graph_materialized"] is True
    if result["environment"]["cpp_available"]:
        assert metrics["persistence"]["compact"]["available"] is True
        assert metrics["persistence"]["compact"]["loaded_graph"]["python_graph_materialized"] is False
        assert metrics["persistence"]["compact"]["loaded_graph"]["cpp_edges"] > 0
    else:
        assert metrics["persistence"]["compact"]["available"] is False

    assert result["environment"]["git_commit"]
    assert "python" in result["environment"]
    assert "numpy" in result["environment"]
    assert "cpp_available" in result["environment"]
    if result["environment"]["cpp_available"]:
        assert metrics["memory"]["graph"]["python_edges"] == 0
        assert metrics["memory"]["graph"]["cpp_edges"] > 0
        cpp_build_stats = metrics["cpp_build_stats"]
        assert cpp_build_stats["distance_evaluations"] == (
            cpp_build_stats["search_distance_evaluations"]
            + cpp_build_stats["neighbor_selection_distance_evaluations"]
            + cpp_build_stats["prune_distance_evaluations"]
        )
        assert cpp_build_stats["visited_nodes"] == cpp_build_stats["search_distance_evaluations"]
        assert cpp_build_stats["max_visited_nodes_per_search"] > 0
        assert cpp_build_stats["candidate_heap_pushes"] >= cpp_build_stats["visited_nodes"]
        assert cpp_build_stats["result_heap_pushes"] >= cpp_build_stats["visited_nodes"]
        assert cpp_build_stats["neighbor_selection_calls"] > 0
        assert cpp_build_stats["average_selected_degree"] > 0
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
    assert "C++ Build Total" in markdown
    assert "C++ Visited Resizes" in markdown
    assert "C++ Distance Evaluations" in markdown
    assert "C++ Visited Nodes" in markdown
    assert "C++ Average Selected Degree" in markdown
    assert "C++ Average Prune Input Size" in markdown
    assert "Compact Save Time" in markdown
    assert "Materialized Save Time" in markdown
