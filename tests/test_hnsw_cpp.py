from pathlib import Path

import numpy as np
import pytest

from src.index.hnsw import HNSWIndex, HNSWNode


def test_hnsw_insert_uses_cpp_pruning_when_available(monkeypatch):
    import src.index.hnsw as hnsw_module

    calls = []

    class FakeCppModule:
        def prune_connections(
            self,
            vectors,
            node_id,
            connection_ids,
            max_connections,
            metric,
        ):
            calls.append(
                {
                    "vectors_dtype": vectors.dtype,
                    "node_id": node_id,
                    "connection_count": len(connection_ids),
                    "max_connections": max_connections,
                    "metric": metric,
                }
            )
            return list(connection_ids)[:max_connections]

    monkeypatch.setattr(hnsw_module, "CPP_AVAILABLE", True)
    monkeypatch.setattr(hnsw_module, "hnsw_cpp", FakeCppModule())
    monkeypatch.setattr(HNSWIndex, "_assign_layer", lambda self: 0)

    index = HNSWIndex(M=1, ef_construction=4, metric="euclidean")
    vectors = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.2, 0.0],
            [0.3, 0.0],
        ],
        dtype=np.float32,
    )
    index.vectors = vectors

    for vector_id, vector in enumerate(vectors):
        index.insert(vector, vector_id=vector_id)

    assert calls
    assert calls[0]["vectors_dtype"] == np.float32
    assert all(call["max_connections"] == 2 for call in calls)
    assert all(call["metric"] == "euclidean" for call in calls)


def test_cpp_search_layer_matches_euclidean_order():
    from src.index import hnsw_cpp

    vectors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 2.0],
            [5.0, 5.0, 5.0],
        ],
        dtype=np.float32,
    )
    query = np.array([0.9, 0.0, 0.0], dtype=np.float32)

    offsets = np.array([0, 2, 4, 6, 8], dtype=np.int32)
    neighbors = np.array([1, 2, 0, 3, 0, 3, 1, 2], dtype=np.int32)

    results = hnsw_cpp.search_layer(
        query=query,
        vectors=vectors,
        offsets=offsets,
        neighbors=neighbors,
        entry_points=[1],
        num_closest=2,
        metric="euclidean",
    )

    assert [vector_id for vector_id, _distance in results] == [0, 1]
    assert results[0][1] == pytest.approx(0.1, abs=1e-6)


def test_cpp_build_graph_returns_searchable_connections():
    from src.index import hnsw_cpp

    vectors = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.2, 0.0],
            [3.0, 3.0],
            [3.1, 3.0],
        ],
        dtype=np.float32,
    )
    levels = np.array([0, 1, 0, 0, 0], dtype=np.int32)

    graph = hnsw_cpp.build_graph(
        vectors=vectors,
        levels=levels,
        max_connections=2,
        ef_construction=4,
        metric="euclidean",
    )

    assert graph["entry_point"] == 1
    assert graph["max_layer"] == 1
    assert graph["levels"] == [0, 1, 0, 0, 0]
    assert "layers" in graph

    connections = {
        (node_id, layer): set(neighbors)
        for node_id, layer, neighbors in graph["connections"]
    }
    assert set(connections[(0, 0)])
    assert all(len(neighbors) <= 4 for (_node_id, layer), neighbors in connections.items() if layer == 0)
    assert all(0 <= neighbor < len(vectors) for neighbors in connections.values() for neighbor in neighbors)

    layer_zero = graph["layers"][0]
    assert layer_zero["offsets"].dtype == np.int32
    assert layer_zero["neighbors"].dtype == np.int32
    assert len(layer_zero["offsets"]) == len(vectors) + 1
    assert layer_zero["offsets"][0] == 0
    assert layer_zero["offsets"][-1] == len(layer_zero["neighbors"])


def test_cpp_build_graph_wrapper_releases_gil_for_parallel_segments():
    source = Path("src/index/hnsw_cpp.pyx").read_text()
    assert "CppBuildGraphResult cpp_build_graph" in source
    assert "cpp_build_graph" in source and "nogil" in source
    assert "with nogil:" in source


def test_cpp_build_graph_returns_phase_stats():
    from src.index import hnsw_cpp

    vectors = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.2, 0.0],
            [3.0, 3.0],
            [3.1, 3.0],
        ],
        dtype=np.float32,
    )
    levels = np.array([0, 1, 0, 0, 0], dtype=np.int32)

    graph = hnsw_cpp.build_graph(
        vectors=vectors,
        levels=levels,
        max_connections=2,
        ef_construction=4,
        metric="euclidean",
        include_connections=False,
    )

    stats = graph["build_stats"]
    assert stats["vectors"] == len(vectors)
    assert stats["dimensions"] == vectors.shape[1]
    assert stats["max_layer"] == graph["max_layer"]
    assert stats["directed_edges"] == sum(
        int(layer["neighbors"].size) for layer in graph["layers"].values()
    )
    assert stats["total_seconds"] >= 0.0
    assert stats["construction_seconds"] >= 0.0
    assert stats["search_seconds"] >= 0.0
    assert stats["greedy_search_seconds"] >= 0.0
    assert stats["candidate_search_seconds"] >= 0.0
    assert stats["search_seconds"] == pytest.approx(
        stats["greedy_search_seconds"] + stats["candidate_search_seconds"]
    )
    assert stats["prune_seconds"] >= 0.0
    assert stats["csr_export_seconds"] >= 0.0


def test_cpp_build_graph_reports_detailed_build_counters():
    from src.index import hnsw_cpp

    vectors = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.2, 0.0],
            [0.3, 0.0],
            [2.0, 2.0],
            [2.1, 2.0],
            [2.2, 2.0],
        ],
        dtype=np.float32,
    )
    levels = np.array([0, 1, 0, 0, 1, 0, 0], dtype=np.int32)

    graph = hnsw_cpp.build_graph(
        vectors=vectors,
        levels=levels,
        max_connections=2,
        ef_construction=4,
        metric="euclidean",
        include_connections=False,
    )

    stats = graph["build_stats"]
    assert stats["distance_evaluations"] > 0
    assert stats["search_distance_evaluations"] > 0
    assert stats["neighbor_selection_distance_evaluations"] > 0
    assert stats["prune_distance_evaluations"] >= 0
    assert stats["distance_evaluations"] == (
        stats["search_distance_evaluations"]
        + stats["neighbor_selection_distance_evaluations"]
        + stats["prune_distance_evaluations"]
    )
    assert stats["visited_nodes"] == stats["search_distance_evaluations"]
    assert stats["max_visited_nodes_per_search"] > 0
    assert 0 < stats["candidate_heap_pushes"] <= stats["visited_nodes"]
    assert 0 < stats["result_heap_pushes"] <= stats["visited_nodes"]
    assert stats["neighbor_selection_calls"] > 0
    assert stats["selected_degree_total"] > 0
    assert stats["average_selected_degree"] > 0.0
    assert stats["max_selected_degree"] <= 4
    assert stats["prune_calls"] >= 0
    assert stats["max_prune_input_size"] >= 0
    assert stats["average_prune_input_size"] >= 0.0


def test_cpp_build_graph_skips_unpromising_search_heap_candidates():
    from src.index import hnsw_cpp

    rng = np.random.default_rng(1)
    vectors = rng.standard_normal((60, 8), dtype=np.float32)
    levels = np.zeros(len(vectors), dtype=np.int32)

    graph = hnsw_cpp.build_graph(
        vectors=vectors,
        levels=levels,
        max_connections=2,
        ef_construction=10,
        metric="euclidean",
        include_connections=False,
    )

    stats = graph["build_stats"]
    assert stats["visited_nodes"] == stats["search_distance_evaluations"]
    assert stats["candidate_heap_pushes"] < stats["visited_nodes"]
    assert stats["result_heap_pushes"] < stats["visited_nodes"]


def test_cpp_build_graph_reuses_visited_scratch():
    from src.index import hnsw_cpp

    vectors = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.2, 0.0],
            [0.3, 0.0],
            [0.4, 0.0],
            [0.5, 0.0],
        ],
        dtype=np.float32,
    )
    levels = np.zeros(len(vectors), dtype=np.int32)

    graph = hnsw_cpp.build_graph(
        vectors=vectors,
        levels=levels,
        max_connections=2,
        ef_construction=4,
        metric="euclidean",
        include_connections=False,
    )

    stats = graph["build_stats"]
    assert stats["search_calls"] == len(vectors) - 1
    assert stats["greedy_search_calls"] == 0
    assert stats["candidate_search_calls"] == len(vectors) - 1
    assert stats["search_calls"] == stats["greedy_search_calls"] + stats["candidate_search_calls"]
    assert stats["visited_resizes"] == 1
    assert stats["visited_resizes"] < stats["search_calls"]
    assert stats["uses_reusable_search_heaps"] is True
    assert stats["search_heap_resizes"] > 0
    assert stats["search_heap_resizes"] < stats["search_calls"] * 2


def test_cpp_build_graph_reports_bounded_adjacency_layout():
    from src.index import hnsw_cpp

    vectors = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.2, 0.0],
            [0.3, 0.0],
            [2.0, 2.0],
            [2.1, 2.0],
        ],
        dtype=np.float32,
    )
    levels = np.array([0, 1, 0, 2, 0, 0], dtype=np.int32)
    max_connections = 2

    graph = hnsw_cpp.build_graph(
        vectors=vectors,
        levels=levels,
        max_connections=max_connections,
        ef_construction=4,
        metric="euclidean",
        include_connections=True,
    )

    stats = graph["build_stats"]
    assert stats["uses_bounded_adjacency"] is True
    assert stats["uses_heuristic_neighbors"] is True
    assert stats["uses_heuristic_reverse_pruning"] is False
    assert stats["adjacency_layers_allocated"] >= int(np.sum(levels + 1))
    assert stats["adjacency_layers_allocated"] <= len(vectors) * (graph["max_layer"] + 1)
    assert stats["max_observed_degree"] <= max_connections * 2

    for node_id, layer, neighbors in graph["connections"]:
        expected_limit = max_connections * 2 if layer == 0 else max_connections
        assert len(neighbors) <= expected_limit, (node_id, layer, neighbors)


def test_cpp_select_heuristic_neighbors_prefers_diverse_euclidean_neighbors():
    from src.index import hnsw_cpp

    vectors = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.1, 0.0],
            [0.0, 1.2],
        ],
        dtype=np.float32,
    )

    selected = hnsw_cpp.select_heuristic_neighbors(
        vectors=vectors,
        node_id=0,
        candidate_ids=[1, 2, 3],
        max_connections=2,
        metric="euclidean",
    )

    assert selected == [1, 3]


def test_cpp_build_graph_reports_squared_l2_mode_for_euclidean():
    from src.index import hnsw_cpp

    vectors = np.array(
        [
            [0.0, 0.0],
            [3.0, 4.0],
            [6.0, 8.0],
        ],
        dtype=np.float32,
    )
    levels = np.array([0, 0, 0], dtype=np.int32)

    euclidean_graph = hnsw_cpp.build_graph(
        vectors=vectors,
        levels=levels,
        max_connections=2,
        ef_construction=4,
        metric="euclidean",
        include_connections=False,
    )
    assert euclidean_graph["build_stats"]["uses_squared_l2"] is True
    assert euclidean_graph["build_stats"]["uses_float_l2_accumulation"] is True

    cosine_graph = hnsw_cpp.build_graph(
        vectors=vectors,
        levels=levels,
        max_connections=2,
        ef_construction=4,
        metric="cosine",
        include_connections=False,
    )
    assert cosine_graph["build_stats"]["uses_squared_l2"] is False
    assert cosine_graph["build_stats"]["uses_float_l2_accumulation"] is False


def test_cpp_build_graph_can_skip_connection_rows():
    from src.index import hnsw_cpp

    vectors = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.2, 0.0],
            [3.0, 3.0],
            [3.1, 3.0],
        ],
        dtype=np.float32,
    )
    levels = np.array([0, 1, 0, 0, 0], dtype=np.int32)

    graph = hnsw_cpp.build_graph(
        vectors=vectors,
        levels=levels,
        max_connections=2,
        ef_construction=4,
        metric="euclidean",
        include_connections=False,
    )

    assert graph["connections"] == []
    assert graph["layers"][0]["neighbors"].size > 0


def test_hnsw_build_stores_cpp_build_stats(monkeypatch):
    import src.index.hnsw as hnsw_module

    class FakeCppModule:
        def build_graph(
            self,
            vectors,
            levels,
            max_connections,
            ef_construction,
            metric,
            include_connections=True,
        ):
            return {
                "entry_point": 0,
                "max_layer": 0,
                "levels": [0, 0],
                "connections": [],
                "layers": {
                    0: {
                        "offsets": np.array([0, 1, 2], dtype=np.int32),
                        "neighbors": np.array([1, 0], dtype=np.int32),
                    },
                },
                "build_stats": {
                    "vectors": 2,
                    "dimensions": 2,
                    "max_layer": 0,
                    "directed_edges": 2,
                    "total_seconds": 0.25,
                    "construction_seconds": 0.2,
                    "search_seconds": 0.15,
                    "greedy_search_seconds": 0.0,
                    "candidate_search_seconds": 0.15,
                    "prune_seconds": 0.01,
                    "csr_export_seconds": 0.05,
                    "uses_squared_l2": True,
                    "uses_float_l2_accumulation": True,
                    "search_calls": 1,
                    "greedy_search_calls": 0,
                    "candidate_search_calls": 1,
                    "visited_resizes": 1,
                    "uses_reusable_search_heaps": True,
                    "search_heap_resizes": 2,
                    "uses_bounded_adjacency": True,
                    "uses_heuristic_neighbors": True,
                    "uses_heuristic_reverse_pruning": False,
                    "adjacency_layers_allocated": 2,
                    "max_observed_degree": 1,
                    "distance_evaluations": 12,
                    "search_distance_evaluations": 8,
                    "neighbor_selection_distance_evaluations": 3,
                    "prune_distance_evaluations": 1,
                    "visited_nodes": 8,
                    "max_visited_nodes_per_search": 4,
                    "candidate_heap_pushes": 8,
                    "result_heap_pushes": 8,
                    "neighbor_selection_calls": 1,
                    "selected_degree_total": 2,
                    "average_selected_degree": 2.0,
                    "max_selected_degree": 2,
                    "prune_calls": 1,
                    "prune_input_total": 3,
                    "average_prune_input_size": 3.0,
                    "max_prune_input_size": 3,
                },
            }

    monkeypatch.setattr(hnsw_module, "CPP_AVAILABLE", True)
    monkeypatch.setattr(hnsw_module, "hnsw_cpp", FakeCppModule())
    monkeypatch.setattr(HNSWIndex, "_assign_layer", lambda self: 0)

    index = HNSWIndex(M=2, ef_construction=4, metric="euclidean")
    vectors = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )

    index.build(vectors)

    assert index._last_cpp_build_stats == {
        "vectors": 2,
        "dimensions": 2,
        "max_layer": 0,
        "directed_edges": 2,
        "total_seconds": 0.25,
        "construction_seconds": 0.2,
        "search_seconds": 0.15,
        "greedy_search_seconds": 0.0,
        "candidate_search_seconds": 0.15,
        "prune_seconds": 0.01,
        "csr_export_seconds": 0.05,
        "uses_squared_l2": True,
        "uses_float_l2_accumulation": True,
        "search_calls": 1,
        "greedy_search_calls": 0,
        "candidate_search_calls": 1,
        "visited_resizes": 1,
        "uses_reusable_search_heaps": True,
        "search_heap_resizes": 2,
        "uses_bounded_adjacency": True,
        "uses_heuristic_neighbors": True,
        "uses_heuristic_reverse_pruning": False,
        "adjacency_layers_allocated": 2,
        "max_observed_degree": 1,
        "distance_evaluations": 12,
        "search_distance_evaluations": 8,
        "neighbor_selection_distance_evaluations": 3,
        "prune_distance_evaluations": 1,
        "visited_nodes": 8,
        "max_visited_nodes_per_search": 4,
        "candidate_heap_pushes": 8,
        "result_heap_pushes": 8,
        "neighbor_selection_calls": 1,
        "selected_degree_total": 2,
        "average_selected_degree": 2.0,
        "max_selected_degree": 2,
        "prune_calls": 1,
        "prune_input_total": 3,
        "average_prune_input_size": 3.0,
        "max_prune_input_size": 3,
    }


def test_hnsw_build_uses_cpp_batch_builder_when_available(monkeypatch):
    import src.index.hnsw as hnsw_module

    calls = []

    class FakeCppModule:
        def build_graph(
            self,
            vectors,
            levels,
            max_connections,
            ef_construction,
            metric,
            include_connections=True,
        ):
            calls.append(
                {
                    "vectors_dtype": vectors.dtype,
                    "levels": levels.tolist(),
                    "max_connections": max_connections,
                    "ef_construction": ef_construction,
                    "metric": metric,
                    "include_connections": include_connections,
                }
            )
            return {
                "entry_point": 1,
                "max_layer": 1,
                "levels": [0, 1, 0],
                "connections": [],
                "layers": {
                    0: {
                        "offsets": np.array([0, 1, 3, 4], dtype=np.int32),
                        "neighbors": np.array([1, 0, 2, 1], dtype=np.int32),
                    },
                    1: {
                        "offsets": np.array([0, 0, 1, 1], dtype=np.int32),
                        "neighbors": np.array([0], dtype=np.int32),
                    },
                },
            }

    assigned_layers = iter([0, 1, 0])

    monkeypatch.setattr(hnsw_module, "CPP_AVAILABLE", True)
    monkeypatch.setattr(hnsw_module, "hnsw_cpp", FakeCppModule())
    monkeypatch.setattr(HNSWIndex, "_assign_layer", lambda self: next(assigned_layers))

    index = HNSWIndex(M=2, ef_construction=8, metric="euclidean")
    vectors = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.2, 0.0],
        ],
        dtype=np.float64,
    )

    index.build(vectors)

    assert calls == [
        {
            "vectors_dtype": np.float32,
            "levels": [0, 1, 0],
            "max_connections": 2,
            "ef_construction": 8,
            "metric": "euclidean",
            "include_connections": False,
        }
    ]
    assert index.entry_point == 1
    assert index.max_layer == 1
    assert index.nodes[1].layer == 1
    assert index.nodes[1].connections == {}

    index.materialize_python_graph()

    assert index.nodes[1].connections[0] == {0, 2}
    assert index.nodes[1].connections[1] == {0}
    assert 0 in index._cpp_graph_cache
    assert np.array_equal(index._cpp_graph_cache[0][0], np.array([0, 1, 3, 4], dtype=np.int32))
    assert np.array_equal(index._cpp_graph_cache[0][1], np.array([1, 0, 2, 1], dtype=np.int32))


def test_hnsw_build_reuses_cpp_builder_cache(monkeypatch):
    import src.index.hnsw as hnsw_module

    class FakeCppModule:
        def build_graph(
            self,
            vectors,
            levels,
            max_connections,
            ef_construction,
            metric,
            include_connections=True,
        ):
            return {
                "entry_point": 0,
                "max_layer": 0,
                "levels": [0, 0],
                "connections": [],
                "layers": {
                    0: {
                        "offsets": np.array([0, 1, 2], dtype=np.int32),
                        "neighbors": np.array([1, 0], dtype=np.int32),
                    },
                },
            }

    def fail_rebuild(self):
        raise AssertionError("build should reuse c++ builder cache")

    monkeypatch.setattr(hnsw_module, "CPP_AVAILABLE", True)
    monkeypatch.setattr(hnsw_module, "hnsw_cpp", FakeCppModule())
    monkeypatch.setattr(HNSWIndex, "_assign_layer", lambda self: 0)
    monkeypatch.setattr(HNSWIndex, "_build_cpp_cache", fail_rebuild)

    index = HNSWIndex(M=2, ef_construction=4, metric="euclidean")
    vectors = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )

    index.build(vectors)

    assert np.array_equal(index._cpp_graph_cache[0][0], np.array([0, 1, 2], dtype=np.int32))
    assert np.array_equal(index._cpp_graph_cache[0][1], np.array([1, 0], dtype=np.int32))


def test_hnsw_search_batch_uses_cpp_batch_cache(monkeypatch):
    import src.index.hnsw as hnsw_module

    calls = []

    class FakeCppModule:
        def search_batch(
            self,
            queries,
            vectors,
            layers,
            entry_point,
            max_layer,
            k,
            ef,
            metric,
        ):
            calls.append(
                {
                    "queries_shape": queries.shape,
                    "vectors_dtype": vectors.dtype,
                    "layers": sorted(layers),
                    "entry_point": entry_point,
                    "max_layer": max_layer,
                    "k": k,
                    "ef": ef,
                    "metric": metric,
                }
            )
            return [[(0, 0.0), (1, 1.0)], [(1, 0.0), (0, 1.0)]]

    monkeypatch.setattr(hnsw_module, "CPP_AVAILABLE", True)
    monkeypatch.setattr(hnsw_module, "hnsw_cpp", FakeCppModule())

    index = HNSWIndex(M=2, ef_search=20, metric="euclidean")
    index.vectors = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )
    index._vectors_f32_cache = np.ascontiguousarray(index.vectors, dtype=np.float32)
    index._cpp_graph_cache = {
        0: (
            np.array([0, 1, 2], dtype=np.int32),
            np.array([1, 0], dtype=np.int32),
        )
    }
    index._python_graph_materialized = False
    index.entry_point = 0
    index.max_layer = 0

    queries = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)

    results = index.search_batch(queries, k=2, ef=7)

    assert results == [[(0, 0.0), (1, 1.0)], [(1, 0.0), (0, 1.0)]]
    assert calls == [
        {
            "queries_shape": (2, 2),
            "vectors_dtype": np.dtype("float32"),
            "layers": [0],
            "entry_point": 0,
            "max_layer": 0,
            "k": 2,
            "ef": 7,
            "metric": "euclidean",
        }
    ]


def test_save_load_preserves_compact_cpp_graph_without_materializing(monkeypatch, tmp_path):
    def fail_materialize(self):
        raise AssertionError("save/load should keep compact CSR graph shape")

    vectors = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ],
        dtype=np.float32,
    )
    offsets = np.array([0, 1, 3, 4], dtype=np.int32)
    neighbors = np.array([1, 0, 2, 1], dtype=np.int32)

    index = HNSWIndex(M=2, ef_construction=8, metric="euclidean")
    index.vectors = vectors
    index.nodes = {
        0: HNSWNode(vector_id=0, layer=0),
        1: HNSWNode(vector_id=1, layer=0),
        2: HNSWNode(vector_id=2, layer=0),
    }
    index.entry_point = 0
    index.max_layer = 0
    index._cpp_graph_cache = {0: (offsets, neighbors)}
    index._vectors_f32_cache = vectors
    index._python_graph_materialized = False

    monkeypatch.setattr(HNSWIndex, "_ensure_python_graph_materialized", fail_materialize)

    filepath = tmp_path / "index.npz"
    index.save(str(filepath))

    assert index._python_graph_materialized is False
    assert all(node.connections == {} for node in index.nodes.values())

    loaded = HNSWIndex()
    loaded.load(str(filepath))

    assert loaded._python_graph_materialized is False
    assert all(node.connections == {} for node in loaded.nodes.values())
    assert loaded._vectors_f32_cache is not None
    assert np.array_equal(loaded._cpp_graph_cache[0][0], offsets)
    assert np.array_equal(loaded._cpp_graph_cache[0][1], neighbors)


def test_insert_after_cpp_batch_build_materializes_existing_graph(monkeypatch):
    import src.index.hnsw as hnsw_module

    if not hnsw_module.CPP_AVAILABLE:
        pytest.skip("C++ extension is not available")

    monkeypatch.setattr(HNSWIndex, "_assign_layer", lambda self: 0)

    index = HNSWIndex(M=2, ef_construction=8, metric="euclidean")
    vectors = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ],
        dtype=np.float32,
    )
    index.build(vectors)

    assert index._python_graph_materialized is False
    assert index.nodes[1].connections == {}

    extended_vectors = np.vstack([vectors, np.array([[3.0, 0.0]], dtype=np.float32)])
    index.vectors = extended_vectors
    assert index.graph_storage_mode == "compact_csr"
    with pytest.warns(RuntimeWarning, match="materializes the compact CSR graph"):
        index.insert(extended_vectors[3], vector_id=3)

    assert index._python_graph_materialized is True
    assert index.graph_storage_mode == "materialized_python"
    assert index._graph_connections_cache is not None
    assert any(3 in neighbors for node in index.nodes.values() for neighbors in node.connections.values())
    assert index.search(extended_vectors[3], k=1)[0][0] == 3


def test_delete_after_cpp_batch_build_materializes_with_warning(monkeypatch):
    import src.index.hnsw as hnsw_module

    if not hnsw_module.CPP_AVAILABLE:
        pytest.skip("C++ extension is not available")

    monkeypatch.setattr(HNSWIndex, "_assign_layer", lambda self: 0)

    index = HNSWIndex(M=2, ef_construction=8, metric="euclidean")
    vectors = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
        ],
        dtype=np.float32,
    )
    index.build(vectors)

    assert index.graph_storage_mode == "compact_csr"
    with pytest.warns(RuntimeWarning, match="materializes the compact CSR graph"):
        index.delete(2)

    assert index._python_graph_materialized is True
    assert index.graph_storage_mode == "materialized_python"
    assert 2 not in index.nodes
    assert all(2 not in neighbors for node in index.nodes.values() for neighbors in node.connections.values())


def test_cpp_search_layer_matches_cosine_order():
    from src.index import hnsw_cpp

    vectors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.707, 0.707, 0.0],
        ],
        dtype=np.float32,
    )
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    offsets = np.array([0, 2, 4, 6], dtype=np.int32)
    neighbors = np.array([1, 2, 0, 2, 0, 1], dtype=np.int32)

    results = hnsw_cpp.search_layer(
        query=query,
        vectors=vectors,
        offsets=offsets,
        neighbors=neighbors,
        entry_points=[1],
        num_closest=2,
        metric="cosine",
    )

    assert [vector_id for vector_id, _similarity in results] == [0, 2]
    assert results[0][1] == pytest.approx(1.0, abs=1e-6)


def test_cpp_prune_connections_keeps_nearest_euclidean_neighbors():
    from src.index import hnsw_cpp

    vectors = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [2.0, 0.0],
            [0.0, 0.2],
            [3.0, 3.0],
        ],
        dtype=np.float32,
    )

    pruned = hnsw_cpp.prune_connections(
        vectors=vectors,
        node_id=0,
        connection_ids=[1, 2, 3, 4],
        max_connections=2,
        metric="euclidean",
    )

    assert pruned == [1, 3]


def test_cpp_prune_connections_keeps_most_similar_cosine_neighbors():
    from src.index import hnsw_cpp

    vectors = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.7, 0.7],
        ],
        dtype=np.float32,
    )

    pruned = hnsw_cpp.prune_connections(
        vectors=vectors,
        node_id=0,
        connection_ids=[1, 2, 3],
        max_connections=2,
        metric="cosine",
    )

    assert pruned == [1, 3]


def test_hnsw_layer_search_can_use_cpp_cache():
    index = HNSWIndex(M=2, metric="euclidean")
    index.vectors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 2.0],
            [5.0, 5.0, 5.0],
        ],
        dtype=np.float32,
    )
    index.nodes[0] = HNSWNode(vector_id=0, layer=0, connections={0: {1, 2}})
    index.nodes[1] = HNSWNode(vector_id=1, layer=0, connections={0: {0, 3}})
    index.nodes[2] = HNSWNode(vector_id=2, layer=0, connections={0: {0, 3}})
    index.nodes[3] = HNSWNode(vector_id=3, layer=0, connections={0: {1, 2}})

    query = np.array([0.9, 0.0, 0.0], dtype=np.float32)

    python_results = index._search_layer(
        query,
        entry_points=[1],
        num_closest=2,
        layer=0,
        use_cython=False,
        use_cpp=False,
    )

    index._build_cpp_cache()
    cpp_results = index._search_layer(
        query,
        entry_points=[1],
        num_closest=2,
        layer=0,
        use_cython=False,
        use_cpp=True,
    )

    assert [vector_id for vector_id, _distance in cpp_results] == [
        vector_id for vector_id, _distance in python_results
    ]


def test_cpp_search_batch_matches_repeated_index_search():
    from src.index import hnsw_cpp

    index = HNSWIndex(M=2, ef_construction=8, ef_search=8, metric="euclidean")
    vectors = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.2, 0.0],
            [2.0, 2.0],
            [2.1, 2.0],
            [2.2, 2.0],
        ],
        dtype=np.float32,
    )
    queries = np.array(
        [
            [0.05, 0.0],
            [2.15, 2.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )

    np.random.seed(7)
    index.build(vectors)

    repeated_results = [index.search(query, k=3, ef=8) for query in queries]
    batch_results = hnsw_cpp.search_batch(
        queries=queries,
        vectors=index._vectors_f32_cache,
        layers=index._cpp_graph_cache,
        entry_point=index.entry_point,
        max_layer=index.max_layer,
        k=3,
        ef=8,
        metric=index.metric,
    )

    assert len(batch_results) == len(repeated_results)
    for batch, repeated in zip(batch_results, repeated_results):
        assert [vector_id for vector_id, _distance in batch] == [
            vector_id for vector_id, _distance in repeated
        ]
        assert [distance for _vector_id, distance in batch] == pytest.approx(
            [distance for _vector_id, distance in repeated],
            abs=1e-6,
        )
