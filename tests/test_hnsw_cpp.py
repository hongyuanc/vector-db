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

    connections = {
        (node_id, layer): set(neighbors)
        for node_id, layer, neighbors in graph["connections"]
    }
    assert set(connections[(0, 0)])
    assert all(len(neighbors) <= 4 for (_node_id, layer), neighbors in connections.items() if layer == 0)
    assert all(0 <= neighbor < len(vectors) for neighbors in connections.values() for neighbor in neighbors)


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
        ):
            calls.append(
                {
                    "vectors_dtype": vectors.dtype,
                    "levels": levels.tolist(),
                    "max_connections": max_connections,
                    "ef_construction": ef_construction,
                    "metric": metric,
                }
            )
            return {
                "entry_point": 1,
                "max_layer": 1,
                "levels": [0, 1, 0],
                "connections": [
                    (0, 0, [1]),
                    (1, 0, [0, 2]),
                    (1, 1, [0]),
                    (2, 0, [1]),
                ],
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
        }
    ]
    assert index.entry_point == 1
    assert index.max_layer == 1
    assert index.nodes[1].layer == 1
    assert index.nodes[1].connections[0] == {0, 2}
    assert index.nodes[1].connections[1] == {0}


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
