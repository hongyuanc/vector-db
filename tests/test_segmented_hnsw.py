import numpy as np
import pytest


class FakeExactSegment:
    def __init__(self, M=16, ef_construction=200, ef_search=50, ml=None, metric="euclidean"):
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.ml = ml
        self.metric = metric
        self.vectors = None
        self._last_cpp_build_stats = None

    @property
    def graph_storage_mode(self):
        return "compact_csr"

    def build(self, vectors):
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self._last_cpp_build_stats = {
            "vectors": int(len(self.vectors)),
            "dimensions": int(self.vectors.shape[1]),
            "max_layer": 0,
            "directed_edges": int(max(0, len(self.vectors) - 1)),
            "total_seconds": float(len(self.vectors)),
            "construction_seconds": float(len(self.vectors)),
            "search_seconds": float(len(self.vectors)) / 2.0,
            "greedy_search_seconds": 0.0,
            "candidate_search_seconds": float(len(self.vectors)) / 2.0,
            "prune_seconds": 0.0,
            "csr_export_seconds": 0.0,
            "uses_squared_l2": True,
            "uses_float_l2_accumulation": True,
            "search_calls": int(len(self.vectors)),
            "greedy_search_calls": 0,
            "candidate_search_calls": int(len(self.vectors)),
            "visited_resizes": 1,
            "uses_reusable_search_heaps": True,
            "search_heap_resizes": 1,
            "uses_bounded_adjacency": True,
            "uses_heuristic_neighbors": True,
            "uses_heuristic_reverse_pruning": False,
            "adjacency_layers_allocated": int(len(self.vectors)),
            "max_observed_degree": 2,
            "distance_evaluations": int(len(self.vectors) * 10),
            "search_distance_evaluations": int(len(self.vectors) * 6),
            "neighbor_selection_distance_evaluations": int(len(self.vectors) * 2),
            "prune_distance_evaluations": int(len(self.vectors) * 2),
            "visited_nodes": int(len(self.vectors) * 6),
            "max_visited_nodes_per_search": int(len(self.vectors)),
            "candidate_heap_pushes": int(len(self.vectors) * 3),
            "result_heap_pushes": int(len(self.vectors) * 3),
            "neighbor_selection_calls": int(len(self.vectors)),
            "selected_degree_total": int(len(self.vectors) * 2),
            "average_selected_degree": 2.0,
            "max_selected_degree": 2,
            "prune_calls": int(len(self.vectors)),
            "prune_input_total": int(len(self.vectors) * 2),
            "average_prune_input_size": 2.0,
            "max_prune_input_size": 2,
        }

    def search(self, query, k, ef=None):
        distances = np.linalg.norm(self.vectors - query, axis=1)
        order = np.argsort(distances, kind="stable")[:k]
        return [(int(local_id), float(distances[local_id])) for local_id in order]

    def search_batch(self, queries, k, ef=None):
        return [self.search(query, k=k, ef=ef) for query in queries]


def test_segmented_hnsw_builds_contiguous_segments_and_merges_global_results():
    from src.index.segmented_hnsw import SegmentedHNSWIndex

    vectors = np.array(
        [
            [0.0, 0.0],
            [0.2, 0.0],
            [0.4, 0.0],
            [1.0, 0.0],
            [1.2, 0.0],
            [1.4, 0.0],
            [2.0, 0.0],
            [2.2, 0.0],
            [2.4, 0.0],
            [2.6, 0.0],
        ],
        dtype=np.float32,
    )

    index = SegmentedHNSWIndex(
        M=2,
        ef_construction=8,
        ef_search=6,
        metric="euclidean",
        segment_count=3,
        build_threads=1,
        segment_factory=FakeExactSegment,
    )
    index.build(vectors)

    assert index.segment_offsets == [0, 4, 7]
    assert index.segment_sizes == [4, 3, 3]
    assert index.graph_storage_mode == "segmented_csr"

    results = index.search(np.array([2.05, 0.0], dtype=np.float32), k=4, ef=6)
    assert [vector_id for vector_id, _distance in results] == [6, 7, 8, 9]

    stats = index.segmented_build_stats
    assert stats["uses_segmented_build"] is True
    assert stats["segment_count"] == 3
    assert stats["build_threads"] == 1
    assert stats["segment_offsets"] == [0, 4, 7]
    assert stats["segment_sizes"] == [4, 3, 3]
    assert len(stats["segment_build_seconds"]) == 3
    assert stats["max_segment_build_seconds"] >= 0.0
    assert index._last_cpp_build_stats["vectors"] == 10
    assert index._last_cpp_build_stats["distance_evaluations"] == 100


def test_segmented_hnsw_search_batch_preserves_query_order_and_global_ids():
    from src.index.segmented_hnsw import SegmentedHNSWIndex

    vectors = np.array(
        [
            [0.0, 0.0],
            [0.5, 0.0],
            [10.0, 0.0],
            [10.5, 0.0],
        ],
        dtype=np.float32,
    )
    queries = np.array([[0.1, 0.0], [10.4, 0.0]], dtype=np.float32)

    index = SegmentedHNSWIndex(
        M=2,
        ef_construction=8,
        ef_search=6,
        metric="euclidean",
        segment_count=2,
        build_threads=1,
        segment_factory=FakeExactSegment,
    )
    index.build(vectors)

    batch = index.search_batch(queries, k=2, ef=6)
    assert [[vector_id for vector_id, _distance in row] for row in batch] == [
        [0, 1],
        [3, 2],
    ]


class FakeCosineSegment(FakeExactSegment):
    def search(self, query, k, ef=None):
        query_norm = np.linalg.norm(query)
        vector_norms = np.linalg.norm(self.vectors, axis=1)
        scores = (self.vectors @ query) / (vector_norms * query_norm)
        order = np.argsort(-scores, kind="stable")[:k]
        return [(int(local_id), float(scores[local_id])) for local_id in order]


def test_segmented_hnsw_merges_cosine_results_by_descending_similarity():
    from src.index.segmented_hnsw import SegmentedHNSWIndex

    vectors = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.8, 0.2],
            [0.6, 0.4],
        ],
        dtype=np.float32,
    )

    index = SegmentedHNSWIndex(
        metric="cosine",
        segment_count=2,
        build_threads=1,
        segment_factory=FakeCosineSegment,
    )
    index.build(vectors)

    results = index.search(np.array([1.0, 0.0], dtype=np.float32), k=3)
    assert [vector_id for vector_id, _score in results] == [0, 2, 3]


class RecordingSegment(FakeExactSegment):
    created_ml_values = []

    def __init__(self, M=16, ef_construction=200, ef_search=50, ml=None, metric="euclidean"):
        super().__init__(
            M=M,
            ef_construction=ef_construction,
            ef_search=ef_search,
            ml=ml,
            metric=metric,
        )
        self.created_ml_values.append(ml)


def test_segmented_hnsw_forwards_ml_to_segment_factory():
    from src.index.segmented_hnsw import SegmentedHNSWIndex

    RecordingSegment.created_ml_values = []
    vectors = np.arange(12, dtype=np.float32).reshape(6, 2)

    index = SegmentedHNSWIndex(
        ml=0.75,
        segment_count=2,
        build_threads=1,
        segment_factory=RecordingSegment,
    )
    index.build(vectors)

    assert RecordingSegment.created_ml_values == [0.75, 0.75]


class FakeMaterializedSegment(FakeExactSegment):
    @property
    def graph_storage_mode(self):
        return "materialized_python"


def test_segmented_hnsw_reports_mixed_graph_storage_when_segments_are_not_all_compact():
    from src.index.segmented_hnsw import SegmentedHNSWIndex

    vectors = np.arange(12, dtype=np.float32).reshape(6, 2)
    index = SegmentedHNSWIndex(segment_count=2, segment_factory=FakeMaterializedSegment)
    index.build(vectors)

    assert index.graph_storage_mode == "segmented_mixed"


class FakePythonGraphSegment(FakeExactSegment):
    @property
    def graph_storage_mode(self):
        return "materialized_python"

    def build(self, vectors):
        super().build(vectors)
        self.nodes = {
            0: type("Node", (), {"connections": {0: {1}}})(),
            1: type("Node", (), {"connections": {0: {0}}})(),
        }
        self._python_graph_materialized = True
        self._cpp_graph_cache = None


def test_segmented_hnsw_memory_estimate_counts_python_materialized_segments():
    from src.index.segmented_hnsw import SegmentedHNSWIndex

    vectors = np.arange(8, dtype=np.float32).reshape(4, 2)
    index = SegmentedHNSWIndex(segment_count=2, segment_factory=FakePythonGraphSegment)
    index.build(vectors)

    memory = index.estimate_graph_memory()

    assert memory["python_graph_materialized"] is True
    assert memory["python_layers"] == 2
    assert memory["python_edges"] == 4
    assert memory["cpp_edges"] == 0
    assert memory["total_edges_counted"] == 4


def test_segmented_hnsw_rejects_invalid_segment_settings():
    from src.index.segmented_hnsw import SegmentedHNSWIndex

    with pytest.raises(ValueError, match="segment_count must be positive"):
        SegmentedHNSWIndex(segment_count=0)

    with pytest.raises(ValueError, match="build_threads must be positive"):
        SegmentedHNSWIndex(segment_count=2, build_threads=0)
