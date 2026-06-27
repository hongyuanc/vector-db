from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable

import numpy as np

from src.index.hnsw import CPP_AVAILABLE, HNSWIndex, hnsw_cpp, sample_hnsw_levels


CPP_SUM_KEYS = {
    "vectors",
    "directed_edges",
    "total_seconds",
    "construction_seconds",
    "search_seconds",
    "greedy_search_seconds",
    "candidate_search_seconds",
    "prune_seconds",
    "csr_export_seconds",
    "search_calls",
    "greedy_search_calls",
    "candidate_search_calls",
    "visited_resizes",
    "search_heap_resizes",
    "adjacency_layers_allocated",
    "distance_evaluations",
    "search_distance_evaluations",
    "neighbor_selection_distance_evaluations",
    "prune_distance_evaluations",
    "visited_nodes",
    "candidate_heap_pushes",
    "result_heap_pushes",
    "neighbor_selection_calls",
    "selected_degree_total",
    "prune_calls",
    "prune_input_total",
}

CPP_MAX_KEYS = {
    "max_layer",
    "max_observed_degree",
    "max_visited_nodes_per_search",
    "max_selected_degree",
    "max_prune_input_size",
}

CPP_BOOL_ALL_KEYS = {
    "uses_squared_l2",
    "uses_float_l2_accumulation",
    "uses_reusable_search_heaps",
    "uses_bounded_adjacency",
    "uses_heuristic_neighbors",
}


@dataclass
class HNSWSegment:
    start: int
    end: int
    index: HNSWIndex
    build_seconds: float

    @property
    def size(self) -> int:
        return self.end - self.start


def _default_segment_factory(
    M: int,
    ef_construction: int,
    ef_search: int,
    ml: float,
    metric: str,
) -> HNSWIndex:
    return HNSWIndex(
        M=M,
        ef_construction=ef_construction,
        ef_search=ef_search,
        ml=ml,
        metric=metric,
    )


class SegmentedHNSWIndex:
    """Opt-in segmented HNSW index built from independent HNSWIndex segments."""

    def __init__(
        self,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        ml: float = 1.0 / np.log(2.0),
        metric: str = "euclidean",
        segment_count: int = 2,
        build_threads: int = 1,
        segment_search_k: int | None = None,
        segment_factory: Callable[..., HNSWIndex] = _default_segment_factory,
        executor_factory=ThreadPoolExecutor,
    ):
        if segment_count <= 0:
            raise ValueError("segment_count must be positive")
        if build_threads <= 0:
            raise ValueError("build_threads must be positive")
        if segment_search_k is not None and segment_search_k <= 0:
            raise ValueError("segment_search_k must be positive")
        if metric not in {"euclidean", "cosine"}:
            raise ValueError(f"Unsupported metric: {metric}. Use 'euclidean' or 'cosine'")

        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.ml = ml
        self.metric = metric
        self.segment_count = segment_count
        self.build_threads = build_threads
        self.segment_search_k = segment_search_k
        self._segment_factory = segment_factory
        self._executor_factory = executor_factory

        self.vectors: np.ndarray | None = None
        self.segments: list[HNSWSegment] = []
        self.segment_offsets: list[int] = []
        self.segment_sizes: list[int] = []
        self.segmented_build_stats: dict | None = None
        self._last_cpp_build_stats: dict | None = None
        self._last_search_used_native_segmented = False

    @property
    def graph_storage_mode(self) -> str:
        if not self.segments:
            return "empty"
        if all(segment.index.graph_storage_mode == "compact_csr" for segment in self.segments):
            return "segmented_csr"
        return "segmented_mixed"

    def build(self, vectors: np.ndarray) -> None:
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.segments = []
        self.segment_offsets = []
        self.segment_sizes = []
        self.segmented_build_stats = None
        self._last_cpp_build_stats = None
        self._last_search_used_native_segmented = False

        if self.vectors.ndim != 2:
            raise ValueError("vectors must be a 2D array")
        if len(self.vectors) == 0:
            return

        ranges = self._segment_ranges(len(self.vectors), self.segment_count)
        build_start = time.perf_counter()
        levels = sample_hnsw_levels(len(self.vectors), self.ml)
        work_items = [(start, end, levels[start:end]) for start, end in ranges]
        if self.build_threads > 1 and len(work_items) > 1:
            max_workers = min(self.build_threads, len(work_items))
            with self._executor_factory(max_workers=max_workers) as executor:
                self.segments = list(executor.map(self._build_one_segment_from_item, work_items))
        else:
            self.segments = [self._build_one_segment_from_item(item) for item in work_items]
        build_seconds = time.perf_counter() - build_start

        self.segment_offsets = [segment.start for segment in self.segments]
        self.segment_sizes = [segment.size for segment in self.segments]
        segment_seconds = [segment.build_seconds for segment in self.segments]
        self._last_cpp_build_stats = self._aggregate_cpp_stats()
        self.segmented_build_stats = {
            "uses_segmented_build": True,
            "segment_count": len(self.segments),
            "build_threads": self.build_threads,
            "segment_offsets": self.segment_offsets,
            "segment_sizes": self.segment_sizes,
            "segment_build_seconds": segment_seconds,
            "total_wall_seconds": build_seconds,
            "sum_segment_build_seconds": sum(segment_seconds),
            "max_segment_build_seconds": max(segment_seconds) if segment_seconds else 0.0,
        }

    def search(
        self,
        query: np.ndarray,
        k: int,
        ef: int | None = None,
    ) -> list[tuple[int, float]]:
        self._last_search_used_native_segmented = False
        if k <= 0 or not self.segments:
            return []
        if ef is None:
            ef = self.ef_search
        ef = max(ef, k)

        query_array = np.asarray(query, dtype=np.float32)
        if query_array.ndim != 1:
            raise ValueError("query must be a 1D array")

        native_results = self._search_batch_native(
            np.ascontiguousarray(query_array.reshape(1, -1), dtype=np.float32),
            k=k,
            ef=ef,
        )
        if native_results is not None:
            return native_results[0]

        candidates: list[tuple[int, float]] = []
        per_segment_k = self._effective_segment_search_k(k)
        for segment in self.segments:
            for local_id, distance in segment.index.search(query_array, k=per_segment_k, ef=ef):
                candidates.append((segment.start + int(local_id), float(distance)))
        if self.metric == "cosine":
            candidates.sort(key=lambda item: (-item[1], item[0]))
        else:
            candidates.sort(key=lambda item: (item[1], item[0]))
        return candidates[:k]

    def search_batch(
        self,
        queries: np.ndarray,
        k: int,
        ef: int | None = None,
    ) -> list[list[tuple[int, float]]]:
        self._last_search_used_native_segmented = False
        query_array = np.asarray(queries, dtype=np.float32)
        if query_array.ndim != 2:
            raise ValueError("queries must be a 2D array shaped (n_queries, dimension)")
        if query_array.shape[0] == 0:
            return []
        if k <= 0:
            return [[] for _query in range(query_array.shape[0])]
        if not self.segments:
            return [[] for _query in range(query_array.shape[0])]
        if ef is None:
            ef = self.ef_search
        ef = max(ef, k)

        native_results = self._search_batch_native(
            np.ascontiguousarray(query_array, dtype=np.float32),
            k=k,
            ef=ef,
        )
        if native_results is not None:
            return native_results

        return [self.search(query, k=k, ef=ef) for query in query_array]

    def estimate_graph_memory(self) -> dict:
        python_nodes = 0
        python_graph_materialized = False
        python_layer_count = 0
        python_edge_count = 0
        cpp_layer_count = 0
        cpp_edges = 0
        cpp_bytes = 0
        for segment in self.segments:
            segment_index = segment.index
            nodes = getattr(segment_index, "nodes", {}) or {}
            python_nodes += len(nodes)
            segment_python_layers = set()
            for node in nodes.values():
                connections = getattr(node, "connections", {}) or {}
                for layer, neighbors in connections.items():
                    segment_python_layers.add(layer)
                    python_edge_count += len(neighbors)
            python_layer_count += len(segment_python_layers)
            python_graph_materialized = (
                python_graph_materialized
                or bool(getattr(segment_index, "_python_graph_materialized", False))
                or python_edge_count > 0
            )

            graph_cache = getattr(segment_index, "_cpp_graph_cache", None) or {}
            for layer in graph_cache.values():
                offsets, neighbors = layer
                cpp_layer_count += 1
                cpp_edges += int(len(neighbors))
                cpp_bytes += int(offsets.nbytes + neighbors.nbytes)
        python_bytes = python_nodes * 128 + python_edge_count * 72
        total_bytes = python_bytes + cpp_bytes
        bytes_per_mib = 1024 * 1024
        return {
            "python_nodes": python_nodes,
            "python_graph_materialized": python_graph_materialized,
            "python_layers": python_layer_count,
            "python_edges": python_edge_count,
            "python_graph_mb": round(python_bytes / bytes_per_mib, 4),
            "cpp_layers": cpp_layer_count,
            "cpp_edges": cpp_edges,
            "cpp_graph_mb": round(cpp_bytes / bytes_per_mib, 4),
            "total_edges_counted": python_edge_count + cpp_edges,
            "total_graph_mb": round(total_bytes / bytes_per_mib, 4),
        }

    def _build_one_segment_from_item(self, item: tuple[int, int, np.ndarray]) -> HNSWSegment:
        start, end, levels = item
        segment_index = self._segment_factory(
            M=self.M,
            ef_construction=self.ef_construction,
            ef_search=self.ef_search,
            ml=self.ml,
            metric=self.metric,
        )
        build_start = time.perf_counter()
        if isinstance(segment_index, HNSWIndex):
            segment_index.build(self.vectors[start:end], levels=levels)
        else:
            segment_index.build(self.vectors[start:end])
        build_seconds = time.perf_counter() - build_start
        return HNSWSegment(start=start, end=end, index=segment_index, build_seconds=build_seconds)

    def _effective_segment_search_k(self, k: int) -> int:
        configured = self.segment_search_k if self.segment_search_k is not None else k
        return max(k, configured)

    def _native_segment_descriptors(self) -> list[dict] | None:
        if (
            not CPP_AVAILABLE
            or hnsw_cpp is None
            or not hasattr(hnsw_cpp, "search_segmented_batch")
            or not self.segments
        ):
            return None

        descriptors = []
        for segment in self.segments:
            segment_index = segment.index
            if getattr(segment_index, "graph_storage_mode", None) != "compact_csr":
                return None

            vectors = getattr(segment_index, "_vectors_f32_cache", None)
            graph_cache = getattr(segment_index, "_cpp_graph_cache", None)
            entry_point = getattr(segment_index, "entry_point", None)
            max_layer = getattr(segment_index, "max_layer", None)
            if vectors is None or graph_cache is None or entry_point is None or max_layer is None:
                return None
            if not all(layer in graph_cache for layer in range(int(max_layer) + 1)):
                return None

            descriptors.append(
                {
                    "vectors": np.ascontiguousarray(vectors, dtype=np.float32),
                    "layers": graph_cache,
                    "entry_point": int(entry_point),
                    "max_layer": int(max_layer),
                    "global_offset": int(segment.start),
                }
            )

        return descriptors

    def _search_batch_native(
        self,
        query_array: np.ndarray,
        k: int,
        ef: int,
    ) -> list[list[tuple[int, float]]] | None:
        segments = self._native_segment_descriptors()
        if segments is None:
            return None

        raw_results = hnsw_cpp.search_segmented_batch(
            queries=query_array,
            segments=segments,
            k=k,
            ef=ef,
            segment_search_k=self._effective_segment_search_k(k),
            metric=self.metric,
        )
        self._last_search_used_native_segmented = True
        return [
            [(int(vector_id), float(distance)) for vector_id, distance in query_results]
            for query_results in raw_results
        ]

    @staticmethod
    def _segment_ranges(total: int, segment_count: int) -> list[tuple[int, int]]:
        actual_segments = min(total, segment_count)
        base_size = total // actual_segments
        remainder = total % actual_segments
        ranges = []
        start = 0
        for segment_id in range(actual_segments):
            size = base_size + (1 if segment_id < remainder else 0)
            end = start + size
            ranges.append((start, end))
            start = end
        return ranges

    def _aggregate_cpp_stats(self) -> dict:
        stats_by_segment = [
            segment.index._last_cpp_build_stats
            for segment in self.segments
            if segment.index._last_cpp_build_stats
        ]
        if not stats_by_segment:
            return {}

        aggregate: dict = {
            "dimensions": int(self.vectors.shape[1]) if self.vectors is not None else None,
            "average_selected_degree": 0.0,
            "average_prune_input_size": 0.0,
            "uses_heuristic_reverse_pruning": any(
                bool(stats.get("uses_heuristic_reverse_pruning")) for stats in stats_by_segment
            ),
        }
        for key in CPP_SUM_KEYS:
            aggregate[key] = sum(stats.get(key, 0) or 0 for stats in stats_by_segment)
        for key in CPP_MAX_KEYS:
            aggregate[key] = max(stats.get(key, 0) or 0 for stats in stats_by_segment)
        for key in CPP_BOOL_ALL_KEYS:
            aggregate[key] = all(bool(stats.get(key)) for stats in stats_by_segment)

        selected_calls = aggregate.get("neighbor_selection_calls", 0) or 0
        selected_total = aggregate.get("selected_degree_total", 0) or 0
        aggregate["average_selected_degree"] = (
            float(selected_total) / float(selected_calls) if selected_calls else 0.0
        )
        prune_calls = aggregate.get("prune_calls", 0) or 0
        prune_input_total = aggregate.get("prune_input_total", 0) or 0
        aggregate["average_prune_input_size"] = (
            float(prune_input_total) / float(prune_calls) if prune_calls else 0.0
        )
        return aggregate
