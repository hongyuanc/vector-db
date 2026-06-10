from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from src.index.hnsw import HNSWIndex


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
        segment_factory: Callable[..., HNSWIndex] = _default_segment_factory,
    ):
        if segment_count <= 0:
            raise ValueError("segment_count must be positive")
        if build_threads <= 0:
            raise ValueError("build_threads must be positive")
        if metric not in {"euclidean", "cosine"}:
            raise ValueError(f"Unsupported metric: {metric}. Use 'euclidean' or 'cosine'")

        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.ml = ml
        self.metric = metric
        self.segment_count = segment_count
        self.build_threads = build_threads
        self._segment_factory = segment_factory

        self.vectors: np.ndarray | None = None
        self.segments: list[HNSWSegment] = []
        self.segment_offsets: list[int] = []
        self.segment_sizes: list[int] = []
        self.segmented_build_stats: dict | None = None
        self._last_cpp_build_stats: dict | None = None

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

        if self.vectors.ndim != 2:
            raise ValueError("vectors must be a 2D array")
        if len(self.vectors) == 0:
            return

        ranges = self._segment_ranges(len(self.vectors), self.segment_count)
        build_start = time.perf_counter()
        self.segments = [self._build_one_segment(start, end) for start, end in ranges]
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
        if k <= 0 or not self.segments:
            return []
        candidates: list[tuple[int, float]] = []
        per_segment_k = max(k, 1)
        for segment in self.segments:
            for local_id, distance in segment.index.search(query, k=per_segment_k, ef=ef):
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
        query_array = np.asarray(queries, dtype=np.float32)
        if query_array.ndim != 2:
            raise ValueError("queries must be a 2D array shaped (n_queries, dimension)")
        return [self.search(query, k=k, ef=ef) for query in query_array]

    def estimate_graph_memory(self) -> dict:
        python_nodes = sum(len(segment.index.nodes) for segment in self.segments)
        cpp_edges = 0
        cpp_bytes = 0
        for segment in self.segments:
            graph_cache = segment.index._cpp_graph_cache or {}
            for layer in graph_cache.values():
                offsets, neighbors = layer
                cpp_edges += int(len(neighbors))
                cpp_bytes += int(offsets.nbytes + neighbors.nbytes)
        python_bytes = python_nodes * 128
        total_bytes = python_bytes + cpp_bytes
        bytes_per_mib = 1024 * 1024
        return {
            "python_nodes": python_nodes,
            "python_graph_materialized": False,
            "python_layers": 0,
            "python_edges": 0,
            "python_graph_mb": round(python_bytes / bytes_per_mib, 4),
            "cpp_layers": sum(len(segment.index._cpp_graph_cache or {}) for segment in self.segments),
            "cpp_edges": cpp_edges,
            "cpp_graph_mb": round(cpp_bytes / bytes_per_mib, 4),
            "total_edges_counted": cpp_edges,
            "total_graph_mb": round(total_bytes / bytes_per_mib, 4),
        }

    def _build_one_segment(self, start: int, end: int) -> HNSWSegment:
        segment_index = self._segment_factory(
            M=self.M,
            ef_construction=self.ef_construction,
            ef_search=self.ef_search,
            ml=self.ml,
            metric=self.metric,
        )
        build_start = time.perf_counter()
        segment_index.build(self.vectors[start:end])
        build_seconds = time.perf_counter() - build_start
        return HNSWSegment(start=start, end=end, index=segment_index, build_seconds=build_seconds)

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
