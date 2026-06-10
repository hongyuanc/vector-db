# HNSW Segmented Parallel Build Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in segmented HNSW build mode that can build independent native HNSW segments in parallel and merge search results globally.

**Architecture:** Keep `HNSWIndex` as the single-graph baseline. Add a `SegmentedHNSWIndex` wrapper that owns multiple `HNSWIndex` segment instances, maps local segment ids to global vector ids, fans out search across segments, and aggregates stats. Benchmark support chooses segmented mode only when `segment_count > 1`, so existing build/search behavior remains unchanged by default.

**Tech Stack:** Python 3.11, Cython, C++ HNSW extension, NumPy, `concurrent.futures.ThreadPoolExecutor`, pytest, existing benchmark CLI.

---

## File Structure

- Create `src/index/segmented_hnsw.py`
  - Owns the opt-in segmented index wrapper.
  - Splits vectors into contiguous segments.
  - Builds one `HNSWIndex` per segment.
  - Converts local segment ids back to global ids.
  - Aggregates per-segment native build stats.

- Modify `src/index/__init__.py`
  - Export `SegmentedHNSWIndex` for benchmark and test imports.

- Modify `src/index/hnsw_cpp.pyx`
  - Release the GIL around the native `cpp_build_graph()` call so Python build threads can overlap actual C++ work.

- Modify `benchmarks/benchmark.py`
  - Add `segment_count` and `build_threads` benchmark parameters.
  - Use `SegmentedHNSWIndex` only when `segment_count > 1`.
  - Add `metrics.segmented_build_stats` to JSON output.
  - Add segmented build rows to Markdown output.
  - Skip compact/materialized persistence benchmarking for segmented indexes until segmented persistence is explicitly designed.

- Create `tests/test_segmented_hnsw.py`
  - Owns segmented wrapper tests using deterministic fake segments.

- Modify `tests/test_hnsw_cpp.py`
  - Add a source-level contract test that the Cython wrapper releases the GIL for native build calls.

- Modify `tests/test_benchmark_cli.py`
  - Add benchmark schema coverage for segmented mode and CLI options.

- Modify `TECHNICAL.md`
  - Record what changed, why segmented build was chosen over same-graph parallel insertion, benchmark results, and next steps.

---

### Task 1: Specify Segmented Index Behavior

**Files:**
- Create: `tests/test_segmented_hnsw.py`
- Test: `tests/test_segmented_hnsw.py`

- [ ] **Step 1: Add tests for contiguous split, global-id merge, and stats shape**

Create `tests/test_segmented_hnsw.py` with this content:

```python
import numpy as np
import pytest


class FakeExactSegment:
    def __init__(self, M=16, ef_construction=200, ef_search=50, metric="euclidean"):
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
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


def test_segmented_hnsw_rejects_invalid_segment_settings():
    from src.index.segmented_hnsw import SegmentedHNSWIndex

    with pytest.raises(ValueError, match="segment_count must be positive"):
        SegmentedHNSWIndex(segment_count=0)

    with pytest.raises(ValueError, match="build_threads must be positive"):
        SegmentedHNSWIndex(segment_count=2, build_threads=0)
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run:

```bash
/Users/hong/projects/personal/vector-db/venv/bin/python -m pytest tests/test_segmented_hnsw.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.index.segmented_hnsw'`.

- [ ] **Step 3: Commit the failing segmented-wrapper tests**

```bash
git add tests/test_segmented_hnsw.py
git commit -m "test: specify segmented hnsw wrapper behavior"
```

---

### Task 2: Implement Sequential Segmented HNSW Wrapper

**Files:**
- Create: `src/index/segmented_hnsw.py`
- Modify: `src/index/__init__.py`
- Test: `tests/test_segmented_hnsw.py`

- [ ] **Step 1: Create the sequential segmented wrapper**

Create `src/index/segmented_hnsw.py`:

```python
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
    metric: str,
) -> HNSWIndex:
    return HNSWIndex(
        M=M,
        ef_construction=ef_construction,
        ef_search=ef_search,
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
        return "segmented_csr"

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
```

- [ ] **Step 2: Export the wrapper from `src/index/__init__.py`**

Add this import to `src/index/__init__.py`:

```python
from src.index.segmented_hnsw import SegmentedHNSWIndex
```

- [ ] **Step 3: Run the segmented wrapper tests**

Run:

```bash
/Users/hong/projects/personal/vector-db/venv/bin/python -m pytest tests/test_segmented_hnsw.py -q
```

Expected: PASS.

- [ ] **Step 4: Run the focused HNSW suite**

Run:

```bash
/Users/hong/projects/personal/vector-db/venv/bin/python -m pytest tests/test_segmented_hnsw.py tests/test_hnsw_cpp.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit the sequential segmented wrapper**

```bash
git add src/index/segmented_hnsw.py src/index/__init__.py tests/test_segmented_hnsw.py
git commit -m "feat: add segmented hnsw wrapper"
```

---

### Task 3: Add Segmented Benchmark Schema and CLI

**Files:**
- Modify: `benchmarks/benchmark.py`
- Modify: `tests/test_benchmark_cli.py`
- Test: `tests/test_benchmark_cli.py`

- [ ] **Step 1: Add a failing benchmark schema test for segmented mode**

Append this test to `tests/test_benchmark_cli.py`:

```python
def test_run_benchmark_suite_reports_segmented_build_stats():
    benchmark = import_module("benchmarks.benchmark")

    result = benchmark.run_benchmark_suite(
        dataset="random",
        size=90,
        dimension=8,
        n_queries=6,
        k=3,
        metric="euclidean",
        M=4,
        ef_construction=20,
        ef_search=10,
        seed=123,
        warmup_queries=1,
        segment_count=3,
        build_threads=1,
    )

    assert result["config"]["hnsw"]["segment_count"] == 3
    assert result["config"]["hnsw"]["build_threads"] == 1

    segmented = result["metrics"]["segmented_build_stats"]
    assert segmented["uses_segmented_build"] is True
    assert segmented["segment_count"] == 3
    assert segmented["build_threads"] == 1
    assert segmented["segment_sizes"] == [30, 30, 30]
    assert len(segmented["segment_build_seconds"]) == 3
    assert segmented["max_segment_build_seconds"] >= 0.0

    cpp_build_stats = result["metrics"]["cpp_build_stats"]
    assert cpp_build_stats["vectors"] == 90
    assert cpp_build_stats["candidate_search_seconds"] >= 0.0
    assert result["metrics"]["persistence"]["compact"]["available"] is False
    assert result["metrics"]["persistence"]["materialized"]["available"] is False
```

- [ ] **Step 2: Add a failing CLI test for segmented options and Markdown rows**

Append this test to `tests/test_benchmark_cli.py`:

```python
def test_main_writes_segmented_markdown_rows(tmp_path):
    benchmark = import_module("benchmarks.benchmark")
    json_path = tmp_path / "segmented.json"
    markdown_path = tmp_path / "segmented.md"

    exit_code = benchmark.main(
        [
            "--dataset",
            "random",
            "--size",
            "90",
            "--dimension",
            "8",
            "--queries",
            "6",
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
            "--segments",
            "3",
            "--build-threads",
            "1",
            "--output",
            str(json_path),
            "--markdown-output",
            str(markdown_path),
        ]
    )

    assert exit_code == 0
    data = json.loads(json_path.read_text())
    assert data["metrics"]["segmented_build_stats"]["segment_count"] == 3

    markdown = markdown_path.read_text()
    assert "Segmented Build" in markdown
    assert "Segment Count" in markdown
    assert "Build Threads" in markdown
    assert "Max Segment Build" in markdown
```

- [ ] **Step 3: Run the new benchmark tests to verify they fail**

Run:

```bash
/Users/hong/projects/personal/vector-db/venv/bin/python -m pytest tests/test_benchmark_cli.py::test_run_benchmark_suite_reports_segmented_build_stats tests/test_benchmark_cli.py::test_main_writes_segmented_markdown_rows -q
```

Expected: FAIL with `TypeError: run_benchmark_suite() got an unexpected keyword argument 'segment_count'`.

- [ ] **Step 4: Import `SegmentedHNSWIndex` in `benchmarks/benchmark.py`**

Near the existing `HNSWIndex = _hnsw_module.HNSWIndex` line, add:

```python
SegmentedHNSWIndex = importlib.import_module("src.index.segmented_hnsw").SegmentedHNSWIndex
```

- [ ] **Step 5: Add segmented parameters to `run_benchmark_suite()`**

Change the function signature to include:

```python
    segment_count: int = 1,
    build_threads: int = 1,
```

After the existing `k <= 0` validation, add:

```python
    if segment_count <= 0:
        raise ValueError("segment_count must be positive")
    if build_threads <= 0:
        raise ValueError("build_threads must be positive")
```

- [ ] **Step 6: Instantiate the correct index type in the benchmark**

Replace the current `index = HNSWIndex(...)` block with:

```python
    if segment_count > 1:
        index = SegmentedHNSWIndex(
            M=M,
            ef_construction=ef_construction,
            ef_search=ef_search,
            metric=metric,
            segment_count=segment_count,
            build_threads=build_threads,
        )
    else:
        index = HNSWIndex(
            M=M,
            ef_construction=ef_construction,
            ef_search=ef_search,
            metric=metric,
        )
```

- [ ] **Step 7: Use segmented memory and persistence behavior**

Replace:

```python
    graph_memory = _estimate_graph_memory(index)
```

with:

```python
    if hasattr(index, "estimate_graph_memory"):
        graph_memory = index.estimate_graph_memory()
    else:
        graph_memory = _estimate_graph_memory(index)
```

Replace:

```python
    persistence = _benchmark_persistence(index)
```

with:

```python
    if segment_count > 1:
        persistence = {
            "compact": _unavailable_persistence_result(),
            "materialized": _unavailable_persistence_result(),
        }
    else:
        persistence = _benchmark_persistence(index)
```

- [ ] **Step 8: Add segmented config and metrics to the result**

In the `"hnsw"` config dictionary, add:

```python
                "segment_count": segment_count,
                "build_threads": build_threads,
```

In the `"metrics"` dictionary, add:

```python
            "segmented_build_stats": getattr(index, "segmented_build_stats", None),
```

- [ ] **Step 9: Add Markdown rows for segmented builds**

In `format_markdown_report()`, after `cpp_build_stats = metrics["cpp_build_stats"]`, add:

```python
    segmented_build_stats = metrics.get("segmented_build_stats")
```

Add these rows after the `C++ Max Prune Input Size` row:

```python
            *(
                [
                    f"| Segmented Build | {segmented_build_stats['uses_segmented_build']} |",
                    f"| Segment Count | {segmented_build_stats['segment_count']} |",
                    f"| Build Threads | {segmented_build_stats['build_threads']} |",
                    f"| Max Segment Build | {segmented_build_stats['max_segment_build_seconds']:.6f}s |",
                    f"| Sum Segment Build | {segmented_build_stats['sum_segment_build_seconds']:.6f}s |",
                ]
                if segmented_build_stats
                else []
            ),
```

- [ ] **Step 10: Add CLI options**

In `_build_parser()`, add:

```python
    parser.add_argument("--segments", type=int, default=1, help="Number of independent HNSW build segments")
    parser.add_argument("--build-threads", type=int, default=1, help="Parallel segment build worker count")
```

In the `run_benchmark_suite()` call inside `main()`, pass:

```python
        segment_count=args.segments,
        build_threads=args.build_threads,
```

- [ ] **Step 11: Update the existing benchmark config assertion**

In `tests/test_benchmark_cli.py`, update the existing `result["config"]["hnsw"]`
assertion in `test_run_benchmark_suite_returns_structured_metrics()` from:

```python
    assert result["config"]["hnsw"] == {
        "M": 4,
        "ef_construction": 20,
        "ef_search": 10,
        "metric": "euclidean",
    }
```

to:

```python
    assert result["config"]["hnsw"] == {
        "M": 4,
        "ef_construction": 20,
        "ef_search": 10,
        "metric": "euclidean",
        "segment_count": 1,
        "build_threads": 1,
    }
```

- [ ] **Step 12: Run benchmark schema tests**

Run:

```bash
/Users/hong/projects/personal/vector-db/venv/bin/python -m pytest tests/test_benchmark_cli.py -q
```

Expected: PASS.

- [ ] **Step 13: Commit benchmark segmented mode**

```bash
git add benchmarks/benchmark.py tests/test_benchmark_cli.py
git commit -m "feat: benchmark segmented hnsw builds"
```

---

### Task 4: Add Native Build Parallelism

**Files:**
- Modify: `src/index/segmented_hnsw.py`
- Modify: `src/index/hnsw_cpp.pyx`
- Modify: `tests/test_segmented_hnsw.py`
- Modify: `tests/test_hnsw_cpp.py`
- Test: `tests/test_segmented_hnsw.py`
- Test: `tests/test_hnsw_cpp.py`

- [ ] **Step 1: Add a failing test for the configured build-thread path**

Append this test to `tests/test_segmented_hnsw.py`:

```python
def test_segmented_hnsw_uses_configured_build_thread_executor():
    from src.index.segmented_hnsw import SegmentedHNSWIndex

    created_workers = []

    class RecordingExecutor:
        def __init__(self, max_workers):
            created_workers.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def map(self, func, items):
            return [func(item) for item in items]

    vectors = np.arange(24, dtype=np.float32).reshape(12, 2)
    index = SegmentedHNSWIndex(
        M=2,
        ef_construction=8,
        ef_search=6,
        metric="euclidean",
        segment_count=4,
        build_threads=3,
        segment_factory=FakeExactSegment,
        executor_factory=RecordingExecutor,
    )

    index.build(vectors)

    assert created_workers == [3]
    assert index.segmented_build_stats["build_threads"] == 3
    assert index.segment_sizes == [3, 3, 3, 3]
```

- [ ] **Step 2: Add a failing Cython GIL-release source contract test**

Append this test to `tests/test_hnsw_cpp.py`:

```python
from pathlib import Path


def test_cpp_build_graph_wrapper_releases_gil_for_parallel_segments():
    source = Path("src/index/hnsw_cpp.pyx").read_text()
    assert "CppBuildGraphResult cpp_build_graph" in source
    assert "cpp_build_graph" in source and "nogil" in source
    assert "with nogil:" in source
```

- [ ] **Step 3: Run the new tests to verify they fail**

Run:

```bash
/Users/hong/projects/personal/vector-db/venv/bin/python -m pytest tests/test_segmented_hnsw.py::test_segmented_hnsw_uses_configured_build_thread_executor tests/test_hnsw_cpp.py::test_cpp_build_graph_wrapper_releases_gil_for_parallel_segments -q
```

Expected: FAIL because `SegmentedHNSWIndex` does not accept `executor_factory` and the Cython wrapper does not use `with nogil:`.

- [ ] **Step 4: Add executor support to `SegmentedHNSWIndex`**

In `src/index/segmented_hnsw.py`, add the import:

```python
from concurrent.futures import ThreadPoolExecutor
```

Change the constructor signature to include:

```python
        executor_factory=ThreadPoolExecutor,
```

Store it in the constructor:

```python
        self._executor_factory = executor_factory
```

Replace the sequential segment build line:

```python
        self.segments = [self._build_one_segment(start, end) for start, end in ranges]
```

with:

```python
        work_items = list(ranges)
        if self.build_threads > 1 and len(work_items) > 1:
            max_workers = min(self.build_threads, len(work_items))
            with self._executor_factory(max_workers=max_workers) as executor:
                self.segments = list(executor.map(self._build_one_segment_from_item, work_items))
        else:
            self.segments = [self._build_one_segment_from_item(item) for item in work_items]
```

Rename `_build_one_segment(self, start: int, end: int)` to:

```python
    def _build_one_segment_from_item(self, item: tuple[int, int]) -> HNSWSegment:
        start, end = item
```

Keep the rest of the method body the same.

- [ ] **Step 5: Release the GIL around native build in `src/index/hnsw_cpp.pyx`**

In the Cython extern declaration for `cpp_build_graph`, add `nogil`:

```cython
    CppBuildGraphResult cpp_build_graph "vectordb::build_graph"(
        const float* vectors,
        int n_vectors,
        int dimension,
        const int* levels,
        int n_levels,
        int max_connections,
        int ef_construction,
        const string& metric,
        bool include_connections
    ) nogil
```

In `def build_graph(...)`, replace:

```cython
    cdef CppBuildGraphResult raw = cpp_build_graph(
        <const float*> vectors_arr.data,
        <int> vectors_arr.shape[0],
        <int> vectors_arr.shape[1],
        <const int*> levels_arr.data,
        <int> levels_arr.shape[0],
        max_connections,
        ef_construction,
        metric_cpp,
        include_connections,
    )
```

with:

```cython
    cdef CppBuildGraphResult raw
    with nogil:
        raw = cpp_build_graph(
            <const float*> vectors_arr.data,
            <int> vectors_arr.shape[0],
            <int> vectors_arr.shape[1],
            <const int*> levels_arr.data,
            <int> levels_arr.shape[0],
            max_connections,
            ef_construction,
            metric_cpp,
            include_connections,
        )
```

- [ ] **Step 6: Rebuild native extensions**

Run:

```bash
/Users/hong/projects/personal/vector-db/venv/bin/python setup.py build_ext --inplace
```

Expected: exit code 0. Existing setuptools deprecation warnings are acceptable.

- [ ] **Step 7: Run parallelism-focused tests**

Run:

```bash
/Users/hong/projects/personal/vector-db/venv/bin/python -m pytest tests/test_segmented_hnsw.py tests/test_hnsw_cpp.py::test_cpp_build_graph_wrapper_releases_gil_for_parallel_segments -q
```

Expected: PASS.

- [ ] **Step 8: Commit native parallel build support**

```bash
git add src/index/segmented_hnsw.py src/index/hnsw_cpp.pyx tests/test_segmented_hnsw.py tests/test_hnsw_cpp.py
git commit -m "feat: parallelize segmented hnsw builds"
```

---

### Task 5: Verify Segmented Benchmarks

**Files:**
- Test-only benchmark outputs: `/tmp/hnsw-segmented-*.json`
- Test-only benchmark outputs: `/tmp/hnsw-segmented-*.md`
- Test: `benchmarks/benchmark.py`

- [ ] **Step 1: Run the focused test suite**

Run:

```bash
/Users/hong/projects/personal/vector-db/venv/bin/python -m pytest tests/test_segmented_hnsw.py tests/test_hnsw_cpp.py tests/test_benchmark_cli.py -q
```

Expected: PASS.

- [ ] **Step 2: Run a small synthetic segmented benchmark**

Run:

```bash
/Users/hong/projects/personal/vector-db/venv/bin/python benchmarks/benchmark.py \
  --dataset random \
  --size 10000 \
  --dimension 128 \
  --queries 100 \
  --k 10 \
  --ef-search 50 \
  --segments 4 \
  --build-threads 4 \
  --output /tmp/hnsw-segmented-10k.json \
  --markdown-output /tmp/hnsw-segmented-10k.md
```

Expected: exit code 0 and Markdown rows for `Segmented Build`, `Segment Count`, `Build Threads`, `Max Segment Build`, and `Sum Segment Build`.

- [ ] **Step 3: Run SIFT1M 100k with 2 segments**

Because this isolated worktree may not contain the downloaded SIFT data, use the original checkout data directory read-only. Run:

```bash
/Users/hong/projects/personal/vector-db/venv/bin/python -c 'import json; from pathlib import Path; import tests.benchmarks.real_datasets as rd; rd.get_data_dir = lambda: Path("/Users/hong/projects/personal/vector-db/tests/benchmarks/data"); from benchmarks.benchmark import run_benchmark_suite, format_markdown_report; result = run_benchmark_suite(dataset="sift1m", size=100000, dimension=128, n_queries=100, k=10, metric="euclidean", M=16, ef_construction=200, ef_search=100, subset="medium", warmup_queries=10, segment_count=2, build_threads=2); Path("/tmp/hnsw-segmented-sift100k-2.json").write_text(json.dumps(result, indent=2) + "\n"); Path("/tmp/hnsw-segmented-sift100k-2.md").write_text(format_markdown_report(result)); print(f"segments=2 build={result[\"metrics\"][\"build_time_seconds\"]:.4f}s recall={result[\"metrics\"][\"recall_at_k\"]:.4f} qps={result[\"metrics\"][\"qps\"]:.2f}")'
```

Expected: exit code 0.

- [ ] **Step 4: Run SIFT1M 100k with 4 segments**

Run:

```bash
/Users/hong/projects/personal/vector-db/venv/bin/python -c 'import json; from pathlib import Path; import tests.benchmarks.real_datasets as rd; rd.get_data_dir = lambda: Path("/Users/hong/projects/personal/vector-db/tests/benchmarks/data"); from benchmarks.benchmark import run_benchmark_suite, format_markdown_report; result = run_benchmark_suite(dataset="sift1m", size=100000, dimension=128, n_queries=100, k=10, metric="euclidean", M=16, ef_construction=200, ef_search=100, subset="medium", warmup_queries=10, segment_count=4, build_threads=4); Path("/tmp/hnsw-segmented-sift100k-4.json").write_text(json.dumps(result, indent=2) + "\n"); Path("/tmp/hnsw-segmented-sift100k-4.md").write_text(format_markdown_report(result)); print(f"segments=4 build={result[\"metrics\"][\"build_time_seconds\"]:.4f}s recall={result[\"metrics\"][\"recall_at_k\"]:.4f} qps={result[\"metrics\"][\"qps\"]:.2f}")'
```

Expected: exit code 0.

- [ ] **Step 5: Decide whether to run 8 segments**

Run the 8-segment benchmark only if the 2- and 4-segment runs keep Recall@10 at or above `0.9860` or show a build-time improvement large enough to justify documenting a recall trade-off.

Command:

```bash
/Users/hong/projects/personal/vector-db/venv/bin/python -c 'import json; from pathlib import Path; import tests.benchmarks.real_datasets as rd; rd.get_data_dir = lambda: Path("/Users/hong/projects/personal/vector-db/tests/benchmarks/data"); from benchmarks.benchmark import run_benchmark_suite, format_markdown_report; result = run_benchmark_suite(dataset="sift1m", size=100000, dimension=128, n_queries=100, k=10, metric="euclidean", M=16, ef_construction=200, ef_search=100, subset="medium", warmup_queries=10, segment_count=8, build_threads=8); Path("/tmp/hnsw-segmented-sift100k-8.json").write_text(json.dumps(result, indent=2) + "\n"); Path("/tmp/hnsw-segmented-sift100k-8.md").write_text(format_markdown_report(result)); print(f"segments=8 build={result[\"metrics\"][\"build_time_seconds\"]:.4f}s recall={result[\"metrics\"][\"recall_at_k\"]:.4f} qps={result[\"metrics\"][\"qps\"]:.2f}")'
```

Expected: exit code 0 if run.

- [ ] **Step 6: Commit if benchmark-only task produced no tracked changes**

If `git status --short` shows no tracked changes, do not create an empty commit. Continue to Task 6.

---

### Task 6: Document Segmented Build Results

**Files:**
- Modify: `TECHNICAL.md`
- Optional create: `benchmarks/results/hnsw-segmented-sift100k-2.json`
- Optional create: `benchmarks/results/hnsw-segmented-sift100k-2.md`
- Optional create: `benchmarks/results/hnsw-segmented-sift100k-4.json`
- Optional create: `benchmarks/results/hnsw-segmented-sift100k-4.md`
- Test: documentation and benchmark artifacts

- [ ] **Step 1: Add the measured result to `TECHNICAL.md`**

In the `Planned Segmented Native Parallel Build` section, add a new subsection named `Measured Prototype Result` with this structure, replacing the numeric values with the benchmark output from Task 5:

```markdown
### Measured Prototype Result

What changed: segmented build is now available as an opt-in benchmark mode. The
default `HNSWIndex.build()` path remains a single global graph. Segmented mode
builds independent compact native graphs and merges per-segment results by
global vector id and distance.

Why it matters: this tests CPU parallelism without introducing same-graph
parallel mutation. The result must be judged as a trade-off: elapsed build time,
recall, query latency, QPS, memory, and segment count all matter.

| Mode | Build Time | Recall@10 | QPS | Candidate Search | Notes |
|---|---:|---:|---:|---:|---|
| Single graph | 40.1198s | 0.9860 | 3378.48 | 30.4112s | Baseline after bounded enqueues |

Add one row for each completed segmented benchmark. Each row must use the exact
numeric values from that run's Markdown or JSON artifact. If the 2-segment run
finishes and the 4-segment run fails, commit only the 2-segment row and describe
the 4-segment failure below the table.

Next step: keep segmented build opt-in until the benchmark shows a build-time
win that justifies the recall and query-latency trade-off. If recall drops,
evaluate per-segment overfetching before changing segment assignment.
```

The final committed text must contain concrete measured numbers for every
completed segmented benchmark row.

- [ ] **Step 2: Preserve successful benchmark artifacts**

If both the 2-segment and 4-segment SIFT1M runs exited 0, copy their `/tmp`
artifacts into `benchmarks/results/`:

```bash
mkdir -p benchmarks/results
cp /tmp/hnsw-segmented-sift100k-2.json benchmarks/results/hnsw-segmented-sift100k-2.json
cp /tmp/hnsw-segmented-sift100k-2.md benchmarks/results/hnsw-segmented-sift100k-2.md
cp /tmp/hnsw-segmented-sift100k-4.json benchmarks/results/hnsw-segmented-sift100k-4.json
cp /tmp/hnsw-segmented-sift100k-4.md benchmarks/results/hnsw-segmented-sift100k-4.md
```

If one of those runs failed, leave artifacts in `/tmp`, commit only
`TECHNICAL.md`, and document the failing command and error output in the final
response.

- [ ] **Step 3: Verify no unfinished measurement markers remain**

Run:

```bash
python - <<'PY'
from pathlib import Path

markers = ["MEASURED" + "_", "T" + "BD", "UNFINISHED" + "_MEASUREMENT"]
paths = [
    Path("TECHNICAL.md"),
    Path("docs/superpowers/plans/2026-06-10-hnsw-segmented-parallel-build.md"),
]
matches = []
for path in paths:
    text = path.read_text()
    for marker in markers:
        if marker in text:
            matches.append(f"{path}: contains {marker}")
if matches:
    raise SystemExit("\n".join(matches))
PY
```

Expected: no output.

- [ ] **Step 4: Run final verification**

Run:

```bash
/Users/hong/projects/personal/vector-db/venv/bin/python setup.py build_ext --inplace
/Users/hong/projects/personal/vector-db/venv/bin/python -m pytest tests/test_segmented_hnsw.py tests/test_hnsw_cpp.py tests/test_benchmark_cli.py -q
git status --short --branch
```

Expected:

- build exits 0
- pytest reports all focused tests passing
- only intentional tracked documentation/artifact changes remain
- untracked `AGENTS.md` remains untouched

- [ ] **Step 5: Commit documentation and selected artifacts**

If only `TECHNICAL.md` changed:

```bash
git add TECHNICAL.md
git commit -m "docs: record segmented hnsw build benchmark"
```

If benchmark artifacts are preserved:

```bash
git add TECHNICAL.md benchmarks/results/hnsw-segmented-sift100k-2.json benchmarks/results/hnsw-segmented-sift100k-2.md benchmarks/results/hnsw-segmented-sift100k-4.json benchmarks/results/hnsw-segmented-sift100k-4.md
git commit -m "docs: record segmented hnsw build benchmark"
```

---

## Final Verification

Run these commands before reporting the milestone complete:

```bash
/Users/hong/projects/personal/vector-db/venv/bin/python setup.py build_ext --inplace
/Users/hong/projects/personal/vector-db/venv/bin/python -m pytest tests/test_segmented_hnsw.py tests/test_hnsw_cpp.py tests/test_benchmark_cli.py -q
/Users/hong/projects/personal/vector-db/venv/bin/python benchmarks/benchmark.py --dataset random --size 10000 --dimension 128 --queries 100 --k 10 --ef-search 50 --segments 4 --build-threads 4 --output /tmp/hnsw-segmented-final-smoke.json --markdown-output /tmp/hnsw-segmented-final-smoke.md
git status --short --branch
```

Expected:

- native extension rebuild exits 0
- focused tests pass
- segmented benchmark exits 0
- Markdown report contains segmented rows
- tracked files are clean after commits
- `AGENTS.md` remains untracked and untouched

## Execution Notes

- Do not replace the current single-graph `HNSWIndex` path.
- Keep segmented build opt-in through benchmark parameters until benchmark evidence says otherwise.
- Do not implement same-graph parallel insertion in this plan.
- If Cython `with nogil` fails to compile around `cpp_build_graph()`, stop and inspect the exact Cython error before changing the architecture. The fallback is native C++ thread pooling inside a later plan, not same-graph insertion.
- If segmented recall drops below `0.9860`, report the drop directly and keep segmented mode documented as a build-speed/recall trade-off.
