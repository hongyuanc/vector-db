# Native Segmented HNSW Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve segmented HNSW query throughput and latency by adding a native C++/Cython multi-segment batch search path with global top-k merge.

**Architecture:** Add a C++ `HnswSegmentView` and `search_segmented_batch()` function that searches multiple compact CSR segment graphs and merges global results natively. Expose it through `hnsw_cpp.pyx`, route `SegmentedHNSWIndex.search_batch()` and `search()` through it when segments are compact CSR, and extend benchmark reporting with `segment_search_k` and native segmented search provenance.

**Tech Stack:** Python 3.11, NumPy, pytest, Cython, C++11, setuptools extension build, Markdown benchmark artifacts.

---

## File Map

- `src/index/hnsw_cpp_core.hpp`: declare native segmented search structs and function.
- `src/index/hnsw_cpp_core.cpp`: implement multi-segment native search and global merge.
- `src/index/hnsw_cpp.pyx`: expose the new native function to Python while keeping NumPy arrays alive for raw pointers.
- `src/index/segmented_hnsw.py`: add `segment_search_k`, compact-native eligibility checks, native `search_batch()`, and single-query delegation.
- `benchmarks/benchmark.py`: expose `--segment-search-k`, use batch search for benchmark QPS, and record native segmented search provenance.
- `tests/test_hnsw_cpp.py`: C++/Cython wrapper tests for native segmented batch search.
- `tests/test_segmented_hnsw.py`: wrapper-level tests for fallback, tuning, and native dispatch.
- `tests/test_benchmark_cli.py`: benchmark schema and Markdown tests.
- `TECHNICAL.md`: document what changed, why, benchmark results, and next steps.
- `README.md`: update segmented benchmark table only after tracked benchmark artifacts are produced.
- `benchmarks/results/*.json` and `benchmarks/results/*.md`: store new benchmark outputs after implementation.

---

### Task 1: C++ Native Segmented Batch Search

**Files:**
- Modify: `src/index/hnsw_cpp_core.hpp`
- Modify: `src/index/hnsw_cpp_core.cpp`
- Test: `tests/test_hnsw_cpp.py`

- [ ] **Step 1: Add failing C++ wrapper-facing tests**

Append these tests to `tests/test_hnsw_cpp.py`:

```python
def test_cpp_search_segmented_batch_merges_global_euclidean_results():
    from src.index import hnsw_cpp

    queries = np.array([[0.05, 0.0], [10.1, 0.0]], dtype=np.float32)
    left_vectors = np.array([[0.0, 0.0], [0.2, 0.0]], dtype=np.float32)
    right_vectors = np.array([[10.0, 0.0], [10.3, 0.0]], dtype=np.float32)
    full_offsets = np.array([0, 1, 2], dtype=np.int32)
    full_neighbors = np.array([1, 0], dtype=np.int32)

    segments = [
        {
            "vectors": left_vectors,
            "layers": {0: (full_offsets, full_neighbors)},
            "entry_point": 0,
            "max_layer": 0,
            "global_offset": 0,
        },
        {
            "vectors": right_vectors,
            "layers": {0: (full_offsets, full_neighbors)},
            "entry_point": 0,
            "max_layer": 0,
            "global_offset": 2,
        },
    ]

    results = hnsw_cpp.search_segmented_batch(
        queries=queries,
        segments=segments,
        k=2,
        ef=4,
        segment_search_k=2,
        metric="euclidean",
    )

    assert [[vector_id for vector_id, _distance in row] for row in results] == [
        [0, 1],
        [2, 3],
    ]
    assert results[0][0][1] == pytest.approx(0.05, abs=1e-6)
    assert results[1][0][1] == pytest.approx(0.1, abs=1e-6)


def test_cpp_search_segmented_batch_merges_cosine_by_descending_similarity():
    from src.index import hnsw_cpp

    queries = np.array([[1.0, 0.0]], dtype=np.float32)
    left_vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    right_vectors = np.array([[0.8, 0.2], [0.6, 0.4]], dtype=np.float32)
    full_offsets = np.array([0, 1, 2], dtype=np.int32)
    full_neighbors = np.array([1, 0], dtype=np.int32)

    segments = [
        {
            "vectors": left_vectors,
            "layers": {0: (full_offsets, full_neighbors)},
            "entry_point": 0,
            "max_layer": 0,
            "global_offset": 0,
        },
        {
            "vectors": right_vectors,
            "layers": {0: (full_offsets, full_neighbors)},
            "entry_point": 0,
            "max_layer": 0,
            "global_offset": 2,
        },
    ]

    results = hnsw_cpp.search_segmented_batch(
        queries=queries,
        segments=segments,
        k=3,
        ef=4,
        segment_search_k=2,
        metric="cosine",
    )

    assert [vector_id for vector_id, _score in results[0]] == [0, 2, 3]
    assert results[0][0][1] == pytest.approx(1.0, abs=1e-6)


def test_cpp_search_segmented_batch_rejects_dimension_mismatch():
    from src.index import hnsw_cpp

    queries = np.array([[0.0, 0.0]], dtype=np.float32)
    vectors = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    offsets = np.array([0, 0], dtype=np.int32)
    neighbors = np.array([], dtype=np.int32)

    with pytest.raises(ValueError, match="segment dimension must match queries dimension"):
        hnsw_cpp.search_segmented_batch(
            queries=queries,
            segments=[
                {
                    "vectors": vectors,
                    "layers": {0: (offsets, neighbors)},
                    "entry_point": 0,
                    "max_layer": 0,
                    "global_offset": 0,
                }
            ],
            k=1,
            ef=1,
            segment_search_k=1,
            metric="euclidean",
        )
```

- [ ] **Step 2: Run the failing tests**

Run:

```bash
./venv/bin/python -m pytest tests/test_hnsw_cpp.py::test_cpp_search_segmented_batch_merges_global_euclidean_results tests/test_hnsw_cpp.py::test_cpp_search_segmented_batch_merges_cosine_by_descending_similarity tests/test_hnsw_cpp.py::test_cpp_search_segmented_batch_rejects_dimension_mismatch -q
```

Expected: FAIL with `AttributeError` because `hnsw_cpp.search_segmented_batch` does not exist yet.

- [ ] **Step 3: Add native declarations**

In `src/index/hnsw_cpp_core.hpp`, add this struct after `CsrLayerView`:

```cpp
struct HnswSegmentView {
    const float* vectors;
    int n_vectors;
    int dimension;
    const CsrLayerView* layers;
    int n_layers;
    int entry_point;
    int max_layer;
    int global_offset;
};
```

In the same header, add this declaration after `search_batch(...)`:

```cpp
std::vector<std::vector<SearchResult>> search_segmented_batch(
    const float* queries,
    int n_queries,
    int dimension,
    const HnswSegmentView* segments,
    int n_segments,
    int k,
    int ef,
    int segment_search_k,
    const std::string& metric
);
```

- [ ] **Step 4: Implement native segmented search**

In `src/index/hnsw_cpp_core.cpp`, add this helper in the anonymous namespace after `search_csr_layer(...)`:

```cpp
bool public_result_less(
    const SearchResult& left,
    const SearchResult& right,
    bool use_euclidean
) {
    if (left.distance == right.distance) {
        return left.id < right.id;
    }
    if (use_euclidean) {
        return left.distance < right.distance;
    }
    return left.distance > right.distance;
}
```

Then add this function after the existing `search_batch(...)` implementation:

```cpp
std::vector<std::vector<SearchResult>> search_segmented_batch(
    const float* queries,
    int n_queries,
    int dimension,
    const HnswSegmentView* segments,
    int n_segments,
    int k,
    int ef,
    int segment_search_k,
    const std::string& metric
) {
    if (metric != "euclidean" && metric != "cosine") {
        throw std::invalid_argument("metric must be 'euclidean' or 'cosine'");
    }

    std::vector<std::vector<SearchResult>> output;
    if (n_queries > 0) {
        output.reserve(static_cast<std::size_t>(n_queries));
    }

    if (
        queries == nullptr ||
        segments == nullptr ||
        n_queries <= 0 ||
        dimension <= 0 ||
        n_segments <= 0 ||
        k <= 0
    ) {
        for (int query_index = 0; query_index < n_queries; ++query_index) {
            output.push_back({});
        }
        return output;
    }

    const int per_segment_k = std::max(segment_search_k, k);
    const int segment_beam_width = std::max(ef, per_segment_k);
    const bool use_euclidean = metric == "euclidean";

    struct PreparedSegment {
        const HnswSegmentView* segment = nullptr;
        std::vector<const CsrLayerView*> layer_by_number;
        SearchScratch scratch;
    };

    std::vector<PreparedSegment> prepared_segments;
    prepared_segments.reserve(static_cast<std::size_t>(n_segments));

    for (int segment_index = 0; segment_index < n_segments; ++segment_index) {
        const HnswSegmentView& segment = segments[segment_index];
        if (
            segment.vectors == nullptr ||
            segment.layers == nullptr ||
            segment.n_vectors <= 0 ||
            segment.dimension != dimension ||
            segment.n_layers <= 0 ||
            segment.entry_point < 0 ||
            segment.entry_point >= segment.n_vectors ||
            segment.max_layer < 0
        ) {
            continue;
        }

        PreparedSegment prepared;
        prepared.segment = &segment;
        prepared.layer_by_number.assign(
            static_cast<std::size_t>(segment.max_layer + 1),
            nullptr
        );
        prepared.scratch.ensure_size(static_cast<std::size_t>(segment.n_vectors));

        for (int layer_index = 0; layer_index < segment.n_layers; ++layer_index) {
            const CsrLayerView& layer = segment.layers[layer_index];
            if (layer.layer >= 0 && layer.layer <= segment.max_layer) {
                prepared.layer_by_number[static_cast<std::size_t>(layer.layer)] = &layer;
            }
        }

        prepared_segments.push_back(std::move(prepared));
    }

    for (int query_index = 0; query_index < n_queries; ++query_index) {
        const float* query = queries + (static_cast<std::size_t>(query_index) * dimension);
        std::vector<SearchResult> merged;
        merged.reserve(static_cast<std::size_t>(prepared_segments.size() * per_segment_k));

        for (PreparedSegment& prepared : prepared_segments) {
            const HnswSegmentView& segment = *prepared.segment;
            std::vector<int> nearest;
            nearest.push_back(segment.entry_point);

            for (int layer = segment.max_layer; layer > 0; --layer) {
                const CsrLayerView* layer_view =
                    prepared.layer_by_number[static_cast<std::size_t>(layer)];
                if (layer_view == nullptr) {
                    continue;
                }

                const std::vector<SearchResult> candidates = search_csr_layer(
                    query,
                    segment.vectors,
                    segment.n_vectors,
                    segment.dimension,
                    layer_view->offsets,
                    layer_view->offsets_len,
                    layer_view->neighbors,
                    layer_view->neighbors_len,
                    nearest.data(),
                    static_cast<int>(nearest.size()),
                    1,
                    use_euclidean,
                    prepared.scratch
                );
                if (!candidates.empty()) {
                    nearest.clear();
                    nearest.push_back(candidates[0].id);
                }
            }

            const CsrLayerView* base_layer = prepared.layer_by_number[0];
            if (base_layer == nullptr) {
                continue;
            }

            std::vector<SearchResult> candidates = search_csr_layer(
                query,
                segment.vectors,
                segment.n_vectors,
                segment.dimension,
                base_layer->offsets,
                base_layer->offsets_len,
                base_layer->neighbors,
                base_layer->neighbors_len,
                nearest.data(),
                static_cast<int>(nearest.size()),
                segment_beam_width,
                use_euclidean,
                prepared.scratch
            );
            if (static_cast<int>(candidates.size()) > per_segment_k) {
                candidates.resize(static_cast<std::size_t>(per_segment_k));
            }

            for (SearchResult candidate : candidates) {
                candidate.id += segment.global_offset;
                merged.push_back(candidate);
            }
        }

        std::sort(
            merged.begin(),
            merged.end(),
            [use_euclidean](const SearchResult& left, const SearchResult& right) {
                return public_result_less(left, right, use_euclidean);
            }
        );
        if (static_cast<int>(merged.size()) > k) {
            merged.resize(static_cast<std::size_t>(k));
        }
        output.push_back(merged);
    }

    return output;
}
```

- [ ] **Step 5: Continue to the Cython wrapper before committing**

Do not commit this task yet. The Python tests still fail until `hnsw_cpp.pyx`
exposes the new native function. Keep the test, header, and C++ implementation
changes in the working tree and continue directly to Task 2.

Expected git status at this point: `tests/test_hnsw_cpp.py`,
`src/index/hnsw_cpp_core.hpp`, and `src/index/hnsw_cpp_core.cpp` are modified.

---

### Task 2: Cython Wrapper For Native Segmented Search

**Files:**
- Modify: `src/index/hnsw_cpp.pyx`
- Test: `tests/test_hnsw_cpp.py`

- [ ] **Step 1: Add Cython declarations**

In `src/index/hnsw_cpp.pyx`, inside the `cdef extern from "hnsw_cpp_core.hpp" namespace "vectordb":` block, add this C++ class after `CppCsrLayerView`:

```cython
    cdef cppclass CppHnswSegmentView "vectordb::HnswSegmentView":
        const float* vectors
        int n_vectors
        int dimension
        const CppCsrLayerView* layers
        int n_layers
        int entry_point
        int max_layer
        int global_offset
```

Add this native function declaration after `cpp_search_batch(...)`:

```cython
    vector[vector[CppSearchResult]] cpp_search_segmented_batch "vectordb::search_segmented_batch"(
        const float* queries,
        int n_queries,
        int dimension,
        const CppHnswSegmentView* segments,
        int n_segments,
        int k,
        int ef,
        int segment_search_k,
        const string& metric
    ) except + nogil
```

- [ ] **Step 2: Add the Python wrapper**

In `src/index/hnsw_cpp.pyx`, add this function after `search_batch(...)` and before `prune_connections(...)`:

```cython
def search_segmented_batch(
    queries,
    segments,
    int k,
    int ef,
    int segment_search_k,
    str metric,
):
    """
    Search multiple compact CSR HNSW segments for a batch of query vectors.
    """
    cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] queries_arr = np.ascontiguousarray(
        queries, dtype=np.float32
    )
    if queries_arr.ndim != 2:
        raise ValueError("queries must be a 2D array shaped (n_queries, dimension)")

    cdef vector[CppHnswSegmentView] segment_views
    cdef vector[vector[CppCsrLayerView]] all_layer_views
    cdef CppHnswSegmentView segment_view
    cdef CppCsrLayerView layer_view
    cdef vector[CppCsrLayerView] current_layer_views
    cdef list keepalive = []
    cdef object segment
    cdef object layers
    cdef object layer
    cdef object layer_data
    cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] vectors_arr
    cdef cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] offsets_arr
    cdef cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] neighbors_arr
    cdef Py_ssize_t segment_index

    segment_views.reserve(len(segments))
    all_layer_views.reserve(len(segments))

    for segment_index, segment in enumerate(segments):
        vectors_arr = np.ascontiguousarray(segment["vectors"], dtype=np.float32)
        if vectors_arr.ndim != 2:
            raise ValueError("segment vectors must be a 2D array")
        if vectors_arr.shape[1] != queries_arr.shape[1]:
            raise ValueError("segment dimension must match queries dimension")
        keepalive.append(vectors_arr)

        current_layer_views.clear()
        layers = segment["layers"]
        for layer in sorted(layers):
            layer_data = layers[layer]
            offsets_arr = np.ascontiguousarray(layer_data[0], dtype=np.int32)
            neighbors_arr = np.ascontiguousarray(layer_data[1], dtype=np.int32)
            keepalive.append(offsets_arr)
            keepalive.append(neighbors_arr)

            layer_view.layer = <int> layer
            layer_view.offsets = <const int*> offsets_arr.data
            layer_view.offsets_len = <int> offsets_arr.shape[0]
            layer_view.neighbors = <const int*> neighbors_arr.data
            layer_view.neighbors_len = <int> neighbors_arr.shape[0]
            current_layer_views.push_back(layer_view)

        all_layer_views.push_back(current_layer_views)

        segment_view.vectors = <const float*> vectors_arr.data
        segment_view.n_vectors = <int> vectors_arr.shape[0]
        segment_view.dimension = <int> vectors_arr.shape[1]
        if all_layer_views[segment_index].size() > 0:
            segment_view.layers = &all_layer_views[segment_index][0]
        else:
            segment_view.layers = NULL
        segment_view.n_layers = <int> all_layer_views[segment_index].size()
        segment_view.entry_point = <int> segment["entry_point"]
        segment_view.max_layer = <int> segment["max_layer"]
        segment_view.global_offset = <int> segment["global_offset"]
        segment_views.push_back(segment_view)

    cdef const CppHnswSegmentView* segment_ptr = NULL
    if segment_views.size() > 0:
        segment_ptr = &segment_views[0]

    cdef string metric_cpp = metric.encode("utf-8")
    cdef vector[vector[CppSearchResult]] raw
    with nogil:
        raw = cpp_search_segmented_batch(
            <const float*> queries_arr.data,
            <int> queries_arr.shape[0],
            <int> queries_arr.shape[1],
            segment_ptr,
            <int> segment_views.size(),
            k,
            ef,
            segment_search_k,
            metric_cpp,
        )

    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef list output = []
    cdef list query_results
    for i in range(raw.size()):
        query_results = []
        for j in range(raw[i].size()):
            query_results.append((raw[i][j].id, raw[i][j].distance))
        output.append(query_results)

    return output
```

- [ ] **Step 3: Build the Cython/C++ extension**

Run:

```bash
./venv/bin/python setup.py build_ext --inplace
```

Expected: build succeeds. If Cython rejects `all_layer_views[segment_index]`, replace the index with an `int` variable named `segment_position` assigned from `segment_views.size()` before `all_layer_views.push_back(current_layer_views)`, then index `all_layer_views[segment_position]`. Re-run the build.

- [ ] **Step 4: Run C++ wrapper tests**

Run:

```bash
./venv/bin/python -m pytest tests/test_hnsw_cpp.py::test_cpp_search_segmented_batch_merges_global_euclidean_results tests/test_hnsw_cpp.py::test_cpp_search_segmented_batch_merges_cosine_by_descending_similarity tests/test_hnsw_cpp.py::test_cpp_search_segmented_batch_rejects_dimension_mismatch -q
```

Expected: PASS.

- [ ] **Step 5: Commit Cython wrapper**

Run:

```bash
git add src/index/hnsw_cpp_core.hpp src/index/hnsw_cpp_core.cpp src/index/hnsw_cpp.pyx tests/test_hnsw_cpp.py
git commit -m "feat: add native segmented hnsw search"
```

---

### Task 3: SegmentedHNSWIndex Native Dispatch And Tuning

**Files:**
- Modify: `src/index/segmented_hnsw.py`
- Modify: `tests/test_segmented_hnsw.py`

- [ ] **Step 1: Add failing segmented wrapper tests**

Append these tests to `tests/test_segmented_hnsw.py`:

```python
def test_segmented_hnsw_uses_native_segmented_search_batch(monkeypatch):
    import src.index.segmented_hnsw as segmented_module
    from src.index.hnsw import HNSWIndex
    from src.index.segmented_hnsw import SegmentedHNSWIndex

    calls = []

    class FakeNative:
        def search_segmented_batch(self, queries, segments, k, ef, segment_search_k, metric):
            calls.append(
                {
                    "queries_shape": queries.shape,
                    "segment_count": len(segments),
                    "k": k,
                    "ef": ef,
                    "segment_search_k": segment_search_k,
                    "metric": metric,
                    "offsets": [segment["global_offset"] for segment in segments],
                }
            )
            return [[(0, 0.0), (3, 0.2)], [(2, 0.1), (1, 0.4)]]

    monkeypatch.setattr(segmented_module, "CPP_AVAILABLE", True)
    monkeypatch.setattr(segmented_module, "hnsw_cpp", FakeNative())

    def make_segment(vectors):
        index = HNSWIndex(M=2, ef_search=6, metric="euclidean")
        index.vectors = np.asarray(vectors, dtype=np.float32)
        index._vectors_f32_cache = np.ascontiguousarray(index.vectors, dtype=np.float32)
        index._cpp_graph_cache = {
            0: (
                np.array([0, 1, 2], dtype=np.int32),
                np.array([1, 0], dtype=np.int32),
            )
        }
        index.entry_point = 0
        index.max_layer = 0
        index._python_graph_materialized = False
        return index

    segmented = SegmentedHNSWIndex(segment_count=2, segment_search_k=5)
    segmented.segments = [
        segmented_module.HNSWSegment(0, 2, make_segment([[0.0, 0.0], [0.1, 0.0]]), 0.0),
        segmented_module.HNSWSegment(2, 4, make_segment([[2.0, 0.0], [2.1, 0.0]]), 0.0),
    ]

    queries = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    results = segmented.search_batch(queries, k=2, ef=4)

    assert results == [[(0, 0.0), (3, 0.2)], [(2, 0.1), (1, 0.4)]]
    assert calls == [
        {
            "queries_shape": (2, 2),
            "segment_count": 2,
            "k": 2,
            "ef": 4,
            "segment_search_k": 5,
            "metric": "euclidean",
            "offsets": [0, 2],
        }
    ]
    assert segmented._last_search_used_native_segmented is True


def test_segmented_hnsw_segment_search_k_is_clamped_to_k(monkeypatch):
    import src.index.segmented_hnsw as segmented_module
    from src.index.hnsw import HNSWIndex
    from src.index.segmented_hnsw import SegmentedHNSWIndex

    seen_segment_search_k = []

    class FakeNative:
        def search_segmented_batch(self, queries, segments, k, ef, segment_search_k, metric):
            seen_segment_search_k.append(segment_search_k)
            return [[(0, 0.0), (1, 0.1), (2, 0.2)]]

    monkeypatch.setattr(segmented_module, "CPP_AVAILABLE", True)
    monkeypatch.setattr(segmented_module, "hnsw_cpp", FakeNative())

    index = HNSWIndex(M=2, ef_search=6, metric="euclidean")
    index.vectors = np.arange(6, dtype=np.float32).reshape(3, 2)
    index._vectors_f32_cache = np.ascontiguousarray(index.vectors, dtype=np.float32)
    index._cpp_graph_cache = {
        0: (
            np.array([0, 1, 2, 3], dtype=np.int32),
            np.array([1, 0, 1], dtype=np.int32),
        )
    }
    index.entry_point = 0
    index.max_layer = 0
    index._python_graph_materialized = False

    segmented = SegmentedHNSWIndex(segment_count=1, segment_search_k=1)
    segmented.segments = [segmented_module.HNSWSegment(0, 3, index, 0.0)]

    segmented.search_batch(np.array([[0.0, 1.0]], dtype=np.float32), k=3, ef=3)

    assert seen_segment_search_k == [3]


def test_segmented_hnsw_native_fallback_for_fake_segments(monkeypatch):
    import src.index.segmented_hnsw as segmented_module
    from src.index.segmented_hnsw import SegmentedHNSWIndex

    class FailingNative:
        def search_segmented_batch(self, *args, **kwargs):
            raise AssertionError("fake segments should not use native segmented search")

    monkeypatch.setattr(segmented_module, "CPP_AVAILABLE", True)
    monkeypatch.setattr(segmented_module, "hnsw_cpp", FailingNative())

    vectors = np.array([[0.0, 0.0], [0.5, 0.0], [10.0, 0.0], [10.5, 0.0]], dtype=np.float32)
    queries = np.array([[0.1, 0.0]], dtype=np.float32)

    index = SegmentedHNSWIndex(segment_count=2, segment_factory=FakeExactSegment)
    index.build(vectors)

    assert [[vector_id for vector_id, _distance in row] for row in index.search_batch(queries, k=2)] == [
        [0, 1]
    ]
    assert index._last_search_used_native_segmented is False
```

- [ ] **Step 2: Run failing segmented wrapper tests**

Run:

```bash
./venv/bin/python -m pytest tests/test_segmented_hnsw.py::test_segmented_hnsw_uses_native_segmented_search_batch tests/test_segmented_hnsw.py::test_segmented_hnsw_segment_search_k_is_clamped_to_k tests/test_segmented_hnsw.py::test_segmented_hnsw_native_fallback_for_fake_segments -q
```

Expected: FAIL because `SegmentedHNSWIndex.__init__()` does not accept `segment_search_k`.

- [ ] **Step 3: Import native availability in segmented wrapper**

In `src/index/segmented_hnsw.py`, replace:

```python
from src.index.hnsw import HNSWIndex, sample_hnsw_levels
```

with:

```python
from src.index.hnsw import CPP_AVAILABLE, HNSWIndex, hnsw_cpp, sample_hnsw_levels
```

- [ ] **Step 4: Add constructor state and validation**

In `SegmentedHNSWIndex.__init__`, add the argument before `segment_factory`:

```python
        segment_search_k: int | None = None,
```

After the existing `build_threads` validation, add:

```python
        if segment_search_k is not None and segment_search_k <= 0:
            raise ValueError("segment_search_k must be positive when provided")
```

After `self.build_threads = build_threads`, add:

```python
        self.segment_search_k = segment_search_k
```

After `self._last_cpp_build_stats: dict | None = None`, add:

```python
        self._last_search_used_native_segmented = False
```

- [ ] **Step 5: Add helper methods and native dispatch**

In `src/index/segmented_hnsw.py`, add these methods before `search(...)`:

```python
    def _effective_segment_search_k(self, k: int) -> int:
        if self.segment_search_k is None:
            return max(k, 1)
        return max(k, self.segment_search_k)

    def _native_segment_descriptors(self) -> list[dict] | None:
        if not (
            CPP_AVAILABLE
            and hnsw_cpp is not None
            and hasattr(hnsw_cpp, "search_segmented_batch")
        ):
            return None

        descriptors: list[dict] = []
        for segment in self.segments:
            segment_index = segment.index
            if not isinstance(segment_index, HNSWIndex):
                return None
            if (
                segment_index._cpp_graph_cache is None
                or segment_index._vectors_f32_cache is None
                or segment_index.entry_point is None
                or segment_index.max_layer < 0
            ):
                return None
            if not all(
                layer in segment_index._cpp_graph_cache
                for layer in range(segment_index.max_layer + 1)
            ):
                return None

            descriptors.append(
                {
                    "vectors": segment_index._vectors_f32_cache,
                    "layers": segment_index._cpp_graph_cache,
                    "entry_point": segment_index.entry_point,
                    "max_layer": segment_index.max_layer,
                    "global_offset": segment.start,
                }
            )

        return descriptors

    def _search_batch_native(
        self,
        query_array: np.ndarray,
        k: int,
        ef: int,
    ) -> list[list[tuple[int, float]]] | None:
        descriptors = self._native_segment_descriptors()
        if descriptors is None:
            self._last_search_used_native_segmented = False
            return None

        segment_search_k = self._effective_segment_search_k(k)
        self._last_search_used_native_segmented = True
        return hnsw_cpp.search_segmented_batch(
            queries=np.ascontiguousarray(query_array, dtype=np.float32),
            segments=descriptors,
            k=k,
            ef=ef,
            segment_search_k=segment_search_k,
            metric=self.metric,
        )
```

- [ ] **Step 6: Route search and search_batch through native path**

In `search(...)`, replace:

```python
        candidates: list[tuple[int, float]] = []
        per_segment_k = max(k, 1)
        for segment in self.segments:
            for local_id, distance in segment.index.search(query, k=per_segment_k, ef=ef):
                candidates.append((segment.start + int(local_id), float(distance)))
```

with:

```python
        if ef is None:
            ef = self.ef_search
        ef = max(ef, k)

        query_array = np.asarray([query], dtype=np.float32)
        native_results = self._search_batch_native(query_array, k=k, ef=ef)
        if native_results is not None:
            return native_results[0]

        candidates: list[tuple[int, float]] = []
        per_segment_k = self._effective_segment_search_k(k)
        for segment in self.segments:
            for local_id, distance in segment.index.search(query, k=per_segment_k, ef=ef):
                candidates.append((segment.start + int(local_id), float(distance)))
```

In `search_batch(...)`, replace the final line:

```python
        return [self.search(query, k=k, ef=ef) for query in query_array]
```

with:

```python
        if k <= 0:
            return [[] for _query in query_array]
        if ef is None:
            ef = self.ef_search
        ef = max(ef, k)

        native_results = self._search_batch_native(query_array, k=k, ef=ef)
        if native_results is not None:
            return native_results

        return [self.search(query, k=k, ef=ef) for query in query_array]
```

- [ ] **Step 7: Run segmented wrapper tests**

Run:

```bash
./venv/bin/python -m pytest tests/test_segmented_hnsw.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit segmented wrapper integration**

Run:

```bash
git add src/index/segmented_hnsw.py tests/test_segmented_hnsw.py
git commit -m "feat: route segmented hnsw search through native batch"
```

---

### Task 4: Benchmark Runner Schema And Batch Measurement

**Files:**
- Modify: `benchmarks/benchmark.py`
- Modify: `tests/test_benchmark_cli.py`

- [ ] **Step 1: Add failing benchmark schema tests**

In `tests/test_benchmark_cli.py`, inside `test_run_benchmark_suite_reports_segmented_build_stats()`, add these assertions after the build thread assertion:

```python
    assert result["config"]["hnsw"]["segment_search_k"] is None
    assert "search" in result["metrics"]
    assert result["metrics"]["search"]["uses_batch_api"] is True
    assert "native_segmented_batch" in result["metrics"]["search"]
```

In `test_main_writes_segmented_markdown_rows(...)`, add CLI args after `"--build-threads", "1"`:

```python
            "--segment-search-k",
            "6",
```

Add these assertions after the JSON load:

```python
    assert data["config"]["hnsw"]["segment_search_k"] == 6
    assert data["metrics"]["search"]["uses_batch_api"] is True
```

Add this Markdown assertion:

```python
    assert "Segment Search K" in markdown
    assert "Native Segmented Batch" in markdown
```

- [ ] **Step 2: Run failing benchmark tests**

Run:

```bash
./venv/bin/python -m pytest tests/test_benchmark_cli.py::test_run_benchmark_suite_reports_segmented_build_stats tests/test_benchmark_cli.py::test_main_writes_segmented_markdown_rows -q
```

Expected: FAIL because `segment_search_k` and search provenance are not in the schema yet.

- [ ] **Step 3: Add `segment_search_k` to the benchmark API**

In `benchmarks/benchmark.py`, add this parameter to `run_benchmark_suite(...)` after `build_threads`:

```python
    segment_search_k: int | None = None,
```

After the `build_threads` validation, add:

```python
    if segment_search_k is not None and segment_search_k <= 0:
        raise ValueError("segment_search_k must be positive when provided")
```

When constructing `SegmentedHNSWIndex`, pass:

```python
            segment_search_k=segment_search_k,
```

In the result config, add:

```python
                "segment_search_k": segment_search_k,
```

- [ ] **Step 4: Use batch search for benchmark QPS**

In `run_benchmark_suite(...)`, replace the current search loop block:

```python
    latencies_ms: list[float] = []
    predictions: list[list[int]] = []

    search_start = time.perf_counter()
    for query in queries:
        query_start = time.perf_counter()
        results = index.search(query, k=effective_k, ef=ef_search)
        latencies_ms.append((time.perf_counter() - query_start) * 1000)
        predictions.append([int(vector_id) for vector_id, _distance in results[:effective_k]])
    search_time = time.perf_counter() - search_start
```

with:

```python
    latencies_ms: list[float] = []
    predictions: list[list[int]] = []

    search_start = time.perf_counter()
    if hasattr(index, "search_batch"):
        batch_results = index.search_batch(queries, k=effective_k, ef=ef_search)
        search_time = time.perf_counter() - search_start
        predictions = [
            [int(vector_id) for vector_id, _distance in results[:effective_k]]
            for results in batch_results
        ]
        uses_batch_api = True
    else:
        batch_results = []
        for query in queries:
            results = index.search(query, k=effective_k, ef=ef_search)
            batch_results.append(results)
        search_time = time.perf_counter() - search_start
        predictions = [
            [int(vector_id) for vector_id, _distance in results[:effective_k]]
            for results in batch_results
        ]
        uses_batch_api = False

    for query in queries:
        query_start = time.perf_counter()
        index.search(query, k=effective_k, ef=ef_search)
        latencies_ms.append((time.perf_counter() - query_start) * 1000)
```

In the `metrics` dict, add:

```python
            "search": {
                "uses_batch_api": uses_batch_api,
                "native_segmented_batch": bool(
                    getattr(index, "_last_search_used_native_segmented", False)
                ),
            },
```

- [ ] **Step 5: Add CLI flag and Markdown rows**

In `_build_parser()`, add:

```python
    parser.add_argument(
        "--segment-search-k",
        type=int,
        default=None,
        help="Candidates requested from each segment before global merge",
    )
```

In `main(...)`, pass:

```python
        segment_search_k=args.segment_search_k,
```

In `format_markdown_report(...)`, add `search = metrics["search"]` after the `metrics` assignment. Then add these Markdown rows after the segmented build rows:

```python
            *(
                [
                    f"| Segment Search K | {config['segment_search_k']} |",
                    f"| Native Segmented Batch | {search['native_segmented_batch']} |",
                ]
                if segmented_build_stats
                else []
            ),
```

- [ ] **Step 6: Run benchmark schema tests**

Run:

```bash
./venv/bin/python -m pytest tests/test_benchmark_cli.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit benchmark runner update**

Run:

```bash
git add benchmarks/benchmark.py tests/test_benchmark_cli.py
git commit -m "feat: benchmark native segmented search"
```

---

### Task 5: Full Focused Verification

**Files:**
- No source edits expected.

- [ ] **Step 1: Rebuild extension from clean sources**

Run:

```bash
./venv/bin/python setup.py build_ext --inplace
```

Expected: build succeeds.

- [ ] **Step 2: Run focused tests**

Run:

```bash
./venv/bin/python -m pytest tests/test_hnsw_cpp.py tests/test_segmented_hnsw.py tests/test_benchmark_cli.py -q
```

Expected: PASS.

- [ ] **Step 3: Run fast suite**

Run:

```bash
make test PYTHON=./venv/bin/python
```

Expected: PASS.

---

### Task 6: Benchmark Matrix And Documentation

**Files:**
- Modify: `TECHNICAL.md`
- Modify: `README.md`
- Create: `benchmarks/results/native-segmented-sift100k-2-k10.json`
- Create: `benchmarks/results/native-segmented-sift100k-2-k10.md`
- Create: `benchmarks/results/native-segmented-sift100k-2-k20.json`
- Create: `benchmarks/results/native-segmented-sift100k-2-k20.md`
- Create: `benchmarks/results/native-segmented-sift100k-4-k10.json`
- Create: `benchmarks/results/native-segmented-sift100k-4-k10.md`
- Create: `benchmarks/results/native-segmented-sift100k-4-k20.json`
- Create: `benchmarks/results/native-segmented-sift100k-4-k20.md`
- Create: `benchmarks/results/native-segmented-sift100k-8-k10.json`
- Create: `benchmarks/results/native-segmented-sift100k-8-k10.md`
- Create: `benchmarks/results/native-segmented-sift100k-8-k20.json`
- Create: `benchmarks/results/native-segmented-sift100k-8-k20.md`

- [ ] **Step 1: Run a small synthetic smoke benchmark**

Run:

```bash
./venv/bin/python benchmarks/benchmark.py \
  --dataset random \
  --size 1000 \
  --dimension 32 \
  --queries 20 \
  --k 10 \
  --ef-search 50 \
  --segments 4 \
  --build-threads 4 \
  --segment-search-k 10 \
  --output /tmp/native-segmented-smoke.json \
  --markdown-output /tmp/native-segmented-smoke.md
```

Expected: exits 0 and prints a summary with `recall@10`, `qps`, and `p99`.

- [ ] **Step 2: Run SIFT1M 100k benchmark matrix**

Run these six commands only if SIFT1M data is already available locally:

```bash
./venv/bin/python benchmarks/benchmark.py --dataset sift1m --subset medium --size 100000 --dimension 128 --queries 100 --k 10 --M 16 --ef-construction 200 --ef-search 100 --segments 2 --build-threads 2 --segment-search-k 10 --output benchmarks/results/native-segmented-sift100k-2-k10.json --markdown-output benchmarks/results/native-segmented-sift100k-2-k10.md
./venv/bin/python benchmarks/benchmark.py --dataset sift1m --subset medium --size 100000 --dimension 128 --queries 100 --k 10 --M 16 --ef-construction 200 --ef-search 100 --segments 2 --build-threads 2 --segment-search-k 20 --output benchmarks/results/native-segmented-sift100k-2-k20.json --markdown-output benchmarks/results/native-segmented-sift100k-2-k20.md
./venv/bin/python benchmarks/benchmark.py --dataset sift1m --subset medium --size 100000 --dimension 128 --queries 100 --k 10 --M 16 --ef-construction 200 --ef-search 100 --segments 4 --build-threads 4 --segment-search-k 10 --output benchmarks/results/native-segmented-sift100k-4-k10.json --markdown-output benchmarks/results/native-segmented-sift100k-4-k10.md
./venv/bin/python benchmarks/benchmark.py --dataset sift1m --subset medium --size 100000 --dimension 128 --queries 100 --k 10 --M 16 --ef-construction 200 --ef-search 100 --segments 4 --build-threads 4 --segment-search-k 20 --output benchmarks/results/native-segmented-sift100k-4-k20.json --markdown-output benchmarks/results/native-segmented-sift100k-4-k20.md
./venv/bin/python benchmarks/benchmark.py --dataset sift1m --subset medium --size 100000 --dimension 128 --queries 100 --k 10 --M 16 --ef-construction 200 --ef-search 100 --segments 8 --build-threads 8 --segment-search-k 10 --output benchmarks/results/native-segmented-sift100k-8-k10.json --markdown-output benchmarks/results/native-segmented-sift100k-8-k10.md
./venv/bin/python benchmarks/benchmark.py --dataset sift1m --subset medium --size 100000 --dimension 128 --queries 100 --k 10 --M 16 --ef-construction 200 --ef-search 100 --segments 8 --build-threads 8 --segment-search-k 20 --output benchmarks/results/native-segmented-sift100k-8-k20.json --markdown-output benchmarks/results/native-segmented-sift100k-8-k20.md
```

Expected: each command exits 0 and writes JSON/Markdown artifacts.

- [ ] **Step 3: Update `TECHNICAL.md`**

Add a section before `## Future Improvements` titled:

```markdown
## Native Segmented HNSW Batch Search
```

Include:

- what was there before: Python `queries x segments` loop
- what was implemented: native C++ multi-segment search and merge
- why: improve segmented QPS/latency while keeping segmented build opt-in
- benchmark table with the six matrix results
- next steps: tune `segment_search_k`, avoid same-graph insertion, consider SIMD only after measuring this path

- [ ] **Step 4: Update `README.md`**

Update the `Segmented Build Trade-off` table only with benchmark rows that were actually produced in Step 2. Keep the old tracked numbers if the new benchmark was not run.

- [ ] **Step 5: Commit benchmark results and docs**

Run:

```bash
git add TECHNICAL.md README.md benchmarks/results/native-segmented-sift100k-*.json benchmarks/results/native-segmented-sift100k-*.md
git commit -m "docs: record native segmented search benchmark"
```

---

## Final Verification

Run:

```bash
./venv/bin/python setup.py build_ext --inplace
./venv/bin/python -m pytest tests/test_hnsw_cpp.py tests/test_segmented_hnsw.py tests/test_benchmark_cli.py -q
make test PYTHON=./venv/bin/python
git status --short --branch
```

Expected:

- extension build succeeds
- focused tests pass
- fast suite passes
- git status is clean on the feature branch
