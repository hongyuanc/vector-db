# Native Segmented HNSW Batch Search Design

## Goal

Improve segmented HNSW query throughput and latency by moving multi-segment
batch search and global result merging into the native C++ path.

The target benchmark shape is the current SIFT1M 100k segmented build matrix:

- Dataset: SIFT1M 100k subset
- Metric: Euclidean
- HNSW config: `M=16`, `ef_construction=200`, `ef_search=100`
- Queries: 100
- Result count: `k=10`
- Segments: `2`, `4`, and `8`

The current segmented results show the trade-off clearly:

| Mode | Build Time | Recall@10 | QPS | p99 Latency |
|---|---:|---:|---:|---:|
| Single graph baseline | 40.1198s | 0.9860 | 3378.48 | not recorded in table |
| 2 segments / 2 threads | 18.1246s | 0.9940 | 1900.17 | 0.7728ms |
| 4 segments / 4 threads | 8.1387s | 0.9980 | 1035.96 | 1.4544ms |
| 8 segments / 8 threads | 4.6599s | 0.9940 | 570.12 | 2.4883ms |

Segmented build gives a strong wall-clock build-time win, but query throughput
drops as segment count increases.

## Current Bottleneck

`SegmentedHNSWIndex.search()` loops over every segment for one query, then
merges candidates in Python. `SegmentedHNSWIndex.search_batch()` currently loops
over each query and calls `search()`:

```python
return [self.search(query, k=k, ef=ef) for query in query_array]
```

That means segmented benchmark search pays Python overhead across:

```text
queries x segments
```

Each segment already has a compact CSR graph and each individual `HNSWIndex`
can use native C++/Cython batch search. The missing piece is a native API that
searches multiple segment graphs for a query batch and performs the global
top-k merge without returning to Python for every query/segment pair.

## Decision

Add a fully native multi-segment batch search path.

The new path should:

1. Accept a batch of query vectors.
2. Accept multiple compact CSR HNSW segment graphs.
3. Search each segment graph natively.
4. Convert local segment ids to global vector ids.
5. Merge per-segment candidates into global top-k results per query.
6. Return the same Python result shape as existing search methods:
   `list[list[tuple[int, float]]]`.

The Python fallback remains for non-compact segments, fake test segments, and
any case where the native extension is unavailable.

## Non-Goals

- Do not make segmented build the default.
- Do not change segmented build construction.
- Do not add same-graph parallel insertion.
- Do not change HNSW graph quality parameters by default.
- Do not replace the existing single-graph `search_batch()` path.
- Do not require all benchmarks to use segmented mode.
- Do not optimize single-graph build time in this phase.

## Proposed Native API

### C++ Segment View

Add a native segment view structure in `src/index/hnsw_cpp_core.hpp` and
`src/index/hnsw_cpp_core.cpp`.

Each segment view should carry:

- pointer to the segment vector matrix
- segment vector count
- dimension
- CSR layer views
- number of CSR layers
- entry point
- max layer
- global id offset

Conceptually:

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

### C++ Search Function

Add a new native function:

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

`segment_search_k` controls how many candidates to request from each segment
before global merge. It must be at least `k` by default, but benchmarks should
be able to test values such as `k`, `ceil(1.5k)`, and `2k`.

### Merge Rules

The native merge must preserve existing ordering:

- Euclidean: lower distance is better, tie-break by lower global vector id.
- Cosine: higher similarity is better, tie-break by lower global vector id.

The returned result for each query must contain at most `k` candidates.

## Python/Cython Integration

### Cython Wrapper

Expose the native function from `src/index/hnsw_cpp.pyx`.

The wrapper should accept a Python list of segment descriptors and build C++
views from existing NumPy arrays:

- segment vectors as contiguous `float32`
- CSR layer offsets as contiguous `int32`
- CSR layer neighbors as contiguous `int32`
- entry point and max layer
- global id offset

The wrapper must keep Python references to all arrays alive until the native
call completes.

### SegmentedHNSWIndex

Update `SegmentedHNSWIndex.search_batch()` to use the native path when:

- the C++ extension is available
- every segment index is an `HNSWIndex`
- every segment has compact CSR cache available
- every segment has contiguous `float32` vectors
- every segment has a valid entry point

Fallback behavior stays unchanged:

```python
return [self.search(query, k=k, ef=ef) for query in query_array]
```

### Tuning Parameter

Add a segmented search tuning parameter:

- constructor argument: `segment_search_k: int | None = None`
- if `None`, use `k`
- during search, effective per-segment candidate count is
  `max(k, segment_search_k)`

The benchmark runner should expose this as:

```text
--segment-search-k
```

If omitted, current behavior is preserved by using `k`.

## Benchmark Plan

Run the benchmark matrix after implementation:

| Segments | Build Threads | segment_search_k |
|---:|---:|---:|
| 2 | 2 | 10 |
| 2 | 2 | 20 |
| 4 | 4 | 10 |
| 4 | 4 | 20 |
| 8 | 8 | 10 |
| 8 | 8 | 20 |

Primary metrics:

- QPS
- average latency
- p95 latency
- p99 latency
- Recall@10
- build time

Success criteria:

- Improve 4-segment and 8-segment QPS materially over current tracked results.
- Keep Recall@10 at or above `0.9860` where possible.
- Preserve segmented build-time advantage.
- Keep segmented mode opt-in.

Current tracked results to beat:

| Mode | QPS | p99 Latency | Recall@10 |
|---|---:|---:|---:|
| 2 segments | 1900.17 | 0.7728ms | 0.9940 |
| 4 segments | 1035.96 | 1.4544ms | 0.9980 |
| 8 segments | 570.12 | 2.4883ms | 0.9940 |

## Testing Strategy

### Unit Tests

Add focused tests before implementation:

- Native segmented batch search matches existing Python segmented `search_batch`
  results on a small deterministic index.
- Global ids are offset correctly from local segment ids.
- Euclidean merge sorts by ascending distance and then global id.
- Cosine merge sorts by descending similarity and then global id.
- `segment_search_k` is clamped to at least `k`.
- Fallback path still works for fake/non-native segment indexes.

### C++/Cython Boundary Tests

Add tests in `tests/test_hnsw_cpp.py` for:

- wrapper accepts multiple segment descriptors
- empty segments return empty result rows safely
- invalid dimensions raise clear errors
- native segmented results match repeated single-segment native batch searches
  plus Python merge on small data

### Benchmark Schema Tests

Update benchmark CLI tests so JSON and Markdown include:

- `segment_search_k`
- whether native segmented batch search was used
- segmented search timing metrics if added

## Risks

### C++/Cython Boundary Complexity

The wrapper must keep NumPy array references alive while C++ uses raw pointers.
The implementation should build the segment view data inside one wrapper call
and avoid storing raw pointers beyond the call lifetime.

### Recall/Speed Trade-off

If `segment_search_k = k` misses true global top-k candidates, recall can drop.
Benchmarks must report this directly instead of hiding it. `segment_search_k`
exists so the trade-off can be measured explicitly.

### Memory and Copying

If the wrapper copies vectors or CSR arrays per search, the optimization may
lose much of its benefit. The design should pass existing contiguous arrays
without copying when possible.

### Over-Optimizing Before Measurement

Do not add native threading or same-graph build changes in this phase. First
remove the Python segmented search loop and measure the result.

## Documentation Updates

After implementation and benchmarks:

- Add a `TECHNICAL.md` section documenting what changed, why native segmented
  batch search was chosen, benchmark results, and next steps.
- Update `README.md` segmented benchmark table only if new tracked benchmark
  artifacts improve or clarify the current headline numbers.
- Store benchmark JSON and Markdown under `benchmarks/results/`.

## Open Questions Resolved

The headline optimization target is segmented search QPS and latency.

The selected approach is the high-ceiling native multi-segment search path, not
the smaller Python merge improvement. This is more complex, but it attacks the
actual segmented benchmark cost directly and creates a stronger foundation for
future tuning.
