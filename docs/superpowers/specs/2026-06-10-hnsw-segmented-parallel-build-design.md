# HNSW Segmented Parallel Native Build Design

## Goal

Reduce SIFT1M 100k build time by using CPU parallelism without making the
current single-graph HNSW builder nondeterministic or lock-heavy.

The immediate target remains the existing 100k comparison shape:

- Dataset: SIFT1M 100k subset
- Metric: Euclidean
- HNSW config: `M=16`, `ef_construction=200`, `ef_search=100`
- Queries: 100
- Result count: `k=10`

The latest measured single-graph result after bounded build-search enqueues is:

| Metric | Value |
|---|---:|
| Build Time | 40.1198s |
| Native Candidate Search | 30.4112s |
| Native Greedy Search | 1.2506s |
| Native Prune | 4.2298s |
| Recall@10 | 0.9860 |
| QPS | 3378.48 |

The remaining build-time gap is still dominated by candidate collection during
graph construction.

## Decision

Use segmented parallel native build before attempting same-graph parallel HNSW
insertion.

Segmented build means the dataset is split into several independent vector
segments. Each segment builds its own compact native HNSW graph using the
existing C++ builder. Search fans out across the segment graphs and merges the
per-segment top-k results into one global top-k list.

Same-graph parallel insertion means multiple worker threads insert into one
shared mutable HNSW graph at the same time. That is not the next step for this
project.

## Why Segmented Parallel Build

### HNSW Insertion Is Order-Dependent

The current builder inserts vector `i` into a graph that already contains
vectors `0..i-1`. The candidate search, selected neighbors, reverse edges, and
pruning decisions all depend on the graph state at that exact point. If several
vectors insert into the same graph concurrently, they either see different
partial graph states or require synchronization around the mutable graph.

That makes same-graph parallel insertion a construction algorithm change, not a
simple performance optimization.

### Shared-Graph Mutation Needs Heavy Coordination

Same-graph parallel insertion would need coordination for:

- node and layer allocation
- entry point and max-layer updates
- mutable adjacency writes
- reverse edge additions
- overflow pruning
- build counters and scratch buffers
- deterministic level/order handling

Coarse locks would likely erase much of the parallel speedup. Fine-grained locks
would make the code harder to reason about and could introduce deadlocks,
nondeterminism, or hard-to-reproduce recall regressions.

### Segments Reuse the Trusted Builder

Segmented build keeps each native graph build single-writer and isolated. Each
worker can call the current C++ `build_graph()` path with its own vector slice,
levels, scratch memory, counters, and compact CSR output. The existing builder
remains the correctness baseline for each segment.

This is a safer educational step because the new complexity lives at clear
boundaries:

- split vectors into segments
- build each segment independently
- keep per-segment compact CSR graphs
- search each segment independently
- merge sorted result candidates by global vector id and distance

### Segments Match Production Architecture

Production vector databases often scale indexing through segments, shards, or
partitions before trying to make one mutable graph accept fully parallel
inserts. This design teaches that architecture directly. It also gives the
project a realistic path toward 1M-scale indexing, where independent segments
can be built, persisted, loaded, compacted, or rebuilt separately.

### The Trade-off Is Measurable

Segmented build changes the search surface. A single global graph can route
through any vector during search. Multiple segment graphs require query fanout
and result merging. This can reduce recall if the true nearest neighbors are
spread across segments and each segment returns too few candidates.

That trade-off is explicit and measurable:

- build time should improve as segment count and worker count increase
- search latency may increase because each query searches multiple graphs
- recall may move depending on segment count and per-segment candidate depth
- memory may increase because each segment has its own entry point and layer
  metadata

The design should be accepted only if benchmarks show a good build-time gain
without hiding recall loss.

## Non-Goals

- Do not replace the current single-graph builder.
- Do not make segmented build the default before it has benchmark evidence.
- Do not add same-graph parallel insertion in this phase.
- Do not add distributed indexing or networked shards.
- Do not change metadata filtering, persistence format compatibility, or online
  mutation behavior in this phase.

## Proposed Architecture

### Python Orchestration

Add a segmented build path above the existing native C++ builder. Python remains
responsible for choosing whether to build one graph or multiple segments.

Initial public shape should be opt-in, for example:

- `build(vectors)` keeps the current single-graph behavior.
- A benchmark-only or explicit option enables segmented build with
  `segment_count` and `build_threads`.

The exact public API should be finalized during the implementation plan, but the
first implementation should avoid surprising existing users.

### Segment Object

Each segment needs:

- start offset in the original vector array
- local vector slice
- local levels
- local entry point
- local max layer
- local compact CSR layer arrays
- local build stats

Search results from a segment use local ids internally and convert to global ids
by adding the segment offset.

### Build Flow

1. Validate vectors and HNSW parameters once.
2. Split the vector matrix into contiguous segments.
3. Generate deterministic levels for each global vector id before dispatching
   workers, so repeated runs keep the same sampled levels.
4. Build each segment with the existing native `hnsw_cpp.build_graph()` path.
5. Store segment metadata and per-segment CSR graph caches.
6. Aggregate build stats into a segmented build report.

Contiguous segments are the first design because they preserve simple id mapping
and reduce implementation complexity. Randomized assignment can be evaluated
later if recall suffers.

### Search Flow

1. For each query batch, search every segment graph using the existing native
   compact CSR batch search.
2. Convert local segment ids to global ids.
3. Merge all segment candidates by distance.
4. Return the global top-k results.

The first version can search segments sequentially if that keeps the
implementation smaller. Parallel query fanout is a follow-up once build behavior
and recall are measured.

### Stats and Reporting

Build stats should make segmented behavior explicit:

- `uses_segmented_build`
- `segment_count`
- `build_threads`
- per-segment build time
- aggregate build time
- max segment build time
- sum of native candidate-search time across segments
- total directed edges across segments
- total heap pushes, visited nodes, and distance evaluations across segments

The benchmark Markdown should show both total wall-clock build time and summed
native work. Wall-clock time tells the user what they waited for. Summed native
work explains whether parallelism reduced elapsed time by overlapping
independent work or whether it also changed graph work.

## Testing Strategy

Focused tests should come before implementation:

- segmented build returns valid top-k results with global ids
- segmented build preserves compact CSR storage per segment
- `segment_count=1` matches current single-graph behavior or delegates to the
  existing path
- per-segment local ids are converted to global ids correctly
- benchmark JSON and Markdown expose segmented build stats
- empty vectors, `k <= 0`, and segment counts larger than vector count have
  defined behavior

Benchmark tests should measure:

- SIFT1M 100k with 1, 2, 4, and 8 segments
- build time
- recall@10
- QPS and latency
- memory
- per-segment candidate-search time

## Success Criteria

The first milestone is not "parallelism exists." The first milestone is evidence
that segmentation helps the 100k target.

An acceptable first result would be:

- build time below 25s on SIFT1M 100k
- Recall@10 no worse than the current `0.9860` without explicitly documenting
  the trade-off
- benchmark output clearly identifies segmented build settings
- current single-graph build and tests remain intact

A stronger result would be:

- build time below 15s on SIFT1M 100k
- Recall@10 at or above `0.9900`
- query QPS closer to the saved ChromaDB comparison

## Risks

- Recall can drop because each segment is searched independently.
- Query latency can increase because every query fans out to multiple graphs.
- Memory can increase because graph metadata is duplicated per segment.
- Contiguous segments may be worse than randomized segments on clustered data.
- Too many segments may reduce graph quality even if build time improves.
- Python thread orchestration may not help if the Cython wrapper holds the GIL
  during native build calls.

## Open Implementation Questions

- Whether the first implementation should use Python threads, Python processes,
  or a native C++ thread pool.
- Whether Cython needs `with nogil` around `cpp_build_graph()` before Python
  threads can overlap native builds.
- Whether segmented search should fan out sequentially first or parallelize in
  the same milestone.
- Whether segment assignment should be contiguous, deterministic shuffled, or
  profile-controlled.

These questions should be answered in the implementation plan with small,
testable chunks.

## Next Step

Write an implementation plan for an opt-in segmented build prototype. The plan
should start with tests and stats schema, then add a minimal sequential
segmented path, then add build parallelism only after the segmented correctness
boundary is verified.
