# HNSW C++ Parity Design

## Goal

Close the remaining ChromaDB performance and accuracy gap for batch-built,
read-optimized HNSW indexes while keeping the project educational and preserving
the existing Python public API.

## Current Baseline

The current comparison target is the saved SIFT1M 100k ChromaDB benchmark:

| System | Build Time | Batch QPS | Batch Avg Latency | Single p99 | Recall@10 |
|--------|-----------:|----------:|------------------:|-----------:|----------:|
| Our HNSW C++/CSR | 49.79s | 4,266.4 | 0.2344ms | 1.0822ms | 98.50% |
| ChromaDB | 4.83s | 6,476.6 | 0.1544ms | 0.5752ms | 99.70% |

The largest gap is build time. Search is closer, but still slower in batch
average latency and single-query tail latency. Recall is strong, but still below
ChromaDB with identical public HNSW parameters.

## Scope

This phase optimizes only the batch-built read index path:

- `HNSWIndex.build()`
- the native C++ graph builder in `src/index/hnsw_cpp_core.cpp`
- compact CSR graph export and search
- benchmark instrumentation and documentation

Online mutation is explicitly out of scope. `insert()` and `delete()` will keep
the existing compatibility behavior: mutating a compact CSR index materializes
Python connection sets and emits a warning. This avoids mixing native build
parity work with a separate native mutable graph design.

## Architecture

The Python layer remains the product boundary. Python owns validation, the
public `HNSWIndex` API, persistence shape, collection integration, tests, and
benchmark reporting. The C++ layer owns the hot path for batch construction and
compact-CSR batch search.

The C++ implementation will evolve from a straightforward educational builder
into a more production-like HNSW core in small commit-sized steps. Each step
should keep behavior observable through tests and benchmark artifacts.

## Proposed Optimization Track

### 1. Native Phase Instrumentation

Add build/search phase timing to identify where time is spent inside the C++
extension. The useful build phases are:

- level assignment in Python
- native insertion and upper-layer traversal
- layer-0 construction search
- neighbor selection and pruning
- duplicate checks and adjacency mutation
- CSR export
- Python wrapper conversion

This prevents optimizing by assumption. Each later change should compare against
the same JSON artifact schema.

### 2. Squared L2 Distance Internals

Euclidean HNSW ordering does not require `sqrt` during traversal, candidate
comparison, or pruning. Replace internal L2 distance with squared L2 for
Euclidean metric ordering, then convert to public L2 distance only when returning
results.

This should reduce build and search CPU cost without changing nearest-neighbor
ordering.

### 3. Reusable Native Scratch Memory

The current C++ search path allocates a fresh `visited` array and heap storage
for each layer search. Replace repeated allocation with reusable scratch state:

- generation-mark visited arrays
- preallocated candidate/result buffers
- scratch reset by incrementing a visit generation instead of clearing memory

This targets both build time and search latency because HNSW performs many small
graph searches.

### 4. Lower-Overhead Mutable Adjacency

The native builder currently stores adjacency as nested vectors and uses linear
duplicate checks. Replace this with a layout tuned for HNSW's bounded degree:

- compact per-node, per-layer connection arrays
- explicit connection counts
- duplicate checks over bounded arrays
- fewer vector reallocations during insertion and pruning

Layer 0 still allows `2M` connections, while upper layers allow `M`.

### 5. HNSW Heuristic Neighbor Selection

The current builder selects the nearest candidates directly. Add the HNSW
diversity heuristic used by production implementations: candidates are processed
by distance, and a candidate is accepted when it is not made redundant by already
selected neighbors.

This should improve graph navigability and recall at the same
`ef_construction`. If recall improves enough, later benchmarks can test lower
`ef_construction` values for faster builds.

### 6. Native Batch Search Cleanup and Optional Threading

Keep single-query behavior deterministic and simple. For `search_batch()`, remove
repeated per-query setup where possible and then add optional native threading
only for batch queries. Threading should be opt-in or controlled by an explicit
argument/environment variable so benchmark comparisons remain clear.

## Success Targets

The first target is SIFT1M 100k with `M=16`, `ef_construction=200`,
`ef_search=100`, `k=10`, and 100 queries:

| Metric | Current | Target | Stretch |
|--------|--------:|-------:|--------:|
| Build Time | 49.79s | < 15s | < 8s |
| Recall@10 | 98.50% | >= 99.30% | >= 99.70% |
| Batch QPS | 4,266 | >= 6,000 | >= 6,500 |
| Batch Avg Latency | 0.2344ms | <= 0.1700ms | <= 0.1550ms |

The secondary target is keeping the compact CSR storage shape intact:

- `graph_storage_mode == "compact_csr"` after batch build
- Python graph edges remain unmaterialized after batch build and load
- save/load behavior remains compatible with current compact CSR files

## Testing Strategy

Each implementation chunk should include focused tests before the implementation:

- unit tests for new native behavior exposed through Cython wrappers
- parity tests that compare native results to existing Python/C++ behavior on
  small deterministic graphs
- save/load tests confirming compact CSR ownership is preserved
- benchmark CLI or ChromaDB comparison artifacts after meaningful performance
  changes

Long SIFT benchmarks should not be required for every unit-level change, but
each optimization milestone should record a benchmark artifact and update the
technical documentation.

## Documentation Strategy

`TECHNICAL.md` remains the main interview-review narrative. Each commit-sized
optimization should add a short section explaining:

- what existed before
- what changed
- why the change matters
- measured result
- next step

Benchmark artifacts should remain in `benchmarks/results/` when generated.

## Risks and Trade-offs

- Deeper C++ work makes the code harder to inspect than the original Python
  implementation, so each step needs a clear educational explanation.
- Squared L2 distance must preserve public distance values for search results,
  even if internal ordering uses squared distances.
- The HNSW diversity heuristic can improve recall but may change exact neighbor
  choices on small deterministic tests. Tests should assert invariants and recall
  behavior instead of overfitting to incidental graph edges.
- Native batch threading can improve QPS but can make benchmarks noisier. It
  should be introduced after single-threaded native inefficiencies are reduced.

## Out of Scope

- Replacing this project with hnswlib, FAISS, Annoy, or another ANN backend
- Native online insertion/deletion for compact CSR indexes
- Distributed indexing or sharding
- Quantization and vector compression
- Metadata filtering changes
