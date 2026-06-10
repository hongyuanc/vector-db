# HNSW 100k Build-Time Reduction Design

## Goal

Reduce HNSW index build time on the SIFT1M 100k benchmark before attempting
larger 1M-scale optimization work.

The immediate comparison target is ChromaDB because it uses hnswlib's native
C++ HNSW implementation under the hood. The goal is not to replace this project
with hnswlib. The goal is to understand which production techniques matter,
implement the most relevant ones in this codebase, and document the learning
path clearly.

## Current Baseline

The saved 100k ChromaDB comparison reports:

| System | Build Time | Batch QPS | Single p99 | Recall@10 |
|---|---:|---:|---:|---:|
| This DB, C++/CSR HNSW | 44.9216s | 4,768.04 | 1.2123ms | 98.60% |
| ChromaDB | 4.6508s | 6,372.68 | 0.5693ms | 99.90% |

The remaining gap is mostly build time. Search is closer, but build still takes
about 9.7x longer than ChromaDB on the same 100k-vector shape.

A fresh 10k synthetic benchmark on the current checkout showed:

| Phase | Time |
|---|---:|
| Total build | 3.1353s |
| Native C++ build total | 3.1033s |
| Native construction | 3.0986s |
| Native search inside construction | 2.3864s |
| Candidate search | 2.3158s |
| Greedy upper-layer search | 0.0707s |
| Pruning | 0.3891s |
| CSR export | 0.0046s |

This means the next work should focus on graph construction search and pruning,
not Python wrapper overhead or persistence.

## What Exists Now

The project already moved beyond the original pure-Python HNSW builder:

- `HNSWIndex.build()` dispatches to a native C++ batch builder when available.
- The native builder returns compact CSR graph layers for post-build search.
- Batch-built indexes avoid materializing Python connection sets for normal
  read-only search workloads.
- Euclidean ordering uses squared L2 internally and converts back to public L2
  distances only when returning results.
- Native build stats already track broad timing buckets such as construction
  search, greedy search, candidate search, pruning, and CSR export.
- Native search scratch memory already avoids repeated visited-array allocation.
- Heuristic neighbor selection is present for new-node outbound links.

These are meaningful production-style improvements. The next phase should avoid
repeating that work and instead narrow the remaining unknowns.

## Design Direction

Optimize 100k first, then use the same measurement and architecture to scale to
1M. This keeps each implementation chunk small enough to reason about and gives
fast feedback from benchmarks.

The recommended sequence is:

1. Improve instrumentation before changing behavior.
2. Optimize the measured single-threaded construction hot path.
3. Add configurable build-quality trade-offs once the baseline is understood.
4. Consider native build parallelism only after the single-threaded core is
   efficient enough to justify parallel work.

This order keeps the project educational. Each step should explain the
bottleneck, the change, why it matters, and the measured result.

## Success Targets

Primary benchmark:

- Dataset: SIFT1M 100k subset
- Metric: Euclidean
- HNSW config: `M=16`, `ef_construction=200`, `ef_search=100`
- Queries: 100
- Result count: `k=10`

Milestones:

| Milestone | Build Time | Recall@10 | Purpose |
|---|---:|---:|---|
| Baseline | 44.9216s | 98.60% | Current saved comparison |
| Milestone 1 | < 25s | >= 98.60% | Remove obvious measured waste |
| Milestone 2 | < 15s | >= 99.00% | Competitive single-thread target |
| Stretch | < 10s | >= 99.00% | Close enough to justify 1M work |

ChromaDB's saved 100k build time is 4.6508s, so matching it exactly may require
parallelism, lower-level SIMD, or deeper hnswlib-style memory layout changes.
The first goal is to close the largest gap without making the code opaque.

## Approach 1: Measured Single-Thread Optimization

This is the first approach to implement.

### Before

The native builder reports broad phase timings, but it does not explain why
candidate search dominates. It does not expose distance-call counts, visited
node counts, candidate expansions, per-layer costs, average prune input sizes,
or how often heuristic selection and reverse pruning recompute distances.

### Change

Extend native build stats with deeper counters:

- total distance evaluations during build
- distance evaluations from candidate search
- distance evaluations from neighbor selection
- distance evaluations from pruning
- visited nodes per candidate search
- candidate heap pushes and result heap updates
- per-layer search call counts and cumulative time
- average and max prune input size
- average and max selected degree

Then optimize only the largest measured bucket. Likely candidates are:

- reduce duplicate distance computation between search results, neighbor
  selection, and pruning
- avoid full sorting when only the top `M` or top `ef_construction` candidates
  are needed
- reuse candidate buffers returned by `search_mutable_layer()`
- reduce heap maintenance overhead in small bounded queues
- make pruning use the same heuristic consistently when the recall gain
  justifies the extra cost

### Why

Build time is dominated by repeated graph searches. Without better counters, an
optimization can make one phase faster while silently hurting recall or moving
work into another phase. The instrumentation makes each commit teach something
specific.

## Approach 2: Build-Quality Modes

This should come after the measured single-thread cleanup.

### Before

The benchmark uses one HNSW configuration: `M=16`, `ef_construction=200`,
`ef_search=100`. That is useful for fair comparison, but production databases
often expose trade-offs between indexing speed, memory, and recall.

### Change

Add explicit build profiles instead of ad hoc parameter changes:

- `high_recall`: current comparison shape, optimized for recall.
- `balanced`: lower construction work if recall remains acceptable.
- `fast_build`: lower construction work for ingestion-heavy use cases.

The public API should remain parameter-based. Profiles can live in benchmarks or
documentation first, then become API helpers only if they prove useful.

### Why

ChromaDB-style competitiveness is partly implementation speed and partly
operational tuning. This project should teach both, but tuning should not hide
real implementation bottlenecks. That is why profiles come after instrumentation
and core cleanup.

## Approach 3: Native Build Parallelism

This is a later approach, not the first step.

### Before

The HNSW build is order-sensitive. Each new vector is inserted into a graph that
depends on all earlier insertions. Naively inserting vectors in parallel can
change graph quality, recall, and determinism.

### Change

Investigate parallel work only after the single-threaded builder is cleaner.
Potential safe areas:

- parallel distance evaluation inside candidate expansion
- parallel batch ground-truth or benchmark support
- staged graph construction experiments with deterministic merge rules
- optional thread count for benchmark-only experiments

### Why

Production systems use CPU parallelism, but parallel construction is a design
change, not a small optimization. It should be introduced when the project can
measure whether the extra complexity pays for itself.

## Architecture

The Python layer remains the public boundary:

- API compatibility
- input validation
- benchmark orchestration
- persistence shape
- technical documentation
- tests

The C++ layer remains responsible for the hot HNSW read-index path:

- batch graph construction
- construction-time layer search
- neighbor selection
- pruning
- compact CSR export
- compact CSR search

The compact CSR storage model should remain intact:

- `graph_storage_mode == "compact_csr"` after batch build
- Python connection sets remain unmaterialized for read-only search
- save/load keeps compact CSR arrays available
- `insert()` and `delete()` may continue to materialize Python graph sets with a
  warning until native mutable graph support is designed separately

## Testing Strategy

Each implementation chunk should include focused tests before broad benchmarks:

- unit tests for new build-stat counters
- deterministic small-graph tests for public search behavior
- recall tests that avoid overfitting to exact incidental graph edges
- save/load tests confirming compact CSR ownership remains intact
- benchmark CLI output tests when the JSON or Markdown schema changes

Benchmark cadence:

- run small synthetic benchmarks during development for fast feedback
- run SIFT1M 100k comparison after meaningful native changes
- update saved benchmark artifacts only when the result is reproducible enough
  to document

## Documentation Strategy

`TECHNICAL.md` remains the long-form interview-review document. Each
optimization commit should add or update a short section covering:

- what existed before
- what changed
- why the change matters
- measured result
- next step

Benchmark artifacts should remain in `benchmarks/results/` when generated.
The design spec records the plan; `TECHNICAL.md` records the learning narrative.

## Risks and Trade-offs

- More C++ counters can clutter the API if exposed directly. Keep them grouped
  under build stats and benchmark reporting.
- Optimizing for 100k may not automatically solve 1M. The 100k work should
  capture scaling signals such as visited nodes and distance calls per vector.
- Faster pruning can reduce build time but may reduce graph quality. Recall must
  be measured with each pruning change.
- Build profiles can make results look better by lowering quality. They should
  be documented as explicit trade-offs, not presented as equivalent to the
  high-recall comparison configuration.
- Parallel construction can create nondeterminism and harder debugging. It
  should wait until the single-threaded design is measured and stable.

## Out of Scope

- Replacing the implementation with hnswlib, FAISS, Annoy, or another ANN
  backend
- Distributed indexing or sharding
- Product quantization or vector compression
- Native online mutation for compact CSR indexes
- Metadata filtering changes
- API server changes

## Next Step

Write an implementation plan for the first milestone:

1. Add deeper native build instrumentation.
2. Expose the new counters through the Cython wrapper.
3. Add focused tests for the benchmark schema.
4. Run a small synthetic benchmark and record what the counters show.
5. Use those results to choose the first actual hot-path optimization.
