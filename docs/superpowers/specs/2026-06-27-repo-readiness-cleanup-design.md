# Repository Readiness Cleanup Design

## Goal

Prepare the repository for new engineering work by removing small sources of
friction, making the project claims match the current implementation, and
rewriting the README after the code and workflow cleanup is complete.

This is an educational vector database project. The cleanup should preserve the
learning trail and make future work easier to reason about, not hide trade-offs
behind polished marketing language.

## Current State

The current implementation includes:

- Python collection, storage, metadata, and FastAPI layers.
- HNSW index implementations with Python, Cython, and C++ paths.
- Compact CSR graph storage for native HNSW search.
- A native C++ batch build path wrapped through Cython.
- An opt-in segmented HNSW build path for parallel native builds.
- Reproducible benchmark JSON and Markdown artifacts under
  `benchmarks/results/`.
- A long-form technical learning document in `TECHNICAL.md`.

The repository is mostly functional, but several small issues make future work
less clear:

- `README.md` mixes older Python/Cython benchmark claims with newer C++/CSR and
  segmented-build results.
- Some top-level language overstates the project as production-grade instead of
  educational and production-inspired.
- `Makefile` commands reference a missing `scripts/` directory and
  `docker/Dockerfile`, while the repository has a root `Dockerfile`.
- `pytest.mark.real_data` is used but not registered in `pyproject.toml`.
- Future benchmark results under `benchmarks/results/` can be easy to miss
  because the ignore rules still broadly ignore generated benchmark output.
- `dot_product` is described in some user-facing places as an HNSW/collection
  metric even though the HNSW index validates only `euclidean` and `cosine`.
- Package metadata still contains a placeholder repository URL.

## Decision

Use small engineering-cleanup chunks first, then rewrite the README last.

This keeps the final README honest: it should summarize the repository after the
workflow and capability clarity issues are fixed, rather than documenting those
issues as if they were intentional design decisions.

## Non-Goals

- Do not reorganize the whole source tree.
- Do not change HNSW search or build algorithms in this cleanup.
- Do not run long SIFT1M or ChromaDB benchmark suites unless a later tuning plan
  explicitly calls for it.
- Do not make segmented build the default.
- Do not remove historical benchmark discussion from `TECHNICAL.md`; clarify it
  where needed instead.
- Do not polish the project into marketing copy. Keep claims measurable and
  tied to benchmark artifacts.

## Cleanup Chunks

### Chunk 1: Developer Workflow Cleanup

What was there before: common commands have avoidable traps. `pytest` may not be
on the shell path, the Docker build target points at `docker/Dockerfile`, the
format target references a missing `scripts/` directory, and the `real_data`
marker raises warnings during test collection.

What to implement:

- Register the `real_data` pytest marker in `pyproject.toml`.
- Update Makefile test commands to use `python -m pytest`.
- Point `make docker-build` at the root `Dockerfile`.
- Remove the missing `scripts/` directory from formatting commands unless the
  directory is added later.
- Update package metadata that still uses placeholder project text or URLs.

Why: future changes should start from commands that behave predictably in a
fresh venv and do not produce avoidable warnings.

### Chunk 2: Benchmark Artifact and Capability Clarity

What was there before: curated benchmark artifacts are tracked, but future files
under `benchmarks/results/` can be obscured by broad ignore patterns. Supported
metric language is also inconsistent: distance utilities include dot product,
but the HNSW index and collection configuration currently support only Euclidean
and cosine.

What to implement:

- Adjust ignore rules so curated `benchmarks/results/*.json` and
  `benchmarks/results/*.md` can be added normally.
- Keep raw benchmark data, downloaded datasets, temporary outputs, and local
  generated files ignored.
- Align collection/API/README wording around supported HNSW metrics:
  `euclidean` and `cosine`.
- Mention `dot_product` only as a utility-level distance function unless HNSW
  support is actually added in a future feature.

Why: benchmark evidence is part of this project's learning record, and metric
claims should match the code path users can actually run.

### Chunk 3: Technical Documentation Update

What was there before: `TECHNICAL.md` contains the long learning history, but
some early sections still describe older goals and older benchmark assumptions.

What to implement:

- Add a new section documenting this repository-readiness cleanup.
- Explain what was cleaned, why it matters, and what remains intentionally
  unchanged.
- Clarify that the early 1M numbers are historical benchmark evidence unless
  they are rerun on the current C++/CSR path.
- Add a focused engineering-tuning plan for the numbers that are still not good
  enough.

Why: the technical document should remain useful for interview review and for
resuming the project later.

### Chunk 4: README Rewrite Last

What was there before: the README opens with stale production-grade language
and old headline performance tables, then later discusses newer C++/CSR and
segmented-build work. That makes the repository look less coherent than it is.

What to implement:

- Reframe the project as an educational vector database built from scratch.
- Put the current architecture first: Python API/storage, Cython utilities,
  C++/CSR HNSW core, and optional segmented builds.
- Present benchmark numbers with clear provenance:
  - current tracked 100k C++/CSR versus ChromaDB comparison
  - current tracked segmented-build trade-off
  - historical 1M result labeled as historical unless rerun
- Add clean setup, test, benchmark, and project-structure sections.
- Add a concise "Current Engineering Gaps" section.

Why: the README should be the final public entry point after the repo itself is
less surprising.

## Engineering Tuning Plan

Some numbers are still not good enough to treat the implementation as
production-competitive.

### Build Time

Current evidence: the tracked 100k ChromaDB comparison shows the C++/CSR path
building slower than ChromaDB. Segmented build reduces wall-clock build time
substantially, but it does so by building independent graphs and changing the
search trade-off.

Plan:

- Keep segmented build opt-in.
- Tune per-segment search overfetch and merge behavior before recommending
  segmented mode for query-heavy workloads.
- Profile the native builder around candidate search and pruning before adding
  more threading.
- Avoid same-graph parallel insertion until there is a separate design, because
  HNSW construction is order-dependent and shared mutation is risky.

### Query Throughput

Current evidence: the native C++/CSR batch search path narrowed the gap, but
ChromaDB remains faster in tracked 100k comparisons.

Plan:

- Profile distance computation inside native search.
- Evaluate SIMD-friendly distance kernels for squared L2.
- Reduce per-query fanout cost in segmented search before making segmented mode
  more visible.
- Keep measuring recall and latency together; do not optimize QPS alone.

### Benchmark Provenance

Current evidence: some benchmark artifacts were generated with dirty worktrees
or older commits, and that is already documented for segmented results.

Plan:

- Prefer benchmark artifacts produced from clean commits.
- Keep JSON and Markdown outputs together.
- Record git commit, dirty state, environment, HNSW config, and dataset size in
  every benchmark artifact.
- Update README headline numbers only from tracked artifacts with clear
  provenance.

## Testing Strategy

For cleanup work, use focused verification instead of long benchmark runs:

- `python -m pytest tests/test_*.py -q -m "not slow and not benchmark"`
- `python -m pytest --collect-only -q` to confirm pytest markers no longer warn.
- `make test` if the Makefile changes are part of the chunk.
- `make docker-build` only if Docker availability is confirmed or the user asks
  for container validation.
- No full SIFT1M or production database comparison runs unless a performance
  tuning plan explicitly requires them.

## Acceptance Criteria

- The repository has no unrelated dirty tracked changes after each chunk.
- `AGENTS.md` is tracked if the project should keep those instructions.
- Developer commands in the README and Makefile match the actual repository
  layout.
- Test collection does not warn about an unknown `real_data` marker.
- Supported HNSW metric claims match the implementation.
- Curated benchmark artifacts under `benchmarks/results/` can be added without
  force-add workarounds.
- `README.md` presents current results honestly and marks historical numbers as
  historical.
- `TECHNICAL.md` records the cleanup with "what was there before", "what was
  implemented", "why", and "next steps".
