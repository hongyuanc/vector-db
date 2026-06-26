# Repository Readiness Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the repository easier to extend by cleaning developer workflow, benchmark artifact handling, capability claims, technical documentation, and the README.

**Architecture:** Keep source layout stable. Apply small, commit-sized cleanup chunks that improve command reliability and documentation accuracy before rewriting the README around the current C++/CSR and segmented HNSW implementation.

**Tech Stack:** Python 3.11, pytest, Cython, C++, FastAPI, NumPy, Makefile, Markdown documentation.

---

## File Map

- `AGENTS.md`: track project-local instructions that were previously untracked.
- `pyproject.toml`: package description and pytest marker registration.
- `setup.py`: package description and repository URL used by legacy setuptools invocation.
- `Makefile`: local developer commands.
- `.gitignore`: benchmark artifact allow rules.
- `src/__init__.py`: package-level description.
- `src/api/server.py`: FastAPI description string.
- `src/collection/collection.py`: collection metric documentation.
- `src/index/__init__.py`: public index package exports.
- `tests/test_index_exports.py`: focused test for exported index types and metric support boundary.
- `TECHNICAL.md`: learning-log update for this cleanup and tuning plan.
- `README.md`: final public project overview after engineering cleanup.

---

### Task 1: Developer Workflow And Metadata Cleanup

**Files:**
- Modify: `AGENTS.md`
- Modify: `pyproject.toml`
- Modify: `setup.py`
- Modify: `Makefile`
- Modify: `src/__init__.py`
- Modify: `src/api/server.py`

- [ ] **Step 1: Confirm baseline marker warning**

Run:

```bash
./venv/bin/python -m pytest --collect-only -q
```

Expected before the fix: collection succeeds, but warnings include `PytestUnknownMarkWarning` for `pytest.mark.real_data`.

- [ ] **Step 2: Track project instructions**

Stage the existing `AGENTS.md` file as part of this chunk. Do not rewrite its policy content in this task.

Run later:

```bash
git add AGENTS.md
```

- [ ] **Step 3: Update package metadata and marker registration**

In `pyproject.toml`, change the project description to:

```toml
description = "Educational vector database with HNSW indexing built from scratch"
```

In `pyproject.toml`, add `real_data` to the pytest markers list:

```toml
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "very_slow: marks tests as very slow long-running benchmarks",
    "integration: marks tests as integration tests",
    "benchmark: marks tests as benchmarks",
    "real_data: marks tests that require downloaded benchmark datasets",
]
```

- [ ] **Step 4: Update setuptools metadata**

In `setup.py`, change:

```python
description="A production-grade vector database with HNSW indexing",
```

to:

```python
description="Educational vector database with HNSW indexing built from scratch",
```

Change:

```python
url="https://github.com/yourusername/vector-db",
```

to:

```python
url="https://github.com/hongyuanc/vector-db",
```

- [ ] **Step 5: Update package and API descriptions**

Replace `src/__init__.py` with:

```python
"""
Vector DB - an educational vector database with HNSW indexing.
"""

__version__ = "0.1.0"
```

In `src/api/server.py`, change the FastAPI description from:

```python
description="Production-grade vector database with HNSW indexing",
```

to:

```python
description="Educational vector database with HNSW indexing",
```

- [ ] **Step 6: Fix Makefile developer commands**

Update `Makefile` by adding the `PYTHON` variable after the `.PHONY` line:

```makefile
PYTHON ?= python
```

Change the test targets to:

```makefile
test:
	$(PYTHON) -m pytest tests/test_*.py -q -m "not slow and not benchmark"

test-cov:
	$(PYTHON) -m pytest tests/test_*.py -q -m "not slow and not benchmark" --cov=src --cov-report=term-missing --cov-report=html
```

Change the format target to:

```makefile
format:
	black src/ tests/ benchmarks/
	ruff check --fix src/ tests/ benchmarks/
```

Change the Docker build target to:

```makefile
docker-build:
	docker build -t vector-db:latest -f Dockerfile .
```

Change the benchmark target to:

```makefile
benchmark:
	$(PYTHON) benchmarks/benchmark.py
```

- [ ] **Step 7: Verify workflow cleanup**

Run:

```bash
./venv/bin/python -m pytest --collect-only -q
```

Expected: collection succeeds without `PytestUnknownMarkWarning`.

Run:

```bash
make test
```

Expected: the fast non-benchmark unit suite runs through the Makefile.

- [ ] **Step 8: Commit workflow cleanup**

Run:

```bash
git add AGENTS.md pyproject.toml setup.py Makefile src/__init__.py src/api/server.py
git commit -m "chore: clean developer workflow metadata"
```

---

### Task 2: Benchmark Artifact And Metric Boundary Cleanup

**Files:**
- Create: `tests/test_index_exports.py`
- Modify: `.gitignore`
- Modify: `src/index/__init__.py`
- Modify: `src/collection/collection.py`

- [ ] **Step 1: Add a failing export and metric-boundary test**

Create `tests/test_index_exports.py`:

```python
import pytest

from src.index import BruteForceIndex, HNSWIndex, SegmentedHNSWIndex, VectorIndex


def test_index_package_exports_public_index_types():
    assert issubclass(BruteForceIndex, VectorIndex)
    assert HNSWIndex.__name__ == "HNSWIndex"
    assert SegmentedHNSWIndex.__name__ == "SegmentedHNSWIndex"


def test_hnsw_index_rejects_dot_product_metric():
    with pytest.raises(ValueError, match="Unsupported metric"):
        HNSWIndex(metric="dot_product")
```

- [ ] **Step 2: Run the new test to verify it fails**

Run:

```bash
./venv/bin/python -m pytest tests/test_index_exports.py -q
```

Expected before the export fix: FAIL with an import error for `BruteForceIndex`, `HNSWIndex`, or `VectorIndex` from `src.index`.

- [ ] **Step 3: Export public index types**

Replace `src/index/__init__.py` with:

```python
"""
Index implementations for vector search.
"""

from .base import VectorIndex
from .brute_force import BruteForceIndex
from .hnsw import HNSWIndex
from .segmented_hnsw import SegmentedHNSWIndex

__all__ = [
    "VectorIndex",
    "BruteForceIndex",
    "HNSWIndex",
    "SegmentedHNSWIndex",
]
```

- [ ] **Step 4: Clarify collection-supported metrics**

In `src/collection/collection.py`, change the `metric` argument docstring from:

```python
metric: Distance metric ("euclidean", "cosine", "dot_product")
```

to:

```python
metric: HNSW distance metric ("euclidean" or "cosine")
```

- [ ] **Step 5: Make curated benchmark artifacts addable**

In `.gitignore`, replace the benchmark section:

```gitignore
# Benchmarks and results
benchmarks/results/
benchmarks/datasets/
*.csv
*.json
!pyproject.toml
!package.json
```

with:

```gitignore
# Benchmarks and results
benchmarks/datasets/
*.csv
*.json
!pyproject.toml
!package.json

# Curated benchmark artifacts are part of the learning record.
!benchmarks/results/
!benchmarks/results/*.json
!benchmarks/results/*.md
```

- [ ] **Step 6: Verify test and ignore behavior**

Run:

```bash
./venv/bin/python -m pytest tests/test_index_exports.py -q
```

Expected: PASS.

Run:

```bash
git check-ignore --no-index benchmarks/results/example-future-result.json
```

Expected: no output and exit code `1`, meaning a future curated result file is not ignored.

Run:

```bash
git check-ignore --no-index scratch-result.json
```

Expected: output shows the repository-wide `*.json` ignore rule still catches local JSON files outside curated benchmark artifacts.

- [ ] **Step 7: Commit metric and artifact cleanup**

Run:

```bash
git add .gitignore src/index/__init__.py src/collection/collection.py tests/test_index_exports.py
git commit -m "chore: clarify index exports and benchmark artifacts"
```

---

### Task 3: Technical Documentation Update

**Files:**
- Modify: `TECHNICAL.md`

- [ ] **Step 1: Insert repository-readiness learning section**

Insert this section immediately before `## Future Improvements` in `TECHNICAL.md`:

```markdown
## Repository Readiness Cleanup

### What Was There Before

The codebase had grown from a pure Python HNSW prototype into a mixed
Python/Cython/C++ implementation with compact CSR graph storage and an opt-in
segmented build path. The implementation had moved forward, but some repository
surface area still described older assumptions:

- the README emphasized older Python/Cython benchmark numbers before the newer
  C++/CSR results
- package and API descriptions called the project production-grade instead of
  educational and production-inspired
- developer commands referenced paths that did not exist in the current tree
- `pytest.mark.real_data` was used without being registered
- future benchmark artifacts under `benchmarks/results/` could be easy to miss
  because broad ignore rules still ignored generated JSON files
- `dot_product` appeared in some collection-facing metric descriptions even
  though HNSW currently accepts only `euclidean` and `cosine`

### What Was Implemented

The cleanup made the repository easier to extend without changing the HNSW
algorithms:

- registered the `real_data` pytest marker
- fixed Makefile commands so tests, formatting, benchmarking, and Docker builds
  match the current repository layout
- updated package and API descriptions to describe the project as educational
- tracked the project-local `AGENTS.md` instructions
- made public index exports explicit from `src.index`
- documented the supported HNSW metric boundary in code-facing text
- adjusted ignore rules so curated benchmark JSON and Markdown files under
  `benchmarks/results/` can be added normally
- rewrote the README after the cleanup so current benchmark claims have clear
  provenance

### Why This Matters

This project is meant to teach vector database internals. That means the
repository should make trade-offs visible instead of smoothing them over. The
cleanup separates historical benchmark evidence from current implementation
claims, makes future benchmark artifacts easier to preserve, and reduces small
workflow surprises before adding new features.

### Remaining Engineering Gaps

The current C++/CSR path is much stronger than the original Python/Cython path,
but it is not production-competitive with ChromaDB on all dimensions.

Build time remains the largest gap. Segmented build can reduce wall-clock build
time by building independent graphs in parallel, but query throughput drops as
segment count increases because each query fans out across more graphs before
merging global top-k results.

Search throughput is closer to ChromaDB than before, but tracked 100k benchmark
artifacts still show ChromaDB ahead. The next useful tuning work should focus
on native distance-kernel profiling, SIMD-friendly squared L2 computation,
per-segment overfetch and merge behavior, and reducing segmented query fanout
cost. Same-graph parallel insertion should remain a separate design because
HNSW construction is order-dependent.

### Next Steps

- Keep segmented build opt-in until query-throughput trade-offs are better
  understood.
- Use clean-commit benchmark runs for future headline numbers.
- Record both JSON and Markdown benchmark artifacts for performance changes.
- Add new technical documentation sections alongside each future feature or
  optimization so the learning record remains current.

---

```

- [ ] **Step 2: Verify the section is present**

Run:

```bash
rg -n "Repository Readiness Cleanup|Remaining Engineering Gaps|Next Steps" TECHNICAL.md
```

Expected: output includes all three headings or phrases from the inserted section.

- [ ] **Step 3: Commit technical documentation update**

Run:

```bash
git add TECHNICAL.md
git commit -m "docs: record repository readiness cleanup"
```

---

### Task 4: README Rewrite Last

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Replace README with current project summary**

Replace `README.md` with the content in Appendix A at the bottom of this plan.

- [ ] **Step 2: Verify README references current artifacts and avoids stale claims**

Run:

```bash
rg -n "production-grade|dot_product|docker/Dockerfile|scripts/|48x faster|core algorithm remains Python" README.md
```

Expected: no output.

Run:

```bash
rg -n "C\\+\\+/CSR|Segmented|Historical 1M|Current Engineering Gaps|benchmarks/results/sift1m-100k-chromadb-comparison.md" README.md
```

Expected: output includes the current implementation and benchmark provenance sections.

- [ ] **Step 3: Run final focused verification**

Run:

```bash
./venv/bin/python -m pytest tests/test_*.py -q -m "not slow and not benchmark"
```

Expected: PASS.

Run:

```bash
./venv/bin/python -m pytest --collect-only -q
```

Expected: collection succeeds without unknown-marker warnings.

- [ ] **Step 4: Commit README rewrite**

Run:

```bash
git add README.md
git commit -m "docs: refresh project readme"
```

---

## Appendix A: Replacement README

````markdown
# Vector Database

An educational vector database built from scratch to understand how approximate
nearest-neighbor systems work under the hood. The project implements HNSW
indexing, vector storage, metadata filtering, persistence, a FastAPI service
layer, Cython helpers, and a native C++/CSR HNSW path.

This repository is intentionally learning-focused. It is not a wrapper around
FAISS or hnswlib; the core data structures and benchmark tooling are implemented
directly so the performance trade-offs are visible.

## Current Status

The current implementation includes:

- HNSW approximate nearest-neighbor search
- compact CSR graph storage for native search
- C++ batch build and batch search paths wrapped through Cython
- optional segmented HNSW builds for parallel native construction
- memory-mapped vector storage
- SQLite-backed metadata storage and soft deletes
- FastAPI endpoints for collection creation, insert, search, and index build
- benchmark tooling that records JSON and Markdown artifacts

The strongest current path is the C++/CSR HNSW implementation. Segmented build
is available as an opt-in benchmarking mode because it improves build time but
changes query throughput trade-offs.

## Architecture

```text
FastAPI service
    |
Collection
    |
    +-- VectorStore: memory-mapped float32 vector data
    +-- MetadataStore: SQLite metadata and tombstones
    +-- HNSWIndex: Python API with Cython/C++ acceleration
            |
            +-- compact CSR graph cache for native search
            +-- optional SegmentedHNSWIndex wrapper
```

Python owns the public API, collection orchestration, storage, metadata, and
persistence compatibility. Cython and C++ own the hot HNSW build/search paths
where the Python object model would otherwise dominate runtime.

## Performance Snapshot

Benchmark numbers are hardware- and configuration-dependent. The tracked
artifacts in `benchmarks/results/` are the source of truth for the current
headline numbers.

### Current 100k C++/CSR Comparison

Tracked artifact:
`benchmarks/results/sift1m-100k-chromadb-comparison.md`

Dataset and configuration:

- Dataset: SIFT1M 100k subset
- Queries: 100
- Metric: Euclidean
- HNSW: `M=16`, `ef_construction=200`, `ef_search=100`

| System | Build Time | Batch QPS | Batch Avg Latency | Single p99 | Recall@10 |
|---|---:|---:|---:|---:|---:|
| This DB, C++/CSR | 44.9216s | 4,768.04 | 0.2097ms | 1.2123ms | 0.9860 |
| ChromaDB | 4.6508s | 6,372.68 | 0.1569ms | 0.5693ms | 0.9990 |

Takeaway: the native path made search much closer to production libraries, but
build time remains the largest gap.

### Segmented Build Trade-off

Tracked artifacts:
`benchmarks/results/hnsw-segmented-sift100k-{2,4,8}.md`

| Mode | Build Time | Recall@10 | QPS | p99 Latency |
|---|---:|---:|---:|---:|
| Single graph baseline | 40.1198s | 0.9860 | 3,378.48 | not recorded in table |
| 2 segments / 2 threads | 18.1246s | 0.9940 | 1,900.17 | 0.7728ms |
| 4 segments / 4 threads | 8.1387s | 0.9980 | 1,035.96 | 1.4544ms |
| 8 segments / 8 threads | 4.6599s | 0.9940 | 570.12 | 2.4883ms |

Takeaway: segmented build gives a large wall-clock build-time win, but every
query searches more independent graphs before merging results. It should remain
opt-in until the query-throughput trade-off is tuned.

### Historical 1M Result

The earlier Python/Cython-era 1M SIFT1M run is retained as historical scale
evidence, not as the current C++/CSR headline result:

| Vectors | Recall@10 | QPS | Avg Latency | Build Time |
|---|---:|---:|---:|---:|
| 1M | 93.7% | 675 | 1.48ms | 66.2min |

A clean current 1M C++/CSR benchmark should be rerun before using 1M numbers as
headline performance claims.

## Quick Start

### Local Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python setup.py build_ext --inplace
```

### Run The API

```bash
python -m uvicorn src.api.server:app --host 0.0.0.0 --port 8000
```

Interactive API docs will be available at:

```text
http://localhost:8000/docs
```

### Docker

```bash
docker-compose up -d
```

## Basic Usage

Create a collection:

```bash
curl -X POST "http://localhost:8000/collections/create?dimension=384&name=default"
```

Insert a vector:

```bash
curl -X POST "http://localhost:8000/insert" \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [0.1, 0.2, 0.3],
    "metadata": {"title": "Document 1", "category": "tech"}
  }'
```

Build the HNSW index:

```bash
curl -X POST "http://localhost:8000/collections/default/build_index"
```

Search:

```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [0.1, 0.2, 0.3],
    "k": 10,
    "filter": {"category": "tech"}
  }'
```

## Development

Run the main test suite:

```bash
make test
```

Run focused non-benchmark tests:

```bash
python -m pytest tests/test_*.py -q -m "not slow and not benchmark"
```

Run linting and formatting:

```bash
make lint
make format
```

Build Cython/C++ extensions:

```bash
python setup.py build_ext --inplace
```

## Benchmarking

Run a synthetic benchmark without downloading external datasets:

```bash
python benchmarks/benchmark.py \
  --dataset random \
  --size 10000 \
  --dimension 128 \
  --queries 100 \
  --k 10 \
  --ef-search 50 \
  --output benchmarks/results/random-10k.json \
  --markdown-output benchmarks/results/random-10k.md
```

Compare two benchmark reports:

```bash
python benchmarks/compare_results.py \
  benchmarks/results/baseline.json \
  benchmarks/results/candidate.json \
  --output benchmarks/results/comparison.md
```

Run SIFT1M benchmarks only after downloading the dataset:

```bash
python tests/benchmarks/download_datasets.py --sift
python -m pytest tests/benchmarks/test_real_world.py::TestRealWorldDatasets::test_sift1m_small -v -s
```

Long-running production comparisons live under
`tests/benchmarks/test_chromadb_comparison.py`.

## Project Structure

```text
vector-db/
├── src/
│   ├── api/              # FastAPI models and server
│   ├── collection/       # Collection orchestration
│   ├── index/            # HNSW, segmented HNSW, brute force, Cython/C++ core
│   ├── storage/          # Vector and metadata stores
│   └── utils/            # Distance metrics
├── benchmarks/           # Reproducible benchmark runner and comparisons
├── benchmarks/results/   # Curated benchmark JSON/Markdown artifacts
├── docs/superpowers/     # Design specs and implementation plans
├── tests/                # Unit tests and benchmark tests
├── TECHNICAL.md          # Long-form technical learning document
├── Dockerfile
├── docker-compose.yml
├── Makefile
└── setup.py
```

## Supported HNSW Metrics

The HNSW index currently supports:

- `euclidean`
- `cosine`

The distance utility module also contains a dot-product helper, but dot product
is not currently accepted as an HNSW index metric.

## Current Engineering Gaps

Build time is still the largest gap versus production libraries. The tracked
100k comparison shows ChromaDB building much faster than this implementation.

Segmented build improves wall-clock build time, but query throughput drops as
segment count increases. The next useful work is to tune per-segment overfetch,
merge behavior, and query fanout cost before recommending segmented mode beyond
build-heavy experiments.

Search throughput is closer than it was before the C++/CSR path, but ChromaDB
still leads. Native distance-kernel profiling and SIMD-friendly squared L2
computation are the next likely optimization areas.

## Learning Notes

The long-form project explanation lives in `TECHNICAL.md`. It records what was
there before each major change, what was implemented, why it was implemented,
and what should happen next.
````
