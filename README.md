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
curl -X POST "http://localhost:8000/collections/create?dimension=3&name=default"
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
|-- src/
|   |-- api/              # FastAPI models and server
|   |-- collection/       # Collection orchestration
|   |-- index/            # HNSW, segmented HNSW, brute force, Cython/C++ core
|   |-- storage/          # Vector and metadata stores
|   `-- utils/            # Distance metrics
|-- benchmarks/           # Reproducible benchmark runner and comparisons
|-- benchmarks/results/   # Curated benchmark JSON/Markdown artifacts
|-- docs/superpowers/     # Design specs and implementation plans
|-- tests/                # Unit tests and benchmark tests
|-- TECHNICAL.md          # Long-form technical learning document
|-- Dockerfile
|-- docker-compose.yml
|-- Makefile
`-- setup.py
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
