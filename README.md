# Vector Database

A production-grade vector database built from scratch with HNSW indexing for fast similarity search. Supports millions of high-dimensional vectors with sub-millisecond query latency.

## Why This Project?

This project was built as an educational deep-dive into how modern vector databases like Pinecone, ChromaDB, and Weaviate work under the hood. Rather than using existing libraries (FAISS, Annoy), every component was implemented from scratch to understand the fundamental algorithms, data structures, and optimization techniques that power semantic search at scale.

## The Journey

The implementation evolved through multiple optimization phases:

**Phase 1: Pure Python** - Initial implementation using Python and NumPy. Simple and clear but slow, achieving only ~600 QPS on 10k vectors due to Python's interpreter overhead in the core search loops.

**Phase 2: Cython Optimization** - Rewrote performance-critical code (distance calculations, search loops) in Cython, compiling to C while maintaining Python's development ergonomics. This brought a 3x speedup (build: 17.6s → 5.8s, search: 600 → 1,810 QPS), demonstrating that 70-80% of C++ performance is achievable without a full rewrite.

**Phase 3: Competitive Benchmarking** - Benchmarked against production systems (ChromaDB, Qdrant) on SIFT1M to validate the approach and understand the remaining performance gap. ChromaDB's pure C++ with SIMD instructions is 4-5x faster, but the gap narrows at scale, showing good algorithmic scaling properties.

## Features

- **HNSW Algorithm**: Hierarchical Navigable Small World graphs for approximate nearest neighbor search
- **High Performance**: Sub-10ms p99 latency, 500+ QPS on single machine
- **Multiple Metrics**: Cosine similarity, Euclidean distance, dot product
- **Metadata Filtering**: Hybrid search with SQLite-backed metadata storage
- **REST API**: FastAPI server with OpenAPI documentation
- **Docker Ready**: Production deployment with docker-compose

## Performance

Benchmarked on SIFT1M (industry-standard dataset):

| Vectors | Recall@10 | QPS | Avg Latency | Build Time |
|---------|-----------|-----|-------------|------------|
| 10k | 99.5% | 1,810 | 0.55ms | 13.2s |
| 100k | 98.1% | 888 | 1.13ms | 4.5min |
| 1M | 93.7% | 675 | 1.48ms | 66.2min |

**Test Environment:** Apple M4 Pro, 24GB RAM, Python 3.11 with Cython
**Configuration:** M=16, ef_construction=200, ef_search=100

**Scalability Insights:**
- Query latency scales logarithmically: 10k→1M (100x vectors) adds only 0.93ms
- QPS decreases sub-linearly: 100x more vectors = only 2.7x slower
- Build time is O(n log n): scales from 13s to 66min for 100x data

### Competitive Comparison - Production Vector Databases

Head-to-head comparison on SIFT1M (10k subset, identical parameters):

| System | Recall@10 | QPS | Latency | Build Time | Implementation |
|--------|-----------|-----|---------|------------|----------------|
| **ChromaDB** | **100.0%** | **9,234** | **0.11ms** | **0.34s** | C++ hnswlib + SIMD |
| **This DB** | 99.5% | 1,810 | 0.55ms | 13.20s | Python + Cython |
| Qdrant | 100.0% | 621 | 1.61ms | 0.79s | Rust |

**Test Environment:** Apple M4 Pro, 24GB RAM, Python 3.11, in-memory mode
**Parameters:** M=16, ef_construction=200, ef_search=100

**Key Insights:**
- ChromaDB leads with C++/SIMD optimization (5x faster search, 39x faster build)
- Our implementation shows 2.9x higher QPS than Qdrant in Python client mode
- All systems achieve strong recall (99.5%+) with identical HNSW parameters
- Build time gap highlights Python overhead in graph construction

**Why the Performance Gap?**
- **ChromaDB** uses hnswlib (pure C++ with SIMD/AVX instructions)
- **This implementation** uses Cython for hot loops but core algorithm remains Python
- **Qdrant** in `:memory:` mode with Python client has serialization overhead; production Qdrant server would likely perform differently

### Scaling to 1M Vectors

Full SIFT1M dataset (1M vectors) comparison:

| System | Recall@10 | QPS | Latency | Build Time | Implementation |
|--------|-----------|-----|---------|------------|----------------|
| **ChromaDB** | **98.8%** | **2,902** | **0.34ms** | **1.1 min** | C++ hnswlib + SIMD |
| **This DB** | 93.7% | 675 | 1.48ms | 66.2 min | Python + Cython |

**Key Insights:**
- ChromaDB maintains 4.3x faster search at scale (2902 vs 675 QPS)
- Build time gap widens to 60x (1.1 min vs 66.2 min) due to O(n log n) complexity
- Both systems maintain strong recall (93.7%+ at 1M scale)
- Our HNSW scales reasonably: 10k→1M (100x data) only reduces QPS by 2.7x

Note: Qdrant's Python `:memory:` client degrades significantly beyond 20k vectors and is not viable for 1M scale benchmarking.

See [Benchmark Methodology](#running-benchmarks) for reproduction steps.

## Quick Start

### Using Docker (Recommended)

```bash
# Start the server
docker-compose up -d

# Access interactive API docs
open http://localhost:8000/docs
```

### Local Installation

```bash
# Clone and setup
git clone <repo-url>
cd vector-db
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Build Cython/C++ extensions
python setup.py build_ext --inplace

# Start server
python -m uvicorn src.api.server:app --host 0.0.0.0 --port 8000
```

## Basic Usage

### 1. Create a Collection

```bash
curl -X POST "http://localhost:8000/collections/create?dimension=384&name=default"
```

### 2. Insert Vectors

```bash
curl -X POST "http://localhost:8000/insert" \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [0.1, 0.2, ...],
    "metadata": {"title": "Document 1", "category": "tech"}
  }'
```

### 3. Build Index

```bash
curl -X POST "http://localhost:8000/collections/default/build_index"
```

### 4. Search

```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [0.1, 0.2, ...],
    "k": 10,
    "filter": {"category": "tech"}
  }'
```

## Running Benchmarks

### Reproducible Metrics Report

Use the top-level benchmark runner when you want a clean JSON/Markdown artifact
for comparing changes across commits:

```bash
# Synthetic benchmark, no downloads required
python benchmarks/benchmark.py \
  --dataset random \
  --size 10000 \
  --dimension 128 \
  --queries 100 \
  --k 10 \
  --ef-search 50 \
  --output benchmarks/results.json \
  --markdown-output benchmarks/results.md
```

The JSON output records the dataset, HNSW configuration, git commit, dirty
worktree state, Python/NumPy versions, Cython availability, build time, QPS,
p50/p95/p99 latency, recall@k, and memory estimates, including Python graph and
C++ CSR graph storage. The Markdown output is intended for copying benchmark
snapshots into the technical documentation.

Compare two JSON reports after an optimization:

```bash
python benchmarks/compare_results.py \
  benchmarks/baseline.json \
  benchmarks/candidate.json \
  --output benchmarks/comparison.md
```

The comparison marks each metric as improved, regressed, or unchanged based on
whether higher or lower is better for that metric.

### C++ Search Core

The current build includes C++ HNSW helpers wrapped through Cython. Python still
owns the public API, storage, metadata, persistence shape, and incremental
updates, but batch `build()` can construct the graph in C++ and post-build search
can use compact CSR adjacency arrays with C++ priority queues for layer traversal.

On SIFT1M 10k with `M=16`, `ef_construction=200`, and `ef_search=50`, the C++
search cache moved query throughput from about 2.2k QPS to about 10.4k QPS at
the same 99.1% Recall@10. The C++ batch builder then reduced build time from
17.50s to 2.46s, with the same recall and about 11.1k QPS.

The earlier insertion-time pruning helper did not improve build time on its own.
The useful boundary was moving construction traversal and mutable adjacency
together. The builder now also returns the CSR search cache directly, avoiding a
post-build walk over Python sets to recreate adjacency arrays.

The batch build path now skips Python edge materialization for normal search-only
workloads. On SIFT1M 10k, the benchmark reports zero Python graph edges, about
1.4 MiB of Python node metadata, and about 2.5 MiB for the C++ CSR graph. Save,
delete, and explicit graph inspection can still lazily rebuild Python connection
sets from the CSR cache.

### Basic Performance Benchmarks

Reproduce the SIFT1M results:

```bash
# Download SIFT1M dataset (~500MB)
python tests/benchmarks/download_datasets.py --sift

# Run benchmark on 10k vectors (~30 seconds)
pytest tests/benchmarks/test_real_world.py::TestRealWorldDatasets::test_sift1m_small -v -s

# Run parameter sweep (shows speed/accuracy trade-off)
pytest tests/benchmarks/test_real_world.py::TestRealWorldDatasets::test_sift1m_parameter_sweep -v -s
```

### Competitive Comparison Benchmarks

Compare against production vector databases:

```bash
# Install production vector databases (optional)
pip install chromadb qdrant-client

# 10k vectors: Our HNSW vs ChromaDB vs Qdrant (~2 minutes)
pytest tests/benchmarks/test_chromadb_comparison.py::TestProductionComparison::test_all_systems_10k -v -s

# 1M vectors: Our HNSW vs ChromaDB (~90 minutes)
pytest tests/benchmarks/test_chromadb_comparison.py::TestProductionComparison::test_all_systems_1m -v -s

# Individual comparisons
pytest tests/benchmarks/test_chromadb_comparison.py::TestChromaDBComparison::test_chromadb_vs_ours_10k -v -s
```

**Methodology:**
- All systems use identical HNSW parameters (M=16, ef_construction=200, ef_search=100)
- In-memory mode for fair comparison (no disk I/O overhead)
- Same SIFT1M dataset and ground truth
- Batch queries to remove Python loop overhead
- Single-threaded execution for consistent comparison

## Architecture

```
API Layer (FastAPI)
    ↓
Collection Manager
    ↓
┌───────────────┬──────────────────┐
│ HNSW Index    │ Metadata Store   │
│ (fast search) │ (SQLite filters) │
└───────────────┴──────────────────┘
    ↓
Vector Storage (memory-mapped files)
```

**Key Components:**
- **HNSW Index**: Multi-layer graph for logarithmic search time
- **Vector Store**: Memory-mapped files for efficient I/O
- **Metadata Store**: SQLite database for filtering
- **Collection**: Unified interface coordinating all components

## Implementation Highlights

### Performance Optimizations

**Cython Implementation** (current):
- Core search loop in Cython with C++ `std::set` for visited tracking
- Inline C distance calculations with `nogil` for no GIL overhead
- Memory views for zero-copy array access
- **Result**: 3x faster build, 2.4x faster search vs pure Python

**Bottlenecks Identified**:
- 73% of build time spent in graph search (unavoidable - O(n log n) algorithm)
- Python `heapq` and `dict` still used for priority queues and graph storage
- Query dtype conversion (float32 to float64) on each search call

**Gap vs Production Systems** (ChromaDB: 48x faster):
- ChromaDB uses hnswlib (pure C++ with SIMD/AVX instructions)
- Our Cython optimizes hot loops but core algorithm remains Python
- Full C++/Go/Rust rewrite needed to match production speed

**Other Optimizations**:
- Numba JIT for distance calculations (fallback when Cython unavailable)
- Memory-mapped storage for zero-copy vector access
- Greedy + beam search hybrid for optimal speed/accuracy

### Algorithm Implementation
- **Multi-Layer Graph**: Exponential decay layer assignment (probability = 1/2^level) creates logarithmic search complexity
- **Heuristic Neighbor Selection**: M-nearest selection during construction maintains graph connectivity
- **Dynamic ef_search**: Runtime tunable search quality (50=fast, 100=balanced, 200=maximum accuracy)

### Data Management
- **Hybrid Metadata Filtering**: Post-filtering approach - fetch extra candidates, filter by metadata, return top-k
- **Soft Deletes**: Tombstone pattern in SQLite allows delete operations without expensive graph reconstruction
- **Persistence**: Complete index serialization with pickle for zero-downtime restarts

### Scalability Features
- **Incremental Index Building**: Add vectors one-at-a-time or bulk insert, index updates incrementally
- **Lazy Loading**: Index loaded on-demand at first search, not at server startup
- **Memory Efficiency**: Float32 precision (4 bytes/dim) balances accuracy and memory usage

## Project Structure

```
vector-db/
├── src/
│   ├── api/              # REST API (FastAPI)
│   ├── collection/       # Collection management
│   ├── index/            # HNSW + brute force implementations
│   ├── storage/          # Vector and metadata storage
│   └── utils/            # Distance metrics
├── tests/                # Unit and integration tests
│   └── benchmarks/       # SIFT1M and performance tests
├── data/                 # Vector database files (gitignored)
├── Dockerfile            # Container definition
└── docker-compose.yml    # Orchestration
```

## Development

```bash
# Run tests
pytest tests/ -v

# Format code
black src/ tests/

# Type check
mypy src/
```

## Technology Stack

- **Python 3.11**: Core implementation
- **Cython**: C-compiled search loops (3x speedup)
- **NumPy**: Vectorized operations
- **Numba**: JIT fallback for distance calculations
- **FastAPI**: Async REST API
- **SQLite**: Metadata storage
- **Docker**: Containerized deployment

## Configuration

Key HNSW parameters (tunable via API):

| Parameter | Default | Effect |
|-----------|---------|--------|
| `M` | 16 | Connections per node (higher = better recall, more memory) |
| `ef_construction` | 200 | Build quality (higher = better graph, slower build) |
| `ef_search` | 50 | Search accuracy (higher = better recall, slower queries) |
| `metric` | euclidean | Distance metric (euclidean, cosine, dot_product) |
