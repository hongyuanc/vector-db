# Vector Database - Technical Documentation

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Core Components](#core-components)
4. [Implementation Details](#implementation-details)
5. [Performance Optimizations](#performance-optimizations)
6. [Design Decisions and Trade-offs](#design-decisions-and-trade-offs)
7. [Key Learnings](#key-learnings)
8. [Future Improvements](#future-improvements)

---

## Project Overview

### What is This?

A production-grade vector database built from scratch in Python, implementing the HNSW (Hierarchical Navigable Small World) algorithm for approximate nearest neighbor search. The system supports billions of high-dimensional vectors with sub-100ms retrieval times.

### Why Build This?

**Educational Goals:**
- Understand how vector databases work at a fundamental level
- Learn approximate nearest neighbor (ANN) algorithms
- Practice system design for data-intensive applications
- Build something production-ready without using existing libraries (no FAISS, Annoy, etc.)

**Real-World Applications:**
- RAG (Retrieval Augmented Generation) systems
- Semantic search engines
- Recommendation systems
- Image similarity search
- Anomaly detection

### Technology Stack

- **Python 3.11**: Main implementation language
- **NumPy**: Vectorized operations and BLAS acceleration
- **Numba**: JIT compilation for performance-critical code
- **FastAPI**: High-performance async web framework
- **SQLite**: Metadata storage and filtering
- **Docker**: Containerized deployment
- **pytest**: Testing framework

---

## Architecture

### High-Level System Design

```
┌─────────────────────────────────────────────────────────────┐
│                         API Layer                            │
│  FastAPI REST endpoints: /insert, /search, /delete          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Collection Manager                      │
│  Coordinates VectorStore, HNSW Index, and Metadata Store    │
└─────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┴─────────────┐
                ▼                           ▼
┌───────────────────────────┐   ┌──────────────────────────┐
│    Indexing Layer         │   │   Metadata Store         │
│  - HNSW Index             │   │  - SQLite Database       │
│  - Brute Force (fallback) │   │  - Filtering Support     │
│  - Layer management       │   │  - Lazy Deletion         │
└───────────────────────────┘   └──────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────┐
│                      Storage Engine                          │
│  - Memory-mapped vector files (vectors.mmap)                │
│  - Efficient disk I/O                                        │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

**Insert Operation:**
1. API receives vector + metadata
2. Validate dimension
3. VectorStore appends to memory-mapped file
4. MetadataStore inserts into SQLite
5. Mark index as stale (needs rebuild)
6. Return vector ID

**Search Operation:**
1. API receives query vector
2. Check if index is built
3. HNSW searches for approximate k-NN
4. Filter out deleted vectors
5. Apply metadata filters (post-filtering)
6. Return results with metadata + latency

**Index Build:**
1. Read all vectors from VectorStore
2. HNSW constructs hierarchical graph
3. Save graph to disk (index.npz)
4. Mark index as built

---

## Core Components

### 1. Vector Storage (`src/storage/vector_store.py`)

**Purpose:** Efficiently store and retrieve high-dimensional vectors.

**Implementation:**

```python
class VectorStore:
    def __init__(self, dimension: int, max_vectors: int, data_dir: str):
        # Memory-mapped file for O(1) random access
        self.vectors = np.memmap(
            f"{data_dir}/vectors.mmap",
            dtype=np.float32,
            mode='r+',
            shape=(max_vectors, dimension)
        )
```

**Key Decisions:**

- **Memory-mapped files:** Allows working with datasets larger than RAM. OS handles paging automatically.
- **Float32:** Balance between precision and memory (vs float64). Good enough for similarity search.
- **Pre-allocation:** Allocate max_vectors upfront to avoid resizing overhead.
- **Row-major layout:** Vectors stored contiguously for cache efficiency.

**Trade-offs:**

| Decision | Advantage | Disadvantage |
|----------|-----------|--------------|
| Memory mapping | Can handle > RAM datasets | Slower than in-memory arrays |
| Pre-allocation | No reallocation overhead | Wastes space if under-utilized |
| Float32 | 50% memory savings vs float64 | Slight precision loss (negligible for ANN) |

**What I Learned:**

Memory-mapped files are perfect for databases because they give you the illusion of having everything in memory while the OS handles the complexity of paging. The trick is to ensure sequential access patterns when possible to minimize page faults.

---

### 2. HNSW Index (`src/index/hnsw.py`)

**Purpose:** Enable sub-linear time approximate nearest neighbor search.

**Algorithm Overview:**

HNSW builds a multi-layer graph where:
- **Bottom layer (0):** Contains all vectors
- **Upper layers:** Contain progressively fewer vectors (exponential decay)
- **Connections:** Each node connects to M nearest neighbors
- **Search:** Start from top layer, greedily navigate to nearest neighbors, descend layers

**Why HNSW?**

| Algorithm | Time Complexity | Recall | Notes |
|-----------|----------------|--------|-------|
| Brute Force | O(n) | 100% | Too slow for large n |
| LSH | O(log n) | ~90% | Hash collisions hurt recall |
| HNSW | O(log n) | 95-99% | Best recall/speed trade-off |
| Product Quantization | O(log n) | ~85% | Lossy compression |

HNSW achieves the best balance of speed and accuracy for high-dimensional data.

**Key Parameters:**

```python
M = 16                # Connections per node (more = better recall, slower build)
ef_construction = 200 # Search width during build (more = better quality graph)
ef_search = 50        # Search width during query (more = better recall, slower)
```

**Implementation Highlights:**

1. **Layer Assignment:**
   ```python
   def _get_random_layer(self):
       # Exponential decay: P(layer=l) = e^(-l)
       return int(-np.log(np.random.uniform()) * self.m_L)
   ```
   Most vectors in layer 0, fewer in upper layers for faster navigation.

2. **Greedy Search:**
   ```python
   def _search_layer(self, query, entry_points, ef, layer):
       visited = set()
       candidates = []  # Min-heap of (distance, node)

       while candidates:
           current = heappop(candidates)
           # Navigate to neighbors, keep best ef candidates
   ```
   Beam search with width `ef`. More candidates = better recall.

3. **Graph Construction:**
   ```python
   def build(self, vectors):
       for i, vector in enumerate(vectors):
           layer = self._get_random_layer()
           # Insert into all layers from 0 to layer
           # Connect to M nearest neighbors in each layer
   ```
   Incremental construction ensures graph quality.

**Trade-offs:**

- **M (connections):** Higher M = better recall but more memory and slower inserts
  - Typical range: 8-64
  - We use 16 (good default)

- **ef_construction:** Higher = better graph quality but slower build
  - Typical range: 100-500
  - We use 200 (balanced)

- **ef_search:** Higher = better recall but slower queries
  - Typical range: 16-512
  - We use 50 (can tune per query)

**What I Learned:**

HNSW's brilliance is the multi-layer structure. By starting searches in sparse upper layers, you quickly get "close" to the target, then refine in denser lower layers. It's like zooming in on a map: start with continents, then countries, then cities, then streets.

The parameter tuning is crucial: for production, you'd benchmark with your actual data and query patterns to find the sweet spot between speed and accuracy.

---

### 3. Metadata Store (`src/storage/metadata_store.py`)

**Purpose:** Store document metadata and enable filtering during search.

**Implementation:**

```python
class MetadataStore:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                vector_id INTEGER PRIMARY KEY,
                metadata TEXT,
                deleted INTEGER DEFAULT 0
            )
        """)
```

**Why SQLite?**

- Lightweight: Single file, no server process
- ACID compliance: Crash-safe
- Fast lookups: B-tree indexes
- JSON support: Store arbitrary metadata as JSON

**Filtering Strategy:**

We use **post-filtering** (not pre-filtering):

```python
# Search workflow
results = hnsw.search(query, k=k*10)  # Over-fetch
filtered = [r for r in results if matches_filter(r)][:k]  # Filter after
```

**Why Post-Filter?**

| Approach | Pros | Cons |
|----------|------|------|
| Pre-filter | Exact k results | Very slow (can't use index effectively) |
| Post-filter | Fast (uses HNSW) | May return < k results if selective |
| Hybrid | Balanced | Complex implementation |

For our use case (educational), post-filtering is simpler and works well when filters aren't too selective.

**Lazy Deletion:**

Instead of removing vectors from the index (expensive):
```python
def delete(self, vector_id: int):
    # Just mark as deleted
    self.cursor.execute(
        "UPDATE metadata SET deleted = 1 WHERE vector_id = ?",
        (vector_id,)
    )
```

Then filter during search. This is O(1) vs O(n) for rebuilding the index.

**What I Learned:**

Hybrid search (vector + metadata) is crucial for real applications. Users don't just want "similar documents" - they want "similar tech documents from 2024". The post-filtering approach is a pragmatic trade-off that works well in practice.

For production at scale, you'd implement pre-filtering with inverted indexes or use a hybrid approach where you maintain separate HNSW indexes per metadata category.

---

### 4. Collection (`src/collection/collection.py`)

**Purpose:** High-level abstraction that coordinates all components.

**Design Pattern:** Facade pattern - provides a simple interface to complex subsystems.

```python
class Collection:
    def __init__(self, name, dimension, ...):
        self.store = VectorStore(...)       # Vector storage
        self.metadata_store = MetadataStore(...)  # Metadata
        self.index = HNSWIndex(...)         # Search index
```

**Key Methods:**

1. **insert():** Atomic operation to add vector + metadata
2. **batch_insert():** Optimized for bulk loading
3. **search():** Unified search with filtering
4. **build_index():** Construct HNSW from vectors
5. **save()/load():** Persistence

**Why This Design?**

Separating concerns (storage, indexing, metadata) makes the code:
- Easier to test (mock individual components)
- Easier to optimize (swap implementations)
- Easier to understand (single responsibility)

**What I Learned:**

The Collection class is where "batteries included" meets "separation of concerns". It hides complexity from the API layer while keeping internals modular. This is a common pattern in database design.

---

### 5. REST API (`src/api/server.py`)

**Purpose:** HTTP interface for remote access.

**Framework Choice:** FastAPI

Why FastAPI over Flask/Django?
- Built-in async support (important for I/O-bound operations)
- Automatic OpenAPI docs (great for testing)
- Pydantic validation (type safety)
- High performance (comparable to Node.js)

**Endpoints:**

```python
POST /collections/create      # Create new collection
POST /insert                  # Insert single vector
POST /batch_insert            # Insert multiple vectors
POST /search                  # k-NN search
DELETE /vector/{id}           # Delete vector
POST /collections/{name}/build_index  # Build HNSW index
GET /health                   # Health check
```

**Request/Response Models:**

Using Pydantic for type safety:
```python
class SearchRequest(BaseModel):
    vector: List[float]
    k: int = Field(default=10, ge=1, le=100)
    ef: Optional[int] = None
    filter: Optional[Dict[str, Any]] = None
```

**CORS Middleware:**

Enabled for development:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production: whitelist specific origins
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**What I Learned:**

FastAPI's automatic docs (`/docs`) are invaluable for development. The Pydantic validation catches bugs early - invalid requests never reach your business logic. For production, you'd add:
- Authentication (API keys, JWT)
- Rate limiting
- Request logging
- Metrics (Prometheus)

---

## Implementation Details

### Distance Metrics (`src/utils/distance.py`)

**Supported Metrics:**

1. **Euclidean Distance:**
   ```python
   def euclidean_distance(a, b):
       return np.linalg.norm(a - b)
   ```
   Use for: Absolute distance matters (e.g., measuring physical proximity)

2. **Cosine Similarity:**
   ```python
   def cosine_similarity(a, b):
       return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
   ```
   Use for: Direction matters, not magnitude (e.g., text embeddings, where document length shouldn't affect similarity)

3. **Dot Product:**
   ```python
   def dot_product(a, b):
       return -np.dot(a, b)  # Negative for min-heap
   ```
   Use for: When vectors are already normalized and you want fastest computation

**Performance Optimization:**

Using NumPy's BLAS-backed operations gives 10-100x speedup over naive Python loops:
```python
# Slow (pure Python)
distance = sum((a[i] - b[i])**2 for i in range(len(a)))**0.5

# Fast (NumPy)
distance = np.linalg.norm(a - b)
```

**What I Learned:**

Cosine similarity is the default for text embeddings because it normalizes for length. A 1000-word document and a 10-word document can have high similarity if they're about the same topic. For our TMDB demo, cosine similarity works perfectly.

---

### Persistence and Recovery

**File Structure:**

```
data/
└── {collection_name}/
    ├── vectors.mmap       # Memory-mapped vectors
    ├── metadata.db        # SQLite database
    ├── index.npz          # HNSW graph (NumPy compressed)
    └── config.json        # Collection configuration
```

**Save Process:**

```python
def save(self):
    # 1. Save HNSW graph
    np.savez_compressed(
        "index.npz",
        layers=self.layers,
        entry_point=self.entry_point,
        ...
    )

    # 2. Save config
    json.dump(config, open("config.json", "w"))

    # 3. VectorStore auto-saves (memory-mapped)
    # 4. SQLite auto-commits
```

**Load Process:**

```python
def _try_load(self):
    if index_path.exists():
        data = np.load(index_path)
        self.index.layers = data['layers']
        # Restore graph structure
```

**Crash Recovery:**

Current implementation:
- VectorStore: Safe (memory-mapped, OS handles writes)
- SQLite: Safe (WAL mode with atomic commits)
- HNSW index: Rebuilds if corrupt (detected during load)

For educational purposes, simple crash recovery (rebuild index) is acceptable. For production, we would need:
- Write-Ahead Log (WAL) for HNSW index
- Versioning for rolling back bad updates
- Snapshots for backup/restore

---

## Performance Optimizations

### 0. Cython Compilation (Phase 2)

**What:** Move performance-critical code from Python to Cython (Python-like syntax compiled to C).

**Why:** Python's GIL and dynamic typing create overhead. Hot loops in HNSW (search, distance calculations) execute millions of times.

**What We Optimized:**

1. **Core Search Loop** (`src/index/hnsw_core.pyx`)
   ```cython
   def search_layer(
       cnp.ndarray[cnp.float64_t, ndim=1] query,
       cnp.ndarray[cnp.float64_t, ndim=2] vectors,
       list entry_points,
       int num_closest,
       ...
   ):
       # Cython-compiled beam search
   ```

2. **Inline Distance Functions**
   ```cython
   cdef inline double c_euclidean_distance_inline(
       double[::1] a, double[::1] b, int n
   ) nogil:  # Release GIL for true parallelism potential
       cdef double squared_sum = 0.0
       cdef double diff
       cdef int i

       for i in range(n):
           diff = a[i] - b[i]
           squared_sum += diff * diff

       return sqrt(squared_sum)
   ```

3. **C++ Data Structures for Visited Tracking**
   ```cython
   from libcpp.set cimport set as cppset
   cdef cppset[int] visited  # Much faster than Python set()
   ```

**Results:**
- Build time: 3x faster (17.6s → 5.8s on 10k vectors)
- Search time: 2.4x faster (1.2ms → 0.5ms average latency)
- QPS improved: ~600 → ~1,800 on 10k SIFT1M

**Bottlenecks Remaining:**
- Python heapq still used (no good Cython alternative without complexity)
- Graph structure stored as Python dict: `{(node_id, layer): set}`
- Type conversions (float32 → float64) on each search call

**Why Not Full C++?**
- Cython gives 70-80% of C++ performance with 20% of the effort
- Maintains Python debugging and testing workflow
- Educational value: shows optimization technique without full rewrite

### 1. Vectorization with NumPy

**Before (Pure Python):**
```python
def compute_distances(query, vectors):
    distances = []
    for vec in vectors:
        dist = sum((q - v)**2 for q, v in zip(query, vec))**0.5
        distances.append(dist)
    return distances
```

**After (NumPy):**
```python
def compute_distances(query, vectors):
    return np.linalg.norm(vectors - query, axis=1)
```

**Speedup:** 50-100x faster due to:
- Vectorized operations (SIMD)
- Optimized BLAS libraries
- No Python interpreter overhead

### 2. Memory-Mapped Files

**Benefits:**
- Load "instantly" (no reading entire file)
- Work with datasets larger than RAM
- OS handles caching intelligently

**Access Pattern Optimization:**
```python
# Bad: Random access (lots of page faults)
for i in random_order:
    process(vectors[i])

# Good: Sequential access (cache-friendly)
for i in range(len(vectors)):
    process(vectors[i])
```

### 3. Batch Operations

**Single Insert:** ~1ms per vector (API overhead dominates)
**Batch Insert:** ~0.01ms per vector (amortized)

Always prefer batch operations for bulk loading.

### 4. HNSW Parameter Tuning

Based on SIFT1M benchmarks:

| ef_search | Recall | QPS | Latency |
|-----------|--------|-----|---------|
| 16 | 0.92 | 2000 | 0.5ms |
| 50 | 0.98 | 850 | 1.2ms |
| 200 | 0.99 | 200 | 5ms |

**Recommendation:** Use ef_search=50 as default, allow override per query.

### 5. JIT Compilation with Numba

For hot paths (distance calculations in tight loops):
```python
@numba.jit(nopython=True)
def euclidean_distance_batch(query, vectors):
    # Compiled to machine code
    distances = np.empty(len(vectors))
    for i in range(len(vectors)):
        distances[i] = np.sum((query - vectors[i])**2)
    return np.sqrt(distances)
```

**When to use Numba:**
- Tight loops over NumPy arrays
- Not using complex Python objects
- Profiling shows it's a bottleneck

**What I Learned:**

Premature optimization is evil, but knowing where to optimize matters. Profile first (`line_profiler`), then optimize the hot 20% that takes 80% of the time. For vector databases, it's usually:
1. Distance calculations (vectorize with NumPy)
2. Graph traversal (optimize data structures)
3. I/O (use memory-mapping)

---

## Design Decisions and Trade-offs

### 1. Python vs C++/Rust

**Decision:** Python

**Reasoning:**
- Educational project: clarity over performance
- NumPy + Numba get you 80% of C++ speed
- Faster development iteration
- Better ecosystem for ML/data work

**Trade-off:** 2-5x slower than C++ (but still fast enough for millions of vectors)

### 2. Single vs Multi-threaded

**Decision:** Single-threaded (for now)

**Reasoning:**
- Simpler implementation
- Python GIL makes threading complex
- Single-threaded can still handle 500+ QPS

**For Production:**
- Use multiprocessing (bypass GIL)
- Run multiple container instances (horizontal scaling)
- Use async I/O for concurrency without threads

### 3. Approximate vs Exact Search

**Decision:** Approximate (HNSW)

**Reasoning:**
- Exact k-NN is O(n) - too slow for large n
- 98% recall is good enough for most applications
- 100x+ speedup

**When Exact is Better:**
- Small datasets (< 10k vectors)
- Need guaranteed correctness
- Legal/compliance requirements

### 4. In-Memory vs Disk-Based

**Decision:** Hybrid (memory-mapped)

**Reasoning:**
- Best of both worlds
- OS handles caching better than we could
- Scales beyond RAM

**Trade-off:**
- Slower than pure in-memory
- Faster than pure disk-based

### 5. SQLite vs PostgreSQL for Metadata

**Decision:** SQLite

**Reasoning:**
- No server to manage
- Perfect for single-node deployment
- ACID compliance
- Fast for reads (our primary use case)

**When to use PostgreSQL:**
- Multi-node deployment
- Complex queries with joins
- High write concurrency
- Need replication

### 6. REST API vs gRPC

**Decision:** REST with FastAPI

**Reasoning:**
- Easier to test (curl, Postman)
- Better documentation (OpenAPI)
- More familiar to most developers
- Good enough performance for our use case

**When to use gRPC:**
- Need lowest possible latency
- Strict type contracts
- Streaming large amounts of data
- Microservices with service mesh

---

## Key Learnings

### Technical Insights

1. **HNSW is remarkably effective**
   - The multi-layer graph structure is elegant
   - Parameter tuning has predictable effects
   - Works well even with naive implementation

2. **Memory-mapped files are underrated**
   - OS is better at caching than you are
   - Lets you work with huge datasets easily
   - Key for database performance

3. **NumPy is essential for Python performance**
   - Vectorization gives orders of magnitude speedup
   - But requires thinking in arrays, not loops
   - Understanding broadcasting is crucial

4. **API design matters**
   - FastAPI's automatic docs save hours
   - Type hints catch bugs early
   - Async support is table stakes

5. **Testing is critical for databases**
   - Correctness is paramount
   - Benchmark on real datasets (SIFT1M)
   - Compare against ground truth

### System Design Lessons

1. **Separate concerns**
   - Storage, indexing, and metadata are independent
   - Makes testing and optimization easier
   - Allows swapping implementations

2. **Design for observability**
   - Return latency metrics
   - Health checks for monitoring
   - Logging for debugging

3. **Start simple, add complexity judiciously**
   - Single-threaded is fine initially
   - Add parallelism only when needed
   - Measure before optimizing

4. **Data structures dominate algorithms**
   - Memory layout affects cache performance
   - Right structure makes implementation obvious
   - Premature abstraction hurts

---

## Competitive Benchmarking (Phase 3)

### Methodology

To understand where our implementation stands, we conducted fair head-to-head comparisons against production vector databases.

**Systems Tested:**
1. **ChromaDB** - Uses hnswlib (pure C++ with SIMD/AVX)
2. **Qdrant** - Rust-based vector database
3. **Our HNSW** - Python + Cython implementation

**Fair Comparison Criteria:**
- Identical HNSW parameters (M=16, ef_construction=200, ef_search=100)
- In-memory mode for all systems (no disk I/O overhead)
  - ChromaDB: EphemeralClient
  - Qdrant: `:memory:` storage (Python client, local mode)
  - Ours: In-memory vectors
- Same SIFT1M dataset (industry standard)
- Batch queries where possible (remove Python loop overhead)
- Single-threaded execution

**Limitations:**
- Qdrant tested via Python client in `:memory:` mode, not production server with gRPC
- Results reflect embedded/local performance, not distributed production deployments
- Focus on single-machine, single-threaded performance (real production often uses multiple cores/machines)

### Results (SIFT1M 10k Subset)

| System | Recall@10 | QPS | Latency | Build Time | Implementation |
|--------|-----------|-----|---------|------------|----------------|
| **ChromaDB** | 100.0% | 9,234 | 0.11ms | 0.34s | C++ hnswlib + SIMD |
| **Our HNSW** | 99.5% | 1,810 | 0.55ms | 13.20s | Python + Cython |
| **Qdrant** | 100.0% | 621 | 1.61ms | 0.79s | Rust |

**Test Environment:** Apple M4 Pro, 24GB RAM, Python 3.11

### Key Insights

**1. ChromaDB Leads the Pack**
- 5.1x faster search (9234 vs 1810 QPS)
- 38.8x faster build (0.34s vs 13.20s)
- Why: Pure C++ with hand-tuned SIMD (AVX2/AVX512), optimized memory layout

**2. Qdrant Python Client Performance**
- Our implementation: 1810 QPS vs Qdrant `:memory:` mode: 621 QPS
- **Important caveat**: This tests Qdrant's Python client in local mode, not production server
- Possible reasons for the difference:
  - Python client serialization/deserialization overhead
  - `:memory:` mode not optimized for performance (dev/testing mode)
  - Qdrant's architecture optimized for distributed server deployments, not embedded use
  - Our Cython is specifically tuned for single-machine, single-threaded performance

**3. Build Time is Our Biggest Gap**
- 38.8x slower than ChromaDB (13.20s vs 0.34s)
- Python overhead in graph construction dominates
- 73% of time spent in graph search (O(n log n) unavoidable)
- Remaining 27%: Python dict/set operations, heap management

**4. Recall is Competitive**
- 99.5% vs 100% (within 0.5%)
- All systems use identical HNSW parameters
- Slight difference likely due to implementation details (tie-breaking, float precision)

### What We Learned

**About Performance Gaps:**
- The 5x search gap (vs ChromaDB) comes from:
  - SIMD vectorization (2-3x): ChromaDB uses AVX instructions for distance calculations
  - Memory layout (1.5x): Cache-friendly data structures
  - Compiler optimizations (1.5x): Loop unrolling, inlining
  - No interpreter overhead (1.2x): Pure C++ vs Python/Cython hybrid

- The 38x build gap is mostly Python overhead in graph construction
  - Each insert requires searching the graph (millions of operations)
  - Python dict/set operations vs C++ std::unordered_map/set
  - No SIMD in graph traversal (hard to vectorize)

**About Our Implementation:**
- Cython gave us 3x improvement, bringing us from ~600 to ~1,800 QPS
- To match ChromaDB would require:
  - Full C++/Rust rewrite
  - Hand-tuned SIMD for distance calculations
  - Optimized memory layout (AoS vs SoA)
  - This would be a 2-3 week effort vs educational value

**About Trade-offs:**
- For an educational project, 1,810 QPS is excellent
- Python enables rapid iteration and debugging
- Cython provides "good enough" performance without full rewrite
- For production at 10k scale, our implementation is viable
- For 1M+ scale, ChromaDB's C++ is worth the complexity

### Results (SIFT1M Full Dataset - 1M Vectors)

| System | Recall@10 | QPS | Latency | Build Time | Implementation |
|--------|-----------|-----|---------|------------|----------------|
| **ChromaDB** | 98.8% | 2,902 | 0.34ms | 1.1 min | C++ hnswlib + SIMD |
| **Our HNSW** | 93.7% | 675 | 1.48ms | 66.2 min | Python + Cython |

**Test Environment:** Apple M4 Pro, 24GB RAM, Python 3.11

**Scaling Analysis:**

1. **Search Performance Scales Well**
   - ChromaDB: 9,234 → 2,902 QPS (3.2x slowdown for 100x data)
   - Our HNSW: 1,810 → 675 QPS (2.7x slowdown for 100x data)
   - Both exhibit logarithmic scaling as expected from HNSW

2. **Build Time Shows O(n log n) Complexity**
   - ChromaDB: 0.34s → 66s (194x slower for 100x data)
   - Our HNSW: 13.2s → 3,972s (301x slower for 100x data)
   - Graph construction cost dominates at scale

3. **Performance Gap Narrows at Scale**
   - 10k: ChromaDB 5.1x faster search → 1M: 4.3x faster search
   - Our Cython optimizations show better relative performance at larger scale
   - Gap from 5.1x to 4.3x suggests our graph traversal is efficient

4. **Recall Remains Strong**
   - Both systems maintain >93% recall at 1M scale
   - ChromaDB: 100% → 98.8% (minimal degradation)
   - Our HNSW: 99.5% → 93.7% (reasonable for Python implementation)

**Qdrant Note:** Qdrant's Python `:memory:` client degraded to 6 QPS (175ms latency) at 1M scale - a 100x slowdown. This is due to Python client limitations beyond 20k vectors, making it unsuitable for fair comparison at this scale.

### Benchmarking Commands

To reproduce these results:

```bash
# Install production vector databases
pip install chromadb qdrant-client

# 10k vectors comparison (all systems: Our HNSW, ChromaDB, Qdrant)
pytest tests/benchmarks/test_chromadb_comparison.py::TestProductionComparison::test_all_systems_10k -v -s

# 1M vectors comparison (ChromaDB vs our HNSW only - takes ~90 min)
pytest tests/benchmarks/test_chromadb_comparison.py::TestProductionComparison::test_all_systems_1m -v -s

# Individual system comparisons
pytest tests/benchmarks/test_chromadb_comparison.py::TestChromaDBComparison::test_chromadb_vs_ours_10k -v -s
```

---

## Reproducible Metrics Pipeline

### What Was There Before

The project already had several useful benchmark paths:

- pytest benchmark tests for search latency, recall, real datasets, and production system comparisons
- `benchmarks/benchmark_cython.py` for measuring the Cython-accelerated HNSW path
- Markdown benchmark summaries documenting SIFT1M and ChromaDB/Qdrant comparisons

The gap was that `benchmarks/benchmark.py`, the command exposed by `make benchmark`,
was still a placeholder. It described build time, p50/p95/p99 latency, throughput,
memory, and recall, but the functions were `TODO/pass`. That made it harder to run
one consistent command before and after an optimization and then compare artifacts.

### What Was Implemented

`benchmarks/benchmark.py` now runs one benchmark configuration end to end and writes
structured output:

```bash
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

The JSON report includes:

- dataset name, size, dimension, query count, and seed
- HNSW parameters: `M`, `ef_construction`, `ef_search`, and metric
- environment metadata: git commit, dirty worktree state, Python, NumPy, platform, and Cython availability
- build time and vectors-per-second
- search QPS and latency percentiles: p50, p95, p99, and average
- recall@k against brute-force ground truth
- vector/query memory estimates, graph memory estimates, and process peak RSS when available
- compact CSR versus materialized Python graph save/load timing, file size,
  loaded graph shape, and process peak RSS samples

The Markdown report contains the same information in a compact table so each
benchmark snapshot can be appended to this document or interview notes.

`benchmarks/compare_results.py` compares two JSON reports:

```bash
python benchmarks/compare_results.py \
  benchmarks/baseline.json \
  benchmarks/candidate.json \
  --output benchmarks/comparison.md
```

The comparison report shows the baseline value, candidate value, absolute delta,
percentage delta, and status for each metric. Status is directional: lower build
time and latency are improvements, while higher QPS, build throughput, and recall
are improvements.

### Why This Matters

Optimization work is only useful if the measurement is repeatable. Before changing
graph storage, Cython boundaries, filtering, or a possible C++ core, this command
creates a baseline tied to a specific commit and configuration. That prevents
ambiguous claims like "faster" or "better recall" and replaces them with numbers
that can be compared across commits.

This also keeps the project educational: every performance change can now be
explained as a hypothesis, a code change, and a measured result.

### Persistence Benchmark Update

What was there before: the benchmark runner showed the memory shape of the graph
after build/search, but it did not exercise `save()` or `load()`. After adding
compact CSR persistence, that left an important gap: unit tests proved the format
could reload without Python edge sets, but there was no repeatable measurement of
save time, load time, file size, or loaded graph memory shape.

What was implemented: `run_benchmark_suite()` now measures two persistence
variants after search metrics are collected:

- **Compact:** saves the index as built when `_cpp_graph_cache` is present and
  `_python_graph_materialized` is false. Loading this file should keep graph
  edges in CSR arrays and leave Python node connection sets empty.
- **Materialized:** calls `materialize_python_graph()`, clears the CSR cache
  before saving, and then loads through the legacy Python connection path. This
  gives a baseline for the older memory shape where Python edge sets are present
  after reload.

Each variant records whether it was available, save/load seconds, `.npz` file
size, process peak RSS sampled after load, and `_estimate_graph_memory()` for the
loaded index.

Why: the CSR persistence change was about preserving memory ownership across
reload, not just making save/load pass. Measuring both persistence shapes in the
same benchmark artifact makes that tradeoff visible and gives future work a
stable way to check whether file format or graph ownership changes improve or
regress persistence behavior.

### Next Steps

- Store benchmark JSON/Markdown snapshots under a versioned `benchmarks/results/` directory.
- Extend the runner to use already-downloaded SIFT1M subsets for the same structured report format.

---

## C++ HNSW Core Spike

### What Was There Before

The Cython HNSW path had already moved distance calculations and some layer
search work out of pure Python, but the hot search path still depended on Python
containers:

- graph edges were cached as Python `dict[(node_id, layer)] -> set(neighbor_ids)`
- candidate/result queues still used Python `heapq`
- visited tracking crossed the Python/Cython boundary
- vectors were converted to float64 for the Cython search cache

Benchmarks showed that search was usable, but a production C++ HNSW core such as
ChromaDB's hnswlib-backed implementation was still much faster.

### What Was Implemented

The C++ work keeps the database architecture unchanged and ports bounded HNSW
primitives into native code:

- `src/index/hnsw_cpp_core.cpp` implements HNSW layer search in C++.
- `src/index/hnsw_cpp.pyx` wraps that C++ function for Python.
- `HNSWIndex._build_cpp_cache()` converts each graph layer into CSR-style arrays:
  `offsets` and `neighbors`.
- `HNSWIndex._search_layer()` now prefers the C++ cache after index build, with
  the existing Cython and Python paths retained as fallbacks.
- `hnsw_cpp.prune_connections()` also exposes C++ distance calculation and
  sorting for insertion-time neighbor pruning.
- `hnsw_cpp.build_graph()` now owns batch HNSW construction for `HNSWIndex.build()`:
  it samples no randomness itself, but receives Python-sampled levels, builds
  mutable adjacency in C++, and returns graph metadata plus CSR search layers to
  Python.
- The C++ builder also returns CSR search layers directly. `HNSWIndex.build()`
  now reuses those arrays instead of walking Python sets to rebuild the same
  C++ search cache after construction.
- The C++ builder can skip Python connection-row output. `HNSWIndex.build()`
  now keeps batch-built graph edges only in the compact CSR cache by default.
- `HNSWIndex.materialize_python_graph()` lazily rebuilds Python `set`-based
  connections from CSR when graph inspection, save, delete, or mutation needs
  the older Python representation.
- `HNSWIndex.graph_storage_mode` exposes whether the graph is currently
  `compact_csr`, `materialized_python`, or empty.

This is not a full C++ rewrite. Python still owns:

- the public `HNSWIndex` API
- incremental `insert()` and delete behavior
- persistence shape and Python `HNSWNode` objects
- vector storage, metadata storage, collections, and API routing

### Why This Scope

This chunk answers a narrow question: does moving only layer traversal from
Python/Cython containers into C++ materially improve search? If the answer were
no, a larger C++ rewrite would be hard to justify. If the answer were yes, the
next target would be graph construction. The pruning-only experiment then tested
whether a tiny build wrapper was enough. It was not, so the next chunk moved
construction traversal and mutable adjacency together.

### Measured Result

SIFT1M 10k, `M=16`, `ef_construction=200`, `ef_search=50`, 100 queries:

| Version | Build Time | QPS | Avg Latency | p99 Latency | Recall@10 |
|---------|-----------:|----:|------------:|------------:|----------:|
| Cython search cache | 16.68s | 2,240.91 | 0.4451ms | 0.6565ms | 99.10% |
| C++ search cache | 17.14s | 10,443.59 | 0.0950ms | 0.1387ms | 99.10% |
| C++ search + C++ pruning | 17.50s | 10,146.75 | 0.0977ms | 0.1450ms | 99.10% |
| C++ batch build + C++ search | 2.46s | 11,138.39 | 0.0892ms | 0.1348ms | 99.10% |
| C++ batch build + direct CSR cache | 2.65s | 10,041.38 | 0.0988ms | 0.1560ms | 99.10% |
| C++ batch build + lazy Python graph | 2.43s | 9,999.63 | 0.0992ms | 0.1792ms | 99.10% |

The C++ layer traversal improved QPS by about 4.7x and reduced p99 latency by
about 79% at the same recall. Build time did not improve because graph
construction still runs through the existing Python/Cython insertion path.

At `ef_search=100`, the same C++ search path reached 7,726 QPS with 99.60%
Recall@10. That puts search much closer to the earlier ChromaDB 10k result
(8,825 QPS), while build remains far behind.

The C++ pruning update did not improve the 10k build benchmark. It moved one
small calculation into C++, but each insertion still crosses Python objects,
Python sets, and Python-owned graph mutation. In this run, build time regressed
by about 2.1% versus the C++ search-only baseline, with unchanged recall. That
is close enough to benchmark noise that it should not be over-interpreted, but
it is clear that isolated pruning is not the build bottleneck.

The C++ batch builder is the first construction change that moves the right
boundary. It reduced 10k build time from 17.50s to 2.46s, an 85.9% improvement,
while keeping Recall@10 unchanged at 99.10%. Query latency also improved slightly
because the finished graph still uses the C++ search cache. Peak process RSS was
higher in this run: 4,636.92 MiB versus 4,343.42 MiB. That likely reflects the
temporary C++ graph, Python graph materialization, and post-build CSR cache all
being alive during the benchmark process.

The direct CSR cache update removes one redundant post-build conversion: C++
already has the final adjacency, so it now returns CSR arrays for search along
with the Python connection rows. This reduced peak process RSS in the 10k sample
from 4,636.92 MiB to 4,144.39 MiB, while Recall@10 stayed unchanged. It did not
improve speed in this run: build time was 2.65s and QPS was 10,041.38. That is
useful as a memory/ownership cleanup, but it confirms that the next meaningful
step is to stop materializing the same graph twice, not just move another
conversion boundary.

The benchmark report now breaks graph memory out from vector memory. On the same
SIFT1M 10k run, the graph had 482,450 directed edges in Python node/set storage
and the same 482,450 directed edges in the C++ CSR search cache. The Python graph
estimate was 44.5242 MiB, while the C++ CSR cache was 2.4890 MiB. The combined
reported graph total was 47.0131 MiB. This makes the next ownership problem
visible: most graph memory is in Python sets, not in the compact C++ search
arrays.

The lazy Python graph update removes those duplicate Python edge sets from the
normal batch-build search path. On the same SIFT1M 10k benchmark, Python graph
materialization was false, Python graph edges dropped from 482,450 to 0, and the
reported graph total dropped from 47.0131 MiB to 3.9147 MiB. The remaining
Python graph estimate was 1.4257 MiB for node metadata, while the C++ CSR cache
still held 482,450 directed edges in 2.4890 MiB. Recall stayed at 99.10%.

The CSR persistence update keeps that compact memory shape across save/load.
Previously, `HNSWIndex.save()` always called
`_ensure_python_graph_materialized()`, so saving a compact C++ batch-built graph
rebuilt Python `set` edges before writing the index. Loading then restored those
pickled Python connections and rebuilt the C++ cache from them. The new file
format stores node identity/layer metadata separately from the graph edges and
writes each CSR layer directly as `csr_offsets_<layer>` and
`csr_neighbors_<layer>` arrays. A compact save records
`python_graph_materialized=false`; load uses the direct CSR arrays to restore
`_cpp_graph_cache`, keeps node `connections` empty, and initializes the float32
vector cache needed by C++ search. Materialized and legacy files still load
through the existing pickled Python graph path.

This matters because persistence should not undo the memory ownership win from
lazy Python graph materialization. Reloading an index can now keep graph edges in
compact arrays instead of temporarily or permanently reconstructing the Python
edge sets just to reach the same searchable C++ graph.

The persistence benchmark was then run on SIFT1M 10k and 100k at
`M=16`, `ef_construction=200`, and `ef_search=50` on commit `b6e0ebd`.
The benchmark worktree was dirty because `AGENTS.md` was untracked, but the code
under test was the committed benchmark and persistence code.

| Dataset | Build Time | QPS | p99 Latency | Recall@10 |
|---------|-----------:|----:|------------:|----------:|
| SIFT1M 10k | 2.4171s | 10,902.50 | 0.1293ms | 99.10% |
| SIFT1M 100k | 48.6693s | 5,668.19 | 0.2569ms | 94.67% |

| Dataset | Shape | Save Time | Load Time | File Size | Loaded Graph |
|---------|-------|----------:|----------:|----------:|-------------:|
| SIFT1M 10k | Compact CSR | 0.0093s | 0.0066s | 7.6885 MiB | 3.9147 MiB |
| SIFT1M 10k | Materialized Python graph | 0.0424s | 0.1853s | 6.6740 MiB | 47.0131 MiB |
| SIFT1M 100k | Compact CSR | 0.1090s | 0.0684s | 77.1979 MiB | 41.6182 MiB |
| SIFT1M 100k | Materialized Python graph | 0.7714s | 2.7478s | 69.1201 MiB | 470.7456 MiB |

At 10k, compact CSR save was about 4.6x faster and load was about 28.2x faster
than the materialized path. At 100k, compact CSR save was about 7.1x faster and
load was about 40.2x faster. The compact files were larger in these runs because
the CSR arrays are stored directly in the `.npz`, while the Python graph path is
stored through pickle. For this project stage, the important result is reload
shape and load cost: compact reload kept Python graph edges at zero and loaded
only one graph copy, while the materialized path restored Python edge sets and
then rebuilt the C++ CSR cache, counting the same edges twice.

The process RSS values in this benchmark are `ru_maxrss` high-water marks within
one process, so they are useful as run context but not as isolated per-variant
memory deltas. The loaded graph estimates are the clearer signal for this
specific persistence comparison.

The compact graph is now explicitly treated as a read-optimized batch index
shape. What was there before: `insert()` and `delete()` already called
`_ensure_python_graph_materialized()`, but that conversion happened silently.
That made it easy to accidentally turn a compact CSR index back into Python
edge sets without noticing the memory-shape change.

What was implemented: mutating a compact CSR index still works, but `insert()`
and `delete()` now emit a `RuntimeWarning` before materializing Python
connection sets. The warning states that mutation keeps compatibility available
but increases graph memory until the index is rebuilt compactly.
`graph_storage_mode` gives tests, benchmarks, and users a small public signal
for the current graph shape. This keeps the project honest about the current
tradeoff: batch-built indexes are compact for build/search/save/load, while
online mutation remains supported through the older materialized Python graph.

The 100k ChromaDB comparison was refreshed again after moving
`HNSWIndex.search_batch()` from a Python loop over `search()` into a native
C++/Cython compact-CSR batch path. What was there before: callers had a batch
method, but it still performed one full Python-level search per query. What was
implemented: `hnsw_cpp.search_batch()` now receives the query matrix, vector
matrix, CSR layer views, entry point, max layer, `k`, and `ef`, then performs
the full top-down HNSW traversal for each query inside C++. The Python method
uses this path only when the compact CSR cache is complete, and keeps the
existing repeated-search fallback for other graph shapes.

This used
`tests/benchmarks/test_chromadb_comparison.py::TestChromaDBComparison::test_chromadb_vs_ours_100k`
on SIFT1M 100k with `M=16`, `ef_construction=200`, `ef_search=100`, and 100
queries. The benchmark now writes structured output to
`benchmarks/results/sift1m-100k-chromadb-comparison.json` and a Markdown summary
to `benchmarks/results/sift1m-100k-chromadb-comparison.md`.

| System | Build Time | Batch QPS | Batch Avg Latency | Single p99 | Recall@10 |
|--------|-----------:|----------:|------------------:|-----------:|----------:|
| Our HNSW C++/CSR | 49.79s | 4,266.4 | 0.2344ms | 1.0822ms | 98.50% |
| ChromaDB | 4.83s | 6,476.6 | 0.1544ms | 0.5752ms | 99.70% |

Compared with the previous batch-boundary run, our batch QPS improved from
3,921.1 to 4,266.4, about an 8.8% gain, and batch average latency improved from
0.2550ms to 0.2344ms. This confirms that removing Python's per-query full-search
loop is useful, but it is not enough to match ChromaDB. Compared with ChromaDB,
the current implementation is still about 10.3x slower to build and about 1.5x
slower on batch average latency at this scale. Single-query p99 is not expected
to improve from this change because single `search()` still uses the existing
layer-level path; the p99 value also has benchmark noise at only 100 measured
queries.

The next useful performance work is deeper than the batch call boundary:
optimize the C++ layer traversal itself, reduce per-layer visited allocation, or
vectorize distance computation. Those changes should be measured against the
same JSON artifact schema.

## HNSW C++ Parity Plan

What was there before: the project had already moved batch build, compact CSR
graph ownership, persistence, and batch search into a C++/Cython path. On the
saved SIFT1M 100k ChromaDB comparison, this reached 98.50% Recall@10 and
4,266.4 batch QPS, but build time was still 49.79s versus ChromaDB's 4.83s.
Search was closer than build, but ChromaDB still led on batch average latency,
single-query p99 latency, and recall.

What is planned: the next phase focuses on deeper C++ work for batch-built
read indexes only. `HNSWIndex.build()` remains the target path. `insert()` and
`delete()` keep the current compatibility behavior, where mutating a compact CSR
index materializes Python connection sets and emits a warning.

Why this matters: the remaining build gap is no longer mostly Python API
overhead. The native builder still performs avoidable work, including repeated
visited allocation, scalar Euclidean distance with `sqrt`, nested vector
adjacency management, linear duplicate checks, and nearest-only neighbor
selection. These are the same classes of implementation detail that separate a
straightforward HNSW implementation from production libraries such as hnswlib.

The implementation plan will proceed in commit-sized chunks:

1. Add native phase instrumentation so build and search time are split into
   measurable substeps.
2. Use squared L2 distance internally for Euclidean ordering and convert back to
   public L2 distances only at result boundaries.
3. Reuse native scratch memory for visited marks and candidate/result buffers.
4. Replace nested mutable adjacency with a lower-overhead bounded connection
   layout.
5. Add HNSW heuristic neighbor selection to improve graph navigability and
   recall at the same construction parameters.
6. Clean up native batch search setup and evaluate optional batch-only threading
   after the single-threaded core is tighter.

The first success target is SIFT1M 100k at `M=16`, `ef_construction=200`,
`ef_search=100`, and `k=10`: reduce build time below 15s, raise Recall@10 to at
least 99.30%, and raise batch QPS to at least 6,000 while preserving compact CSR
storage.

## Native Build Instrumentation

What was there before: benchmark reports measured Python-level `index.build()`
wall time, but the native builder itself did not report where construction time
was spent. Once batch build moved into C++, that made the next optimization
steps harder to prioritize. A 49.79s build could be dominated by graph search,
neighbor pruning, adjacency mutation, CSR export, or wrapper conversion, but the
benchmark schema could not distinguish those phases.

What was implemented: the C++ `BuildGraphResult` now carries a `BuildStats`
record with native build phase timings and graph shape counters. The Cython
wrapper converts that struct into a Python `build_stats` dictionary, and
`HNSWIndex` stores it on `_last_cpp_build_stats` after a C++ batch build. The
benchmark CLI now includes `metrics.cpp_build_stats` in JSON output and renders
the main native phase timings in the Markdown report. The comparison helper can
also compare native phase timings between benchmark artifacts.

Why this matters: this is instrumentation, not an optimization. Its job is to
make later optimization chunks measurable. Before changing distance math,
visited tracking, adjacency layout, or neighbor selection, the project can now
record whether a change reduced native search time, pruning time, CSR export
time, or only shifted work between phases.

Verification added:

- `tests/test_hnsw_cpp.py` checks that `hnsw_cpp.build_graph()` returns native
  phase stats and that `HNSWIndex.build()` stores them.
- `tests/test_benchmark_cli.py` checks that benchmark JSON and Markdown reports
  expose the C++ build stats.

### Detailed Build Counter Baseline

What was there before: native build instrumentation exposed broad phase timings,
which made it possible to separate graph construction, CSR export, and wrapper
conversion. It still did not expose the detailed work counters needed to explain
why graph construction was expensive: distance evaluations, visited nodes, heap
operations, neighbor-selection work, and prune input sizes were not visible in
benchmark artifacts.

What changed: the native build path now exposes detailed build counters through
the benchmark JSON and Markdown reports. The counters split C++ distance work
across layer search, heuristic neighbor selection, and pruning, while also
reporting visited-node totals and average graph-selection/prune sizes.

Why this matters: the next optimization can target measured distance work
instead of guessing. If most distance evaluations come from layer search, the
best next step is different from a result where pruning or neighbor selection
dominates. This keeps the HNSW work educational and evidence-driven: every
optimization should have a counter that explains whether it reduced the intended
cost.

Measured result from the 10k random-vector benchmark:

| Metric | Value |
|---|---:|
| Build Time | 2.9704s |
| C++ Distance Evaluations | 53141500 |
| C++ Search Distance Evaluations | 31351236 |
| C++ Neighbor Selection Distance Evaluations | 8553272 |
| C++ Prune Distance Evaluations | 13236992 |
| C++ Visited Nodes | 31351236 |
| C++ Average Selected Degree | 23.838104688662863 |
| C++ Average Prune Input Size | 27.665596575708935 |

Next step: target the largest measured source of distance work while preserving
the SIFT1M 100k recall target. Based on this baseline, layer search distance
evaluations are the first area to inspect before changing pruning or neighbor
selection behavior.

## Squared L2 Native Ordering

What was there before: the C++ path computed full Euclidean distance with
`sqrt` during graph traversal, neighbor pruning, and search ordering. This was
simple and matched the public distance value directly, but the square root is
not needed for ordering because squared L2 distance preserves the same nearest
neighbor order as L2 distance.

What was implemented: native Euclidean heap comparisons and pruning now use
squared L2 distance internally. Public search results still return normal L2
distance by converting the selected result distances back with `sqrt` at the
result boundary. Cosine behavior is unchanged. Build stats now include
`uses_squared_l2`, so benchmark artifacts can show when this path is active.

Why this matters: HNSW build and search perform many distance calculations. This
change removes an avoidable square root from internal candidate ordering without
changing graph semantics or public result distances. It is a small optimization,
but it also creates a clean boundary for later distance-kernel work: internal
ordering can use the cheapest monotonic distance form, while the public API can
still expose user-facing distances.

Verification added:

- `tests/test_hnsw_cpp.py` checks that Euclidean native builds report
  `uses_squared_l2=True`, while cosine builds report `False`.
- Existing C++ search tests continue to check public Euclidean distances, so the
  result boundary still returns normal L2 values.
- `tests/test_benchmark_cli.py` checks that benchmark JSON includes the new
  build-stat field.

## Reusable Native Visited Scratch

What was there before: every native layer search allocated a fresh visited array.
That happened during batch graph construction and during compact CSR search.
HNSW performs many small layer searches, especially while building, so repeated
visited allocation and clearing added allocator and memory-write overhead to a
hot path.

What was implemented: the C++ core now has a `SearchScratch` helper with a
generation-mark visited array. Instead of clearing memory for each search, the
scratch object increments a generation counter and marks visited node IDs with
that generation. The builder preallocates scratch storage once for the full
vector count, and `search_batch()` reuses one scratch object across all layer
searches in the batch. The public single-layer `search_layer()` keeps the same
API and creates local scratch internally.

Why this matters: this removes repeated visited-array allocation without
changing search semantics. The win should show up most clearly in native build
search time, because construction performs one or more graph searches for almost
every inserted vector. Benchmark stats now include `search_calls` and
`visited_resizes`, making the reuse visible in artifacts. For a normal batch
build, `visited_resizes` should stay near one while `search_calls` grows with the
number of inserted vectors and layers searched.

Verification added:

- `tests/test_hnsw_cpp.py` checks that a deterministic build performs multiple
  native search calls while resizing visited scratch only once.
- Existing C++ Euclidean, cosine, and batch-search parity tests continue to
  verify that result ordering and public distances are unchanged.
- `tests/test_benchmark_cli.py` checks that JSON and Markdown benchmark reports
  expose the scratch counters.

## Bounded Native Adjacency

What was there before: the native builder represented mutable graph edges as
nested `std::vector<std::vector<int>>` containers. Each layer's connection list
grew like an ordinary vector, even though HNSW has a known degree limit:
`2M` connections at layer 0 and `M` connections on upper layers. The code still
enforced degree limits through pruning, but the mutable adjacency shape did not
make those limits explicit.

What was implemented: each native build layer now uses a `BuildLayer` wrapper
with an explicit `max_connections` value and a pre-reserved neighbor vector. The
builder allocates layer storage through one `ensure_layer()` helper that assigns
the expected limit, reserves space for the bounded degree plus one overflow
entry, and tracks how many adjacency layer blocks were allocated. CSR export is
unchanged, so the public compact graph shape remains the same.

Why this matters: this is a layout cleanup that prepares the builder for deeper
native optimization. It reduces accidental vector growth in a bounded-degree
data structure and makes the degree contract easier to inspect. It also gives
future chunks a clearer place to replace the remaining vector-backed neighbor
storage with fixed-capacity or small-vector storage if measurements justify it.

An implementation detail surfaced during testing: the current construction
semantics can allocate a connection block above a node's sampled level when a
higher-layer insertion connects through that node. This behavior existed before
the bounded wrapper. The new tests preserve that behavior but verify the
important invariants: all emitted connection lists stay within the HNSW degree
limits, and allocated adjacency layers stay bounded by the possible node/layer
matrix.

Verification added:

- `tests/test_hnsw_cpp.py` checks that native build stats report bounded
  adjacency, adjacency layer allocation count, and maximum observed degree.
- Existing searchable-graph and batch-search parity tests continue to verify
  that CSR output and search results remain compatible.
- `tests/test_benchmark_cli.py` checks that benchmark JSON includes the bounded
  adjacency counters.

## Heuristic Native Neighbor Selection

What was there before: native graph construction collected candidate neighbors
with the HNSW layer search, sorted those candidates by distance to the inserted
vector, and kept the nearest `M` or `2M` ids. Overflow pruning used the same
nearest-only rule. This was simple and fast, but it could keep several neighbors
that sit in the same local direction and discard a slightly farther neighbor
that improves graph navigability.

What was implemented: the native selector now applies the standard HNSW
diversity heuristic when building each layer's neighbor set and when pruning a
neighbor list after adding a reverse edge. Candidates are still processed in
distance order, but a candidate is skipped when an already selected neighbor is
closer to that candidate than the inserted vector is. If the heuristic produces
fewer than the requested degree, the selector fills the remaining slots with the
closest non-selected candidates so the graph keeps useful degree density. Greedy
entry-point descent still uses nearest-only selection because that phase is
supposed to choose one closest waypoint, not a diversified neighbor set.

Why this matters: HNSW recall depends on graph navigability, not just local
nearest-neighbor degree. The heuristic tends to keep neighbors that point into
different regions of the vector space, which improves the chance that search can
escape a narrow local cluster. This is an accuracy-oriented C++ change: it may
add a small amount of distance work during construction and pruning, but it
targets recall quality and brings the native builder closer to production HNSW
implementations.

Verification added:

- `tests/test_hnsw_cpp.py` exposes the native selector through Cython and checks
  a geometry where nearest-only would keep a redundant candidate while the
  heuristic keeps a more diverse neighbor.
- Native build stats now report `uses_heuristic_neighbors` so benchmark reports
  make the active construction policy visible.
- Existing pruning, graph construction, batch-search, and benchmark contract
  tests continue to verify that the native graph remains bounded and searchable.

## Fast Reverse-Edge Pruning

What was there before: after adding heuristic native neighbor selection, the
builder applied the heuristic both when selecting the new node's outward
neighbors and when pruning an existing neighbor's reverse-edge list after an
overflow. The 100k SIFT1M run showed that this doubled build time: build time
rose to `107.75s`, with native prune time alone taking `40.89s`.

What was implemented: outward neighbor selection still uses the HNSW diversity
heuristic, preserving the recall-oriented edge choice for the newly inserted
node. Reverse-edge overflow pruning now uses the fast nearest-only selector.
The public `prune_connections()` wrapper still exposes heuristic pruning for
direct use, but the batch builder avoids the expensive heuristic loop on the
high-frequency reverse-prune path. Build stats now report
`uses_heuristic_reverse_pruning=false` so benchmark artifacts make this policy
visible.

Why this matters: reverse-edge pruning is called frequently during construction,
and each heuristic prune can compare candidate vectors against already selected
vectors. On `M=16`, that turns many small overflow repairs into repeated
candidate-to-selected distance scans. The policy split keeps the main accuracy
benefit from diversified outward links while reducing the build-time regression
from more than `2x` to about `1.36x` against the prior 100k benchmark.

Verification added:

- `tests/test_hnsw_cpp.py` checks that native build stats expose the reverse
  pruning policy.
- `tests/test_benchmark_cli.py` checks that benchmark JSON includes
  `uses_heuristic_reverse_pruning`.
- A 100k SIFT1M instrumented run after the change reported build time `66.17s`,
  recall@10 `0.9860`, native search time `51.80s`, and native prune time
  `7.36s`.
- A 100k SIFT1M ChromaDB comparison after the change reported our build time
  `66.19s`, recall@10 `98.40%`, and QPS `3779.7`, compared with ChromaDB build
  time `5.30s`, recall@10 `99.90%`, and QPS `5778.7`.

## Split Native Build-Search Profiling

What was there before: after fast reverse-edge pruning, the 100k SIFT1M run
showed native build search as the dominant phase, but the benchmark could not
separate upper-layer greedy descent from candidate collection on insertion
layers. That made the next optimization target ambiguous.

What was implemented: native build stats now split `search_seconds` and
`search_calls` into `greedy_search_seconds`, `candidate_search_seconds`,
`greedy_search_calls`, and `candidate_search_calls`. The aggregate
`search_seconds` and `search_calls` remain available for compatibility, and the
tests verify that the aggregate equals the split phases.

Why this matters: HNSW build uses search in two different ways. Upper-layer
descent chooses a single waypoint, while insertion-layer search collects an
`ef_construction` candidate set. They have different cost profiles and should
not be optimized blindly as one bucket. The 10k SIFT1M profiling run after this
change reported `126,005` greedy calls taking `0.124s`, while `20,155`
candidate-search calls took `1.668s`. That points the next work at candidate
collection and distance evaluation, not greedy descent.

Verification added:

- `tests/test_hnsw_cpp.py` checks the split search timing and call counters.
- `tests/test_benchmark_cli.py` checks that benchmark JSON exposes the split
  fields.
- `benchmarks/compare_results.py` can compare greedy and candidate search time
  between benchmark reports.
- A 10k SIFT1M benchmark reported build time `2.99s`, recall@10 `0.9980`, and
  the split described above.

## Float Native L2 Accumulation

What was there before: the native Euclidean squared-L2 kernel read float32
vectors but widened every component difference and accumulator to `double`.
That preserved extra precision, but SIFT vectors are float32 and HNSW internal
ordering does not need double precision for distance accumulation. The split
profiling showed insertion-layer candidate search dominated build-search time,
and every candidate expansion pays for distance evaluation.

What was implemented: the native squared-L2 kernel now accumulates in `float`
with four independent partial sums, then converts the final squared distance
back to `double` for the existing heap and result structures. Cosine remains on
its existing dot/norm path. Build stats now report
`uses_float_l2_accumulation` for Euclidean builds so benchmark artifacts make the
active distance kernel visible.

Why this matters: this moves the hot Euclidean path closer to production vector
index implementations that operate on float32 vectors with float32 distance
kernels. On 100k SIFT1M, this reduced native search time from `51.80s` to
`41.48s` and native prune time from `7.36s` to `4.19s`, while keeping recall@10
at `0.9860` in the standalone benchmark.

Verification added:

- `tests/test_hnsw_cpp.py` checks that Euclidean builds report
  `uses_float_l2_accumulation=true`, while cosine builds report `false`.
- `tests/test_benchmark_cli.py` checks that benchmark JSON exposes the flag.
- A 10k SIFT1M run improved build time from `2.99s` to `2.03s` with recall@10
  unchanged at `0.9980`.
- A 100k SIFT1M standalone run improved build time from `66.17s` to `49.95s`,
  with recall@10 unchanged at `0.9860`.
- A 100k SIFT1M ChromaDB comparison reported our build time `53.15s`, recall@10
  `98.50%`, and QPS `4692.6`, compared with ChromaDB build time `4.81s`,
  recall@10 `99.90%`, and QPS `6054.8`.

## Reusable Native Search Heaps

What was there before: every mutable-layer build search created fresh
`std::priority_queue` containers for the candidate frontier and result set. At
100k SIFT1M, the builder performed `1,742,434` mutable-layer searches, so even
small per-call heap container allocation overhead showed up in the candidate
search phase.

What was implemented: `SearchScratch` now owns reusable candidate and result
heap vectors for mutable-layer search. Each call clears the vectors and uses
`std::push_heap` / `std::pop_heap` with the same min-heap and max-heap
comparators as the old `std::priority_queue` path. The vectors keep their
capacity between searches, so the build avoids repeated heap container
allocation while preserving the existing HNSW search semantics. Build stats now
report `uses_reusable_search_heaps` and `search_heap_resizes`.

Why this matters: after float L2 accumulation, the remaining build bottleneck
was still insertion-layer candidate search. Heap reuse targets that cost without
changing graph quality or HNSW parameters. It is not the final answer to the
ChromaDB gap, but it is a low-risk C++ memory-management improvement on a path
called hundreds of thousands to millions of times per build.

Verification added:

- `tests/test_hnsw_cpp.py` checks that native builds report reusable search
  heaps and that heap reserves are far below total search calls.
- `tests/test_benchmark_cli.py` checks that benchmark JSON exposes the heap
  reuse fields.
- `benchmarks/compare_results.py` can compare `search_heap_resizes` between
  benchmark reports.
- A 10k SIFT1M run improved build time from `2.03s` to `1.91s`, with recall@10
  unchanged at `0.9980`.
- A 100k SIFT1M standalone run improved build time from `49.95s` to `44.64s`,
  with recall@10 unchanged at `0.9860`; candidate-search time dropped from
  `39.70s` to `35.30s`.
- A 100k SIFT1M ChromaDB comparison reported our build time `44.92s`, recall@10
  `98.60%`, and QPS `4768.0`, compared with ChromaDB build time `4.65s`,
  recall@10 `99.90%`, and QPS `6372.7`.

### Next Steps

- Write the implementation plan for the C++ parity phase as small, testable,
  commit-sized chunks.
- Use the instrumentation to compare native build phases before and after the
  squared-L2, visited-scratch, bounded-adjacency, and heuristic-neighbor
  changes on the same benchmark shape.
- Add an accuracy-focused benchmark slice that reports recall deltas across
  nearest-only versus heuristic construction on clustered data.
- Optimize insertion-layer candidate search next. After reusable search heaps,
  the 100k run still spends `35.30s` in candidate search, while greedy descent
  is only `1.43s`.
- Leave native online mutation for a separate future design after the
  batch-built read index path is closer to ChromaDB.

---

## Future Improvements

### Phase 6 (Advanced Features)

**1. Write-Ahead Log (WAL)**
- Record all mutations before applying
- Enable point-in-time recovery
- Allow replay for crash recovery

**2. Multi-threading**
- Use multiprocessing for parallel queries
- Thread pool for I/O operations
- Lock-free data structures where possible

**3. Distributed Architecture**
- Shard vectors across nodes
- Consistent hashing for routing
- Replication for fault tolerance

**4. Advanced Filtering**
- Pre-filtering with inverted indexes
- Range queries on metadata
- Complex boolean filters

**5. Monitoring and Metrics**
- Prometheus metrics export
- Query latency histograms
- Cache hit rates
- Resource utilization

**6. Compression**
- Product Quantization for smaller index
- Scalar quantization (float32 to int8)
- Trade precision for memory

### Nice-to-Have Features

- **Hot-reload:** Update index without downtime
- **Versioning:** Rollback to previous index versions
- **Streaming inserts:** Real-time index updates
- **GPU acceleration:** Use CUDA for distance calculations
- **Multi-modal search:** Handle images + text
- **Approximate joins:** Find similar pairs in dataset

---

## Conclusion

This project demonstrates a deep understanding of:
- Vector similarity search algorithms (HNSW)
- Database systems design (storage, indexing, metadata)
- Performance optimization (vectorization, memory-mapping)
- API design (REST with FastAPI)
- Production considerations (Docker, persistence, monitoring)

The implementation is educational but production-capable for small-to-medium scale deployments (millions of vectors). Scaling to billions would require distributed systems work.

**Most importantly:** Building this from scratch provides insights you can't get from using a library. Understanding the trade-offs between accuracy and speed, the impact of parameter tuning, and the challenges of persistence gives you the foundation to make informed decisions when working with any vector database.
