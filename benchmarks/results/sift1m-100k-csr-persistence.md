# Benchmark Results

## Run Context

- Commit: `b6e0ebd`
- Dirty worktree: `True`
- Python: `3.11.14`
- NumPy: `2.3.5`
- Cython extensions available: `True`
- C++ HNSW extension available: `True`

## Dataset

- Name: `sift1m`
- Vectors: `100000`
- Dimension: `128`
- Queries: `1000`
- Seed: `42`

## HNSW Configuration

- M: `16`
- ef_construction: `200`
- ef_search: `50`
- Metric: `euclidean`

## Metrics

| Metric | Value |
|---|---:|
| Build Time | 48.6693s |
| Build Throughput | 2054.68 vectors/sec |
| QPS | 5668.19 |
| Average Latency | 0.1758 ms |
| p50 Latency | 0.1738 ms |
| p95 Latency | 0.2274 ms |
| p99 Latency | 0.2569 ms |
| Recall@10 | 0.9467 |
| Vector Data | 48.83 MiB |
| Python Graph Materialized | False |
| Python Graph | 16.4442 MiB |
| Python Graph Edges | 0 |
| C++ CSR Graph | 25.1740 MiB |
| C++ CSR Edges | 4799206 |
| Graph Total | 41.6182 MiB |
| Process Peak RSS | 5463.17 MiB |
| Compact Persistence Available | True |
| Compact Save Time | 0.108971s |
| Compact Load Time | 0.068366s |
| Compact File Size | 77.1979 MiB |
| Compact Load Peak RSS | 5463.1700 MiB |
| Materialized Save Time | 0.771423s |
| Materialized Load Time | 2.747778s |
| Materialized File Size | 69.1201 MiB |
| Materialized Load Peak RSS | 5463.1700 MiB |
