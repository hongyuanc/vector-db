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
- Vectors: `10000`
- Dimension: `128`
- Queries: `100`
- Seed: `42`

## HNSW Configuration

- M: `16`
- ef_construction: `200`
- ef_search: `50`
- Metric: `euclidean`

## Metrics

| Metric | Value |
|---|---:|
| Build Time | 2.4171s |
| Build Throughput | 4137.25 vectors/sec |
| QPS | 10902.50 |
| Average Latency | 0.0911 ms |
| p50 Latency | 0.0918 ms |
| p95 Latency | 0.1187 ms |
| p99 Latency | 0.1293 ms |
| Recall@10 | 0.9910 |
| Vector Data | 4.88 MiB |
| Python Graph Materialized | False |
| Python Graph | 1.4257 MiB |
| Python Graph Edges | 0 |
| C++ CSR Graph | 2.4890 MiB |
| C++ CSR Edges | 482450 |
| Graph Total | 3.9147 MiB |
| Process Peak RSS | 4608.78 MiB |
| Compact Persistence Available | True |
| Compact Save Time | 0.009300s |
| Compact Load Time | 0.006571s |
| Compact File Size | 7.6885 MiB |
| Compact Load Peak RSS | 4608.7800 MiB |
| Materialized Save Time | 0.042441s |
| Materialized Load Time | 0.185330s |
| Materialized File Size | 6.6740 MiB |
| Materialized Load Peak RSS | 4608.7800 MiB |
