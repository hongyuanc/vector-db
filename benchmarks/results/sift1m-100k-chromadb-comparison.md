# ChromaDB Comparison

- Dataset: `sift1m`
- Vectors: `100000`
- Queries: `100`
- M: `16`
- ef_construction: `200`
- ef_search: `100`

| System | Build Time | Batch QPS | Batch Avg Latency | Single p50 | Single p95 | Single p99 | Recall@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Our HNSW C++/CSR | 44.9216s | 4768.04 | 0.2097ms | 0.3506ms | 0.5282ms | 1.2123ms | 0.9860 |
| ChromaDB | 4.6508s | 6372.68 | 0.1569ms | 0.3600ms | 0.4306ms | 0.5693ms | 0.9990 |
