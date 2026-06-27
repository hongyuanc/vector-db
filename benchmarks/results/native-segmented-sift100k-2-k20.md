# Benchmark Results

## Run Context

- Commit: `1a5e15a`
- Dirty worktree: `True`
- Python: `3.11.14`
- NumPy: `2.3.5`
- Cython extensions available: `True`
- C++ HNSW extension available: `True`

## Dataset

- Name: `sift1m`
- Vectors: `100000`
- Dimension: `128`
- Queries: `100`
- Seed: `42`

## HNSW Configuration

- M: `16`
- ef_construction: `200`
- ef_search: `100`
- Metric: `euclidean`
- Segment Search K: `20`

## Metrics

| Metric | Value |
|---|---:|
| Build Time | 13.0733s |
| Build Throughput | 7649.15 vectors/sec |
| C++ Build Total | 25.662059s |
| C++ Build Construction | 25.604948s |
| C++ Build Search | 18.417167s |
| C++ Greedy Search | 0.748079s |
| C++ Candidate Search | 17.669087s |
| C++ Build Prune | 3.876380s |
| C++ CSR Export | 0.057105s |
| C++ Float L2 Accumulation | True |
| C++ Directed Edges | 4799184 |
| C++ Build Search Calls | 1657450 |
| C++ Greedy Search Calls | 1457505 |
| C++ Candidate Search Calls | 199945 |
| C++ Visited Resizes | 2 |
| C++ Reusable Search Heaps | True |
| C++ Search Heap Resizes | 6 |
| C++ Bounded Adjacency | True |
| C++ Heuristic Neighbors | True |
| C++ Heuristic Reverse Pruning | False |
| C++ Adjacency Layers | 199980 |
| C++ Max Observed Degree | 32 |
| C++ Distance Evaluations | 464918578 |
| C++ Search Distance Evaluations | 244019229 |
| C++ Neighbor Selection Distance Evaluations | 88296069 |
| C++ Prune Distance Evaluations | 132603280 |
| C++ Visited Nodes | 244019229 |
| C++ Max Visited Nodes Per Search | 4820 |
| C++ Candidate Heap Pushes | 67484115 |
| C++ Result Heap Pushes | 67484115 |
| C++ Neighbor Selection Calls | 199945 |
| C++ Selected Degree Total | 4794800 |
| C++ Average Selected Degree | 23.98059466353247 |
| C++ Max Selected Degree | 32 |
| C++ Prune Calls | 4790416 |
| C++ Prune Input Total | 132603280 |
| C++ Average Prune Input Size | 27.68095296942896 |
| C++ Max Prune Input Size | 33 |
| Segmented Build | True |
| Segment Count | 2 |
| Build Threads | 2 |
| Max Segment Build | 12.987988s |
| Sum Segment Build | 25.968410s |
| QPS | 2077.30 |
| Batch Search API | True |
| Native Segmented Batch | True |
| Average Latency | 0.3626 ms |
| p50 Latency | 0.3679 ms |
| p95 Latency | 0.5161 ms |
| p99 Latency | 0.5695 ms |
| Recall@10 | 0.9940 |
| Vector Data | 48.83 MiB |
| Python Graph Materialized | False |
| Python Graph | 12.2070 MiB |
| Python Graph Edges | 0 |
| C++ CSR Graph | 24.9833 MiB |
| C++ CSR Edges | 4799184 |
| Graph Total | 37.1903 MiB |
| Process Peak RSS | 5258.56 MiB |
| Compact Persistence Available | False |
| Compact Save Time | n/a |
| Compact Load Time | n/a |
| Compact File Size | n/a |
| Compact Load Peak RSS | 5258.5600 MiB |
| Materialized Save Time | n/a |
| Materialized Load Time | n/a |
| Materialized File Size | n/a |
| Materialized Load Peak RSS | 5258.5600 MiB |
