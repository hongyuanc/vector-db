# Benchmark Results

## Run Context

- Commit: `9e2fb66`
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

## Metrics

| Metric | Value |
|---|---:|
| Build Time | 18.1246s |
| Build Throughput | 5517.36 vectors/sec |
| C++ Build Total | 35.644196s |
| C++ Build Construction | 35.581559s |
| C++ Build Search | 26.956488s |
| C++ Greedy Search | 1.112366s |
| C++ Candidate Search | 25.844122s |
| C++ Build Prune | 4.630792s |
| C++ CSR Export | 0.062630s |
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
| Max Segment Build | 18.024841s |
| Sum Segment Build | 35.852703s |
| QPS | 1900.17 |
| Average Latency | 0.5251 ms |
| p50 Latency | 0.5223 ms |
| p95 Latency | 0.7102 ms |
| p99 Latency | 0.7728 ms |
| Recall@10 | 0.9940 |
| Vector Data | 48.83 MiB |
| Python Graph Materialized | False |
| Python Graph | 12.2070 MiB |
| Python Graph Edges | 0 |
| C++ CSR Graph | 24.9833 MiB |
| C++ CSR Edges | 4799184 |
| Graph Total | 37.1903 MiB |
| Process Peak RSS | 4596.34 MiB |
| Compact Persistence Available | False |
| Compact Save Time | n/a |
| Compact Load Time | n/a |
| Compact File Size | n/a |
| Compact Load Peak RSS | 4596.3400 MiB |
| Materialized Save Time | n/a |
| Materialized Load Time | n/a |
| Materialized File Size | n/a |
| Materialized Load Peak RSS | 4596.3400 MiB |
