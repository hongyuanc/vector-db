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
| Build Time | 8.1387s |
| Build Throughput | 12287.04 vectors/sec |
| C++ Build Total | 30.990439s |
| C++ Build Construction | 30.924880s |
| C++ Build Search | 22.242519s |
| C++ Greedy Search | 0.952338s |
| C++ Candidate Search | 21.290182s |
| C++ Build Prune | 4.651564s |
| C++ CSR Export | 0.065553s |
| C++ Float L2 Accumulation | True |
| C++ Directed Edges | 4798986 |
| C++ Build Search Calls | 1539122 |
| C++ Greedy Search Calls | 1339191 |
| C++ Candidate Search Calls | 199931 |
| C++ Visited Resizes | 4 |
| C++ Reusable Search Heaps | True |
| C++ Search Heap Resizes | 12 |
| C++ Bounded Adjacency | True |
| C++ Heuristic Neighbors | True |
| C++ Heuristic Reverse Pruning | False |
| C++ Adjacency Layers | 199999 |
| C++ Max Observed Degree | 32 |
| C++ Distance Evaluations | 418968012 |
| C++ Search Distance Evaluations | 200615415 |
| C++ Neighbor Selection Distance Evaluations | 85916661 |
| C++ Prune Distance Evaluations | 132435936 |
| C++ Visited Nodes | 200615415 |
| C++ Max Visited Nodes Per Search | 3932 |
| C++ Candidate Heap Pushes | 62566858 |
| C++ Result Heap Pushes | 62566858 |
| C++ Neighbor Selection Calls | 199931 |
| C++ Selected Degree Total | 4790773 |
| C++ Average Selected Degree | 23.962131935517753 |
| C++ Max Selected Degree | 32 |
| C++ Prune Calls | 4782560 |
| C++ Prune Input Total | 132435936 |
| C++ Average Prune Input Size | 27.691432203673347 |
| C++ Max Prune Input Size | 33 |
| Segmented Build | True |
| Segment Count | 4 |
| Build Threads | 4 |
| Max Segment Build | 8.036649s |
| Sum Segment Build | 31.279521s |
| QPS | 1035.96 |
| Average Latency | 0.9641 ms |
| p50 Latency | 0.9494 ms |
| p95 Latency | 1.3680 ms |
| p99 Latency | 1.4544 ms |
| Recall@10 | 0.9980 |
| Vector Data | 48.83 MiB |
| Python Graph Materialized | False |
| Python Graph | 12.2070 MiB |
| Python Graph Edges | 0 |
| C++ CSR Graph | 24.7919 MiB |
| C++ CSR Edges | 4798986 |
| Graph Total | 36.9990 MiB |
| Process Peak RSS | 4781.34 MiB |
| Compact Persistence Available | False |
| Compact Save Time | n/a |
| Compact Load Time | n/a |
| Compact File Size | n/a |
| Compact Load Peak RSS | 4781.3400 MiB |
| Materialized Save Time | n/a |
| Materialized Load Time | n/a |
| Materialized File Size | n/a |
| Materialized Load Peak RSS | 4781.3400 MiB |
