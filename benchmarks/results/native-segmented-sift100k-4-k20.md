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
| Build Time | 6.2569s |
| Build Throughput | 15982.48 vectors/sec |
| C++ Build Total | 23.899788s |
| C++ Build Construction | 23.823247s |
| C++ Build Search | 15.928359s |
| C++ Greedy Search | 0.720883s |
| C++ Candidate Search | 15.207476s |
| C++ Build Prune | 4.237270s |
| C++ CSR Export | 0.076533s |
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
| Max Segment Build | 6.166097s |
| Sum Segment Build | 24.090486s |
| QPS | 1574.19 |
| Batch Search API | True |
| Native Segmented Batch | True |
| Average Latency | 0.5310 ms |
| p50 Latency | 0.5117 ms |
| p95 Latency | 0.8053 ms |
| p99 Latency | 0.9079 ms |
| Recall@10 | 0.9980 |
| Vector Data | 48.83 MiB |
| Python Graph Materialized | False |
| Python Graph | 12.2070 MiB |
| Python Graph Edges | 0 |
| C++ CSR Graph | 24.7919 MiB |
| C++ CSR Edges | 4798986 |
| Graph Total | 36.9990 MiB |
| Process Peak RSS | 5398.27 MiB |
| Compact Persistence Available | False |
| Compact Save Time | n/a |
| Compact Load Time | n/a |
| Compact File Size | n/a |
| Compact Load Peak RSS | 5398.2700 MiB |
| Materialized Save Time | n/a |
| Materialized Load Time | n/a |
| Materialized File Size | n/a |
| Materialized Load Peak RSS | 5398.2700 MiB |
