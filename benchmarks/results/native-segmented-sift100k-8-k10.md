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
- Segment Search K: `10`

## Metrics

| Metric | Value |
|---|---:|
| Build Time | 2.8071s |
| Build Throughput | 35623.82 vectors/sec |
| C++ Build Total | 20.519156s |
| C++ Build Construction | 20.462565s |
| C++ Build Search | 12.449477s |
| C++ Greedy Search | 0.574862s |
| C++ Candidate Search | 11.874615s |
| C++ Build Prune | 4.463005s |
| C++ CSR Export | 0.056583s |
| C++ Float L2 Accumulation | True |
| C++ Directed Edges | 4798778 |
| C++ Build Search Calls | 1352966 |
| C++ Greedy Search Calls | 1153039 |
| C++ Candidate Search Calls | 199927 |
| C++ Visited Resizes | 8 |
| C++ Reusable Search Heaps | True |
| C++ Search Heap Resizes | 20 |
| C++ Bounded Adjacency | True |
| C++ Heuristic Neighbors | True |
| C++ Heuristic Reverse Pruning | False |
| C++ Adjacency Layers | 200052 |
| C++ Max Observed Degree | 32 |
| C++ Distance Evaluations | 372298665 |
| C++ Search Distance Evaluations | 159280987 |
| C++ Neighbor Selection Distance Evaluations | 80887054 |
| C++ Prune Distance Evaluations | 132130624 |
| C++ Visited Nodes | 159280987 |
| C++ Max Visited Nodes Per Search | 3173 |
| C++ Candidate Heap Pushes | 57544010 |
| C++ Result Heap Pushes | 57544010 |
| C++ Neighbor Selection Calls | 199927 |
| C++ Selected Degree Total | 4783677 |
| C++ Average Selected Degree | 23.92711839821535 |
| C++ Max Selected Degree | 32 |
| C++ Prune Calls | 4768576 |
| C++ Prune Input Total | 132130624 |
| C++ Average Prune Input Size | 27.70861238239676 |
| C++ Max Prune Input Size | 33 |
| Segmented Build | True |
| Segment Count | 8 |
| Build Threads | 8 |
| Max Segment Build | 2.714923s |
| Sum Segment Build | 20.851606s |
| QPS | 953.44 |
| Batch Search API | True |
| Native Segmented Batch | True |
| Average Latency | 0.9773 ms |
| p50 Latency | 0.9498 ms |
| p95 Latency | 1.4484 ms |
| p99 Latency | 1.5424 ms |
| Recall@10 | 0.9940 |
| Vector Data | 48.83 MiB |
| Python Graph Materialized | False |
| Python Graph | 12.2070 MiB |
| Python Graph Edges | 0 |
| C++ CSR Graph | 24.2668 MiB |
| C++ CSR Edges | 4798778 |
| Graph Total | 36.4739 MiB |
| Process Peak RSS | 5436.58 MiB |
| Compact Persistence Available | False |
| Compact Save Time | n/a |
| Compact Load Time | n/a |
| Compact File Size | n/a |
| Compact Load Peak RSS | 5436.5800 MiB |
| Materialized Save Time | n/a |
| Materialized Load Time | n/a |
| Materialized File Size | n/a |
| Materialized Load Peak RSS | 5436.5800 MiB |
