# HNSW 100k Build-Time Instrumentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add deeper native build instrumentation so the next HNSW 100k build-time optimization is chosen from measured construction costs instead of assumptions.

**Architecture:** Keep Python as the public API and reporting layer. Add native C++ counters inside the HNSW batch builder, expose them through the Cython wrapper, normalize them through the benchmark runner, then record the instrumentation baseline in the technical documentation.

**Tech Stack:** C++17, Cython, NumPy, pytest, project `venv`, existing benchmark CLI.

---

## File Structure

- Modify `src/index/hnsw_cpp_core.hpp`
  - Owns the native `BuildStats` shape returned by the C++ builder.
  - Add counter fields for distance evaluations, visited nodes, heap pushes, pruning, and selected degrees.

- Modify `src/index/hnsw_cpp_core.cpp`
  - Owns the native HNSW construction hot path.
  - Increment counters inside `search_mutable_layer()`, neighbor selection, and pruning.

- Modify `src/index/hnsw_cpp.pyx`
  - Owns conversion from native `BuildStats` into Python dictionaries.
  - Add the same fields to the Cython extern declaration and returned `build_stats`.

- Modify `benchmarks/benchmark.py`
  - Owns stable benchmark JSON and Markdown reporting.
  - Add the new keys to `CPP_BUILD_STATS_KEYS` and Markdown output.

- Modify `tests/test_hnsw_cpp.py`
  - Owns focused native wrapper and HNSW C++ behavior tests.
  - Add failing tests for the new counters, then update fake build-stat fixtures.

- Modify `tests/test_benchmark_cli.py`
  - Owns benchmark schema tests.
  - Add expected keys and Markdown rows for the new counters.

- Modify `TECHNICAL.md`
  - Owns the educational project narrative.
  - Add a short instrumentation-baseline section after running the small benchmark.

---

### Task 1: Add Native Build Counter Tests

**Files:**
- Modify: `tests/test_hnsw_cpp.py`
- Test: `tests/test_hnsw_cpp.py`

- [ ] **Step 1: Add the failing native counter test**

Append this test after `test_cpp_build_graph_returns_phase_stats()` in `tests/test_hnsw_cpp.py`:

```python
def test_cpp_build_graph_reports_detailed_build_counters():
    from src.index import hnsw_cpp

    vectors = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.2, 0.0],
            [0.3, 0.0],
            [2.0, 2.0],
            [2.1, 2.0],
            [2.2, 2.0],
        ],
        dtype=np.float32,
    )
    levels = np.array([0, 1, 0, 0, 1, 0, 0], dtype=np.int32)

    graph = hnsw_cpp.build_graph(
        vectors=vectors,
        levels=levels,
        max_connections=2,
        ef_construction=4,
        metric="euclidean",
        include_connections=False,
    )

    stats = graph["build_stats"]
    assert stats["distance_evaluations"] > 0
    assert stats["search_distance_evaluations"] > 0
    assert stats["neighbor_selection_distance_evaluations"] > 0
    assert stats["prune_distance_evaluations"] >= 0
    assert stats["distance_evaluations"] == (
        stats["search_distance_evaluations"]
        + stats["neighbor_selection_distance_evaluations"]
        + stats["prune_distance_evaluations"]
    )
    assert stats["visited_nodes"] == stats["search_distance_evaluations"]
    assert stats["max_visited_nodes_per_search"] > 0
    assert stats["candidate_heap_pushes"] >= stats["visited_nodes"]
    assert stats["result_heap_pushes"] >= stats["visited_nodes"]
    assert stats["neighbor_selection_calls"] > 0
    assert stats["selected_degree_total"] > 0
    assert stats["average_selected_degree"] > 0.0
    assert stats["max_selected_degree"] <= 4
    assert stats["prune_calls"] >= 0
    assert stats["max_prune_input_size"] >= 0
    assert stats["average_prune_input_size"] >= 0.0
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
venv/bin/python -m pytest tests/test_hnsw_cpp.py::test_cpp_build_graph_reports_detailed_build_counters -q
```

Expected: FAIL with `KeyError: 'distance_evaluations'`.

- [ ] **Step 3: Update the fake stats fixture used by `HNSWIndex.build()`**

In `tests/test_hnsw_cpp.py`, find the `build_stats` dictionary inside
`test_hnsw_build_stores_cpp_build_stats()` and add these keys after
`"max_observed_degree": 1,`:

```python
                    "distance_evaluations": 12,
                    "search_distance_evaluations": 8,
                    "neighbor_selection_distance_evaluations": 3,
                    "prune_distance_evaluations": 1,
                    "visited_nodes": 8,
                    "max_visited_nodes_per_search": 4,
                    "candidate_heap_pushes": 8,
                    "result_heap_pushes": 8,
                    "neighbor_selection_calls": 1,
                    "selected_degree_total": 2,
                    "average_selected_degree": 2.0,
                    "max_selected_degree": 2,
                    "prune_calls": 1,
                    "prune_input_total": 3,
                    "average_prune_input_size": 3.0,
                    "max_prune_input_size": 3,
```

Update the final `assert index._last_cpp_build_stats == {...}` dictionary in the
same test with the identical keys and values.

- [ ] **Step 4: Run the fake stats test to verify it still passes after fixture update**

Run:

```bash
venv/bin/python -m pytest tests/test_hnsw_cpp.py::test_hnsw_build_stores_cpp_build_stats -q
```

Expected: PASS. This test uses a fake module and does not require the native
implementation to expose the new keys yet.

- [ ] **Step 5: Commit the failing test chunk**

```bash
git add tests/test_hnsw_cpp.py
git commit -m "test: specify hnsw build instrumentation counters"
```

---

### Task 2: Implement Native Build Counters

**Files:**
- Modify: `src/index/hnsw_cpp_core.hpp`
- Modify: `src/index/hnsw_cpp_core.cpp`
- Modify: `src/index/hnsw_cpp.pyx`
- Test: `tests/test_hnsw_cpp.py`

- [ ] **Step 1: Add fields to the native `BuildStats` struct**

In `src/index/hnsw_cpp_core.hpp`, add these fields at the end of `struct BuildStats`:

```cpp
    long long distance_evaluations = 0;
    long long search_distance_evaluations = 0;
    long long neighbor_selection_distance_evaluations = 0;
    long long prune_distance_evaluations = 0;
    long long visited_nodes = 0;
    int max_visited_nodes_per_search = 0;
    long long candidate_heap_pushes = 0;
    long long result_heap_pushes = 0;
    long long neighbor_selection_calls = 0;
    long long selected_degree_total = 0;
    double average_selected_degree = 0.0;
    int max_selected_degree = 0;
    long long prune_calls = 0;
    long long prune_input_total = 0;
    double average_prune_input_size = 0.0;
    int max_prune_input_size = 0;
```

- [ ] **Step 2: Add instrumentation helper structs in C++**

In `src/index/hnsw_cpp_core.cpp`, inside the anonymous namespace after
`struct SearchScratch`, add:

```cpp
struct SearchInstrumentation {
    long long distance_evaluations = 0;
    long long visited_nodes = 0;
    int max_visited_nodes_per_search = 0;
    long long candidate_heap_pushes = 0;
    long long result_heap_pushes = 0;
};

struct DistanceInstrumentation {
    long long neighbor_selection_distance_evaluations = 0;
    long long prune_distance_evaluations = 0;
};
```

- [ ] **Step 3: Count distance work inside mutable layer search**

Change the `search_mutable_layer()` signature in `src/index/hnsw_cpp_core.cpp`
from:

```cpp
std::vector<HeapItem> search_mutable_layer(
    const float* query,
    const float* vectors,
    int n_vectors,
    int dimension,
    const std::vector<BuildNode>& nodes,
    const std::vector<int>& entry_points,
    int num_closest,
    int layer,
    bool use_euclidean,
    SearchScratch& scratch
)
```

to:

```cpp
std::vector<HeapItem> search_mutable_layer(
    const float* query,
    const float* vectors,
    int n_vectors,
    int dimension,
    const std::vector<BuildNode>& nodes,
    const std::vector<int>& entry_points,
    int num_closest,
    int layer,
    bool use_euclidean,
    SearchScratch& scratch,
    SearchInstrumentation* instrumentation
)
```

At the start of the function, after `scratch.prepare_heaps(...)`, add:

```cpp
    long long visited_this_search = 0;
```

After each `scratch.mark_visited(entry_id);`, add:

```cpp
        ++visited_this_search;
        if (instrumentation != nullptr) {
            ++instrumentation->visited_nodes;
            ++instrumentation->distance_evaluations;
        }
```

After each `candidates.push_back({distance, entry_id});`, add:

```cpp
        if (instrumentation != nullptr) {
            ++instrumentation->candidate_heap_pushes;
        }
```

After each `results.push_back({distance, entry_id});`, add:

```cpp
        if (instrumentation != nullptr) {
            ++instrumentation->result_heap_pushes;
        }
```

Repeat the same four increments for the neighbor path immediately after
`scratch.mark_visited(neighbor_id);`, `candidates.push_back({distance, neighbor_id});`,
and `results.push_back({distance, neighbor_id});`.

Before `return order_heap_results(results);`, add:

```cpp
    if (instrumentation != nullptr) {
        instrumentation->max_visited_nodes_per_search = std::max(
            instrumentation->max_visited_nodes_per_search,
            static_cast<int>(visited_this_search)
        );
    }
```

- [ ] **Step 4: Count neighbor-selection and prune distance work**

Change the `select_heuristic_neighbor_ids()` signature from:

```cpp
std::vector<int> select_heuristic_neighbor_ids(
    const float* vectors,
    int n_vectors,
    int dimension,
    int node_id,
    const std::vector<HeapItem>& candidates,
    int max_connections,
    bool use_euclidean
)
```

to:

```cpp
std::vector<int> select_heuristic_neighbor_ids(
    const float* vectors,
    int n_vectors,
    int dimension,
    int node_id,
    const std::vector<HeapItem>& candidates,
    int max_connections,
    bool use_euclidean,
    long long* distance_counter
)
```

Inside the inner loop, immediately after the `heap_distance(...)` call that
computes `selected_distance`, add:

```cpp
            if (distance_counter != nullptr) {
                ++(*distance_counter);
            }
```

Update all calls to `select_heuristic_neighbor_ids()`:

```cpp
        nullptr
```

for public helper paths where build stats are not being collected.

For the build path, pass:

```cpp
                &distance_instrumentation.neighbor_selection_distance_evaluations
```

Change `prune_connection_vector()` to accept one extra argument:

```cpp
    long long* distance_counter
```

After each distance computation in `prune_connection_vector()`, add:

```cpp
        if (distance_counter != nullptr) {
            ++(*distance_counter);
        }
```

When `prune_connection_vector()` calls `select_heuristic_neighbor_ids()`, pass
the same `distance_counter`. Existing calls outside the build path should pass
`nullptr`.

- [ ] **Step 5: Wire counters through `build_graph()`**

In `build_graph()` in `src/index/hnsw_cpp_core.cpp`, after `SearchScratch search_scratch;`, add:

```cpp
    SearchInstrumentation search_instrumentation;
    DistanceInstrumentation distance_instrumentation;
    long long neighbor_selection_calls = 0;
    long long selected_degree_total = 0;
    int max_selected_degree = 0;
    long long prune_calls = 0;
    long long prune_input_total = 0;
    int max_prune_input_size = 0;
```

Pass `&search_instrumentation` into both `search_mutable_layer()` calls in
`build_graph()`.

Immediately before the build-path `select_heuristic_neighbor_ids()` call, add:

```cpp
            ++neighbor_selection_calls;
```

Immediately after `selected_neighbors` is returned, add:

```cpp
            selected_degree_total += static_cast<long long>(selected_neighbors.size());
            max_selected_degree = std::max(
                max_selected_degree,
                static_cast<int>(selected_neighbors.size())
            );
```

Immediately before build-path pruning, add:

```cpp
                    ++prune_calls;
                    prune_input_total += static_cast<long long>(neighbor_connections.size());
                    max_prune_input_size = std::max(
                        max_prune_input_size,
                        static_cast<int>(neighbor_connections.size())
                    );
```

Pass `&distance_instrumentation.prune_distance_evaluations` into the build-path
`prune_connection_vector()` call.

- [ ] **Step 6: Populate `BuildStats`**

At the end of `build_graph()`, before assigning `result.build_stats.total_seconds`, add:

```cpp
    result.build_stats.search_distance_evaluations =
        search_instrumentation.distance_evaluations;
    result.build_stats.neighbor_selection_distance_evaluations =
        distance_instrumentation.neighbor_selection_distance_evaluations;
    result.build_stats.prune_distance_evaluations =
        distance_instrumentation.prune_distance_evaluations;
    result.build_stats.distance_evaluations =
        result.build_stats.search_distance_evaluations
        + result.build_stats.neighbor_selection_distance_evaluations
        + result.build_stats.prune_distance_evaluations;
    result.build_stats.visited_nodes = search_instrumentation.visited_nodes;
    result.build_stats.max_visited_nodes_per_search =
        search_instrumentation.max_visited_nodes_per_search;
    result.build_stats.candidate_heap_pushes =
        search_instrumentation.candidate_heap_pushes;
    result.build_stats.result_heap_pushes =
        search_instrumentation.result_heap_pushes;
    result.build_stats.neighbor_selection_calls = neighbor_selection_calls;
    result.build_stats.selected_degree_total = selected_degree_total;
    result.build_stats.average_selected_degree = neighbor_selection_calls == 0
        ? 0.0
        : static_cast<double>(selected_degree_total) / static_cast<double>(neighbor_selection_calls);
    result.build_stats.max_selected_degree = max_selected_degree;
    result.build_stats.prune_calls = prune_calls;
    result.build_stats.prune_input_total = prune_input_total;
    result.build_stats.average_prune_input_size = prune_calls == 0
        ? 0.0
        : static_cast<double>(prune_input_total) / static_cast<double>(prune_calls);
    result.build_stats.max_prune_input_size = max_prune_input_size;
```

- [ ] **Step 7: Expose new fields through Cython**

In `src/index/hnsw_cpp.pyx`, add these fields to `cdef cppclass CppBuildStats`:

```cython
        long long distance_evaluations
        long long search_distance_evaluations
        long long neighbor_selection_distance_evaluations
        long long prune_distance_evaluations
        long long visited_nodes
        int max_visited_nodes_per_search
        long long candidate_heap_pushes
        long long result_heap_pushes
        long long neighbor_selection_calls
        long long selected_degree_total
        double average_selected_degree
        int max_selected_degree
        long long prune_calls
        long long prune_input_total
        double average_prune_input_size
        int max_prune_input_size
```

In the returned `"build_stats"` dictionary, add:

```cython
            "distance_evaluations": raw.build_stats.distance_evaluations,
            "search_distance_evaluations": raw.build_stats.search_distance_evaluations,
            "neighbor_selection_distance_evaluations": raw.build_stats.neighbor_selection_distance_evaluations,
            "prune_distance_evaluations": raw.build_stats.prune_distance_evaluations,
            "visited_nodes": raw.build_stats.visited_nodes,
            "max_visited_nodes_per_search": raw.build_stats.max_visited_nodes_per_search,
            "candidate_heap_pushes": raw.build_stats.candidate_heap_pushes,
            "result_heap_pushes": raw.build_stats.result_heap_pushes,
            "neighbor_selection_calls": raw.build_stats.neighbor_selection_calls,
            "selected_degree_total": raw.build_stats.selected_degree_total,
            "average_selected_degree": raw.build_stats.average_selected_degree,
            "max_selected_degree": raw.build_stats.max_selected_degree,
            "prune_calls": raw.build_stats.prune_calls,
            "prune_input_total": raw.build_stats.prune_input_total,
            "average_prune_input_size": raw.build_stats.average_prune_input_size,
            "max_prune_input_size": raw.build_stats.max_prune_input_size,
```

- [ ] **Step 8: Rebuild native extensions**

Run:

```bash
venv/bin/python setup.py build_ext --inplace
```

Expected: build completes without C++ or Cython errors.

- [ ] **Step 9: Run native counter tests**

Run:

```bash
venv/bin/python -m pytest tests/test_hnsw_cpp.py::test_cpp_build_graph_reports_detailed_build_counters tests/test_hnsw_cpp.py::test_cpp_build_graph_returns_phase_stats tests/test_hnsw_cpp.py::test_hnsw_build_stores_cpp_build_stats -q
```

Expected: PASS.

- [ ] **Step 10: Commit native instrumentation**

```bash
git add src/index/hnsw_cpp_core.hpp src/index/hnsw_cpp_core.cpp src/index/hnsw_cpp.pyx tests/test_hnsw_cpp.py
git commit -m "feat: instrument hnsw native build counters"
```

---

### Task 3: Report Counters Through Benchmark Artifacts

**Files:**
- Modify: `benchmarks/benchmark.py`
- Modify: `tests/test_benchmark_cli.py`
- Test: `tests/test_benchmark_cli.py`

- [ ] **Step 1: Add failing benchmark schema expectations**

In `tests/test_benchmark_cli.py`, extend the expected
`set(metrics["cpp_build_stats"])` with:

```python
        "distance_evaluations",
        "search_distance_evaluations",
        "neighbor_selection_distance_evaluations",
        "prune_distance_evaluations",
        "visited_nodes",
        "max_visited_nodes_per_search",
        "candidate_heap_pushes",
        "result_heap_pushes",
        "neighbor_selection_calls",
        "selected_degree_total",
        "average_selected_degree",
        "max_selected_degree",
        "prune_calls",
        "prune_input_total",
        "average_prune_input_size",
        "max_prune_input_size",
```

After the existing `assert metrics["cpp_build_stats"]["dimensions"] == 8`, add:

```python
    if result["environment"]["cpp_available"]:
        stats = metrics["cpp_build_stats"]
        assert stats["distance_evaluations"] == (
            stats["search_distance_evaluations"]
            + stats["neighbor_selection_distance_evaluations"]
            + stats["prune_distance_evaluations"]
        )
        assert stats["visited_nodes"] == stats["search_distance_evaluations"]
        assert stats["max_visited_nodes_per_search"] > 0
        assert stats["candidate_heap_pushes"] >= stats["visited_nodes"]
        assert stats["result_heap_pushes"] >= stats["visited_nodes"]
        assert stats["neighbor_selection_calls"] > 0
        assert stats["average_selected_degree"] > 0
```

In `test_main_writes_json_and_markdown_reports()`, add:

```python
    assert "C++ Distance Evaluations" in markdown
    assert "C++ Visited Nodes" in markdown
    assert "C++ Average Selected Degree" in markdown
    assert "C++ Average Prune Input Size" in markdown
```

- [ ] **Step 2: Run benchmark CLI tests to verify they fail before reporting changes**

Run:

```bash
venv/bin/python -m pytest tests/test_benchmark_cli.py::test_run_benchmark_suite_returns_structured_metrics tests/test_benchmark_cli.py::test_main_writes_json_and_markdown_reports -q
```

Expected: FAIL because `benchmarks/benchmark.py` has not added the new keys or
Markdown rows yet.

- [ ] **Step 3: Add keys to benchmark normalization**

In `benchmarks/benchmark.py`, extend `CPP_BUILD_STATS_KEYS` after
`"max_observed_degree",` with:

```python
    "distance_evaluations",
    "search_distance_evaluations",
    "neighbor_selection_distance_evaluations",
    "prune_distance_evaluations",
    "visited_nodes",
    "max_visited_nodes_per_search",
    "candidate_heap_pushes",
    "result_heap_pushes",
    "neighbor_selection_calls",
    "selected_degree_total",
    "average_selected_degree",
    "max_selected_degree",
    "prune_calls",
    "prune_input_total",
    "average_prune_input_size",
    "max_prune_input_size",
```

- [ ] **Step 4: Add Markdown report rows**

In `format_markdown_report()` in `benchmarks/benchmark.py`, after the
`C++ Max Observed Degree` row, add:

```python
            f"| C++ Distance Evaluations | {cpp_build_stats['distance_evaluations']} |",
            f"| C++ Search Distance Evaluations | {cpp_build_stats['search_distance_evaluations']} |",
            f"| C++ Neighbor Selection Distance Evaluations | {cpp_build_stats['neighbor_selection_distance_evaluations']} |",
            f"| C++ Prune Distance Evaluations | {cpp_build_stats['prune_distance_evaluations']} |",
            f"| C++ Visited Nodes | {cpp_build_stats['visited_nodes']} |",
            f"| C++ Max Visited Nodes Per Search | {cpp_build_stats['max_visited_nodes_per_search']} |",
            f"| C++ Candidate Heap Pushes | {cpp_build_stats['candidate_heap_pushes']} |",
            f"| C++ Result Heap Pushes | {cpp_build_stats['result_heap_pushes']} |",
            f"| C++ Neighbor Selection Calls | {cpp_build_stats['neighbor_selection_calls']} |",
            f"| C++ Selected Degree Total | {cpp_build_stats['selected_degree_total']} |",
            f"| C++ Average Selected Degree | {cpp_build_stats['average_selected_degree']} |",
            f"| C++ Max Selected Degree | {cpp_build_stats['max_selected_degree']} |",
            f"| C++ Prune Calls | {cpp_build_stats['prune_calls']} |",
            f"| C++ Prune Input Total | {cpp_build_stats['prune_input_total']} |",
            f"| C++ Average Prune Input Size | {cpp_build_stats['average_prune_input_size']} |",
            f"| C++ Max Prune Input Size | {cpp_build_stats['max_prune_input_size']} |",
```

- [ ] **Step 5: Run benchmark CLI tests**

Run:

```bash
venv/bin/python -m pytest tests/test_benchmark_cli.py::test_run_benchmark_suite_returns_structured_metrics tests/test_benchmark_cli.py::test_main_writes_json_and_markdown_reports -q
```

Expected: PASS.

- [ ] **Step 6: Commit benchmark reporting**

```bash
git add benchmarks/benchmark.py tests/test_benchmark_cli.py
git commit -m "feat: report hnsw build instrumentation in benchmarks"
```

---

### Task 4: Record the First Instrumentation Baseline

**Files:**
- Modify: `TECHNICAL.md`
- Optional create: `benchmarks/results/hnsw-100k-build-instrumentation-baseline.json`
- Optional create: `benchmarks/results/hnsw-100k-build-instrumentation-baseline.md`
- Test: benchmark CLI smoke run

- [ ] **Step 1: Run a fast benchmark to validate counters on the current machine**

Run:

```bash
venv/bin/python benchmarks/benchmark.py --dataset random --size 10000 --dimension 128 --queries 100 --k 10 --ef-search 50 --output /tmp/hnsw-build-instrumentation.json --markdown-output /tmp/hnsw-build-instrumentation.md
```

Expected: command exits 0, prints `Results written to /tmp/hnsw-build-instrumentation.json`, and the Markdown report contains the new rows.

- [ ] **Step 2: Inspect the generated Markdown report**

Run:

```bash
sed -n '1,180p' /tmp/hnsw-build-instrumentation.md
```

Expected: the report includes rows for:

```text
C++ Distance Evaluations
C++ Search Distance Evaluations
C++ Neighbor Selection Distance Evaluations
C++ Prune Distance Evaluations
C++ Visited Nodes
C++ Average Selected Degree
C++ Average Prune Input Size
```

- [ ] **Step 3: Decide whether to save the artifact**

If the worktree is dirty from unrelated user changes or the run is only a local
smoke check, leave the artifact in `/tmp` and do not add it to git.

If this run is intended as a shared baseline, rerun the command with repo paths:

```bash
venv/bin/python benchmarks/benchmark.py --dataset random --size 10000 --dimension 128 --queries 100 --k 10 --ef-search 50 --output benchmarks/results/hnsw-100k-build-instrumentation-baseline.json --markdown-output benchmarks/results/hnsw-100k-build-instrumentation-baseline.md
```

Expected: the JSON and Markdown files are created under `benchmarks/results/`.

- [ ] **Step 4: Add a technical documentation section**

Generate the exact metric table from the benchmark JSON:

```bash
venv/bin/python -c 'import json; data=json.load(open("/tmp/hnsw-build-instrumentation.json")); stats=data["metrics"]["cpp_build_stats"]; metrics=data["metrics"]; rows=[("Build Time", f"{metrics[\"build_time_seconds\"]:.4f}s"), ("C++ Distance Evaluations", stats["distance_evaluations"]), ("C++ Search Distance Evaluations", stats["search_distance_evaluations"]), ("C++ Neighbor Selection Distance Evaluations", stats["neighbor_selection_distance_evaluations"]), ("C++ Prune Distance Evaluations", stats["prune_distance_evaluations"]), ("C++ Visited Nodes", stats["visited_nodes"]), ("C++ Average Selected Degree", stats["average_selected_degree"]), ("C++ Average Prune Input Size", stats["average_prune_input_size"])]; print("| Metric | Value |"); print("|---|---:|"); [print(f"| {name} | {value} |") for name, value in rows]'
```

Expected: stdout is a Markdown table with concrete numeric values.

In `TECHNICAL.md`, add this section under the existing HNSW performance or C++
search-core discussion. Paste the generated metric table in place of the sample
table shown here:

```markdown
### Build-Time Instrumentation Baseline

Before this step, the native HNSW builder reported broad timing buckets such as
construction search, pruning, and CSR export. That showed candidate search was
the largest phase, but it did not explain whether time was driven by distance
evaluations, visited nodes, heap operations, neighbor selection, or pruning
input sizes.

This step adds detailed native build counters for distance evaluations, visited
nodes, heap pushes, neighbor-selection calls, selected degree, prune calls, and
prune input size. The goal is to make the next optimization educational and
measurable: instead of guessing at the bottleneck, each follow-up change can
show which counter moved and whether recall stayed stable.

On the fast 10k synthetic smoke benchmark, the generated values were:

| Metric | Value |
|---|---:|
| Build Time | 3.1353s |
| C++ Distance Evaluations | 0 |
| C++ Search Distance Evaluations | 0 |
| C++ Neighbor Selection Distance Evaluations | 0 |
| C++ Prune Distance Evaluations | 0 |
| C++ Visited Nodes | 0 |
| C++ Average Selected Degree | 0.0 |
| C++ Average Prune Input Size | 0.0 |

The next implementation step should target the largest measured source of
distance work while keeping the SIFT1M 100k recall target intact.
```

The numbers in the sample block are only there to show the exact table shape.
Replace the whole table with the command output from this task before committing.

- [ ] **Step 5: Run focused tests after the documentation update**

Run:

```bash
venv/bin/python -m pytest tests/test_hnsw_cpp.py::test_cpp_build_graph_reports_detailed_build_counters tests/test_benchmark_cli.py::test_run_benchmark_suite_returns_structured_metrics tests/test_benchmark_cli.py::test_main_writes_json_and_markdown_reports -q
```

Expected: PASS.

- [ ] **Step 6: Commit the baseline documentation**

If no benchmark artifact was saved:

```bash
git add TECHNICAL.md
git commit -m "docs: record hnsw build instrumentation baseline"
```

If benchmark artifacts were saved:

```bash
git add TECHNICAL.md benchmarks/results/hnsw-100k-build-instrumentation-baseline.json benchmarks/results/hnsw-100k-build-instrumentation-baseline.md
git commit -m "docs: record hnsw build instrumentation baseline"
```

---

### Task 5: Final Verification for the Instrumentation Milestone

**Files:**
- Verify: `src/index/hnsw_cpp_core.hpp`
- Verify: `src/index/hnsw_cpp_core.cpp`
- Verify: `src/index/hnsw_cpp.pyx`
- Verify: `benchmarks/benchmark.py`
- Verify: `tests/test_hnsw_cpp.py`
- Verify: `tests/test_benchmark_cli.py`
- Verify: `TECHNICAL.md`

- [ ] **Step 1: Rebuild native extensions from source**

Run:

```bash
venv/bin/python setup.py build_ext --inplace
```

Expected: build completes successfully.

- [ ] **Step 2: Run focused test suite**

Run:

```bash
venv/bin/python -m pytest tests/test_hnsw_cpp.py tests/test_benchmark_cli.py -q
```

Expected: PASS.

- [ ] **Step 3: Run benchmark smoke check**

Run:

```bash
venv/bin/python benchmarks/benchmark.py --dataset random --size 1000 --dimension 32 --queries 20 --k 5 --ef-search 20 --output /tmp/hnsw-build-final-smoke.json --markdown-output /tmp/hnsw-build-final-smoke.md
```

Expected: command exits 0 and the Markdown report contains `C++ Distance Evaluations`.

- [ ] **Step 4: Check git status**

Run:

```bash
git status --short
```

Expected: only intentional files are modified. Pre-existing unrelated worktree
changes may still be present; do not revert them.

- [ ] **Step 5: Commit any final verification-only documentation corrections**

If verification revealed a documentation typo or benchmark-label fix:

```bash
git add TECHNICAL.md benchmarks/benchmark.py tests/test_benchmark_cli.py
git commit -m "docs: polish hnsw instrumentation reporting"
```

If no corrections were needed, skip this commit.

---

## Self-Review Notes

- Spec coverage: This plan implements the approved first milestone only:
  instrumentation, reporting, benchmark smoke run, and documentation. It does
  not implement the later hot-path optimization, build-quality profiles, or
  parallelism.
- Placeholder scan: No open-ended implementation markers remain. Benchmark
  documentation values are produced by the explicit JSON-to-Markdown command in
  Task 4 before commit.
- Type consistency: Native `BuildStats`, Cython `CppBuildStats`, benchmark
  `CPP_BUILD_STATS_KEYS`, and tests all use the same field names.
