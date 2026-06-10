"""
Compare two benchmark JSON reports.

The comparison is intentionally small and explicit: it reads two reports emitted
by benchmarks/benchmark.py, computes deltas for the key metrics, and marks each
metric as improved, regressed, or unchanged based on whether higher or lower is
better for that metric.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

METRIC_SPECS = {
    "build_time_seconds": {"higher_is_better": False},
    "vectors_per_second": {"higher_is_better": True},
    "cpp_build_stats.total_seconds": {"higher_is_better": False},
    "cpp_build_stats.construction_seconds": {"higher_is_better": False},
    "cpp_build_stats.search_seconds": {"higher_is_better": False},
    "cpp_build_stats.greedy_search_seconds": {"higher_is_better": False},
    "cpp_build_stats.candidate_search_seconds": {"higher_is_better": False},
    "cpp_build_stats.prune_seconds": {"higher_is_better": False},
    "cpp_build_stats.csr_export_seconds": {"higher_is_better": False},
    "cpp_build_stats.visited_resizes": {"higher_is_better": False},
    "cpp_build_stats.search_heap_resizes": {"higher_is_better": False},
    "qps": {"higher_is_better": True},
    "latency_ms.average": {"higher_is_better": False},
    "latency_ms.p50": {"higher_is_better": False},
    "latency_ms.p95": {"higher_is_better": False},
    "latency_ms.p99": {"higher_is_better": False},
    "recall_at_k": {"higher_is_better": True},
    "memory.graph.total_graph_mb": {"higher_is_better": False},
    "memory.process_peak_rss_mb": {"higher_is_better": False},
    "persistence.compact.save_time_seconds": {"higher_is_better": False},
    "persistence.compact.load_time_seconds": {"higher_is_better": False},
    "persistence.compact.file_size_mb": {"higher_is_better": False},
    "persistence.compact.process_peak_rss_mb": {"higher_is_better": False},
    "persistence.compact.loaded_graph.total_graph_mb": {"higher_is_better": False},
    "persistence.materialized.save_time_seconds": {"higher_is_better": False},
    "persistence.materialized.load_time_seconds": {"higher_is_better": False},
    "persistence.materialized.file_size_mb": {"higher_is_better": False},
    "persistence.materialized.process_peak_rss_mb": {"higher_is_better": False},
    "persistence.materialized.loaded_graph.total_graph_mb": {"higher_is_better": False},
}


def load_result(path: str | Path) -> dict[str, Any]:
    """Load a benchmark JSON report from disk."""
    result = json.loads(Path(path).read_text())
    schema_version = result.get("schema_version")
    if schema_version != "benchmark.v1":
        raise ValueError(f"Unsupported benchmark schema: {schema_version!r}")
    return result


def _nested_get(data: dict[str, Any], dotted_path: str) -> float | None:
    current: Any = data
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]

    if current is None:
        return None
    return float(current)


def _status(delta: float, higher_is_better: bool) -> str:
    if abs(delta) < 1e-12:
        return "unchanged"

    if higher_is_better:
        return "improved" if delta > 0 else "regressed"

    return "improved" if delta < 0 else "regressed"


def _compare_metric(
    baseline_metrics: dict[str, Any],
    candidate_metrics: dict[str, Any],
    name: str,
    higher_is_better: bool,
) -> dict[str, Any] | None:
    baseline = _nested_get(baseline_metrics, name)
    candidate = _nested_get(candidate_metrics, name)
    if baseline is None or candidate is None:
        return None

    delta = candidate - baseline
    percent_delta = None if baseline == 0 else (delta / baseline) * 100

    return {
        "baseline": round(baseline, 6),
        "candidate": round(candidate, 6),
        "delta": round(delta, 6),
        "percent_delta": None if percent_delta is None else round(percent_delta, 6),
        "higher_is_better": higher_is_better,
        "status": _status(delta, higher_is_better),
    }


def compare_results(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Compare two loaded benchmark results."""
    baseline_metrics = baseline["metrics"]
    candidate_metrics = candidate["metrics"]

    metrics = {}
    for name, spec in METRIC_SPECS.items():
        comparison = _compare_metric(
            baseline_metrics,
            candidate_metrics,
            name,
            higher_is_better=spec["higher_is_better"],
        )
        if comparison is not None:
            metrics[name] = comparison

    return {
        "schema_version": "benchmark-comparison.v1",
        "baseline": {
            "commit": baseline.get("environment", {}).get("git_commit", "unknown"),
            "dataset": baseline.get("dataset", {}),
            "config": baseline.get("config", {}),
        },
        "candidate": {
            "commit": candidate.get("environment", {}).get("git_commit", "unknown"),
            "dataset": candidate.get("dataset", {}),
            "config": candidate.get("config", {}),
        },
        "metrics": metrics,
    }


def format_markdown_comparison(comparison: dict[str, Any]) -> str:
    """Render a comparison as a Markdown table."""
    lines = [
        "# Benchmark Comparison",
        "",
        "## Context",
        "",
        f"- Baseline commit: `{comparison['baseline']['commit']}`",
        f"- Candidate commit: `{comparison['candidate']['commit']}`",
        "",
        "## Metric Deltas",
        "",
        "| Metric | Baseline | Candidate | Delta | Delta % | Status |",
        "|---|---:|---:|---:|---:|---|",
    ]

    for name, metric in comparison["metrics"].items():
        percent = metric["percent_delta"]
        percent_text = "n/a" if percent is None else f"{percent:.2f}%"
        lines.append(
            "| "
            f"{name} | "
            f"{metric['baseline']:.6g} | "
            f"{metric['candidate']:.6g} | "
            f"{metric['delta']:.6g} | "
            f"{percent_text} | "
            f"{metric['status']} |"
        )

    lines.append("")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare two vector DB benchmark reports")
    parser.add_argument("baseline", help="Baseline benchmark JSON path")
    parser.add_argument("candidate", help="Candidate benchmark JSON path")
    parser.add_argument("--output", default=None, help="Optional Markdown output path")
    parser.add_argument("--json-output", default=None, help="Optional structured JSON output path")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    comparison = compare_results(load_result(args.baseline), load_result(args.candidate))
    markdown = format_markdown_comparison(comparison)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown)
    else:
        print(markdown)

    if args.json_output:
        json_path = Path(args.json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(comparison, indent=2) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
