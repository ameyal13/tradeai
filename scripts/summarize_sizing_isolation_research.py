"""Summarize Sizing Isolation Research v1 results.

Offline research only. This script reads the sizing_isolation_v1 registry and
compares matched configs against focused_v2A to determine whether changing only
ATR/RR/horizon improved validation PF and drawdown.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.compare_research_phases import build_comparison, load_phase  # noqa: E402
from scripts.summarize_research_registry import summarize_registry  # noqa: E402


DEFAULT_SIZING_REGISTRY = PROJECT_ROOT / "reports" / "research_daemon" / "sizing_isolation_v1_registry.jsonl"
DEFAULT_BASELINE_REGISTRY = PROJECT_ROOT / "reports" / "research_daemon" / "focused_v2a_registry.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "research_daemon"


def _conclusion_from_summary(summary: dict[str, Any], comparison: dict[str, Any]) -> list[str]:
    overview = summary.get("summary") or {}
    stability = summary.get("stability_analysis") or {}
    comparison_overview = comparison.get("summary") or {}
    verdicts = comparison_overview.get("verdict_counts") or {}
    watchlist = int(overview.get("unstable_watchlist", 0) or 0)
    stable = int(overview.get("stable_research_candidate", 0) or 0)
    median_pf = stability.get("median_validation_pf")
    median_avg = stability.get("median_validation_avg_return")
    median_dd = stability.get("median_worst_drawdown")
    improved = int(verdicts.get("improved_validation", 0) or 0)
    worse = int(verdicts.get("worse_validation", 0) or 0)
    conclusions = ["Research only. No trading signal."]
    if stable > 0:
        conclusions.append("Sizing/exit variants produced stable candidates; they still require shadow validation.")
    elif watchlist > 0:
        conclusions.append("Sizing/exit variants produced watchlist configs, but no stable candidates.")
    else:
        conclusions.append("Sizing/exit variants did not produce watchlist or stable candidates yet.")
    if improved > worse:
        conclusions.append("Against focused_v2A matched configs, sizing changes improved validation more often than they worsened it.")
    elif worse > improved:
        conclusions.append("Against focused_v2A matched configs, sizing changes worsened validation more often than they improved it.")
    else:
        conclusions.append("Against focused_v2A matched configs, sizing changes were mixed or inconclusive.")
    conclusions.append(
        f"Aggregate sizing median validation PF={median_pf}, avg_return={median_avg}, median_worst_drawdown={median_dd}."
    )
    if watchlist == 0 and stable == 0 and (median_pf is None or median_pf < 1):
        conclusions.append("Current evidence points more toward direction/features than sizing alone.")
    else:
        conclusions.append("If drawdown improves while PF stays positive, sizing/exit remains a live hypothesis.")
    conclusions.append("Validation is the selector; test metrics are diagnostic only.")
    return conclusions


def summarize_sizing_isolation_registry(
    registry_path: str | Path = DEFAULT_SIZING_REGISTRY,
    baseline_registry: str | Path = DEFAULT_BASELINE_REGISTRY,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    top_limit: int = 15,
) -> dict[str, Any]:
    summary = summarize_registry(
        registry_path=registry_path,
        output_dir=output_dir,
        filename_prefix="sizing_isolation_v1_global_summary",
        top_limit=top_limit,
        refined=True,
    )
    comparison = build_comparison(
        load_phase(baseline_registry),
        load_phase(registry_path),
        baseline_name="focused_v2a",
        candidate_name="sizing_isolation_v1",
    )
    summary["sizing_baseline_comparison"] = {
        "summary": comparison["summary"],
        "top_improvements": comparison["top_improvements"],
        "top_regressions": comparison["top_regressions"],
        "conclusion": comparison["conclusion"],
    }
    summary["sizing_conclusion"] = _conclusion_from_summary(summary, comparison)
    json_path = Path(summary["json_path"])
    markdown_path = Path(summary["markdown_path"])
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    markdown = markdown_path.read_text(encoding="utf-8")
    markdown += "\n## Sizing Isolation Conclusion\n\n"
    markdown += "\n".join(f"- {item}" for item in summary["sizing_conclusion"]) + "\n"
    markdown += "\n## Focused v2A Baseline Comparison\n\n"
    comparison_summary = comparison["summary"]
    markdown += f"- matched configs: `{comparison_summary.get('matched_configs')}`\n"
    markdown += f"- verdict counts: `{comparison_summary.get('verdict_counts')}`\n"
    markdown += f"- mean delta validation PF: `{comparison_summary.get('mean_delta_validation_pf')}`\n"
    markdown += f"- mean delta validation avg return: `{comparison_summary.get('mean_delta_validation_avg_return')}`\n"
    markdown += f"- mean delta worst drawdown: `{comparison_summary.get('mean_delta_worst_drawdown')}`\n"
    markdown += "\n### Top Sizing Improvements vs focused_v2A\n\n"
    for row in comparison["top_improvements"][:top_limit]:
        markdown += (
            f"- `{row.get('label')}` dPF `{row.get('delta_validation_pf')}`, "
            f"dAvg `{row.get('delta_validation_avg_return')}`, dDD `{row.get('delta_worst_drawdown')}`\n"
        )
    markdown += "\n### Top Sizing Regressions vs focused_v2A\n\n"
    for row in comparison["top_regressions"][:top_limit]:
        markdown += (
            f"- `{row.get('label')}` dPF `{row.get('delta_validation_pf')}`, "
            f"dAvg `{row.get('delta_validation_avg_return')}`, dDD `{row.get('delta_worst_drawdown')}`\n"
        )
    markdown_path.write_text(markdown, encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize Sizing Isolation Research v1.")
    parser.add_argument("--registry", default=str(DEFAULT_SIZING_REGISTRY))
    parser.add_argument("--baseline-registry", default=str(DEFAULT_BASELINE_REGISTRY))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--top-limit", type=int, default=15)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = summarize_sizing_isolation_registry(
        registry_path=args.registry,
        baseline_registry=args.baseline_registry,
        output_dir=args.output_dir,
        top_limit=args.top_limit,
    )
    overview = summary["summary"]
    comparison = summary["sizing_baseline_comparison"]["summary"]
    print("Sizing Isolation Research v1 global summary generated")
    print("Research only. No trading signal.")
    print(f"total_configs: {overview['unique_configs']}")
    print(f"completed: {overview['unique_completed_configs']}")
    print(f"classification_counts: {overview['classification_counts']}")
    print(f"matched_to_focused_v2a: {comparison['matched_configs']}")
    print(f"comparison_verdict_counts: {comparison['verdict_counts']}")
    print(f"json: {summary['json_path']}")
    print(f"markdown: {summary['markdown_path']}")


if __name__ == "__main__":
    main()
