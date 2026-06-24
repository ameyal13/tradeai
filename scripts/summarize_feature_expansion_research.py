"""Summarize Feature Expansion Grid v1 research results.

Offline research only. This script reads the feature_expansion_v1 registry and
answers whether opt-in feature families beat the time_only PF 0.81 reference
and the real PF > 1.0 objective.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.feature_expansion_grid import OBJECTIVE_PROFIT_FACTOR, TIME_ONLY_BASELINE_PF  # noqa: E402
from scripts.run_feature_expansion_research import FEATURE_EXPANSION_REGISTRY_PATH  # noqa: E402
from scripts.summarize_research_registry import enrich_registry_records, load_latest_registry_records  # noqa: E402


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "research_daemon"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _metric(row: dict[str, Any], name: str) -> float | None:
    return _float((row.get("aggregate") or {}).get(name))


def _classification(row: dict[str, Any]) -> str:
    if row.get("status") == "failed":
        return "failed"
    return str(row.get("classification") or row.get("status") or "unknown")


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2:
        return round(values[mid], 6)
    return round((values[mid - 1] + values[mid]) / 2, 6)


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def _counts(rows: list[dict[str, Any]], key: str | None = None) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        value = _classification(row) if key is None else str((row.get("config") or {}).get(key))
        out[value] = out.get(value, 0) + 1
    return out


def _row_label(row: dict[str, Any]) -> str:
    config = row.get("config") or {}
    return (
        f"{config.get('symbol')} {config.get('timeframe')} {config.get('feature_set')} "
        f"h{config.get('horizon_candles')} RR{config.get('risk_reward')} "
        f"ATR{config.get('atr_stop_multiplier')} {config.get('cost_mode')}"
    )


def _row_summary(row: dict[str, Any]) -> dict[str, Any]:
    aggregate = row.get("aggregate") or {}
    return {
        "config_id": row.get("config_id"),
        "label": _row_label(row),
        "classification": _classification(row),
        "config": row.get("config") or {},
        "median_validation_pf": aggregate.get("median_validation_pf"),
        "median_validation_avg_return": aggregate.get("median_validation_avg_return"),
        "beats_time_only_rate": aggregate.get("beats_time_only_rate"),
        "beats_dummy_random_rate": aggregate.get("beats_dummy_random_rate"),
        "objective_pf_rate": aggregate.get("objective_pf_rate"),
        "time_only_median_pf": aggregate.get("time_only_median_pf"),
        "dummy_random_median_pf": aggregate.get("dummy_random_median_pf"),
        "valid_windows": aggregate.get("valid_windows"),
        "json_path": row.get("json_path"),
        "json_loaded": row.get("json_loaded"),
        "json_missing": row.get("json_missing"),
    }


def _top(rows: list[dict[str, Any]], metric: str, limit: int) -> list[dict[str, Any]]:
    valid = [row for row in rows if _metric(row, metric) is not None]
    return [_row_summary(row) for row in sorted(valid, key=lambda row: _metric(row, metric) or float("-inf"), reverse=True)[:limit]]


def group_records(rows: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str((row.get("config") or {}).get(field))].append(row)
    out: dict[str, dict[str, Any]] = {}
    for key, items in sorted(groups.items()):
        pf_values = [_metric(row, "median_validation_pf") for row in items]
        avg_values = [_metric(row, "median_validation_avg_return") for row in items]
        beats_time = [_metric(row, "beats_time_only_rate") for row in items]
        objective = [_metric(row, "objective_pf_rate") for row in items]
        out[key] = {
            "count": len(items),
            "classification_counts": _counts(items),
            "median_validation_pf": _median([value for value in pf_values if value is not None]),
            "median_validation_avg_return": _median([value for value in avg_values if value is not None]),
            "avg_beats_time_only_rate": _mean([value for value in beats_time if value is not None]),
            "avg_objective_pf_rate": _mean([value for value in objective if value is not None]),
        }
    return out


def automatic_conclusion(summary: dict[str, Any]) -> list[str]:
    counts = summary["summary"]["classification_counts"]
    feature_groups = summary["groupings"]["feature_set"]
    best_feature = None
    if feature_groups:
        best_feature = sorted(
            feature_groups.items(),
            key=lambda item: item[1].get("median_validation_pf") or float("-inf"),
            reverse=True,
        )[0]
    conclusions = ["Research only. No trading signal."]
    if counts.get("feature_expansion_candidate", 0) > 0:
        conclusions.append("At least one feature expansion crossed PF > 1.0 with positive validation evidence; it still needs further validation.")
    elif counts.get("feature_expansion_watchlist", 0) > 0:
        conclusions.append("Some feature expansions beat the time_only reference, but none reached robust candidate status.")
    else:
        conclusions.append("No feature expansion beat the reference strongly enough; current evidence still points to weak/noisy features.")
    if best_feature:
        conclusions.append(
            f"Best median validation PF group: {best_feature[0]} with PF {best_feature[1].get('median_validation_pf')}."
        )
    funding_pf = feature_groups.get("baseline_plus_funding", {}).get("median_validation_pf")
    mtf_pf = feature_groups.get("baseline_plus_4h", {}).get("median_validation_pf")
    all_new_pf = feature_groups.get("baseline_plus_all_new", {}).get("median_validation_pf")
    conclusions.append(f"Funding median PF: {funding_pf}; 4h median PF: {mtf_pf}; combined new-feature median PF: {all_new_pf}.")
    conclusions.append(
        f"Minimum reference is time_only PF {TIME_ONLY_BASELINE_PF}; real objective remains PF > {OBJECTIVE_PROFIT_FACTOR}."
    )
    return conclusions


def build_summary(records: list[dict[str, Any]], top_limit: int = 15) -> dict[str, Any]:
    completed = [row for row in records if row.get("status") == "completed"]
    summary = {
        "generated_at": utc_now(),
        "report_type": "feature_expansion_v1_global_summary",
        "guardrails": {
            "research_only": True,
            "no_trading_signal": True,
            "no_supabase_writes": True,
            "test_not_used_for_selection": True,
            "accuracy_not_used": True,
        },
        "benchmarks": {
            "time_only_reference_pf": TIME_ONLY_BASELINE_PF,
            "objective_profit_factor": OBJECTIVE_PROFIT_FACTOR,
        },
        "summary": {
            "unique_configs": len(records),
            "completed": len(completed),
            "classification_counts": _counts(records),
            "json_loaded": sum(1 for row in records if row.get("json_loaded")),
            "json_missing": sum(1 for row in records if row.get("json_missing")),
        },
        "top": {
            "median_validation_pf": _top(records, "median_validation_pf", top_limit),
            "median_validation_avg_return": _top(records, "median_validation_avg_return", top_limit),
            "beats_time_only_rate": _top(records, "beats_time_only_rate", top_limit),
            "objective_pf_rate": _top(records, "objective_pf_rate", top_limit),
        },
        "groupings": {
            "symbol": group_records(records, "symbol"),
            "feature_set": group_records(records, "feature_set"),
        },
        "records": [_row_summary(row) for row in records],
    }
    summary["conclusion"] = automatic_conclusion(summary)
    return summary


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Feature Expansion v1 Global Summary",
        "",
        f"Generated at: `{summary['generated_at']}`",
        "",
        "## Guardrails",
        "",
        "- Research only. No trading signal.",
        "- No live/shadow feature changes.",
        "- No Supabase writes.",
        "- Validation metrics decide; test/accuracy are not used.",
        "",
        "## Summary",
        "",
        f"- unique configs: `{summary['summary']['unique_configs']}`",
        f"- completed: `{summary['summary']['completed']}`",
        f"- classification counts: `{summary['summary']['classification_counts']}`",
        f"- time_only reference PF: `{TIME_ONLY_BASELINE_PF}`",
        f"- real objective PF: `>{OBJECTIVE_PROFIT_FACTOR}`",
        "",
        "## Conclusion",
        "",
    ]
    lines.extend(f"- {item}" for item in summary.get("conclusion", []))

    def row_lines(title: str, rows: list[dict[str, Any]]) -> None:
        lines.extend(["", f"## {title}", ""])
        if not rows:
            lines.append("None.")
            return
        for row in rows:
            lines.append(
                f"- `{row['classification']}: {row['label']}` PF `{row['median_validation_pf']}`, "
                f"avg `{row['median_validation_avg_return']}`, beats_time `{row['beats_time_only_rate']}`, "
                f"objective_rate `{row['objective_pf_rate']}`"
            )

    row_lines("Top By Median Validation PF", summary["top"]["median_validation_pf"])
    row_lines("Top By Median Validation Avg Return", summary["top"]["median_validation_avg_return"])
    row_lines("Top By Beats Time Only Rate", summary["top"]["beats_time_only_rate"])
    row_lines("Top By Objective PF Rate", summary["top"]["objective_pf_rate"])

    def group_table(title: str, groups: dict[str, dict[str, Any]]) -> None:
        lines.extend(["", f"## {title}", ""])
        lines.append("| Group | Count | Classes | Median PF | Median Avg | Avg Beats Time | Avg Objective PF Rate |")
        lines.append("| --- | ---: | --- | ---: | ---: | ---: | ---: |")
        for key, row in groups.items():
            lines.append(
                f"| {key} | {row['count']} | `{row['classification_counts']}` | "
                f"{row['median_validation_pf']} | {row['median_validation_avg_return']} | "
                f"{row['avg_beats_time_only_rate']} | {row['avg_objective_pf_rate']} |"
            )

    group_table("By Feature Set", summary["groupings"]["feature_set"])
    group_table("By Symbol", summary["groupings"]["symbol"])
    return "\n".join(lines).rstrip() + "\n"


def summarize_feature_expansion_registry(
    registry_path: str | Path = FEATURE_EXPANSION_REGISTRY_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    top_limit: int = 15,
) -> dict[str, Any]:
    records = enrich_registry_records(load_latest_registry_records(registry_path))
    summary = build_summary(records, top_limit=top_limit)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    stamp = utc_stamp()
    json_path = target / f"feature_expansion_v1_global_summary_{stamp}.json"
    markdown_path = target / f"feature_expansion_v1_global_summary_{stamp}.md"
    summary["json_path"] = str(json_path)
    summary["markdown_path"] = str(markdown_path)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(render_markdown(summary), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize Feature Expansion Grid v1.")
    parser.add_argument("--registry", default=str(FEATURE_EXPANSION_REGISTRY_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--top-limit", type=int, default=15)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = summarize_feature_expansion_registry(args.registry, args.output_dir, args.top_limit)
    print("Feature Expansion v1 global summary generated")
    print("Research only. No trading signal.")
    print(f"total_configs: {summary['summary']['unique_configs']}")
    print(f"completed: {summary['summary']['completed']}")
    print(f"classification_counts: {summary['summary']['classification_counts']}")
    print(f"json: {summary['json_path']}")
    print(f"markdown: {summary['markdown_path']}")


if __name__ == "__main__":
    main()
