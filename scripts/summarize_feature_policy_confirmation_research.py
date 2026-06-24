"""Summarize Feature Policy Confirmation v1 results."""
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

from scripts.run_feature_policy_confirmation_research import FEATURE_POLICY_REGISTRY_PATH  # noqa: E402
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


def _counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        value = _classification(row)
        out[value] = out.get(value, 0) + 1
    return out


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return round(ordered[mid], 6)
    return round((ordered[mid - 1] + ordered[mid]) / 2, 6)


def _row_summary(row: dict[str, Any]) -> dict[str, Any]:
    config = row.get("config") or {}
    aggregate = row.get("aggregate") or {}
    return {
        "config_id": row.get("config_id"),
        "symbol": config.get("symbol"),
        "feature_set": config.get("feature_set"),
        "classification": _classification(row),
        "median_validation_pf": aggregate.get("median_validation_pf"),
        "median_validation_avg_return": aggregate.get("median_validation_avg_return"),
        "beats_time_only_rate": aggregate.get("beats_time_only_rate"),
        "beats_dummy_random_rate": aggregate.get("beats_dummy_random_rate"),
        "valid_windows": aggregate.get("valid_windows"),
    }


def group_by_symbol(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[str((row.get("config") or {}).get("symbol"))].append(row)
    out: dict[str, dict[str, Any]] = {}
    for symbol, rows in sorted(grouped.items()):
        summaries = [_row_summary(row) for row in rows]
        best = sorted(
            summaries,
            key=lambda row: _float(row.get("median_validation_pf")) or float("-inf"),
            reverse=True,
        )[0] if summaries else None
        out[symbol] = {
            "configs": summaries,
            "best_by_validation_pf": best,
            "classification_counts": _counts(rows),
        }
    return out


def control_comparison(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Compare candidate feature policies against baseline and time_only controls."""
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in records:
        config = row.get("config") or {}
        grouped[str(config.get("symbol"))][str(config.get("feature_set"))] = row

    output: dict[str, dict[str, Any]] = {}
    for symbol, feature_rows in sorted(grouped.items()):
        baseline = feature_rows.get("baseline")
        time_only = feature_rows.get("time_only")
        candidates = {
            feature_set: row
            for feature_set, row in feature_rows.items()
            if feature_set not in {"baseline", "time_only"}
        }
        best_candidate_key = None
        best_candidate_row = None
        for feature_set, row in candidates.items():
            if best_candidate_row is None:
                best_candidate_key = feature_set
                best_candidate_row = row
                continue
            if (_metric(row, "median_validation_pf") or float("-inf")) > (
                _metric(best_candidate_row, "median_validation_pf") or float("-inf")
            ):
                best_candidate_key = feature_set
                best_candidate_row = row

        baseline_pf = _metric(baseline or {}, "median_validation_pf")
        time_only_pf = _metric(time_only or {}, "median_validation_pf")
        candidate_pf = _metric(best_candidate_row or {}, "median_validation_pf")
        baseline_avg = _metric(baseline or {}, "median_validation_avg_return")
        time_only_avg = _metric(time_only or {}, "median_validation_avg_return")
        candidate_avg = _metric(best_candidate_row or {}, "median_validation_avg_return")

        beats_baseline = candidate_pf is not None and baseline_pf is not None and candidate_pf > baseline_pf
        beats_time_only = candidate_pf is not None and time_only_pf is not None and candidate_pf > time_only_pf
        beats_controls = bool(beats_baseline and beats_time_only)
        if baseline is None or time_only is None or best_candidate_row is None:
            recommendation = "incomplete_controls"
        elif beats_controls and candidate_pf is not None and candidate_pf > 1.0:
            recommendation = "candidate_feature_policy_warrants_next_validation"
        elif best_candidate_key == "time_only":
            recommendation = "time_only_control_wins"
        elif not beats_baseline and not beats_time_only:
            recommendation = "candidate_does_not_beat_controls"
        else:
            recommendation = "mixed_control_result"

        output[symbol] = {
            "baseline": _row_summary(baseline) if baseline else None,
            "time_only": _row_summary(time_only) if time_only else None,
            "best_candidate_feature_set": best_candidate_key,
            "best_candidate": _row_summary(best_candidate_row) if best_candidate_row else None,
            "baseline_pf": baseline_pf,
            "time_only_pf": time_only_pf,
            "best_candidate_pf": candidate_pf,
            "pf_delta_vs_baseline": round(candidate_pf - baseline_pf, 6) if candidate_pf is not None and baseline_pf is not None else None,
            "pf_delta_vs_time_only": round(candidate_pf - time_only_pf, 6) if candidate_pf is not None and time_only_pf is not None else None,
            "baseline_avg_return": baseline_avg,
            "time_only_avg_return": time_only_avg,
            "best_candidate_avg_return": candidate_avg,
            "avg_delta_vs_baseline": round(candidate_avg - baseline_avg, 6) if candidate_avg is not None and baseline_avg is not None else None,
            "avg_delta_vs_time_only": round(candidate_avg - time_only_avg, 6) if candidate_avg is not None and time_only_avg is not None else None,
            "beats_baseline_pf": beats_baseline,
            "beats_time_only_pf": beats_time_only,
            "beats_controls_pf": beats_controls,
            "recommendation": recommendation,
        }
    return output


def group_by_feature_set(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[str((row.get("config") or {}).get("feature_set"))].append(row)
    out: dict[str, dict[str, Any]] = {}
    for feature_set, rows in sorted(grouped.items()):
        pf = [_metric(row, "median_validation_pf") for row in rows]
        avg = [_metric(row, "median_validation_avg_return") for row in rows]
        out[feature_set] = {
            "count": len(rows),
            "classification_counts": _counts(rows),
            "median_validation_pf": _median([value for value in pf if value is not None]),
            "median_validation_avg_return": _median([value for value in avg if value is not None]),
        }
    return out


def automatic_conclusion(summary: dict[str, Any]) -> list[str]:
    conclusions = ["Research only. No trading signal."]
    recommendations = {
        symbol: row.get("recommendation")
        for symbol, row in summary["control_comparison"].items()
    }
    winners = {
        symbol: row.get("best_candidate_feature_set")
        for symbol, row in summary["control_comparison"].items()
    }
    conclusions.append(f"Best candidate feature set by symbol: {winners}.")
    conclusions.append(f"Control-gated recommendations: {recommendations}.")
    passed = [
        symbol
        for symbol, row in summary["control_comparison"].items()
        if row.get("beats_controls_pf")
    ]
    if passed:
        conclusions.append(f"Feature candidates beat both baseline and time_only in validation for: {', '.join(passed)}.")
    else:
        conclusions.append("No candidate feature policy beat both baseline and time_only controls.")
    conclusions.append("Do not promote any feature policy unless it beats both baseline and time_only in validation.")
    return conclusions


def build_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in records if row.get("status") == "completed"]
    summary = {
        "generated_at": utc_now(),
        "report_type": "feature_policy_confirmation_v1_global_summary",
        "guardrails": {
            "research_only": True,
            "no_trading_signal": True,
            "no_supabase_writes": True,
            "test_not_used_for_selection": True,
            "accuracy_not_used": True,
        },
        "summary": {
            "unique_configs": len(records),
            "completed": len(completed),
            "classification_counts": _counts(records),
            "json_loaded": sum(1 for row in records if row.get("json_loaded")),
            "json_missing": sum(1 for row in records if row.get("json_missing")),
        },
        "by_symbol": group_by_symbol(records),
        "by_feature_set": group_by_feature_set(records),
        "records": [_row_summary(row) for row in records],
    }
    summary["control_comparison"] = control_comparison(records)
    summary["conclusion"] = automatic_conclusion(summary)
    return summary


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Feature Policy Confirmation v1 Global Summary",
        "",
        f"Generated at: `{summary['generated_at']}`",
        "",
        "## Guardrails",
        "",
        "- Research only. No trading signal.",
        "- No live/shadow feature changes.",
        "- Baseline/time_only controls must be beaten before promotion.",
        "",
        "## Summary",
        "",
        f"- unique configs: `{summary['summary']['unique_configs']}`",
        f"- completed: `{summary['summary']['completed']}`",
        f"- classification counts: `{summary['summary']['classification_counts']}`",
        "",
        "## Conclusion",
        "",
    ]
    lines.extend(f"- {item}" for item in summary.get("conclusion", []))
    lines.extend(["", "## By Symbol", ""])
    lines.append("| Symbol | Best Feature Set | Best PF | Best Avg | Classes |")
    lines.append("| --- | --- | ---: | ---: | --- |")
    for symbol, row in summary["by_symbol"].items():
        best = row.get("best_by_validation_pf") or {}
        lines.append(
            f"| {symbol} | {best.get('feature_set')} | {best.get('median_validation_pf')} | "
            f"{best.get('median_validation_avg_return')} | `{row.get('classification_counts')}` |"
        )
    lines.extend(["", "## Control Comparison", ""])
    lines.append("| Symbol | Baseline PF | Time PF | Best Candidate | Candidate PF | Delta vs Baseline | Delta vs Time | Beats Controls | Recommendation |")
    lines.append("| --- | ---: | ---: | --- | ---: | ---: | ---: | --- | --- |")
    for symbol, row in summary["control_comparison"].items():
        lines.append(
            f"| {symbol} | {row.get('baseline_pf')} | {row.get('time_only_pf')} | "
            f"{row.get('best_candidate_feature_set')} | {row.get('best_candidate_pf')} | "
            f"{row.get('pf_delta_vs_baseline')} | {row.get('pf_delta_vs_time_only')} | "
            f"{row.get('beats_controls_pf')} | {row.get('recommendation')} |"
        )
    lines.extend(["", "## By Feature Set", ""])
    lines.append("| Feature Set | Count | Classes | Median PF | Median Avg |")
    lines.append("| --- | ---: | --- | ---: | ---: |")
    for feature_set, row in summary["by_feature_set"].items():
        lines.append(
            f"| {feature_set} | {row['count']} | `{row['classification_counts']}` | "
            f"{row['median_validation_pf']} | {row['median_validation_avg_return']} |"
        )
    return "\n".join(lines).rstrip() + "\n"


def summarize_feature_policy_confirmation_registry(
    registry_path: str | Path = FEATURE_POLICY_REGISTRY_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    records = enrich_registry_records(load_latest_registry_records(registry_path))
    summary = build_summary(records)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    stamp = utc_stamp()
    json_path = target / f"feature_policy_confirmation_v1_global_summary_{stamp}.json"
    markdown_path = target / f"feature_policy_confirmation_v1_global_summary_{stamp}.md"
    summary["json_path"] = str(json_path)
    summary["markdown_path"] = str(markdown_path)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(render_markdown(summary), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize Feature Policy Confirmation v1.")
    parser.add_argument("--registry", default=str(FEATURE_POLICY_REGISTRY_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = summarize_feature_policy_confirmation_registry(args.registry, args.output_dir)
    print("Feature Policy Confirmation v1 global summary generated")
    print("Research only. No trading signal.")
    print(f"total_configs: {summary['summary']['unique_configs']}")
    print(f"completed: {summary['summary']['completed']}")
    print(f"classification_counts: {summary['summary']['classification_counts']}")
    print(f"json: {summary['json_path']}")
    print(f"markdown: {summary['markdown_path']}")


if __name__ == "__main__":
    main()
