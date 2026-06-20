"""Compare two research registries over matching configurations.

Research only. This script does not run experiments, generate signals, change
models, or use test metrics for selection. It is meant to answer whether a new
research phase improved validation evidence over a baseline phase.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.summarize_research_registry import enrich_registry_records, load_latest_registry_records  # noqa: E402


DEFAULT_BASELINE_REGISTRY = PROJECT_ROOT / "reports" / "research_daemon" / "focused_v2a_registry.jsonl"
DEFAULT_CANDIDATE_REGISTRY = PROJECT_ROOT / "reports" / "research_daemon" / "market_context_v1_registry.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "research_daemon"
MATCH_FIELDS = [
    "symbol",
    "timeframe",
    "horizon_candles",
    "risk_reward",
    "atr_stop_multiplier",
    "cost_mode",
    "strategy_mode",
]


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


def phase_key(config: dict[str, Any]) -> str:
    return "|".join(f"{field}={config.get(field)}" for field in MATCH_FIELDS)


def _metric(row: dict[str, Any], name: str) -> float | None:
    return _float((row.get("aggregate") or {}).get(name))


def _classification(row: dict[str, Any] | None) -> str:
    if not row:
        return "missing"
    if row.get("status") == "failed":
        return "failed"
    return str(row.get("classification") or row.get("status") or "unknown")


def _row_label(row: dict[str, Any]) -> str:
    config = row.get("config") or {}
    return (
        f"{config.get('symbol')} {config.get('timeframe')} h{config.get('horizon_candles')} "
        f"RR{config.get('risk_reward')} ATR{config.get('atr_stop_multiplier')} "
        f"{config.get('cost_mode')} {config.get('strategy_mode')}"
    )


def _index_records(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        indexed[phase_key(row.get("config") or {})] = row
    return indexed


def load_phase(path: str | Path) -> list[dict[str, Any]]:
    return enrich_registry_records(load_latest_registry_records(path))


def compare_rows(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    baseline_pf = _metric(baseline, "median_validation_pf")
    candidate_pf = _metric(candidate, "median_validation_pf")
    baseline_avg = _metric(baseline, "median_validation_avg_return")
    candidate_avg = _metric(candidate, "median_validation_avg_return")
    baseline_drawdown = _metric(baseline, "worst_validation_drawdown")
    candidate_drawdown = _metric(candidate, "worst_validation_drawdown")
    baseline_test_confirm = _metric(baseline, "test_confirm_rate")
    candidate_test_confirm = _metric(candidate, "test_confirm_rate")

    delta_pf = None if baseline_pf is None or candidate_pf is None else round(candidate_pf - baseline_pf, 6)
    delta_avg = None if baseline_avg is None or candidate_avg is None else round(candidate_avg - baseline_avg, 6)
    delta_drawdown = (
        None if baseline_drawdown is None or candidate_drawdown is None else round(candidate_drawdown - baseline_drawdown, 6)
    )
    delta_test_confirm = (
        None if baseline_test_confirm is None or candidate_test_confirm is None
        else round(candidate_test_confirm - baseline_test_confirm, 6)
    )

    if delta_pf is None or delta_avg is None:
        verdict = "incomplete"
    elif delta_pf > 0.05 and delta_avg > 0:
        verdict = "improved_validation"
    elif delta_pf < -0.05 or delta_avg < 0:
        verdict = "worse_validation"
    else:
        verdict = "neutral_validation"

    return {
        "key": phase_key(baseline.get("config") or {}),
        "label": _row_label(baseline),
        "baseline_config_id": baseline.get("config_id"),
        "candidate_config_id": candidate.get("config_id"),
        "baseline_classification": _classification(baseline),
        "candidate_classification": _classification(candidate),
        "baseline_feature_family": (baseline.get("config") or {}).get("feature_family"),
        "candidate_feature_family": (candidate.get("config") or {}).get("feature_family"),
        "baseline_validation_pf": baseline_pf,
        "candidate_validation_pf": candidate_pf,
        "delta_validation_pf": delta_pf,
        "baseline_validation_avg_return": baseline_avg,
        "candidate_validation_avg_return": candidate_avg,
        "delta_validation_avg_return": delta_avg,
        "baseline_worst_drawdown": baseline_drawdown,
        "candidate_worst_drawdown": candidate_drawdown,
        "delta_worst_drawdown": delta_drawdown,
        "baseline_test_confirm_rate": baseline_test_confirm,
        "candidate_test_confirm_rate": candidate_test_confirm,
        "delta_test_confirm_rate": delta_test_confirm,
        "verdict": verdict,
    }


def _counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key))
        counts[value] = counts.get(value, 0) + 1
    return counts


def build_comparison(
    baseline_records: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
    baseline_name: str = "baseline",
    candidate_name: str = "candidate",
) -> dict[str, Any]:
    baseline_index = _index_records(baseline_records)
    candidate_index = _index_records(candidate_records)
    matched_keys = sorted(set(baseline_index) & set(candidate_index))
    rows = [compare_rows(baseline_index[key], candidate_index[key]) for key in matched_keys]
    improved = [row for row in rows if row["verdict"] == "improved_validation"]
    worse = [row for row in rows if row["verdict"] == "worse_validation"]
    neutral = [row for row in rows if row["verdict"] == "neutral_validation"]

    avg_delta_pf_values = [row["delta_validation_pf"] for row in rows if row.get("delta_validation_pf") is not None]
    avg_delta_avg_values = [row["delta_validation_avg_return"] for row in rows if row.get("delta_validation_avg_return") is not None]
    avg_delta_drawdown_values = [row["delta_worst_drawdown"] for row in rows if row.get("delta_worst_drawdown") is not None]

    def mean(values: list[float]) -> float | None:
        return round(sum(values) / len(values), 6) if values else None

    conclusion: list[str] = ["Research only. No trading signal."]
    if len(worse) > len(improved):
        conclusion.append(f"{candidate_name} worsened validation evidence versus {baseline_name} on matched configs.")
    elif len(improved) > len(worse):
        conclusion.append(f"{candidate_name} improved validation evidence versus {baseline_name} on matched configs.")
    else:
        conclusion.append(f"{candidate_name} was mixed or neutral versus {baseline_name} on matched configs.")
    conclusion.append("Validation metrics decide comparison; test metrics are diagnostic only.")

    return {
        "generated_at": utc_now(),
        "baseline_name": baseline_name,
        "candidate_name": candidate_name,
        "match_fields": MATCH_FIELDS,
        "guardrails": {
            "research_only": True,
            "no_trading": True,
            "no_signal_generation": True,
            "test_not_used_for_selection": True,
            "accuracy_not_used": True,
        },
        "summary": {
            "baseline_records": len(baseline_records),
            "candidate_records": len(candidate_records),
            "matched_configs": len(rows),
            "baseline_only": len(set(baseline_index) - set(candidate_index)),
            "candidate_only": len(set(candidate_index) - set(baseline_index)),
            "verdict_counts": _counts(rows, "verdict"),
            "baseline_classification_counts_on_matched": _counts(rows, "baseline_classification"),
            "candidate_classification_counts_on_matched": _counts(rows, "candidate_classification"),
            "mean_delta_validation_pf": mean(avg_delta_pf_values),
            "mean_delta_validation_avg_return": mean(avg_delta_avg_values),
            "mean_delta_worst_drawdown": mean(avg_delta_drawdown_values),
        },
        "top_improvements": sorted(improved, key=lambda row: row.get("delta_validation_pf") or -999, reverse=True)[:10],
        "top_regressions": sorted(worse, key=lambda row: row.get("delta_validation_pf") or 999)[:10],
        "neutral": neutral,
        "rows": rows,
        "conclusion": conclusion,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    overview = summary["summary"]
    lines = [
        "# Research Phase Comparison",
        "",
        f"Generated at: `{summary['generated_at']}`",
        "",
        "## Guardrails",
        "",
        "- Research only. No trading signal.",
        "- Validation metrics decide comparison.",
        "- Test metrics are diagnostic only and not selectable.",
        "- Accuracy is not used.",
        "",
        "## Compared Phases",
        "",
        f"- baseline: `{summary['baseline_name']}`",
        f"- candidate: `{summary['candidate_name']}`",
        f"- match fields: `{summary['match_fields']}`",
        "",
        "## Summary",
        "",
        f"- baseline records: `{overview['baseline_records']}`",
        f"- candidate records: `{overview['candidate_records']}`",
        f"- matched configs: `{overview['matched_configs']}`",
        f"- baseline only: `{overview['baseline_only']}`",
        f"- candidate only: `{overview['candidate_only']}`",
        f"- verdict counts: `{overview['verdict_counts']}`",
        f"- baseline classes on matched: `{overview['baseline_classification_counts_on_matched']}`",
        f"- candidate classes on matched: `{overview['candidate_classification_counts_on_matched']}`",
        f"- mean delta validation PF: `{overview['mean_delta_validation_pf']}`",
        f"- mean delta validation avg return: `{overview['mean_delta_validation_avg_return']}`",
        f"- mean delta worst drawdown: `{overview['mean_delta_worst_drawdown']}`",
        "",
        "## Conclusion",
        "",
    ]
    lines.extend(f"- {item}" for item in summary.get("conclusion", []))

    def table(title: str, rows: list[dict[str, Any]]) -> None:
        lines.extend(["", f"## {title}", ""])
        if not rows:
            lines.append("None.")
            return
        lines.append(
            "| Config | Base Class | Cand Class | Base PF | Cand PF | dPF | Base Avg | Cand Avg | dAvg | Base DD | Cand DD | dDD |"
        )
        lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in rows:
            lines.append(
                f"| {row['label']} | {row['baseline_classification']} | {row['candidate_classification']} | "
                f"{row['baseline_validation_pf']} | {row['candidate_validation_pf']} | {row['delta_validation_pf']} | "
                f"{row['baseline_validation_avg_return']} | {row['candidate_validation_avg_return']} | "
                f"{row['delta_validation_avg_return']} | {row['baseline_worst_drawdown']} | "
                f"{row['candidate_worst_drawdown']} | {row['delta_worst_drawdown']} |"
            )

    table("Top Validation Improvements", summary["top_improvements"])
    table("Top Validation Regressions", summary["top_regressions"])
    table("All Matched Configs", summary["rows"])
    return "\n".join(lines).rstrip() + "\n"


def write_comparison(summary: dict[str, Any], output_dir: str | Path, filename_prefix: str) -> dict[str, str]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    stamp = utc_stamp()
    json_path = target / f"{filename_prefix}_{stamp}.json"
    markdown_path = target / f"{filename_prefix}_{stamp}.md"
    summary["json_path"] = str(json_path)
    summary["markdown_path"] = str(markdown_path)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(render_markdown(summary), encoding="utf-8")
    return {"json_path": str(json_path), "markdown_path": str(markdown_path)}


def run_comparison(args: argparse.Namespace) -> dict[str, Any]:
    baseline = load_phase(args.baseline_registry)
    candidate = load_phase(args.candidate_registry)
    summary = build_comparison(
        baseline,
        candidate,
        baseline_name=args.baseline_name,
        candidate_name=args.candidate_name,
    )
    paths = write_comparison(summary, args.output_dir, args.filename_prefix)
    summary.update(paths)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare two research phases on matched configs.")
    parser.add_argument("--baseline-registry", default=str(DEFAULT_BASELINE_REGISTRY))
    parser.add_argument("--candidate-registry", default=str(DEFAULT_CANDIDATE_REGISTRY))
    parser.add_argument("--baseline-name", default="focused_v2a")
    parser.add_argument("--candidate-name", default="market_context_v1")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--filename-prefix", default="research_phase_comparison")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_comparison(args)
    overview = summary["summary"]
    print("Research phase comparison generated")
    print("Research only. No trading signal.")
    print(f"matched_configs: {overview['matched_configs']}")
    print(f"verdict_counts: {overview['verdict_counts']}")
    print(f"mean_delta_validation_pf: {overview['mean_delta_validation_pf']}")
    print(f"mean_delta_validation_avg_return: {overview['mean_delta_validation_avg_return']}")
    print(f"json: {summary['json_path']}")
    print(f"markdown: {summary['markdown_path']}")


if __name__ == "__main__":
    main()
