"""Summarize 4h Focused Research v1 results.

Offline research only. This script reads the 4h_focused_v1 registry and answers
whether 4h_only is consistently superior to baseline_plus_btc_context across
ADA, ETH, and SOL.
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

from scripts.run_4h_focused_research import FOUR_H_FOCUSED_REGISTRY_PATH  # noqa: E402
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


def _counts(rows: list[dict[str, Any]], key: str | None = None) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        value = _classification(row) if key is None else str((row.get("config") or {}).get(key))
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
        "valid_windows": aggregate.get("valid_windows"),
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
        out[key] = {
            "count": len(items),
            "classification_counts": _counts(items),
            "median_validation_pf": _median([value for value in pf_values if value is not None]),
            "median_validation_avg_return": _median([value for value in avg_values if value is not None]),
        }
    return out


def symbol_comparison(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_symbol: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        config = row.get("config") or {}
        by_symbol[str(config.get("symbol"))][str(config.get("feature_set"))] = row

    output: dict[str, dict[str, Any]] = {}
    for symbol, feature_rows in sorted(by_symbol.items()):
        four_h = feature_rows.get("4h_only")
        btc = feature_rows.get("baseline_plus_btc_context")
        four_h_pf = _metric(four_h or {}, "median_validation_pf")
        btc_pf = _metric(btc or {}, "median_validation_pf")
        if four_h_pf is None or btc_pf is None:
            verdict = "incomplete"
        elif four_h_pf > btc_pf:
            verdict = "4h_only_better"
        elif btc_pf > four_h_pf:
            verdict = "btc_context_better"
        else:
            verdict = "tie"
        output[symbol] = {
            "4h_only": _row_summary(four_h) if four_h else None,
            "baseline_plus_btc_context": _row_summary(btc) if btc else None,
            "delta_pf_4h_minus_btc_context": round(four_h_pf - btc_pf, 6) if four_h_pf is not None and btc_pf is not None else None,
            "verdict": verdict,
        }
    return output


def automatic_conclusion(summary: dict[str, Any]) -> list[str]:
    comparisons = summary["symbol_comparison"]
    verdicts = [row.get("verdict") for row in comparisons.values()]
    conclusions = ["Research only. No trading signal."]
    if comparisons and all(verdict == "4h_only_better" for verdict in verdicts):
        conclusions.append("4h_only is consistently superior to baseline_plus_btc_context across the tested symbols.")
    elif "4h_only_better" in verdicts:
        conclusions.append("4h_only improves at least one symbol, but is not consistently superior across all tested symbols.")
    else:
        conclusions.append("4h_only is not superior to baseline_plus_btc_context in this registry.")
    conclusions.append("Validation metrics decide comparison; no operational promotion is implied.")
    return conclusions


def build_summary(records: list[dict[str, Any]], top_limit: int = 15) -> dict[str, Any]:
    completed = [row for row in records if row.get("status") == "completed"]
    summary = {
        "generated_at": utc_now(),
        "report_type": "4h_focused_v1_global_summary",
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
        "top": {
            "median_validation_pf": _top(records, "median_validation_pf", top_limit),
            "median_validation_avg_return": _top(records, "median_validation_avg_return", top_limit),
        },
        "groupings": {
            "symbol": group_records(records, "symbol"),
            "feature_set": group_records(records, "feature_set"),
        },
        "symbol_comparison": symbol_comparison(records),
        "records": [_row_summary(row) for row in records],
    }
    summary["conclusion"] = automatic_conclusion(summary)
    return summary


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# 4h Focused v1 Global Summary",
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
        "",
        "## Conclusion",
        "",
    ]
    lines.extend(f"- {item}" for item in summary.get("conclusion", []))
    lines.extend(["", "## Symbol Comparison", ""])
    lines.append("| Symbol | 4h PF | BTC Context PF | Delta PF | Verdict |")
    lines.append("| --- | ---: | ---: | ---: | --- |")
    for symbol, row in summary["symbol_comparison"].items():
        four_h = row.get("4h_only") or {}
        btc = row.get("baseline_plus_btc_context") or {}
        lines.append(
            f"| {symbol} | {four_h.get('median_validation_pf')} | "
            f"{btc.get('median_validation_pf')} | {row.get('delta_pf_4h_minus_btc_context')} | {row.get('verdict')} |"
        )
    lines.extend(["", "## By Feature Set", ""])
    lines.append("| Feature Set | Count | Classes | Median PF | Median Avg |")
    lines.append("| --- | ---: | --- | ---: | ---: |")
    for key, row in summary["groupings"]["feature_set"].items():
        lines.append(
            f"| {key} | {row['count']} | `{row['classification_counts']}` | "
            f"{row['median_validation_pf']} | {row['median_validation_avg_return']} |"
        )
    lines.extend(["", "## Top By Median Validation PF", ""])
    for row in summary["top"]["median_validation_pf"]:
        lines.append(
            f"- `{row['classification']}: {row['label']}` PF `{row['median_validation_pf']}`, "
            f"avg `{row['median_validation_avg_return']}`"
        )
    return "\n".join(lines).rstrip() + "\n"


def summarize_4h_focused_registry(
    registry_path: str | Path = FOUR_H_FOCUSED_REGISTRY_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    top_limit: int = 15,
) -> dict[str, Any]:
    records = enrich_registry_records(load_latest_registry_records(registry_path))
    summary = build_summary(records, top_limit=top_limit)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    stamp = utc_stamp()
    json_path = target / f"4h_focused_v1_global_summary_{stamp}.json"
    markdown_path = target / f"4h_focused_v1_global_summary_{stamp}.md"
    summary["json_path"] = str(json_path)
    summary["markdown_path"] = str(markdown_path)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(render_markdown(summary), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize 4h Focused Research v1.")
    parser.add_argument("--registry", default=str(FOUR_H_FOCUSED_REGISTRY_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--top-limit", type=int, default=15)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = summarize_4h_focused_registry(args.registry, args.output_dir, args.top_limit)
    print("4h Focused Research v1 global summary generated")
    print("Research only. No trading signal.")
    print(f"total_configs: {summary['summary']['unique_configs']}")
    print(f"completed: {summary['summary']['completed']}")
    print(f"classification_counts: {summary['summary']['classification_counts']}")
    print(f"json: {summary['json_path']}")
    print(f"markdown: {summary['markdown_path']}")


if __name__ == "__main__":
    main()
