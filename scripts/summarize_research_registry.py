"""Summarize the complete local Research Daemon registry.

This is an offline research report. It does not run experiments, change models,
send signals, write Supabase, or use test metrics for selection.
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


DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "reports" / "research_daemon" / "registry.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "research_daemon"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_registry_rows(path: str | Path) -> list[dict[str, Any]]:
    registry_path = Path(path)
    if not registry_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in registry_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_latest_registry_records(path: str | Path) -> list[dict[str, Any]]:
    """Return the latest registry row per config_id, preserving latest order."""
    latest: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for row in load_registry_rows(path):
        cid = str(row.get("config_id"))
        if cid not in latest:
            order.append(cid)
        latest[cid] = row
    return [latest[cid] for cid in order if cid in latest]


def _resolve_path(path: str | Path | None, base_dir: str | Path | None = None) -> Path | None:
    if not path:
        return None
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    if base_dir is not None:
        base_candidate = Path(base_dir) / candidate
        if base_candidate.exists():
            return base_candidate
    return PROJECT_ROOT / candidate


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _first_setup_result(payload: dict[str, Any]) -> dict[str, Any]:
    setups = payload.get("setups") or []
    return setups[0] if setups else {}


def enrich_registry_records(records: list[dict[str, Any]], base_dir: str | Path | None = None) -> list[dict[str, Any]]:
    """Attach aggregate metrics from each result JSON when available."""
    enriched: list[dict[str, Any]] = []
    for record in records:
        row = dict(record)
        config = row.get("config") or {}
        row["config"] = config
        row["aggregate"] = {}
        row["json_loaded"] = False
        row["json_missing"] = False
        row["json_error"] = None
        json_path = _resolve_path(row.get("json_path"), base_dir=base_dir)
        if json_path is not None:
            try:
                payload = _read_json(json_path)
                setup_result = _first_setup_result(payload)
                row["aggregate"] = setup_result.get("aggregate") or {}
                row["result_classification"] = setup_result.get("classification")
                row["json_loaded"] = True
            except FileNotFoundError:
                row["json_missing"] = True
            except Exception as exc:  # noqa: BLE001 - report should survive bad result files.
                row["json_error"] = f"{type(exc).__name__}: {exc}"
        enriched.append(row)
    return enriched


def _classification(row: dict[str, Any]) -> str:
    if row.get("status") == "failed":
        return "failed"
    if row.get("status") == "insufficient_data":
        return "insufficient_data"
    return str(row.get("classification") or row.get("status") or "unknown")


def _classification_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        label = _classification(row)
        counts[label] = counts.get(label, 0) + 1
    return counts


def _float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _metric(row: dict[str, Any], name: str) -> float | None:
    return _float((row.get("aggregate") or {}).get(name))


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return round(ordered[mid], 6)
    return round((ordered[mid - 1] + ordered[mid]) / 2, 6)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def _top_rows(rows: list[dict[str, Any]], metric: str, limit: int = 10) -> list[dict[str, Any]]:
    valid = [row for row in rows if _metric(row, metric) is not None]
    return sorted(valid, key=lambda row: _metric(row, metric) or float("-inf"), reverse=True)[:limit]


def _config_label(config: dict[str, Any]) -> str:
    return (
        f"{config.get('symbol')} {config.get('timeframe')} "
        f"h{config.get('horizon_candles')} RR{config.get('risk_reward')} "
        f"ATR{config.get('atr_stop_multiplier')} {config.get('cost_mode')} "
        f"{config.get('strategy_mode')}"
    )


def _group_key(config: dict[str, Any], fields: list[str]) -> str:
    return " | ".join(f"{field}={config.get(field)}" for field in fields)


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pf_values = [_metric(row, "median_validation_pf") for row in rows]
    avg_values = [_metric(row, "median_validation_avg_return") for row in rows]
    validation_rates = [_metric(row, "validation_positive_rate") for row in rows]
    random_rates = [_metric(row, "beats_random_rate") for row in rows]
    deterministic_rates = [_metric(row, "beats_deterministic_rate") for row in rows]
    test_rates = [_metric(row, "test_confirm_rate") for row in rows]
    drawdowns = [_metric(row, "worst_validation_drawdown") for row in rows]
    return {
        "count": len(rows),
        "classification_counts": _classification_counts(rows),
        "median_validation_pf": _median([value for value in pf_values if value is not None]),
        "median_validation_avg_return": _median([value for value in avg_values if value is not None]),
        "avg_validation_positive_rate": _mean([value for value in validation_rates if value is not None]),
        "avg_beats_random_rate": _mean([value for value in random_rates if value is not None]),
        "avg_beats_deterministic_rate": _mean([value for value in deterministic_rates if value is not None]),
        "avg_test_confirm_rate": _mean([value for value in test_rates if value is not None]),
        "median_worst_drawdown": _median([value for value in drawdowns if value is not None]),
    }


def group_by_config_fields(rows: list[dict[str, Any]], fields: list[str]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_group_key(row.get("config") or {}, fields)].append(row)
    return {key: summarize_group(group_rows) for key, group_rows in sorted(groups.items())}


def _row_summary(row: dict[str, Any]) -> dict[str, Any]:
    config = row.get("config") or {}
    aggregate = row.get("aggregate") or {}
    return {
        "config_id": row.get("config_id"),
        "classification": _classification(row),
        "status": row.get("status"),
        "config": config,
        "label": _config_label(config),
        "median_validation_pf": aggregate.get("median_validation_pf"),
        "median_validation_avg_return": aggregate.get("median_validation_avg_return"),
        "median_test_pf": aggregate.get("median_test_pf"),
        "test_confirm_rate": aggregate.get("test_confirm_rate"),
        "validation_positive_rate": aggregate.get("validation_positive_rate"),
        "beats_random_rate": aggregate.get("beats_random_rate"),
        "beats_deterministic_rate": aggregate.get("beats_deterministic_rate"),
        "worst_validation_drawdown": aggregate.get("worst_validation_drawdown"),
        "json_path": row.get("json_path"),
        "json_loaded": row.get("json_loaded"),
        "json_missing": row.get("json_missing"),
        "json_error": row.get("json_error"),
    }


def build_conclusion(counts: dict[str, int]) -> list[str]:
    conclusions: list[str] = []
    stable = counts.get("stable_research_candidate", 0)
    watchlist = counts.get("unstable_watchlist", 0)
    if stable == 0:
        conclusions.append("No hay configuracion lista para paper trading.")
    if stable == 0 and watchlist > 0:
        conclusions.append("Hay zonas de investigacion, no senales operativas.")
    if stable > 0:
        conclusions.append("Hay candidates de investigacion; requieren validacion adicional antes de paper trading.")
    conclusions.append("Test PF se reporta como diagnostico no seleccionable; validation sigue siendo la base de seleccion.")
    return conclusions


def build_global_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts = _classification_counts(records)
    completed = [row for row in records if row.get("status") == "completed"]
    return {
        "generated_at": utc_now(),
        "guardrails": {
            "research_only": True,
            "no_trading": True,
            "no_paper_trading": True,
            "models_unchanged": True,
            "test_not_used_for_selection": True,
            "accuracy_not_used": True,
        },
        "summary": {
            "unique_configs": len(records),
            "unique_completed_configs": len(completed),
            "classification_counts": counts,
            "stable_research_candidate": counts.get("stable_research_candidate", 0),
            "unstable_watchlist": counts.get("unstable_watchlist", 0),
            "multi_window_reject": counts.get("multi_window_reject", 0),
            "failed": counts.get("failed", 0),
            "insufficient_data": counts.get("insufficient_data", 0),
            "json_loaded": sum(1 for row in records if row.get("json_loaded")),
            "json_missing": sum(1 for row in records if row.get("json_missing")),
            "json_errors": sum(1 for row in records if row.get("json_error")),
        },
        "top": {
            "median_validation_pf": [_row_summary(row) for row in _top_rows(records, "median_validation_pf")],
            "median_validation_avg_return": [_row_summary(row) for row in _top_rows(records, "median_validation_avg_return")],
            "median_test_pf_diagnostic_only": [_row_summary(row) for row in _top_rows(records, "median_test_pf")],
            "test_confirm_rate_diagnostic_only": [_row_summary(row) for row in _top_rows(records, "test_confirm_rate")],
        },
        "groupings": {
            "horizon_candles": group_by_config_fields(records, ["horizon_candles"]),
            "risk_reward": group_by_config_fields(records, ["risk_reward"]),
            "atr_stop_multiplier": group_by_config_fields(records, ["atr_stop_multiplier"]),
            "cost_mode": group_by_config_fields(records, ["cost_mode"]),
            "horizon_rr_atr_cost": group_by_config_fields(records, ["horizon_candles", "risk_reward", "atr_stop_multiplier", "cost_mode"]),
        },
        "cost_analysis": group_by_config_fields(records, ["cost_mode"]),
        "stability_analysis": summarize_group(records),
        "conclusion": build_conclusion(counts),
        "records": [_row_summary(row) for row in records],
    }


def _markdown_row(row: dict[str, Any]) -> str:
    return (
        f"- `{row.get('classification')}: {row.get('label')}` "
        f"val PF `{row.get('median_validation_pf')}`, "
        f"val avg `{row.get('median_validation_avg_return')}`, "
        f"test PF `{row.get('median_test_pf')}`, "
        f"test confirm `{row.get('test_confirm_rate')}`"
    )


def _markdown_group_table(groups: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        "| Group | Count | Classes | Median Val PF | Median Val Avg | Avg Val+ | Avg Beats Random | Avg Beats Det | Avg Test Confirm | Median Worst DD |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, value in groups.items():
        lines.append(
            f"| {key} | {value.get('count')} | `{value.get('classification_counts')}` | "
            f"{value.get('median_validation_pf')} | {value.get('median_validation_avg_return')} | "
            f"{value.get('avg_validation_positive_rate')} | {value.get('avg_beats_random_rate')} | "
            f"{value.get('avg_beats_deterministic_rate')} | {value.get('avg_test_confirm_rate')} | "
            f"{value.get('median_worst_drawdown')} |"
        )
    return lines


def render_markdown(summary: dict[str, Any]) -> str:
    overview = summary["summary"]
    lines = [
        "# Research Daemon Global Summary",
        "",
        f"Generated at: `{summary.get('generated_at')}`",
        "",
        "## Guardrails",
        "",
        "- Research only. No trading signal.",
        "- No models, features, thresholds, costs, grid, classification, or validation logic changed.",
        "- Validation selects; test metrics are diagnostic only and not selectable.",
        "- Accuracy is not used.",
        "",
        "## General Summary",
        "",
        f"- unique configs: `{overview.get('unique_configs')}`",
        f"- unique completed configs: `{overview.get('unique_completed_configs')}`",
        f"- classification counts: `{overview.get('classification_counts')}`",
        f"- stable_research_candidate: `{overview.get('stable_research_candidate')}`",
        f"- unstable_watchlist: `{overview.get('unstable_watchlist')}`",
        f"- multi_window_reject: `{overview.get('multi_window_reject')}`",
        f"- failed: `{overview.get('failed')}`",
        f"- insufficient_data: `{overview.get('insufficient_data')}`",
        f"- json loaded/missing/errors: `{overview.get('json_loaded')}` / `{overview.get('json_missing')}` / `{overview.get('json_errors')}`",
        "",
        "## Conclusion",
        "",
    ]
    lines.extend(f"- {item}" for item in summary.get("conclusion", []))
    lines.extend(["", "## Top 10 By Median Validation PF", ""])
    lines.extend(_markdown_row(row) for row in summary["top"]["median_validation_pf"])
    lines.extend(["", "## Top 10 By Median Validation Avg Return", ""])
    lines.extend(_markdown_row(row) for row in summary["top"]["median_validation_avg_return"])
    lines.extend(["", "## Top 10 By Median Test PF (Diagnostic Only, Not Selectable)", ""])
    lines.extend(_markdown_row(row) for row in summary["top"]["median_test_pf_diagnostic_only"])
    lines.extend(["", "## Top 10 By Test Confirm Rate (Diagnostic Only, Not Selectable)", ""])
    lines.extend(_markdown_row(row) for row in summary["top"]["test_confirm_rate_diagnostic_only"])
    lines.extend(["", "## Cost Analysis", ""])
    lines.extend(_markdown_group_table(summary["cost_analysis"]))
    lines.extend(["", "## Stability Analysis", ""])
    lines.append(f"`{summary['stability_analysis']}`")
    lines.extend(["", "## Groupings By Horizon", ""])
    lines.extend(_markdown_group_table(summary["groupings"]["horizon_candles"]))
    lines.extend(["", "## Groupings By Risk Reward", ""])
    lines.extend(_markdown_group_table(summary["groupings"]["risk_reward"]))
    lines.extend(["", "## Groupings By ATR Stop Multiplier", ""])
    lines.extend(_markdown_group_table(summary["groupings"]["atr_stop_multiplier"]))
    lines.extend(["", "## Groupings By Horizon + RR + ATR + Cost", ""])
    lines.extend(_markdown_group_table(summary["groupings"]["horizon_rr_atr_cost"]))
    return "\n".join(lines).rstrip() + "\n"


def write_global_summary(summary: dict[str, Any], output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Path]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    stamp = utc_stamp()
    json_path = target / f"global_summary_{stamp}.json"
    markdown_path = target / f"global_summary_{stamp}.md"
    summary["json_path"] = str(json_path)
    summary["markdown_path"] = str(markdown_path)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(render_markdown(summary), encoding="utf-8")
    return {"json_path": json_path, "markdown_path": markdown_path}


def summarize_registry(registry_path: str | Path = DEFAULT_REGISTRY_PATH, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    records = load_latest_registry_records(registry_path)
    enriched = enrich_registry_records(records)
    summary = build_global_summary(enriched)
    paths = write_global_summary(summary, output_dir=output_dir)
    summary["json_path"] = str(paths["json_path"])
    summary["markdown_path"] = str(paths["markdown_path"])
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize Research Daemon registry.")
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = summarize_registry(args.registry, args.output_dir)
    overview = summary["summary"]
    print("Research registry global summary generated")
    print(f"unique_configs: {overview['unique_configs']}")
    print(f"unique_completed_configs: {overview['unique_completed_configs']}")
    print(f"classification_counts: {overview['classification_counts']}")
    print(f"json: {summary['json_path']}")
    print(f"markdown: {summary['markdown_path']}")


if __name__ == "__main__":
    main()
