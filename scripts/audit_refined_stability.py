"""Audit stability of the best Refined Grid 2A watchlist configs.

Offline research only. This script reads the refined registry and saved
multi-window result JSON files. It does not run experiments, change models,
generate signals, write Supabase, or use test metrics for selection.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.summarize_research_registry import (  # noqa: E402
    enrich_registry_records,
    load_latest_registry_records,
    stable_failure_reasons,
)


DEFAULT_REFINED_REGISTRY = PROJECT_ROOT / "reports" / "research_daemon" / "refined_registry.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "research_daemon"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return round(ordered[mid], 6)
    return round((ordered[mid - 1] + ordered[mid]) / 2, 6)


def _metric(row: dict[str, Any], name: str) -> float | None:
    return _float((row.get("aggregate") or {}).get(name))


def _config_label(config: dict[str, Any]) -> str:
    return (
        f"{config.get('symbol')} {config.get('timeframe')} "
        f"h{config.get('horizon_candles')} RR{config.get('risk_reward')} "
        f"ATR{config.get('atr_stop_multiplier')} {config.get('cost_mode')} "
        f"{config.get('strategy_mode')}"
    )


def _resolve_path(path: str | Path | None) -> Path | None:
    if not path:
        return None
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def load_result_payload(record: dict[str, Any]) -> dict[str, Any] | None:
    """Load a saved multi-window JSON payload, returning None if unavailable."""
    json_path = _resolve_path(record.get("json_path"))
    if json_path is None or not json_path.exists():
        return None
    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - bad result files should not kill the audit.
        return None


def select_top_watchlist_configs(records: list[dict[str, Any]], top_n: int = 5) -> list[dict[str, Any]]:
    """Select top watchlist configs using validation metrics only."""
    watchlist = [
        row for row in records
        if row.get("status") == "completed"
        and row.get("classification") == "unstable_watchlist"
    ]
    return sorted(
        watchlist,
        key=lambda row: (
            _metric(row, "median_validation_pf") or float("-inf"),
            _metric(row, "median_validation_avg_return") or float("-inf"),
            _metric(row, "validation_positive_rate") or float("-inf"),
        ),
        reverse=True,
    )[:top_n]


def window_is_test_contradiction(row: dict[str, Any]) -> bool:
    test_avg = _float(row.get("test_avg_return"))
    test_pf = _float(row.get("test_profit_factor"))
    return test_avg is not None and test_pf is not None and test_avg <= 0 and test_pf < 1.0


def summarize_windows(windows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in windows if row.get("window_status") == "valid"]
    positive = [row for row in valid if bool(row.get("validation_positive"))]
    negative = [row for row in valid if not bool(row.get("validation_positive"))]
    beats_random = [row for row in valid if bool(row.get("beats_random_validation"))]
    beats_deterministic = [row for row in valid if bool(row.get("beats_deterministic_validation"))]
    test_contradictions = [row for row in valid if window_is_test_contradiction(row)]
    n_trades_values = [
        float(row.get("n_trades"))
        for row in valid
        if _float(row.get("n_trades")) is not None
    ]
    bias_counts = Counter(str(row.get("directional_bias", "unknown")) for row in valid)
    return {
        "valid_windows": len(valid),
        "n_trades_total": int(sum(n_trades_values)) if n_trades_values else 0,
        "n_trades_median_per_window": _median(n_trades_values),
        "directional_bias_counts": dict(bias_counts),
        "directional_bias_by_window": [
            {
                "window_index": row.get("window_index"),
                "directional_bias": row.get("directional_bias"),
                "n_trades": row.get("n_trades"),
            }
            for row in valid
        ],
        "positive_windows": [row.get("window_index") for row in positive],
        "negative_windows": [row.get("window_index") for row in negative],
        "beats_random_windows": [row.get("window_index") for row in beats_random],
        "beats_deterministic_windows": [row.get("window_index") for row in beats_deterministic],
        "test_contradiction_windows": [row.get("window_index") for row in test_contradictions],
    }


def diagnose_watchlist_class(aggregate: dict[str, Any], window_summary: dict[str, Any]) -> str:
    """Assign a diagnostic class without changing selection rules."""
    valid_windows = int(aggregate.get("valid_windows") or 0)
    if valid_windows < 3:
        return "weak_watchlist"

    validation_rate = float(aggregate.get("validation_positive_rate") or 0.0)
    beats_random = float(aggregate.get("beats_random_rate") or 0.0)
    beats_deterministic = float(aggregate.get("beats_deterministic_rate") or 0.0)
    median_pf = float(aggregate.get("median_validation_pf") or 0.0)
    median_avg = float(aggregate.get("median_validation_avg_return") or 0.0)
    test_confirm = float(aggregate.get("test_confirm_rate") or 0.0)
    test_contradiction = float(aggregate.get("test_contradiction_rate") or 0.0)
    worst_drawdown = _float(aggregate.get("worst_validation_drawdown")) or 0.0

    validation_near_stable = (
        validation_rate >= 0.55
        and beats_random >= 0.55
        and beats_deterministic >= 0.45
        and median_pf > 1.05
        and median_avg > 0
    )
    if validation_near_stable and (test_confirm < 0.40 or test_contradiction > 0.40):
        return "near_stable_but_low_test_confirm"
    if median_pf > 1.05 and median_avg > 0 and worst_drawdown > 50:
        return "near_stable_but_drawdown_high"
    if median_pf > 1.0 and median_avg > 0 and beats_random < 0.60:
        return "near_stable_but_random_not_beaten"
    if median_pf > 1.0 and median_avg > 0 and beats_deterministic < 0.50:
        return "near_stable_but_deterministic_not_beaten"
    if validation_rate < 0.60 or len(window_summary.get("negative_windows") or []) > len(window_summary.get("positive_windows") or []):
        return "unstable_due_to_window_concentration"
    return "weak_watchlist"


def audit_record(record: dict[str, Any]) -> dict[str, Any]:
    payload = load_result_payload(record)
    setup_result = (payload.get("setups") or [{}])[0] if payload else {}
    setup = setup_result.get("setup") or record.get("config") or {}
    aggregate = setup_result.get("aggregate") or record.get("aggregate") or {}
    windows = setup_result.get("windows") or []
    window_summary = summarize_windows(windows)
    reasons = stable_failure_reasons({**record, "aggregate": aggregate})
    return {
        "config_id": record.get("config_id"),
        "label": _config_label(setup),
        "classification": record.get("classification"),
        "diagnostic_classification": diagnose_watchlist_class(aggregate, window_summary),
        "config": setup,
        "json_path": record.get("json_path"),
        "json_loaded": bool(payload),
        "validation_positive_rate": aggregate.get("validation_positive_rate"),
        "beats_random_rate": aggregate.get("beats_random_rate"),
        "beats_deterministic_rate": aggregate.get("beats_deterministic_rate"),
        "test_confirm_rate": aggregate.get("test_confirm_rate"),
        "median_validation_pf": aggregate.get("median_validation_pf"),
        "median_validation_avg_return": aggregate.get("median_validation_avg_return"),
        "worst_validation_drawdown": aggregate.get("worst_validation_drawdown"),
        "median_test_pf": aggregate.get("median_test_pf"),
        "median_test_avg_return": aggregate.get("median_test_avg_return"),
        "valid_windows": aggregate.get("valid_windows"),
        "test_contradiction_rate": aggregate.get("test_contradiction_rate"),
        "n_trades_total": window_summary["n_trades_total"],
        "n_trades_median_per_window": window_summary["n_trades_median_per_window"],
        "directional_bias_counts": window_summary["directional_bias_counts"],
        "directional_bias_by_window": window_summary["directional_bias_by_window"],
        "positive_windows": window_summary["positive_windows"],
        "negative_windows": window_summary["negative_windows"],
        "beats_random_windows": window_summary["beats_random_windows"],
        "beats_deterministic_windows": window_summary["beats_deterministic_windows"],
        "test_contradiction_windows": window_summary["test_contradiction_windows"],
        "stable_failure_reasons": reasons,
    }


def build_stability_audit(records: list[dict[str, Any]], top_n: int = 5) -> dict[str, Any]:
    selected = select_top_watchlist_configs(records, top_n=top_n)
    audits = [audit_record(row) for row in selected]
    return {
        "generated_at": utc_now(),
        "report_type": "refined_stability_audit",
        "guardrails": {
            "research_only": True,
            "no_trading": True,
            "no_paper_trading": True,
            "models_unchanged": True,
            "features_unchanged": True,
            "thresholds_unchanged": True,
            "costs_unchanged": True,
            "test_not_used_for_selection": True,
            "selection_metric": "median_validation_pf",
        },
        "summary": {
            "records_loaded": len(records),
            "watchlist_records": sum(1 for row in records if row.get("classification") == "unstable_watchlist"),
            "audited_configs": len(audits),
            "missing_json_in_selected": sum(1 for row in audits if not row.get("json_loaded")),
            "diagnostic_counts": dict(Counter(row["diagnostic_classification"] for row in audits)),
        },
        "audits": audits,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Refined Stability Audit",
        "",
        f"Generated at: `{summary.get('generated_at')}`",
        "",
        "## Guardrails",
        "",
        "- Research only. No trading signal.",
        "- No models, features, thresholds, costs, grid, classification, or multi-window validation changed.",
        "- Selection uses validation metrics only; test is diagnostic and not selectable.",
        "",
        "## Summary",
        "",
        f"- records loaded: `{summary['summary'].get('records_loaded')}`",
        f"- watchlist records: `{summary['summary'].get('watchlist_records')}`",
        f"- audited configs: `{summary['summary'].get('audited_configs')}`",
        f"- missing selected JSON files: `{summary['summary'].get('missing_json_in_selected')}`",
        f"- diagnostic counts: `{summary['summary'].get('diagnostic_counts')}`",
        "",
        "## Audited Configs",
        "",
    ]
    for idx, row in enumerate(summary.get("audits", []), start=1):
        lines.extend([
            f"### {idx}. {row.get('label')}",
            "",
            f"- diagnostic classification: `{row.get('diagnostic_classification')}`",
            f"- original classification: `{row.get('classification')}`",
            f"- validation_positive_rate: `{row.get('validation_positive_rate')}`",
            f"- beats_random_rate: `{row.get('beats_random_rate')}`",
            f"- beats_deterministic_rate: `{row.get('beats_deterministic_rate')}`",
            f"- test_confirm_rate: `{row.get('test_confirm_rate')}`",
            f"- median_validation_pf: `{row.get('median_validation_pf')}`",
            f"- median_validation_avg_return: `{row.get('median_validation_avg_return')}`",
            f"- worst_validation_drawdown: `{row.get('worst_validation_drawdown')}`",
            f"- median_test_pf: `{row.get('median_test_pf')}`",
            f"- median_test_avg_return: `{row.get('median_test_avg_return')}`",
            f"- valid_windows: `{row.get('valid_windows')}`",
            f"- n_trades total / median per window: `{row.get('n_trades_total')}` / `{row.get('n_trades_median_per_window')}`",
            f"- directional_bias_counts: `{row.get('directional_bias_counts')}`",
            f"- positive_windows: `{row.get('positive_windows')}`",
            f"- negative_windows: `{row.get('negative_windows')}`",
            f"- beats_random_windows: `{row.get('beats_random_windows')}`",
            f"- beats_deterministic_windows: `{row.get('beats_deterministic_windows')}`",
            f"- test_contradiction_windows: `{row.get('test_contradiction_windows')}`",
            f"- why not stable: `{row.get('stable_failure_reasons')}`",
            "",
        ])
    lines.extend([
        "## Interpretation",
        "",
        "- `near_stable_but_random_not_beaten` means validation has positive shape, but it does not beat random often enough.",
        "- `near_stable_but_deterministic_not_beaten` means XGBoost is not clearly better than the deterministic baseline often enough.",
        "- `near_stable_but_low_test_confirm` means validation is close, but test frequently contradicts or fails to confirm.",
        "- `near_stable_but_drawdown_high` means the setup may have positive validation median metrics but unacceptable instability.",
        "- `unstable_due_to_window_concentration` means positive behavior is not spread across enough windows.",
    ])
    return "\n".join(lines).rstrip() + "\n"


def save_stability_audit(summary: dict[str, Any], output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Path]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    stamp = utc_stamp()
    json_path = target / f"refined_stability_audit_{stamp}.json"
    markdown_path = target / f"refined_stability_audit_{stamp}.md"
    summary["json_path"] = str(json_path)
    summary["markdown_path"] = str(markdown_path)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(render_markdown(summary), encoding="utf-8")
    return {"json_path": json_path, "markdown_path": markdown_path}


def run_audit(
    registry_path: str | Path = DEFAULT_REFINED_REGISTRY,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    top_n: int = 5,
) -> dict[str, Any]:
    records = enrich_registry_records(load_latest_registry_records(registry_path))
    summary = build_stability_audit(records, top_n=top_n)
    paths = save_stability_audit(summary, output_dir=output_dir)
    summary["json_path"] = str(paths["json_path"])
    summary["markdown_path"] = str(paths["markdown_path"])
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit stability of refined watchlist configs.")
    parser.add_argument("--registry", default=str(DEFAULT_REFINED_REGISTRY))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--top-n", type=int, default=5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_audit(
        registry_path=args.registry,
        output_dir=args.output_dir,
        top_n=args.top_n,
    )
    print("Refined stability audit generated")
    print(f"audited_configs: {summary['summary']['audited_configs']}")
    print(f"diagnostic_counts: {summary['summary']['diagnostic_counts']}")
    print(f"json: {summary['json_path']}")
    print(f"markdown: {summary['markdown_path']}")
    for row in summary.get("audits", []):
        print(
            f"- {row['label']} | {row['diagnostic_classification']} | "
            f"val_pf={row['median_validation_pf']} | val_avg={row['median_validation_avg_return']} | "
            f"val+={row['validation_positive_rate']} | beats_random={row['beats_random_rate']}"
        )


if __name__ == "__main__":
    main()
