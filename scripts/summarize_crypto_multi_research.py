"""Summarize Crypto Multi-Asset Research Grid v1 results.

Offline research only. This script reads the crypto_multi registry and result
JSON files, then writes a research summary. It does not run experiments, change
models, generate signals, or use test metrics for selection.
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

from scripts.summarize_research_registry import (  # noqa: E402
    _classification_counts,
    _config_label,
    _metric,
    _row_summary,
    _top_rows,
    enrich_registry_records,
    group_by_config_fields,
    load_latest_registry_records,
    summarize_group,
    utc_now,
    utc_stamp,
)


DEFAULT_CRYPTO_MULTI_REGISTRY = PROJECT_ROOT / "reports" / "research_daemon" / "crypto_multi_registry.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "research_daemon"


def _sorted_groups_by_metric(groups: dict[str, dict[str, Any]], metric: str, limit: int = 10) -> list[dict[str, Any]]:
    rows = [{"group": key, **value} for key, value in groups.items()]
    return sorted(
        rows,
        key=lambda row: row.get(metric) if row.get(metric) is not None else float("-inf"),
        reverse=True,
    )[:limit]


def _asset_diagnostics(symbol_groups: dict[str, dict[str, Any]]) -> dict[str, Any]:
    diagnostics: list[dict[str, Any]] = []
    for key, group in symbol_groups.items():
        symbol = key.split("=", 1)[-1]
        counts = group.get("classification_counts") or {}
        stable = int(counts.get("stable_research_candidate", 0))
        watchlist = int(counts.get("unstable_watchlist", 0))
        rejects = int(counts.get("multi_window_reject", 0))
        median_pf = group.get("median_validation_pf")
        median_avg = group.get("median_validation_avg_return")
        promising = bool(
            stable > 0
            or (
                watchlist > 0
                and median_pf is not None
                and median_pf > 1.0
                and median_avg is not None
                and median_avg > 0
            )
        )
        discardable = bool(
            stable == 0
            and watchlist == 0
            or (
                median_pf is not None
                and median_pf < 1.0
                and median_avg is not None
                and median_avg <= 0
                and rejects >= watchlist
            )
        )
        diagnostics.append({
            "symbol": symbol,
            "count": group.get("count", 0),
            "classification_counts": counts,
            "stable_research_candidate": stable,
            "unstable_watchlist": watchlist,
            "multi_window_reject": rejects,
            "median_validation_pf": median_pf,
            "median_validation_avg_return": median_avg,
            "avg_validation_positive_rate": group.get("avg_validation_positive_rate"),
            "avg_beats_random_rate": group.get("avg_beats_random_rate"),
            "avg_beats_deterministic_rate": group.get("avg_beats_deterministic_rate"),
            "promising": promising,
            "discardable": discardable,
        })
    by_watchlist = sorted(
        diagnostics,
        key=lambda row: (
            row.get("unstable_watchlist", 0),
            row.get("stable_research_candidate", 0),
            row.get("median_validation_pf") or float("-inf"),
            row.get("median_validation_avg_return") or float("-inf"),
        ),
        reverse=True,
    )
    return {
        "assets_by_watchlist_count": by_watchlist,
        "promising_assets": [row["symbol"] for row in diagnostics if row["promising"]],
        "discardable_assets": [row["symbol"] for row in diagnostics if row["discardable"]],
    }


def _crypto_conclusion(summary_counts: dict[str, int], asset_diagnostics: dict[str, Any]) -> list[str]:
    stable = int(summary_counts.get("stable_research_candidate", 0))
    watchlist = int(summary_counts.get("unstable_watchlist", 0))
    promising_assets = asset_diagnostics.get("promising_assets") or []
    discardable_assets = asset_diagnostics.get("discardable_assets") or []
    conclusions = ["Research only. No trading signal."]
    if stable > 0:
        conclusions.append(
            "Hay stable_research_candidate en crypto_multi; requieren shadow/paper controlado antes de cualquier decision real."
        )
    elif watchlist > 0:
        conclusions.append("No hay stable candidates; solo hay unstable watchlist para investigacion adicional.")
    else:
        conclusions.append("No hay candidates ni watchlist en crypto_multi.")
    if promising_assets:
        conclusions.append(f"Activos prometedores para investigacion: {', '.join(promising_assets)}.")
    else:
        conclusions.append("No se detectaron activos prometedores bajo criterios de validation.")
    if discardable_assets:
        conclusions.append(f"Activos descartables temporalmente: {', '.join(discardable_assets)}.")
    conclusions.append("Median test PF es diagnostico no seleccionable; validation sigue siendo la base de seleccion.")
    return conclusions


def build_crypto_multi_summary(records: list[dict[str, Any]], top_limit: int = 15) -> dict[str, Any]:
    counts = _classification_counts(records)
    completed = [row for row in records if row.get("status") == "completed"]
    groupings = {
        "symbol": group_by_config_fields(records, ["symbol"]),
        "horizon_candles": group_by_config_fields(records, ["horizon_candles"]),
        "risk_reward": group_by_config_fields(records, ["risk_reward"]),
        "atr_stop_multiplier": group_by_config_fields(records, ["atr_stop_multiplier"]),
        "symbol_horizon_rr_atr": group_by_config_fields(
            records,
            ["symbol", "horizon_candles", "risk_reward", "atr_stop_multiplier"],
        ),
    }
    asset_diagnostics = _asset_diagnostics(groupings["symbol"])
    return {
        "generated_at": utc_now(),
        "report_type": "crypto_multi_global_summary",
        "top_limit": int(top_limit),
        "guardrails": {
            "research_only": True,
            "no_trading": True,
            "no_paper_trading": True,
            "models_unchanged": True,
            "features_unchanged": True,
            "thresholds_unchanged": True,
            "costs_unchanged": True,
            "test_not_used_for_selection": True,
            "accuracy_not_used": True,
        },
        "summary": {
            "total_configs": len(records),
            "completed": len(completed),
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
            "median_validation_pf": [_row_summary(row) for row in _top_rows(records, "median_validation_pf", limit=top_limit)],
            "median_validation_avg_return": [
                _row_summary(row)
                for row in _top_rows(records, "median_validation_avg_return", limit=top_limit)
            ],
            "median_test_pf_diagnostic_only": [
                _row_summary(row)
                for row in _top_rows(records, "median_test_pf", limit=top_limit)
            ],
        },
        "groupings": groupings,
        "best_symbol_horizon_rr_atr": _sorted_groups_by_metric(
            groupings["symbol_horizon_rr_atr"],
            "median_validation_pf",
            limit=top_limit,
        ),
        "asset_diagnostics": asset_diagnostics,
        "stability_analysis": summarize_group(records),
        "conclusion": _crypto_conclusion(counts, asset_diagnostics),
        "records": [_row_summary(row) for row in records],
    }


def _markdown_row(row: dict[str, Any]) -> str:
    config = row.get("config") or {}
    return (
        f"- `{row.get('classification')}: {_config_label(config)}` "
        f"val PF `{row.get('median_validation_pf')}`, "
        f"val avg `{row.get('median_validation_avg_return')}`, "
        f"test PF `{row.get('median_test_pf')}`"
    )


def _markdown_group_table(groups: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        "| Group | Count | Classes | Median Val PF | Median Val Avg | Avg Val+ | Avg Beats Random | Avg Beats Det | Avg Test Confirm |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, value in groups.items():
        lines.append(
            f"| {key} | {value.get('count')} | `{value.get('classification_counts')}` | "
            f"{value.get('median_validation_pf')} | {value.get('median_validation_avg_return')} | "
            f"{value.get('avg_validation_positive_rate')} | {value.get('avg_beats_random_rate')} | "
            f"{value.get('avg_beats_deterministic_rate')} | {value.get('avg_test_confirm_rate')} |"
        )
    return lines


def render_crypto_multi_markdown(summary: dict[str, Any]) -> str:
    overview = summary["summary"]
    top_limit = summary.get("top_limit", 15)
    lines = [
        "# Crypto Multi-Asset Global Summary",
        "",
        f"Generated at: `{summary.get('generated_at')}`",
        "",
        "## Guardrails",
        "",
        "- Research only. No trading signal.",
        "- No models, features, thresholds, costs, multi-window validation, or classification logic changed.",
        "- Validation selects; test metrics are diagnostic only and not selectable.",
        "- Accuracy is not used.",
        "",
        "## General Summary",
        "",
        f"- total configs: `{overview.get('total_configs')}`",
        f"- completed: `{overview.get('completed')}`",
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
    lines.extend(["", "## Asset Diagnostics", ""])
    lines.append("| Symbol | Count | Classes | Watchlist | Stable | Median Val PF | Median Val Avg | Promising | Discardable |")
    lines.append("| --- | ---: | --- | ---: | ---: | ---: | ---: | --- | --- |")
    for row in summary["asset_diagnostics"]["assets_by_watchlist_count"]:
        lines.append(
            f"| {row.get('symbol')} | {row.get('count')} | `{row.get('classification_counts')}` | "
            f"{row.get('unstable_watchlist')} | {row.get('stable_research_candidate')} | "
            f"{row.get('median_validation_pf')} | {row.get('median_validation_avg_return')} | "
            f"{row.get('promising')} | {row.get('discardable')} |"
        )
    lines.extend(["", f"## Top {top_limit} By Median Validation PF", ""])
    lines.extend(_markdown_row(row) for row in summary["top"]["median_validation_pf"])
    lines.extend(["", f"## Top {top_limit} By Median Validation Avg Return", ""])
    lines.extend(_markdown_row(row) for row in summary["top"]["median_validation_avg_return"])
    lines.extend(["", f"## Top {top_limit} By Median Test PF (Diagnostic Only, Not Selectable)", ""])
    lines.extend(_markdown_row(row) for row in summary["top"]["median_test_pf_diagnostic_only"])
    lines.extend(["", "## Grouping By Symbol", ""])
    lines.extend(_markdown_group_table(summary["groupings"]["symbol"]))
    lines.extend(["", "## Grouping By Horizon", ""])
    lines.extend(_markdown_group_table(summary["groupings"]["horizon_candles"]))
    lines.extend(["", "## Grouping By Risk Reward", ""])
    lines.extend(_markdown_group_table(summary["groupings"]["risk_reward"]))
    lines.extend(["", "## Grouping By ATR Stop Multiplier", ""])
    lines.extend(_markdown_group_table(summary["groupings"]["atr_stop_multiplier"]))
    lines.extend(["", "## Best Symbol + Horizon + RR + ATR Combinations", ""])
    lines.append("| Group | Count | Classes | Median Val PF | Median Val Avg |")
    lines.append("| --- | ---: | --- | ---: | ---: |")
    for row in summary["best_symbol_horizon_rr_atr"]:
        lines.append(
            f"| {row.get('group')} | {row.get('count')} | `{row.get('classification_counts')}` | "
            f"{row.get('median_validation_pf')} | {row.get('median_validation_avg_return')} |"
        )
    return "\n".join(lines).rstrip() + "\n"


def write_crypto_multi_summary(summary: dict[str, Any], output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Path]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    stamp = utc_stamp()
    json_path = target / f"crypto_multi_global_summary_{stamp}.json"
    markdown_path = target / f"crypto_multi_global_summary_{stamp}.md"
    summary["json_path"] = str(json_path)
    summary["markdown_path"] = str(markdown_path)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(render_crypto_multi_markdown(summary), encoding="utf-8")
    return {"json_path": json_path, "markdown_path": markdown_path}


def summarize_crypto_multi_registry(
    registry_path: str | Path = DEFAULT_CRYPTO_MULTI_REGISTRY,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    top_limit: int = 15,
) -> dict[str, Any]:
    records = load_latest_registry_records(registry_path)
    enriched = enrich_registry_records(records)
    summary = build_crypto_multi_summary(enriched, top_limit=top_limit)
    paths = write_crypto_multi_summary(summary, output_dir=output_dir)
    summary["json_path"] = str(paths["json_path"])
    summary["markdown_path"] = str(paths["markdown_path"])
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize Crypto Multi-Asset Research Grid v1.")
    parser.add_argument("--registry", default=str(DEFAULT_CRYPTO_MULTI_REGISTRY))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--top-limit", type=int, default=15)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = summarize_crypto_multi_registry(
        registry_path=args.registry,
        output_dir=args.output_dir,
        top_limit=args.top_limit,
    )
    overview = summary["summary"]
    print("Crypto multi-asset global summary generated")
    print("Research only. No trading signal.")
    print(f"total_configs: {overview['total_configs']}")
    print(f"completed: {overview['completed']}")
    print(f"classification_counts: {overview['classification_counts']}")
    print(f"promising_assets: {summary['asset_diagnostics']['promising_assets']}")
    print(f"discardable_assets: {summary['asset_diagnostics']['discardable_assets']}")
    print(f"json: {summary['json_path']}")
    print(f"markdown: {summary['markdown_path']}")


if __name__ == "__main__":
    main()
