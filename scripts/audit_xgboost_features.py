"""Audit whether current XGBoost features contain out-of-sample signal.

This is an offline research script. It does not persist to Supabase, does not
place trades, and does not change strategy thresholds.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_historical_experiments import (  # noqa: E402
    DEFAULT_CACHE_DIR,
    effective_horizon_minutes,
    horizon_candles_for_interval,
    load_experiment_candles,
)
from tools.feature_research import run_feature_audit  # noqa: E402
from tools.historical_replay import run_historical_replay  # noqa: E402


def first_metric(result: dict[str, Any]) -> dict[str, Any]:
    metrics = result.get("metrics") or []
    return metrics[0] if metrics else {}


def deterministic_baseline(
    candles,
    symbol: str,
    timeframe: str,
    horizon_candles: int,
    horizon_minutes: int,
    max_predictions: int,
) -> dict[str, Any]:
    result = run_historical_replay(
        candles,
        symbol=symbol,
        timeframe=timeframe,
        strategy_mode="deterministic",
        horizon_candles=horizon_candles,
        horizon_minutes=horizon_minutes,
        max_predictions=max_predictions,
        min_history=50,
        strategy_params={},
    )
    metric = first_metric(result)
    return {
        "total_predictions": len(result.get("predictions") or []),
        "evaluated_predictions": metric.get("evaluated_predictions", len(result.get("outcomes") or [])),
        "win_rate": metric.get("win_rate", 0),
        "average_return": metric.get("average_return", 0),
        "total_return_pct": metric.get("total_return_pct", 0),
        "profit_factor": metric.get("profit_factor", 0),
        "max_drawdown": metric.get("max_drawdown", 0),
        "sharpe": metric.get("sharpe", 0),
    }


def classify_feature_audit(audit: dict[str, Any]) -> str:
    all_current = audit.get("ablation_results", {}).get("all_current", {})
    random_baseline = audit.get("ablation_results", {}).get("dummy_random", {})
    if all_current.get("status") != "ok":
        return "insufficient_data"
    if all_current.get("trades", 0) < 30:
        return "insufficient_trades"
    improves_random = (
        all_current.get("average_return", 0) > random_baseline.get("average_return", 0)
        and all_current.get("profit_factor", 0) > random_baseline.get("profit_factor", 0)
    )
    if all_current.get("average_return", 0) > 0 and all_current.get("profit_factor", 0) > 1.1 and improves_random:
        return "signal_candidate"
    if all_current.get("average_return", 0) > -0.05 and improves_random:
        return "weak_signal_candidate"
    return "no_feature_edge_detected"


def top_items(values: dict[str, float], limit: int = 5, reverse: bool = True) -> list[tuple[str, float]]:
    return sorted(values.items(), key=lambda item: item[1], reverse=reverse)[:limit]


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# XGBoost Feature Audit",
        "",
        f"Generated at: `{report['generated_at']}`",
        "",
        "## Method",
        "",
        "- Purged walk-forward validation by temporal blocks.",
        "- The purge equals `horizon_candles`, so train labels do not consume validation candles.",
        "- XGBoost is compared against deterministic baseline and dummy random probabilities.",
        "- This report audits feature signal; it is not a trading recommendation.",
        "",
        "## Summary",
        "",
    ]
    for row in report["results"]:
        if row.get("error"):
            lines.extend([
                f"### {row['symbol']} {row['timeframe']}: DATA ERROR",
                "",
                f"- Error: `{row['error']}`",
                "",
            ])
            continue
        audit = row["audit"]
        all_current = audit["ablation_results"].get("all_current", {})
        dummy = audit["ablation_results"].get("dummy_random", {})
        deterministic = row["deterministic_baseline"]
        lines.extend([
            f"### {row['symbol']} {row['timeframe']}: {row['classification']}",
            "",
            f"- Data source: `{row['data_source']}`",
            f"- Horizon candles: `{row['horizon_candles']}`",
            f"- Raw BUY labels: `{audit['raw_buy_label_count']}` / positives `{audit['raw_buy_positive_count']}`",
            f"- Raw SELL labels: `{audit['raw_sell_label_count']}` / positives `{audit['raw_sell_positive_count']}`",
            f"- XGBoost all_current: trades `{all_current.get('trades', 0)}`, avg_return `{all_current.get('average_return', 0)}`, PF `{all_current.get('profit_factor', 0)}`",
            f"- Dummy random: trades `{dummy.get('trades', 0)}`, avg_return `{dummy.get('average_return', 0)}`, PF `{dummy.get('profit_factor', 0)}`",
            f"- Deterministic baseline: evaluated `{deterministic.get('evaluated_predictions', 0)}`, avg_return `{deterministic.get('average_return', 0)}`, PF `{deterministic.get('profit_factor', 0)}`",
            f"- Removal candidates: `{', '.join(audit.get('removal_candidates') or []) or 'none'}`",
            "",
            "Feature importance, top:",
        ])
        for feature, value in top_items(all_current.get("model_importance", {})):
            corr = audit["feature_correlations_to_future_return"].get(feature)
            lines.append(f"- `{feature}` importance `{value}`, future-return corr `{corr}`")
        lines.extend(["", "Ablations:", ""])
        for name, result in audit["ablation_results"].items():
            lines.append(
                f"- `{name}`: status `{result.get('status')}`, trades `{result.get('trades', 0)}`, "
                f"avg_return `{result.get('average_return', 0)}`, PF `{result.get('profit_factor', 0)}`"
            )
        lines.extend(["", "Methodology notes:"])
        for note in audit["methodology"]["leakage_notes"]:
            lines.append(f"- {note}")
        lines.append("")
    lines.extend([
        "## Interpretation Rules",
        "",
        "- Signal candidate requires positive average return, profit factor above 1.1, and improvement over random baseline.",
        "- Weak candidate requires near-flat average return and improvement over random baseline.",
        "- If all feature families lose similarly, the current XGBoost feature set is likely learning noise.",
        "- Do not move to paper trading from this report unless results are stable across symbols, timeframes, and future windows.",
        "",
    ])
    return "\n".join(lines)


def write_report(report: dict[str, Any], reports_dir: str | Path) -> dict[str, str]:
    target = Path(reports_dir)
    target.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = target / f"xgboost_feature_audit_{stamp}.json"
    md_path = target / f"xgboost_feature_audit_{stamp}.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


async def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for symbol in [value.upper() for value in args.symbols]:
        for timeframe in args.timeframes:
            horizon_candles = horizon_candles_for_interval(args.horizon_minutes, timeframe)
            horizon_minutes = effective_horizon_minutes(args.horizon_minutes, timeframe)
            try:
                loaded = await load_experiment_candles(
                    symbol,
                    timeframe,
                    max_candles=args.max_candles,
                    use_cache=args.use_cache,
                    refresh_cache=args.refresh_cache,
                    cache_dir=args.cache_dir,
                )
                candles = loaded["candles"]
                audit = run_feature_audit(
                    candles,
                    horizon_candles=horizon_candles,
                    n_splits=args.splits,
                    min_train_rows=args.min_train_rows,
                    trade_label_scheme=args.trade_label_scheme,
                )
                deterministic = deterministic_baseline(
                    candles,
                    symbol=symbol,
                    timeframe=timeframe,
                    horizon_candles=horizon_candles,
                    horizon_minutes=horizon_minutes,
                    max_predictions=args.max_predictions,
                )
                results.append({
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "data_source": loaded.get("data_source", ""),
                    "data_cache_path": loaded.get("data_cache_path", ""),
                    "data_warning": loaded.get("data_warning", ""),
                    "horizon_candles": horizon_candles,
                    "effective_horizon_minutes": horizon_minutes,
                    "classification": classify_feature_audit(audit),
                    "audit": audit,
                    "deterministic_baseline": deterministic,
                })
            except Exception as exc:  # noqa: BLE001 - one symbol/timeframe should not stop the audit.
                results.append({
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "classification": "data_error",
                    "error": f"{exc.__class__.__name__}: {exc}",
                })
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "symbols": [value.upper() for value in args.symbols],
            "timeframes": args.timeframes,
            "horizon_minutes": args.horizon_minutes,
            "max_candles": args.max_candles,
            "max_predictions": args.max_predictions,
            "splits": args.splits,
            "min_train_rows": args.min_train_rows,
            "trade_label_scheme": args.trade_label_scheme,
            "use_cache": args.use_cache,
            "refresh_cache": args.refresh_cache,
            "cache_dir": str(args.cache_dir),
        },
        "results": results,
    }
    report["report_paths"] = write_report(report, args.reports_dir)
    return report


def print_summary(report: dict[str, Any]) -> None:
    print("symbol | timeframe | classification | xgb_trades | xgb_avg_return | xgb_pf | random_avg_return | deterministic_avg_return")
    print("-------+-----------+----------------+------------+----------------+--------+-------------------+-------------------------")
    for row in report["results"]:
        if row.get("error"):
            print(f"{row['symbol']} | {row['timeframe']} | data_error | 0 | 0 | 0 | 0 | 0")
            continue
        all_current = row["audit"]["ablation_results"].get("all_current", {})
        random_baseline = row["audit"]["ablation_results"].get("dummy_random", {})
        deterministic = row["deterministic_baseline"]
        print(
            f"{row['symbol']} | {row['timeframe']} | {row['classification']} | "
            f"{all_current.get('trades', 0)} | {all_current.get('average_return', 0)} | "
            f"{all_current.get('profit_factor', 0)} | {random_baseline.get('average_return', 0)} | "
            f"{deterministic.get('average_return', 0)}"
        )
    print(f"Markdown report: {report['report_paths']['markdown']}")
    print(f"JSON report: {report['report_paths']['json']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit XGBoost feature signal with purged walk-forward validation.")
    parser.add_argument("--symbols", nargs="+", default=["BTC", "ETH", "SOL"])
    parser.add_argument("--timeframes", nargs="+", default=["15m", "1h"])
    parser.add_argument("--max-candles", type=int, default=1500)
    parser.add_argument("--max-predictions", type=int, default=200)
    parser.add_argument("--horizon-minutes", type=int, default=60)
    parser.add_argument("--splits", type=int, default=4)
    parser.add_argument("--min-train-rows", type=int, default=120)
    parser.add_argument("--trade-label-scheme", default="expected_value_classification")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--no-cache", action="store_false", dest="use_cache", default=True)
    parser.add_argument("--refresh-cache", action="store_true")
    return parser


async def main() -> None:
    report = await run_audit(build_parser().parse_args())
    print_summary(report)


if __name__ == "__main__":
    asyncio.run(main())
