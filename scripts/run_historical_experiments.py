"""Run historical replay experiments across symbols, timeframes, and strategies.

This script is for offline/historical research. It does not place trades and
does not persist to Supabase unless --persist is explicitly passed.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.historical_data import fetch_binance_klines
from tools.historical_replay import run_historical_replay
from tools.prediction_journal import PredictionStore


DEFAULT_SYMBOLS = ["BTC", "ETH", "SOL"]
DEFAULT_TIMEFRAMES = ["15m", "1h"]
DEFAULT_STRATEGY_MODES = ["deterministic"]
DEFAULT_HORIZON_MINUTES = 60
DEFAULT_MAX_CANDLES = 500
DEFAULT_MAX_PREDICTIONS = 100


def interval_to_minutes(interval: str) -> int:
    unit = interval[-1].lower()
    value = int(interval[:-1])
    if unit == "m":
        return value
    if unit == "h":
        return value * 60
    if unit == "d":
        return value * 1440
    raise ValueError(f"Unsupported interval: {interval}")


def horizon_candles_for_interval(horizon_minutes: int, interval: str) -> int:
    interval_minutes = interval_to_minutes(interval)
    return max(1, int((horizon_minutes + interval_minutes - 1) / interval_minutes))


def first_metric(result: dict[str, Any]) -> dict[str, Any]:
    metrics = result.get("metrics") or []
    return metrics[0] if metrics else {}


def summarize_run(
    symbol: str,
    timeframe: str,
    strategy_mode: str,
    result: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    metric = first_metric(result or {})
    outcomes = (result or {}).get("outcomes") or []
    invalid_count = sum(1 for outcome in outcomes if outcome.get("outcome") == "INVALID_DATA")
    warnings = []
    assumptions = (result or {}).get("assumptions") or {}
    if assumptions.get("error"):
        warnings.append(str(assumptions["error"]))
    if error:
        warnings.append(error)

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "strategy_mode": strategy_mode,
        "total_predictions": len((result or {}).get("predictions") or []),
        "evaluated_predictions": metric.get("evaluated_predictions", len(outcomes)),
        "win_rate": metric.get("win_rate", 0),
        "average_return": metric.get("average_return", 0),
        "total_return_pct": metric.get("total_return_pct", 0),
        "profit_factor": metric.get("profit_factor", 0),
        "max_drawdown": metric.get("max_drawdown", 0),
        "sharpe": metric.get("sharpe", 0),
        "invalid_count": invalid_count,
        "warnings": "; ".join(warnings),
    }


async def run_experiments(
    symbols: list[str],
    timeframes: list[str],
    strategy_modes: list[str],
    horizon_minutes: int = DEFAULT_HORIZON_MINUTES,
    max_candles: int = DEFAULT_MAX_CANDLES,
    max_predictions: int = DEFAULT_MAX_PREDICTIONS,
    persist: bool = False,
    reports_dir: str | Path = "reports",
) -> dict[str, Any]:
    store = PredictionStore() if persist else None
    summaries: list[dict[str, Any]] = []
    raw_runs: list[dict[str, Any]] = []

    for symbol in symbols:
        for timeframe in timeframes:
            candles = None
            try:
                candles = await fetch_binance_klines(symbol, timeframe, limit=max_candles)
            except Exception as exc:
                error = f"historical_data_error: {exc}"
                for strategy_mode in strategy_modes:
                    summaries.append(summarize_run(symbol, timeframe, strategy_mode, error=error))
                continue

            for strategy_mode in strategy_modes:
                try:
                    result = run_historical_replay(
                        candles,
                        symbol=symbol,
                        timeframe=timeframe,
                        strategy_mode=strategy_mode,
                        horizon_candles=horizon_candles_for_interval(horizon_minutes, timeframe),
                        horizon_minutes=horizon_minutes,
                        max_predictions=max_predictions,
                        store=store,
                    )
                    summaries.append(summarize_run(symbol, timeframe, strategy_mode, result=result))
                    raw_runs.append({
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "strategy_mode": strategy_mode,
                        "assumptions": result.get("assumptions", {}),
                        "metrics": result.get("metrics", []),
                    })
                except Exception as exc:
                    summaries.append(summarize_run(symbol, timeframe, strategy_mode, error=f"replay_error: {exc}"))

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "symbols": symbols,
            "timeframes": timeframes,
            "strategy_modes": strategy_modes,
            "horizon_minutes": horizon_minutes,
            "max_candles": max_candles,
            "max_predictions_per_run": max_predictions,
            "persist": persist,
        },
        "summary": summaries,
        "runs": raw_runs,
    }
    paths = write_report(report, reports_dir)
    report["report_paths"] = paths
    return report


def write_report(report: dict[str, Any], reports_dir: str | Path = "reports") -> dict[str, str]:
    target = Path(reports_dir)
    target.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = target / f"historical_experiments_{stamp}.json"
    csv_path = target / f"historical_experiments_{stamp}.csv"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    rows = report["summary"]
    fieldnames = [
        "symbol", "timeframe", "strategy_mode", "total_predictions",
        "evaluated_predictions", "win_rate", "average_return",
        "total_return_pct", "profit_factor", "max_drawdown", "sharpe",
        "invalid_count", "warnings",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return {"json": str(json_path), "csv": str(csv_path)}


def print_summary(rows: list[dict[str, Any]]) -> None:
    headers = [
        "symbol", "timeframe", "strategy_mode", "total_predictions",
        "evaluated_predictions", "win_rate", "average_return",
        "total_return_pct", "profit_factor", "max_drawdown", "sharpe",
        "invalid_count", "warnings",
    ]
    widths = {header: max(len(header), *(len(str(row.get(header, ""))) for row in rows)) for header in headers}
    print(" | ".join(header.ljust(widths[header]) for header in headers))
    print("-+-".join("-" * widths[header] for header in headers))
    for row in rows:
        print(" | ".join(str(row.get(header, "")).ljust(widths[header]) for header in headers))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run historical replay experiments.")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--timeframes", nargs="+", default=DEFAULT_TIMEFRAMES)
    parser.add_argument("--strategy-modes", nargs="+", default=DEFAULT_STRATEGY_MODES)
    parser.add_argument("--max-candles", type=int, default=DEFAULT_MAX_CANDLES)
    parser.add_argument("--max-predictions", type=int, default=DEFAULT_MAX_PREDICTIONS)
    parser.add_argument("--horizon-minutes", type=int, default=DEFAULT_HORIZON_MINUTES)
    parser.add_argument("--persist", action="store_true")
    return parser


async def main() -> None:
    args = build_parser().parse_args()
    report = await run_experiments(
        symbols=[symbol.upper() for symbol in args.symbols],
        timeframes=args.timeframes,
        strategy_modes=args.strategy_modes,
        horizon_minutes=args.horizon_minutes,
        max_candles=args.max_candles,
        max_predictions=args.max_predictions,
        persist=args.persist,
    )
    print_summary(report["summary"])
    print(f"JSON report: {report['report_paths']['json']}")
    print(f"CSV report: {report['report_paths']['csv']}")


if __name__ == "__main__":
    asyncio.run(main())
