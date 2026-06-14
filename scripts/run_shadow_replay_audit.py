"""Run an offline replay of the current shadow-ops policy.

Research only. This script does not write the live shadow journal, does not send
Telegram, and does not place exchange orders.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.asset_universe import normalize_crypto_symbols  # noqa: E402
from research.shadow_replay import (  # noqa: E402
    combine_replay_reports,
    load_shadow_replay_configs,
    run_shadow_replay_for_candles,
    save_shadow_replay_report,
)
from scripts.run_historical_experiments import load_experiment_candles  # noqa: E402
from tools.runtime_env import load_project_env  # noqa: E402


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "shadow_replay"
DEFAULT_STATUS_PATH = DEFAULT_OUTPUT_DIR / "current_status.json"


def _write_status(path: str | Path, status: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(status, indent=2, sort_keys=True, default=str), encoding="utf-8")


async def run_shadow_replay_audit(
    *,
    registry: str = "crypto_multi",
    symbols: list[str] | None = None,
    timeframe: str = "1h",
    days: int = 60,
    max_candles: int = 2000,
    max_signals: int = 1,
    max_configs_scanned: int = 21,
    cycle_step_candles: int = 1,
    min_history_candles: int = 300,
    max_cycles: int | None = None,
    use_sentiment: bool = False,
    refresh_cache: bool = False,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    write_report: bool = True,
    progress: bool = True,
    quiet: bool = False,
    progress_every_cycles: int = 5,
    status_path: str | Path | None = DEFAULT_STATUS_PATH,
    max_runtime_seconds: int | None = None,
) -> dict[str, Any]:
    selected_symbols = normalize_crypto_symbols(symbols or ["ADA", "ETH", "SOL"])
    configs = load_shadow_replay_configs(registry=registry, symbols=selected_symbols, allow_watchlist_shadow=True)
    runs: list[dict[str, Any]] = []
    data_warnings: list[str] = []
    started_at = time.time()
    started_iso = datetime.now(timezone.utc).isoformat()

    def make_progress_callback(symbol: str, symbol_index: int):
        last_printed_cycle = 0

        def callback(update: dict[str, Any]) -> None:
            nonlocal last_printed_cycle
            cycle = int(update.get("cycle") or 0)
            total_cycles = int(update.get("total_cycles") or 0)
            summary = update.get("summary") or {}
            status = {
                "research_only": True,
                "cycle_started_at": started_iso,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "symbols": selected_symbols,
                "current_symbol": symbol,
                "current_symbol_index": symbol_index,
                "total_symbols": len(selected_symbols),
                "completed_symbols": [run.get("symbol") for run in runs],
                "current_progress": update,
                "elapsed_seconds": round(time.time() - started_at, 3),
                "runs_completed": len(runs),
            }
            if status_path:
                _write_status(status_path, status)
            should_print = (
                progress
                and not quiet
                and (
                    cycle == 1
                    or cycle >= total_cycles
                    or cycle - last_printed_cycle >= max(1, int(progress_every_cycles))
                    or update.get("stop_reason")
                )
            )
            if should_print:
                last_printed_cycle = cycle
                print(
                    f"[shadow-replay] {symbol} {cycle}/{total_cycles} "
                    f"signals={summary.get('total')} closed={summary.get('closed')} "
                    f"wins={summary.get('wins')} losses={summary.get('losses')} "
                    f"pf={summary.get('profit_factor')} avg={summary.get('avg_return')} "
                    f"last={update.get('last_event_status')} elapsed={status['elapsed_seconds']}s",
                    flush=True,
                )

        return callback

    for symbol in selected_symbols:
        symbol_index = len(runs) + 1
        if progress and not quiet:
            print(f"[shadow-replay] loading {symbol} {timeframe} candles...", flush=True)
        loaded = await load_experiment_candles(
            symbol,
            timeframe,
            max_candles=max_candles,
            use_cache=True,
            refresh_cache=refresh_cache,
        )
        warning = loaded.get("data_warning")
        if warning:
            data_warnings.append(f"{symbol} {timeframe}: {warning}")
        candles = loaded["candles"]
        run = run_shadow_replay_for_candles(
            candles=candles,
            configs=configs,
            symbol=symbol,
            timeframe=timeframe,
            days=days,
            max_signals=max_signals,
            max_configs_scanned=max_configs_scanned,
            cycle_step_candles=cycle_step_candles,
            min_history_candles=min_history_candles,
            use_sentiment=use_sentiment,
            max_cycles=max_cycles,
            progress_callback=make_progress_callback(symbol, symbol_index),
            max_runtime_seconds=max_runtime_seconds,
        )
        run["data_source"] = loaded.get("data_source")
        run["data_cache_path"] = loaded.get("data_cache_path")
        run["data_warning"] = warning
        runs.append(run)

    report = {
        "config": {
            "registry": registry,
            "symbols": selected_symbols,
            "timeframe": timeframe,
            "days": days,
            "max_candles": max_candles,
            "max_signals": max_signals,
            "max_configs_scanned": max_configs_scanned,
            "cycle_step_candles": cycle_step_candles,
            "min_history_candles": min_history_candles,
            "max_cycles": max_cycles,
            "use_sentiment": use_sentiment,
            "refresh_cache": refresh_cache,
            "progress": progress,
            "quiet": quiet,
            "progress_every_cycles": progress_every_cycles,
            "status_path": str(status_path) if status_path else None,
            "max_runtime_seconds": max_runtime_seconds,
            "research_only": True,
            "no_live_journal_writes": True,
            "no_exchange_orders": True,
        },
        "data_warnings": data_warnings,
        "runs": runs,
        "combined": combine_replay_reports(runs),
    }
    if write_report:
        report.update(save_shadow_replay_report(report, output_dir))
    if status_path:
        _write_status(status_path, {
            "research_only": True,
            "cycle_started_at": started_iso,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "symbols": selected_symbols,
            "completed": True,
            "elapsed_seconds": round(time.time() - started_at, 3),
            "summary": report["combined"]["summary"],
            "event_status_counts": report["combined"].get("event_status_counts", {}),
            "json_path": report.get("json_path"),
            "markdown_path": report.get("markdown_path"),
        })
    return report


def print_replay_summary(report: dict[str, Any]) -> None:
    summary = report["combined"]["summary"]
    events = report["combined"].get("event_status_counts", {})
    print("Shadow Replay Audit")
    print("Research only. No trading signal. No exchange orders.")
    print(f"symbols: {', '.join(report['config']['symbols'])}")
    print(f"timeframe: {report['config']['timeframe']}")
    print(f"days: {report['config']['days']}")
    print(f"cycles: {sum(run.get('cycles', 0) for run in report.get('runs', []))}")
    print(f"signals total/open/closed: {summary['total']}/{summary['open']}/{summary['closed']}")
    print(f"wins/losses/expired: {summary['wins']}/{summary['losses']}/{summary['expired']}")
    print(f"win_rate: {summary['win_rate']}")
    print(f"profit_factor: {summary['profit_factor']}")
    print(f"avg_return: {summary['avg_return']}")
    print(f"total_return_pct: {summary['total_return_pct']}")
    print(f"max_drawdown: {summary['max_drawdown']}")
    print(f"event_status_counts: {events}")
    if report.get("markdown_path"):
        print(f"json: {report['json_path']}")
        print(f"markdown: {report['markdown_path']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run offline shadow replay audit.")
    parser.add_argument("--registry", default="crypto_multi")
    parser.add_argument("--symbols", nargs="*", default=["ADA", "ETH", "SOL"])
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--max-candles", type=int, default=2000)
    parser.add_argument("--max-signals", type=int, default=1)
    parser.add_argument("--max-configs-scanned", type=int, default=21)
    parser.add_argument("--cycle-step-candles", type=int, default=1)
    parser.add_argument("--min-history-candles", type=int, default=300)
    parser.add_argument("--max-cycles", type=int, default=None)
    parser.add_argument("--use-sentiment", action="store_true", help="Disabled by default to avoid current sentiment leakage.")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--no-write-report", action="store_false", dest="write_report", default=True)
    parser.add_argument("--progress", action="store_true", dest="progress", default=True)
    parser.add_argument("--quiet", action="store_true", help="Suppress progress lines; final summary still prints.")
    parser.add_argument("--progress-every-cycles", type=int, default=5)
    parser.add_argument("--status-path", default=str(DEFAULT_STATUS_PATH))
    parser.add_argument("--no-status-file", action="store_const", const=None, dest="status_path")
    parser.add_argument("--max-runtime-seconds", type=int, default=None)
    return parser


async def main() -> None:
    load_project_env()
    args = build_parser().parse_args()
    report = await run_shadow_replay_audit(
        registry=args.registry,
        symbols=args.symbols,
        timeframe=args.timeframe,
        days=args.days,
        max_candles=args.max_candles,
        max_signals=args.max_signals,
        max_configs_scanned=args.max_configs_scanned,
        cycle_step_candles=args.cycle_step_candles,
        min_history_candles=args.min_history_candles,
        max_cycles=args.max_cycles,
        use_sentiment=args.use_sentiment,
        refresh_cache=args.refresh_cache,
        output_dir=args.output_dir,
        write_report=args.write_report,
        progress=args.progress,
        quiet=args.quiet,
        progress_every_cycles=args.progress_every_cycles,
        status_path=args.status_path,
        max_runtime_seconds=args.max_runtime_seconds,
    )
    print_replay_summary(report)


if __name__ == "__main__":
    asyncio.run(main())
