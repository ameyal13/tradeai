"""Run local multi-window validation for Research Autopilot watchlist setups.

This script is research-only. It does not trade, paper trade, write Supabase, or
generate operational signals.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.multi_window_validator import DEFAULT_OUTPUT_DIR, run_multi_window_validation  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run multi-window validation for fixed watchlist setups.")
    parser.add_argument("--symbol", default="SOL")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--window-size-candles", type=int, default=600)
    parser.add_argument("--step-size-candles", type=int, default=250)
    parser.add_argument("--max-candles", type=int, default=1500)
    parser.add_argument("--refresh-cache", action="store_true", default=False)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser


async def main() -> None:
    args = build_parser().parse_args()
    summary = await run_multi_window_validation(
        symbol=args.symbol,
        timeframe=args.timeframe,
        max_candles=args.max_candles,
        window_size_candles=args.window_size_candles,
        step_size_candles=args.step_size_candles,
        refresh_cache=args.refresh_cache,
        output_dir=args.output_dir,
    )
    print("Multi-window validation finished")
    print(f"symbol: {summary['data']['symbol']}")
    print(f"timeframe: {summary['data']['timeframe']}")
    print(f"requested_max_candles: {summary['data']['requested_max_candles']}")
    print(f"actual_rows_loaded: {summary['data']['actual_rows_loaded']}")
    print(f"data_source: {summary['data']['data_source']}")
    print(f"data_cache_path: {summary['data']['data_cache_path']}")
    if summary["data"].get("data_warning"):
        print(f"data_warning: {summary['data']['data_warning']}")
    print(f"classification_counts: {summary['classification_counts']}")
    for setup_result in summary["setups"]:
        setup = setup_result["setup"]
        aggregate = setup_result["aggregate"]
        print(
            f"- {setup['symbol']} {setup['timeframe']} h{setup['horizon_candles']} "
            f"RR{setup['risk_reward']} ATR{setup['atr_stop_multiplier']} {setup['cost_mode']}: "
            f"{setup_result['classification']} | valid_windows "
            f"{aggregate['valid_windows']}/{aggregate['total_windows']} | "
            f"median_val_pf {aggregate['median_validation_pf']} | "
            f"median_val_avg {aggregate['median_validation_avg_return']}"
        )
    print(f"json: {summary['json_path']}")
    print(f"markdown: {summary['markdown_path']}")


if __name__ == "__main__":
    asyncio.run(main())
