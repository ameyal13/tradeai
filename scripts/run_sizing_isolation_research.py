"""Run Sizing Isolation Research v1 locally.

Research only: no trading, no paper trading, no Supabase writes, no endpoints,
and no operational signals. This phase freezes baseline XGBoost features and
varies only horizon, risk/reward, and ATR stop multiplier.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.research_daemon import DEFAULT_DAEMON_DIR, run_research_cycle  # noqa: E402
from research.sizing_isolation_grid import SIZING_HORIZON_CANDLES, build_sizing_isolation_grid  # noqa: E402
from scripts.run_crypto_multi_asset_research import parse_symbols, print_running_records, running_records  # noqa: E402
from scripts.run_research_daemon import print_progress  # noqa: E402


SIZING_REGISTRY_PATH = DEFAULT_DAEMON_DIR / "sizing_isolation_v1_registry.jsonl"
SIZING_CYCLES_DIR = DEFAULT_DAEMON_DIR / "sizing_isolation_v1_cycles"
SIZING_RESULTS_DIR = DEFAULT_DAEMON_DIR / "sizing_isolation_v1_results"
SIZING_STATUS_PATH = DEFAULT_DAEMON_DIR / "sizing_isolation_v1_current_status.json"


def parse_horizons(values: list[str] | None) -> list[int] | None:
    if not values:
        return None
    parsed: list[int] = []
    for value in values:
        for part in str(value).split(","):
            if part.strip():
                parsed.append(int(part.strip()))
    return parsed or None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Sizing Isolation Research v1.")
    parser.add_argument("--once", action="store_true", default=True)
    parser.add_argument("--max-configs-per-cycle", type=int, default=5)
    parser.add_argument("--symbols", nargs="*", default=None, help="Subset of ADA, ETH.")
    parser.add_argument("--horizons", nargs="*", default=None, help=f"Subset of {SIZING_HORIZON_CANDLES}.")
    parser.add_argument("--quick-ada", action="store_true", default=False, help="Shortcut: ADA only, horizons 10/12/14.")
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", action="store_true", dest="resume", default=True)
    resume_group.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--retry-failed", action="store_true", default=False)
    parser.add_argument("--retry-running", action="store_true", default=False)
    parser.add_argument("--list-running", action="store_true", default=False)
    parser.add_argument("--notify-telegram", action="store_true", default=False)
    parser.add_argument("--progress", action="store_true", default=True)
    parser.add_argument("--quiet", action="store_true", default=False)
    return parser


async def main() -> None:
    args = build_parser().parse_args()
    symbols = ["ADA"] if args.quick_ada else parse_symbols(args.symbols)
    horizons = [10, 12, 14] if args.quick_ada else parse_horizons(args.horizons)
    if args.list_running:
        print_running_records(running_records(SIZING_REGISTRY_PATH, symbols=symbols))
        return
    grid = build_sizing_isolation_grid(symbols=symbols, horizons=horizons)
    result = await run_research_cycle(
        grid=grid,
        max_configs_per_cycle=args.max_configs_per_cycle,
        resume=args.resume,
        retry_failed=args.retry_failed,
        retry_running=args.retry_running,
        notify_telegram=args.notify_telegram,
        registry_path=SIZING_REGISTRY_PATH,
        cycles_dir=SIZING_CYCLES_DIR,
        results_dir=SIZING_RESULTS_DIR,
        status_path=SIZING_STATUS_PATH,
        progress_callback=print_progress if args.progress and not args.quiet else None,
    )
    summary = result["summary"]
    print("Sizing Isolation Research v1 cycle finished")
    print("Research only. No trading signal.")
    print(f"interrupted: {result['interrupted']}")
    print(f"grid_size: {result['grid_size']}")
    print(f"runnable_before_cycle: {result['runnable_before_cycle']}")
    print(f"selected_configs: {result['selected_configs']}")
    print(f"evaluated: {summary['evaluated']}")
    print(f"classification_counts: {summary['classification_counts']}")
    print(f"stable_candidates: {len(summary['stable_candidates'])}")
    print(f"unstable_watchlist: {len(summary['unstable_watchlist'])}")
    if args.notify_telegram:
        print(f"telegram_sent: {result.get('telegram_sent', False)}")
    print(f"registry: {SIZING_REGISTRY_PATH}")
    print(f"status: {SIZING_STATUS_PATH}")
    print(f"json: {result['json_path']}")
    print(f"markdown: {result['markdown_path']}")


if __name__ == "__main__":
    asyncio.run(main())
