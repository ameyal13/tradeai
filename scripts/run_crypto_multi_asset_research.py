"""Run Crypto Multi-Asset Research Grid v1 locally.

Research only: no trading, no paper trading, no Supabase, no endpoints, no
shadow journal writes, and no operational signals.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.crypto_multi_asset_grid import build_crypto_multi_asset_grid  # noqa: E402
from research.research_daemon import DEFAULT_DAEMON_DIR, run_research_cycle  # noqa: E402
from scripts.run_research_daemon import print_progress  # noqa: E402


CRYPTO_MULTI_REGISTRY_PATH = DEFAULT_DAEMON_DIR / "crypto_multi_registry.jsonl"
CRYPTO_MULTI_CYCLES_DIR = DEFAULT_DAEMON_DIR / "crypto_multi_cycles"
CRYPTO_MULTI_RESULTS_DIR = DEFAULT_DAEMON_DIR / "crypto_multi_results"
CRYPTO_MULTI_STATUS_PATH = DEFAULT_DAEMON_DIR / "crypto_multi_current_status.json"


def parse_symbols(raw: list[str] | None) -> list[str] | None:
    if not raw:
        return None
    symbols: list[str] = []
    for item in raw:
        symbols.extend(part.strip() for part in str(item).replace(",", " ").split() if part.strip())
    return symbols or None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Crypto Multi-Asset Research Grid v1.")
    parser.add_argument("--once", action="store_true", default=True, help="Run one crypto multi-asset research cycle.")
    parser.add_argument("--max-configs-per-cycle", type=int, default=5)
    parser.add_argument("--symbols", nargs="*", default=None, help="Crypto symbols, e.g. BTC,ETH,SOL or BTC ETH SOL.")
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", action="store_true", dest="resume", default=True)
    resume_group.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--retry-failed", action="store_true", default=False)
    parser.add_argument("--notify-telegram", action="store_true", default=False)
    parser.add_argument("--progress", action="store_true", default=True)
    parser.add_argument("--quiet", action="store_true", default=False)
    return parser


async def main() -> None:
    args = build_parser().parse_args()
    grid = build_crypto_multi_asset_grid(symbols=parse_symbols(args.symbols))
    result = await run_research_cycle(
        grid=grid,
        max_configs_per_cycle=args.max_configs_per_cycle,
        resume=args.resume,
        retry_failed=args.retry_failed,
        notify_telegram=args.notify_telegram,
        registry_path=CRYPTO_MULTI_REGISTRY_PATH,
        cycles_dir=CRYPTO_MULTI_CYCLES_DIR,
        results_dir=CRYPTO_MULTI_RESULTS_DIR,
        status_path=CRYPTO_MULTI_STATUS_PATH,
        progress_callback=print_progress if args.progress and not args.quiet else None,
    )
    summary = result["summary"]
    print("Crypto Multi-Asset Research Grid v1 cycle finished")
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
    print(f"registry: {CRYPTO_MULTI_REGISTRY_PATH}")
    print(f"status: {CRYPTO_MULTI_STATUS_PATH}")
    print(f"json: {result['json_path']}")
    print(f"markdown: {result['markdown_path']}")


if __name__ == "__main__":
    asyncio.run(main())
