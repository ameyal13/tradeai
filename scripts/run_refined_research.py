"""Run Refined Research Grid 2A locally.

This script runs a focused SOL 1h low_costs grid. It is research-only: no
trading, no paper trading, no Supabase, no endpoints, and no operational
signals.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.refined_grid import build_refined_sol_1h_grid  # noqa: E402
from research.research_daemon import DEFAULT_DAEMON_DIR, run_research_cycle  # noqa: E402
from scripts.run_research_daemon import print_progress  # noqa: E402


REFINED_REGISTRY_PATH = DEFAULT_DAEMON_DIR / "refined_registry.jsonl"
REFINED_CYCLES_DIR = DEFAULT_DAEMON_DIR / "refined_cycles"
REFINED_RESULTS_DIR = DEFAULT_DAEMON_DIR / "refined_results"
REFINED_STATUS_PATH = DEFAULT_DAEMON_DIR / "refined_current_status.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Refined Research Grid 2A.")
    parser.add_argument("--once", action="store_true", default=True, help="Run one refined research cycle.")
    parser.add_argument("--max-configs-per-cycle", type=int, default=5)
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
    grid = build_refined_sol_1h_grid()
    result = await run_research_cycle(
        grid=grid,
        max_configs_per_cycle=args.max_configs_per_cycle,
        resume=args.resume,
        retry_failed=args.retry_failed,
        notify_telegram=args.notify_telegram,
        registry_path=REFINED_REGISTRY_PATH,
        cycles_dir=REFINED_CYCLES_DIR,
        results_dir=REFINED_RESULTS_DIR,
        status_path=REFINED_STATUS_PATH,
        progress_callback=print_progress if args.progress and not args.quiet else None,
    )
    summary = result["summary"]
    print("Refined Research Grid 2A cycle finished")
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
    print(f"registry: {REFINED_REGISTRY_PATH}")
    print(f"status: {REFINED_STATUS_PATH}")
    print(f"json: {result['json_path']}")
    print(f"markdown: {result['markdown_path']}")


if __name__ == "__main__":
    asyncio.run(main())
