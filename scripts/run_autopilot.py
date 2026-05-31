"""Run the local Research Autopilot.

This script is offline research only: no trading, no paper trading, no
Supabase writes, no scheduler, and no operational signals.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.autopilot import run_autopilot  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local Research Autopilot.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--resume", action="store_true", dest="resume", default=True)
    group.add_argument("--no-resume", action="store_false", dest="resume")
    notify_group = parser.add_mutually_exclusive_group()
    notify_group.add_argument("--notify-telegram", action="store_true", dest="notify_telegram")
    notify_group.add_argument("--no-notify-telegram", action="store_false", dest="notify_telegram")
    parser.set_defaults(notify_telegram=False)
    parser.add_argument("--max-experiments", type=int, default=None)
    return parser


async def main() -> None:
    args = build_parser().parse_args()
    result = await run_autopilot(
        resume=args.resume,
        max_experiments=args.max_experiments,
        notify_telegram=args.notify_telegram,
    )
    print("Research Autopilot finished")
    print(f"interrupted: {result['interrupted']}")
    print(f"ran: {result['ran']}")
    print(f"completed: {result['completed']} / {result['total']}")
    print(f"candidates: {result.get('candidates', 0)}")
    print(f"watchlist: {result.get('watchlist', 0)}")
    if args.notify_telegram:
        print(f"telegram_sent: {result.get('telegram_sent', False)}")
    print(f"jsonl: {result['jsonl_path']}")
    print(f"markdown: {result['markdown_path']}")
    print(f"checkpoint: {result['checkpoint_path']}")


if __name__ == "__main__":
    asyncio.run(main())
