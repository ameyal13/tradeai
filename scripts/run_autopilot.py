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
    parser.add_argument("--max-experiments", type=int, default=None)
    return parser


async def main() -> None:
    args = build_parser().parse_args()
    result = await run_autopilot(resume=args.resume, max_experiments=args.max_experiments)
    print("Research Autopilot finished")
    print(f"interrupted: {result['interrupted']}")
    print(f"ran: {result['ran']}")
    print(f"completed: {result['completed']} / {result['total']}")
    print(f"candidates: {result.get('candidates', 0)}")
    print(f"jsonl: {result['jsonl_path']}")
    print(f"markdown: {result['markdown_path']}")
    print(f"checkpoint: {result['checkpoint_path']}")


if __name__ == "__main__":
    asyncio.run(main())
