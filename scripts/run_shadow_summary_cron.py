"""Railway cron entrypoint for daily shadow signal summary.

Research only. No trading signal. No exchange orders.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.summarize_shadow_signals import summarize_shadow_signals  # noqa: E402
from tools.runtime_env import load_project_env  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one daily shadow summary cron.")
    parser.add_argument("--notify-telegram", action="store_true", default=True)
    parser.add_argument("--no-notify-telegram", action="store_false", dest="notify_telegram")
    return parser


def main() -> None:
    load_project_env()
    args = build_parser().parse_args()
    report = summarize_shadow_signals(notify_telegram=args.notify_telegram)
    summary = report.get("summary") or {}
    print("Shadow summary cron finished")
    print("Research only. No trading signal.")
    print(f"open: {summary.get('open')}")
    print(f"closed: {summary.get('closed')}")
    print(f"win_rate: {summary.get('win_rate')}")
    print(f"profit_factor: {summary.get('profit_factor')}")
    print(f"markdown: {report.get('markdown_path')}")


if __name__ == "__main__":
    main()
