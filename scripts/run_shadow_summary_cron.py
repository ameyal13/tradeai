"""Railway cron entrypoint for daily shadow signal summary.

Research only. No trading signal. No exchange orders.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_shadow_cycle_once import default_journal_path  # noqa: E402
from scripts.summarize_shadow_signals import save_shadow_summary, summarize_shadow_signals  # noqa: E402
from scripts.sync_shadow_journal_to_supabase import build_supabase_client_from_env  # noqa: E402
from tools.runtime_env import load_project_env  # noqa: E402
from tools.shadow_signal_repository import ShadowSignalRepository  # noqa: E402


def supabase_first_requested() -> bool:
    value = str(os.getenv("TRADEAI_SUPABASE_FIRST") or "").strip().lower()
    return bool(os.getenv("RAILWAY_ENVIRONMENT")) or bool(os.getenv("GITHUB_ACTIONS")) or value in {"1", "true", "yes", "on"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one daily shadow summary cron.")
    parser.add_argument("--notify-telegram", action="store_true", default=True)
    parser.add_argument("--no-notify-telegram", action="store_false", dest="notify_telegram")
    return parser


def main() -> None:
    load_project_env()
    args = build_parser().parse_args()
    if supabase_first_requested():
        supabase = build_supabase_client_from_env()
        if supabase is None:
            report = summarize_shadow_signals(notify_telegram=args.notify_telegram)
            report["source"] = "local_jsonl_supabase_not_configured"
        else:
            report = ShadowSignalRepository(
                supabase_client=supabase,
                journal_path=default_journal_path(),
            ).summary(prefer_supabase=True)
            paths = save_shadow_summary(report)
            report["json_path"] = str(paths["json_path"])
            report["markdown_path"] = str(paths["markdown_path"])
            if args.notify_telegram:
                from research.telegram_notifier import format_shadow_daily_summary, send_telegram_message

                closed = [row for row in report["signals"] if row.get("status") in {"CLOSED", "EXPIRED"}]
                send_telegram_message(format_shadow_daily_summary(closed))
    else:
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
