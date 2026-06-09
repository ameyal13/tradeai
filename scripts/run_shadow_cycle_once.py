"""Run one local production-shadow cycle.

This is intended to be Railway-Cron friendly later, but remains local and safe:
no real trading, no exchange orders, no scheduler loop, no Supabase writes, and
no LLM trader.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.telegram_notifier import send_telegram_message  # noqa: E402
from scripts.evaluate_shadow_signals_once import evaluate_shadow_signals_once  # noqa: E402
from scripts.generate_shadow_signals_once import generate_shadow_signals_once, summarize_generation_rows  # noqa: E402
from scripts.summarize_shadow_signals import summarize_shadow_signals  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def data_dir() -> Path:
    return Path(os.getenv("TRADEAI_DATA_DIR", "data"))


def reports_dir() -> Path:
    return Path(os.getenv("TRADEAI_REPORTS_DIR", "reports"))


def default_journal_path() -> Path:
    return data_dir() / "shadow_signal_journal.jsonl"


def default_lock_path() -> Path:
    return data_dir() / "shadow_cycle.lock"


def default_shadow_reports_dir() -> Path:
    return reports_dir() / "shadow"


def registry_path_for_choice(choice: str, base_reports_dir: str | Path | None = None) -> str:
    base = Path(base_reports_dir) if base_reports_dir else reports_dir()
    if choice == "refined":
        return str(base / "research_daemon" / "refined_registry.jsonl")
    if choice == "general":
        return str(base / "research_daemon" / "registry.jsonl")
    if choice == "crypto_multi":
        return str(base / "research_daemon" / "crypto_multi_registry.jsonl")
    return choice


class ShadowCycleLocked(RuntimeError):
    pass


@contextmanager
def shadow_cycle_lock(lock_path: str | Path) -> Iterator[None]:
    path = Path(lock_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        handle = path.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise ShadowCycleLocked(f"Shadow cycle already running: {path}") from exc
    try:
        handle.write(json.dumps({"started_at": utc_now(), "pid": os.getpid()}))
        handle.close()
        yield
    finally:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


async def run_shadow_cycle_once(
    registry: str = "refined",
    symbols: list[str] | None = None,
    max_signals: int = 5,
    max_configs_scanned: int | None = None,
    allow_watchlist_shadow: bool = False,
    notify_telegram: bool = False,
    dry_run: bool = False,
    min_classification: str = "stable_research_candidate",
    refresh_cache: bool = True,
    journal_path: str | Path | None = None,
    reports_output_dir: str | Path | None = None,
    lock_path: str | Path | None = None,
) -> dict[str, Any]:
    journal = Path(journal_path) if journal_path else default_journal_path()
    output_dir = Path(reports_output_dir) if reports_output_dir else default_shadow_reports_dir()
    lock = Path(lock_path) if lock_path else default_lock_path()
    registry_path = registry_path_for_choice(registry)
    effective_max_configs_scanned = int(max_configs_scanned) if max_configs_scanned is not None else int(max_signals)
    result: dict[str, Any] = {
        "started_at": utc_now(),
        "dry_run": bool(dry_run),
        "max_signals": int(max_signals),
        "max_configs_scanned": effective_max_configs_scanned,
        "journal_path": str(journal),
        "reports_output_dir": str(output_dir),
        "lock_path": str(lock),
        "registry": registry,
        "registry_path": registry_path,
        "guardrails": {
            "research_only": True,
            "no_real_trading": True,
            "no_exchange_orders": True,
            "no_llm_trader": True,
        },
    }

    if dry_run:
        evaluation = {
            "found_open": 0,
            "closed": 0,
            "still_open": 0,
            "errors": [],
            "closed_signals": [],
            "dry_run_note": "evaluation skipped in dry-run to avoid writes",
        }
        generated = await generate_shadow_signals_once(
            registry=registry_path,
            symbols=symbols,
            max_signals=max_signals,
            allow_watchlist_shadow=allow_watchlist_shadow,
            notify_telegram=False,
            dry_run=True,
            min_classification=min_classification,
            journal_path=journal,
            refresh_cache=refresh_cache,
            max_configs_scanned=effective_max_configs_scanned,
        )
        summary = summarize_shadow_signals(
            journal_path=journal,
            output_dir=output_dir,
            notify_telegram=False,
            write_report=False,
        )
    else:
        with shadow_cycle_lock(lock):
            evaluation = await evaluate_shadow_signals_once(journal_path=journal, notify_telegram=notify_telegram)
            generated = await generate_shadow_signals_once(
                registry=registry_path,
                symbols=symbols,
                max_signals=max_signals,
                allow_watchlist_shadow=allow_watchlist_shadow,
                notify_telegram=notify_telegram,
                dry_run=False,
                min_classification=min_classification,
                journal_path=journal,
                refresh_cache=refresh_cache,
                max_configs_scanned=effective_max_configs_scanned,
            )
            summary = summarize_shadow_signals(
                journal_path=journal,
                output_dir=output_dir,
                notify_telegram=False,
                write_report=True,
            )

    generation_summary = summarize_generation_rows(
        generated,
        journal_path=journal,
        max_signals=int(max_signals),
        max_configs_scanned=effective_max_configs_scanned,
    )
    result.update({
        "finished_at": utc_now(),
        "evaluation": evaluation,
        "generation": generated,
        "generation_summary": generation_summary,
        "shadow_summary": summary.get("summary"),
        "summary_json_path": summary.get("json_path"),
        "summary_markdown_path": summary.get("markdown_path"),
    })
    if notify_telegram and not dry_run:
        text = (
            "Shadow cycle finished\n"
            "Research only. No trading signal.\n\n"
            f"Evaluated closed: {evaluation.get('closed')}\n"
            f"Opened signals: {generation_summary.get('opened_signals')}\n"
            f"Configs scanned: {generation_summary.get('configs_scanned')}\n"
            f"Skipped hold: {generation_summary.get('skipped_hold')}\n"
            f"Skipped duplicate similar: {generation_summary.get('skipped_duplicate_similar')}\n"
            f"Open now: {(summary.get('summary') or {}).get('open')}\n"
            f"Closed total: {(summary.get('summary') or {}).get('closed')}\n"
            f"Markdown: {summary.get('markdown_path')}"
        )
        result["cycle_telegram_sent"] = send_telegram_message(text)
    else:
        result["cycle_telegram_sent"] = False
    return result


def print_cycle_result(result: dict[str, Any]) -> None:
    generation = result.get("generation_summary") or {}
    evaluation = result.get("evaluation") or {}
    shadow_summary = result.get("shadow_summary") or {}
    print("Shadow cycle finished")
    print("Research only. No trading signal.")
    print(f"dry_run: {result.get('dry_run')}")
    print(f"registry_path: {result.get('registry_path')}")
    print(f"journal_path: {result.get('journal_path')}")
    print(f"evaluated_closed: {evaluation.get('closed')}")
    print(f"evaluation_errors: {len(evaluation.get('errors') or [])}")
    print(f"configs_scanned: {generation.get('configs_scanned')}")
    print(f"opened_signals: {generation.get('opened_signals')}")
    print(f"skipped_hold: {generation.get('skipped_hold')}")
    print(f"skipped_duplicate_open: {generation.get('skipped_duplicate_open')}")
    print(f"skipped_duplicate_similar: {generation.get('skipped_duplicate_similar')}")
    print(f"skipped_errors: {generation.get('skipped_errors')}")
    print(f"max_signals: {generation.get('max_signals')}")
    print(f"max_configs_scanned: {generation.get('max_configs_scanned')}")
    print(f"shadow_open: {shadow_summary.get('open')}")
    print(f"shadow_closed: {shadow_summary.get('closed')}")
    if result.get("summary_json_path"):
        print(f"summary_json: {result.get('summary_json_path')}")
        print(f"summary_markdown: {result.get('summary_markdown_path')}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one local shadow cycle.")
    parser.add_argument("--registry", default="refined", help="refined, general, crypto_multi, or a registry JSONL path.")
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--max-signals", type=int, default=5)
    parser.add_argument("--max-configs-scanned", type=int, default=None)
    parser.add_argument("--allow-watchlist-shadow", action="store_true")
    parser.add_argument("--notify-telegram", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--min-classification", choices=["stable_research_candidate", "unstable_watchlist"], default="stable_research_candidate")
    parser.add_argument("--refresh-cache", action="store_true", default=True)
    parser.add_argument("--no-refresh-cache", action="store_false", dest="refresh_cache")
    parser.add_argument("--journal-path", default=None)
    parser.add_argument("--reports-output-dir", default=None)
    parser.add_argument("--lock-path", default=None)
    return parser


async def main() -> None:
    args = build_parser().parse_args()
    result = await run_shadow_cycle_once(
        registry=args.registry,
        symbols=args.symbols,
        max_signals=args.max_signals,
        max_configs_scanned=args.max_configs_scanned,
        allow_watchlist_shadow=args.allow_watchlist_shadow,
        notify_telegram=args.notify_telegram,
        dry_run=args.dry_run,
        min_classification=args.min_classification,
        refresh_cache=args.refresh_cache,
        journal_path=args.journal_path,
        reports_output_dir=args.reports_output_dir,
        lock_path=args.lock_path,
    )
    print_cycle_result(result)


if __name__ == "__main__":
    asyncio.run(main())
