"""Run one safe local Shadow Ops cycle.

This orchestrates healthcheck, evaluation, summary, and optional generation.
It never places real orders and never uses LLMs to create trades.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.telegram_notifier import send_telegram_message  # noqa: E402
from scripts.evaluate_shadow_signals_once import evaluate_shadow_signals_once  # noqa: E402
from scripts.run_shadow_cycle_once import default_journal_path, default_shadow_reports_dir, run_shadow_cycle_once  # noqa: E402
from scripts.shadow_ops_healthcheck import build_healthcheck_report  # noqa: E402
from scripts.summarize_shadow_signals import summarize_shadow_signals  # noqa: E402
from tools.runtime_env import load_project_env  # noqa: E402


async def run_shadow_ops_once(
    *,
    notify_telegram: bool = False,
    dry_run: bool = False,
    allow_open_more_signals: bool = False,
    max_signals: int = 1,
    max_configs_scanned: int = 21,
    use_news_context: bool = False,
    use_market_context: bool = False,
    refresh_cache: bool = True,
    journal_path: str | Path | None = None,
    reports_output_dir: str | Path | None = None,
) -> dict[str, Any]:
    journal = Path(journal_path) if journal_path else default_journal_path()
    output_dir = Path(reports_output_dir) if reports_output_dir else default_shadow_reports_dir()
    health_before = await build_healthcheck_report(journal_path=journal)

    if dry_run:
        evaluation = {
            "found_open": (health_before.get("shadow_journal") or {}).get("open", 0),
            "closed": 0,
            "still_open": (health_before.get("shadow_journal") or {}).get("open", 0),
            "errors": [],
            "closed_signals": [],
            "dry_run_note": "evaluation skipped in dry-run to avoid journal writes",
        }
    else:
        evaluation = await evaluate_shadow_signals_once(journal_path=journal, notify_telegram=notify_telegram)

    first_summary = summarize_shadow_signals(
        journal_path=journal,
        output_dir=output_dir,
        notify_telegram=False,
        write_report=not dry_run,
    )
    open_after_eval = int((first_summary.get("summary") or {}).get("open") or 0)

    generation_cycle: dict[str, Any] | None = None
    generation_skipped_reason: str | None = None
    if open_after_eval > 0 and not allow_open_more_signals:
        generation_skipped_reason = "open_signals_exist"
    else:
        generation_cycle = await run_shadow_cycle_once(
            registry="crypto_multi",
            symbols=["ADA", "ETH", "SOL"],
            max_signals=max_signals,
            max_configs_scanned=max_configs_scanned,
            allow_watchlist_shadow=True,
            notify_telegram=notify_telegram,
            dry_run=dry_run,
            min_classification="stable_research_candidate",
            refresh_cache=refresh_cache,
            journal_path=journal,
            reports_output_dir=output_dir,
            use_news_context=use_news_context,
            use_market_context=use_market_context,
        )

    final_summary = summarize_shadow_signals(
        journal_path=journal,
        output_dir=output_dir,
        notify_telegram=False,
        write_report=not dry_run,
    )
    result = {
        "guardrails": {
            "research_only": True,
            "no_real_trading": True,
            "no_exchange_orders": True,
            "no_llm_trader": True,
        },
        "dry_run": bool(dry_run),
        "use_news_context": bool(use_news_context),
        "use_market_context": bool(use_market_context),
        "health_before": health_before,
        "evaluation": evaluation,
        "summary_before_generation": first_summary.get("summary"),
        "open_after_evaluation": open_after_eval,
        "generation_skipped_reason": generation_skipped_reason,
        "generation_cycle": generation_cycle,
        "final_summary": final_summary.get("summary"),
        "summary_json_path": final_summary.get("json_path"),
        "summary_markdown_path": final_summary.get("markdown_path"),
        "journal_path": str(journal),
    }
    if notify_telegram and not dry_run:
        text = (
            "Shadow Ops cycle finished\n"
            "Research only. No trading signal.\n\n"
            f"Health: {health_before.get('health_status')}\n"
            f"Evaluated closed: {evaluation.get('closed')}\n"
            f"Open after evaluation: {open_after_eval}\n"
            f"Generation skipped: {generation_skipped_reason or 'no'}\n"
            f"Final open: {(final_summary.get('summary') or {}).get('open')}\n"
            f"Final closed: {(final_summary.get('summary') or {}).get('closed')}\n"
            f"Markdown: {final_summary.get('markdown_path')}"
        )
        result["ops_telegram_sent"] = send_telegram_message(text)
    else:
        result["ops_telegram_sent"] = False
    return result


def print_ops_result(result: dict[str, Any]) -> None:
    print("Shadow Ops cycle finished")
    print("Research only. No trading signal.")
    print(f"dry_run: {result.get('dry_run')}")
    print(f"use_news_context: {result.get('use_news_context')}")
    print(f"use_market_context: {result.get('use_market_context')}")
    print(f"health_status: {(result.get('health_before') or {}).get('health_status')}")
    print(f"evaluated_closed: {(result.get('evaluation') or {}).get('closed')}")
    print(f"evaluation_errors: {len((result.get('evaluation') or {}).get('errors') or [])}")
    print(f"open_after_evaluation: {result.get('open_after_evaluation')}")
    print(f"generation_skipped_reason: {result.get('generation_skipped_reason')}")
    cycle = result.get("generation_cycle") or {}
    generation = cycle.get("generation_summary") or {}
    print(f"opened_signals: {generation.get('opened_signals', 0)}")
    print(f"configs_scanned: {generation.get('configs_scanned', 0)}")
    final = result.get("final_summary") or {}
    print(f"final_open: {final.get('open')}")
    print(f"final_closed: {final.get('closed')}")
    if result.get("summary_markdown_path"):
        print(f"summary_markdown: {result.get('summary_markdown_path')}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one safe local Shadow Ops cycle.")
    parser.add_argument("--notify-telegram", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-open-more-signals", action="store_true")
    parser.add_argument("--max-signals", type=int, default=1)
    parser.add_argument("--max-configs-scanned", type=int, default=21)
    parser.add_argument("--use-news-context", action="store_true")
    parser.add_argument("--use-market-context", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true", default=True)
    parser.add_argument("--no-refresh-cache", action="store_false", dest="refresh_cache")
    parser.add_argument("--journal-path", default=None)
    parser.add_argument("--reports-output-dir", default=None)
    parser.add_argument("--json", action="store_true", dest="json_output")
    return parser


async def main() -> None:
    load_project_env()
    args = build_parser().parse_args()
    result = await run_shadow_ops_once(
        notify_telegram=args.notify_telegram,
        dry_run=args.dry_run,
        allow_open_more_signals=args.allow_open_more_signals,
        max_signals=args.max_signals,
        max_configs_scanned=args.max_configs_scanned,
        use_news_context=args.use_news_context,
        use_market_context=args.use_market_context,
        refresh_cache=args.refresh_cache,
        journal_path=args.journal_path,
        reports_output_dir=args.reports_output_dir,
    )
    if args.json_output:
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
    else:
        print_ops_result(result)


if __name__ == "__main__":
    asyncio.run(main())
