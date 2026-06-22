"""Run one safe local Shadow Ops cycle.

This orchestrates healthcheck, evaluation, summary, and optional generation.
It never places real orders and never uses LLMs to create trades.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.telegram_notifier import send_telegram_message  # noqa: E402
from scripts.evaluate_shadow_signals_once import evaluate_shadow_signals_once  # noqa: E402
from scripts.run_shadow_cycle_once import default_journal_path, default_shadow_reports_dir, run_shadow_cycle_once  # noqa: E402
from scripts.shadow_ops_healthcheck import build_healthcheck_report  # noqa: E402
from scripts.summarize_shadow_signals import summarize_shadow_signals  # noqa: E402
from scripts.sync_shadow_journal_to_supabase import build_supabase_client_from_env  # noqa: E402
from tools.runtime_env import load_project_env  # noqa: E402
from tools.shadow_ops_cycle_repository import DEFAULT_SHADOW_OPS_CYCLES_PATH, ShadowOpsCycleRepository  # noqa: E402
from tools.shadow_signal_repository import ShadowSignalRepository  # noqa: E402


SHADOW_OPS_LOCK_NAME = "shadow_ops_once"
SHADOW_OPS_LOCK_MAX_AGE_MINUTES = 10


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_now_dt() -> datetime:
    return datetime.now(timezone.utc)


def shadow_ops_owner_id() -> str:
    return f"{socket.gethostname()}_{os.getpid()}"


def parse_lock_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        try:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def lock_is_active(row: dict[str, Any], now: datetime | None = None) -> bool:
    acquired_at = parse_lock_dt(row.get("acquired_at"))
    if acquired_at is None:
        return False
    return acquired_at > (now or utc_now_dt()) - timedelta(minutes=SHADOW_OPS_LOCK_MAX_AGE_MINUTES)


def acquire_shadow_ops_lock(supabase: Any, *, owner_id: str, cycle_id: str) -> dict[str, Any]:
    if supabase is None:
        return {"acquired": True, "reason": "supabase_not_configured", "owner_id": owner_id, "lock_enabled": False}
    now = utc_now_dt()
    payload = {
        "lock_name": SHADOW_OPS_LOCK_NAME,
        "owner_id": owner_id,
        "acquired_at": now.isoformat(),
        "expires_at": (now + timedelta(minutes=SHADOW_OPS_LOCK_MAX_AGE_MINUTES)).isoformat(),
        "heartbeat_at": now.isoformat(),
        "cycle_id": cycle_id,
        "metadata": {"source": "run_shadow_ops_once.py", "research_only": True},
    }
    try:
        supabase.table("shadow_ops_locks").insert(payload).execute()
        return {"acquired": True, "reason": "inserted", "owner_id": owner_id, "lock_enabled": True}
    except Exception:
        existing = (
            supabase.table("shadow_ops_locks")
            .select("*")
            .eq("lock_name", SHADOW_OPS_LOCK_NAME)
            .limit(1)
            .execute()
            .data
            or []
        )
        row = existing[0] if existing else None
        if row and lock_is_active(row, now=now):
            return {
                "acquired": False,
                "reason": "active_lock",
                "owner_id": owner_id,
                "held_by": row.get("owner_id"),
                "lock_enabled": True,
            }
        supabase.table("shadow_ops_locks").upsert(payload, on_conflict="lock_name").execute()
        return {"acquired": True, "reason": "stale_lock_overwritten", "owner_id": owner_id, "lock_enabled": True}


def release_shadow_ops_lock(supabase: Any, *, owner_id: str) -> dict[str, Any]:
    if supabase is None:
        return {"released": False, "reason": "supabase_not_configured", "lock_enabled": False}
    supabase.table("shadow_ops_locks").delete().eq("lock_name", SHADOW_OPS_LOCK_NAME).eq("owner_id", owner_id).execute()
    return {"released": True, "reason": None, "owner_id": owner_id, "lock_enabled": True}


def sync_shadow_journal_to_supabase_safe(journal_path: str | Path) -> dict[str, Any]:
    """Best-effort sync from local shadow journal to Supabase.

    The local JSONL journal remains the source of truth. Supabase sync is allowed
    to fail without breaking the shadow ops cycle.
    """
    try:
        supabase = build_supabase_client_from_env()
        repo = ShadowSignalRepository(supabase_client=supabase, journal_path=journal_path)
        result = repo.sync_local_to_supabase()
        result["attempted"] = True
        return result
    except Exception as exc:
        return {
            "ok": False,
            "attempted": True,
            "reason": "sync_error",
            "error_type": type(exc).__name__,
            "error": str(exc)[:240],
            "signals_upserted": 0,
            "events_upserted": 0,
            "journal_path": str(journal_path),
        }


def sync_shadow_ops_cycles_to_supabase_safe(cycles_path: str | Path) -> dict[str, Any]:
    try:
        supabase = build_supabase_client_from_env()
        repo = ShadowOpsCycleRepository(supabase_client=supabase, path=cycles_path)
        result = repo.sync_local_to_supabase()
        result["attempted"] = True
        return result
    except Exception as exc:
        return {
            "ok": False,
            "attempted": True,
            "reason": "sync_error",
            "error_type": type(exc).__name__,
            "error": str(exc)[:240],
            "cycles_upserted": 0,
            "path": str(cycles_path),
        }


def build_shadow_ops_cycle_record(result: dict[str, Any], *, cycle_id: str, started_at: str, finished_at: str) -> dict[str, Any]:
    generation = ((result.get("generation_cycle") or {}).get("generation_summary") or {})
    evaluation = result.get("evaluation") or {}
    final = result.get("final_summary") or {}
    signal_sync = result.get("supabase_sync") or {}
    return {
        "cycle_id": cycle_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "dry_run": bool(result.get("dry_run")),
        "health_status": (result.get("health_before") or {}).get("health_status"),
        "evaluated_closed": evaluation.get("closed", 0),
        "evaluation_errors": len(evaluation.get("errors") or []),
        "open_after_evaluation": result.get("open_after_evaluation", 0),
        "generation_skipped_reason": result.get("generation_skipped_reason"),
        "opened_signals": generation.get("opened_signals", 0),
        "configs_scanned": generation.get("configs_scanned", 0),
        "skipped_hold": generation.get("skipped_hold", 0),
        "skipped_duplicate_open": generation.get("skipped_duplicate_open", 0),
        "skipped_duplicate_similar": generation.get("skipped_duplicate_similar", 0),
        "skipped_errors": generation.get("skipped_errors", 0),
        "status_counts": generation.get("status_counts") or {},
        "final_open": final.get("open", 0),
        "final_closed": final.get("closed", 0),
        "sync_supabase": bool(result.get("sync_supabase")),
        "supabase_sync_ok": bool(signal_sync.get("ok")),
        "supabase_sync_reason": signal_sync.get("reason"),
        "research_only": True,
        "raw": result,
    }


async def run_shadow_ops_once(
    *,
    notify_telegram: bool = False,
    dry_run: bool = False,
    allow_open_more_signals: bool = False,
    max_signals: int = 1,
    max_configs_scanned: int = 21,
    use_news_context: bool = False,
    use_market_context: bool = False,
    sync_supabase: bool = False,
    refresh_cache: bool = True,
    journal_path: str | Path | None = None,
    reports_output_dir: str | Path | None = None,
    cycles_path: str | Path | None = None,
) -> dict[str, Any]:
    cycle_id = str(uuid4())
    started_at = utc_now_iso()
    journal = Path(journal_path) if journal_path else default_journal_path()
    output_dir = Path(reports_output_dir) if reports_output_dir else default_shadow_reports_dir()
    cycles = Path(cycles_path) if cycles_path else DEFAULT_SHADOW_OPS_CYCLES_PATH
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
    if sync_supabase and not dry_run:
        supabase_sync = sync_shadow_journal_to_supabase_safe(journal)
    elif sync_supabase and dry_run:
        supabase_sync = {
            "ok": False,
            "attempted": False,
            "reason": "dry_run",
            "signals_upserted": 0,
            "events_upserted": 0,
            "journal_path": str(journal),
        }
    else:
        supabase_sync = {
            "ok": False,
            "attempted": False,
            "reason": "disabled",
            "signals_upserted": 0,
            "events_upserted": 0,
            "journal_path": str(journal),
        }
    result = {
        "cycle_id": cycle_id,
        "started_at": started_at,
        "finished_at": utc_now_iso(),
        "guardrails": {
            "research_only": True,
            "no_real_trading": True,
            "no_exchange_orders": True,
            "no_llm_trader": True,
        },
        "dry_run": bool(dry_run),
        "use_news_context": bool(use_news_context),
        "use_market_context": bool(use_market_context),
        "sync_supabase": bool(sync_supabase),
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
        "supabase_sync": supabase_sync,
        "cycles_path": str(cycles),
    }
    cycle_record = build_shadow_ops_cycle_record(
        result,
        cycle_id=cycle_id,
        started_at=started_at,
        finished_at=result["finished_at"],
    )
    if not dry_run:
        cycle_record = ShadowOpsCycleRepository(path=cycles).append_cycle(cycle_record)
    if sync_supabase and not dry_run:
        cycles_sync = sync_shadow_ops_cycles_to_supabase_safe(cycles)
    elif sync_supabase and dry_run:
        cycles_sync = {"ok": False, "attempted": False, "reason": "dry_run", "cycles_upserted": 0, "path": str(cycles)}
    else:
        cycles_sync = {"ok": False, "attempted": False, "reason": "disabled", "cycles_upserted": 0, "path": str(cycles)}
    result["cycle_record"] = cycle_record
    result["cycles_sync"] = cycles_sync
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
            f"Supabase sync: {supabase_sync.get('ok')} ({supabase_sync.get('reason')})\n"
            f"Cycle sync: {cycles_sync.get('ok')} ({cycles_sync.get('reason')})\n"
            f"Markdown: {final_summary.get('markdown_path')}"
        )
        result["ops_telegram_sent"] = send_telegram_message(text)
    else:
        result["ops_telegram_sent"] = False
    return result


def print_ops_result(result: dict[str, Any]) -> None:
    print("Shadow Ops cycle finished")
    print("Research only. No trading signal.")
    print(f"cycle_id: {result.get('cycle_id')}")
    print(f"dry_run: {result.get('dry_run')}")
    print(f"use_news_context: {result.get('use_news_context')}")
    print(f"use_market_context: {result.get('use_market_context')}")
    print(f"sync_supabase: {result.get('sync_supabase')}")
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
    sync = result.get("supabase_sync") or {}
    print(f"supabase_sync_attempted: {sync.get('attempted')}")
    print(f"supabase_sync_ok: {sync.get('ok')}")
    print(f"supabase_sync_reason: {sync.get('reason')}")
    print(f"supabase_signals_upserted: {sync.get('signals_upserted')}")
    print(f"supabase_events_upserted: {sync.get('events_upserted')}")
    cycle_sync = result.get("cycles_sync") or {}
    print(f"cycles_sync_attempted: {cycle_sync.get('attempted')}")
    print(f"cycles_sync_ok: {cycle_sync.get('ok')}")
    print(f"cycles_sync_reason: {cycle_sync.get('reason')}")
    print(f"cycles_upserted: {cycle_sync.get('cycles_upserted')}")
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
    parser.add_argument("--sync-supabase", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true", default=True)
    parser.add_argument("--no-refresh-cache", action="store_false", dest="refresh_cache")
    parser.add_argument("--journal-path", default=None)
    parser.add_argument("--reports-output-dir", default=None)
    parser.add_argument("--cycles-path", default=None)
    parser.add_argument("--json", action="store_true", dest="json_output")
    return parser


async def main() -> None:
    load_project_env()
    args = build_parser().parse_args()
    cycle_id = str(uuid4())
    owner_id = shadow_ops_owner_id()
    supabase = build_supabase_client_from_env()
    lock = acquire_shadow_ops_lock(supabase, owner_id=owner_id, cycle_id=cycle_id)
    if not lock.get("acquired"):
        print("lock held by another process, skipping")
        return
    try:
        result = await run_shadow_ops_once(
            notify_telegram=args.notify_telegram,
            dry_run=args.dry_run,
            allow_open_more_signals=args.allow_open_more_signals,
            max_signals=args.max_signals,
            max_configs_scanned=args.max_configs_scanned,
            use_news_context=args.use_news_context,
            use_market_context=args.use_market_context,
            sync_supabase=args.sync_supabase,
            refresh_cache=args.refresh_cache,
            journal_path=args.journal_path,
            reports_output_dir=args.reports_output_dir,
            cycles_path=args.cycles_path,
        )
    finally:
        release_shadow_ops_lock(supabase, owner_id=owner_id)
    if args.json_output:
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
    else:
        print_ops_result(result)


if __name__ == "__main__":
    asyncio.run(main())
