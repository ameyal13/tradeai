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

from research.telegram_notifier import format_shadow_ops_cycle_brief, send_telegram_message  # noqa: E402
from scripts.evaluate_shadow_signals_once import evaluate_shadow_signals_once  # noqa: E402
from scripts.run_shadow_cycle_once import default_journal_path, default_shadow_reports_dir, run_shadow_cycle_once  # noqa: E402
from scripts.generate_shadow_signals_once import generate_shadow_signals_once, summarize_generation_rows  # noqa: E402
from scripts.shadow_ops_healthcheck import build_healthcheck_report  # noqa: E402
from scripts.summarize_shadow_signals import summarize_shadow_signals  # noqa: E402
from scripts.sync_shadow_journal_to_supabase import build_supabase_client_from_env  # noqa: E402
from tools.runtime_env import load_project_env  # noqa: E402
from tools.shadow_ops_cycle_repository import DEFAULT_SHADOW_OPS_CYCLES_PATH, ShadowOpsCycleRepository  # noqa: E402
from tools.shadow_signal_repository import ShadowSignalRepository, SupabaseShadowSignalStore  # noqa: E402


SHADOW_OPS_LOCK_NAME = "shadow_ops_once"
SHADOW_OPS_LOCK_MAX_AGE_MINUTES = 10


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_now_dt() -> datetime:
    return datetime.now(timezone.utc)


def shadow_ops_owner_id() -> str:
    return f"{socket.gethostname()}_{os.getpid()}"


def running_on_railway() -> bool:
    return bool(os.getenv("RAILWAY_ENVIRONMENT"))


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


def supabase_shadow_summary(supabase: Any, journal_path: str | Path) -> dict[str, Any]:
    return ShadowSignalRepository(supabase_client=supabase, journal_path=journal_path).summary(prefer_supabase=True)


def _count_values(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "")
        counts[value] = counts.get(value, 0) + 1
    return counts


def _config_symbol(row: dict[str, Any]) -> str:
    config = row.get("config") or {}
    return str(config.get("symbol") or row.get("symbol") or "").upper()


def _config_from_research_row(row: dict[str, Any]) -> dict[str, Any]:
    config = dict(row.get("config") or {})
    config.setdefault("symbol", row.get("symbol"))
    config.setdefault("timeframe", row.get("timeframe"))
    config.setdefault("strategy_mode", row.get("strategy_mode"))
    config.setdefault("horizon_candles", row.get("horizon_candles"))
    config.setdefault("risk_reward", row.get("risk_reward"))
    config.setdefault("atr_stop_multiplier", row.get("atr_stop_multiplier"))
    config.setdefault("cost_mode", row.get("cost_mode"))
    config["config_id"] = row.get("config_id") or config.get("config_id")
    config["_source_classification"] = row.get("classification")
    return config


def _fetch_supabase_research_config_rows(supabase: Any, source: str = "crypto_multi", limit: int = 10_000) -> list[dict[str, Any]]:
    """Fetch research configs directly from Supabase without local fallback.

    Railway is stateless, so silently falling back to a missing local registry
    hides the real issue. This helper lets the ops cycle report whether
    Supabase had rows, classifications, or query errors.
    """
    if supabase is None:
        return []
    data = (
        supabase.table("research_configs")
        .select("*")
        .eq("source", source)
        .limit(max(1, int(limit)))
        .execute()
        .data
        or []
    )
    if not isinstance(data, list):
        raise TypeError("research_configs query did not return a list")
    return data


def supabase_research_config_diagnostics(
    supabase: Any,
    *,
    source: str = "crypto_multi",
    symbols: list[str],
    allow_watchlist_shadow: bool,
) -> dict[str, Any]:
    """Return safe counts explaining why Railway can or cannot scan configs."""
    allowed = {"stable_research_candidate"}
    if allow_watchlist_shadow:
        allowed.add("unstable_watchlist")
    symbol_filter = {symbol.upper() for symbol in symbols}
    diagnostics: dict[str, Any] = {
        "source": source,
        "symbols": sorted(symbol_filter),
        "allow_watchlist_shadow": bool(allow_watchlist_shadow),
        "allowed_classifications": sorted(allowed),
        "query_ok": False,
        "query_error_type": None,
        "query_error": None,
        "source_rows": 0,
        "completed_rows": 0,
        "eligible_classification_rows": 0,
        "symbol_filtered_rows": 0,
        "classification_counts": {},
        "status_counts": {},
        "symbol_counts": {},
    }
    try:
        rows = _fetch_supabase_research_config_rows(supabase, source=source)
    except Exception as exc:  # noqa: BLE001 - diagnostics should not crash ops.
        diagnostics.update({
            "query_ok": False,
            "query_error_type": type(exc).__name__,
            "query_error": str(exc)[:240],
        })
        return diagnostics

    completed = [row for row in rows if row.get("status") == "completed"]
    eligible = [row for row in completed if row.get("classification") in allowed]
    symbol_filtered = [row for row in eligible if _config_symbol(row) in symbol_filter]
    diagnostics.update({
        "query_ok": True,
        "source_rows": len(rows),
        "completed_rows": len(completed),
        "eligible_classification_rows": len(eligible),
        "symbol_filtered_rows": len(symbol_filtered),
        "classification_counts": _count_values(rows, "classification"),
        "status_counts": _count_values(rows, "status"),
        "symbol_counts": _count_values([{"symbol": _config_symbol(row)} for row in rows], "symbol"),
    })
    return diagnostics


def supabase_candidate_configs(
    supabase: Any,
    *,
    symbols: list[str],
    allow_watchlist_shadow: bool,
    source: str = "crypto_multi",
) -> list[dict[str, Any]]:
    allowed = {"stable_research_candidate"}
    if allow_watchlist_shadow:
        allowed.add("unstable_watchlist")
    symbol_filter = {symbol.upper() for symbol in symbols}
    rows = _fetch_supabase_research_config_rows(supabase, source=source)
    configs: list[dict[str, Any]] = []
    for row in rows:
        if row.get("status") != "completed":
            continue
        classification = row.get("classification")
        if classification not in allowed:
            continue
        config = _config_from_research_row(row)
        if str(config.get("symbol", "")).upper() not in symbol_filter:
            continue
        configs.append(config)
    return configs


def summarize_evaluation_errors(errors: list[dict[str, Any]] | None) -> dict[str, Any]:
    rows = list(errors or [])
    code_counts = _count_values(rows, "error_code")
    category_counts = _count_values(rows, "error_category")
    return {
        "count": len(rows),
        "error_code_counts": code_counts,
        "error_category_counts": category_counts,
        "samples": [
            {
                "shadow_signal_id": row.get("shadow_signal_id"),
                "symbol": row.get("symbol"),
                "timeframe": row.get("timeframe"),
                "error_type": row.get("error_type"),
                "error_code": row.get("error_code"),
                "error_category": row.get("error_category"),
                "signal_left_open": row.get("signal_left_open"),
                "error": str(row.get("error") or "")[:180],
            }
            for row in rows[:3]
        ],
    }


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
    evaluation_error_summary = result.get("evaluation_error_summary") or {}
    final = result.get("final_summary") or {}
    signal_sync = result.get("supabase_sync") or {}
    config_diagnostics = result.get("research_config_diagnostics") or {}
    return {
        "cycle_id": cycle_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "dry_run": bool(result.get("dry_run")),
        "health_status": (result.get("health_before") or {}).get("health_status"),
        "evaluated_closed": evaluation.get("closed", 0),
        "evaluation_errors": len(evaluation.get("errors") or []),
        "evaluation_error_summary": evaluation_error_summary,
        "open_after_evaluation": result.get("open_after_evaluation", 0),
        "research_config_diagnostics": config_diagnostics,
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
    railway_mode = running_on_railway()
    supabase = build_supabase_client_from_env() if railway_mode else None
    supabase_first = railway_mode and supabase is not None
    signal_store = SupabaseShadowSignalStore(supabase) if supabase_first else None
    research_config_diagnostics = (
        supabase_research_config_diagnostics(
            supabase,
            symbols=["ADA", "ETH", "SOL"],
            allow_watchlist_shadow=True,
        )
        if supabase_first
        else None
    )
    candidate_configs_error: dict[str, Any] | None = None
    if supabase_first:
        try:
            candidate_configs = supabase_candidate_configs(
                supabase,
                symbols=["ADA", "ETH", "SOL"],
                allow_watchlist_shadow=True,
            )
        except Exception as exc:  # noqa: BLE001 - record and keep cycle alive.
            candidate_configs = []
            candidate_configs_error = {
                "error_type": type(exc).__name__,
                "error": str(exc)[:240],
            }
    else:
        candidate_configs = None
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
        evaluation = await evaluate_shadow_signals_once(
            journal_path=journal,
            notify_telegram=notify_telegram,
            signal_store=signal_store,
        )
    evaluation_error_summary = summarize_evaluation_errors(evaluation.get("errors") or [])

    first_summary = (
        supabase_shadow_summary(supabase, journal) if supabase_first else summarize_shadow_signals(
            journal_path=journal,
            output_dir=output_dir,
            notify_telegram=False,
            write_report=not dry_run,
        )
    )
    open_after_eval = int((first_summary.get("summary") or {}).get("open") or 0)

    generation_cycle: dict[str, Any] | None = None
    generation_skipped_reason: str | None = None
    if open_after_eval > 0 and not allow_open_more_signals:
        generation_skipped_reason = "open_signals_exist"
    elif railway_mode and not supabase_first:
        generation_skipped_reason = "supabase_not_configured"
    elif supabase_first:
        generated = await generate_shadow_signals_once(
            registry="crypto_multi",
            symbols=["ADA", "ETH", "SOL"],
            max_signals=max_signals,
            max_configs_scanned=max_configs_scanned,
            allow_watchlist_shadow=True,
            notify_telegram=notify_telegram,
            dry_run=dry_run,
            min_classification="stable_research_candidate",
            journal_path=journal,
            refresh_cache=refresh_cache,
            use_news_context=use_news_context,
            use_market_context=use_market_context,
            signal_store=signal_store,
            candidate_configs=candidate_configs,
        )
        generation_cycle = {
            "generation_summary": summarize_generation_rows(
                generated,
                journal_path=journal,
                max_signals=max_signals,
                max_configs_scanned=max_configs_scanned,
            ),
            "generation": generated,
            "railway_supabase_first": True,
        }
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

    final_summary = (
        supabase_shadow_summary(supabase, journal) if supabase_first else summarize_shadow_signals(
            journal_path=journal,
            output_dir=output_dir,
            notify_telegram=False,
            write_report=not dry_run,
        )
    )
    if supabase_first:
        supabase_sync = {
            "ok": True,
            "attempted": False,
            "reason": "supabase_first",
            "signals_upserted": 0,
            "events_upserted": 0,
            "journal_path": str(journal),
        }
    elif sync_supabase and not dry_run:
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
        "railway_mode": bool(railway_mode),
        "supabase_first": bool(supabase_first),
        "sync_supabase": bool(sync_supabase),
        "health_before": health_before,
        "evaluation": evaluation,
        "evaluation_error_summary": evaluation_error_summary,
        "research_config_diagnostics": research_config_diagnostics,
        "candidate_configs_count": len(candidate_configs or []) if supabase_first else None,
        "candidate_configs_error": candidate_configs_error,
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
        result["ops_telegram_sent"] = send_telegram_message(format_shadow_ops_cycle_brief(result))
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
    print(f"railway_mode: {result.get('railway_mode')}")
    print(f"supabase_first: {result.get('supabase_first')}")
    print(f"sync_supabase: {result.get('sync_supabase')}")
    print(f"health_status: {(result.get('health_before') or {}).get('health_status')}")
    print(f"evaluated_closed: {(result.get('evaluation') or {}).get('closed')}")
    error_summary = result.get("evaluation_error_summary") or {}
    print(f"evaluation_errors: {error_summary.get('count', len((result.get('evaluation') or {}).get('errors') or []))}")
    print(f"evaluation_error_codes: {error_summary.get('error_code_counts')}")
    print(f"evaluation_error_categories: {error_summary.get('error_category_counts')}")
    print(f"open_after_evaluation: {result.get('open_after_evaluation')}")
    print(f"generation_skipped_reason: {result.get('generation_skipped_reason')}")
    cycle = result.get("generation_cycle") or {}
    generation = cycle.get("generation_summary") or {}
    print(f"opened_signals: {generation.get('opened_signals', 0)}")
    print(f"configs_scanned: {generation.get('configs_scanned', 0)}")
    config_diag = result.get("research_config_diagnostics") or {}
    if config_diag:
        print(f"research_config_query_ok: {config_diag.get('query_ok')}")
        print(f"research_config_source_rows: {config_diag.get('source_rows')}")
        print(f"research_config_completed_rows: {config_diag.get('completed_rows')}")
        print(f"research_config_eligible_classification_rows: {config_diag.get('eligible_classification_rows')}")
        print(f"research_config_symbol_filtered_rows: {config_diag.get('symbol_filtered_rows')}")
        print(f"research_config_classification_counts: {config_diag.get('classification_counts')}")
        if config_diag.get("query_error_type"):
            print(f"research_config_query_error: {config_diag.get('query_error_type')}: {config_diag.get('query_error')}")
    print(f"candidate_configs_count: {result.get('candidate_configs_count')}")
    if result.get("candidate_configs_error"):
        err = result["candidate_configs_error"]
        print(f"candidate_configs_error: {err.get('error_type')}: {err.get('error')}")
    for sample in (error_summary.get("samples") or []):
        print(
            "evaluation_error_sample: "
            f"id={sample.get('shadow_signal_id')} "
            f"symbol={sample.get('symbol')} "
            f"code={sample.get('error_code')} "
            f"category={sample.get('error_category')} "
            f"type={sample.get('error_type')} "
            f"left_open={sample.get('signal_left_open')}"
        )
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
