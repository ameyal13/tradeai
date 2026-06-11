"""Local Shadow Ops healthcheck.

Research/shadow only. This script checks local readiness without reading or
printing secrets, without placing orders, and without writing journals.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.telegram_notifier import send_telegram_message, telegram_enabled  # noqa: E402
from scripts.run_shadow_cycle_once import default_lock_path, registry_path_for_choice  # noqa: E402
from scripts.summarize_shadow_signals import build_shadow_summary, default_journal_path, default_output_dir  # noqa: E402
from tools.historical_data import classify_historical_data_error, fetch_binance_klines  # noqa: E402
from tools.prediction_journal import parse_dt  # noqa: E402
from tools.runtime_env import load_project_env  # noqa: E402


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _latest_shadow_summary_path(output_dir: str | Path | None = None) -> str | None:
    directory = Path(output_dir) if output_dir else default_output_dir()
    matches = sorted(directory.glob("shadow_summary_*.md")) if directory.exists() else []
    return str(matches[-1]) if matches else None


def _load_registry_latest(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    latest: dict[str, dict[str, Any]] = {}
    for line in target.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        latest[str(row.get("config_id"))] = row
    return list(latest.values())


def _git_pending_changes() -> dict[str, Any]:
    try:
        proc = subprocess.run(
            ["git", "status", "--short"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        lines = [line for line in proc.stdout.splitlines() if line.strip()]
        return {
            "ok": proc.returncode == 0,
            "pending": bool(lines),
            "pending_count": len(lines),
            "error": proc.stderr.strip()[:200] if proc.returncode else None,
        }
    except Exception as exc:  # noqa: BLE001 - healthcheck should be best effort.
        return {"ok": False, "pending": None, "pending_count": None, "error": f"{type(exc).__name__}: {exc}"}


async def _check_market_connectivity() -> dict[str, Any]:
    try:
        candles = await fetch_binance_klines("BTC", "1h", limit=1, retries=0)
        return {"ok": bool(len(candles)), "rows": int(len(candles)), "error": None, "category": None}
    except Exception as exc:  # noqa: BLE001 - report, do not crash.
        return {
            "ok": False,
            "rows": 0,
            "error": f"{type(exc).__name__}: {str(exc)[:160]}",
            "category": classify_historical_data_error(exc),
        }


async def _check_news_context() -> dict[str, Any]:
    try:
        from research.news_context_engine import build_news_context

        context = await build_news_context("BTC", limit=3)
        return {
            "available": True,
            "provider_status": context.provider_status,
            "item_count": context.item_count,
            "risk_score": context.risk_score,
            "risk_flags": context.risk_flags,
        }
    except Exception as exc:  # noqa: BLE001 - optional layer must never block ops.
        return {"available": False, "provider_status": f"error:{type(exc).__name__}", "error": str(exc)[:160]}


async def build_healthcheck_report(
    *,
    journal_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    lock_path: str | Path | None = None,
    check_news_context: bool = False,
) -> dict[str, Any]:
    journal = Path(journal_path) if journal_path else default_journal_path()
    registry = Path(registry_path) if registry_path else Path(registry_path_for_choice("crypto_multi"))
    lock = Path(lock_path) if lock_path else default_lock_path()
    summary = build_shadow_summary(journal)
    open_rows = [row for row in summary["signals"] if row.get("status") == "OPEN"]
    now = utc_now()
    overdue = []
    for row in open_rows:
        try:
            if parse_dt(row["expires_at"]) <= now:
                overdue.append(row)
        except Exception:
            overdue.append(row)

    registry_rows = _load_registry_latest(registry)
    completed = sum(1 for row in registry_rows if row.get("status") == "completed")
    classifications: dict[str, int] = {}
    for row in registry_rows:
        classification = str(row.get("classification") or "unknown")
        classifications[classification] = classifications.get(classification, 0) + 1

    telegram_configured = telegram_enabled()
    news_engine_path = PROJECT_ROOT / "research" / "news_context_engine.py"
    cycle_script = (PROJECT_ROOT / "scripts" / "run_shadow_cycle_once.py").read_text(encoding="utf-8")
    market = await _check_market_connectivity()
    git = _git_pending_changes()
    news_context = await _check_news_context() if check_news_context else {"checked": False}

    warnings: list[str] = []
    blockers: list[str] = []
    if not journal.exists():
        warnings.append("shadow_journal_missing")
    if not telegram_configured:
        warnings.append("telegram_env_missing")
    if not registry.exists():
        blockers.append("crypto_multi_registry_missing")
    elif completed != 64:
        warnings.append(f"crypto_multi_completed_count_{completed}")
    if overdue:
        warnings.append("open_signals_due_for_evaluation")
    if lock.exists():
        blockers.append("shadow_cycle_lock_active")
    if git.get("pending"):
        warnings.append("git_pending_changes")
    if not news_engine_path.exists():
        warnings.append("news_context_engine_missing")
    if "--use-news-context" not in cycle_script:
        warnings.append("use_news_context_flag_missing")
    if not market["ok"]:
        warnings.append(f"market_connectivity_{market.get('category') or 'failed'}")

    health_status = "HEALTH_BLOCKED" if blockers else "HEALTH_WARNING" if warnings else "HEALTH_OK"
    return {
        "generated_at": now.isoformat(),
        "health_status": health_status,
        "warnings": warnings,
        "blockers": blockers,
        "shadow_journal": {
            "path": str(journal),
            "exists": journal.exists(),
            "open": summary["summary"].get("open"),
            "closed": summary["summary"].get("closed"),
            "total": summary["summary"].get("total"),
            "overdue_open": len(overdue),
        },
        "latest_shadow_summary_md": _latest_shadow_summary_path(),
        "telegram": {
            "configured": telegram_configured,
            "bot_token_present": bool(os.getenv("TELEGRAM_BOT_TOKEN")),
            "chat_id_present": bool(os.getenv("TELEGRAM_CHAT_ID")),
        },
        "crypto_multi_registry": {
            "path": str(registry),
            "exists": registry.exists(),
            "completed": completed,
            "total_latest": len(registry_rows),
            "classification_counts": classifications,
        },
        "lock_file": {"path": str(lock), "exists": lock.exists()},
        "git": git,
        "news_context": news_context,
        "news_context_engine_exists": news_engine_path.exists(),
        "use_news_context_flag_available": "--use-news-context" in cycle_script,
        "market_connectivity": market,
        "guardrails": {
            "research_only": True,
            "no_real_trading": True,
            "no_exchange_orders": True,
            "no_secrets_printed": True,
        },
    }


def print_healthcheck(report: dict[str, Any], quiet: bool = False) -> None:
    if quiet:
        print(report["health_status"])
        return
    print("Shadow Ops Healthcheck")
    print("Research only. No trading signal.")
    print(f"status: {report['health_status']}")
    print(f"warnings: {report.get('warnings')}")
    print(f"blockers: {report.get('blockers')}")
    journal = report["shadow_journal"]
    print(f"journal: {journal['path']} exists={journal['exists']}")
    print(f"signals open/closed/total: {journal['open']}/{journal['closed']}/{journal['total']}")
    print(f"overdue_open: {journal['overdue_open']}")
    telegram = report["telegram"]
    print(f"telegram_configured: {telegram['configured']}")
    registry = report["crypto_multi_registry"]
    print(f"crypto_multi completed/total: {registry['completed']}/{registry['total_latest']}")
    print(f"market_connectivity_ok: {report['market_connectivity']['ok']}")
    print(f"git_pending_changes: {report['git']['pending']}")
    print(f"use_news_context_flag_available: {report['use_news_context_flag_available']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local Shadow Ops healthcheck.")
    parser.add_argument("--notify-telegram", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--check-news-context", action="store_true")
    parser.add_argument("--test-telegram", action="store_true")
    parser.add_argument("--journal-path", default=None)
    parser.add_argument("--registry-path", default=None)
    parser.add_argument("--lock-path", default=None)
    return parser


async def main() -> None:
    load_project_env()
    args = build_parser().parse_args()
    report = await build_healthcheck_report(
        journal_path=args.journal_path,
        registry_path=args.registry_path,
        lock_path=args.lock_path,
        check_news_context=args.check_news_context,
    )
    if args.test_telegram:
        if not telegram_enabled():
            report["telegram_test"] = {"attempted": False, "sent": False, "reason": "missing_env"}
        else:
            sent = send_telegram_message("TRADEAI Telegram test: notifications configured. Research only.")
            report["telegram_test"] = {"attempted": True, "sent": bool(sent)}
    if args.json_output:
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
    else:
        print_healthcheck(report, quiet=args.quiet)
        if args.test_telegram:
            if report["telegram_test"]["attempted"]:
                print(f"Telegram test sent: {report['telegram_test']['sent']}")
            else:
                print("Telegram test: missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID")
    if args.notify_telegram:
        send_telegram_message(
            "Shadow Ops Healthcheck\n"
            "Research only. No trading signal.\n\n"
            f"Status: {report['health_status']}\n"
            f"Warnings: {', '.join(report['warnings']) or 'none'}\n"
            f"Blockers: {', '.join(report['blockers']) or 'none'}\n"
            f"Open signals: {report['shadow_journal']['open']}\n"
            f"Market connectivity: {report['market_connectivity']['ok']}"
        )


if __name__ == "__main__":
    asyncio.run(main())
