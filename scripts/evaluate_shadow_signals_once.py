"""Evaluate local shadow/paper signals once.

Research only. Uses prediction_journal evaluator semantics and never places
orders or writes Supabase.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.telegram_notifier import format_shadow_signal_evaluated, send_telegram_message  # noqa: E402
from tools.historical_data import classify_historical_data_error  # noqa: E402
from tools.prediction_journal import fetch_future_klines, parse_dt, utc_now  # noqa: E402
from tools.runtime_env import load_project_env  # noqa: E402
from tools.shadow_signal_journal import (  # noqa: E402
    DEFAULT_SHADOW_JOURNAL_PATH,
    OPEN,
    ShadowSignalJournal,
    evaluate_shadow_signal_with_candles,
)


def _is_retryable_fetch_error(exc: Exception) -> bool:
    return classify_historical_data_error(exc) in {
        "dns_resolution",
        "timeout",
        "network",
        "rate_limited",
        "endpoint_server_error",
    }


async def fetch_future_klines_with_retry(
    symbol: str,
    timeframe: str,
    start: Any,
    end: Any,
    retries: int = 2,
    backoff_seconds: float = 0.75,
) -> Any:
    """Fetch future candles with bounded retry/backoff for transient failures."""
    attempts = max(1, int(retries) + 1)
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return await fetch_future_klines(symbol, timeframe, start, end)
        except Exception as exc:  # noqa: BLE001 - network boundary.
            last_error = exc
            if attempt >= attempts - 1 or not _is_retryable_fetch_error(exc):
                raise
            await asyncio.sleep(float(backoff_seconds) * (2 ** attempt))
    raise last_error or RuntimeError("unknown fetch error")


async def evaluate_shadow_signals_once(
    journal_path: str | Path = DEFAULT_SHADOW_JOURNAL_PATH,
    notify_telegram: bool = False,
    retries: int = 2,
    backoff_seconds: float = 0.75,
    signal_store: Any | None = None,
) -> dict[str, Any]:
    journal = signal_store or ShadowSignalJournal(journal_path)
    open_signals = journal.list_signals(status=OPEN)
    result = {
        "found_open": len(open_signals),
        "closed": 0,
        "still_open": 0,
        "errors": [],
        "telegram_errors": [],
        "closed_signals": [],
    }
    now = utc_now()
    for signal in open_signals:
        try:
            start = parse_dt(signal["generated_at"])
            end = min(now, parse_dt(signal["expires_at"]))
            if end <= start:
                result["still_open"] += 1
                continue
            candles = await fetch_future_klines_with_retry(
                signal["symbol"],
                signal["timeframe"],
                start,
                end,
                retries=retries,
                backoff_seconds=backoff_seconds,
            )
            updates = evaluate_shadow_signal_with_candles(signal, candles, now=now)
            if updates is None:
                result["still_open"] += 1
                continue
            closed = journal.update_signal(signal, updates)
            result["closed"] += 1
            result["closed_signals"].append(closed)
            if notify_telegram:
                try:
                    send_telegram_message(format_shadow_signal_evaluated(closed))
                except Exception as exc:  # noqa: BLE001 - notifications cannot break evaluation.
                    result["telegram_errors"].append({
                        "shadow_signal_id": signal.get("shadow_signal_id"),
                        "error_type": type(exc).__name__,
                        "error": f"{type(exc).__name__}: {exc}",
                    })
        except Exception as exc:  # noqa: BLE001 - keep evaluating other signals.
            category = classify_historical_data_error(exc)
            result["errors"].append({
                "shadow_signal_id": signal.get("shadow_signal_id"),
                "symbol": signal.get("symbol"),
                "timeframe": signal.get("timeframe"),
                "error_type": type(exc).__name__,
                "error_category": category,
                "error_code": "network_error_retry_later" if _is_retryable_fetch_error(exc) else "evaluation_error",
                "signal_left_open": True,
                "error": f"{type(exc).__name__}: {exc}",
            })
            result["still_open"] += 1
    return result


def print_result(result: dict[str, Any]) -> None:
    print("Evaluate shadow signals")
    print(f"found_open: {result['found_open']}")
    print(f"closed: {result['closed']}")
    print(f"still_open: {result['still_open']}")
    print(f"errors: {len(result['errors'])}")
    print(f"telegram_errors: {len(result.get('telegram_errors') or [])}")
    for row in result["closed_signals"]:
        print(
            " | ".join([
                f"id={row.get('shadow_signal_id')}",
                f"symbol={row.get('symbol')}",
                f"outcome={row.get('outcome')}",
                f"pnl_pct={row.get('pnl_pct')}",
                f"exit_reason={row.get('exit_reason')}",
            ])
        )
    for error in result["errors"]:
        print(
            f"error {error.get('shadow_signal_id')}: "
            f"{error.get('error_code')} | {error.get('error_category')} | {error.get('error')}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate local shadow signals once.")
    parser.add_argument("--journal-path", default=str(DEFAULT_SHADOW_JOURNAL_PATH))
    parser.add_argument("--notify-telegram", action="store_true")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--backoff-seconds", type=float, default=0.75)
    return parser


async def main() -> None:
    load_project_env()
    args = build_parser().parse_args()
    print_result(await evaluate_shadow_signals_once(
        args.journal_path,
        notify_telegram=args.notify_telegram,
        retries=args.retries,
        backoff_seconds=args.backoff_seconds,
    ))


if __name__ == "__main__":
    asyncio.run(main())
