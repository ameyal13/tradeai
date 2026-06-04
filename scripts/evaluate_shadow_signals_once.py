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
from tools.prediction_journal import fetch_future_klines, parse_dt, utc_now  # noqa: E402
from tools.shadow_signal_journal import (  # noqa: E402
    DEFAULT_SHADOW_JOURNAL_PATH,
    OPEN,
    ShadowSignalJournal,
    evaluate_shadow_signal_with_candles,
)


async def evaluate_shadow_signals_once(
    journal_path: str | Path = DEFAULT_SHADOW_JOURNAL_PATH,
    notify_telegram: bool = False,
) -> dict[str, Any]:
    journal = ShadowSignalJournal(journal_path)
    open_signals = journal.list_signals(status=OPEN)
    result = {
        "found_open": len(open_signals),
        "closed": 0,
        "still_open": 0,
        "errors": [],
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
            candles = await fetch_future_klines(signal["symbol"], signal["timeframe"], start, end)
            updates = evaluate_shadow_signal_with_candles(signal, candles, now=now)
            if updates is None:
                result["still_open"] += 1
                continue
            closed = journal.update_signal(signal, updates)
            result["closed"] += 1
            result["closed_signals"].append(closed)
            if notify_telegram:
                send_telegram_message(format_shadow_signal_evaluated(closed))
        except Exception as exc:  # noqa: BLE001 - keep evaluating other signals.
            result["errors"].append({
                "shadow_signal_id": signal.get("shadow_signal_id"),
                "symbol": signal.get("symbol"),
                "error": f"{type(exc).__name__}: {exc}",
            })
    return result


def print_result(result: dict[str, Any]) -> None:
    print("Evaluate shadow signals")
    print(f"found_open: {result['found_open']}")
    print(f"closed: {result['closed']}")
    print(f"still_open: {result['still_open']}")
    print(f"errors: {len(result['errors'])}")
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
        print(f"error {error.get('shadow_signal_id')}: {error.get('error')}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate local shadow signals once.")
    parser.add_argument("--journal-path", default=str(DEFAULT_SHADOW_JOURNAL_PATH))
    parser.add_argument("--notify-telegram", action="store_true")
    return parser


async def main() -> None:
    args = build_parser().parse_args()
    print_result(await evaluate_shadow_signals_once(args.journal_path, notify_telegram=args.notify_telegram))


if __name__ == "__main__":
    asyncio.run(main())
