"""Generate one manual paper-trading signal batch.

This is a safe, manual loop helper. It records measurable predictions only; it
does not place orders, schedule jobs, or require LLM API keys.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.trading_agent import generate_trading_signal
from main import prediction_store
from tools.prediction_journal import PredictionStore, parse_dt, prediction_payload_from_signal_response, utc_now


DEFAULT_SYMBOLS = ["BTC", "ETH", "SOL"]
DEFAULT_TIMEFRAME = "1h"
DEFAULT_STRATEGY_MODE = "deterministic"
DEFAULT_PROVIDER = "none"
DEFAULT_HORIZON_MINUTES = 60


def has_recent_pending_prediction(
    store: PredictionStore,
    symbol: str,
    timeframe: str,
    strategy_mode: str,
    horizon_minutes: int,
) -> bool:
    cutoff = utc_now().timestamp() - (horizon_minutes * 60)
    pending = store.list_predictions(status="pending", symbol=symbol, limit=100)
    for prediction in pending:
        if prediction.get("timeframe") != timeframe:
            continue
        if prediction.get("strategy_mode") != strategy_mode:
            continue
        if parse_dt(prediction["created_at"]).timestamp() >= cutoff:
            return True
    return False


async def generate_signals_once(
    symbols: list[str] | None = None,
    timeframe: str = DEFAULT_TIMEFRAME,
    strategy_mode: str = DEFAULT_STRATEGY_MODE,
    provider: str = DEFAULT_PROVIDER,
    horizon_minutes: int = DEFAULT_HORIZON_MINUTES,
    store: PredictionStore | None = None,
) -> list[dict[str, Any]]:
    active_store = store or prediction_store
    results = []

    for symbol in symbols or DEFAULT_SYMBOLS:
        row: dict[str, Any] = {
            "symbol": symbol.upper(),
            "timeframe": timeframe,
            "strategy_mode": strategy_mode,
        }
        try:
            if has_recent_pending_prediction(active_store, symbol, timeframe, strategy_mode, horizon_minutes):
                row["status"] = "skipped_recent_pending"
                results.append(row)
                continue

            signal = await generate_trading_signal(
                symbol,
                timeframe,
                provider,
                strategy_mode=strategy_mode,
                horizon_minutes=horizon_minutes,
            )
            if "error" in signal:
                row["status"] = "error"
                row["error"] = signal["error"]
                results.append(row)
                continue

            prediction = active_store.create_prediction(
                prediction_payload_from_signal_response(
                    signal,
                    timeframe=timeframe,
                    requested_strategy_mode=strategy_mode,
                    requested_horizon_minutes=horizon_minutes,
                    provider=provider,
                )
            )
            row.update({
                "signal": prediction["signal"],
                "confidence": prediction["confidence"],
                "entry_price": prediction["entry_price"],
                "stop_loss": prediction["stop_loss"],
                "take_profit": prediction["take_profit"],
                "prediction_id": prediction["id"],
                "status": prediction["status"],
            })
        except Exception as exc:
            row["status"] = "error"
            row["error"] = str(exc)
        results.append(row)

    return results


def print_results(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        print(
            " | ".join([
                f"symbol={row.get('symbol')}",
                f"timeframe={row.get('timeframe')}",
                f"strategy_mode={row.get('strategy_mode')}",
                f"signal={row.get('signal', '')}",
                f"confidence={row.get('confidence', '')}",
                f"entry_price={row.get('entry_price', '')}",
                f"stop_loss={row.get('stop_loss', '')}",
                f"take_profit={row.get('take_profit', '')}",
                f"prediction_id={row.get('prediction_id', '')}",
                f"status={row.get('status')}",
                f"error={row.get('error', '')}",
            ])
        )


async def main() -> None:
    print_results(await generate_signals_once())


if __name__ == "__main__":
    asyncio.run(main())
