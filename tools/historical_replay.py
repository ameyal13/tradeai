"""Historical replay for generated strategy signals."""
from __future__ import annotations

from typing import Any

import pandas as pd

from tools.historical_data import normalize_ohlcv
from tools.prediction_journal import (
    PredictionStore,
    evaluate_prediction_against_candles,
    metrics_by_strategy_mode,
    normalize_prediction,
)
from tools.strategy_signals import generate_strategy_signal_from_df


def normalize_replay_prediction(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        return normalize_prediction(payload)
    except ValueError as exc:
        if payload.get("strategy_mode") != "xgboost":
            raise
        prediction = normalize_prediction({**payload, "strategy_mode": "model_based"})
        prediction["strategy_mode"] = "xgboost"
        prediction["input_features"] = {
            **(prediction.get("input_features") or {}),
            "normalization_note": "xgboost accepted by historical replay without changing prediction_journal validation",
            "normalization_warning": str(exc),
        }
        return prediction


def _entry_adjusted_prediction(signal: dict[str, Any], symbol: str, timeframe: str, created_at: pd.Timestamp, entry_price: float) -> dict[str, Any]:
    original_entry = float(signal["entry_price"])
    stop_loss = signal.get("stop_loss")
    take_profit = signal.get("take_profit")
    if stop_loss is not None and take_profit is not None and original_entry > 0:
        stop_pct = abs(original_entry - stop_loss) / original_entry
        take_pct = abs(take_profit - original_entry) / original_entry
        if signal["signal"] == "BUY":
            stop_loss = entry_price * (1 - stop_pct)
            take_profit = entry_price * (1 + take_pct)
        elif signal["signal"] == "SELL":
            stop_loss = entry_price * (1 + stop_pct)
            take_profit = entry_price * (1 - take_pct)

    return normalize_replay_prediction({
        "symbol": symbol,
        "timeframe": timeframe,
        "strategy_mode": signal["strategy_mode"],
        "strategy_name": signal["strategy_name"],
        "strategy_version": signal["strategy_version"],
        "signal": signal["signal"],
        "confidence": signal["confidence"],
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "risk_reward_ratio": signal.get("risk_reward_ratio"),
        "horizon_minutes": signal["horizon_minutes"],
        "input_features": signal["input_features"],
        "reasoning": signal["reasoning"],
        "model_provider": signal.get("model_provider"),
        "model_name": signal.get("model_name"),
        "created_at": created_at.isoformat(),
    })


def run_historical_replay(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    strategy_mode: str,
    horizon_candles: int = 3,
    horizon_minutes: int = 60,
    commission_pct: float = 0.001,
    slippage_pct: float = 0.0005,
    spread_pct: float = 0.0003,
    step_size: int = 1,
    min_history: int = 50,
    max_predictions: int | None = None,
    strategy_params: dict[str, Any] | None = None,
    store: PredictionStore | None = None,
) -> dict[str, Any]:
    candles = normalize_ohlcv(df)
    if len(candles) < min_history + horizon_candles + 1:
        return {
            "predictions": [],
            "outcomes": [],
            "metrics": [],
            "assumptions": {"error": "insufficient_data", "min_history": min_history, "horizon_candles": horizon_candles},
        }

    predictions = []
    outcomes = []
    count = 0
    last_n = len(candles) - horizon_candles - 2
    for index_n in range(min_history - 1, last_n + 1, step_size):
        history = candles.iloc[: index_n + 1]
        entry_row = candles.iloc[index_n + 1]
        signal = generate_strategy_signal_from_df(
            history,
            strategy_mode=strategy_mode,
            horizon_minutes=horizon_minutes,
            strategy_params=strategy_params,
        ).to_dict()
        prediction = _entry_adjusted_prediction(
            signal,
            symbol=symbol,
            timeframe=timeframe,
            created_at=candles.iloc[index_n]["timestamp"],
            entry_price=float(entry_row["open"]),
        )
        predictions.append(prediction)
        if store is not None:
            store.create_prediction(prediction)

        if prediction["signal"] != "HOLD":
            future = candles.iloc[index_n + 1:index_n + 1 + horizon_candles]
            outcome = evaluate_prediction_against_candles(
                prediction,
                future,
                commission_pct=commission_pct,
                slippage_pct=slippage_pct,
                spread_pct=spread_pct,
            )
            outcomes.append(outcome)
            if store is not None:
                store.create_outcome(outcome)

        count += 1
        if max_predictions is not None and count >= max_predictions:
            break

    return {
        "predictions": predictions,
        "outcomes": outcomes,
        "metrics": metrics_by_strategy_mode(predictions, outcomes),
        "assumptions": {
            "entry_delay_candles": 1,
            "uses_history_through_index_n": True,
            "horizon_candles": horizon_candles,
            "commission_pct": commission_pct,
            "slippage_pct": slippage_pct,
            "spread_pct": spread_pct,
            "strategy_params": strategy_params or {},
        },
    }
