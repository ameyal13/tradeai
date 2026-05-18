"""Build supervised trade labels from historical OHLCV candles."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from tools.historical_data import normalize_ohlcv


def _levels(entry: float, side: str, stop_loss_pct: float, take_profit_pct: float) -> tuple[float, float]:
    if side == "BUY":
        return entry * (1 - stop_loss_pct), entry * (1 + take_profit_pct)
    return entry * (1 + stop_loss_pct), entry * (1 - take_profit_pct)


def _net_return_pct(entry: float, exit_price: float, side: str, commission_pct: float, slippage_pct: float, spread_pct: float) -> float:
    raw = (exit_price - entry) / entry * 100 if side == "BUY" else (entry - exit_price) / entry * 100
    costs = ((entry + exit_price) * (commission_pct + slippage_pct + spread_pct / 2)) / entry * 100
    return raw - costs


def label_trade_at_index(
    df: pd.DataFrame,
    index_n: int,
    side: str,
    horizon_candles: int,
    stop_loss_pct: float = 0.03,
    take_profit_pct: float = 0.045,
    commission_pct: float = 0.001,
    slippage_pct: float = 0.0005,
    spread_pct: float = 0.0003,
) -> dict[str, Any]:
    candles = normalize_ohlcv(df)
    side = side.upper()
    if side not in {"BUY", "SELL"}:
        return {"outcome": "INVALID_DATA", "tp_before_sl": np.nan, "net_return_positive": 0, "notes": ["invalid_side"]}
    entry_idx = index_n + 1
    if index_n < 0 or entry_idx >= len(candles) or horizon_candles <= 0:
        return {"outcome": "INVALID_DATA", "tp_before_sl": np.nan, "net_return_positive": 0, "notes": ["insufficient_data"]}

    entry = float(candles.iloc[entry_idx]["open"])
    stop_loss, take_profit = _levels(entry, side, stop_loss_pct, take_profit_pct)
    path = candles.iloc[entry_idx:min(len(candles), entry_idx + horizon_candles)]
    if path.empty:
        return {"outcome": "INVALID_DATA", "tp_before_sl": np.nan, "net_return_positive": 0, "notes": ["insufficient_data"]}

    notes = []
    exit_price = float(path.iloc[-1]["close"])
    outcome = "EXPIRED"
    tp_before_sl = np.nan

    for _, row in path.iterrows():
        high = float(row["high"])
        low = float(row["low"])
        if side == "BUY":
            hit_sl = low <= stop_loss
            hit_tp = high >= take_profit
        else:
            hit_sl = high >= stop_loss
            hit_tp = low <= take_profit

        if hit_sl and hit_tp:
            outcome = "LOSS"
            tp_before_sl = 0
            exit_price = stop_loss
            notes.append("ambiguous_intrabar_conservative_loss")
            break
        if hit_sl:
            outcome = "LOSS"
            tp_before_sl = 0
            exit_price = stop_loss
            break
        if hit_tp:
            outcome = "WIN"
            tp_before_sl = 1
            exit_price = take_profit
            break

    net_return = _net_return_pct(entry, exit_price, side, commission_pct, slippage_pct, spread_pct)
    if outcome == "EXPIRED" and abs(net_return) <= 0.05:
        outcome = "BREAKEVEN"

    return {
        "entry_index": entry_idx,
        "entry_time": candles.iloc[entry_idx]["timestamp"].isoformat(),
        "entry_price": entry,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "exit_price": exit_price,
        "tp_before_sl": tp_before_sl,
        "net_return_positive": int(net_return > 0),
        "net_return_pct": net_return,
        "outcome": outcome,
        "side": side,
        "notes": notes,
    }


def build_trade_labels(
    df: pd.DataFrame,
    side: str,
    horizon_candles: int,
    **kwargs: Any,
) -> pd.DataFrame:
    candles = normalize_ohlcv(df)
    rows = []
    for index_n in range(len(candles)):
        label = label_trade_at_index(candles, index_n, side, horizon_candles, **kwargs)
        rows.append({"timestamp": candles.iloc[index_n]["timestamp"], **label})
    return pd.DataFrame(rows)
