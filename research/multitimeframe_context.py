"""Multi-timeframe context diagnostics for evaluated shadow signals.

This module is read-only research code. It does not generate signals, train
models, write Supabase rows, or change trade decisions.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Literal

import pandas as pd

from tools.historical_data import fetch_binance_klines


Trend = Literal["bullish", "bearish", "neutral"]


def normalize_symbol(symbol: str) -> str:
    value = str(symbol or "").upper().strip()
    for suffix in ("/USDT", "-USDT", "USDT"):
        if value.endswith(suffix):
            return value[: -len(suffix)]
    return value


def normalize_signal_side(side: str) -> str:
    value = str(side or "").upper().strip()
    if value in {"BUY", "LONG"}:
        return "LONG"
    if value in {"SELL", "SHORT"}:
        return "SHORT"
    return value


def ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def trend_from_ema20_ema50(candles: pd.DataFrame) -> Trend:
    """Classify trend using EMA20 vs EMA50 and a 0.1% neutral band."""
    if candles.empty or "close" not in candles.columns:
        return "neutral"
    close = pd.to_numeric(candles["close"], errors="coerce").dropna()
    if len(close) < 50:
        return "neutral"
    ema20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
    ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
    price = close.iloc[-1]
    if price == 0 or pd.isna(price) or pd.isna(ema20) or pd.isna(ema50):
        return "neutral"
    diff = float(ema20 - ema50)
    neutral_band = abs(float(price)) * 0.001
    if abs(diff) < neutral_band:
        return "neutral"
    return "bullish" if diff > 0 else "bearish"


def is_trend_aligned(trend: str, signal_side: str) -> bool:
    side = normalize_signal_side(signal_side)
    if trend == "bullish" and side == "LONG":
        return True
    if trend == "bearish" and side == "SHORT":
        return True
    return False


async def _fetch_4h_context_candles(symbol: str, signal_time: datetime, lookback_candles: int) -> pd.DataFrame:
    end_time = ensure_utc(signal_time)
    start_time = end_time - timedelta(hours=4 * int(lookback_candles))
    return await fetch_binance_klines(
        normalize_symbol(symbol),
        interval="4h",
        start_time=start_time,
        end_time=end_time,
        limit=int(lookback_candles),
    )


async def compute_4h_context(
    symbol: str,
    signal_time: datetime,
    signal_side: str,
    lookback_candles: int = 50,
) -> dict[str, Any]:
    """Compute asset and BTC 4h trend alignment at a shadow signal timestamp."""
    if lookback_candles <= 0:
        raise ValueError("lookback_candles must be positive")

    asset_candles = await _fetch_4h_context_candles(symbol, signal_time, lookback_candles)
    btc_candles = await _fetch_4h_context_candles("BTC", signal_time, lookback_candles)
    asset_trend = trend_from_ema20_ema50(asset_candles)
    btc_trend = trend_from_ema20_ema50(btc_candles)
    asset_aligned = is_trend_aligned(asset_trend, signal_side)
    btc_aligned = is_trend_aligned(btc_trend, signal_side)
    return {
        "asset_4h_trend": asset_trend,
        "asset_trend_aligned": asset_aligned,
        "btc_4h_trend": btc_trend,
        "btc_trend_aligned": btc_aligned,
        "full_alignment": asset_aligned and btc_aligned,
    }
