"""Deterministic market-context review for shadow signals.

This module does not generate trades and does not modify trade levels. It
summarizes technical context that can be measured later: trend alignment,
nearby support/resistance, volatility regime, volume confirmation, and an
optional benchmark/BTC trend check.
"""
from __future__ import annotations

from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field


ContextStatus = Literal["APPROVE", "CAUTION", "BLOCK"]


class MarketContextResult(BaseModel):
    context_status: ContextStatus = "APPROVE"
    confidence_adjustment: int = Field(default=0, ge=-10, le=5)
    risk_flags: list[str] = Field(default_factory=list)
    context_summary: str
    metrics: dict[str, Any] = Field(default_factory=dict)
    can_modify_trade_levels: bool = False
    research_only: bool = True


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_candles(candles: pd.DataFrame) -> pd.DataFrame:
    df = candles.copy()
    if "timestamp" not in df.columns:
        df = df.reset_index().rename(columns={"index": "timestamp"})
    required = ["open", "high", "low", "close", "volume"]
    for column in required:
        if column not in df.columns:
            df[column] = 0.0
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.sort_values("timestamp")
    return df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)


def _atr_pct(df: pd.DataFrame, period: int = 14) -> pd.Series:
    previous_close = df["close"].shift(1)
    true_range = pd.concat([
        df["high"] - df["low"],
        (df["high"] - previous_close).abs(),
        (df["low"] - previous_close).abs(),
    ], axis=1).max(axis=1)
    atr = true_range.rolling(period, min_periods=period).mean()
    return atr / df["close"].replace(0, pd.NA) * 100


def _trend_from_close(close: float, ema20: float | None, ema50: float | None) -> str:
    if ema20 is None or ema50 is None:
        return "unknown"
    if close > ema50 and ema20 > ema50:
        return "bullish"
    if close < ema50 and ema20 < ema50:
        return "bearish"
    return "mixed"


def build_market_context(
    *,
    candles: pd.DataFrame,
    symbol: str,
    timeframe: str,
    side: Literal["LONG", "SHORT"],
    entry_price: float,
    stop_loss: float | None = None,
    take_profit: float | None = None,
    benchmark_candles: pd.DataFrame | None = None,
) -> MarketContextResult:
    """Build a bounded technical review for an already-generated signal."""
    df = _normalize_candles(candles)
    if len(df) < 60:
        return MarketContextResult(
            context_status="CAUTION",
            confidence_adjustment=-3,
            risk_flags=["insufficient_market_context_history"],
            context_summary=f"{symbol} {timeframe}: insufficient candles for market context.",
            metrics={"rows": len(df)},
        )

    close = float(df["close"].iloc[-1])
    entry = float(entry_price)
    ema20_series = df["close"].ewm(span=20, adjust=False).mean()
    ema50_series = df["close"].ewm(span=50, adjust=False).mean()
    ema20 = _float_or_none(ema20_series.iloc[-1])
    ema50 = _float_or_none(ema50_series.iloc[-1])
    atr_series = _atr_pct(df)
    current_atr_pct = _float_or_none(atr_series.iloc[-1])
    atr_window = atr_series.dropna().tail(120)
    atr_percentile = None
    if current_atr_pct is not None and len(atr_window) >= 10:
        atr_percentile = float((atr_window <= current_atr_pct).mean())

    lookback = min(80, len(df))
    recent = df.tail(lookback)
    support = float(recent["low"].min())
    resistance = float(recent["high"].max())
    distance_to_support_pct = (entry - support) / entry * 100 if entry else None
    distance_to_resistance_pct = (resistance - entry) / entry * 100 if entry else None
    volume_mean = _float_or_none(df["volume"].tail(20).mean())
    volume_relative = float(df["volume"].iloc[-1] / volume_mean) if volume_mean and volume_mean > 0 else None
    trend = _trend_from_close(close, ema20, ema50)

    risk_flags: list[str] = []
    if side == "LONG" and trend == "bearish":
        risk_flags.append("long_against_local_trend")
    if side == "SHORT" and trend == "bullish":
        risk_flags.append("short_against_local_trend")
    if current_atr_pct is not None and atr_percentile is not None and atr_percentile >= 0.85 and current_atr_pct >= 3.0:
        risk_flags.append("high_volatility_regime")
    if volume_relative is not None and volume_relative < 0.70:
        risk_flags.append("low_relative_volume")

    atr_buffer = max(float(current_atr_pct or 0), 0.75)
    if side == "LONG" and trend != "bullish" and distance_to_resistance_pct is not None and distance_to_resistance_pct < atr_buffer:
        risk_flags.append("long_near_resistance")
    if side == "SHORT" and trend != "bearish" and distance_to_support_pct is not None and distance_to_support_pct < atr_buffer:
        risk_flags.append("short_near_support")

    benchmark_trend = None
    if benchmark_candles is not None and len(benchmark_candles) >= 60:
        benchmark_df = _normalize_candles(benchmark_candles)
        if len(benchmark_df) >= 60:
            benchmark_close = float(benchmark_df["close"].iloc[-1])
            benchmark_ema20 = _float_or_none(benchmark_df["close"].ewm(span=20, adjust=False).mean().iloc[-1])
            benchmark_ema50 = _float_or_none(benchmark_df["close"].ewm(span=50, adjust=False).mean().iloc[-1])
            benchmark_trend = _trend_from_close(benchmark_close, benchmark_ema20, benchmark_ema50)
            if side == "LONG" and benchmark_trend == "bearish":
                risk_flags.append("benchmark_bearish_for_long")
            if side == "SHORT" and benchmark_trend == "bullish":
                risk_flags.append("benchmark_bullish_for_short")

    severe_count = sum(
        1 for flag in risk_flags
        if flag in {
            "long_against_local_trend",
            "short_against_local_trend",
            "long_near_resistance",
            "short_near_support",
            "benchmark_bearish_for_long",
            "benchmark_bullish_for_short",
        }
    )
    if severe_count >= 3:
        status: ContextStatus = "BLOCK"
        adjustment = -10
    elif risk_flags:
        status = "CAUTION"
        adjustment = -5
    else:
        status = "APPROVE"
        adjustment = 0

    metrics = {
        "rows": len(df),
        "close": round(close, 8),
        "ema20": round(ema20, 8) if ema20 is not None else None,
        "ema50": round(ema50, 8) if ema50 is not None else None,
        "trend": trend,
        "atr_pct": round(current_atr_pct, 6) if current_atr_pct is not None else None,
        "atr_percentile": round(atr_percentile, 6) if atr_percentile is not None else None,
        "support": round(support, 8),
        "resistance": round(resistance, 8),
        "distance_to_support_pct": round(distance_to_support_pct, 6) if distance_to_support_pct is not None else None,
        "distance_to_resistance_pct": round(distance_to_resistance_pct, 6) if distance_to_resistance_pct is not None else None,
        "volume_relative": round(volume_relative, 6) if volume_relative is not None else None,
        "benchmark_trend": benchmark_trend,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
    }
    summary = (
        f"{symbol.upper()} {timeframe} {side}: trend={trend}, "
        f"ATR%={metrics['atr_pct']}, vol_rel={metrics['volume_relative']}, "
        f"support={metrics['support']}, resistance={metrics['resistance']}."
    )
    return MarketContextResult(
        context_status=status,
        confidence_adjustment=adjustment,
        risk_flags=risk_flags,
        context_summary=summary,
        metrics=metrics,
    )
