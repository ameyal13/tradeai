"""Multi-timeframe and BTC context research features.

All joins are as-of joins against closed candles. These features are opt-in for
research grids and do not affect live/shadow signal generation unless a caller
explicitly wires them in.
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from tools.historical_data import fetch_binance_klines


def _timestamps(df: pd.DataFrame) -> pd.Series:
    if "timestamp" in df.columns:
        return pd.to_datetime(df["timestamp"], utc=True)
    return pd.to_datetime(df.index, utc=True)


def _ema(series: pd.Series, span: int) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").ewm(span=span, adjust=False, min_periods=span).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = pd.to_numeric(close, errors="coerce").diff()
    gain = delta.clip(lower=0).rolling(period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).rolling(period, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


async def fetch_4h_candles_aligned(
    symbol: str,
    ohlcv_1h_df: pd.DataFrame,
    lookback_4h: int = 200,
) -> pd.DataFrame:
    """Fetch enough 4h candles to cover the 1h range plus indicator lookback."""
    if ohlcv_1h_df.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    times = _timestamps(ohlcv_1h_df)
    start = times.min() - timedelta(hours=4 * int(lookback_4h))
    end = times.max()
    needed = int(((end - start).total_seconds() // (4 * 3600)) + 10)
    return await fetch_binance_klines(symbol, "4h", start_time=start, end_time=end, limit=max(lookback_4h, needed))


def compute_4h_features(ohlcv_1h_df: pd.DataFrame, ohlcv_4h_df: pd.DataFrame) -> pd.DataFrame:
    """Compute closed-4h trend features aligned to 1h candles."""
    out = pd.DataFrame(index=ohlcv_1h_df.index)
    for column in ["tf4h_ema20", "tf4h_ema50", "tf4h_trend", "tf4h_trend_strength", "tf4h_rsi", "tf4h_atr_ratio", "tf4h_1h_aligned"]:
        out[column] = np.nan
    if ohlcv_1h_df.empty or ohlcv_4h_df is None or ohlcv_4h_df.empty:
        return out

    four_h = ohlcv_4h_df.copy()
    four_h["timestamp"] = _timestamps(four_h)
    four_h["feature_time"] = four_h["timestamp"] + pd.Timedelta(hours=4)
    four_h["tf4h_ema20"] = _ema(four_h["close"], 20)
    four_h["tf4h_ema50"] = _ema(four_h["close"], 50)
    close = pd.to_numeric(four_h["close"], errors="coerce").replace(0, np.nan)
    diff = four_h["tf4h_ema20"] - four_h["tf4h_ema50"]
    neutral = diff.abs() < close * 0.001
    four_h["tf4h_trend"] = np.where(neutral, 0.0, np.where(diff > 0, 1.0, -1.0))
    four_h.loc[diff.isna(), "tf4h_trend"] = np.nan
    four_h["tf4h_trend_strength"] = diff / close
    four_h["tf4h_rsi"] = _rsi(four_h["close"])
    four_h["tf4h_atr_ratio"] = _atr(four_h) / close

    left = pd.DataFrame({"timestamp": _timestamps(ohlcv_1h_df), "__row": np.arange(len(ohlcv_1h_df))})
    right_cols = ["feature_time", "tf4h_ema20", "tf4h_ema50", "tf4h_trend", "tf4h_trend_strength", "tf4h_rsi", "tf4h_atr_ratio"]
    merged = pd.merge_asof(
        left.sort_values("timestamp"),
        four_h[right_cols].dropna(subset=["feature_time"]).sort_values("feature_time"),
        left_on="timestamp",
        right_on="feature_time",
        direction="backward",
    ).sort_values("__row")

    aligned = merged[["tf4h_ema20", "tf4h_ema50", "tf4h_trend", "tf4h_trend_strength", "tf4h_rsi", "tf4h_atr_ratio"]]
    out.loc[:, aligned.columns] = aligned.to_numpy()

    one_h_close = pd.to_numeric(ohlcv_1h_df["close"], errors="coerce")
    one_h_fast = _ema(one_h_close, 12)
    one_h_slow = _ema(one_h_close, 26)
    one_h_trend = np.where(one_h_fast > one_h_slow, 1.0, -1.0)
    one_h_trend[pd.isna(one_h_fast) | pd.isna(one_h_slow)] = np.nan
    tf_trend = out["tf4h_trend"].to_numpy(dtype=float)
    out["tf4h_1h_aligned"] = np.where(
        np.isfinite(tf_trend) & (tf_trend != 0) & np.isfinite(one_h_trend),
        (tf_trend == one_h_trend).astype(float),
        np.nan,
    )
    return out


def compute_btc_context_features(ohlcv_1h_df: pd.DataFrame, btc_1h_df: pd.DataFrame) -> pd.DataFrame:
    """Compute lagged BTC context aligned to the asset 1h candles."""
    columns = ["btc_return_1h", "btc_return_4h", "btc_ema_trend", "asset_btc_corr_20", "asset_btc_diverge"]
    out = pd.DataFrame(index=ohlcv_1h_df.index)
    for column in columns:
        out[column] = np.nan
    if ohlcv_1h_df.empty or btc_1h_df is None or btc_1h_df.empty:
        return out

    btc = btc_1h_df.copy()
    btc["timestamp"] = _timestamps(btc)
    btc_close = pd.to_numeric(btc["close"], errors="coerce")
    btc["btc_return_1h"] = btc_close.pct_change(1).shift(1)
    btc["btc_return_4h"] = btc_close.pct_change(4).shift(1)
    btc_fast = _ema(btc_close, 12)
    btc_slow = _ema(btc_close, 26)
    btc["btc_ema_trend"] = np.where(btc_fast > btc_slow, 1.0, -1.0)
    btc.loc[btc_fast.isna() | btc_slow.isna(), "btc_ema_trend"] = np.nan
    btc["btc_ema_trend"] = btc["btc_ema_trend"].shift(1)

    left = pd.DataFrame({"timestamp": _timestamps(ohlcv_1h_df), "__row": np.arange(len(ohlcv_1h_df))})
    merged = pd.merge_asof(
        left.sort_values("timestamp"),
        btc[["timestamp", "close", "btc_return_1h", "btc_return_4h", "btc_ema_trend"]].sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    ).sort_values("__row")
    out["btc_return_1h"] = merged["btc_return_1h"].to_numpy()
    out["btc_return_4h"] = merged["btc_return_4h"].to_numpy()
    out["btc_ema_trend"] = merged["btc_ema_trend"].to_numpy()

    asset_return = pd.to_numeric(ohlcv_1h_df["close"], errors="coerce").pct_change(1).shift(1)
    btc_return = pd.Series(merged["close"].to_numpy(), index=ohlcv_1h_df.index).pct_change(1).shift(1)
    corr = asset_return.rolling(20, min_periods=20).corr(btc_return)
    out["asset_btc_corr_20"] = corr
    out["asset_btc_diverge"] = np.where(corr.notna(), (corr.abs() < 0.3).astype(float), np.nan)
    return out
