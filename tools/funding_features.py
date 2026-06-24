"""Funding rate and open-interest research features.

These utilities are research-only. They fetch public Binance Futures data,
align it to OHLCV candles, and produce lag-safe features for offline audits.
Missing or unsupported futures symbols return empty frames/features instead of
failing the whole research run.
"""
from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tools.historical_data import to_millis


FUTURES_BASE = "https://fapi.binance.com"


def normalize_futures_symbol(symbol: str) -> str:
    value = str(symbol).upper().replace("/", "").replace("-", "").strip()
    if value.endswith("USDT"):
        return value
    return f"{value}USDT"


def _utc_datetime(value: datetime | str | int | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, int):
        return datetime.fromtimestamp(value / 1000, tz=timezone.utc)
    if isinstance(value, str):
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        dt = value
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _cache_dir() -> Path:
    return Path(os.getenv("TRADEAI_DATA_DIR", "data")) / "funding_cache"


def _safe_cache_name(parts: list[Any]) -> str:
    raw = "_".join(str(part) for part in parts if part is not None)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)


def _range_cache_path(kind: str, symbol: str, start_time: datetime | str | int | None, end_time: datetime | str | int | None, extra: str = "") -> Path:
    start = _utc_datetime(start_time)
    end = _utc_datetime(end_time)
    start_part = start.strftime("%Y%m%d") if start is not None else "none"
    end_part = end.strftime("%Y%m%d") if end is not None else "none"
    name = _safe_cache_name([normalize_futures_symbol(symbol), kind, extra, start_part, end_part])
    return _cache_dir() / f"{name}.csv"


def _read_cache(path: Path, timestamp_col: str = "timestamp") -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if timestamp_col in df.columns:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)
    return df


def _write_cache(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _empty_funding() -> pd.DataFrame:
    return pd.DataFrame(columns=["timestamp", "funding_rate"])


def _empty_oi() -> pd.DataFrame:
    return pd.DataFrame(columns=["timestamp", "open_interest", "open_interest_value"])


async def fetch_funding_rate_history(
    symbol: str,
    start_time: datetime | str | int | None,
    end_time: datetime | str | int | None,
    limit: int = 1000,
) -> pd.DataFrame:
    """Fetch public Binance Futures funding history with CSV caching."""
    import httpx

    ticker = normalize_futures_symbol(symbol)
    cache_path = _range_cache_path("funding", ticker, start_time, end_time)
    cached = _read_cache(cache_path)
    if cached is not None:
        return cached

    total_limit = max(1, int(limit))
    start_ms = to_millis(start_time)
    end_ms = to_millis(end_time)
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            rows: list[dict[str, Any]] = []
            next_start = start_ms
            while len(rows) < total_limit:
                page_limit = min(1000, total_limit - len(rows))
                params: dict[str, Any] = {"symbol": ticker, "limit": page_limit}
                if next_start is not None:
                    params["startTime"] = next_start
                if end_ms is not None:
                    params["endTime"] = end_ms
                response = await client.get(f"{FUTURES_BASE}/fapi/v1/fundingRate", params=params)
                response.raise_for_status()
                page = response.json()
                if not page:
                    break
                rows.extend(page)
                last_time = int(page[-1].get("fundingTime", 0))
                next_start = last_time + 1
                if len(page) < page_limit or (end_ms is not None and last_time >= end_ms):
                    break
    except Exception as exc:  # noqa: BLE001 - research should continue if futures data is unavailable.
        print(f"warning: funding history unavailable for {ticker}: {type(exc).__name__}: {str(exc)[:160]}")
        return _empty_funding()

    if not rows:
        return _empty_funding()
    df = pd.DataFrame(rows)
    if "fundingTime" not in df.columns or "fundingRate" not in df.columns:
        return _empty_funding()
    out = pd.DataFrame({
        "timestamp": pd.to_datetime(df["fundingTime"].astype("int64"), unit="ms", utc=True),
        "funding_rate": pd.to_numeric(df["fundingRate"], errors="coerce"),
    }).dropna(subset=["timestamp"])
    out = out.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    _write_cache(cache_path, out)
    return out


async def fetch_open_interest_history(
    symbol: str,
    start_time: datetime | str | int | None,
    end_time: datetime | str | int | None,
    period: str = "1h",
    limit: int = 500,
) -> pd.DataFrame:
    """Fetch public Binance Futures open-interest history with CSV caching."""
    import httpx

    ticker = normalize_futures_symbol(symbol)
    cache_path = _range_cache_path("open_interest", ticker, start_time, end_time, period)
    cached = _read_cache(cache_path)
    if cached is not None:
        return cached

    total_limit = max(1, int(limit))
    start_ms = to_millis(start_time)
    end_ms = to_millis(end_time)
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            rows: list[dict[str, Any]] = []
            next_end = end_ms
            while len(rows) < total_limit:
                page_limit = min(500, total_limit - len(rows))
                params: dict[str, Any] = {"symbol": ticker, "period": period, "limit": page_limit}
                if next_end is not None:
                    params["endTime"] = next_end
                response = await client.get(f"{FUTURES_BASE}/futures/data/openInterestHist", params=params)
                response.raise_for_status()
                page = response.json()
                if not page:
                    break
                rows.extend(page)
                earliest_time = int(page[0].get("timestamp", 0))
                next_end = earliest_time - 1
                if start_ms is not None and earliest_time <= start_ms:
                    break
                if len(page) < page_limit or next_end <= 0:
                    break
    except Exception as exc:  # noqa: BLE001
        print(f"warning: open-interest history unavailable for {ticker}: {type(exc).__name__}: {str(exc)[:160]}")
        return _empty_oi()

    if not rows:
        return _empty_oi()
    df = pd.DataFrame(rows)
    if "timestamp" not in df.columns:
        return _empty_oi()
    open_interest = df["sumOpenInterest"] if "sumOpenInterest" in df.columns else pd.Series(np.nan, index=df.index)
    open_interest_value = (
        df["sumOpenInterestValue"] if "sumOpenInterestValue" in df.columns else pd.Series(np.nan, index=df.index)
    )
    out = pd.DataFrame({
        "timestamp": pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True),
        "open_interest": pd.to_numeric(open_interest, errors="coerce"),
        "open_interest_value": pd.to_numeric(open_interest_value, errors="coerce"),
    }).dropna(subset=["timestamp"])
    if start_ms is not None:
        start_ts = pd.to_datetime(start_ms, unit="ms", utc=True)
        out = out[out["timestamp"] >= start_ts]
    if end_ms is not None:
        end_ts = pd.to_datetime(end_ms, unit="ms", utc=True)
        out = out[out["timestamp"] <= end_ts]
    out = out.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    if len(out) > total_limit:
        out = out.tail(total_limit).reset_index(drop=True)
    _write_cache(cache_path, out)
    return out


def _ohlcv_timestamps(ohlcv_df: pd.DataFrame) -> pd.Series:
    if "timestamp" in ohlcv_df.columns:
        return pd.to_datetime(ohlcv_df["timestamp"], utc=True)
    return pd.to_datetime(ohlcv_df.index, utc=True)


def _align_column(ohlcv_df: pd.DataFrame, source_df: pd.DataFrame, column: str) -> pd.Series:
    timestamps = _ohlcv_timestamps(ohlcv_df)
    if source_df is None or source_df.empty or column not in source_df.columns:
        return pd.Series(np.nan, index=ohlcv_df.index, name=column)
    source = source_df[["timestamp", column]].copy()
    source["timestamp"] = pd.to_datetime(source["timestamp"], utc=True)
    source[column] = pd.to_numeric(source[column], errors="coerce")
    left = pd.DataFrame({"timestamp": timestamps, "__row": np.arange(len(ohlcv_df))})
    merged = pd.merge_asof(
        left.sort_values("timestamp"),
        source.dropna(subset=["timestamp"]).sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    ).sort_values("__row")
    return pd.Series(merged[column].to_numpy(), index=ohlcv_df.index, name=column)


def align_funding_to_ohlcv(ohlcv_df: pd.DataFrame, funding_df: pd.DataFrame) -> pd.Series:
    """Forward-fill 8h funding observations onto OHLCV candle timestamps."""
    return _align_column(ohlcv_df, funding_df, "funding_rate").rename("funding_rate")


def compute_funding_features(ohlcv_df: pd.DataFrame, funding_df: pd.DataFrame, oi_df: pd.DataFrame) -> pd.DataFrame:
    """Compute funding/open-interest features aligned to OHLCV timestamps."""
    out = pd.DataFrame(index=ohlcv_df.index)
    funding = funding_df.copy() if funding_df is not None else _empty_funding()
    if not funding.empty and "funding_rate" in funding.columns:
        funding["timestamp"] = pd.to_datetime(funding["timestamp"], utc=True)
        funding["funding_rate"] = pd.to_numeric(funding["funding_rate"], errors="coerce")
        funding["funding_rate_ma3"] = funding["funding_rate"].rolling(3, min_periods=1).mean()
    out["funding_rate"] = align_funding_to_ohlcv(ohlcv_df, funding)
    out["funding_rate_ma3"] = _align_column(ohlcv_df, funding, "funding_rate_ma3")
    out["funding_extreme_long"] = np.where(out["funding_rate"].notna(), (out["funding_rate"] > 0.0005).astype(float), np.nan)
    out["funding_extreme_short"] = np.where(out["funding_rate"].notna(), (out["funding_rate"] < -0.0005).astype(float), np.nan)

    oi = oi_df.copy() if oi_df is not None else _empty_oi()
    aligned_oi = _align_column(ohlcv_df, oi, "open_interest")
    out["oi_change_1h"] = aligned_oi.pct_change(1)
    out["oi_change_4h"] = aligned_oi.pct_change(4)

    close = pd.to_numeric(ohlcv_df["close"], errors="coerce") if "close" in ohlcv_df.columns else pd.Series(np.nan, index=ohlcv_df.index)
    price_return = close.pct_change(1)
    diverge = pd.Series(np.nan, index=ohlcv_df.index, dtype=float)
    valid = price_return.notna() & out["oi_change_1h"].notna()
    diverge.loc[valid] = 0.0
    diverge.loc[valid & (price_return > 0) & (out["oi_change_1h"] < 0)] = 1.0
    diverge.loc[valid & (price_return < 0) & (out["oi_change_1h"] < 0)] = -1.0
    out["oi_price_diverge"] = diverge

    regime = pd.Series(np.nan, index=ohlcv_df.index, dtype=float)
    regime.loc[out["oi_change_4h"] > 0] = 1.0
    regime.loc[out["oi_change_4h"] < 0] = -1.0
    regime.loc[out["oi_change_4h"] == 0] = 0.0
    out["oi_trend_regime"] = regime
    return out
