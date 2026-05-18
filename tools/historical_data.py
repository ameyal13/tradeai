"""Historical OHLCV data layer for replay, labels, and validation."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd


BINANCE_BASE = "https://api.binance.com/api/v3"
SYMBOL_MAP = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "BNB": "BNBUSDT",
    "SOL": "SOLUSDT",
    "XRP": "XRPUSDT",
    "ADA": "ADAUSDT",
    "DOGE": "DOGEUSDT",
    "AVAX": "AVAXUSDT",
    "LINK": "LINKUSDT",
    "DOT": "DOTUSDT",
}


def to_millis(value: datetime | str | int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        dt = value
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def normalize_klines(raw: list[list[Any]]) -> pd.DataFrame:
    columns = [
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore",
    ]
    df = pd.DataFrame(raw, columns=columns[: len(raw[0])] if raw else columns)
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    for column in ["open", "high", "low", "close", "volume"]:
        df[column] = df[column].astype(float)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "open_time" in df.columns and "timestamp" not in df.columns:
        df = df.rename(columns={"open_time": "timestamp"})
    if "timestamp" not in df.columns:
        df = df.reset_index().rename(columns={"index": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    for column in ["open", "high", "low", "close", "volume"]:
        if column not in df.columns:
            raise ValueError(f"Missing OHLCV column: {column}")
        df[column] = df[column].astype(float)
    return df[["timestamp", "open", "high", "low", "close", "volume"]].drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


async def fetch_binance_klines(
    symbol: str,
    interval: str,
    start_time: datetime | str | int | None = None,
    end_time: datetime | str | int | None = None,
    limit: int = 1000,
) -> pd.DataFrame:
    """Fetch Binance klines with pagination beyond the 1000-candle API limit."""
    import httpx

    ticker = SYMBOL_MAP.get(symbol.upper(), symbol.upper() + "USDT")
    start_ms = to_millis(start_time)
    end_ms = to_millis(end_time)
    remaining = limit
    all_rows: list[list[Any]] = []
    next_start = start_ms

    async with httpx.AsyncClient(timeout=30) as client:
        while remaining > 0:
            page_limit = min(1000, remaining)
            params: dict[str, Any] = {"symbol": ticker, "interval": interval, "limit": page_limit}
            if next_start is not None:
                params["startTime"] = next_start
            if end_ms is not None:
                params["endTime"] = end_ms
            response = await client.get(f"{BINANCE_BASE}/klines", params=params)
            response.raise_for_status()
            rows = response.json()
            if not rows:
                break
            all_rows.extend(rows)
            remaining -= len(rows)
            last_open = int(rows[-1][0])
            next_start = last_open + 1
            if len(rows) < page_limit or (end_ms is not None and last_open >= end_ms):
                break

    return normalize_klines(all_rows).head(limit)
