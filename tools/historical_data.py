"""Historical OHLCV data layer for replay, labels, and validation."""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


BINANCE_BASE = "https://api.binance.com/api/v3"
BINANCE_DATA_API_BASE = "https://data-api.binance.vision/api/v3"
BINANCE_KLINE_BASES = [BINANCE_BASE, BINANCE_DATA_API_BASE]
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


class HistoricalDataError(RuntimeError):
    """Typed error so experiment reports can separate network, API, and data issues."""

    def __init__(self, category: str, message: str, original: Exception | None = None):
        super().__init__(message)
        self.category = category
        self.original = original


@dataclass(frozen=True)
class CachedOHLCV:
    df: pd.DataFrame
    source: str
    cache_path: str | None = None
    warning: str | None = None


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


def binance_symbol(symbol: str) -> str:
    return SYMBOL_MAP.get(symbol.upper(), symbol.upper() + "USDT")


def classify_historical_data_error(exc: Exception) -> str:
    if isinstance(exc, HistoricalDataError):
        return exc.category
    name = exc.__class__.__name__.lower()
    text = str(exc).lower()
    status_code = getattr(getattr(exc, "response", None), "status_code", None)
    if status_code is not None:
        if int(status_code) in {418, 429}:
            return "rate_limited"
        if 500 <= int(status_code) < 600:
            return "endpoint_server_error"
        return "endpoint_http_error"
    if "getaddrinfo" in text or "name or service not known" in text or "temporary failure in name resolution" in text:
        return "dns_resolution"
    if "timeout" in name or "timeout" in text:
        return "timeout"
    if "connect" in name or "network" in text:
        return "network"
    if "empty" in text or "no rows" in text:
        return "empty_data"
    return "unknown"


def cache_path_for_ohlcv(
    symbol: str,
    interval: str,
    limit: int,
    cache_dir: str | Path = "data/historical",
    start_time: datetime | str | int | None = None,
    end_time: datetime | str | int | None = None,
) -> Path:
    ticker = binance_symbol(symbol)
    start = to_millis(start_time)
    end = to_millis(end_time)
    range_part = "latest" if start is None and end is None else f"{start or 'none'}_{end or 'none'}"
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{ticker}_{interval}_{limit}_{range_part}")
    return Path(cache_dir) / f"{safe}.csv"


def load_ohlcv_cache(path: str | Path, limit: int | None = None) -> pd.DataFrame:
    cache_path = Path(path)
    if not cache_path.exists():
        raise FileNotFoundError(str(cache_path))
    df = normalize_ohlcv(pd.read_csv(cache_path))
    if limit is not None and limit > 0:
        df = df.tail(limit).reset_index(drop=True)
    if df.empty:
        raise HistoricalDataError("cache_empty", f"OHLCV cache is empty: {cache_path}")
    return df


def save_ohlcv_cache(df: pd.DataFrame, path: str | Path) -> Path:
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = normalize_ohlcv(df)
    normalized.to_csv(cache_path, index=False)
    return cache_path


async def fetch_binance_klines(
    symbol: str,
    interval: str,
    start_time: datetime | str | int | None = None,
    end_time: datetime | str | int | None = None,
    limit: int = 1000,
    retries: int = 3,
    backoff_seconds: float = 0.5,
) -> pd.DataFrame:
    """Fetch Binance klines with pagination beyond the 1000-candle API limit.

    The primary Spot REST host can return HTTP 451 from some cloud regions. When
    that happens, retry the same public market-data request against Binance's
    market-data-only host before failing the whole fetch.
    """
    import httpx

    ticker = binance_symbol(symbol)
    start_ms = to_millis(start_time)
    end_ms = to_millis(end_time)
    attempts = max(1, retries + 1)
    last_error: Exception | None = None
    use_backward_latest_pagination = limit > 1000 and start_ms is None

    for attempt in range(attempts):
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                for base_url in BINANCE_KLINE_BASES:
                    remaining = limit
                    all_rows: list[list[Any]] = []
                    next_start = start_ms
                    next_end = end_ms
                    try:
                        while remaining > 0:
                            page_limit = min(1000, remaining)
                            params: dict[str, Any] = {"symbol": ticker, "interval": interval, "limit": page_limit}
                            if use_backward_latest_pagination:
                                if next_end is not None:
                                    params["endTime"] = next_end
                            else:
                                if next_start is not None:
                                    params["startTime"] = next_start
                                if end_ms is not None:
                                    params["endTime"] = end_ms
                            response = await client.get(f"{base_url}/klines", params=params)
                            response.raise_for_status()
                            rows = response.json()
                            if not rows:
                                break
                            all_rows.extend(rows)
                            remaining -= len(rows)
                            if use_backward_latest_pagination:
                                earliest_open = int(rows[0][0])
                                next_end = earliest_open - 1
                                if next_end <= 0 or len(rows) < page_limit:
                                    break
                            else:
                                last_open = int(rows[-1][0])
                                next_start = last_open + 1
                                if len(rows) < page_limit or (end_ms is not None and last_open >= end_ms):
                                    break
                        df = normalize_klines(all_rows)
                        df = df.tail(limit).reset_index(drop=True) if use_backward_latest_pagination else df.head(limit).reset_index(drop=True)
                        if df.empty:
                            raise HistoricalDataError("empty_data", f"Binance returned no OHLCV rows for {ticker} {interval}")
                        return df
                    except Exception as exc:  # noqa: BLE001 - try alternate public endpoint before retrying.
                        last_error = exc
                        continue
        except Exception as exc:  # noqa: BLE001 - API boundary needs typed retry diagnostics.
            last_error = exc
        if attempt < attempts - 1:
            await asyncio.sleep(backoff_seconds * (2 ** attempt))

    category = classify_historical_data_error(last_error or RuntimeError("unknown historical data error"))
    raise HistoricalDataError(
        category,
        f"Failed to fetch Binance klines for {ticker} {interval} after {attempts} attempt(s) "
        f"across {len(BINANCE_KLINE_BASES)} endpoint(s): {last_error}",
        original=last_error,
    )
