import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

import pandas as pd

from tools.historical_data import (
    cache_path_for_ohlcv,
    classify_historical_data_error,
    fetch_binance_klines,
    load_ohlcv_cache,
    normalize_klines,
    normalize_ohlcv,
    save_ohlcv_cache,
)


def raw_rows(start: int, count: int) -> list[list[str]]:
    return [
        [start + idx * 60_000, "1", "2", "0.5", "1.5", "10"]
        for idx in range(count)
    ]


class FakeResponse:
    def __init__(self, rows, error=None):
        self._rows = rows
        self._error = error

    def raise_for_status(self):
        if self._error:
            raise self._error
        return None

    def json(self):
        return self._rows


class FakeAsyncClient:
    pages: list[list[list[str]]] = []
    requests: list[dict] = []
    urls: list[str] = []

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def get(self, url, params):
        self.__class__.urls.append(url)
        self.__class__.requests.append(dict(params))
        return FakeResponse(self.__class__.pages.pop(0))


class FallbackAsyncClient(FakeAsyncClient):
    primary_error: Exception = RuntimeError("HTTP 451")

    async def get(self, url, params):
        self.__class__.urls.append(url)
        self.__class__.requests.append(dict(params))
        if "api.binance.com" in url:
            return FakeResponse([], error=self.__class__.primary_error)
        return FakeResponse(self.__class__.pages.pop(0))


class HistoricalDataTests(unittest.TestCase):
    def test_normalize_klines_orders_and_dedupes(self):
        raw = [
            [2000, "2", "3", "1", "2.5", "20"],
            [1000, "1", "2", "0.5", "1.5", "10"],
            [1000, "1", "2", "0.5", "1.5", "10"],
        ]
        df = normalize_klines(raw)

        self.assertEqual(len(df), 2)
        self.assertTrue(df["timestamp"].is_monotonic_increasing)
        self.assertEqual(list(df.columns), ["timestamp", "open", "high", "low", "close", "volume"])

    def test_normalize_ohlcv_requires_columns(self):
        df = normalize_ohlcv(pd.DataFrame([{
            "timestamp": "2026-01-01T00:00:00Z",
            "open": 1,
            "high": 2,
            "low": 0.5,
            "close": 1.5,
            "volume": 10,
        }]))

        self.assertEqual(float(df.iloc[0]["close"]), 1.5)

    def test_cache_roundtrip_normalizes_ohlcv(self):
        df = pd.DataFrame([{
            "timestamp": "2026-01-01T00:00:00Z",
            "open": 1,
            "high": 2,
            "low": 0.5,
            "close": 1.5,
            "volume": 10,
        }])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "BTCUSDT_1h_1_latest.csv"
            save_ohlcv_cache(df, path)
            loaded = load_ohlcv_cache(path)

        self.assertEqual(len(loaded), 1)
        self.assertEqual(list(loaded.columns), ["timestamp", "open", "high", "low", "close", "volume"])

    def test_cache_path_includes_symbol_interval_and_limit(self):
        path = cache_path_for_ohlcv("BTC", "1h", 500, cache_dir="cache")

        self.assertIn("BTCUSDT_1h_500", str(path))

    def test_error_classifier_detects_dns_resolution(self):
        exc = RuntimeError("[Errno 11001] getaddrinfo failed")

        self.assertEqual(classify_historical_data_error(exc), "dns_resolution")

    def test_fetch_binance_latest_over_1000_paginates_backward(self):
        latest = raw_rows(500 * 60_000, 1000)
        older = raw_rows(0, 500)
        FakeAsyncClient.pages = [latest, older]
        FakeAsyncClient.requests = []
        FakeAsyncClient.urls = []

        async def run():
            with patch("httpx.AsyncClient", FakeAsyncClient):
                return await fetch_binance_klines("SOL", "1h", limit=1500, retries=0)

        df = __import__("asyncio").run(run())

        self.assertEqual(len(df), 1500)
        self.assertTrue(df["timestamp"].is_monotonic_increasing)
        self.assertEqual(FakeAsyncClient.requests[0]["limit"], 1000)
        self.assertNotIn("endTime", FakeAsyncClient.requests[0])
        self.assertEqual(FakeAsyncClient.requests[1]["limit"], 500)
        self.assertEqual(FakeAsyncClient.requests[1]["endTime"], latest[0][0] - 1)

    def test_fetch_binance_falls_back_to_market_data_host(self):
        FallbackAsyncClient.pages = [raw_rows(0, 3)]
        FallbackAsyncClient.requests = []
        FallbackAsyncClient.urls = []

        async def run():
            with patch("httpx.AsyncClient", FallbackAsyncClient):
                return await fetch_binance_klines("ADA", "1h", limit=3, retries=0)

        df = __import__("asyncio").run(run())

        self.assertEqual(len(df), 3)
        self.assertIn("api.binance.com", FallbackAsyncClient.urls[0])
        self.assertIn("data-api.binance.vision", FallbackAsyncClient.urls[1])

    def test_fetch_binance_dedupes_orders_and_does_not_exceed_limit(self):
        latest = raw_rows(500 * 60_000, 1000)
        older = raw_rows(0, 501)
        older[-1] = latest[0]
        FakeAsyncClient.pages = [latest, older]
        FakeAsyncClient.requests = []
        FakeAsyncClient.urls = []

        async def run():
            with patch("httpx.AsyncClient", FakeAsyncClient):
                return await fetch_binance_klines("SOL", "1h", limit=1500, retries=0)

        df = __import__("asyncio").run(run())

        self.assertEqual(len(df), 1500)
        self.assertTrue(df["timestamp"].is_monotonic_increasing)
        self.assertEqual(df["timestamp"].nunique(), 1500)

    def test_fetch_binance_limit_under_1000_keeps_single_latest_request(self):
        FakeAsyncClient.pages = [raw_rows(0, 500)]
        FakeAsyncClient.requests = []
        FakeAsyncClient.urls = []

        async def run():
            with patch("httpx.AsyncClient", FakeAsyncClient):
                return await fetch_binance_klines("SOL", "1h", limit=500, retries=0)

        df = __import__("asyncio").run(run())

        self.assertEqual(len(df), 500)
        self.assertEqual(len(FakeAsyncClient.requests), 1)
        self.assertEqual(FakeAsyncClient.requests[0]["limit"], 500)
        self.assertNotIn("startTime", FakeAsyncClient.requests[0])
        self.assertNotIn("endTime", FakeAsyncClient.requests[0])


if __name__ == "__main__":
    unittest.main()
