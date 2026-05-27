import unittest
from pathlib import Path
import tempfile

import pandas as pd

from tools.historical_data import (
    cache_path_for_ohlcv,
    classify_historical_data_error,
    load_ohlcv_cache,
    normalize_klines,
    normalize_ohlcv,
    save_ohlcv_cache,
)


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


if __name__ == "__main__":
    unittest.main()
