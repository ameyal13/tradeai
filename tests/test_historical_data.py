import unittest

import pandas as pd

from tools.historical_data import normalize_klines, normalize_ohlcv


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


if __name__ == "__main__":
    unittest.main()
