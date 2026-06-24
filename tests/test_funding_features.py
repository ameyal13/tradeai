import unittest

import numpy as np
import pandas as pd

from tools.funding_features import align_funding_to_ohlcv, compute_funding_features, normalize_futures_symbol


def candles(rows=8):
    idx = pd.date_range("2026-01-01", periods=rows, freq="h", tz="UTC")
    base = [100, 101, 102, 101, 100, 99, 100, 101]
    close = pd.Series(base[:rows], dtype=float)
    return pd.DataFrame({
        "timestamp": idx,
        "open": close,
        "high": close + 1,
        "low": close - 1,
        "close": close,
        "volume": 1000.0,
    })


class FundingFeatureTests(unittest.TestCase):
    def test_normalizes_futures_symbol(self):
        self.assertEqual(normalize_futures_symbol("ADA"), "ADAUSDT")
        self.assertEqual(normalize_futures_symbol("ETH/USDT"), "ETHUSDT")

    def test_align_funding_forward_fills_without_future_lookahead(self):
        ohlcv = candles(6)
        funding = pd.DataFrame({
            "timestamp": [
                pd.Timestamp("2026-01-01T00:00:00Z"),
                pd.Timestamp("2026-01-01T04:00:00Z"),
            ],
            "funding_rate": [0.0001, 0.0009],
        })

        aligned = align_funding_to_ohlcv(ohlcv, funding)

        self.assertEqual(aligned.iloc[0], 0.0001)
        self.assertEqual(aligned.iloc[3], 0.0001)
        self.assertEqual(aligned.iloc[4], 0.0009)

    def test_compute_funding_features_divergence_and_regime(self):
        ohlcv = candles(8)
        funding = pd.DataFrame({
            "timestamp": pd.date_range("2026-01-01", periods=2, freq="4h", tz="UTC"),
            "funding_rate": [0.0006, -0.0006],
        })
        oi = pd.DataFrame({
            "timestamp": ohlcv["timestamp"],
            "open_interest": [100, 99, 98, 97, 96, 95, 96, 97],
            "open_interest_value": np.arange(8),
        })

        features = compute_funding_features(ohlcv, funding, oi)

        self.assertIn("funding_rate_ma3", features.columns)
        self.assertEqual(features.loc[1, "oi_price_diverge"], 1.0)
        self.assertEqual(features.loc[3, "oi_price_diverge"], -1.0)
        self.assertEqual(features.loc[4, "oi_trend_regime"], -1.0)
        self.assertTrue(pd.isna(features.loc[0, "oi_change_1h"]))


if __name__ == "__main__":
    unittest.main()
