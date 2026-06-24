import unittest

import numpy as np
import pandas as pd

from tools.multitimeframe_features import compute_4h_features, compute_btc_context_features


def ohlcv(index, close):
    close = pd.Series(close, dtype=float)
    return pd.DataFrame({
        "timestamp": index,
        "open": close,
        "high": close + 1,
        "low": close - 1,
        "close": close,
        "volume": 1000.0,
    })


class MultiTimeframeFeatureTests(unittest.TestCase):
    def test_4h_alignment_uses_closed_candles_only(self):
        four_h_index = pd.date_range("2026-01-01", periods=70, freq="4h", tz="UTC")
        four_h = ohlcv(four_h_index, np.linspace(100, 170, 70))
        target_open = four_h_index[60]
        one_h = ohlcv(
            pd.DatetimeIndex([target_open + pd.Timedelta(hours=3), target_open + pd.Timedelta(hours=4)], tz="UTC"),
            [150, 151],
        )

        features = compute_4h_features(one_h, four_h)

        self.assertNotEqual(features.loc[0, "tf4h_ema20"], features.loc[1, "tf4h_ema20"])
        self.assertTrue(pd.notna(features.loc[1, "tf4h_trend"]))

    def test_btc_context_lags_returns(self):
        idx = pd.date_range("2026-01-01", periods=30, freq="h", tz="UTC")
        asset = ohlcv(idx, np.linspace(50, 60, 30))
        btc = ohlcv(idx, np.linspace(100, 130, 30))

        features = compute_btc_context_features(asset, btc)

        expected = btc.loc[1, "close"] / btc.loc[0, "close"] - 1
        self.assertAlmostEqual(features.loc[2, "btc_return_1h"], expected)
        self.assertIn(features.loc[29, "asset_btc_diverge"], {0.0, 1.0})


if __name__ == "__main__":
    unittest.main()
