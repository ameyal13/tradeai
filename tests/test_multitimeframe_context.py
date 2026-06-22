import unittest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import numpy as np
import pandas as pd

from research.multitimeframe_context import (
    compute_4h_context,
    is_trend_aligned,
    normalize_signal_side,
    normalize_symbol,
    trend_from_ema20_ema50,
)


def candles(values):
    idx = pd.date_range("2026-01-01", periods=len(values), freq="4h", tz="UTC")
    close = np.array(values, dtype=float)
    return pd.DataFrame({
        "timestamp": idx,
        "open": close,
        "high": close + 1,
        "low": close - 1,
        "close": close,
        "volume": 1000,
    })


class MultiTimeframeContextTests(unittest.IsolatedAsyncioTestCase):
    def test_symbol_and_side_normalization(self):
        self.assertEqual(normalize_symbol("ADA/USDT"), "ADA")
        self.assertEqual(normalize_symbol("ethusdt"), "ETH")
        self.assertEqual(normalize_signal_side("BUY"), "LONG")
        self.assertEqual(normalize_signal_side("SELL"), "SHORT")

    def test_trend_from_ema20_ema50(self):
        self.assertEqual(trend_from_ema20_ema50(candles(np.linspace(100, 150, 60))), "bullish")
        self.assertEqual(trend_from_ema20_ema50(candles(np.linspace(150, 100, 60))), "bearish")
        self.assertEqual(trend_from_ema20_ema50(candles(np.full(60, 100))), "neutral")

    def test_alignment(self):
        self.assertTrue(is_trend_aligned("bullish", "LONG"))
        self.assertTrue(is_trend_aligned("bearish", "SHORT"))
        self.assertFalse(is_trend_aligned("neutral", "LONG"))
        self.assertFalse(is_trend_aligned("bullish", "SHORT"))

    async def test_compute_4h_context_fetches_asset_and_btc_with_expected_window(self):
        signal_time = datetime(2026, 6, 20, 12, tzinfo=timezone.utc)
        asset = candles(np.linspace(100, 150, 60))
        btc = candles(np.linspace(150, 100, 60))

        async def fake_fetch(symbol, interval, start_time=None, end_time=None, limit=1000, **kwargs):
            self.assertEqual(interval, "4h")
            self.assertEqual(limit, 50)
            self.assertEqual(end_time, signal_time)
            self.assertEqual(int((end_time - start_time).total_seconds() / 3600), 200)
            return asset if symbol == "ADA" else btc

        with patch("research.multitimeframe_context.fetch_binance_klines", new=AsyncMock(side_effect=fake_fetch)) as fetch:
            context = await compute_4h_context("ADA/USDT", signal_time, "LONG")

        self.assertEqual(fetch.await_count, 2)
        self.assertEqual(context["asset_4h_trend"], "bullish")
        self.assertTrue(context["asset_trend_aligned"])
        self.assertEqual(context["btc_4h_trend"], "bearish")
        self.assertFalse(context["btc_trend_aligned"])
        self.assertFalse(context["full_alignment"])


if __name__ == "__main__":
    unittest.main()
