import unittest

import pandas as pd

from research.market_context_engine import build_market_context
from research.signal_review_agent import SignalReviewRequest, review_shadow_signal


def candles_from_prices(prices, volume=1000.0):
    idx = pd.date_range("2026-01-01", periods=len(prices), freq="1h", tz="UTC")
    return pd.DataFrame({
        "timestamp": idx,
        "open": prices,
        "high": [price * 1.01 for price in prices],
        "low": [price * 0.99 for price in prices],
        "close": prices,
        "volume": [volume] * len(prices),
    })


class MarketContextEngineTests(unittest.TestCase):
    def test_long_in_bullish_context_can_approve(self):
        prices = [100 + i * 0.2 for i in range(100)]
        context = build_market_context(
            candles=candles_from_prices(prices),
            symbol="ETH",
            timeframe="1h",
            side="LONG",
            entry_price=prices[-1],
            stop_loss=prices[-1] * 0.98,
            take_profit=prices[-1] * 1.04,
        )

        self.assertEqual(context.context_status, "APPROVE")
        self.assertEqual(context.risk_flags, [])
        self.assertFalse(context.can_modify_trade_levels)

    def test_long_against_bearish_trend_is_caution(self):
        prices = [120 - i * 0.2 for i in range(100)]
        context = build_market_context(
            candles=candles_from_prices(prices),
            symbol="ADA",
            timeframe="1h",
            side="LONG",
            entry_price=prices[-1],
        )

        self.assertIn("long_against_local_trend", context.risk_flags)
        self.assertEqual(context.context_status, "CAUTION")

    def test_stacked_market_risks_can_block(self):
        prices = [120 - i * 0.2 for i in range(100)]
        benchmark_prices = [50000 - i * 100 for i in range(100)]
        context = build_market_context(
            candles=candles_from_prices(prices),
            benchmark_candles=candles_from_prices(benchmark_prices),
            symbol="ADA",
            timeframe="1h",
            side="LONG",
            entry_price=prices[-1],
        )

        self.assertIn(context.context_status, {"CAUTION", "BLOCK"})
        self.assertIn("long_against_local_trend", context.risk_flags)

    def test_agent_uses_market_context_without_modifying_trade_levels(self):
        response = review_shadow_signal(SignalReviewRequest(
            symbol="SOL",
            timeframe="1h",
            side="LONG",
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=110.0,
            confidence=60.0,
            market_context={
                "context_status": "BLOCK",
                "risk_flags": ["long_against_local_trend", "benchmark_bearish_for_long"],
                "context_summary": "market context test",
            },
        ))

        self.assertEqual(response.review_status, "BLOCK")
        self.assertFalse(response.can_modify_trade_levels)
        self.assertIn("long_against_local_trend", response.risk_flags)


if __name__ == "__main__":
    unittest.main()
