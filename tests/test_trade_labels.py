import unittest

import pandas as pd

from tools.trade_labels import label_trade_at_index


def df(rows):
    return pd.DataFrame(rows)


class TradeLabelTests(unittest.TestCase):
    def test_buy_tp_before_sl(self):
        label = label_trade_at_index(df([
            {"timestamp": "2026-01-01T00:00:00Z", "open": 100, "high": 100, "low": 100, "close": 100, "volume": 1},
            {"timestamp": "2026-01-01T01:00:00Z", "open": 100, "high": 106, "low": 99, "close": 105, "volume": 1},
        ]), 0, "BUY", 1, stop_loss_pct=0.03, take_profit_pct=0.05, commission_pct=0, slippage_pct=0, spread_pct=0)
        self.assertEqual(label["outcome"], "WIN")
        self.assertEqual(label["tp_before_sl"], 1)

    def test_buy_sl_before_tp(self):
        label = label_trade_at_index(df([
            {"timestamp": "2026-01-01T00:00:00Z", "open": 100, "high": 100, "low": 100, "close": 100, "volume": 1},
            {"timestamp": "2026-01-01T01:00:00Z", "open": 100, "high": 101, "low": 96, "close": 97, "volume": 1},
        ]), 0, "BUY", 1, commission_pct=0, slippage_pct=0, spread_pct=0)
        self.assertEqual(label["outcome"], "LOSS")
        self.assertEqual(label["tp_before_sl"], 0)

    def test_sell_tp_before_sl(self):
        label = label_trade_at_index(df([
            {"timestamp": "2026-01-01T00:00:00Z", "open": 100, "high": 100, "low": 100, "close": 100, "volume": 1},
            {"timestamp": "2026-01-01T01:00:00Z", "open": 100, "high": 101, "low": 94, "close": 95, "volume": 1},
        ]), 0, "SELL", 1, stop_loss_pct=0.03, take_profit_pct=0.05, commission_pct=0, slippage_pct=0, spread_pct=0)
        self.assertEqual(label["outcome"], "WIN")
        self.assertEqual(label["tp_before_sl"], 1)

    def test_sell_sl_before_tp(self):
        label = label_trade_at_index(df([
            {"timestamp": "2026-01-01T00:00:00Z", "open": 100, "high": 100, "low": 100, "close": 100, "volume": 1},
            {"timestamp": "2026-01-01T01:00:00Z", "open": 100, "high": 104, "low": 99, "close": 103, "volume": 1},
        ]), 0, "SELL", 1, commission_pct=0, slippage_pct=0, spread_pct=0)
        self.assertEqual(label["outcome"], "LOSS")

    def test_same_candle_both_touched_is_conservative_loss(self):
        label = label_trade_at_index(df([
            {"timestamp": "2026-01-01T00:00:00Z", "open": 100, "high": 100, "low": 100, "close": 100, "volume": 1},
            {"timestamp": "2026-01-01T01:00:00Z", "open": 100, "high": 106, "low": 96, "close": 101, "volume": 1},
        ]), 0, "BUY", 1, stop_loss_pct=0.03, take_profit_pct=0.05, commission_pct=0, slippage_pct=0, spread_pct=0)
        self.assertEqual(label["outcome"], "LOSS")
        self.assertIn("ambiguous_intrabar_conservative_loss", label["notes"])

    def test_net_return_after_costs(self):
        positive = label_trade_at_index(df([
            {"timestamp": "2026-01-01T00:00:00Z", "open": 100, "high": 100, "low": 100, "close": 100, "volume": 1},
            {"timestamp": "2026-01-01T01:00:00Z", "open": 100, "high": 101, "low": 99, "close": 100.2, "volume": 1},
        ]), 0, "BUY", 1, stop_loss_pct=0.5, take_profit_pct=0.5, commission_pct=0, slippage_pct=0, spread_pct=0)
        negative = label_trade_at_index(df([
            {"timestamp": "2026-01-01T00:00:00Z", "open": 100, "high": 100, "low": 100, "close": 100, "volume": 1},
            {"timestamp": "2026-01-01T01:00:00Z", "open": 100, "high": 101, "low": 99, "close": 100.01, "volume": 1},
        ]), 0, "BUY", 1, stop_loss_pct=0.5, take_profit_pct=0.5, commission_pct=0.001, slippage_pct=0.0005, spread_pct=0.0003)
        self.assertEqual(positive["net_return_positive"], 1)
        self.assertEqual(negative["net_return_positive"], 0)

    def test_insufficient_data(self):
        label = label_trade_at_index(df([
            {"timestamp": "2026-01-01T00:00:00Z", "open": 100, "high": 100, "low": 100, "close": 100, "volume": 1},
        ]), 0, "BUY", 1)
        self.assertEqual(label["outcome"], "INVALID_DATA")


if __name__ == "__main__":
    unittest.main()
