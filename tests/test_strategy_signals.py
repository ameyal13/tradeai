import unittest

import numpy as np
import pandas as pd

from tools.prediction_journal import metrics_by_strategy_mode, metrics_by_symbol_timeframe
from tools.strategy_signals import (
    deterministic_signal_from_df,
    generate_strategy_signal_from_df,
    model_based_signal_from_df,
)


def sample_candles(rows=80):
    idx = pd.date_range("2026-01-01", periods=rows, freq="h", tz="UTC")
    base = np.linspace(100, 120, rows) + np.sin(np.arange(rows) / 3) * 2
    return pd.DataFrame({
        "timestamp": idx,
        "open": base,
        "high": base + 2,
        "low": base - 2,
        "close": base + np.sin(np.arange(rows) / 5),
        "volume": np.linspace(1000, 2000, rows),
    })


class StrategySignalTests(unittest.TestCase):
    def test_deterministic_signal_has_required_journal_fields(self):
        signal = deterministic_signal_from_df(sample_candles()).to_dict()

        self.assertEqual(signal["strategy_mode"], "deterministic")
        self.assertIn(signal["signal"], {"BUY", "SELL", "HOLD"})
        self.assertIn("rsi", signal["input_features"])
        self.assertIn("risk_reward_ratio", signal)

    def test_model_based_interface_without_enough_rows_returns_hold(self):
        signal = model_based_signal_from_df(sample_candles(20), min_train_rows=40).to_dict()

        self.assertEqual(signal["strategy_mode"], "model_based")
        self.assertEqual(signal["signal"], "HOLD")
        self.assertFalse(signal["input_features"]["model_available"])
        self.assertEqual(signal["model_provider"], "local_numpy")

    def test_model_based_uses_temporal_split_when_data_available(self):
        signal = model_based_signal_from_df(sample_candles(120), min_train_rows=30).to_dict()

        self.assertEqual(signal["strategy_mode"], "model_based")
        self.assertIn("temporal_split", signal["input_features"])
        self.assertGreater(signal["input_features"]["temporal_split"]["train_rows"], 0)

    def test_hybrid_preserves_deterministic_direction(self):
        deterministic = generate_strategy_signal_from_df(sample_candles(), "deterministic").to_dict()
        hybrid = generate_strategy_signal_from_df(sample_candles(), "hybrid", provider="none").to_dict()

        self.assertEqual(hybrid["strategy_mode"], "hybrid")
        self.assertEqual(hybrid["signal"], deterministic["signal"])
        self.assertFalse(hybrid["input_features"]["llm_can_change_direction"])

    def test_strategy_mode_metrics(self):
        predictions = [
            {"id": "p1", "strategy_mode": "deterministic", "symbol": "BTC", "timeframe": "1h"},
            {"id": "p2", "strategy_mode": "hybrid", "symbol": "ETH", "timeframe": "4h"},
        ]
        outcomes = [
            {"prediction_id": "p1", "outcome": "WIN", "return_pct": 2},
            {"prediction_id": "p2", "outcome": "LOSS", "return_pct": -1},
        ]

        by_mode = metrics_by_strategy_mode(predictions, outcomes)
        by_symbol_timeframe = metrics_by_symbol_timeframe(predictions, outcomes)

        self.assertEqual(by_mode[0]["total_signals"], 1)
        self.assertIn("profit_factor", by_mode[0])
        self.assertEqual({row["symbol_timeframe"] for row in by_symbol_timeframe}, {"BTC:1h", "ETH:4h"})


if __name__ == "__main__":
    unittest.main()
