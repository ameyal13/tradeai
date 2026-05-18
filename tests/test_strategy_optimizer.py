import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from tools.strategy_optimizer import run_walk_forward_optimizer


def candles(rows=160):
    idx = pd.date_range("2026-01-01", periods=rows, freq="h", tz="UTC")
    base = np.linspace(100, 120, rows) + np.sin(np.arange(rows) / 5)
    return pd.DataFrame({
        "timestamp": idx,
        "open": base,
        "high": base + 2,
        "low": base - 2,
        "close": base + 0.2,
        "volume": 1000,
    })


class StrategyOptimizerTests(unittest.TestCase):
    def test_optimizer_uses_distinct_train_and_validation_windows(self):
        result = run_walk_forward_optimizer(candles(), "BTC", "1h", train_size=80, validation_size=40, min_history=20)
        self.assertEqual(result["windows"]["train"]["end_index"], 79)
        self.assertEqual(result["windows"]["validation"]["start_index"], 80)

    def test_candidate_not_winner_if_only_wins_train(self):
        calls = []

        def fake_replay(df, **kwargs):
            calls.append(kwargs.get("strategy_params"))
            is_validation = len(calls) > 2
            candidate = kwargs.get("strategy_params", {}).get("rsi_buy_threshold") == 30
            avg = -1 if is_validation and candidate else 5 if candidate else 0
            return {"metrics": [{"evaluated_predictions": 5, "profit_factor": max(avg, 0), "average_return_pct": avg, "max_drawdown": 0, "sharpe": 0, "win_rate": 50}]}

        with patch("tools.strategy_optimizer.run_historical_replay", side_effect=fake_replay):
            result = run_walk_forward_optimizer(
                candles(), "BTC", "1h", train_size=80, validation_size=40,
                parameter_grid={"rsi_buy_threshold": [30, 35]},
            )
        self.assertFalse(result["candidate_beats_baseline"])

    def test_candidate_wins_if_validation_beats_baseline(self):
        def fake_replay(df, **kwargs):
            candidate = kwargs.get("strategy_params", {}).get("rsi_buy_threshold") == 30
            avg = 3 if candidate else 0
            return {"metrics": [{"evaluated_predictions": 5, "profit_factor": avg + 1, "average_return_pct": avg, "max_drawdown": 0, "sharpe": 0, "win_rate": 60}]}

        with patch("tools.strategy_optimizer.run_historical_replay", side_effect=fake_replay):
            result = run_walk_forward_optimizer(
                candles(), "BTC", "1h", train_size=80, validation_size=40,
                parameter_grid={"rsi_buy_threshold": [30, 35]},
            )
        self.assertTrue(result["candidate_beats_baseline"])

    def test_insufficient_data(self):
        result = run_walk_forward_optimizer(candles(20), "BTC", "1h", train_size=80, validation_size=40)
        self.assertIn("insufficient_data", result["warnings"])

    def test_candidate_params_registered(self):
        result = run_walk_forward_optimizer(candles(), "BTC", "1h", train_size=80, validation_size=40, min_history=20)
        self.assertIsNotNone(result["candidate_params"])
        self.assertIn("rsi_buy_threshold", result["candidate_params"])


if __name__ == "__main__":
    unittest.main()
