import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from tools.historical_replay import run_historical_replay
from tools.prediction_journal import PredictionStore


def candles(rows=90):
    idx = pd.date_range("2026-01-01", periods=rows, freq="h", tz="UTC")
    base = np.linspace(100, 130, rows) + np.sin(np.arange(rows) / 3)
    return pd.DataFrame({
        "timestamp": idx,
        "open": base,
        "high": base + 2,
        "low": base - 2,
        "close": base + 0.5,
        "volume": np.linspace(1000, 2000, rows),
    })


class HistoricalReplayTests(unittest.TestCase):
    def test_no_future_data_and_entry_n_plus_1(self):
        data = candles()
        result = run_historical_replay(data, "BTC", "1h", "deterministic", min_history=30, max_predictions=1)
        pred = result["predictions"][0]
        expected_entry = float(data.iloc[30]["open"])

        self.assertEqual(pred["input_features"].get("prediction_row"), None)
        self.assertAlmostEqual(pred["entry_price"], expected_entry)
        self.assertTrue(result["assumptions"]["uses_history_through_index_n"])

    def test_generate_prediction_and_outcome(self):
        result = run_historical_replay(candles(), "BTC", "1h", "deterministic", min_history=30, max_predictions=3)
        self.assertGreater(len(result["predictions"]), 0)
        self.assertIn("metrics", result)

    def test_each_strategy_mode_runs(self):
        for mode in ["deterministic", "model_based", "hybrid"]:
            result = run_historical_replay(candles(110), "BTC", "1h", mode, min_history=35, max_predictions=2)
            self.assertGreater(len(result["predictions"]), 0)
            self.assertEqual(result["predictions"][0]["strategy_mode"], mode)

    def test_hold_handled_without_outcome(self):
        result = run_historical_replay(candles(35), "BTC", "1h", "model_based", min_history=20, max_predictions=1)
        if result["predictions"][0]["signal"] == "HOLD":
            self.assertEqual(result["outcomes"], [])

    def test_insufficient_data(self):
        result = run_historical_replay(candles(10), "BTC", "1h", "deterministic", min_history=30)
        self.assertEqual(result["assumptions"]["error"], "insufficient_data")

    def test_optional_store(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = PredictionStore(file_path=Path(tmp) / "journal.json")
            result = run_historical_replay(candles(), "BTC", "1h", "deterministic", min_history=30, max_predictions=2, store=store)
            self.assertEqual(len(store.list_predictions()), len(result["predictions"]))


if __name__ == "__main__":
    unittest.main()
