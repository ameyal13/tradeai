import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from tools.ml_engine import walk_forward_accuracy, xgboost_signal
from tools.strategy_signals import add_features, generate_strategy_signal_from_df


def sample_candles(rows=300):
    idx = pd.date_range("2026-01-01", periods=rows, freq="h", tz="UTC")
    base = np.linspace(100, 125, rows) + np.sin(np.arange(rows) / 4) * 3
    return pd.DataFrame({
        "timestamp": idx,
        "open": base,
        "high": base + 2,
        "low": base - 2,
        "close": base + np.sin(np.arange(rows) / 5),
        "volume": np.linspace(1000, 2500, rows),
    })


class MLEngineTests(unittest.TestCase):
    def test_xgboost_signal_requires_min_rows(self):
        """Con < 200 filas debe retornar model_available=False y signal=HOLD"""
        features = add_features(sample_candles(80))
        result = xgboost_signal(features, min_train_rows=200)

        self.assertFalse(result["model_available"])
        self.assertEqual(result["signal"], "HOLD")

    def test_xgboost_signal_with_sufficient_data(self):
        """Con >= 200 filas debe entrenar y retornar una predicción válida"""
        features = add_features(sample_candles(300))
        result = xgboost_signal(features, min_train_rows=200)

        self.assertTrue(result["model_available"])
        self.assertIn(result["signal"], {"BUY", "SELL", "HOLD"})
        self.assertIsNotNone(result["probability_up"])
        self.assertGreaterEqual(result["confidence"], 0)
        self.assertLessEqual(result["confidence"], 100)

    def test_xgboost_signal_no_lookahead(self):
        """La predicción debe usar iloc[-2], nunca la última fila abierta"""
        df = sample_candles(300)
        changed_last = df.copy()
        changed_last.loc[changed_last.index[-1], ["open", "high", "low", "close", "volume"]] = [1, 2, 0.5, 1, 999999]

        result_a = xgboost_signal(add_features(df), min_train_rows=200)
        result_b = xgboost_signal(add_features(changed_last), min_train_rows=200)

        self.assertEqual(result_a["probability_up"], result_b["probability_up"])
        self.assertEqual(result_a["signal"], result_b["signal"])

    def test_walk_forward_accuracy_returns_float(self):
        """walk_forward_accuracy debe retornar float entre 0 y 1"""
        features = add_features(sample_candles(300))
        accuracy = walk_forward_accuracy(features, n_splits=3)

        self.assertIsInstance(accuracy, float)
        self.assertGreaterEqual(accuracy, 0)
        self.assertLessEqual(accuracy, 1)

    def test_xgboost_mode_in_generate_strategy_signal(self):
        """generate_strategy_signal_from_df debe aceptar strategy_mode='xgboost'"""
        with patch("tools.strategy_signals.get_fear_greed_index", return_value={"value": 50, "classification": "Neutral", "timestamp": None}):
            signal = generate_strategy_signal_from_df(sample_candles(300), strategy_mode="xgboost").to_dict()

        self.assertEqual(signal["strategy_mode"], "xgboost")
        self.assertEqual(signal["model_provider"], "local_xgboost")
        self.assertEqual(signal["model_name"], "xgboost_classifier_v1")
        self.assertIn(signal["signal"], {"BUY", "SELL", "HOLD"})
        self.assertEqual(signal["input_features"]["sentiment_features"]["classification"], "Neutral")

    def test_fallback_when_xgboost_not_available(self):
        """Si xgboost no está instalado, debe hacer fallback a model_based sin crash"""
        with patch("tools.strategy_signals.get_fear_greed_index", return_value={"value": 50, "classification": "Neutral", "timestamp": None}):
            with patch("tools.strategy_signals.xgboost_signal", side_effect=ImportError("xgboost missing")):
                signal = generate_strategy_signal_from_df(sample_candles(300), strategy_mode="xgboost").to_dict()

        self.assertEqual(signal["strategy_mode"], "model_based")
        self.assertEqual(signal["model_provider"], "local_numpy")


if __name__ == "__main__":
    unittest.main()
