import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from tools.ml_engine import build_trade_outcome_labels, walk_forward_accuracy, xgboost_signal
from tools.strategy_signals import add_features, generate_strategy_signal_from_df, xgboost_signal_from_df


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

    def test_build_trade_outcome_labels_returns_binary_valid_labels(self):
        labels = build_trade_outcome_labels(
            sample_candles(260),
            horizon_candles=4,
            stop_loss_pct=0.01,
            take_profit_pct=0.01,
            commission_pct=0,
            slippage_pct=0,
        )
        valid = labels.dropna()

        self.assertGreater(len(valid), 0)
        self.assertEqual(set(labels.columns), {"buy_win", "sell_win"})
        self.assertTrue(set(valid["buy_win"].unique()).issubset({0.0, 1.0}))
        self.assertTrue(set(valid["sell_win"].unique()).issubset({0.0, 1.0}))

    def test_buy_loss_does_not_automatically_mark_sell_win(self):
        idx = pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC")
        df = pd.DataFrame({
            "timestamp": idx,
            "open": [100, 100, 100],
            "high": [100, 101, 100],
            "low": [100, 98, 100],
            "close": [100, 99, 100],
            "volume": [1000, 1000, 1000],
        })
        labels = build_trade_outcome_labels(
            df,
            horizon_candles=1,
            stop_loss_pct=0.01,
            take_profit_pct=0.05,
            commission_pct=0,
            slippage_pct=0,
            spread_pct=0,
        )

        self.assertEqual(labels.iloc[0]["buy_win"], 0)
        self.assertEqual(labels.iloc[0]["sell_win"], 0)

    def test_xgboost_signal_with_trade_labels_trains(self):
        features = add_features(sample_candles(300))
        result = xgboost_signal(
            features,
            min_train_rows=30,
            strategy_params={
                "use_trade_labels": True,
                "horizon_candles": 4,
                "stop_loss_pct": 0.01,
                "take_profit_pct": 0.01,
                "commission_pct": 0,
                "slippage_pct": 0,
            },
        )

        self.assertTrue(result["model_available"])
        self.assertEqual(result["label_type"], "trade_outcome_directional")
        self.assertIn("probability_buy_win", result)
        self.assertIn("probability_sell_win", result)
        self.assertIn(result["signal"], {"BUY", "SELL", "HOLD"})

    def test_xgboost_signal_reports_price_return_label_type(self):
        features = add_features(sample_candles(300))
        result = xgboost_signal(features, min_train_rows=30, strategy_params={"use_trade_labels": False})

        self.assertEqual(result["label_type"], "price_return")

    def test_xgboost_input_features_include_label_params(self):
        result = {
            "model_available": True,
            "probability_up": 0.6,
            "probability_buy_win": 0.6,
            "probability_sell_win": 0.3,
            "signal": "BUY",
            "confidence": 60,
            "validation_accuracy": 0.5,
            "walk_forward_accuracy": None,
            "train_rows": 200,
            "reason": "mock",
            "sentiment_features": {},
            "label_type": "trade_outcome_directional",
            "label_stop_loss_pct": 0.03,
            "label_take_profit_pct": 0.045,
            "label_horizon_candles": 4,
            "label_costs": {"commission_pct": 0.001, "slippage_pct": 0.0005, "spread_pct": 0.0003},
            "label_level_note": "fixed_pct_trade_labels_not_atr",
        }
        with patch("tools.strategy_signals.xgboost_signal", return_value=result):
            signal = xgboost_signal_from_df(sample_candles(300), use_sentiment=False).to_dict()

        self.assertEqual(signal["input_features"]["label_type"], "trade_outcome_directional")
        self.assertEqual(signal["input_features"]["label_horizon_candles"], 4)
        self.assertEqual(signal["input_features"]["label_costs"]["spread_pct"], 0.0003)

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

    def test_xgboost_signal_from_df_without_sentiment_does_not_fetch_fear_greed(self):
        result = {
            "model_available": True,
            "probability_up": 0.5,
            "signal": "HOLD",
            "confidence": 10,
            "validation_accuracy": 0.5,
            "walk_forward_accuracy": 0.5,
            "train_rows": 200,
            "reason": "mock",
            "sentiment_features": {},
        }
        with patch("tools.strategy_signals.get_fear_greed_index") as fear_greed:
            with patch("tools.strategy_signals.xgboost_signal", return_value=result) as model:
                signal = xgboost_signal_from_df(sample_candles(300), use_sentiment=False).to_dict()

        fear_greed.assert_not_called()
        self.assertIsNone(model.call_args.kwargs["sentiment_features"])
        self.assertEqual(signal["strategy_mode"], "xgboost")

    def test_xgboost_signal_from_df_with_sentiment_fetches_fear_greed(self):
        fear_greed_payload = {"value": 10, "classification": "Extreme Fear", "timestamp": "1"}
        result = {
            "model_available": True,
            "probability_up": 0.6,
            "signal": "BUY",
            "confidence": 20,
            "validation_accuracy": 0.5,
            "walk_forward_accuracy": 0.5,
            "train_rows": 200,
            "reason": "mock",
            "sentiment_features": fear_greed_payload,
        }
        with patch("tools.strategy_signals.get_fear_greed_index", return_value=fear_greed_payload) as fear_greed:
            with patch("tools.strategy_signals.xgboost_signal", return_value=result) as model:
                signal = xgboost_signal_from_df(sample_candles(300), use_sentiment=True).to_dict()

        fear_greed.assert_called_once()
        self.assertEqual(model.call_args.kwargs["sentiment_features"], fear_greed_payload)
        self.assertIn("Extreme Fear", signal["reasoning"])


if __name__ == "__main__":
    unittest.main()
