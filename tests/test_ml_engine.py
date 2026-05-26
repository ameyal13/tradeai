import unittest
import time
from unittest.mock import patch

import numpy as np
import pandas as pd

from tools.ml_engine import (
    FEATURE_COLS,
    _prepared_trade_dataset,
    build_trade_outcome_labels,
    walk_forward_accuracy,
    xgboost_signal,
)
from tools.strategy_signals import add_features, generate_strategy_signal_from_df, xgboost_signal_from_df
from tools.trade_labels import label_trade_at_index


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

    def test_trade_outcome_labels_support_dynamic_atr_like_levels(self):
        idx = pd.date_range("2026-01-01", periods=8, freq="h", tz="UTC")
        df = pd.DataFrame({
            "timestamp": idx,
            "open": [100, 100, 100, 100, 100, 100, 100, 100],
            "high": [100, 101, 101, 101, 101, 101, 101, 101],
            "low": [100, 99, 99, 99, 99, 99, 99, 99],
            "close": [100, 100, 100, 100, 100, 100, 100, 100],
            "volume": [1000] * 8,
        })

        fixed = build_trade_outcome_labels(
            df,
            horizon_candles=4,
            stop_loss_pct=0.03,
            take_profit_pct=0.045,
            commission_pct=0,
            slippage_pct=0,
            spread_pct=0,
        )
        dynamic = build_trade_outcome_labels(
            df,
            horizon_candles=4,
            stop_loss_pct=0.03,
            take_profit_pct=0.045,
            commission_pct=0,
            slippage_pct=0,
            spread_pct=0,
            stop_loss_pcts=np.full(len(df), 0.005),
            take_profit_pcts=np.full(len(df), 0.008),
        )

        self.assertTrue(np.isnan(fixed.iloc[0]["buy_win"]))
        self.assertEqual(dynamic.iloc[0]["buy_win"], 0.0)
        self.assertEqual(dynamic.iloc[0]["sell_win"], 0.0)

    def test_fast_trade_labels_match_reference_for_buy_and_sell(self):
        df = sample_candles(40)
        labels = build_trade_outcome_labels(
            df,
            horizon_candles=4,
            stop_loss_pct=0.01,
            take_profit_pct=0.015,
            commission_pct=0.001,
            slippage_pct=0.0005,
            spread_pct=0.0003,
        )

        for index_n in [0, 5, 10, 20]:
            buy_reference = label_trade_at_index(
                df, index_n, "BUY", 4,
                stop_loss_pct=0.01,
                take_profit_pct=0.015,
                commission_pct=0.001,
                slippage_pct=0.0005,
                spread_pct=0.0003,
            )
            sell_reference = label_trade_at_index(
                df, index_n, "SELL", 4,
                stop_loss_pct=0.01,
                take_profit_pct=0.015,
                commission_pct=0.001,
                slippage_pct=0.0005,
                spread_pct=0.0003,
            )
            expected_buy = 1.0 if buy_reference["outcome"] == "WIN" else 0.0 if buy_reference["outcome"] == "LOSS" else np.nan
            expected_sell = 1.0 if sell_reference["outcome"] == "WIN" else 0.0 if sell_reference["outcome"] == "LOSS" else np.nan

            if np.isnan(expected_buy):
                self.assertTrue(np.isnan(labels.iloc[index_n]["buy_win"]))
            else:
                self.assertEqual(labels.iloc[index_n]["buy_win"], expected_buy)
            if np.isnan(expected_sell):
                self.assertTrue(np.isnan(labels.iloc[index_n]["sell_win"]))
            else:
                self.assertEqual(labels.iloc[index_n]["sell_win"], expected_sell)

    def test_ambiguous_same_candle_tp_sl_is_loss(self):
        idx = pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC")
        df = pd.DataFrame({
            "timestamp": idx,
            "open": [100, 100, 100],
            "high": [100, 106, 100],
            "low": [100, 94, 100],
            "close": [100, 100, 100],
            "volume": [1000, 1000, 1000],
        })
        labels = build_trade_outcome_labels(
            df,
            horizon_candles=1,
            stop_loss_pct=0.05,
            take_profit_pct=0.05,
            commission_pct=0,
            slippage_pct=0,
            spread_pct=0,
        )

        self.assertEqual(labels.iloc[0]["buy_win"], 0.0)
        self.assertEqual(labels.iloc[0]["sell_win"], 0.0)

    def test_expired_trade_label_stays_nan(self):
        idx = pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC")
        df = pd.DataFrame({
            "timestamp": idx,
            "open": [100, 100, 100],
            "high": [100, 101, 101],
            "low": [100, 99, 99],
            "close": [100, 100, 100],
            "volume": [1000, 1000, 1000],
        })
        labels = build_trade_outcome_labels(
            df,
            horizon_candles=1,
            stop_loss_pct=0.05,
            take_profit_pct=0.05,
            commission_pct=0,
            slippage_pct=0,
            spread_pct=0,
        )

        self.assertTrue(np.isnan(labels.iloc[0]["buy_win"]))
        self.assertTrue(np.isnan(labels.iloc[0]["sell_win"]))

    def test_hybrid_touch_or_expiry_labels_expired_by_net_return(self):
        idx = pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC")
        df = pd.DataFrame({
            "timestamp": idx,
            "open": [100, 100, 100],
            "high": [100, 102, 100],
            "low": [100, 99, 100],
            "close": [100, 101, 100],
            "volume": [1000, 1000, 1000],
        })

        labels = build_trade_outcome_labels(
            df,
            horizon_candles=1,
            stop_loss_pct=0.05,
            take_profit_pct=0.05,
            commission_pct=0,
            slippage_pct=0,
            spread_pct=0,
            label_scheme="hybrid_touch_or_expiry",
            expiry_return_threshold_pct=0.05,
        )

        self.assertEqual(labels.iloc[0]["buy_win"], 1.0)
        self.assertEqual(labels.iloc[0]["sell_win"], 0.0)

    def test_trade_labels_values_are_only_binary_or_nan(self):
        labels = build_trade_outcome_labels(
            sample_candles(120),
            horizon_candles=4,
            stop_loss_pct=0.01,
            take_profit_pct=0.015,
            commission_pct=0.001,
            slippage_pct=0.0005,
        )
        values = labels.to_numpy().ravel()
        valid_values = {value for value in values if not np.isnan(value)}

        self.assertTrue(valid_values.issubset({0.0, 1.0}))

    def test_prepared_trade_dataset_allows_only_buy_labels(self):
        features = pd.DataFrame({column: [1.0, 2.0, 3.0] for column in FEATURE_COLS})
        labels = pd.DataFrame({
            "buy_win": [1.0, 0.0, np.nan],
            "sell_win": [np.nan, np.nan, np.nan],
        })

        x_buy, y_buy, x_sell, y_sell = _prepared_trade_dataset(features, labels, FEATURE_COLS)

        self.assertEqual(len(x_buy), 2)
        self.assertEqual(len(y_buy), 2)
        self.assertEqual(len(x_sell), 0)
        self.assertEqual(len(y_sell), 0)

    def test_prepared_trade_dataset_allows_only_sell_labels(self):
        features = pd.DataFrame({column: [1.0, 2.0, 3.0] for column in FEATURE_COLS})
        labels = pd.DataFrame({
            "buy_win": [np.nan, np.nan, np.nan],
            "sell_win": [1.0, 0.0, np.nan],
        })

        x_buy, y_buy, x_sell, y_sell = _prepared_trade_dataset(features, labels, FEATURE_COLS)

        self.assertEqual(len(x_buy), 0)
        self.assertEqual(len(y_buy), 0)
        self.assertEqual(len(x_sell), 2)
        self.assertEqual(len(y_sell), 2)

    def test_prepared_trade_dataset_does_not_require_buy_sell_intersection(self):
        features = pd.DataFrame({column: [1.0, 2.0, 3.0] for column in FEATURE_COLS})
        labels = pd.DataFrame({
            "buy_win": [1.0, np.nan, np.nan],
            "sell_win": [np.nan, 1.0, np.nan],
        })

        x_buy, y_buy, x_sell, y_sell = _prepared_trade_dataset(features, labels, FEATURE_COLS)

        self.assertEqual(len(x_buy), 1)
        self.assertEqual(len(y_buy), 1)
        self.assertEqual(len(x_sell), 1)
        self.assertEqual(len(y_sell), 1)

    def test_trade_label_builder_benchmark_500_rows_under_one_second(self):
        idx = pd.date_range("2023-01-01", periods=500, freq="15min", tz="UTC")
        rng = np.random.default_rng(42)
        base = 100 + np.cumsum(rng.normal(0, 0.1, 500))
        df = pd.DataFrame({
            "timestamp": idx,
            "open": base,
            "high": base + 0.5,
            "low": base - 0.5,
            "close": base,
            "volume": 1000.0,
        })
        started = time.perf_counter()
        labels = build_trade_outcome_labels(
            df,
            horizon_candles=4,
            stop_loss_pct=0.02,
            take_profit_pct=0.03,
            commission_pct=0.001,
            slippage_pct=0.0005,
        )
        elapsed = time.perf_counter() - started

        self.assertEqual(len(labels), 500)
        self.assertLess(elapsed, 1.0)

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
        self.assertIn("buy_label_count", result)
        self.assertIn("sell_label_count", result)
        self.assertIn("buy_positive_count", result)
        self.assertIn("sell_positive_count", result)
        self.assertIn("buy_positive_rate", result)
        self.assertIn("sell_positive_rate", result)
        self.assertIn("buy_threshold", result)
        self.assertIn("sell_threshold", result)
        self.assertIn("decision_margin", result)
        self.assertIn("hold_reason", result)
        self.assertGreater(result["buy_label_count"], 0)
        self.assertGreater(result["sell_label_count"], 0)
        self.assertIn("raw_buy_label_count", result)
        self.assertIn("raw_sell_label_count", result)
        self.assertIn("feature_valid_count", result)
        self.assertIn("feature_nan_summary", result)
        self.assertIn(result["signal"], {"BUY", "SELL", "HOLD"})

    def test_xgboost_trade_labels_default_to_atr_aligned_levels(self):
        features = add_features(sample_candles(300))
        result = xgboost_signal(
            features,
            min_train_rows=30,
            strategy_params={
                "use_trade_labels": True,
                "horizon_candles": 4,
                "commission_pct": 0,
                "slippage_pct": 0,
            },
        )

        self.assertEqual(result["label_level_mode"], "atr")
        self.assertEqual(result["trade_label_scheme"], "touch_only")
        self.assertIn("atr_aligned_trade_labels", result["label_level_note"])

    def test_xgboost_trade_labels_support_hybrid_touch_or_expiry_scheme(self):
        features = add_features(sample_candles(300))
        result = xgboost_signal(
            features,
            min_train_rows=30,
            strategy_params={
                "use_trade_labels": True,
                "trade_label_scheme": "hybrid_touch_or_expiry",
                "horizon_candles": 4,
                "commission_pct": 0,
                "slippage_pct": 0,
            },
        )

        self.assertEqual(result["label_type"], "hybrid_touch_or_expiry")
        self.assertEqual(result["trade_label_scheme"], "hybrid_touch_or_expiry")
        self.assertEqual(result["expiry_return_threshold_pct"], 0.05)

    def test_xgboost_trade_label_hold_includes_hold_reason(self):
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
                "buy_win_threshold": 1.1,
                "sell_win_threshold": 1.1,
            },
        )

        self.assertEqual(result["signal"], "HOLD")
        self.assertIn(result["hold_reason"], {"probabilities_below_threshold", "no_directional_edge"})

    def test_xgboost_signal_reports_price_return_label_type(self):
        features = add_features(sample_candles(300))
        result = xgboost_signal(features, min_train_rows=30, strategy_params={"use_trade_labels": False})

        self.assertEqual(result["label_type"], "price_return")

    def test_deterministic_mode_still_generates_signal(self):
        signal = generate_strategy_signal_from_df(sample_candles(80), strategy_mode="deterministic").to_dict()

        self.assertEqual(signal["strategy_mode"], "deterministic")
        self.assertIn(signal["signal"], {"BUY", "SELL", "HOLD"})

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
