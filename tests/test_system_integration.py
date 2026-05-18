import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from tools.prediction_journal import (
    PredictionStore,
    evaluate_prediction_against_candles,
    metrics_by_signal,
    metrics_by_strategy,
    metrics_by_strategy_mode,
    metrics_by_symbol_timeframe,
    parse_dt,
    prediction_payload_from_signal_response,
)
from tools.strategy_signals import generate_strategy_signal_from_df, model_based_signal_from_df


REQUIRED_JOURNAL_FIELDS = {
    "symbol",
    "timeframe",
    "strategy_mode",
    "strategy_name",
    "strategy_version",
    "signal",
    "confidence",
    "entry_price",
    "stop_loss",
    "take_profit",
    "risk_reward_ratio",
    "horizon_minutes",
    "input_features",
    "reasoning",
    "created_at",
}


def sample_candles(rows=90):
    idx = pd.date_range("2026-01-01", periods=rows, freq="h", tz="UTC")
    base = np.linspace(120, 100, rows) + np.sin(np.arange(rows) / 4) * 1.5
    return pd.DataFrame({
        "timestamp": idx,
        "open": base,
        "high": base + 2,
        "low": base - 2,
        "close": base + np.sin(np.arange(rows) / 5),
        "volume": np.linspace(2000, 3000, rows),
    })


def response_from_strategy_signal(strategy_signal, symbol="BTC"):
    signal = strategy_signal.to_dict()
    return {
        "symbol": symbol,
        "signal_type": signal["signal"],
        "confidence": signal["confidence"],
        "price_at_signal": signal["entry_price"],
        "entry_price": signal["entry_price"],
        "stop_loss": signal["stop_loss"],
        "take_profit": signal["take_profit"],
        "risk_reward_ratio": signal["risk_reward_ratio"],
        "horizon_minutes": signal["horizon_minutes"],
        "strategy_mode": signal["strategy_mode"],
        "strategy_name": signal["strategy_name"],
        "strategy_version": signal["strategy_version"],
        "input_features": signal["input_features"],
        "reasoning": signal["reasoning"],
        "model_provider": signal["model_provider"],
        "model_name": signal["model_name"],
    }


class SystemIntegrationAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = PredictionStore(file_path=Path(self.tmp.name) / "journal.json")

    def tearDown(self):
        self.tmp.cleanup()

    def generate_and_store(self, strategy_mode):
        strategy_signal = generate_strategy_signal_from_df(
            sample_candles(),
            strategy_mode=strategy_mode,
            provider="none",
            horizon_minutes=60,
        )
        response = response_from_strategy_signal(strategy_signal)
        payload = prediction_payload_from_signal_response(
            response,
            timeframe="1h",
            requested_strategy_mode=strategy_mode,
            requested_horizon_minutes=60,
            provider="none",
        )
        return self.store.create_prediction(payload)

    def assert_valid_journal_prediction(self, prediction, strategy_mode):
        self.assertTrue(REQUIRED_JOURNAL_FIELDS.issubset(prediction.keys()))
        self.assertEqual(prediction["strategy_mode"], strategy_mode)
        self.assertEqual(prediction["symbol"], "BTC")
        self.assertEqual(prediction["timeframe"], "1h")
        self.assertIn(prediction["signal"], {"BUY", "SELL", "HOLD"})
        self.assertIsInstance(prediction["input_features"], dict)
        self.assertTrue(prediction["reasoning"])

    def test_generate_deterministic_signal_and_store_prediction(self):
        prediction = self.generate_and_store("deterministic")

        self.assert_valid_journal_prediction(prediction, "deterministic")
        self.assertEqual(len(self.store.list_predictions()), 1)

    def test_generate_model_based_signal_and_store_prediction(self):
        prediction = self.generate_and_store("model_based")

        self.assert_valid_journal_prediction(prediction, "model_based")
        self.assertIn("model_available", prediction["input_features"])

    def test_generate_hybrid_signal_and_store_prediction(self):
        prediction = self.generate_and_store("hybrid")

        self.assert_valid_journal_prediction(prediction, "hybrid")
        self.assertFalse(prediction["input_features"]["llm_can_change_direction"])

    def test_evaluate_saved_prediction_and_metrics(self):
        prediction = self.generate_and_store("deterministic")
        created_at = parse_dt(prediction["created_at"])
        entry = prediction["entry_price"]
        signal = prediction["signal"]

        if signal == "SELL":
            high = entry + 1
            low = (prediction["take_profit"] or entry * 0.98) - 1
            close = low
        elif signal == "BUY":
            high = (prediction["take_profit"] or entry * 1.02) + 1
            low = entry - 1
            close = high
        else:
            high = entry + 1
            low = entry - 1
            close = entry

        candles = pd.DataFrame([{
            "timestamp": (created_at + pd.Timedelta(minutes=60)).isoformat(),
            "open": entry,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1000,
        }])
        outcome = evaluate_prediction_against_candles(prediction, candles, commission_pct=0, slippage_pct=0)
        self.store.create_outcome(outcome)
        self.store.update_prediction_status(prediction["id"], "evaluated")

        stored = self.store.get_prediction(prediction["id"])
        self.assertEqual(stored["status"], "evaluated")
        self.assertEqual(len(self.store.list_outcomes()), 1)
        self.assertTrue(metrics_by_signal(self.store.list_predictions(), self.store.list_outcomes()))
        self.assertTrue(metrics_by_strategy(self.store.list_predictions(), self.store.list_outcomes()))

    def test_metrics_by_strategy_mode_include_total_signals_and_evaluated_predictions(self):
        for mode in ["deterministic", "model_based", "hybrid"]:
            self.generate_and_store(mode)

        first = self.store.list_predictions()[-1]
        created_at = parse_dt(first["created_at"])
        outcome = evaluate_prediction_against_candles(
            first,
            pd.DataFrame([{
                "timestamp": (created_at + pd.Timedelta(minutes=60)).isoformat(),
                "open": first["entry_price"],
                "high": first["entry_price"] + 1,
                "low": first["entry_price"] - 1,
                "close": first["entry_price"],
                "volume": 1000,
            }]),
            commission_pct=0,
            slippage_pct=0,
        )
        self.store.create_outcome(outcome)

        by_mode = metrics_by_strategy_mode(self.store.list_predictions(), self.store.list_outcomes())
        by_symbol_timeframe = metrics_by_symbol_timeframe(self.store.list_predictions(), self.store.list_outcomes())

        self.assertEqual(sum(row["total_signals"] for row in by_mode), 3)
        self.assertEqual(sum(row["evaluated_predictions"] for row in by_mode), 1)
        self.assertEqual(by_symbol_timeframe[0]["symbol_timeframe"], "BTC:1h")

    def test_model_based_uses_latest_closed_candle_without_training_on_it(self):
        signal = model_based_signal_from_df(sample_candles(120), min_train_rows=30).to_dict()

        if signal["input_features"].get("model_available"):
            split = signal["input_features"]["temporal_split"]
            self.assertGreater(split["train_rows"], 0)
            self.assertGreaterEqual(split["validation_rows"], 0)
            self.assertEqual(signal["input_features"]["prediction_row"], "latest_closed_candle")
        else:
            self.assertEqual(signal["signal"], "HOLD")
            self.assertLessEqual(signal["confidence"], 30)


if __name__ == "__main__":
    unittest.main()
