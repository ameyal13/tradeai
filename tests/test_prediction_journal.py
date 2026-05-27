import unittest

import pandas as pd

from tools.prediction_journal import evaluate_prediction_against_candles, normalize_prediction


def make_prediction(signal="BUY", **overrides):
    prediction = {
        "id": "pred-1",
        "symbol": "BTC",
        "timeframe": "15m",
        "strategy_mode": "deterministic",
        "strategy_name": "test_strategy",
        "strategy_version": "v1",
        "signal": signal,
        "confidence": 75,
        "entry_price": 100,
        "stop_loss": 95 if signal == "BUY" else 105,
        "take_profit": 110 if signal == "BUY" else 90,
        "risk_reward_ratio": 2,
        "horizon_minutes": 60,
        "input_features": {},
        "reasoning": "unit test",
        "status": "pending",
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    prediction.update(overrides)
    return prediction


def candles(rows):
    return pd.DataFrame(rows)


class PredictionOutcomeTests(unittest.TestCase):
    def test_normalize_prediction_accepts_xgboost_strategy_mode(self):
        prediction = normalize_prediction(make_prediction("BUY", strategy_mode="xgboost"))

        self.assertEqual(prediction["strategy_mode"], "xgboost")

    def test_outcome_buy_winner(self):
        outcome = evaluate_prediction_against_candles(
            make_prediction("BUY"),
            candles([
                {"timestamp": "2026-01-01T00:15:00Z", "open": 100, "high": 111, "low": 99, "close": 90},
                {"timestamp": "2026-01-01T01:00:00Z", "open": 108, "high": 109, "low": 102, "close": 106},
            ]),
            commission_pct=0,
            slippage_pct=0,
            spread_pct=0,
        )

        self.assertEqual(outcome["outcome"], "WIN")
        self.assertTrue(outcome["hit_take_profit"])
        self.assertEqual(outcome["exit_price"], 110)
        self.assertGreater(outcome["return_pct"], 0)
        self.assertEqual(outcome["raw_path"]["exit_reason"], "take_profit")

    def test_outcome_buy_loser(self):
        outcome = evaluate_prediction_against_candles(
            make_prediction("BUY"),
            candles([
                {"timestamp": "2026-01-01T00:15:00Z", "open": 100, "high": 102, "low": 94, "close": 110},
                {"timestamp": "2026-01-01T01:00:00Z", "open": 98, "high": 99, "low": 96, "close": 97},
            ]),
            commission_pct=0,
            slippage_pct=0,
            spread_pct=0,
        )

        self.assertEqual(outcome["outcome"], "LOSS")
        self.assertTrue(outcome["hit_stop_loss"])
        self.assertEqual(outcome["exit_price"], 95)
        self.assertLess(outcome["return_pct"], 0)
        self.assertEqual(outcome["raw_path"]["exit_reason"], "stop_loss")

    def test_outcome_sell_winner(self):
        outcome = evaluate_prediction_against_candles(
            make_prediction("SELL"),
            candles([
                {"timestamp": "2026-01-01T00:15:00Z", "open": 100, "high": 101, "low": 89, "close": 110},
                {"timestamp": "2026-01-01T01:00:00Z", "open": 92, "high": 96, "low": 91, "close": 94},
            ]),
            commission_pct=0,
            slippage_pct=0,
            spread_pct=0,
        )

        self.assertEqual(outcome["outcome"], "WIN")
        self.assertTrue(outcome["hit_take_profit"])
        self.assertEqual(outcome["exit_price"], 90)
        self.assertGreater(outcome["return_pct"], 0)
        self.assertEqual(outcome["raw_path"]["exit_reason"], "take_profit")

    def test_outcome_sell_loser(self):
        outcome = evaluate_prediction_against_candles(
            make_prediction("SELL"),
            candles([
                {"timestamp": "2026-01-01T00:15:00Z", "open": 100, "high": 106, "low": 98, "close": 90},
                {"timestamp": "2026-01-01T01:00:00Z", "open": 104, "high": 104, "low": 99, "close": 101},
            ]),
            commission_pct=0,
            slippage_pct=0,
            spread_pct=0,
        )

        self.assertEqual(outcome["outcome"], "LOSS")
        self.assertTrue(outcome["hit_stop_loss"])
        self.assertEqual(outcome["exit_price"], 105)
        self.assertLess(outcome["return_pct"], 0)
        self.assertEqual(outcome["raw_path"]["exit_reason"], "stop_loss")

    def test_mfe_mae_calculation_for_buy(self):
        outcome = evaluate_prediction_against_candles(
            make_prediction("BUY", take_profit=130, stop_loss=80),
            candles([
                {"timestamp": "2026-01-01T00:15:00Z", "open": 100, "high": 112, "low": 96, "close": 105},
                {"timestamp": "2026-01-01T01:00:00Z", "open": 105, "high": 108, "low": 97, "close": 104},
            ]),
            commission_pct=0,
            slippage_pct=0,
            spread_pct=0,
        )

        self.assertEqual(outcome["max_favorable_excursion_pct"], 12)
        self.assertEqual(outcome["max_adverse_excursion_pct"], 4)

    def test_expired_without_tp_sl(self):
        outcome = evaluate_prediction_against_candles(
            make_prediction("BUY", stop_loss=None, take_profit=None),
            candles([
                {"timestamp": "2026-01-01T00:30:00Z", "open": 100, "high": 103, "low": 99, "close": 102},
                {"timestamp": "2026-01-01T01:00:00Z", "open": 102, "high": 104, "low": 101, "close": 103},
            ]),
            commission_pct=0,
            slippage_pct=0,
            spread_pct=0,
        )

        self.assertEqual(outcome["outcome"], "EXPIRED")
        self.assertFalse(outcome["hit_stop_loss"])
        self.assertFalse(outcome["hit_take_profit"])
        self.assertEqual(outcome["exit_price"], 103)
        self.assertEqual(outcome["raw_path"]["exit_reason"], "expired_close")

    def test_invalid_when_insufficient_future_data(self):
        outcome = evaluate_prediction_against_candles(
            make_prediction("BUY"),
            candles([
                {"timestamp": "2025-12-31T23:45:00Z", "open": 100, "high": 105, "low": 99, "close": 103},
            ]),
            commission_pct=0,
            slippage_pct=0,
            spread_pct=0,
        )

        self.assertEqual(outcome["outcome"], "INVALID_DATA")
        self.assertEqual(outcome["raw_path"], [])

    def test_ambiguous_intrabar_buy_is_conservative_loss(self):
        outcome = evaluate_prediction_against_candles(
            make_prediction("BUY"),
            candles([
                {"timestamp": "2026-01-01T00:15:00Z", "open": 100, "high": 111, "low": 94, "close": 102},
            ]),
            commission_pct=0,
            slippage_pct=0,
            spread_pct=0,
        )

        self.assertEqual(outcome["outcome"], "LOSS")
        self.assertEqual(outcome["exit_price"], 95)
        self.assertEqual(outcome["raw_path"]["exit_reason"], "ambiguous_intrabar_conservative_loss")
        self.assertIn("ambiguous_intrabar_conservative_loss", outcome["raw_path"]["notes"])

    def test_ambiguous_intrabar_sell_is_conservative_loss(self):
        outcome = evaluate_prediction_against_candles(
            make_prediction("SELL"),
            candles([
                {"timestamp": "2026-01-01T00:15:00Z", "open": 100, "high": 106, "low": 89, "close": 99},
            ]),
            commission_pct=0,
            slippage_pct=0,
            spread_pct=0,
        )

        self.assertEqual(outcome["outcome"], "LOSS")
        self.assertEqual(outcome["exit_price"], 105)
        self.assertEqual(outcome["raw_path"]["exit_reason"], "ambiguous_intrabar_conservative_loss")
        self.assertIn("ambiguous_intrabar_conservative_loss", outcome["raw_path"]["notes"])

    def test_tp_before_later_sl_exits_at_take_profit(self):
        outcome = evaluate_prediction_against_candles(
            make_prediction("BUY"),
            candles([
                {"timestamp": "2026-01-01T00:15:00Z", "open": 100, "high": 111, "low": 99, "close": 109},
                {"timestamp": "2026-01-01T00:30:00Z", "open": 109, "high": 110, "low": 94, "close": 96},
            ]),
            commission_pct=0,
            slippage_pct=0,
            spread_pct=0,
        )

        self.assertEqual(outcome["outcome"], "WIN")
        self.assertEqual(outcome["exit_price"], 110)
        self.assertEqual(outcome["raw_path"]["exit_candle_index"], 0)
        self.assertGreater(outcome["return_pct"], 0)

    def test_take_profit_that_does_not_cover_costs_is_not_win(self):
        outcome = evaluate_prediction_against_candles(
            make_prediction("BUY", take_profit=100.1, stop_loss=95),
            candles([
                {"timestamp": "2026-01-01T00:15:00Z", "open": 100, "high": 100.2, "low": 99, "close": 100.05},
            ]),
            commission_pct=0.001,
            slippage_pct=0.0005,
            spread_pct=0.0003,
        )

        self.assertEqual(outcome["raw_path"]["exit_reason"], "take_profit")
        self.assertLess(outcome["return_pct"], 0)
        self.assertEqual(outcome["outcome"], "LOSS")
        self.assertIn("take_profit_net_loss_after_costs", outcome["raw_path"]["notes"])

    def test_sl_before_later_tp_exits_at_stop_loss(self):
        outcome = evaluate_prediction_against_candles(
            make_prediction("BUY"),
            candles([
                {"timestamp": "2026-01-01T00:15:00Z", "open": 100, "high": 101, "low": 94, "close": 96},
                {"timestamp": "2026-01-01T00:30:00Z", "open": 96, "high": 112, "low": 96, "close": 110},
            ]),
            commission_pct=0,
            slippage_pct=0,
            spread_pct=0,
        )

        self.assertEqual(outcome["outcome"], "LOSS")
        self.assertEqual(outcome["exit_price"], 95)
        self.assertEqual(outcome["raw_path"]["exit_candle_index"], 0)
        self.assertLess(outcome["return_pct"], 0)

    def test_spread_cost_reduces_return(self):
        no_spread = evaluate_prediction_against_candles(
            make_prediction("BUY", stop_loss=None, take_profit=None, horizon_minutes=30),
            candles([{"timestamp": "2026-01-01T00:15:00Z", "open": 100, "high": 101, "low": 99, "close": 100.5}]),
            commission_pct=0,
            slippage_pct=0,
            spread_pct=0,
        )
        with_spread = evaluate_prediction_against_candles(
            make_prediction("BUY", stop_loss=None, take_profit=None, horizon_minutes=30),
            candles([{"timestamp": "2026-01-01T00:15:00Z", "open": 100, "high": 101, "low": 99, "close": 100.5}]),
            commission_pct=0,
            slippage_pct=0,
            spread_pct=0.01,
        )

        self.assertLess(with_spread["return_pct"], no_spread["return_pct"])


if __name__ == "__main__":
    unittest.main()
