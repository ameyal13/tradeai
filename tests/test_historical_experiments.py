import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np
import pandas as pd


def synthetic_candles(rows=180):
    idx = pd.date_range("2026-01-01", periods=rows, freq="15min", tz="UTC")
    base = np.linspace(100, 120, rows) + np.sin(np.arange(rows) / 4)
    return pd.DataFrame({
        "timestamp": idx,
        "open": base,
        "high": base + 2,
        "low": base - 2,
        "close": base + 0.5,
        "volume": 1000,
    })


def diagnostic_result():
    predictions = [
        {
            "id": "buy-win",
            "signal": "BUY",
            "confidence": 75,
            "entry_price": 100,
            "stop_loss": 95,
            "take_profit": 110,
            "risk_reward_ratio": 2,
            "created_at": "2026-01-01T08:00:00+00:00",
        },
        {
            "id": "sell-loss",
            "signal": "SELL",
            "confidence": 55,
            "entry_price": 100,
            "stop_loss": 105,
            "take_profit": 90,
            "risk_reward_ratio": 2,
            "created_at": "2026-01-01T09:00:00+00:00",
        },
        {
            "id": "hold",
            "signal": "HOLD",
            "confidence": 35,
            "entry_price": 100,
            "created_at": "2026-01-01T09:30:00+00:00",
        },
        {
            "id": "buy-expired",
            "signal": "BUY",
            "confidence": 85,
            "entry_price": 100,
            "stop_loss": 97,
            "take_profit": 106,
            "risk_reward_ratio": 2,
            "created_at": "2026-01-01T10:00:00+00:00",
        },
    ]
    outcomes = [
        {
            "prediction_id": "buy-win",
            "outcome": "WIN",
            "return_pct": 2.0,
            "hit_take_profit": True,
            "hit_stop_loss": False,
        },
        {
            "prediction_id": "sell-loss",
            "outcome": "LOSS",
            "return_pct": -1.0,
            "hit_take_profit": False,
            "hit_stop_loss": True,
        },
        {
            "prediction_id": "buy-expired",
            "outcome": "EXPIRED",
            "return_pct": 0.2,
            "hit_take_profit": False,
            "hit_stop_loss": False,
        },
    ]
    return {
        "predictions": predictions,
        "outcomes": outcomes,
        "metrics": [{
            "evaluated_predictions": 3,
            "win_rate": 33.333333,
            "average_return": 0.4,
            "total_return_pct": 1.2,
            "profit_factor": 2.2,
            "max_drawdown": 1.0,
            "sharpe": 0.5,
        }],
        "assumptions": {},
    }


def xgboost_diagnostic_result():
    return {
        "predictions": [
            {
                "id": "xgb-hold-1",
                "signal": "HOLD",
                "confidence": 44,
                "entry_price": 100,
                "input_features": {
                    "label_type": "trade_outcome_directional",
                    "min_train_rows": 150,
                    "label_level_mode": "atr",
                    "label_horizon_candles": 4,
                    "label_stop_loss_pct": 0.03,
                    "label_take_profit_pct": 0.045,
                    "label_atr_stop_multiplier": 1.5,
                    "label_atr_take_profit_multiplier": None,
                    "label_min_risk_reward": 1.5,
                    "label_costs": {"commission_pct": 0.001, "slippage_pct": 0.0005, "spread_pct": 0.0003},
                    "raw_buy_label_count": 150,
                    "raw_sell_label_count": 148,
                    "raw_buy_positive_count": 40,
                    "raw_sell_positive_count": 38,
                    "feature_valid_count": 130,
                    "feature_nan_summary": {"rsi": 14, "macd_hist": 33},
                    "probability_buy_win": 0.41,
                    "probability_sell_win": 0.38,
                    "buy_label_count": 120,
                    "sell_label_count": 118,
                    "buy_positive_rate": 0.25,
                    "sell_positive_rate": 0.2,
                    "hold_reason": "probabilities_below_threshold",
                },
            },
            {
                "id": "xgb-hold-2",
                "signal": "HOLD",
                "confidence": 30,
                "entry_price": 101,
                "input_features": {
                    "label_type": "trade_outcome_directional",
                    "min_train_rows": 150,
                    "label_level_mode": "atr",
                    "label_horizon_candles": 4,
                    "label_stop_loss_pct": 0.03,
                    "label_take_profit_pct": 0.045,
                    "label_atr_stop_multiplier": 1.5,
                    "label_atr_take_profit_multiplier": None,
                    "label_min_risk_reward": 1.5,
                    "label_costs": {"commission_pct": 0.001, "slippage_pct": 0.0005, "spread_pct": 0.0003},
                    "raw_buy_label_count": 130,
                    "raw_sell_label_count": 128,
                    "raw_buy_positive_count": 20,
                    "raw_sell_positive_count": 18,
                    "feature_valid_count": 110,
                    "feature_nan_summary": {"rsi": 14, "macd_hist": 33},
                    "probability_buy_win": 0.35,
                    "probability_sell_win": 0.33,
                    "buy_label_count": 100,
                    "sell_label_count": 98,
                    "buy_positive_rate": 0.1,
                    "sell_positive_rate": 0.12,
                    "hold_reason": "insufficient_train_rows",
                },
            },
        ],
        "outcomes": [],
        "metrics": [],
        "assumptions": {},
    }


class HistoricalExperimentTests(unittest.IsolatedAsyncioTestCase):
    async def test_generates_report_with_synthetic_data(self):
        import scripts.run_historical_experiments as script

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(script, "fetch_binance_klines", new=AsyncMock(return_value=synthetic_candles())):
                report = await script.run_experiments(
                    symbols=["BTC"],
                    timeframes=["15m"],
                    strategy_modes=["deterministic"],
                    max_predictions=5,
                    reports_dir=tmp,
                )

            self.assertEqual(len(report["summary"]), 1)
            self.assertTrue(Path(report["report_paths"]["json"]).exists())
            self.assertTrue(Path(report["report_paths"]["csv"]).exists())

    async def test_does_not_persist_by_default(self):
        import scripts.run_historical_experiments as script

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(script, "PredictionStore") as store_cls:
                with patch.object(script, "fetch_binance_klines", new=AsyncMock(return_value=synthetic_candles())):
                    await script.run_experiments(
                        symbols=["BTC"],
                        timeframes=["15m"],
                        strategy_modes=["deterministic"],
                        max_predictions=3,
                        persist=False,
                        reports_dir=tmp,
                    )

        store_cls.assert_not_called()

    async def test_continues_if_one_symbol_fails(self):
        import scripts.run_historical_experiments as script

        async def fetch(symbol, *args, **kwargs):
            if symbol == "ETH":
                raise RuntimeError("network unavailable")
            return synthetic_candles()

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(script, "fetch_binance_klines", new=AsyncMock(side_effect=fetch)):
                report = await script.run_experiments(
                    symbols=["BTC", "ETH"],
                    timeframes=["15m"],
                    strategy_modes=["deterministic"],
                    max_predictions=3,
                    reports_dir=tmp,
                )

        by_symbol = {row["symbol"]: row for row in report["summary"]}
        self.assertIn("historical_data_error", by_symbol["ETH"]["warnings"])
        self.assertGreaterEqual(by_symbol["BTC"]["total_predictions"], 1)

    async def test_respects_max_predictions(self):
        import scripts.run_historical_experiments as script

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(script, "fetch_binance_klines", new=AsyncMock(return_value=synthetic_candles())):
                report = await script.run_experiments(
                    symbols=["BTC"],
                    timeframes=["15m"],
                    strategy_modes=["deterministic"],
                    max_predictions=4,
                    reports_dir=tmp,
                )

        self.assertLessEqual(report["summary"][0]["total_predictions"], 4)

    async def test_respects_symbols_and_timeframes(self):
        import scripts.run_historical_experiments as script

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(script, "fetch_binance_klines", new=AsyncMock(return_value=synthetic_candles())):
                report = await script.run_experiments(
                    symbols=["BTC", "SOL"],
                    timeframes=["15m", "1h"],
                    strategy_modes=["deterministic"],
                    max_predictions=2,
                    reports_dir=tmp,
                )

        combos = {(row["symbol"], row["timeframe"]) for row in report["summary"]}
        self.assertEqual(combos, {("BTC", "15m"), ("BTC", "1h"), ("SOL", "15m"), ("SOL", "1h")})

    async def test_summary_includes_minimum_metrics(self):
        import scripts.run_historical_experiments as script

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(script, "fetch_binance_klines", new=AsyncMock(return_value=synthetic_candles())):
                report = await script.run_experiments(
                    symbols=["BTC"],
                    timeframes=["15m"],
                    strategy_modes=["deterministic"],
                    max_predictions=2,
                    reports_dir=tmp,
                )

        row = report["summary"][0]
        for key in [
            "symbol", "timeframe", "strategy_mode", "total_predictions",
            "evaluated_predictions", "win_rate", "average_return",
            "total_return_pct", "profit_factor", "max_drawdown", "sharpe",
            "invalid_count", "warnings",
        ]:
            self.assertIn(key, row)

    async def test_report_includes_diagnostic_metrics(self):
        import scripts.run_historical_experiments as script

        row = script.summarize_run("BTC", "15m", "deterministic", result=diagnostic_result())

        for key in [
            "buy_count", "sell_count", "hold_count", "buy_win_rate",
            "sell_win_rate", "buy_average_return", "sell_average_return",
            "avg_confidence", "avg_risk_reward", "avg_stop_distance_pct",
            "avg_take_profit_distance_pct", "tp_hit_count", "sl_hit_count",
            "expired_count", "confidence_buckets",
            "hourly_performance_utc", "best_hour_utc", "worst_hour_utc",
        ]:
            self.assertIn(key, row)
        self.assertEqual(row["buy_count"], 2)
        self.assertEqual(row["sell_count"], 1)
        self.assertEqual(row["hold_count"], 1)
        self.assertEqual(row["best_hour_utc"], "08:00")
        self.assertEqual(row["worst_hour_utc"], "09:00")

    async def test_hourly_performance_groups_predictions_by_utc_hour(self):
        import scripts.run_historical_experiments as script

        row = script.summarize_run("BTC", "15m", "deterministic", result=diagnostic_result())
        hourly = row["hourly_performance_utc"]

        self.assertEqual(hourly["08:00"]["evaluated_predictions"], 1)
        self.assertEqual(hourly["08:00"]["win_rate"], 100)
        self.assertEqual(hourly["09:00"]["evaluated_predictions"], 1)
        self.assertEqual(hourly["09:00"]["average_return"], -1.0)
        self.assertEqual(hourly["10:00"]["evaluated_predictions"], 1)

    async def test_separates_buy_vs_sell_performance(self):
        import scripts.run_historical_experiments as script

        row = script.summarize_run("BTC", "15m", "deterministic", result=diagnostic_result())

        self.assertEqual(row["buy_win_rate"], 50)
        self.assertEqual(row["sell_win_rate"], 0)
        self.assertEqual(row["buy_average_return"], 1.1)
        self.assertEqual(row["sell_average_return"], -1.0)

    async def test_separates_exit_reasons(self):
        import scripts.run_historical_experiments as script

        row = script.summarize_run("BTC", "15m", "deterministic", result=diagnostic_result())

        self.assertEqual(row["tp_hit_count"], 1)
        self.assertEqual(row["sl_hit_count"], 1)
        self.assertEqual(row["expired_count"], 1)

    async def test_previous_report_fields_remain_available(self):
        import scripts.run_historical_experiments as script

        row = script.summarize_run("BTC", "15m", "deterministic", result=diagnostic_result())

        for key in [
            "symbol", "timeframe", "strategy_mode", "total_predictions",
            "evaluated_predictions", "win_rate", "average_return",
            "total_return_pct", "profit_factor", "max_drawdown", "sharpe",
            "invalid_count", "warnings",
        ]:
            self.assertIn(key, row)

    async def test_xgboost_is_allowed_strategy_mode(self):
        import scripts.run_historical_experiments as script

        self.assertEqual(script.validate_strategy_modes(["deterministic", "xgboost"]), ["deterministic", "xgboost"])

    async def test_4h_horizon_uses_enough_minutes_for_future_candles(self):
        import scripts.run_historical_experiments as script

        self.assertEqual(script.horizon_candles_for_interval(60, "4h"), 4)
        self.assertEqual(script.effective_horizon_minutes(60, "4h"), 960)

    async def test_1h_horizon_extends_when_single_candle_would_expire_too_often(self):
        import scripts.run_historical_experiments as script

        self.assertEqual(script.horizon_candles_for_interval(60, "1h"), 4)
        self.assertEqual(script.effective_horizon_minutes(60, "1h"), 240)

    async def test_run_experiments_passes_effective_horizon_to_replay(self):
        import scripts.run_historical_experiments as script

        replay_result = {
            "predictions": [],
            "outcomes": [],
            "metrics": [],
            "assumptions": {},
        }
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(script, "fetch_binance_klines", new=AsyncMock(return_value=synthetic_candles())):
                with patch.object(script, "run_historical_replay", return_value=replay_result) as replay:
                    report = await script.run_experiments(
                        symbols=["BTC"],
                        timeframes=["4h"],
                        strategy_modes=["xgboost"],
                        horizon_minutes=60,
                        reports_dir=tmp,
                    )

        kwargs = replay.call_args.kwargs
        self.assertEqual(kwargs["strategy_mode"], "xgboost")
        self.assertEqual(kwargs["horizon_candles"], 4)
        self.assertEqual(kwargs["horizon_minutes"], 960)
        self.assertEqual(kwargs["strategy_params"], {"use_sentiment": False})
        self.assertEqual(report["runs"][0]["effective_horizon_minutes"], 960)

    async def test_use_trade_labels_flag_passes_strategy_param(self):
        import scripts.run_historical_experiments as script

        replay_result = {
            "predictions": [],
            "outcomes": [],
            "metrics": [],
            "assumptions": {},
        }
        csv_text = ""
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(script, "fetch_binance_klines", new=AsyncMock(return_value=synthetic_candles())):
                with patch.object(script, "run_historical_replay", return_value=replay_result) as replay:
                    report = await script.run_experiments(
                        symbols=["BTC"],
                        timeframes=["1h"],
                        strategy_modes=["xgboost"],
                        reports_dir=tmp,
                        use_trade_labels=True,
                    )
                    csv_text = Path(report["report_paths"]["csv"]).read_text(encoding="utf-8")

        self.assertEqual(replay.call_args.kwargs["strategy_params"], {
            "use_sentiment": False,
            "use_trade_labels": True,
            "horizon_candles": 4,
            "min_train_rows": 150,
            "commission_pct": 0.001,
            "slippage_pct": 0.0005,
            "spread_pct": 0.0003,
        })
        self.assertGreaterEqual(replay.call_args.kwargs["min_history"], 116)
        self.assertTrue(report["config"]["use_trade_labels"])
        self.assertEqual(report["config"]["trade_label_min_train_rows"], 150)
        self.assertTrue(report["runs"][0]["use_trade_labels"])
        self.assertTrue(report["summary"][0]["use_trade_labels"])
        self.assertIn("use_trade_labels", csv_text)
        self.assertIn("true", csv_text.lower())

    async def test_trade_label_replay_uses_recent_windows_with_enough_history(self):
        import scripts.run_historical_experiments as script

        replay_result = {
            "predictions": [],
            "outcomes": [],
            "metrics": [],
            "assumptions": {},
        }
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(script, "fetch_binance_klines", new=AsyncMock(return_value=synthetic_candles(600))):
                with patch.object(script, "run_historical_replay", return_value=replay_result) as replay:
                    await script.run_experiments(
                        symbols=["BTC"],
                        timeframes=["1h"],
                        strategy_modes=["xgboost"],
                        max_predictions=80,
                        reports_dir=tmp,
                        use_trade_labels=True,
                    )

        self.assertEqual(replay.call_args.kwargs["min_history"], 516)
        self.assertEqual(replay.call_args.kwargs["strategy_params"]["min_train_rows"], 150)

    async def test_historical_report_includes_sentiment_disabled_note(self):
        import scripts.run_historical_experiments as script

        replay_result = {
            "predictions": [],
            "outcomes": [],
            "metrics": [],
            "assumptions": {},
        }
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(script, "fetch_binance_klines", new=AsyncMock(return_value=synthetic_candles())):
                with patch.object(script, "run_historical_replay", return_value=replay_result):
                    report = await script.run_experiments(
                        symbols=["BTC"],
                        timeframes=["1h"],
                        strategy_modes=["xgboost"],
                        reports_dir=tmp,
                    )

        self.assertFalse(report["config"]["sentiment_used"])
        self.assertIn("avoid time leakage", report["config"]["sentiment_note"])
        self.assertFalse(report["summary"][0]["sentiment_used"])
        self.assertIn("requested_horizon_minutes", report["summary"][0])
        self.assertIn("effective_horizon_minutes", report["summary"][0])
        self.assertIn("evaluation_horizon_candles", report["summary"][0])

    async def test_historical_report_aggregates_xgboost_diagnostics(self):
        import scripts.run_historical_experiments as script

        row = script.summarize_run("BTC", "1h", "xgboost", result=xgboost_diagnostic_result())

        self.assertEqual(row["label_type"], "trade_outcome_directional")
        self.assertEqual(row["hold_reasons_summary"], {
            "probabilities_below_threshold": 1,
            "insufficient_train_rows": 1,
        })
        self.assertEqual(row["avg_probability_buy_win"], 0.38)
        self.assertEqual(row["max_probability_sell_win"], 0.38)
        self.assertEqual(row["avg_buy_label_count"], 110)
        self.assertEqual(row["avg_sell_positive_rate"], 0.16)
        self.assertEqual(row["raw_buy_label_count"], 140)
        self.assertEqual(row["raw_sell_label_count"], 138)
        self.assertEqual(row["raw_buy_positive_count"], 30)
        self.assertEqual(row["raw_sell_positive_count"], 28)
        self.assertEqual(row["feature_valid_count"], 120)
        self.assertEqual(row["feature_nan_summary"], {"rsi": 28, "macd_hist": 66})
        self.assertEqual(row["label_level_mode"], "atr")
        self.assertEqual(row["label_horizon_candles"], 4)
        self.assertEqual(row["label_params"]["min_train_rows"], 150)
        self.assertEqual(row["label_params"]["label_costs"]["spread_pct"], 0.0003)

    async def test_requirements_does_not_include_unused_lightgbm(self):
        content = Path("requirements.txt").read_text(encoding="utf-8").lower()

        self.assertNotIn("lightgbm", content)


if __name__ == "__main__":
    unittest.main()
