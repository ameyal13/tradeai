import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from research.shadow_replay import (
    combine_replay_reports,
    render_shadow_replay_markdown,
    run_shadow_replay_for_candles,
    save_shadow_replay_report,
)
from scripts.run_shadow_replay_audit import build_parser


class FakeSignal:
    def __init__(self, payload):
        self.payload = payload

    def to_dict(self):
        return dict(self.payload)


def candles(rows=20):
    idx = pd.date_range("2026-01-01", periods=rows, freq="1h", tz="UTC")
    base = [100.0] * rows
    df = pd.DataFrame({
        "timestamp": idx,
        "open": base,
        "high": [101.0] * rows,
        "low": [99.0] * rows,
        "close": base,
        "volume": [1000.0] * rows,
    })
    return df


def config(config_id="cfg1"):
    return {
        "config_id": config_id,
        "_source_classification": "unstable_watchlist",
        "symbol": "ADA",
        "timeframe": "1h",
        "strategy_mode": "xgboost",
        "horizon_candles": 4,
        "risk_reward": 2.0,
        "atr_stop_multiplier": 1.25,
        "cost_mode": "low_costs",
        "min_train_rows": 5,
        "buy_threshold": 0.58,
        "sell_threshold": 0.58,
        "trade_label_scheme": "expected_value_classification",
    }


def buy_signal():
    return FakeSignal({
        "strategy_mode": "xgboost",
        "strategy_name": "xgboost_temporal_split",
        "strategy_version": "v1",
        "signal": "BUY",
        "confidence": 70.0,
        "entry_price": 100.0,
        "stop_loss": 98.0,
        "take_profit": 104.0,
        "risk_reward_ratio": 2.0,
        "horizon_minutes": 240,
        "input_features": {"model_available": True},
        "reasoning": "fake signal",
        "model_provider": "local_xgboost",
        "model_name": "xgboost_classifier_v1",
    })


def hold_signal():
    return FakeSignal({
        "signal": "HOLD",
        "confidence": 20.0,
        "input_features": {"hold_reason": "probabilities_below_threshold"},
        "reasoning": "hold",
    })


class ShadowReplayTests(unittest.TestCase):
    def test_replay_opens_max_one_and_skips_while_open(self):
        df = candles(12)
        df.loc[8, "high"] = 105.0
        with patch("research.shadow_replay.generate_strategy_signal_from_df", return_value=buy_signal()) as generate:
            report = run_shadow_replay_for_candles(
                candles=df,
                configs=[config()],
                symbol="ADA",
                timeframe="1h",
                days=2,
                max_signals=1,
                max_configs_scanned=1,
                min_history_candles=5,
                max_cycles=5,
            )

        self.assertGreaterEqual(generate.call_count, 1)
        self.assertGreater(report["cycles_skipped_open_exists"], 0)
        self.assertEqual(report["summary"]["closed"], 1)
        self.assertEqual(report["summary"]["wins"], 1)
        self.assertEqual(report["summary"]["open"], 1)

    def test_replay_does_not_use_sentiment_by_default(self):
        seen_params = []

        def fake_generate(*args, **kwargs):
            seen_params.append(kwargs["strategy_params"])
            return hold_signal()

        with patch("research.shadow_replay.generate_strategy_signal_from_df", side_effect=fake_generate):
            run_shadow_replay_for_candles(
                candles=candles(10),
                configs=[config()],
                symbol="ADA",
                timeframe="1h",
                days=1,
                max_signals=1,
                max_configs_scanned=1,
                min_history_candles=5,
                max_cycles=2,
            )

        self.assertTrue(seen_params)
        self.assertTrue(all(params["use_sentiment"] is False for params in seen_params))

    def test_combine_and_markdown_report(self):
        run = {
            "signals": [
                {"status": "CLOSED", "outcome": "WIN", "pnl_pct": 1.0, "symbol": "ADA", "side": "LONG", "config_id": "cfg1"},
                {"status": "CLOSED", "outcome": "LOSS", "pnl_pct": -0.5, "symbol": "ADA", "side": "LONG", "config_id": "cfg1"},
            ],
            "events": [{"status": "OPEN"}, {"status": "skipped_hold"}],
        }
        combined = combine_replay_reports([run])
        report = {
            "config": {
                "registry": "crypto_multi",
                "symbols": ["ADA"],
                "timeframe": "1h",
                "days": 1,
                "max_signals": 1,
                "max_configs_scanned": 21,
                "use_sentiment": False,
            },
            "combined": combined,
            "runs": [{"symbol": "ADA", "timeframe": "1h", "cycles": 2, "selected_config_count": 1, "summary": combined["summary"]}],
        }
        markdown = render_shadow_replay_markdown(report)

        self.assertEqual(combined["summary"]["closed"], 2)
        self.assertIn("Shadow Replay Audit", markdown)
        self.assertIn("Research only", markdown)

    def test_save_shadow_replay_report_writes_json_and_markdown(self):
        report = {
            "config": {"registry": "crypto_multi", "symbols": ["ADA"], "timeframe": "1h", "days": 1, "max_signals": 1, "max_configs_scanned": 21, "use_sentiment": False},
            "combined": combine_replay_reports([]),
            "runs": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            paths = save_shadow_replay_report(report, tmp)

            self.assertTrue(Path(paths["json_path"]).exists())
            self.assertTrue(Path(paths["markdown_path"]).exists())

    def test_cli_defaults_are_research_safe(self):
        args = build_parser().parse_args([])

        self.assertEqual(args.registry, "crypto_multi")
        self.assertEqual(args.max_signals, 1)
        self.assertFalse(args.use_sentiment)


if __name__ == "__main__":
    unittest.main()
