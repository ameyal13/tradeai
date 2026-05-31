import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np
import pandas as pd

from research.multi_window_validator import (
    aggregate_multi_window_results,
    build_watchlist_setups,
    classify_multi_window_setup,
    generate_rolling_windows,
    render_multi_window_markdown,
    run_multi_window_validation,
    run_setup_across_windows,
    save_multi_window_report,
)
from scripts.run_multi_window_validation import build_parser


def sample_candles(rows=1500):
    idx = pd.date_range("2026-01-01", periods=rows, freq="1h", tz="UTC")
    base = 100 + np.linspace(0, 8, rows) + np.sin(np.arange(rows) / 12)
    return pd.DataFrame({
        "timestamp": idx,
        "open": base,
        "high": base + 1.0,
        "low": base - 1.0,
        "close": base + 0.1,
        "volume": 1000.0,
    })


def window_row(
    *,
    valid=True,
    validation_positive=True,
    beats_random=True,
    beats_deterministic=True,
    validation_pf=1.2,
    validation_avg=0.2,
    validation_drawdown=8.0,
    test_confirms=True,
    test_pf=1.1,
    test_avg=0.1,
    bias="balanced",
):
    return {
        "window_status": "valid" if valid else "insufficient_data",
        "validation_positive": validation_positive,
        "beats_random_validation": beats_random,
        "beats_deterministic_validation": beats_deterministic,
        "validation_profit_factor": validation_pf,
        "validation_avg_return": validation_avg,
        "validation_drawdown": validation_drawdown,
        "test_confirms": test_confirms,
        "test_profit_factor": test_pf,
        "test_avg_return": test_avg,
        "directional_bias": bias,
        "n_trades": 12,
    }


def fake_result(train_end=120, validation_start=137, validation_end=170, test_start=187):
    return {
        "experiment_id": "fake",
        "classification": "reject",
        "reasons": [],
        "split": {
            "train_rows": 120,
            "validation_rows": 34,
            "test_rows": 20,
            "train_end": train_end,
            "validation_start": validation_start,
            "validation_end": validation_end,
            "test_start": test_start,
        },
        "validation_metrics": {
            "n_trades": 5,
            "avg_return_pct": 0.1,
            "profit_factor": 1.2,
            "max_drawdown_pct": 4,
            "buy_trades": 3,
            "sell_trades": 2,
        },
        "test_metrics": {
            "n_trades": 4,
            "avg_return_pct": 0.1,
            "profit_factor": 1.1,
            "max_drawdown_pct": 5,
            "buy_trades": 2,
            "sell_trades": 2,
        },
        "baselines": {
            "validation": {
                "random_same_count": {"avg_return_pct": -0.1, "profit_factor": 0.8},
                "deterministic": {"avg_return_pct": -0.05, "profit_factor": 0.9},
            }
        },
        "diagnostics": {
            "beats_random_validation": True,
            "beats_deterministic_validation": True,
            "validation_positive": True,
            "test_confirms": True,
            "validation_directional_exposure": {"directional_bias": "balanced"},
        },
    }


class MultiWindowValidatorTests(unittest.TestCase):
    def test_build_watchlist_setups_are_fixed_to_two_sol_configs(self):
        setups = build_watchlist_setups()

        self.assertEqual(len(setups), 2)
        self.assertEqual({row["symbol"] for row in setups}, {"SOL"})
        self.assertEqual({row["timeframe"] for row in setups}, {"1h"})
        self.assertEqual({row["horizon_candles"] for row in setups}, {16})
        self.assertEqual({row["risk_reward"] for row in setups}, {2.0})
        self.assertEqual({row["atr_stop_multiplier"] for row in setups}, {1.25, 1.5})
        self.assertEqual({row["cost_mode"] for row in setups}, {"low_costs"})
        self.assertEqual({row["strategy_mode"] for row in setups}, {"xgboost"})

    def test_rolling_windows_are_generated_correctly(self):
        candles = sample_candles(1500)
        windows = generate_rolling_windows(candles, window_size_candles=600, step_size_candles=250, horizon_candles=16)

        self.assertEqual([row["start_index"] for row in windows], [0, 250, 500, 750, 900])
        self.assertEqual([row["end_index"] for row in windows], [600, 850, 1100, 1350, 1500])
        self.assertTrue(all(row["purge_candles"] == 16 for row in windows))

    def test_run_setup_across_windows_keeps_purged_split_metadata(self):
        setup = build_watchlist_setups(max_candles=700)[0]
        candles = sample_candles(700)

        with patch("research.multi_window_validator.run_experiment_on_candles", return_value=fake_result()) as run:
            result = run_setup_across_windows(setup, candles, window_size_candles=300, step_size_candles=200)

        self.assertGreaterEqual(run.call_count, 3)
        first = result["windows"][0]
        split = first["split"]
        self.assertGreater(split["validation_start"], split["train_end"] + setup["horizon_candles"])
        self.assertGreater(split["test_start"], split["validation_end"] + setup["horizon_candles"])

    def test_less_than_three_valid_windows_is_needs_more_data(self):
        aggregate = aggregate_multi_window_results([window_row(), window_row(valid=False), window_row()])

        self.assertEqual(aggregate["valid_windows"], 2)
        self.assertEqual(classify_multi_window_setup(aggregate), "needs_more_data")

    def test_test_positive_does_not_rescue_failed_validation(self):
        rows = [
            window_row(validation_positive=False, beats_random=False, beats_deterministic=False, validation_pf=0.8, validation_avg=-0.1, test_confirms=True, test_pf=1.5, test_avg=0.4),
            window_row(validation_positive=False, beats_random=False, beats_deterministic=False, validation_pf=0.7, validation_avg=-0.2, test_confirms=True, test_pf=1.4, test_avg=0.3),
            window_row(validation_positive=False, beats_random=False, beats_deterministic=False, validation_pf=0.9, validation_avg=-0.05, test_confirms=True, test_pf=1.2, test_avg=0.2),
        ]

        aggregate = aggregate_multi_window_results(rows)

        self.assertEqual(classify_multi_window_setup(aggregate), "multi_window_reject")

    def test_stable_research_candidate_classifies_correctly(self):
        rows = [window_row(), window_row(), window_row(validation_positive=False, beats_random=True, beats_deterministic=False, validation_pf=1.0, validation_avg=0.01)]

        aggregate = aggregate_multi_window_results(rows)

        self.assertEqual(classify_multi_window_setup(aggregate), "stable_research_candidate")

    def test_unstable_watchlist_classifies_correctly(self):
        rows = [
            window_row(),
            window_row(validation_positive=False, beats_random=False, beats_deterministic=False, validation_pf=0.9, validation_avg=-0.1),
            window_row(validation_positive=False, beats_random=False, beats_deterministic=False, validation_pf=0.8, validation_avg=-0.2),
        ]

        aggregate = aggregate_multi_window_results(rows)

        self.assertEqual(classify_multi_window_setup(aggregate), "unstable_watchlist")

    def test_multi_window_reject_classifies_correctly(self):
        rows = [
            window_row(validation_positive=False, beats_random=False, beats_deterministic=False, validation_pf=0.7, validation_avg=-0.2),
            window_row(validation_positive=False, beats_random=False, beats_deterministic=False, validation_pf=0.8, validation_avg=-0.1),
            window_row(validation_positive=False, beats_random=False, beats_deterministic=False, validation_pf=0.9, validation_avg=-0.05),
        ]

        aggregate = aggregate_multi_window_results(rows)

        self.assertEqual(classify_multi_window_setup(aggregate), "multi_window_reject")

    def test_markdown_is_generated(self):
        setup = build_watchlist_setups()[0]
        aggregate = aggregate_multi_window_results([window_row(), window_row(), window_row()])
        summary = {
            "created_at": "2026-01-01T00:00:00+00:00",
            "data": {
                "requested_max_candles": 1500,
                "actual_rows_loaded": 1000,
                "data_source": "cache",
                "data_cache_path": "cache.csv",
                "data_warning": "cache_incomplete",
            },
            "classification_counts": {"stable_research_candidate": 1},
            "setups": [{
                "setup": setup,
                "aggregate": aggregate,
                "classification": "stable_research_candidate",
                "windows": [window_row(), window_row()],
            }],
        }

        markdown = render_multi_window_markdown(summary)

        self.assertIn("Multi-Window Validation Summary", markdown)
        self.assertIn("stable_research_candidate", markdown)
        self.assertIn("Validation selects; test only confirms", markdown)
        self.assertIn("requested max candles: `1500`", markdown)
        self.assertIn("actual rows loaded: `1000`", markdown)
        self.assertIn("cache_incomplete", markdown)

    def test_save_multi_window_report_writes_json_and_markdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = {"created_at": "now", "classification_counts": {}, "setups": []}
            paths = save_multi_window_report(summary, output_dir=tmp)

            self.assertTrue(Path(paths["json_path"]).exists())
            self.assertTrue(Path(paths["markdown_path"]).exists())

    def test_cli_parser_defaults(self):
        args = build_parser().parse_args([])

        self.assertEqual(args.symbol, "SOL")
        self.assertEqual(args.timeframe, "1h")
        self.assertEqual(args.window_size_candles, 600)
        self.assertEqual(args.step_size_candles, 250)
        self.assertEqual(args.max_candles, 1500)
        self.assertFalse(args.refresh_cache)

    def test_cli_parser_accepts_refresh_cache(self):
        args = build_parser().parse_args(["--refresh-cache"])

        self.assertTrue(args.refresh_cache)

    def test_run_multi_window_report_includes_requested_and_actual_rows(self):
        import asyncio
        import research.multi_window_validator as validator

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(validator, "load_experiment_candles", new=AsyncMock(return_value={
                "candles": sample_candles(1000),
                "data_source": "cache",
                "data_cache_path": "cache.csv",
                "data_warning": "cache_incomplete",
            })) as loader:
                with patch.object(validator, "run_setup_across_windows", return_value={
                    "setup": build_watchlist_setups()[0],
                    "aggregate": {"total_windows": 0, "valid_windows": 0, "insufficient_windows": 0},
                    "classification": "needs_more_data",
                    "windows": [],
                }):
                    summary = asyncio.run(run_multi_window_validation(
                        symbol="SOL",
                        timeframe="1h",
                        max_candles=1500,
                        output_dir=tmp,
                        refresh_cache=True,
                    ))

        self.assertEqual(summary["data"]["requested_max_candles"], 1500)
        self.assertEqual(summary["data"]["actual_rows_loaded"], 1000)
        self.assertEqual(summary["data"]["data_warning"], "cache_incomplete")
        self.assertTrue(summary["data"]["refresh_cache"])
        self.assertTrue(loader.call_args.kwargs["refresh_cache"])


if __name__ == "__main__":
    unittest.main()
