import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np
import pandas as pd

from research.experiment_grid import build_experiment_grid, experiment_id, load_grid_checkpoint, save_grid_checkpoint
from research.experiment_runner import (
    _random_baseline,
    classify_result,
    purged_train_validation_test_split,
    run_experiment,
)
from research.experiment_store import ExperimentStore


def sample_candles(rows=320):
    idx = pd.date_range("2026-01-01", periods=rows, freq="1h", tz="UTC")
    base = 100 + np.linspace(0, 12, rows) + np.sin(np.arange(rows) / 5)
    return pd.DataFrame({
        "timestamp": idx,
        "open": base,
        "high": base + 1.8,
        "low": base - 1.8,
        "close": base + np.sin(np.arange(rows) / 7) * 0.4,
        "volume": 1000 + np.arange(rows),
    })


class ExperimentGridTests(unittest.TestCase):
    def test_grid_generates_only_expected_combinations(self):
        grid = build_experiment_grid()

        self.assertEqual(len(grid), 16)
        self.assertEqual({row["symbol"] for row in grid}, {"ETH", "SOL"})
        self.assertEqual({row["timeframe"] for row in grid}, {"1h"})
        self.assertEqual({row["horizon_candles"] for row in grid}, {16, 24})
        self.assertEqual({row["risk_reward"] for row in grid}, {2.0})
        self.assertEqual({row["atr_stop_multiplier"] for row in grid}, {1.25, 1.5})
        self.assertEqual({row["cost_mode"] for row in grid}, {"low_costs", "medium_costs_current"})
        self.assertEqual({row["strategy_mode"] for row in grid}, {"xgboost"})

    def test_experiment_id_is_stable(self):
        config = build_experiment_grid()[0]
        shuffled = dict(reversed(list(config.items())))

        self.assertEqual(experiment_id(config), experiment_id(shuffled))
        self.assertEqual(config["experiment_id"], experiment_id(config))

    def test_checkpoint_saves_and_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.json"
            grid = build_experiment_grid()[:2]
            save_grid_checkpoint(path, grid, completed_ids={grid[0]["experiment_id"]}, result_path="results.jsonl")
            loaded = load_grid_checkpoint(path)

        self.assertEqual(loaded["grid"], grid)
        self.assertEqual(loaded["completed_ids"], [grid[0]["experiment_id"]])
        self.assertEqual(loaded["result_path"], "results.jsonl")


class ExperimentStoreTests(unittest.TestCase):
    def test_store_writes_jsonl_without_overwriting_and_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.jsonl"
            store = ExperimentStore(jsonl_path=path, reports_dir=tmp)
            store.append_result({"experiment_id": "a", "classification": "reject"})
            store.append_result({"experiment_id": "b", "classification": "weak_candidate"})
            rows = store.load_all()

        self.assertEqual([row["experiment_id"] for row in rows], ["a", "b"])

    def test_get_candidates_and_markdown_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ExperimentStore(jsonl_path=Path(tmp) / "results.jsonl", reports_dir=tmp)
            store.append_result({"experiment_id": "a", "classification": "reject", "config": {}, "validation_metrics": {}, "test_metrics": {}, "baselines": {}})
            store.append_result({"experiment_id": "b", "classification": "candidate_for_further_validation", "config": {}, "validation_metrics": {}, "test_metrics": {}, "baselines": {}})
            markdown = store.generate_markdown_report()

            self.assertEqual(len(store.get_candidates()), 1)
            self.assertTrue(markdown.exists())
            self.assertIn("Research Autopilot Summary", markdown.read_text(encoding="utf-8"))


class ExperimentRunnerTests(unittest.IsolatedAsyncioTestCase):
    def test_purged_split_avoids_leakage(self):
        split = purged_train_validation_test_split(np.arange(250), horizon_candles=16, min_train_rows=80)

        self.assertGreater(len(split["validation"]), 0)
        self.assertGreater(len(split["test"]), 0)
        self.assertGreater(int(split["validation"][0]), int(split["train"][-1]) + 16)
        self.assertGreater(int(split["test"][0]), int(split["validation"][-1]) + 16)

    def test_test_is_not_used_for_initial_selection(self):
        validation = {"profit_factor": 1.2, "avg_return_pct": 0.2, "max_drawdown_pct": 5}
        test = {"profit_factor": 0.5, "avg_return_pct": -0.1, "max_drawdown_pct": 3}
        baselines = {
            "validation": {
                "random_same_count": {"avg_return_pct": -0.1},
                "deterministic": {"avg_return_pct": -0.05},
            }
        }

        classification, reasons = classify_result(validation, test, baselines)

        self.assertEqual(classification, "validation_candidate_test_failed")
        self.assertIn("test_holdout_failed_confirmation", reasons)

    def test_classification_reject_weak_candidate_and_candidate(self):
        reject, _ = classify_result(
            {"profit_factor": 0.9, "avg_return_pct": 0.1, "max_drawdown_pct": 1},
            {"profit_factor": 2, "avg_return_pct": 1, "max_drawdown_pct": 1},
            {"validation": {"random_same_count": {"avg_return_pct": -1}, "deterministic": {"avg_return_pct": -1}}},
        )
        weak, _ = classify_result(
            {"profit_factor": 1.05, "avg_return_pct": 0.1, "max_drawdown_pct": 1},
            {"profit_factor": 1, "avg_return_pct": 0.1, "max_drawdown_pct": 1},
            {"validation": {"random_same_count": {"avg_return_pct": -1}, "deterministic": {"avg_return_pct": 0.2}}},
        )
        candidate, _ = classify_result(
            {"profit_factor": 1.2, "avg_return_pct": 0.3, "max_drawdown_pct": 5},
            {"profit_factor": 1.1, "avg_return_pct": 0.1, "max_drawdown_pct": 5},
            {"validation": {"random_same_count": {"avg_return_pct": -1}, "deterministic": {"avg_return_pct": 0.1}}},
        )

        self.assertEqual(reject, "reject")
        self.assertEqual(weak, "weak_candidate")
        self.assertEqual(candidate, "candidate_for_further_validation")

    def test_random_baseline_uses_same_number_of_trades(self):
        returns = pd.DataFrame({
            "buy_return_pct": np.linspace(-1, 1, 20),
            "sell_return_pct": np.linspace(1, -1, 20),
        })
        baseline = _random_baseline(returns, np.arange(20), trade_count=7)

        self.assertEqual(baseline["n_trades"], 7)

    async def test_run_experiment_returns_validation_and_test_metrics(self):
        import research.experiment_runner as runner

        config = build_experiment_grid(min_train_rows=40, max_candles=320)[0]
        with patch.object(runner, "load_experiment_candles", new=AsyncMock(return_value={
            "candles": sample_candles(),
            "data_source": "mock",
            "data_cache_path": "",
            "data_warning": "",
        })):
            result = await run_experiment(config)

        self.assertEqual(result["experiment_id"], config["experiment_id"])
        self.assertIn("validation_metrics", result)
        self.assertIn("test_metrics", result)
        self.assertTrue(result["guardrails"]["test_not_used_for_selection"])
        self.assertIn(result["classification"], {
            "reject",
            "weak_candidate",
            "candidate_for_further_validation",
            "validation_candidate_test_failed",
        })


class AutopilotTests(unittest.IsolatedAsyncioTestCase):
    async def test_autopilot_resumes_from_checkpoint(self):
        import research.autopilot as autopilot

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "checkpoint.json"
            grid = build_experiment_grid()[:2]
            save_grid_checkpoint(checkpoint, grid, completed_ids={grid[0]["experiment_id"]}, result_path=str(Path(tmp) / "results.jsonl"))
            fake_result = {
                "experiment_id": grid[1]["experiment_id"],
                "classification": "reject",
                "config": grid[1],
                "validation_metrics": {},
                "test_metrics": {},
                "baselines": {},
            }
            with patch.object(autopilot, "run_experiment", new=AsyncMock(return_value=fake_result)) as run:
                result = await autopilot.run_autopilot(
                    resume=True,
                    max_experiments=1,
                    reports_dir=tmp,
                    checkpoint_path=checkpoint,
                )

            self.assertEqual(run.call_count, 1)
            self.assertEqual(result["completed"], 2)

    async def test_keyboard_interrupt_leaves_checkpoint(self):
        import research.autopilot as autopilot

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "checkpoint.json"
            with patch.object(autopilot, "run_experiment", new=AsyncMock(side_effect=KeyboardInterrupt)):
                result = await autopilot.run_autopilot(
                    resume=False,
                    max_experiments=1,
                    reports_dir=tmp,
                    checkpoint_path=checkpoint,
                )

            self.assertTrue(result["interrupted"])
            self.assertTrue(checkpoint.exists())
            loaded = json.loads(checkpoint.read_text(encoding="utf-8"))
            self.assertIn("completed_ids", loaded)


if __name__ == "__main__":
    unittest.main()
