import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from research.refined_grid import (
    REFINED_ATR_STOP_MULTIPLIERS,
    REFINED_HORIZON_CANDLES,
    REFINED_RISK_REWARDS,
    build_refined_sol_1h_grid,
)
from research.research_daemon import build_daemon_grid, run_research_cycle
from scripts.run_refined_research import (
    REFINED_CYCLES_DIR,
    REFINED_REGISTRY_PATH,
    REFINED_RESULTS_DIR,
    REFINED_STATUS_PATH,
    build_parser,
)


def fake_multi_window_result():
    return {
        "setups": [{
            "classification": "multi_window_reject",
            "aggregate": {
                "total_windows": 4,
                "valid_windows": 4,
                "median_validation_pf": 0.8,
                "median_validation_avg_return": -0.1,
                "median_test_pf": 1.0,
                "test_confirm_rate": 0.2,
            },
            "windows": [],
        }],
        "json_path": "result.json",
        "markdown_path": "result.md",
    }


class RefinedGridTests(unittest.IsolatedAsyncioTestCase):
    def test_refined_grid_generates_exactly_48_configs(self):
        grid = build_refined_sol_1h_grid()

        self.assertEqual(len(grid), 48)

    def test_refined_grid_only_sol_1h_xgboost_low_costs(self):
        grid = build_refined_sol_1h_grid()

        self.assertEqual({row["symbol"] for row in grid}, {"SOL"})
        self.assertEqual({row["timeframe"] for row in grid}, {"1h"})
        self.assertEqual({row["strategy_mode"] for row in grid}, {"xgboost"})
        self.assertEqual({row["cost_mode"] for row in grid}, {"low_costs"})

    def test_refined_grid_excludes_rr_1_5_and_medium_costs(self):
        grid = build_refined_sol_1h_grid()

        self.assertNotIn(1.5, {row["risk_reward"] for row in grid})
        self.assertNotIn("medium_costs_current", {row["cost_mode"] for row in grid})

    def test_refined_grid_has_expected_horizon_rr_and_atr_values(self):
        grid = build_refined_sol_1h_grid()

        self.assertEqual({row["horizon_candles"] for row in grid}, set(REFINED_HORIZON_CANDLES))
        self.assertEqual({row["risk_reward"] for row in grid}, set(REFINED_RISK_REWARDS))
        self.assertEqual({row["atr_stop_multiplier"] for row in grid}, set(REFINED_ATR_STOP_MULTIPLIERS))

    def test_refined_grid_config_ids_are_stable_and_unique(self):
        first = build_refined_sol_1h_grid()
        second = build_refined_sol_1h_grid()

        self.assertEqual([row["config_id"] for row in first], [row["config_id"] for row in second])
        self.assertEqual(len({row["config_id"] for row in first}), 48)

    async def test_run_research_cycle_uses_external_grid(self):
        import research.research_daemon as daemon

        grid = build_refined_sol_1h_grid()[:2]
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(daemon, "run_multi_window_validation_for_setups", new=AsyncMock(return_value=fake_multi_window_result())) as run:
                result = await run_research_cycle(
                    grid=grid,
                    max_configs_per_cycle=2,
                    registry_path=Path(tmp) / "refined_registry.jsonl",
                    cycles_dir=Path(tmp) / "refined_cycles",
                    results_dir=Path(tmp) / "refined_results",
                    status_path=Path(tmp) / "refined_current_status.json",
                )

        self.assertEqual(result["grid_size"], 2)
        self.assertEqual(run.call_count, 2)
        self.assertEqual(run.call_args.kwargs["setups"][0]["symbol"], "SOL")

    async def test_run_research_cycle_grid_none_still_uses_general_grid(self):
        import research.research_daemon as daemon

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(daemon, "run_multi_window_validation_for_setups", new=AsyncMock(return_value=fake_multi_window_result())):
                result = await run_research_cycle(
                    grid=None,
                    max_configs_per_cycle=1,
                    registry_path=Path(tmp) / "registry.jsonl",
                    cycles_dir=Path(tmp) / "cycles",
                    results_dir=Path(tmp) / "results",
                    status_path=Path(tmp) / "current_status.json",
                )

        self.assertEqual(result["grid_size"], len(build_daemon_grid()))
        self.assertEqual(result["grid_size"], 96)

    def test_refined_script_uses_separate_paths(self):
        self.assertIn("refined_registry.jsonl", str(REFINED_REGISTRY_PATH))
        self.assertIn("refined_cycles", str(REFINED_CYCLES_DIR))
        self.assertIn("refined_results", str(REFINED_RESULTS_DIR))
        self.assertIn("refined_current_status.json", str(REFINED_STATUS_PATH))

    def test_refined_cli_defaults(self):
        args = build_parser().parse_args([])

        self.assertTrue(args.once)
        self.assertEqual(args.max_configs_per_cycle, 5)
        self.assertTrue(args.resume)
        self.assertFalse(args.retry_failed)
        self.assertFalse(args.notify_telegram)
        self.assertTrue(args.progress)
        self.assertFalse(args.quiet)

    def test_no_test_selection_guardrail_remains_in_daemon_result(self):
        grid = build_refined_sol_1h_grid()[:1]
        self.assertTrue(all(row["strategy_mode"] == "xgboost" for row in grid))
        self.assertTrue(all(row["cost_mode"] == "low_costs" for row in grid))


if __name__ == "__main__":
    unittest.main()
