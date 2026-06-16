import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from research.focused_research_grid import (
    FOCUSED_ATR_STOP_MULTIPLIERS,
    FOCUSED_COST_MODES,
    FOCUSED_HORIZON_CANDLES,
    FOCUSED_PHASE,
    FOCUSED_RISK_REWARDS,
    FOCUSED_SYMBOLS,
    build_focused_research_grid,
)
from research.research_daemon import run_research_cycle
from scripts.run_focused_research import (
    FOCUSED_CYCLES_DIR,
    FOCUSED_REGISTRY_PATH,
    FOCUSED_RESULTS_DIR,
    FOCUSED_STATUS_PATH,
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


class FocusedResearchGridTests(unittest.IsolatedAsyncioTestCase):
    def test_grid_generates_81_configs_by_default(self):
        grid = build_focused_research_grid()

        self.assertEqual(len(grid), 81)
        self.assertEqual({row["symbol"] for row in grid}, set(FOCUSED_SYMBOLS))

    def test_grid_only_expected_values(self):
        grid = build_focused_research_grid()

        self.assertEqual({row["timeframe"] for row in grid}, {"1h"})
        self.assertEqual({row["strategy_mode"] for row in grid}, {"xgboost"})
        self.assertEqual({row["cost_mode"] for row in grid}, set(FOCUSED_COST_MODES))
        self.assertEqual({row["horizon_candles"] for row in grid}, set(FOCUSED_HORIZON_CANDLES))
        self.assertEqual({row["risk_reward"] for row in grid}, set(FOCUSED_RISK_REWARDS))
        self.assertEqual({row["atr_stop_multiplier"] for row in grid}, set(FOCUSED_ATR_STOP_MULTIPLIERS))
        self.assertTrue(all(row["research_phase"] == FOCUSED_PHASE for row in grid))

    def test_grid_filters_symbols_and_rejects_outside_watchlist(self):
        grid = build_focused_research_grid(symbols=["ADA", "SOL"])

        self.assertEqual(len(grid), 54)
        self.assertEqual({row["symbol"] for row in grid}, {"ADA", "SOL"})
        with self.assertRaises(ValueError):
            build_focused_research_grid(symbols=["BTC"])

    def test_cli_defaults_and_separate_paths(self):
        args = build_parser().parse_args([])

        self.assertTrue(args.once)
        self.assertEqual(args.max_configs_per_cycle, 5)
        self.assertIn("focused_v2a_registry.jsonl", str(FOCUSED_REGISTRY_PATH))
        self.assertIn("focused_v2a_cycles", str(FOCUSED_CYCLES_DIR))
        self.assertIn("focused_v2a_results", str(FOCUSED_RESULTS_DIR))
        self.assertIn("focused_v2a_current_status.json", str(FOCUSED_STATUS_PATH))

    async def test_run_research_cycle_accepts_focused_grid(self):
        import research.research_daemon as daemon

        grid = build_focused_research_grid(symbols=["ADA"])[:2]
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = Path(tmp) / "focused_registry.jsonl"
            with patch.object(daemon, "run_multi_window_validation_for_setups", new=AsyncMock(return_value=fake_multi_window_result())) as run:
                result = await run_research_cycle(
                    grid=grid,
                    max_configs_per_cycle=2,
                    registry_path=registry_path,
                    cycles_dir=Path(tmp) / "cycles",
                    results_dir=Path(tmp) / "results",
                    status_path=Path(tmp) / "status.json",
                )

        self.assertEqual(result["grid_size"], 2)
        self.assertEqual(run.call_count, 2)


if __name__ == "__main__":
    unittest.main()
