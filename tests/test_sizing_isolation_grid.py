import unittest

from research.sizing_isolation_grid import (
    SIZING_ATR_STOP_MULTIPLIERS,
    SIZING_COST_MODES,
    SIZING_FEATURE_FAMILY,
    SIZING_HORIZON_CANDLES,
    SIZING_PHASE,
    SIZING_RISK_REWARDS,
    SIZING_SYMBOLS,
    build_sizing_isolation_grid,
)
from scripts.run_sizing_isolation_research import (
    SIZING_CYCLES_DIR,
    SIZING_REGISTRY_PATH,
    SIZING_RESULTS_DIR,
    SIZING_STATUS_PATH,
    build_parser,
    parse_horizons,
)


class SizingIsolationGridTests(unittest.TestCase):
    def test_grid_generates_160_configs_by_default(self):
        grid = build_sizing_isolation_grid()

        self.assertEqual(len(grid), 160)
        self.assertEqual({row["symbol"] for row in grid}, set(SIZING_SYMBOLS))

    def test_grid_only_varies_sizing_fields(self):
        grid = build_sizing_isolation_grid()

        self.assertEqual({row["timeframe"] for row in grid}, {"1h"})
        self.assertEqual({row["strategy_mode"] for row in grid}, {"xgboost"})
        self.assertEqual({row["cost_mode"] for row in grid}, set(SIZING_COST_MODES))
        self.assertEqual({row["horizon_candles"] for row in grid}, set(SIZING_HORIZON_CANDLES))
        self.assertEqual({row["risk_reward"] for row in grid}, set(SIZING_RISK_REWARDS))
        self.assertEqual({row["atr_stop_multiplier"] for row in grid}, set(SIZING_ATR_STOP_MULTIPLIERS))
        self.assertEqual({row["feature_family"] for row in grid}, {SIZING_FEATURE_FAMILY})
        self.assertEqual({row["use_market_context_features"] for row in grid}, {False})
        self.assertTrue(all(row["research_phase"] == SIZING_PHASE for row in grid))

    def test_grid_filters_symbols_and_horizons(self):
        grid = build_sizing_isolation_grid(symbols=["ADA"], horizons=[10, 12, 14])

        self.assertEqual(len(grid), 48)
        self.assertEqual({row["symbol"] for row in grid}, {"ADA"})
        self.assertEqual({row["horizon_candles"] for row in grid}, {10, 12, 14})
        with self.assertRaises(ValueError):
            build_sizing_isolation_grid(symbols=["BTC"])
        with self.assertRaises(ValueError):
            build_sizing_isolation_grid(horizons=[9])

    def test_config_ids_are_stable_and_unique(self):
        first = build_sizing_isolation_grid()
        second = build_sizing_isolation_grid()

        self.assertEqual([row["config_id"] for row in first], [row["config_id"] for row in second])
        self.assertEqual(len({row["config_id"] for row in first}), len(first))

    def test_cli_defaults_and_paths(self):
        args = build_parser().parse_args([])

        self.assertTrue(args.once)
        self.assertTrue(args.resume)
        self.assertEqual(args.max_configs_per_cycle, 5)
        self.assertFalse(args.quick_ada)
        self.assertIn("sizing_isolation_v1_registry.jsonl", str(SIZING_REGISTRY_PATH))
        self.assertIn("sizing_isolation_v1_cycles", str(SIZING_CYCLES_DIR))
        self.assertIn("sizing_isolation_v1_results", str(SIZING_RESULTS_DIR))
        self.assertIn("sizing_isolation_v1_current_status.json", str(SIZING_STATUS_PATH))

    def test_parse_horizons(self):
        self.assertEqual(parse_horizons(["10", "12,14"]), [10, 12, 14])
        self.assertIsNone(parse_horizons(None))


if __name__ == "__main__":
    unittest.main()
