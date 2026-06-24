import unittest

from research.feature_expansion_grid import (
    FEATURE_EXPANSION_ATR_STOP_MULTIPLIERS,
    FEATURE_EXPANSION_FEATURE_SETS,
    FEATURE_EXPANSION_RISK_REWARDS,
    FEATURE_EXPANSION_SYMBOLS,
    build_feature_expansion_grid,
)
from scripts.run_feature_expansion_research import build_parser


class FeatureExpansionGridTests(unittest.TestCase):
    def test_grid_generates_18_configs_by_default(self):
        grid = build_feature_expansion_grid()

        self.assertEqual(len(grid), 18)
        self.assertEqual({row["symbol"] for row in grid}, set(FEATURE_EXPANSION_SYMBOLS))
        self.assertEqual({row["feature_set"] for row in grid}, set(FEATURE_EXPANSION_FEATURE_SETS))

    def test_grid_freezes_sizing_and_costs(self):
        grid = build_feature_expansion_grid()

        self.assertEqual({row["timeframe"] for row in grid}, {"1h"})
        self.assertEqual({row["horizon_candles"] for row in grid}, {10})
        self.assertEqual({row["risk_reward"] for row in grid}, set(FEATURE_EXPANSION_RISK_REWARDS))
        self.assertEqual({row["atr_stop_multiplier"] for row in grid}, set(FEATURE_EXPANSION_ATR_STOP_MULTIPLIERS))
        self.assertEqual({row["cost_mode"] for row in grid}, {"low_costs"})
        self.assertEqual({row["strategy_mode"] for row in grid}, {"xgboost"})

    def test_accepts_usdt_symbol_alias_and_rejects_unsupported(self):
        grid = build_feature_expansion_grid(symbols=["ADAUSDT"], feature_sets=["time_only"])

        self.assertEqual(len(grid), 1)
        self.assertEqual(grid[0]["symbol"], "ADA")
        with self.assertRaises(ValueError):
            build_feature_expansion_grid(symbols=["BTC"])

    def test_config_ids_are_stable_and_unique(self):
        first = build_feature_expansion_grid()
        second = build_feature_expansion_grid()

        self.assertEqual([row["config_id"] for row in first], [row["config_id"] for row in second])
        self.assertEqual(len({row["config_id"] for row in first}), len(first))

    def test_cli_defaults(self):
        args = build_parser().parse_args([])

        self.assertTrue(args.once)
        self.assertTrue(args.resume)
        self.assertEqual(args.max_configs_per_cycle, 5)
        self.assertFalse(args.dry_run)

    def test_cli_accepts_dry_run(self):
        args = build_parser().parse_args(["--dry-run"])

        self.assertTrue(args.dry_run)


if __name__ == "__main__":
    unittest.main()
