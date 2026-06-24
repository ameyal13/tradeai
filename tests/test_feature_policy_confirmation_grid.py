import unittest

from research.feature_policy_confirmation_grid import (
    FEATURE_POLICY_FEATURE_SETS_BY_SYMBOL,
    build_feature_policy_confirmation_grid,
)
from scripts.summarize_feature_policy_confirmation_research import build_summary


class FeaturePolicyConfirmationGridTests(unittest.TestCase):
    def test_default_grid_generates_10_configs(self):
        grid = build_feature_policy_confirmation_grid()

        self.assertEqual(len(grid), 10)
        self.assertEqual({row["symbol"] for row in grid}, {"ADA", "ETH", "SOL"})
        self.assertEqual({row["timeframe"] for row in grid}, {"1h"})
        self.assertEqual({row["horizon_candles"] for row in grid}, {10})
        self.assertEqual({row["risk_reward"] for row in grid}, {2.0})
        self.assertEqual({row["atr_stop_multiplier"] for row in grid}, {1.5})
        self.assertEqual({row["cost_mode"] for row in grid}, {"low_costs"})
        self.assertEqual({row["strategy_mode"] for row in grid}, {"xgboost"})

    def test_feature_sets_are_symbol_specific(self):
        grid = build_feature_policy_confirmation_grid()
        by_symbol = {}
        for row in grid:
            by_symbol.setdefault(row["symbol"], set()).add(row["feature_set"])

        self.assertEqual(by_symbol["ADA"], set(FEATURE_POLICY_FEATURE_SETS_BY_SYMBOL["ADA"]))
        self.assertEqual(by_symbol["ETH"], set(FEATURE_POLICY_FEATURE_SETS_BY_SYMBOL["ETH"]))
        self.assertEqual(by_symbol["SOL"], set(FEATURE_POLICY_FEATURE_SETS_BY_SYMBOL["SOL"]))
        self.assertNotIn("4h_only", by_symbol["ADA"])
        self.assertNotIn("4h_only", by_symbol["ETH"])
        self.assertIn("4h_only", by_symbol["SOL"])

    def test_accepts_usdt_alias_and_rejects_unsupported(self):
        grid = build_feature_policy_confirmation_grid(symbols=["ADAUSDT"])

        self.assertEqual(len(grid), 3)
        self.assertEqual({row["symbol"] for row in grid}, {"ADA"})
        with self.assertRaises(ValueError):
            build_feature_policy_confirmation_grid(symbols=["BTC"])

    def test_config_ids_are_stable_and_unique(self):
        first = build_feature_policy_confirmation_grid()
        second = build_feature_policy_confirmation_grid()

        self.assertEqual([row["config_id"] for row in first], [row["config_id"] for row in second])
        self.assertEqual(len({row["config_id"] for row in first}), len(first))


class FeaturePolicySummaryTests(unittest.TestCase):
    def _record(self, symbol, feature_set, pf, avg=0.0):
        return {
            "config_id": f"{symbol}-{feature_set}",
            "status": "completed",
            "classification": "feature_expansion_watchlist",
            "config": {"symbol": symbol, "feature_set": feature_set},
            "aggregate": {
                "median_validation_pf": pf,
                "median_validation_avg_return": avg,
                "beats_time_only_rate": 0.5,
                "beats_dummy_random_rate": 0.5,
                "valid_windows": 19,
            },
            "json_loaded": True,
            "json_missing": False,
        }

    def test_control_comparison_requires_beating_baseline_and_time_only(self):
        records = [
            self._record("ADA", "baseline", 0.9),
            self._record("ADA", "time_only", 0.8),
            self._record("ADA", "baseline_plus_btc_context", 1.1),
            self._record("SOL", "baseline", 0.9),
            self._record("SOL", "time_only", 0.95),
            self._record("SOL", "4h_only", 0.92),
            self._record("SOL", "baseline_plus_btc_context", 0.91),
        ]

        summary = build_summary(records)

        self.assertTrue(summary["control_comparison"]["ADA"]["beats_controls_pf"])
        self.assertEqual(
            summary["control_comparison"]["ADA"]["recommendation"],
            "candidate_feature_policy_warrants_next_validation",
        )
        self.assertFalse(summary["control_comparison"]["SOL"]["beats_controls_pf"])
        self.assertEqual(summary["control_comparison"]["SOL"]["recommendation"], "mixed_control_result")


if __name__ == "__main__":
    unittest.main()
