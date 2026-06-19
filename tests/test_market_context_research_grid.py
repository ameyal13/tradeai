import unittest

from research.market_context_research_grid import (
    MARKET_CONTEXT_ATR_STOP_MULTIPLIERS,
    MARKET_CONTEXT_HORIZON_CANDLES,
    MARKET_CONTEXT_RISK_REWARDS,
    build_market_context_research_grid,
)
from scripts.run_market_context_research import build_parser


class MarketContextResearchGridTests(unittest.TestCase):
    def test_grid_generates_expected_16_configs(self):
        grid = build_market_context_research_grid()

        self.assertEqual(len(grid), 16)
        self.assertEqual({row["symbol"] for row in grid}, {"ADA", "ETH"})
        self.assertEqual({row["timeframe"] for row in grid}, {"1h"})
        self.assertEqual({row["cost_mode"] for row in grid}, {"low_costs"})
        self.assertEqual({row["strategy_mode"] for row in grid}, {"xgboost"})
        self.assertEqual({row["horizon_candles"] for row in grid}, set(MARKET_CONTEXT_HORIZON_CANDLES))
        self.assertEqual({row["risk_reward"] for row in grid}, set(MARKET_CONTEXT_RISK_REWARDS))
        self.assertEqual({row["atr_stop_multiplier"] for row in grid}, set(MARKET_CONTEXT_ATR_STOP_MULTIPLIERS))
        self.assertTrue(all(row["use_market_context_features"] for row in grid))
        self.assertEqual({row["feature_family"] for row in grid}, {"current_plus_market_context_v1"})

    def test_symbol_filter(self):
        grid = build_market_context_research_grid(symbols=["ADA"])

        self.assertEqual(len(grid), 8)
        self.assertEqual({row["symbol"] for row in grid}, {"ADA"})

    def test_invalid_symbol_rejected(self):
        with self.assertRaises(ValueError):
            build_market_context_research_grid(symbols=["BTC"])

    def test_cli_defaults(self):
        args = build_parser().parse_args([])

        self.assertTrue(args.once)
        self.assertTrue(args.resume)
        self.assertEqual(args.max_configs_per_cycle, 5)
        self.assertFalse(args.notify_telegram)


if __name__ == "__main__":
    unittest.main()
