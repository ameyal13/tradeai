import unittest

import pandas as pd

from tools.backtest_tool import (
    BacktestConfig,
    calculate_max_drawdown,
    calculate_metrics,
    calculate_position_size,
    estimate_slippage_cost,
    simulate_backtest,
)


def candles(rows):
    return pd.DataFrame(rows, index=pd.to_datetime([row["timestamp"] for row in rows])).drop(columns=["timestamp"])


class BacktestEngineV2Tests(unittest.TestCase):
    def base_strategy(self, **overrides):
        strategy = {
            "side": "LONG",
            "entry_conditions": [{"indicator": "close", "operator": ">", "value": 100}],
            "stop_loss_pct": 0.05,
            "take_profit_pct": 0.10,
            "commission_pct": 0,
            "slippage_pct": 0,
            "spread_pct": 0,
            "risk_per_trade_pct": 0.10,
        }
        strategy.update(overrides)
        return strategy

    def test_no_lookahead_bias_entry_uses_next_candle_open(self):
        df = candles([
            {"timestamp": "2026-01-01", "open": 90, "high": 95, "low": 85, "close": 90},
            {"timestamp": "2026-01-02", "open": 90, "high": 130, "low": 89, "close": 110},
            {"timestamp": "2026-01-03", "open": 120, "high": 121, "low": 119, "close": 120},
        ])

        result = simulate_backtest(df, self.base_strategy(take_profit_pct=0.50), initial_capital=1000)

        self.assertEqual(result["number_of_trades"], 1)
        self.assertEqual(result["trades"][0]["entry_time"], "2026-01-03T00:00:00")
        self.assertEqual(result["trades"][0]["entry_price"], 120)

    def test_long_stop_loss_uses_low_not_close(self):
        df = candles([
            {"timestamp": "2026-01-01", "open": 100, "high": 105, "low": 99, "close": 101},
            {"timestamp": "2026-01-02", "open": 100, "high": 104, "low": 94, "close": 103},
        ])

        result = simulate_backtest(df, self.base_strategy(take_profit_pct=0.20), initial_capital=1000)

        self.assertEqual(result["trades"][0]["exit_reason"], "STOP_LOSS")
        self.assertEqual(result["trades"][0]["exit_price"], 95)

    def test_long_take_profit_uses_high_not_close(self):
        df = candles([
            {"timestamp": "2026-01-01", "open": 100, "high": 105, "low": 99, "close": 101},
            {"timestamp": "2026-01-02", "open": 100, "high": 111, "low": 98, "close": 101},
        ])

        result = simulate_backtest(df, self.base_strategy(), initial_capital=1000)

        self.assertEqual(result["trades"][0]["exit_reason"], "TAKE_PROFIT")
        self.assertEqual(result["trades"][0]["exit_price"], 110)

    def test_short_stop_and_take_profit_use_high_low(self):
        short_strategy = self.base_strategy(
            side="SHORT",
            entry_conditions=[{"indicator": "close", "operator": "<", "value": 100}],
            stop_loss_pct=0.05,
            take_profit_pct=0.10,
        )
        stop_df = candles([
            {"timestamp": "2026-01-01", "open": 100, "high": 101, "low": 95, "close": 99},
            {"timestamp": "2026-01-02", "open": 100, "high": 106, "low": 98, "close": 99},
        ])
        take_df = candles([
            {"timestamp": "2026-01-01", "open": 100, "high": 101, "low": 95, "close": 99},
            {"timestamp": "2026-01-02", "open": 100, "high": 102, "low": 89, "close": 99},
        ])

        stop_result = simulate_backtest(stop_df, short_strategy, initial_capital=1000)
        take_result = simulate_backtest(take_df, short_strategy, initial_capital=1000)

        self.assertEqual(stop_result["trades"][0]["exit_reason"], "STOP_LOSS")
        self.assertEqual(stop_result["trades"][0]["exit_price"], 105)
        self.assertEqual(take_result["trades"][0]["exit_reason"], "TAKE_PROFIT")
        self.assertEqual(take_result["trades"][0]["exit_price"], 90)

    def test_commission_calculation(self):
        df = candles([
            {"timestamp": "2026-01-01", "open": 100, "high": 105, "low": 99, "close": 101},
            {"timestamp": "2026-01-02", "open": 100, "high": 111, "low": 98, "close": 101},
        ])
        strategy = self.base_strategy(commission_pct=0.001)

        result = simulate_backtest(df, strategy, initial_capital=1000)

        self.assertAlmostEqual(result["trades"][0]["fees"], 2.1, places=6)
        self.assertAlmostEqual(result["fees_total"], 2.1, places=6)

    def test_slippage_calculation(self):
        config = BacktestConfig(slippage_pct=0.0005)

        self.assertAlmostEqual(estimate_slippage_cost(2, 100, config), 0.1)

    def test_position_sizing_by_risk(self):
        config = BacktestConfig(risk_per_trade_pct=0.01)

        quantity = calculate_position_size(1000, 100, 95, config)

        self.assertAlmostEqual(quantity, 2)

    def test_max_drawdown_calculation(self):
        self.assertAlmostEqual(calculate_max_drawdown([1000, 1100, 900, 1200]), 18.181818, places=6)

    def test_profit_factor_calculation(self):
        df = candles([
            {"timestamp": "2026-01-01", "open": 100, "high": 100, "low": 100, "close": 100},
            {"timestamp": "2026-01-02", "open": 100, "high": 100, "low": 100, "close": 100},
        ])
        metrics = calculate_metrics(
            trades=[
                {"pnl": 20, "pnl_pct": 2, "fees": 0, "slippage_cost": 0},
                {"pnl": -10, "pnl_pct": -1, "fees": 0, "slippage_cost": 0},
            ],
            equity_curve=[{"capital": 1000}, {"capital": 1010}],
            initial_capital=1000,
            df=df,
        )

        self.assertEqual(metrics["profit_factor"], 2)


if __name__ == "__main__":
    unittest.main()
