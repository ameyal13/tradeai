import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np
import pandas as pd

from tools.trade_opportunity_research import (
    break_even_win_rate_pct,
    classify_opportunity,
    cost_formula_diagnostics,
    evaluate_opportunity_config,
    profit_metrics,
    random_same_count_returns,
    run_trade_opportunity_audit,
)
from tools.feature_research import add_research_features


def sample_candles(rows=220):
    idx = pd.date_range("2026-01-01", periods=rows, freq="15min", tz="UTC")
    base = 100 + np.linspace(0, 10, rows) + np.sin(np.arange(rows) / 4)
    return pd.DataFrame({
        "timestamp": idx,
        "open": base,
        "high": base + 1.5,
        "low": base - 1.5,
        "close": base + np.sin(np.arange(rows) / 6) * 0.4,
        "volume": 1000 + np.arange(rows),
    })


class TradeOpportunityResearchTests(unittest.TestCase):
    def test_break_even_win_rate_from_average_win_loss(self):
        self.assertEqual(break_even_win_rate_pct([2.0, 2.0, -1.0, -1.0]), 33.333333)

    def test_profit_metrics_calculates_pf_and_average_return(self):
        metrics = profit_metrics([1.0, -0.5, 2.0])

        self.assertEqual(metrics["trades"], 3)
        self.assertEqual(metrics["profit_factor"], 6.0)
        self.assertEqual(metrics["win_rate"], 66.666667)

    def test_random_same_count_uses_requested_trade_count(self):
        buy = np.array([1.0, -1.0, 0.5, -0.2, 0.3])
        sell = np.array([-0.5, 0.2, -0.1, 0.4, -0.3])
        metrics = random_same_count_returns(buy, sell, trade_count=3, simulations=5)

        self.assertEqual(metrics["trades"], 3)
        self.assertEqual(metrics["simulations"], 5)

    def test_evaluate_opportunity_config_outputs_rates_and_baselines(self):
        features = add_research_features(sample_candles())
        row = evaluate_opportunity_config(
            features,
            horizon_candles=4,
            risk_reward=1.5,
            atr_stop_multiplier=1.0,
            cost_name="current_costs",
            costs={"commission_pct": 0.001, "slippage_pct": 0.0005, "spread_pct": 0.0003},
            top_k=20,
            random_trade_count=20,
        )

        for key in [
            "buy_positive_rate",
            "sell_positive_rate",
            "buy_break_even_win_rate",
            "sell_break_even_win_rate",
            "always_buy",
            "always_sell",
            "oracle_positive",
            "oracle_top_k",
            "random_same_count",
            "classification",
        ]:
            self.assertIn(key, row)
        self.assertEqual(row["oracle_top_k"]["trades"], 20)
        self.assertEqual(row["random_same_count"]["trades"], 20)

    def test_classify_opportunity_detects_oracle_only(self):
        row = {
            "buy_positive_rate": 20,
            "sell_positive_rate": 15,
            "buy_break_even_win_rate": 40,
            "sell_break_even_win_rate": 40,
            "oracle_top_k": {"trades": 50, "average_return": 1.0, "profit_factor": 2.0},
            "random_same_count": {"average_return": -0.2},
        }

        self.assertEqual(classify_opportunity(row), "oracle_only_opportunity")

    def test_run_trade_opportunity_audit_returns_sensitivity_grid(self):
        audit = run_trade_opportunity_audit(
            sample_candles(),
            horizon_values=[4, 8],
            risk_rewards=[1.0, 1.5],
            atr_stop_multipliers=[1.0],
            top_k=10,
            random_trade_count=10,
        )

        self.assertEqual(len(audit["rows"]), 16)
        self.assertIn("classification_counts", audit)
        self.assertIn("current_setup", audit)
        self.assertIn("cost_sensitivity", audit)
        self.assertIn("realistic_candidate_setups", audit)
        self.assertIn("cost_formula_diagnostics", audit)

    def test_cost_formula_diagnostics_documents_current_tier(self):
        diagnostics = cost_formula_diagnostics(entry=100, exit_price=101.5)

        self.assertIn("net_return_pct", diagnostics["formula"])
        self.assertEqual(diagnostics["current_profile"]["commission_pct"], 0.001)
        self.assertIn("low_costs", diagnostics["recommended_research_profiles"])
        self.assertIn("medium_costs_current", diagnostics["recommended_research_profiles"])


class TradeOpportunityScriptTests(unittest.IsolatedAsyncioTestCase):
    async def test_script_writes_markdown_and_json(self):
        import scripts.audit_trade_opportunity as script

        with tempfile.TemporaryDirectory() as tmp:
            args = Namespace(
                symbols=["BTC"],
                timeframes=["15m"],
                max_candles=220,
                max_predictions=20,
                horizon_minutes=60,
                horizons="4",
                risk_rewards="1.5",
                atr_stop_multipliers="1.5",
                top_k=10,
                random_trade_count=10,
                reports_dir=tmp,
                cache_dir=Path(tmp) / "cache",
                use_cache=False,
                refresh_cache=False,
            )
            with patch.object(script, "load_experiment_candles", new=AsyncMock(return_value={
                "candles": sample_candles(),
                "data_source": "mock",
                "data_cache_path": "",
                "data_warning": "",
            })):
                report = await script.run_audit(args)

            self.assertEqual(len(report["results"]), 1)
            self.assertTrue(Path(report["report_paths"]["markdown"]).exists())
            self.assertTrue(Path(report["report_paths"]["json"]).exists())
            self.assertIn("current_setup", report["results"][0]["audit"])
            self.assertIn("cost_formula_diagnostics", report)


if __name__ == "__main__":
    unittest.main()
