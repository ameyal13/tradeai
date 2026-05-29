import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np
import pandas as pd

from tools.feature_research import (
    FEATURE_COLS,
    add_research_features,
    build_directional_net_returns,
    feature_correlations_to_future_return,
    purged_walk_forward_splits,
    run_feature_audit,
)


def sample_candles(rows=260):
    idx = pd.date_range("2026-01-01", periods=rows, freq="15min", tz="UTC")
    base = 100 + np.linspace(0, 8, rows) + np.sin(np.arange(rows) / 5)
    return pd.DataFrame({
        "timestamp": idx,
        "open": base,
        "high": base + 1.2,
        "low": base - 1.2,
        "close": base + np.sin(np.arange(rows) / 7) * 0.3,
        "volume": 1000 + np.arange(rows) * 2,
    })


class FeatureResearchTests(unittest.TestCase):
    def test_add_research_features_keeps_current_and_adds_time_regime(self):
        features = add_research_features(sample_candles(120))

        for column in FEATURE_COLS:
            self.assertIn(column, features.columns)
        for column in ["hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos", "atr_pct", "ema_distance_pct"]:
            self.assertIn(column, features.columns)

    def test_purged_walk_forward_splits_embargo_label_horizon(self):
        positions = np.arange(40, 180)
        folds = purged_walk_forward_splits(positions, n_splits=3, min_train_rows=40, horizon_candles=4)

        self.assertGreater(len(folds), 0)
        for fold in folds:
            self.assertLessEqual(int(fold.train_positions.max()) + 4, int(fold.validation_positions.min()) - 1)

    def test_directional_returns_use_delayed_entry_and_tp_sl_path(self):
        idx = pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC")
        df = pd.DataFrame({
            "timestamp": idx,
            "open": [100, 100, 100, 100],
            "high": [100, 103, 100, 100],
            "low": [100, 100, 100, 100],
            "close": [100, 100, 100, 100],
            "volume": [1000] * 4,
        })
        returns = build_directional_net_returns(
            df,
            horizon_candles=1,
            stop_loss_pct=0.01,
            take_profit_pct=0.02,
            commission_pct=0,
            slippage_pct=0,
            spread_pct=0,
        )

        self.assertEqual(returns.iloc[0]["buy_return_pct"], 2.0)
        self.assertEqual(returns.iloc[0]["sell_return_pct"], -1.0)

    def test_feature_correlations_returns_current_feature_keys(self):
        features = add_research_features(sample_candles(120))
        correlations = feature_correlations_to_future_return(features, FEATURE_COLS, horizon_candles=4)

        self.assertEqual(set(correlations), set(FEATURE_COLS))

    def test_run_feature_audit_returns_ablation_and_candidates(self):
        audit = run_feature_audit(
            sample_candles(260),
            horizon_candles=4,
            n_splits=2,
            min_train_rows=40,
            trade_label_scheme="expected_value_classification",
        )

        self.assertIn("all_current", audit["ablation_results"])
        self.assertIn("dummy_random", audit["ablation_results"])
        self.assertIn("feature_correlations_to_future_return", audit)
        self.assertIn("removal_candidates", audit)
        self.assertEqual(audit["methodology"]["split"], "purged_walk_forward")


class FeatureAuditScriptTests(unittest.IsolatedAsyncioTestCase):
    async def test_script_writes_markdown_and_json_without_keys(self):
        import scripts.audit_xgboost_features as script

        with tempfile.TemporaryDirectory() as tmp:
            args = Namespace(
                symbols=["BTC"],
                timeframes=["15m"],
                max_candles=260,
                max_predictions=20,
                horizon_minutes=60,
                splits=2,
                min_train_rows=40,
                trade_label_scheme="expected_value_classification",
                reports_dir=tmp,
                cache_dir=Path(tmp) / "cache",
                use_cache=False,
                refresh_cache=False,
            )
            with patch.object(script, "load_experiment_candles", new=AsyncMock(return_value={
                "candles": sample_candles(260),
                "data_source": "mock",
                "data_cache_path": "",
                "data_warning": "",
            })):
                report = await script.run_audit(args)

            self.assertEqual(len(report["results"]), 1)
            self.assertTrue(Path(report["report_paths"]["markdown"]).exists())
            self.assertTrue(Path(report["report_paths"]["json"]).exists())
            self.assertIn(report["results"][0]["classification"], {
                "signal_candidate",
                "weak_signal_candidate",
                "no_feature_edge_detected",
                "insufficient_data",
                "insufficient_trades",
            })


if __name__ == "__main__":
    unittest.main()
