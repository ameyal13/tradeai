import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np
import pandas as pd

import scripts.run_feature_audit as script


def sample_candles(rows=80):
    idx = pd.date_range("2026-01-01", periods=rows, freq="h", tz="UTC")
    base = 100 + np.linspace(0, 4, rows)
    return pd.DataFrame({
        "timestamp": idx,
        "open": base,
        "high": base + 1,
        "low": base - 1,
        "close": base,
        "volume": 1000,
    })


def fake_audit():
    return {
        "rows": 80,
        "label_scheme": "expected_value_classification",
        "label_level_mode": "atr",
        "feature_correlations_to_future_return": {
            "rsi": 0.04,
            "macd_hist": -0.02,
        },
        "removal_candidates": ["macd_hist"],
        "ablation_results": {
            "all_current": {
                "status": "ok",
                "trades": 35,
                "average_return": 0.03,
                "profit_factor": 1.2,
                "model_importance": {"rsi": 0.6, "macd_hist": 0.4},
                "permutation_importance": {"rsi": 0.02, "macd_hist": -0.01},
            },
            "returns_only": {
                "status": "ok",
                "trades": 30,
                "average_return": -0.01,
                "profit_factor": 0.8,
            },
            "dummy_random": {
                "status": "ok",
                "trades": 35,
                "average_return": -0.02,
                "profit_factor": 0.7,
            },
        },
    }


class FeatureAuditCliTests(unittest.IsolatedAsyncioTestCase):
    def test_parser_accepts_one_or_more_symbols(self):
        args = script.build_parser().parse_args(["--symbol", "ADA", "--symbol", "ETHUSDT"])

        self.assertEqual(args.symbol, ["ADA", "ETHUSDT"])

    def test_normalizes_usdt_symbol_to_base_asset(self):
        self.assertEqual(script.normalize_symbol_input("ADAUSDT"), "ADA")
        self.assertEqual(script.normalize_symbol_input("ETH/USDT"), "ETH")

    async def test_generates_json_and_markdown_for_default_horizons_without_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = Namespace(
                symbol=["ADAUSDT"],
                timeframe="1h",
                max_candles=80,
                horizon_candles=None,
                output_dir=tmp,
            )
            with patch.object(script, "fetch_binance_klines", new=AsyncMock(return_value=sample_candles())) as fetch:
                with patch.object(script, "run_feature_audit", side_effect=lambda candles, horizon_candles: fake_audit()) as audit:
                    batch = await script.run_feature_audit_cli(args)

            self.assertEqual(fetch.call_args.args[0], "ADA")
            self.assertEqual(len(batch["reports"]), 2)
            self.assertEqual([call.kwargs["horizon_candles"] for call in audit.call_args_list], [4, 10])

            json_paths = sorted(Path(tmp).glob("feature_audit_ADAUSDT_h*_*.json"))
            md_paths = sorted(Path(tmp).glob("feature_audit_ADAUSDT_h*_*.md"))
            self.assertEqual(len(json_paths), 2)
            self.assertEqual(len(md_paths), 2)
            self.assertTrue(any("_h4_" in path.name for path in json_paths))
            self.assertTrue(any("_h10_" in path.name for path in json_paths))

            payload = json.loads(json_paths[0].read_text(encoding="utf-8"))
            self.assertIn("feature_correlations_to_future_return", payload["audit"])
            self.assertIn("ablation_results", payload["audit"])
            self.assertIn("removal_candidates", payload["audit"])
            self.assertEqual(payload["guardrails"]["no_supabase_writes"], True)


if __name__ == "__main__":
    unittest.main()
