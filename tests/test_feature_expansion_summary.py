import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_feature_expansion_research import summarize_feature_expansion_registry


class FeatureExpansionSummaryTests(unittest.TestCase):
    def test_summary_reads_registry_and_result_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_path = root / "result.json"
            payload = {
                "setups": [{
                    "classification": "feature_expansion_watchlist",
                    "aggregate": {
                        "median_validation_pf": 0.92,
                        "median_validation_avg_return": 0.03,
                        "beats_time_only_rate": 0.7,
                        "beats_dummy_random_rate": 0.8,
                        "objective_pf_rate": 0.2,
                    },
                }]
            }
            result_path.write_text(json.dumps(payload), encoding="utf-8")
            registry = root / "registry.jsonl"
            registry.write_text(json.dumps({
                "config_id": "abc",
                "status": "completed",
                "classification": "feature_expansion_watchlist",
                "config": {
                    "symbol": "ADA",
                    "timeframe": "1h",
                    "feature_set": "baseline_plus_funding",
                    "horizon_candles": 10,
                    "risk_reward": 2.0,
                    "atr_stop_multiplier": 1.5,
                    "cost_mode": "low_costs",
                },
                "json_path": str(result_path),
            }) + "\n", encoding="utf-8")

            summary = summarize_feature_expansion_registry(registry_path=registry, output_dir=root)

            self.assertEqual(summary["summary"]["unique_configs"], 1)
            self.assertEqual(summary["summary"]["classification_counts"]["feature_expansion_watchlist"], 1)
            self.assertIn("baseline_plus_funding", summary["groupings"]["feature_set"])
            self.assertTrue(Path(summary["json_path"]).exists())
            self.assertTrue(Path(summary["markdown_path"]).exists())


if __name__ == "__main__":
    unittest.main()
