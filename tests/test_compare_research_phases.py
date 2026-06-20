import json
import tempfile
import unittest
from pathlib import Path

from scripts.compare_research_phases import build_comparison, run_comparison


def row(config_id: str, pf: float, avg: float, dd: float, classification: str = "unstable_watchlist"):
    return {
        "config_id": config_id,
        "status": "completed",
        "classification": classification,
        "config": {
            "symbol": "ETH",
            "timeframe": "1h",
            "horizon_candles": 12,
            "risk_reward": 2.0,
            "atr_stop_multiplier": 1.5,
            "cost_mode": "low_costs",
            "strategy_mode": "xgboost",
        },
        "aggregate": {
            "median_validation_pf": pf,
            "median_validation_avg_return": avg,
            "worst_validation_drawdown": dd,
            "test_confirm_rate": 0.5,
        },
    }


def registry_record(config_id: str, json_path: str, pf: float, avg: float, dd: float, classification: str):
    return {
        "config_id": config_id,
        "status": "completed",
        "classification": classification,
        "config": row(config_id, pf, avg, dd, classification)["config"],
        "json_path": json_path,
        "markdown_path": None,
        "started_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "2026-01-01T01:00:00+00:00",
        "error": None,
    }


def result_json(pf: float, avg: float, dd: float, classification: str):
    return {
        "setups": [{
            "classification": classification,
            "aggregate": {
                "median_validation_pf": pf,
                "median_validation_avg_return": avg,
                "worst_validation_drawdown": dd,
                "test_confirm_rate": 0.5,
            },
        }]
    }


class CompareResearchPhasesTests(unittest.TestCase):
    def test_build_comparison_detects_validation_regression(self):
        summary = build_comparison(
            [row("base", pf=1.2, avg=0.1, dd=30)],
            [row("cand", pf=0.9, avg=-0.1, dd=40, classification="multi_window_reject")],
            baseline_name="base",
            candidate_name="candidate",
        )

        self.assertEqual(summary["summary"]["matched_configs"], 1)
        self.assertEqual(summary["summary"]["verdict_counts"], {"worse_validation": 1})
        self.assertEqual(summary["rows"][0]["delta_validation_pf"], -0.3)
        self.assertTrue(summary["guardrails"]["test_not_used_for_selection"])

    def test_build_comparison_detects_validation_improvement(self):
        summary = build_comparison(
            [row("base", pf=0.9, avg=-0.1, dd=30, classification="multi_window_reject")],
            [row("cand", pf=1.1, avg=0.05, dd=35)],
        )

        self.assertEqual(summary["summary"]["verdict_counts"], {"improved_validation": 1})

    def test_run_comparison_writes_markdown_and_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            base_json = Path(tmp) / "base_result.json"
            cand_json = Path(tmp) / "cand_result.json"
            base_json.write_text(json.dumps(result_json(1.1, 0.1, 30, "unstable_watchlist")), encoding="utf-8")
            cand_json.write_text(json.dumps(result_json(0.9, -0.1, 40, "multi_window_reject")), encoding="utf-8")
            base_registry = Path(tmp) / "base.jsonl"
            cand_registry = Path(tmp) / "cand.jsonl"
            base_registry.write_text(
                json.dumps(registry_record("cfg", str(base_json), 1.1, 0.1, 30, "unstable_watchlist")) + "\n",
                encoding="utf-8",
            )
            cand_registry.write_text(
                json.dumps(registry_record("cfg", str(cand_json), 0.9, -0.1, 40, "multi_window_reject")) + "\n",
                encoding="utf-8",
            )

            class Args:
                baseline_registry = str(base_registry)
                candidate_registry = str(cand_registry)
                baseline_name = "base"
                candidate_name = "candidate"
                output_dir = tmp
                filename_prefix = "comparison"

            summary = run_comparison(Args())
            markdown = Path(summary["markdown_path"]).read_text(encoding="utf-8")
            self.assertTrue(Path(summary["json_path"]).exists())

        self.assertIn("Research only. No trading signal.", markdown)
        self.assertIn("Top Validation Regressions", markdown)


if __name__ == "__main__":
    unittest.main()
