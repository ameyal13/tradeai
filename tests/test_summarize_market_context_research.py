import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_market_context_research import DEFAULT_MARKET_CONTEXT_REGISTRY, build_parser
from scripts.summarize_research_registry import summarize_registry


def market_context_config(config_id="cfg1", symbol="ADA", horizon=12, rr=2.0, atr=1.25):
    return {
        "config_id": config_id,
        "symbol": symbol,
        "timeframe": "1h",
        "horizon_candles": horizon,
        "risk_reward": rr,
        "atr_stop_multiplier": atr,
        "cost_mode": "low_costs",
        "strategy_mode": "xgboost",
        "max_candles": 5000,
        "window_size_candles": 600,
        "step_size_candles": 250,
        "research_phase": "market_context_features_v1",
        "use_market_context_features": True,
        "feature_family": "current_plus_market_context_v1",
    }


def registry_row(config_id="cfg1", classification="unstable_watchlist", json_path=None, **overrides):
    return {
        "config_id": config_id,
        "status": "completed",
        "classification": classification,
        "config": market_context_config(config_id=config_id, **overrides),
        "json_path": json_path,
        "markdown_path": None,
        "started_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "2026-01-01T01:00:00+00:00",
        "error": None,
    }


def result_json(classification="unstable_watchlist", pf=1.2, avg=0.08):
    return {
        "setups": [{
            "classification": classification,
            "aggregate": {
                "total_windows": 19,
                "valid_windows": 19,
                "median_validation_pf": pf,
                "median_validation_avg_return": avg,
                "median_test_pf": 0.9,
                "test_confirm_rate": 0.3,
                "validation_positive_rate": 0.5,
                "beats_random_rate": 0.4,
                "beats_deterministic_rate": 0.3,
                "worst_validation_drawdown": 22.0,
            },
        }]
    }


class MarketContextResearchSummaryTests(unittest.TestCase):
    def write_registry(self, path: Path, rows: list[dict]):
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    def test_parser_defaults_to_market_context_registry(self):
        args = build_parser().parse_args([])

        self.assertEqual(args.top_limit, 15)
        self.assertEqual(Path(args.registry), DEFAULT_MARKET_CONTEXT_REGISTRY)

    def test_generates_market_context_markdown_and_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "market_context_v1_registry.jsonl"
            result_path = Path(tmp) / "result.json"
            result_path.write_text(json.dumps(result_json()), encoding="utf-8")
            self.write_registry(registry, [
                registry_row("cfg1", json_path=str(result_path), symbol="ADA", horizon=12, rr=2.0, atr=1.25),
                registry_row("cfg2", classification="multi_window_reject", symbol="ETH", horizon=10, rr=2.5, atr=1.5),
            ])

            summary = summarize_registry(
                registry_path=registry,
                output_dir=tmp,
                filename_prefix="market_context_v1_global_summary",
                top_limit=15,
                refined=True,
            )
            markdown = Path(summary["markdown_path"]).read_text(encoding="utf-8")

        self.assertIn("market_context_v1_global_summary_", Path(summary["json_path"]).name)
        self.assertIn("Refined Research Global Summary", markdown)
        self.assertIn("Top Watchlist Diagnostics: Why Not Stable", markdown)
        self.assertIn("Research only. No trading signal.", markdown)
        self.assertEqual(summary["summary"]["unique_configs"], 2)

    def test_missing_json_does_not_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "market_context_v1_registry.jsonl"
            self.write_registry(registry, [
                registry_row("cfg1", json_path=str(Path(tmp) / "missing.json")),
            ])

            summary = summarize_registry(
                registry_path=registry,
                output_dir=tmp,
                filename_prefix="market_context_v1_global_summary",
                refined=True,
            )

        self.assertEqual(summary["summary"]["json_missing"], 1)
        self.assertEqual(summary["summary"]["unique_configs"], 1)

    def test_test_metrics_are_diagnostic_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "market_context_v1_registry.jsonl"
            result_path = Path(tmp) / "result.json"
            result_path.write_text(json.dumps(result_json(pf=0.8, avg=-0.1)), encoding="utf-8")
            self.write_registry(registry, [
                registry_row("cfg1", classification="multi_window_reject", json_path=str(result_path)),
            ])

            summary = summarize_registry(
                registry_path=registry,
                output_dir=tmp,
                filename_prefix="market_context_v1_global_summary",
                refined=True,
            )
            markdown = Path(summary["markdown_path"]).read_text(encoding="utf-8")

        self.assertTrue(summary["guardrails"]["test_not_used_for_selection"])
        self.assertIn("Diagnostic Only, Not Selectable", markdown)


if __name__ == "__main__":
    unittest.main()
