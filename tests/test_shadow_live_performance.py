import json
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_shadow_live_performance import (
    build_config_diagnostics,
    build_parser,
    build_report,
    confidence_bucket,
    performance_metrics,
    render_markdown,
    save_report,
)


def signal(signal_id, outcome, pnl, *, symbol="ADA", side="LONG", config_id="cfg-1", confidence=60):
    return {
        "shadow_signal_id": signal_id,
        "config_id": config_id,
        "classification": "unstable_watchlist",
        "symbol": symbol,
        "timeframe": "1h",
        "side": side,
        "status": "CLOSED",
        "outcome": outcome,
        "pnl_pct": pnl,
        "confidence": confidence,
        "generated_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T04:00:00+00:00",
        "agent_review": {"review_status": "CAUTION"},
    }


class ShadowLivePerformanceTests(unittest.TestCase):
    def test_metrics_calculate_profit_factor_and_drawdown(self):
        metrics = performance_metrics([
            signal("1", "WIN", 2.0),
            signal("2", "LOSS", -1.0),
            signal("3", "LOSS", -0.5),
        ])

        self.assertEqual(metrics["closed"], 3)
        self.assertEqual(metrics["wins"], 1)
        self.assertAlmostEqual(metrics["profit_factor"], 2.0 / 1.5, places=6)
        self.assertAlmostEqual(metrics["avg_return_pct"], 0.5 / 3, places=6)
        self.assertAlmostEqual(metrics["max_drawdown_pct"], 1.495, places=3)
        self.assertEqual(metrics["closed_without_pnl"], 0)

    def test_config_diagnostics_do_not_use_test_metrics(self):
        signals = [signal(str(index), "LOSS", -1.0) for index in range(5)]
        configs = [{
            "config_id": "cfg-1",
            "classification": "unstable_watchlist",
            "median_validation_pf": 1.2,
            "median_validation_avg_return": 0.1,
            "median_test_pf": 99.0,
        }]

        diagnostics = build_config_diagnostics(signals, configs)

        self.assertEqual(diagnostics[0]["diagnostic_status"], "negative_live_watch")
        self.assertTrue(diagnostics[0]["historical_live_mismatch"])
        self.assertFalse(diagnostics[0]["selection_uses_test"])
        self.assertNotIn("median_test_pf", json.dumps(diagnostics))

    def test_report_groups_symbol_side_review_and_confidence(self):
        report = build_report(
            signals=[
                signal("1", "WIN", 1.0, symbol="ADA", side="LONG", confidence=59),
                signal("2", "LOSS", -2.0, symbol="SOL", side="SHORT", confidence=72),
            ],
            cycles=[{"health_status": "HEALTH_OK", "opened_signals": 1, "configs_scanned": 6}],
            research_configs=[],
            source="supabase",
        )

        self.assertIn("ADA", report["by_symbol"])
        self.assertIn("SHORT", report["by_side"])
        self.assertIn("CAUTION", report["by_agent_review"])
        self.assertIn("55-60", report["confidence_buckets"])
        self.assertEqual(report["cycle_health"]["total_cycles"], 1)
        self.assertIn("last_24h", report["cycle_health"])
        self.assertTrue(report["guardrails"]["test_metrics_not_used_for_selection"])
        self.assertEqual(report["strategy_performance"]["closed"], 2)

    def test_technical_expiry_is_excluded_from_strategy_performance(self):
        technical = signal("expired", "EXPIRED", None)
        technical["status"] = "EXPIRED"
        technical["exit_reason"] = "evaluation_http_error"
        report = build_report(
            signals=[signal("win", "WIN", 1.0), technical],
            source="supabase",
        )

        self.assertEqual(report["overall"]["closed"], 2)
        self.assertEqual(report["strategy_performance"]["closed"], 1)
        self.assertEqual(report["technical_exclusions"]["count"], 1)

    def test_missing_optional_fields_do_not_fail(self):
        report = build_report(
            signals=[{"status": "OPEN", "symbol": "ETH"}],
            source="local_jsonl",
        )
        self.assertEqual(report["overall"]["open"], 1)
        self.assertEqual(report["overall"]["closed"], 0)
        self.assertIn("Research only", render_markdown(report))

    def test_save_report_generates_markdown_and_json(self):
        report = build_report(signals=[signal("1", "WIN", 1.0)], source="supabase")
        with tempfile.TemporaryDirectory() as directory:
            paths = save_report(report, directory)
            self.assertTrue(Path(paths["json_path"]).exists())
            self.assertTrue(Path(paths["markdown_path"]).exists())
            payload = json.loads(Path(paths["json_path"]).read_text(encoding="utf-8"))
            self.assertEqual(payload["source"], "supabase")

    def test_parser_defaults_to_auto_source(self):
        args = build_parser().parse_args([])
        self.assertEqual(args.source, "auto")
        self.assertTrue(args.write_report)

    def test_confidence_bucket_boundaries(self):
        self.assertEqual(confidence_bucket(49), "<50")
        self.assertEqual(confidence_bucket(59.9), "55-60")
        self.assertEqual(confidence_bucket(70), "70+")


if __name__ == "__main__":
    unittest.main()
