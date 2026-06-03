import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_research_registry import (
    build_global_summary,
    enrich_registry_records,
    load_latest_registry_records,
    render_markdown,
    summarize_registry,
)


def config(config_id="cfg1", horizon=16, rr=2.0, atr=1.25, cost="low_costs"):
    return {
        "config_id": config_id,
        "symbol": "SOL",
        "timeframe": "1h",
        "horizon_candles": horizon,
        "risk_reward": rr,
        "atr_stop_multiplier": atr,
        "cost_mode": cost,
        "strategy_mode": "xgboost",
        "max_candles": 5000,
        "window_size_candles": 600,
        "step_size_candles": 250,
    }


def registry_row(config_id="cfg1", status="completed", classification="multi_window_reject", json_path=None, **overrides):
    row = {
        "config_id": config_id,
        "status": status,
        "classification": classification,
        "config": config(config_id=config_id, **overrides),
        "json_path": json_path,
        "markdown_path": None,
        "started_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "2026-01-01T01:00:00+00:00",
        "error": None,
    }
    return row


def result_json(classification="multi_window_reject", pf=0.8, avg=-0.1, test_pf=1.2, test_confirm=0.5):
    return {
        "setups": [{
            "classification": classification,
            "aggregate": {
                "median_validation_pf": pf,
                "median_validation_avg_return": avg,
                "median_test_pf": test_pf,
                "test_confirm_rate": test_confirm,
                "validation_positive_rate": 0.25,
                "beats_random_rate": 0.3,
                "beats_deterministic_rate": 0.2,
                "worst_validation_drawdown": 20.0,
            },
        }]
    }


class ResearchRegistrySummaryTests(unittest.TestCase):
    def write_registry(self, path: Path, rows: list[dict]):
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    def test_dedupes_registry_by_latest_config_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "registry.jsonl"
            self.write_registry(registry, [
                registry_row("cfg1", status="running", classification=None),
                registry_row("cfg1", status="completed", classification="multi_window_reject"),
            ])

            latest = load_latest_registry_records(registry)

        self.assertEqual(len(latest), 1)
        self.assertEqual(latest[0]["status"], "completed")
        self.assertEqual(latest[0]["classification"], "multi_window_reject")

    def test_ignores_old_running_when_completed_exists_later(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "registry.jsonl"
            self.write_registry(registry, [
                registry_row("cfg1", status="running", classification=None),
                registry_row("cfg2", status="completed", classification="unstable_watchlist"),
                registry_row("cfg1", status="completed", classification="multi_window_reject"),
            ])

            summary = build_global_summary(enrich_registry_records(load_latest_registry_records(registry)))

        self.assertEqual(summary["summary"]["unique_configs"], 2)
        self.assertEqual(summary["summary"]["classification_counts"], {
            "multi_window_reject": 1,
            "unstable_watchlist": 1,
        })

    def test_reads_json_paths_correctly(self):
        with tempfile.TemporaryDirectory() as tmp:
            result_path = Path(tmp) / "result.json"
            result_path.write_text(json.dumps(result_json(pf=1.2, avg=0.05)), encoding="utf-8")
            record = registry_row("cfg1", json_path=str(result_path))

            enriched = enrich_registry_records([record])

        self.assertTrue(enriched[0]["json_loaded"])
        self.assertEqual(enriched[0]["aggregate"]["median_validation_pf"], 1.2)

    def test_generates_markdown_and_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "registry.jsonl"
            result_path = Path(tmp) / "result.json"
            result_path.write_text(json.dumps(result_json(classification="unstable_watchlist", pf=1.05, avg=0.01)), encoding="utf-8")
            self.write_registry(registry, [
                registry_row("cfg1", classification="unstable_watchlist", json_path=str(result_path)),
            ])

            summary = summarize_registry(registry, output_dir=tmp)

            self.assertTrue(Path(summary["json_path"]).exists())
            self.assertTrue(Path(summary["markdown_path"]).exists())
            markdown = Path(summary["markdown_path"]).read_text(encoding="utf-8")

        self.assertIn("Research Daemon Global Summary", markdown)
        self.assertIn("unstable_watchlist", markdown)

    def test_no_test_selection_guardrail_in_report(self):
        summary = build_global_summary(enrich_registry_records([
            registry_row("cfg1", classification="multi_window_reject"),
        ]))

        markdown = render_markdown(summary)

        self.assertTrue(summary["guardrails"]["test_not_used_for_selection"])
        self.assertIn("test metrics are diagnostic only", markdown)
        self.assertIn("Diagnostic Only, Not Selectable", markdown)

    def test_missing_individual_json_does_not_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing.json"
            record = registry_row("cfg1", json_path=str(missing))

            enriched = enrich_registry_records([record])
            summary = build_global_summary(enriched)

        self.assertTrue(enriched[0]["json_missing"])
        self.assertEqual(summary["summary"]["json_missing"], 1)
        self.assertEqual(summary["summary"]["unique_configs"], 1)

    def test_cost_grouping_compares_cost_modes(self):
        records = enrich_registry_records([
            registry_row("cfg1", cost="low_costs"),
            registry_row("cfg2", cost="medium_costs_current", classification="unstable_watchlist"),
        ])
        summary = build_global_summary(records)

        self.assertIn("cost_mode=low_costs", summary["cost_analysis"])
        self.assertIn("cost_mode=medium_costs_current", summary["cost_analysis"])


if __name__ == "__main__":
    unittest.main()
