import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_crypto_multi_research import (
    build_crypto_multi_summary,
    render_crypto_multi_markdown,
    summarize_crypto_multi_registry,
)
from scripts.summarize_research_registry import enrich_registry_records, load_latest_registry_records


def config(
    config_id="cfg1",
    symbol="SOL",
    horizon=12,
    rr=2.0,
    atr=1.25,
    cost="low_costs",
):
    return {
        "config_id": config_id,
        "symbol": symbol,
        "timeframe": "1h",
        "horizon_candles": horizon,
        "risk_reward": rr,
        "atr_stop_multiplier": atr,
        "cost_mode": cost,
        "strategy_mode": "xgboost",
        "max_candles": 5000,
        "window_size_candles": 600,
        "step_size_candles": 250,
        "research_phase": "crypto_multi_asset_grid_v1",
    }


def registry_row(
    config_id="cfg1",
    status="completed",
    classification="multi_window_reject",
    json_path=None,
    **overrides,
):
    return {
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


def result_json(classification="multi_window_reject", pf=0.8, avg=-0.1, test_pf=1.2):
    return {
        "setups": [{
            "classification": classification,
            "aggregate": {
                "total_windows": 4,
                "valid_windows": 4,
                "median_validation_pf": pf,
                "median_validation_avg_return": avg,
                "median_test_pf": test_pf,
                "test_confirm_rate": 0.5,
                "validation_positive_rate": 0.4,
                "beats_random_rate": 0.4,
                "beats_deterministic_rate": 0.3,
                "worst_validation_drawdown": 20.0,
            },
        }]
    }


class CryptoMultiSummaryTests(unittest.TestCase):
    def write_registry(self, path: Path, rows: list[dict]):
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    def test_reads_crypto_multi_registry_and_dedupes_by_config_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "crypto_multi_registry.jsonl"
            self.write_registry(registry, [
                registry_row("cfg1", status="running", classification=None, symbol="SOL"),
                registry_row("cfg1", status="completed", classification="unstable_watchlist", symbol="SOL"),
                registry_row("cfg2", status="completed", classification="multi_window_reject", symbol="BTC"),
            ])

            records = load_latest_registry_records(registry)
            summary = build_crypto_multi_summary(enrich_registry_records(records))

        self.assertEqual(len(records), 2)
        self.assertEqual(summary["summary"]["total_configs"], 2)
        self.assertEqual(summary["summary"]["classification_counts"]["unstable_watchlist"], 1)

    def test_reads_result_json_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            result_path = Path(tmp) / "result.json"
            result_path.write_text(json.dumps(result_json(pf=1.18, avg=0.04)), encoding="utf-8")
            record = registry_row("cfg1", classification="unstable_watchlist", json_path=str(result_path), symbol="SOL")

            summary = build_crypto_multi_summary(enrich_registry_records([record]))

        self.assertEqual(summary["top"]["median_validation_pf"][0]["median_validation_pf"], 1.18)
        self.assertEqual(summary["groupings"]["symbol"]["symbol=SOL"]["median_validation_avg_return"], 0.04)

    def test_missing_result_json_does_not_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing.json"
            record = registry_row("cfg1", json_path=str(missing), symbol="ETH")

            summary = build_crypto_multi_summary(enrich_registry_records([record]))

        self.assertEqual(summary["summary"]["json_missing"], 1)
        self.assertEqual(summary["summary"]["total_configs"], 1)

    def test_generates_markdown_and_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "crypto_multi_registry.jsonl"
            result_path = Path(tmp) / "result.json"
            result_path.write_text(json.dumps(result_json(
                classification="unstable_watchlist",
                pf=1.08,
                avg=0.02,
            )), encoding="utf-8")
            self.write_registry(registry, [
                registry_row("cfg1", classification="unstable_watchlist", json_path=str(result_path), symbol="SOL"),
            ])

            summary = summarize_crypto_multi_registry(registry_path=registry, output_dir=tmp)
            markdown = Path(summary["markdown_path"]).read_text(encoding="utf-8")
            json_exists = Path(summary["json_path"]).exists()
            markdown_exists = Path(summary["markdown_path"]).exists()

        self.assertTrue(json_exists)
        self.assertTrue(markdown_exists)
        self.assertIn("Crypto Multi-Asset Global Summary", markdown)
        self.assertIn("Grouping By Symbol", markdown)
        self.assertIn("symbol=SOL", markdown)

    def test_report_includes_required_groupings_and_best_combinations(self):
        records = enrich_registry_records([
            registry_row("cfg1", symbol="SOL", horizon=12, rr=2.0, atr=1.25),
            registry_row("cfg2", symbol="BTC", horizon=14, rr=2.5, atr=1.0),
        ])

        summary = build_crypto_multi_summary(records)

        self.assertIn("symbol=SOL", summary["groupings"]["symbol"])
        self.assertIn("horizon_candles=12", summary["groupings"]["horizon_candles"])
        self.assertIn("risk_reward=2.5", summary["groupings"]["risk_reward"])
        self.assertIn("atr_stop_multiplier=1.0", summary["groupings"]["atr_stop_multiplier"])
        self.assertTrue(any("symbol=SOL" in row["group"] for row in summary["best_symbol_horizon_rr_atr"]))

    def test_no_test_selection_guardrail(self):
        summary = build_crypto_multi_summary(enrich_registry_records([
            registry_row("cfg1", classification="multi_window_reject", symbol="SOL"),
        ]))
        markdown = render_crypto_multi_markdown(summary)

        self.assertTrue(summary["guardrails"]["test_not_used_for_selection"])
        self.assertIn("Diagnostic Only, Not Selectable", markdown)
        self.assertIn("Validation selects", markdown)

    def test_asset_diagnostics_identifies_watchlist_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            sol_result = Path(tmp) / "sol.json"
            btc_result = Path(tmp) / "btc.json"
            sol_result.write_text(json.dumps(result_json(
                classification="unstable_watchlist",
                pf=1.1,
                avg=0.03,
            )), encoding="utf-8")
            btc_result.write_text(json.dumps(result_json(
                classification="multi_window_reject",
                pf=0.7,
                avg=-0.2,
            )), encoding="utf-8")
            records = enrich_registry_records([
                registry_row("cfg1", classification="unstable_watchlist", json_path=str(sol_result), symbol="SOL"),
                registry_row("cfg2", classification="multi_window_reject", json_path=str(btc_result), symbol="BTC"),
            ])
            summary = build_crypto_multi_summary(records)

        self.assertIn("SOL", summary["asset_diagnostics"]["promising_assets"])
        self.assertIn("BTC", summary["asset_diagnostics"]["discardable_assets"])


if __name__ == "__main__":
    unittest.main()
