import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_refined_stability import (
    audit_record,
    build_stability_audit,
    diagnose_watchlist_class,
    load_result_payload,
    render_markdown,
    run_audit,
    select_top_watchlist_configs,
    summarize_windows,
)
from scripts.summarize_research_registry import enrich_registry_records, load_latest_registry_records


def config(config_id="cfg1", horizon=12, rr=2.8, atr=1.25, cost="low_costs"):
    return {
        "config_id": config_id,
        "symbol": "SOL",
        "timeframe": "1h",
        "horizon_candles": horizon,
        "risk_reward": rr,
        "atr_stop_multiplier": atr,
        "cost_mode": cost,
        "strategy_mode": "xgboost",
    }


def registry_row(config_id="cfg1", classification="unstable_watchlist", json_path=None, **kwargs):
    return {
        "config_id": config_id,
        "status": "completed",
        "classification": classification,
        "config": config(config_id=config_id, **kwargs),
        "json_path": json_path,
        "markdown_path": None,
        "started_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "2026-01-01T01:00:00+00:00",
        "error": None,
    }


def window(
    index,
    *,
    validation_positive=True,
    beats_random=True,
    beats_deterministic=True,
    val_pf=1.2,
    val_avg=0.1,
    test_pf=1.1,
    test_avg=0.05,
    test_confirms=True,
    bias="balanced",
    n_trades=10,
):
    return {
        "window_index": index,
        "window_status": "valid",
        "validation_positive": validation_positive,
        "beats_random_validation": beats_random,
        "beats_deterministic_validation": beats_deterministic,
        "validation_profit_factor": val_pf,
        "validation_avg_return": val_avg,
        "validation_drawdown": 20.0,
        "test_profit_factor": test_pf,
        "test_avg_return": test_avg,
        "test_confirms": test_confirms,
        "directional_bias": bias,
        "n_trades": n_trades,
    }


def result_json(
    *,
    pf=1.2,
    avg=0.1,
    val_rate=0.58,
    random_rate=0.58,
    det_rate=0.45,
    test_confirm=0.30,
    test_contradiction=0.45,
    drawdown=40.0,
    classification="unstable_watchlist",
):
    windows = [
        window(0, validation_positive=True, beats_random=True, beats_deterministic=True, n_trades=12),
        window(1, validation_positive=True, beats_random=True, beats_deterministic=False, test_pf=0.8, test_avg=-0.1, test_confirms=False, n_trades=8),
        window(2, validation_positive=False, beats_random=False, beats_deterministic=False, n_trades=10),
    ]
    return {
        "setups": [{
            "classification": classification,
            "setup": config(),
            "aggregate": {
                "valid_windows": 3,
                "validation_positive_rate": val_rate,
                "beats_random_rate": random_rate,
                "beats_deterministic_rate": det_rate,
                "test_confirm_rate": test_confirm,
                "test_contradiction_rate": test_contradiction,
                "median_validation_pf": pf,
                "median_validation_avg_return": avg,
                "worst_validation_drawdown": drawdown,
                "median_test_pf": 0.9,
                "median_test_avg_return": -0.02,
            },
            "windows": windows,
        }]
    }


class RefinedStabilityAuditTests(unittest.TestCase):
    def write_registry(self, path: Path, rows: list[dict]):
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    def test_loads_refined_registry_and_result_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "refined_registry.jsonl"
            result_path = Path(tmp) / "result.json"
            result_path.write_text(json.dumps(result_json()), encoding="utf-8")
            self.write_registry(registry, [registry_row("cfg1", json_path=str(result_path))])

            records = enrich_registry_records(load_latest_registry_records(registry))
            payload = load_result_payload(records[0])

        self.assertEqual(len(records), 1)
        self.assertIsNotNone(payload)
        self.assertEqual(payload["setups"][0]["aggregate"]["median_validation_pf"], 1.2)

    def test_select_top_configs_uses_validation_not_test(self):
        records = [
            registry_row("low_val_high_test"),
            registry_row("high_val_low_test"),
        ]
        records[0]["aggregate"] = {"median_validation_pf": 1.05, "median_validation_avg_return": 0.1, "median_test_pf": 9.0}
        records[1]["aggregate"] = {"median_validation_pf": 1.30, "median_validation_avg_return": 0.05, "median_test_pf": 0.1}

        selected = select_top_watchlist_configs(records, top_n=1)

        self.assertEqual(selected[0]["config_id"], "high_val_low_test")

    def test_summarize_windows_counts_trades_bias_and_window_lists(self):
        summary = summarize_windows([
            window(0, validation_positive=True, beats_random=True, beats_deterministic=False, n_trades=10, bias="buy_heavy"),
            window(1, validation_positive=False, beats_random=False, beats_deterministic=True, n_trades=20, bias="sell_heavy", test_pf=0.7, test_avg=-0.1),
        ])

        self.assertEqual(summary["n_trades_total"], 30)
        self.assertEqual(summary["n_trades_median_per_window"], 15.0)
        self.assertEqual(summary["directional_bias_counts"], {"buy_heavy": 1, "sell_heavy": 1})
        self.assertEqual(summary["positive_windows"], [0])
        self.assertEqual(summary["negative_windows"], [1])
        self.assertEqual(summary["beats_random_windows"], [0])
        self.assertEqual(summary["beats_deterministic_windows"], [1])
        self.assertEqual(summary["test_contradiction_windows"], [1])

    def test_diagnostic_classes_failure_modes(self):
        self.assertEqual(diagnose_watchlist_class({
            "valid_windows": 19,
            "validation_positive_rate": 0.58,
            "beats_random_rate": 0.58,
            "beats_deterministic_rate": 0.45,
            "median_validation_pf": 1.2,
            "median_validation_avg_return": 0.2,
            "test_confirm_rate": 0.2,
            "test_contradiction_rate": 0.5,
            "worst_validation_drawdown": 30,
        }, {"positive_windows": [1, 2], "negative_windows": [3]}), "near_stable_but_low_test_confirm")
        self.assertEqual(diagnose_watchlist_class({
            "valid_windows": 19,
            "validation_positive_rate": 0.5,
            "beats_random_rate": 0.5,
            "beats_deterministic_rate": 0.5,
            "median_validation_pf": 1.1,
            "median_validation_avg_return": 0.1,
            "test_confirm_rate": 0.5,
            "test_contradiction_rate": 0.1,
            "worst_validation_drawdown": 60,
        }, {"positive_windows": [1, 2], "negative_windows": [3]}), "near_stable_but_drawdown_high")
        self.assertEqual(diagnose_watchlist_class({
            "valid_windows": 19,
            "validation_positive_rate": 0.5,
            "beats_random_rate": 0.45,
            "beats_deterministic_rate": 0.6,
            "median_validation_pf": 1.1,
            "median_validation_avg_return": 0.1,
        }, {"positive_windows": [1, 2], "negative_windows": [3]}), "near_stable_but_random_not_beaten")

    def test_audit_record_calculates_stable_failure_reasons(self):
        with tempfile.TemporaryDirectory() as tmp:
            result_path = Path(tmp) / "result.json"
            result_path.write_text(json.dumps(result_json(random_rate=0.45, det_rate=0.45)), encoding="utf-8")
            record = registry_row("cfg1", json_path=str(result_path))
            record = enrich_registry_records([record])[0]

            audit = audit_record(record)

        self.assertEqual(audit["label"], "SOL 1h h12 RR2.8 ATR1.25 low_costs xgboost")
        self.assertTrue(any("beats_random_rate" in reason for reason in audit["stable_failure_reasons"]))
        self.assertTrue(any("beats_deterministic_rate" in reason for reason in audit["stable_failure_reasons"]))

    def test_build_report_generates_markdown_and_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "refined_registry.jsonl"
            result_path = Path(tmp) / "result.json"
            result_path.write_text(json.dumps(result_json()), encoding="utf-8")
            self.write_registry(registry, [registry_row("cfg1", json_path=str(result_path))])

            summary = run_audit(registry_path=registry, output_dir=tmp, top_n=1)
            markdown = Path(summary["markdown_path"]).read_text(encoding="utf-8")
            loaded_json = json.loads(Path(summary["json_path"]).read_text(encoding="utf-8"))
            self.assertTrue(Path(summary["json_path"]).exists())
            self.assertTrue(Path(summary["markdown_path"]).exists())

        self.assertIn("Refined Stability Audit", markdown)
        self.assertIn("Selection uses validation metrics only", markdown)
        self.assertEqual(loaded_json["summary"]["audited_configs"], 1)

    def test_missing_json_does_not_fail(self):
        records = enrich_registry_records([registry_row("cfg1", json_path="missing.json")])

        summary = build_stability_audit(records, top_n=1)
        markdown = render_markdown(summary)

        self.assertEqual(summary["summary"]["audited_configs"], 1)
        self.assertEqual(summary["summary"]["missing_json_in_selected"], 1)
        self.assertIn("missing selected JSON files", markdown)


if __name__ == "__main__":
    unittest.main()
