import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_focused_stability import (
    DEFAULT_FOCUSED_REGISTRY,
    build_parser,
    run_focused_audit,
)


def focused_config(config_id="cfg1", symbol="ADA", horizon=12, rr=2.0, atr=1.25):
    return {
        "config_id": config_id,
        "symbol": symbol,
        "timeframe": "1h",
        "horizon_candles": horizon,
        "risk_reward": rr,
        "atr_stop_multiplier": atr,
        "cost_mode": "low_costs",
        "strategy_mode": "xgboost",
        "research_phase": "focused_crypto_watchlist_v2a",
    }


def registry_row(config_id="cfg1", classification="unstable_watchlist", json_path=None, **overrides):
    return {
        "config_id": config_id,
        "status": "completed",
        "classification": classification,
        "config": focused_config(config_id=config_id, **overrides),
        "json_path": json_path,
        "markdown_path": None,
        "started_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "2026-01-01T01:00:00+00:00",
        "error": None,
    }


def window(index, validation_positive=True, beats_random=True, beats_deterministic=True, test_pf=0.8, test_avg=-0.1):
    return {
        "window_index": index,
        "window_status": "valid",
        "validation_positive": validation_positive,
        "beats_random_validation": beats_random,
        "beats_deterministic_validation": beats_deterministic,
        "test_profit_factor": test_pf,
        "test_avg_return": test_avg,
        "directional_bias": "balanced",
        "n_trades": 10,
    }


def result_json(pf=1.25, avg=0.1, classification="unstable_watchlist", symbol="ADA"):
    return {
        "setups": [{
            "classification": classification,
            "setup": focused_config(symbol=symbol),
            "aggregate": {
                "valid_windows": 3,
                "validation_positive_rate": 0.58,
                "beats_random_rate": 0.58,
                "beats_deterministic_rate": 0.58,
                "test_confirm_rate": 0.2,
                "test_contradiction_rate": 0.55,
                "median_validation_pf": pf,
                "median_validation_avg_return": avg,
                "worst_validation_drawdown": 72.0,
                "median_test_pf": 0.8,
                "median_test_avg_return": -0.05,
            },
            "windows": [
                window(0, True, True, True),
                window(1, True, True, False),
                window(2, False, False, True),
            ],
        }]
    }


class FocusedStabilityAuditTests(unittest.TestCase):
    def write_registry(self, path: Path, rows: list[dict]):
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    def test_parser_defaults_to_focused_registry(self):
        args = build_parser().parse_args([])

        self.assertEqual(Path(args.registry), DEFAULT_FOCUSED_REGISTRY)
        self.assertEqual(args.top_n, 10)

    def test_generates_focused_audit_markdown_and_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "focused_v2a_registry.jsonl"
            result_path = Path(tmp) / "result.json"
            result_path.write_text(json.dumps(result_json()), encoding="utf-8")
            self.write_registry(registry, [registry_row("cfg1", json_path=str(result_path))])

            summary = run_focused_audit(registry_path=registry, output_dir=tmp, top_n=1)
            markdown = Path(summary["markdown_path"]).read_text(encoding="utf-8")

        self.assertIn("focused_v2a_stability_audit_", Path(summary["json_path"]).name)
        self.assertIn("Focused v2A Stability Audit", markdown)
        self.assertIn("Selection uses validation metrics only", markdown)
        self.assertEqual(summary["summary"]["audited_configs"], 1)
        self.assertEqual(summary["summary"]["watchlist_records"], 1)

    def test_missing_json_does_not_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "focused_v2a_registry.jsonl"
            self.write_registry(registry, [registry_row("cfg1", json_path=str(Path(tmp) / "missing.json"))])

            summary = run_focused_audit(registry_path=registry, output_dir=tmp, top_n=1)

        self.assertEqual(summary["summary"]["missing_json_in_selected"], 1)
        self.assertEqual(summary["summary"]["audited_configs"], 1)

    def test_selection_uses_validation_not_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "focused_v2a_registry.jsonl"
            low_val = Path(tmp) / "low_val.json"
            high_val = Path(tmp) / "high_val.json"
            low_val.write_text(json.dumps(result_json(pf=1.01, avg=0.2, symbol="ADA")), encoding="utf-8")
            high_val.write_text(json.dumps(result_json(pf=1.4, avg=0.01, symbol="ETH")), encoding="utf-8")
            self.write_registry(registry, [
                registry_row("low_val", json_path=str(low_val), symbol="ADA"),
                registry_row("high_val", json_path=str(high_val), symbol="ETH"),
            ])

            summary = run_focused_audit(registry_path=registry, output_dir=tmp, top_n=1)

        self.assertIn("ETH", summary["audits"][0]["label"])


if __name__ == "__main__":
    unittest.main()
