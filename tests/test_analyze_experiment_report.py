import tempfile
import unittest
from pathlib import Path

from scripts.analyze_experiment_report import (
    analyze_trade_calibration,
    analyze_report,
    classify_row,
    load_report,
    load_trades_report,
    normalize_row,
    probability_bucket,
    render_markdown,
    write_outputs,
)


def base_row(**overrides):
    row = {
        "symbol": "BTC",
        "timeframe": "15m",
        "strategy_mode": "xgboost",
        "use_trade_labels": "true",
        "label_type": "trade_outcome_directional",
        "trade_label_scheme": "touch_only",
        "label_level_mode": "atr",
        "label_horizon_candles": "4",
        "total_predictions": "200",
        "evaluated_predictions": "160",
        "hold_count": "40",
        "win_rate": "14.0",
        "average_return": "-0.31",
        "total_return_pct": "-50",
        "profit_factor": "0.05",
        "max_drawdown": "39",
        "sharpe": "-10",
        "tp_hit_count": "25",
        "sl_hit_count": "36",
        "expired_count": "103",
        "buy_count": "145",
        "sell_count": "15",
        "buy_win_rate": "15",
        "sell_win_rate": "0",
        "buy_average_return": "-0.30",
        "sell_average_return": "-0.45",
        "avg_probability_buy_win": "0.67",
        "avg_probability_sell_win": "0.15",
        "max_probability_buy_win": "0.98",
        "max_probability_sell_win": "0.40",
        "avg_confidence": "70",
        "hold_reasons_summary": "{'probabilities_below_threshold': 40}",
        "hourly_performance_utc": '{"07:00": {"evaluated_predictions": 4, "average_return": 0.1, "profit_factor": 1.2}, "12:00": {"evaluated_predictions": 8, "average_return": -1.0, "profit_factor": 0.0}}',
        "confidence_buckets": '{"70-80": {"evaluated_predictions": 30, "average_return": -0.2}}',
        "warnings": "",
    }
    row.update(overrides)
    return row


class AnalyzeExperimentReportTests(unittest.TestCase):
    def test_parse_hold_reasons_summary(self):
        row = normalize_row(base_row())

        self.assertEqual(row["hold_reasons_summary"], {"probabilities_below_threshold": 40})

    def test_parse_hourly_performance(self):
        row = normalize_row(base_row())

        self.assertEqual(row["hourly_performance_utc"]["07:00"]["evaluated_predictions"], 4)

    def test_classifies_reject(self):
        row = normalize_row(base_row())

        self.assertEqual(classify_row(row), "reject")

    def test_classifies_candidate(self):
        row = normalize_row(base_row(
            evaluated_predictions="80",
            average_return="0.2",
            total_return_pct="16",
            profit_factor="1.4",
            max_drawdown="12",
            win_rate="55",
            expired_count="10",
        ))

        self.assertEqual(classify_row(row), "candidate")

    def test_classifies_weak_candidate(self):
        row = normalize_row(base_row(
            evaluated_predictions="50",
            average_return="-0.01",
            total_return_pct="-1",
            profit_factor="0.95",
            max_drawdown="10",
        ))

        self.assertEqual(classify_row(row), "weak_candidate")

    def test_classifies_insufficient_data(self):
        row = normalize_row(base_row(evaluated_predictions="5", total_predictions="100", warnings=""))

        self.assertEqual(classify_row(row), "insufficient_data")

    def test_classifies_data_error(self):
        row = normalize_row(base_row(total_predictions="0", evaluated_predictions="0", warnings="historical_data_error: ConnectError"))

        self.assertEqual(classify_row(row), "data_error")

    def test_generates_markdown_report(self):
        summary = analyze_report([normalize_row(base_row())], {"source": "sample.csv"})
        markdown = render_markdown(summary)

        self.assertIn("TRADEAI Research Summary", markdown)
        self.assertIn("BTC 15m", markdown)
        self.assertIn("REJECT", markdown)

    def test_hybrid_scheme_is_named_in_combo_and_recommendations(self):
        row = normalize_row(base_row(
            label_type="hybrid_touch_or_expiry",
            trade_label_scheme="hybrid_touch_or_expiry",
        ))
        summary = analyze_report([row], {"source": "sample.csv"})
        analysis = summary["analyses"][0]

        self.assertIn("hybrid_touch_or_expiry", analysis["combo"])
        self.assertTrue(any("experimental" in item for item in analysis["recommendations"]))

    def test_does_not_fail_with_missing_optional_columns(self):
        row = normalize_row({
            "symbol": "ETH",
            "timeframe": "1h",
            "strategy_mode": "xgboost",
            "total_predictions": "10",
            "evaluated_predictions": "0",
            "warnings": "",
        })
        summary = analyze_report([row], {"source": "minimal.csv"})

        self.assertEqual(summary["analyses"][0]["classification"], "insufficient_data")

    def test_loads_csv_and_writes_outputs_without_api_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "report.csv"
            csv_path.write_text(
                ",".join(base_row().keys()) + "\n" +
                ",".join(f'"{value}"' for value in base_row().values()) + "\n",
                encoding="utf-8",
            )
            rows, metadata = load_report(csv_path=csv_path)
            summary = analyze_report(rows, metadata)
            paths = write_outputs(summary, tmp)

            self.assertTrue(Path(paths["markdown"]).exists())
            self.assertTrue(Path(paths["json"]).exists())

    def test_probability_bucket_parsing(self):
        self.assertEqual(probability_bucket(0.72), "0.7-0.8")

    def test_trade_calibration_flags_high_probability_negative_return(self):
        trades = [
            {
                "symbol": "BTC",
                "timeframe": "15m",
                "strategy_mode": "xgboost",
                "use_trade_labels": "true",
                "trade_label_scheme": "hybrid_touch_or_expiry",
                "label_type": "hybrid_touch_or_expiry",
                "signal": "BUY",
                "probability_buy_win": "0.82",
                "return_pct": "-0.3",
                "outcome": "LOSS",
            }
            for _ in range(6)
        ]

        calibration = analyze_trade_calibration(trades)
        item = next(iter(calibration.values()))

        self.assertTrue(item["calibration_warning"])
        self.assertIn("0.8-0.9", item["high_probability_bad_buckets"])

    def test_render_markdown_includes_calibration_section(self):
        trades = [
            {
                "symbol": "BTC",
                "timeframe": "15m",
                "strategy_mode": "xgboost",
                "use_trade_labels": "true",
                "trade_label_scheme": "hybrid_touch_or_expiry",
                "label_type": "hybrid_touch_or_expiry",
                "signal": "BUY",
                "probability_buy_win": "0.82",
                "return_pct": "-0.3",
                "outcome": "LOSS",
            }
            for _ in range(6)
        ]
        summary = analyze_report([normalize_row(base_row())], {"source": "sample.csv"}, trades=trades)
        markdown = render_markdown(summary)

        self.assertIn("Probability Calibration", markdown)
        self.assertIn("0.8-0.9", markdown)

    def test_loads_trades_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trades.csv"
            path.write_text("symbol,timeframe,signal,return_pct\nBTC,15m,BUY,1.0\n", encoding="utf-8")

            rows = load_trades_report(path)

        self.assertEqual(rows[0]["symbol"], "BTC")


if __name__ == "__main__":
    unittest.main()
