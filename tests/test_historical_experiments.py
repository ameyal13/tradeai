import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np
import pandas as pd


def synthetic_candles(rows=180):
    idx = pd.date_range("2026-01-01", periods=rows, freq="15min", tz="UTC")
    base = np.linspace(100, 120, rows) + np.sin(np.arange(rows) / 4)
    return pd.DataFrame({
        "timestamp": idx,
        "open": base,
        "high": base + 2,
        "low": base - 2,
        "close": base + 0.5,
        "volume": 1000,
    })


class HistoricalExperimentTests(unittest.IsolatedAsyncioTestCase):
    async def test_generates_report_with_synthetic_data(self):
        import scripts.run_historical_experiments as script

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(script, "fetch_binance_klines", new=AsyncMock(return_value=synthetic_candles())):
                report = await script.run_experiments(
                    symbols=["BTC"],
                    timeframes=["15m"],
                    strategy_modes=["deterministic"],
                    max_predictions=5,
                    reports_dir=tmp,
                )

            self.assertEqual(len(report["summary"]), 1)
            self.assertTrue(Path(report["report_paths"]["json"]).exists())
            self.assertTrue(Path(report["report_paths"]["csv"]).exists())

    async def test_does_not_persist_by_default(self):
        import scripts.run_historical_experiments as script

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(script, "PredictionStore") as store_cls:
                with patch.object(script, "fetch_binance_klines", new=AsyncMock(return_value=synthetic_candles())):
                    await script.run_experiments(
                        symbols=["BTC"],
                        timeframes=["15m"],
                        strategy_modes=["deterministic"],
                        max_predictions=3,
                        persist=False,
                        reports_dir=tmp,
                    )

        store_cls.assert_not_called()

    async def test_continues_if_one_symbol_fails(self):
        import scripts.run_historical_experiments as script

        async def fetch(symbol, *args, **kwargs):
            if symbol == "ETH":
                raise RuntimeError("network unavailable")
            return synthetic_candles()

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(script, "fetch_binance_klines", new=AsyncMock(side_effect=fetch)):
                report = await script.run_experiments(
                    symbols=["BTC", "ETH"],
                    timeframes=["15m"],
                    strategy_modes=["deterministic"],
                    max_predictions=3,
                    reports_dir=tmp,
                )

        by_symbol = {row["symbol"]: row for row in report["summary"]}
        self.assertIn("historical_data_error", by_symbol["ETH"]["warnings"])
        self.assertGreaterEqual(by_symbol["BTC"]["total_predictions"], 1)

    async def test_respects_max_predictions(self):
        import scripts.run_historical_experiments as script

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(script, "fetch_binance_klines", new=AsyncMock(return_value=synthetic_candles())):
                report = await script.run_experiments(
                    symbols=["BTC"],
                    timeframes=["15m"],
                    strategy_modes=["deterministic"],
                    max_predictions=4,
                    reports_dir=tmp,
                )

        self.assertLessEqual(report["summary"][0]["total_predictions"], 4)

    async def test_respects_symbols_and_timeframes(self):
        import scripts.run_historical_experiments as script

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(script, "fetch_binance_klines", new=AsyncMock(return_value=synthetic_candles())):
                report = await script.run_experiments(
                    symbols=["BTC", "SOL"],
                    timeframes=["15m", "1h"],
                    strategy_modes=["deterministic"],
                    max_predictions=2,
                    reports_dir=tmp,
                )

        combos = {(row["symbol"], row["timeframe"]) for row in report["summary"]}
        self.assertEqual(combos, {("BTC", "15m"), ("BTC", "1h"), ("SOL", "15m"), ("SOL", "1h")})

    async def test_summary_includes_minimum_metrics(self):
        import scripts.run_historical_experiments as script

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(script, "fetch_binance_klines", new=AsyncMock(return_value=synthetic_candles())):
                report = await script.run_experiments(
                    symbols=["BTC"],
                    timeframes=["15m"],
                    strategy_modes=["deterministic"],
                    max_predictions=2,
                    reports_dir=tmp,
                )

        row = report["summary"][0]
        for key in [
            "symbol", "timeframe", "strategy_mode", "total_predictions",
            "evaluated_predictions", "win_rate", "average_return",
            "total_return_pct", "profit_factor", "max_drawdown", "sharpe",
            "invalid_count", "warnings",
        ]:
            self.assertIn(key, row)


if __name__ == "__main__":
    unittest.main()
