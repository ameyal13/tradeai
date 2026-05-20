import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch

from tools.prediction_journal import PredictionStore, utc_now


def fake_signal(symbol, signal="BUY"):
    return {
        "symbol": symbol.upper(),
        "signal_type": signal,
        "confidence": 72,
        "price_at_signal": 100,
        "entry_price": 100,
        "stop_loss": 95,
        "take_profit": 110,
        "risk_reward_ratio": 2,
        "horizon_minutes": 60,
        "strategy_mode": "deterministic",
        "strategy_name": "deterministic_signal",
        "strategy_version": "v1",
        "reasoning": "mock signal",
        "input_features": {"source": "test"},
        "model_provider": None,
        "model_name": None,
        "model_used": "deterministic_signal",
    }


class GenerateSignalsOnceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = PredictionStore(file_path=Path(self.tmp.name) / "journal.json")

    async def asyncTearDown(self):
        self.tmp.cleanup()

    async def test_generates_signals_for_multiple_symbols_with_mocked_function(self):
        import scripts.generate_signals_once as script

        async def generate(symbol, *args, **kwargs):
            return fake_signal(symbol)

        with patch.object(script, "generate_trading_signal", new=AsyncMock(side_effect=generate)):
            rows = await script.generate_signals_once(symbols=["BTC", "ETH"], store=self.store)

        self.assertEqual(len(rows), 2)
        self.assertEqual(len(self.store.list_predictions(status="pending")), 2)
        self.assertEqual({row["symbol"] for row in rows}, {"BTC", "ETH"})
        self.assertTrue(all(row["status"] == "pending" for row in rows))

    async def test_avoids_recent_pending_duplicates(self):
        import scripts.generate_signals_once as script

        self.store.create_prediction({
            "symbol": "BTC",
            "timeframe": "1h",
            "strategy_mode": "deterministic",
            "strategy_name": "existing",
            "strategy_version": "v1",
            "signal": "BUY",
            "confidence": 60,
            "entry_price": 100,
            "horizon_minutes": 60,
            "created_at": (utc_now() - timedelta(minutes=10)).isoformat(),
        })

        with patch.object(script, "generate_trading_signal", new=AsyncMock(return_value=fake_signal("BTC"))) as mocked:
            rows = await script.generate_signals_once(symbols=["BTC"], store=self.store)

        self.assertEqual(rows[0]["status"], "skipped_recent_pending")
        self.assertEqual(len(self.store.list_predictions(status="pending")), 1)
        mocked.assert_not_called()

    async def test_continues_when_one_symbol_fails(self):
        import scripts.generate_signals_once as script

        async def generate(symbol, *args, **kwargs):
            if symbol == "ETH":
                raise RuntimeError("market data unavailable")
            return fake_signal(symbol)

        with patch.object(script, "generate_trading_signal", new=AsyncMock(side_effect=generate)):
            rows = await script.generate_signals_once(symbols=["BTC", "ETH", "SOL"], store=self.store)

        by_symbol = {row["symbol"]: row for row in rows}
        self.assertEqual(by_symbol["ETH"]["status"], "error")
        self.assertEqual(by_symbol["BTC"]["status"], "pending")
        self.assertEqual(by_symbol["SOL"]["status"], "pending")
        self.assertEqual(len(self.store.list_predictions(status="pending")), 2)

    async def test_saved_signals_have_pending_status(self):
        import scripts.generate_signals_once as script

        with patch.object(script, "generate_trading_signal", new=AsyncMock(return_value=fake_signal("BTC"))):
            rows = await script.generate_signals_once(symbols=["BTC"], store=self.store)

        prediction = self.store.get_prediction(rows[0]["prediction_id"])
        self.assertEqual(prediction["status"], "pending")
        self.assertEqual(rows[0]["status"], "pending")


if __name__ == "__main__":
    unittest.main()
