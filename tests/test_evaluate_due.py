import importlib.util
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pandas as pd

from tools.prediction_journal import PENDING, PredictionStore, utc_now


def future_candles(created_at, high=111, low=99, close=108):
    return pd.DataFrame([{
        "timestamp": (created_at + timedelta(minutes=60)).isoformat(),
        "open": 100,
        "high": high,
        "low": low,
        "close": close,
        "volume": 1000,
    }])


@unittest.skipIf(importlib.util.find_spec("fastapi") is None, "FastAPI dependencies are not installed")
class EvaluateDueTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        import main

        self.tmp = tempfile.TemporaryDirectory()
        main.supabase = None
        main.prediction_store = PredictionStore(file_path=Path(self.tmp.name) / "journal.json")
        self.main = main

    async def asyncTearDown(self):
        self.tmp.cleanup()

    def create_prediction(self, created_at, **overrides):
        payload = {
            "symbol": "BTC",
            "timeframe": "1h",
            "strategy_mode": "deterministic",
            "strategy_name": "due_test",
            "strategy_version": "v1",
            "signal": "BUY",
            "confidence": 70,
            "entry_price": 100,
            "stop_loss": 95,
            "take_profit": 110,
            "risk_reward_ratio": 2,
            "horizon_minutes": 60,
            "input_features": {"test": "evaluate_due"},
            "reasoning": "test",
            "status": PENDING,
            "created_at": created_at.isoformat(),
        }
        payload.update(overrides)
        return self.main.prediction_store.create_prediction(payload)

    async def test_pending_prediction_not_due_yet(self):
        created_at = utc_now() - timedelta(minutes=10)
        prediction = self.create_prediction(created_at)

        result = await self.main.evaluate_due_predictions_once(now=utc_now())

        self.assertEqual(result["found"], 1)
        self.assertEqual(result["not_due"], 1)
        self.assertEqual(result["evaluated"], 0)
        self.assertEqual(result["invalid"], 0)
        stored = self.main.prediction_store.get_prediction(prediction["id"])
        self.assertEqual(stored["status"], "pending")
        self.assertEqual(self.main.prediction_store.list_outcomes(), [])

    async def test_pending_prediction_due_is_evaluated_and_outcome_created(self):
        created_at = utc_now() - timedelta(minutes=120)
        prediction = self.create_prediction(created_at)

        with patch.object(self.main, "fetch_future_klines", new=AsyncMock(return_value=future_candles(created_at))):
            result = await self.main.evaluate_due_predictions_once(now=utc_now())

        self.assertEqual(result["found"], 1)
        self.assertEqual(result["not_due"], 0)
        self.assertEqual(result["evaluated"], 1)
        self.assertEqual(result["invalid"], 0)
        stored = self.main.prediction_store.get_prediction(prediction["id"])
        self.assertEqual(stored["status"], "evaluated")
        outcomes = self.main.prediction_store.list_outcomes()
        self.assertEqual(len(outcomes), 1)
        self.assertEqual(outcomes[0]["prediction_id"], prediction["id"])
        self.assertEqual(outcomes[0]["outcome"], "WIN")

    async def test_due_prediction_invalid_when_future_data_is_insufficient(self):
        created_at = utc_now() - timedelta(minutes=120)
        prediction = self.create_prediction(created_at)
        empty_candles = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        with patch.object(self.main, "fetch_future_klines", new=AsyncMock(return_value=empty_candles)):
            result = await self.main.evaluate_due_predictions_once(now=utc_now())

        self.assertEqual(result["found"], 1)
        self.assertEqual(result["evaluated"], 0)
        self.assertEqual(result["invalid"], 1)
        stored = self.main.prediction_store.get_prediction(prediction["id"])
        self.assertEqual(stored["status"], "invalid")
        outcomes = self.main.prediction_store.list_outcomes()
        self.assertEqual(len(outcomes), 1)
        self.assertEqual(outcomes[0]["outcome"], "INVALID_DATA")


if __name__ == "__main__":
    unittest.main()
