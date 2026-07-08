import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np
import pandas as pd

from tools.prediction_journal import PredictionStore
from tools.research_result_repository import ResearchResultRepository
from tools.shadow_signal_journal import ShadowSignalJournal
from tools.shadow_signal_repository import ShadowSignalRepository


def candles(rows=140):
    idx = pd.date_range("2026-01-01", periods=rows, freq="h", tz="UTC")
    base = np.linspace(100, 125, rows) + np.sin(np.arange(rows) / 4)
    return [
        {
            "timestamp": ts.isoformat(),
            "open": float(price),
            "high": float(price + 2),
            "low": float(price - 2),
            "close": float(price + 0.4),
            "volume": 1000.0,
        }
        for ts, price in zip(idx, base)
    ]


@unittest.skipIf(importlib.util.find_spec("fastapi") is None, "FastAPI dependencies are not installed in this Python environment")
class ApiSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from fastapi.testclient import TestClient
            import main
        except Exception as exc:
            raise unittest.SkipTest(f"API dependencies unavailable: {exc}")

        cls.tmp = tempfile.TemporaryDirectory()
        main.supabase = None
        main.prediction_store = PredictionStore(file_path=Path(cls.tmp.name) / "journal.json")
        main.shadow_signal_repo = ShadowSignalRepository(journal_path=Path(cls.tmp.name) / "shadow.jsonl")
        main.research_result_repo = ResearchResultRepository(registry_path=Path(cls.tmp.name) / "missing_registry.jsonl")
        cls.main = main
        cls.client = TestClient(main.app)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_predictions_create_list_evaluate_and_metrics(self):
        create = self.client.post("/predictions/create", json={
            "symbol": "BTC",
            "timeframe": "1h",
            "strategy_mode": "deterministic",
            "strategy_name": "smoke",
            "strategy_version": "v1",
            "signal": "BUY",
            "confidence": 70,
            "entry_price": 100,
            "stop_loss": 95,
            "take_profit": 110,
            "risk_reward_ratio": 2,
            "horizon_minutes": 60,
            "input_features": {"source": "smoke"},
            "reasoning": "smoke test",
            "created_at": "2026-01-01T00:00:00+00:00",
        })
        self.assertEqual(create.status_code, 200)
        prediction_id = create.json()["data"]["id"]

        listed = self.client.get("/predictions")
        self.assertEqual(listed.status_code, 200)
        self.assertGreaterEqual(len(listed.json()["data"]), 1)

        evaluated = self.client.post("/predictions/evaluate", json={
            "prediction_id": prediction_id,
            "commission_pct": 0,
            "slippage_pct": 0,
            "spread_pct": 0,
            "candles": [{
                "timestamp": "2026-01-01T01:00:00+00:00",
                "open": 100,
                "high": 111,
                "low": 99,
                "close": 108,
                "volume": 1000,
            }],
        })
        self.assertEqual(evaluated.status_code, 200)
        self.assertEqual(evaluated.json()["data"]["outcome"], "WIN")

        metrics = self.client.get("/metrics/signals")
        self.assertEqual(metrics.status_code, 200)
        self.assertTrue(metrics.json()["data"])

    def test_backtest_run_without_supabase_is_controlled(self):
        fake = {
            "engine_version": "v2",
            "total_return_pct": 0,
            "trades": [],
            "equity_curve": [],
        }
        with patch.object(self.main, "run_backtest", new=AsyncMock(return_value=fake)):
            response = self.client.post("/backtest/run", json={
                "symbol": "BTC",
                "strategy": {"entry_conditions": []},
                "date_from": "2026-01-01",
                "date_to": "2026-01-02",
                "initial_capital": 1000,
                "timeframe": "1h",
            })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "DONE")
        self.assertEqual(response.json()["persistence"], "none")

    def test_replay_run_with_manual_candles(self):
        response = self.client.post("/replay/run", json={
            "symbol": "BTC",
            "interval": "1h",
            "strategy_mode": "deterministic",
            "candles": candles(90),
            "horizon_candles": 3,
            "min_history": 30,
            "max_predictions": 3,
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn("predictions", response.json()["data"])

    def test_shadow_read_only_endpoints_use_local_fallback(self):
        journal_path = Path(self.tmp.name) / "shadow.jsonl"
        journal = ShadowSignalJournal(journal_path)
        journal.append({
            "shadow_signal_id": "sig-smoke",
            "config_id": "cfg-smoke",
            "source_registry": "crypto_multi_registry",
            "classification": "unstable_watchlist",
            "symbol": "ADA",
            "timeframe": "1h",
            "strategy_mode": "xgboost",
            "side": "LONG",
            "entry_price": 1.0,
            "stop_loss": 0.95,
            "take_profit": 1.1,
            "risk_reward": 2.0,
            "horizon_candles": 12,
            "horizon_minutes": 720,
            "confidence": 60.0,
            "generated_at": "2026-01-01T00:00:00+00:00",
            "expires_at": "2026-01-01T12:00:00+00:00",
            "status": "OPEN",
            "research_only": True,
        })

        health = self.client.get("/shadow/health")
        signals = self.client.get("/shadow/signals?status=OPEN&symbol=ADA")
        summary = self.client.get("/shadow/summary")
        config_health = self.client.get("/shadow/config-health")

        self.assertEqual(health.status_code, 200)
        self.assertEqual(signals.status_code, 200)
        self.assertEqual(summary.status_code, 200)
        self.assertEqual(config_health.status_code, 200)
        self.assertTrue(health.json()["research_only"])
        self.assertEqual(signals.json()["count"], 1)
        self.assertEqual(summary.json()["data"]["summary"]["open"], 1)
        self.assertTrue(config_health.json()["research_only"])
        self.assertFalse(config_health.json()["auto_quarantine_enabled"])

    def test_research_summary_endpoint_is_read_only(self):
        response = self.client.get("/research/summary")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["research_only"])
        self.assertTrue(payload["test_not_used_for_selection"])
        self.assertEqual(payload["data"]["summary"]["total_configs"], 0)

    def test_optimizer_run_with_manual_candles(self):
        response = self.client.post("/optimizer/run", json={
            "symbol": "BTC",
            "interval": "1h",
            "strategy_mode": "deterministic",
            "candles": candles(140),
            "train_size": 80,
            "validation_size": 40,
            "horizon_candles": 3,
            "min_history": 20,
            "step_size": 10,
            "parameter_grid": {"rsi_buy_threshold": [30], "rsi_sell_threshold": [70]},
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn("baseline_params", response.json()["data"])


if __name__ == "__main__":
    unittest.main()
