from __future__ import annotations

import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch

from scripts.evaluate_shadow_signals_once import build_parser, evaluate_shadow_signals_once
from scripts.generate_shadow_signals_once import load_shadow_generation_candles_with_retry
from tools.prediction_journal import utc_now
from tools.shadow_signal_journal import EXPIRED, OPEN, ShadowSignalJournal


class FakeResponse:
    status_code = 400


class FakeHTTPStatusError(Exception):
    def __init__(self, message: str = "HTTP 400 from Binance"):
        super().__init__(message)
        self.response = FakeResponse()


def open_signal(signal_id: str = "sig-http-error") -> dict:
    generated = utc_now() - timedelta(hours=3)
    return {
        "shadow_signal_id": signal_id,
        "config_id": "cfg1",
        "source_registry": "crypto_multi",
        "classification": "unstable_watchlist",
        "symbol": "SOL",
        "timeframe": "1h",
        "strategy_mode": "xgboost",
        "side": "LONG",
        "entry_price": 100.0,
        "stop_loss": 98.0,
        "take_profit": 104.0,
        "risk_reward": 2.0,
        "horizon_candles": 2,
        "horizon_minutes": 120,
        "confidence": 60.0,
        "generated_at": generated.isoformat(),
        "expires_at": (generated + timedelta(hours=2)).isoformat(),
        "status": OPEN,
        "outcome": None,
        "commission_pct": 0.001,
        "slippage_pct": 0.0005,
        "spread_pct": 0.0003,
        "research_only": True,
    }


class EvaluateShadowSignalsTests(unittest.IsolatedAsyncioTestCase):
    async def test_endpoint_http_error_retries_three_attempts_then_expires_signal(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal_path = Path(tmp) / "shadow.jsonl"
            journal = ShadowSignalJournal(journal_path)
            journal.create_signal(open_signal())

            with patch(
                "scripts.evaluate_shadow_signals_once.fetch_future_klines",
                new=AsyncMock(side_effect=FakeHTTPStatusError()),
            ) as fetch:
                result = await evaluate_shadow_signals_once(
                    journal_path=journal_path,
                    retries=2,
                    backoff_seconds=0,
                )

            latest = ShadowSignalJournal(journal_path).latest_by_id()["sig-http-error"]

        self.assertEqual(fetch.await_count, 3)
        self.assertEqual(result["closed"], 1)
        self.assertEqual(result["still_open"], 0)
        self.assertEqual(result["errors"][0]["error_category"], "endpoint_http_error")
        self.assertFalse(result["errors"][0]["signal_left_open"])
        self.assertEqual(result["errors"][0]["expiry_reason"], "evaluation_http_error")
        self.assertEqual(latest["status"], EXPIRED)
        self.assertEqual(latest["outcome"], "EXPIRED")
        self.assertEqual(latest["exit_reason"], "evaluation_http_error")
        self.assertEqual(latest["expiry_reason"], "evaluation_http_error")
        self.assertEqual(latest["evaluation_raw_path"]["expiry_reason"], "evaluation_http_error")

    async def test_non_http_network_error_still_left_open_after_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal_path = Path(tmp) / "shadow.jsonl"
            journal = ShadowSignalJournal(journal_path)
            journal.create_signal(open_signal())

            with patch(
                "scripts.evaluate_shadow_signals_once.fetch_future_klines",
                new=AsyncMock(side_effect=RuntimeError("network down")),
            ) as fetch:
                result = await evaluate_shadow_signals_once(
                    journal_path=journal_path,
                    retries=1,
                    backoff_seconds=0,
                )

            open_rows = ShadowSignalJournal(journal_path).list_signals(status=OPEN)

        self.assertEqual(fetch.await_count, 2)
        self.assertEqual(result["closed"], 0)
        self.assertEqual(result["still_open"], 1)
        self.assertTrue(result["errors"][0]["signal_left_open"])
        self.assertEqual(len(open_rows), 1)


class EvaluateShadowSignalsCliTests(unittest.TestCase):
    def test_cli_defaults_to_three_total_attempts_and_five_second_wait(self):
        args = build_parser().parse_args([])

        self.assertEqual(args.retries, 2)
        self.assertEqual(args.backoff_seconds, 5.0)


class GenerateShadowPriceFetchRetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_shadow_generation_price_fetch_retries_three_attempts(self):
        loaded = {"candles": ["ok"], "data_source": "network"}

        with patch(
            "scripts.generate_shadow_signals_once.load_experiment_candles",
            new=AsyncMock(side_effect=[TimeoutError("timeout"), TimeoutError("timeout"), loaded]),
        ) as fetch:
            with patch("scripts.generate_shadow_signals_once.asyncio.sleep", new=AsyncMock()) as sleep:
                result = await load_shadow_generation_candles_with_retry(
                    "SOL",
                    "1h",
                    max_candles=5000,
                    refresh_cache=True,
                    retries=2,
                    backoff_seconds=5,
                )

        self.assertEqual(fetch.await_count, 3)
        self.assertEqual(sleep.await_count, 2)
        self.assertEqual(result["fetch_attempts"], 3)

    async def test_shadow_generation_price_fetch_raises_after_three_retryable_failures(self):
        with patch(
            "scripts.generate_shadow_signals_once.load_experiment_candles",
            new=AsyncMock(side_effect=TimeoutError("timeout")),
        ) as fetch:
            with patch("scripts.generate_shadow_signals_once.asyncio.sleep", new=AsyncMock()) as sleep:
                with self.assertRaises(TimeoutError):
                    await load_shadow_generation_candles_with_retry(
                        "SOL",
                        "1h",
                        max_candles=5000,
                        refresh_cache=True,
                        retries=2,
                        backoff_seconds=5,
                    )

        self.assertEqual(fetch.await_count, 3)
        self.assertEqual(sleep.await_count, 2)


if __name__ == "__main__":
    unittest.main()
