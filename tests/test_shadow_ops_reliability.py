import json
import os
import tempfile
import unittest
from datetime import timedelta
from io import StringIO
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pandas as pd

from scripts.evaluate_shadow_signals_once import evaluate_shadow_signals_once
from scripts.print_shadow_open_signals import open_signal_rows
from scripts.run_shadow_ops_once import run_shadow_ops_once
from scripts.shadow_ops_healthcheck import build_healthcheck_report, build_parser as build_healthcheck_parser, main as healthcheck_main, print_healthcheck
from tools.prediction_journal import utc_now
from tools.runtime_env import load_project_env
from tools.shadow_signal_journal import OPEN, ShadowSignalJournal


def write_registry(path: Path, count: int = 64):
    rows = []
    for index in range(count):
        rows.append({
            "config_id": f"cfg{index}",
            "status": "completed",
            "classification": "unstable_watchlist",
            "config": {"symbol": "ADA", "timeframe": "1h"},
        })
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def open_signal(signal_id="sig1", generated=None, expires=None):
    generated = generated or utc_now() - timedelta(hours=2)
    expires = expires or utc_now() + timedelta(hours=2)
    return {
        "shadow_signal_id": signal_id,
        "config_id": "cfg1",
        "symbol": "ADA",
        "timeframe": "1h",
        "strategy_mode": "xgboost",
        "strategy_name": "xgboost_temporal_split",
        "strategy_version": "v1",
        "side": "SHORT",
        "entry_price": 1.0,
        "stop_loss": 1.1,
        "take_profit": 0.8,
        "risk_reward": 2.0,
        "horizon_candles": 4,
        "horizon_minutes": 240,
        "confidence": 60,
        "generated_at": generated.isoformat(),
        "expires_at": expires.isoformat(),
        "status": OPEN,
        "commission_pct": 0.001,
        "slippage_pct": 0.0005,
        "spread_pct": 0.0003,
        "research_only": True,
        "agent_review": {"review_status": "APPROVE", "risk_flags": []},
    }


def future_candles(start):
    idx = pd.date_range(start, periods=3, freq="1h", tz="UTC")
    return pd.DataFrame({
        "timestamp": idx,
        "open": [1.0, 1.0, 1.0],
        "high": [1.02, 1.03, 1.04],
        "low": [0.79, 0.78, 0.77],
        "close": [0.8, 0.8, 0.8],
        "volume": [1000.0, 1000.0, 1000.0],
    })


class ShadowOpsHealthcheckTests(unittest.IsolatedAsyncioTestCase):
    def test_load_project_env_uses_override_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".env").write_text("TRADEAI_TEST_VALUE=from_env_file\n", encoding="utf-8")
            with patch("tools.runtime_env.project_root", return_value=root):
                with patch.dict(os.environ, {"TRADEAI_TEST_VALUE": "from_process"}, clear=False):
                    loaded = load_project_env()
                    value = os.getenv("TRADEAI_TEST_VALUE")

        self.assertTrue(loaded)
        self.assertEqual(value, "from_process")

    def test_healthcheck_parser_accepts_test_telegram(self):
        args = build_healthcheck_parser().parse_args(["--test-telegram"])

        self.assertTrue(args.test_telegram)

    async def test_healthcheck_test_telegram_missing_env_prints_no_secret(self):
        report = {
            "health_status": "HEALTH_WARNING",
            "warnings": [],
            "blockers": [],
            "shadow_journal": {"path": "journal", "exists": False, "open": 0, "closed": 0, "total": 0, "overdue_open": 0},
            "telegram": {"configured": False},
            "crypto_multi_registry": {"completed": 0, "total_latest": 0},
            "market_connectivity": {"ok": True},
            "git": {"pending": False},
            "use_news_context_flag_available": True,
        }
        stream = StringIO()
        with patch("scripts.shadow_ops_healthcheck.load_project_env"):
            with patch("scripts.shadow_ops_healthcheck.build_healthcheck_report", new=AsyncMock(return_value=report)):
                with patch("scripts.shadow_ops_healthcheck.telegram_enabled", return_value=False):
                    with patch("sys.argv", ["shadow_ops_healthcheck.py", "--test-telegram"]):
                        with patch("sys.stdout", stream):
                            await healthcheck_main()

        output = stream.getvalue()
        self.assertIn("Telegram test: missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID", output)
        self.assertNotIn("secret", output.lower())

    async def test_healthcheck_detects_telegram_config_without_printing_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            registry = tmp_path / "crypto_multi_registry.jsonl"
            journal_path = tmp_path / "shadow.jsonl"
            write_registry(registry)
            ShadowSignalJournal(journal_path).create_signal(open_signal())
            with patch("scripts.shadow_ops_healthcheck._check_market_connectivity", new=AsyncMock(return_value={"ok": True, "rows": 1, "error": None, "category": None})):
                with patch("scripts.shadow_ops_healthcheck._git_pending_changes", return_value={"ok": True, "pending": False, "pending_count": 0, "error": None}):
                    with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "secret-token", "TELEGRAM_CHAT_ID": "secret-chat"}, clear=True):
                        report = await build_healthcheck_report(journal_path=journal_path, registry_path=registry, lock_path=tmp_path / "lock")
                        stream = StringIO()
                        with patch("sys.stdout", stream):
                            print_healthcheck(report)

        output = stream.getvalue()
        self.assertTrue(report["telegram"]["configured"])
        self.assertNotIn("secret-token", output)
        self.assertNotIn("secret-chat", output)

    async def test_healthcheck_detects_missing_journal_and_crypto_multi_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            registry = tmp_path / "crypto_multi_registry.jsonl"
            write_registry(registry)
            with patch("scripts.shadow_ops_healthcheck._check_market_connectivity", new=AsyncMock(return_value={"ok": True, "rows": 1, "error": None, "category": None})):
                with patch("scripts.shadow_ops_healthcheck._git_pending_changes", return_value={"ok": True, "pending": False, "pending_count": 0, "error": None}):
                    report = await build_healthcheck_report(journal_path=tmp_path / "missing.jsonl", registry_path=registry, lock_path=tmp_path / "lock")

        self.assertIn("shadow_journal_missing", report["warnings"])
        self.assertEqual(report["crypto_multi_registry"]["completed"], 64)

    async def test_healthcheck_detects_open_due_signal_and_news_context_failure_safe(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            registry = tmp_path / "crypto_multi_registry.jsonl"
            journal_path = tmp_path / "shadow.jsonl"
            write_registry(registry)
            ShadowSignalJournal(journal_path).create_signal(open_signal(expires=utc_now() - timedelta(hours=1)))
            with patch("scripts.shadow_ops_healthcheck._check_market_connectivity", new=AsyncMock(return_value={"ok": True, "rows": 1, "error": None, "category": None})):
                with patch("scripts.shadow_ops_healthcheck._git_pending_changes", return_value={"ok": True, "pending": False, "pending_count": 0, "error": None}):
                    with patch("research.news_context_engine.build_news_context", new=AsyncMock(side_effect=RuntimeError("network down"))):
                        report = await build_healthcheck_report(
                            journal_path=journal_path,
                            registry_path=registry,
                            lock_path=tmp_path / "lock",
                            check_news_context=True,
                        )

        self.assertIn("open_signals_due_for_evaluation", report["warnings"])
        self.assertEqual(report["news_context"]["provider_status"], "error:RuntimeError")


class ShadowOpsEvaluationTests(unittest.IsolatedAsyncioTestCase):
    async def test_evaluate_retry_keeps_signal_open_on_network_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal_path = Path(tmp) / "shadow.jsonl"
            journal = ShadowSignalJournal(journal_path)
            journal.create_signal(open_signal(expires=utc_now() - timedelta(hours=1)))
            with patch("scripts.evaluate_shadow_signals_once.fetch_future_klines", new=AsyncMock(side_effect=RuntimeError("network down"))) as fetch:
                result = await evaluate_shadow_signals_once(journal_path=journal_path, retries=1, backoff_seconds=0)

            self.assertEqual(fetch.await_count, 2)
            self.assertEqual(result["closed"], 0)
            self.assertEqual(result["still_open"], 1)
            self.assertEqual(result["errors"][0]["error_code"], "network_error_retry_later")
            self.assertEqual(len(ShadowSignalJournal(journal_path).list_signals(status=OPEN)), 1)

    async def test_telegram_failure_does_not_break_evaluation(self):
        generated = utc_now() - timedelta(hours=5)
        with tempfile.TemporaryDirectory() as tmp:
            journal_path = Path(tmp) / "shadow.jsonl"
            journal = ShadowSignalJournal(journal_path)
            journal.create_signal(open_signal(generated=generated, expires=generated + timedelta(hours=3)))
            with patch("scripts.evaluate_shadow_signals_once.fetch_future_klines", new=AsyncMock(return_value=future_candles(generated))):
                with patch("scripts.evaluate_shadow_signals_once.send_telegram_message", side_effect=RuntimeError("telegram down")):
                    result = await evaluate_shadow_signals_once(journal_path=journal_path, notify_telegram=True, retries=0)

        self.assertEqual(result["closed"], 1)
        self.assertEqual(len(result["telegram_errors"]), 1)


class ShadowOpsCycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_shadow_ops_does_not_open_new_when_open_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal_path = Path(tmp) / "shadow.jsonl"
            journal = ShadowSignalJournal(journal_path)
            journal.create_signal(open_signal())
            with patch("scripts.run_shadow_ops_once.build_healthcheck_report", new=AsyncMock(return_value={
                "health_status": "HEALTH_WARNING",
                "shadow_journal": {"open": 1},
            })):
                with patch("scripts.run_shadow_ops_once.evaluate_shadow_signals_once", new=AsyncMock(return_value={
                    "found_open": 1,
                    "closed": 0,
                    "still_open": 1,
                    "errors": [],
                    "closed_signals": [],
                })):
                    with patch("scripts.run_shadow_ops_once.run_shadow_cycle_once", new=AsyncMock()) as cycle:
                        result = await run_shadow_ops_once(journal_path=journal_path, reports_output_dir=Path(tmp) / "reports")

        cycle.assert_not_awaited()
        self.assertEqual(result["generation_skipped_reason"], "open_signals_exist")

    async def test_run_shadow_ops_opens_max_one_when_no_open(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal_path = Path(tmp) / "shadow.jsonl"
            with patch("scripts.run_shadow_ops_once.build_healthcheck_report", new=AsyncMock(return_value={
                "health_status": "HEALTH_OK",
                "shadow_journal": {"open": 0},
            })):
                with patch("scripts.run_shadow_ops_once.evaluate_shadow_signals_once", new=AsyncMock(return_value={
                    "found_open": 0,
                    "closed": 0,
                    "still_open": 0,
                    "errors": [],
                    "closed_signals": [],
                })):
                    with patch("scripts.run_shadow_ops_once.run_shadow_cycle_once", new=AsyncMock(return_value={
                        "generation_summary": {"opened_signals": 1, "configs_scanned": 5},
                    })) as cycle:
                        await run_shadow_ops_once(journal_path=journal_path, reports_output_dir=Path(tmp) / "reports", max_signals=1)

        self.assertEqual(cycle.await_args.kwargs["max_signals"], 1)

    async def test_run_shadow_ops_propagates_news_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal_path = Path(tmp) / "shadow.jsonl"
            with patch("scripts.run_shadow_ops_once.build_healthcheck_report", new=AsyncMock(return_value={
                "health_status": "HEALTH_OK",
                "shadow_journal": {"open": 0},
            })):
                with patch("scripts.run_shadow_ops_once.evaluate_shadow_signals_once", new=AsyncMock(return_value={
                    "found_open": 0,
                    "closed": 0,
                    "still_open": 0,
                    "errors": [],
                    "closed_signals": [],
                })):
                    with patch("scripts.run_shadow_ops_once.run_shadow_cycle_once", new=AsyncMock(return_value={
                        "generation_summary": {"opened_signals": 0, "configs_scanned": 1},
                    })) as cycle:
                        await run_shadow_ops_once(
                            journal_path=journal_path,
                            reports_output_dir=Path(tmp) / "reports",
                            use_news_context=True,
                            dry_run=True,
                        )

        self.assertTrue(cycle.await_args.kwargs["use_news_context"])

    async def test_dry_run_does_not_write_journal(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal_path = Path(tmp) / "shadow.jsonl"
            with patch("scripts.run_shadow_ops_once.build_healthcheck_report", new=AsyncMock(return_value={
                "health_status": "HEALTH_OK",
                "shadow_journal": {"open": 0},
            })):
                with patch("scripts.run_shadow_ops_once.run_shadow_cycle_once", new=AsyncMock(return_value={
                    "generation_summary": {"opened_signals": 0, "configs_scanned": 1},
                })):
                    await run_shadow_ops_once(journal_path=journal_path, reports_output_dir=Path(tmp) / "reports", dry_run=True)

        self.assertFalse(journal_path.exists())

    def test_print_open_signals_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal_path = Path(tmp) / "shadow.jsonl"
            ShadowSignalJournal(journal_path).create_signal(open_signal())

            rows = open_signal_rows(journal_path)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["symbol"], "ADA")


if __name__ == "__main__":
    unittest.main()
