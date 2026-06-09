import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from scripts.run_shadow_cycle_once import (
    ShadowCycleLocked,
    build_parser as build_shadow_cycle_parser,
    default_journal_path,
    default_shadow_reports_dir,
    registry_path_for_choice,
    run_shadow_cycle_once,
    shadow_cycle_lock,
)
from scripts.summarize_shadow_signals import (
    build_shadow_summary,
    max_drawdown_pct,
    summarize_rows,
    summarize_shadow_signals,
)
from tools.shadow_signal_journal import ShadowSignalJournal


def shadow_row(signal_id="sig1", status="CLOSED", outcome="WIN", pnl=1.0, symbol="SOL", config_id="cfg1"):
    return {
        "shadow_signal_id": signal_id,
        "config_id": config_id,
        "source_registry": "refined_registry",
        "classification": "unstable_watchlist",
        "symbol": symbol,
        "timeframe": "1h",
        "strategy_mode": "xgboost",
        "side": "LONG",
        "entry_price": 100.0,
        "stop_loss": 95.0,
        "take_profit": 110.0,
        "risk_reward": 2.0,
        "horizon_candles": 12,
        "horizon_minutes": 720,
        "confidence": 60.0,
        "generated_at": "2026-01-01T00:00:00+00:00",
        "expires_at": "2026-01-01T12:00:00+00:00",
        "status": status,
        "outcome": outcome,
        "exit_price": 110.0,
        "exit_reason": "take_profit",
        "pnl_pct": pnl,
        "fees": 0.1,
        "slippage": 0.05,
        "spread": 0.0001,
        "research_only": True,
    }


class ShadowCycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_cycle_evaluates_before_generating(self):
        events = []

        async def fake_eval(*args, **kwargs):
            events.append("evaluate")
            return {"found_open": 1, "closed": 1, "still_open": 0, "errors": [], "closed_signals": []}

        async def fake_generate(*args, **kwargs):
            events.append("generate")
            return [{"status": "OPEN", "config_id": "cfg1"}]

        def fake_summary(*args, **kwargs):
            events.append("summary")
            return {"summary": {"open": 1, "closed": 1}, "json_path": "summary.json", "markdown_path": "summary.md", "signals": []}

        with tempfile.TemporaryDirectory() as tmp:
            with patch("scripts.run_shadow_cycle_once.evaluate_shadow_signals_once", new=AsyncMock(side_effect=fake_eval)):
                with patch("scripts.run_shadow_cycle_once.generate_shadow_signals_once", new=AsyncMock(side_effect=fake_generate)):
                    with patch("scripts.run_shadow_cycle_once.summarize_shadow_signals", side_effect=fake_summary):
                        result = await run_shadow_cycle_once(
                            registry=str(Path(tmp) / "registry.jsonl"),
                            journal_path=Path(tmp) / "journal.jsonl",
                            reports_output_dir=Path(tmp) / "reports",
                            lock_path=Path(tmp) / "cycle.lock",
                        )

        self.assertEqual(events, ["evaluate", "generate", "summary"])
        self.assertEqual(result["evaluation"]["closed"], 1)
        self.assertEqual(result["generation_summary"]["opened_signals"], 1)

    async def test_cycle_does_not_duplicate_signals_when_generator_skips_duplicate(self):
        async def fake_generate(*args, **kwargs):
            return [{"status": "skipped_duplicate_open", "config_id": "cfg1"}]

        with tempfile.TemporaryDirectory() as tmp:
            with patch("scripts.run_shadow_cycle_once.evaluate_shadow_signals_once", new=AsyncMock(return_value={
                "found_open": 1, "closed": 0, "still_open": 1, "errors": [], "closed_signals": [],
            })):
                with patch("scripts.run_shadow_cycle_once.generate_shadow_signals_once", new=AsyncMock(side_effect=fake_generate)):
                    with patch("scripts.run_shadow_cycle_once.summarize_shadow_signals", return_value={
                        "summary": {"open": 1, "closed": 0}, "signals": [],
                    }):
                        result = await run_shadow_cycle_once(
                            registry=str(Path(tmp) / "registry.jsonl"),
                            journal_path=Path(tmp) / "journal.jsonl",
                            reports_output_dir=Path(tmp) / "reports",
                            lock_path=Path(tmp) / "cycle.lock",
                        )

        self.assertEqual(result["generation_summary"]["opened_signals"], 0)
        self.assertEqual(result["generation_summary"]["skipped_duplicate_open"], 1)

    async def test_lock_avoids_concurrent_cycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = Path(tmp) / "cycle.lock"
            with shadow_cycle_lock(lock):
                with self.assertRaises(ShadowCycleLocked):
                    async with _noop_async_context():
                        await run_shadow_cycle_once(lock_path=lock, journal_path=Path(tmp) / "journal.jsonl")

    async def test_dry_run_does_not_write_journal_or_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = Path(tmp) / "journal.jsonl"
            output = Path(tmp) / "reports"
            lock = Path(tmp) / "cycle.lock"
            with patch("scripts.run_shadow_cycle_once.generate_shadow_signals_once", new=AsyncMock(return_value=[{"status": "skipped_hold", "config_id": "cfg1"}])):
                result = await run_shadow_cycle_once(
                    registry=str(Path(tmp) / "registry.jsonl"),
                    journal_path=journal,
                    reports_output_dir=output,
                    lock_path=lock,
                    dry_run=True,
                )

        self.assertTrue(result["dry_run"])
        self.assertFalse(journal.exists())
        self.assertFalse(output.exists())
        self.assertFalse(lock.exists())

    async def test_cycle_passes_max_configs_scanned_separately_from_max_signals(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("scripts.run_shadow_cycle_once.generate_shadow_signals_once", new=AsyncMock(return_value=[{"status": "skipped_hold", "config_id": "cfg1"}])) as generate:
                result = await run_shadow_cycle_once(
                    registry=str(Path(tmp) / "registry.jsonl"),
                    journal_path=Path(tmp) / "journal.jsonl",
                    reports_output_dir=Path(tmp) / "reports",
                    lock_path=Path(tmp) / "cycle.lock",
                    dry_run=True,
                    max_signals=1,
                    max_configs_scanned=8,
                )

        self.assertEqual(generate.call_args.kwargs["max_signals"], 1)
        self.assertEqual(generate.call_args.kwargs["max_configs_scanned"], 8)
        self.assertEqual(result["max_signals"], 1)
        self.assertEqual(result["max_configs_scanned"], 8)

    async def test_cycle_defaults_max_configs_scanned_to_max_signals(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("scripts.run_shadow_cycle_once.generate_shadow_signals_once", new=AsyncMock(return_value=[])) as generate:
                await run_shadow_cycle_once(
                    registry=str(Path(tmp) / "registry.jsonl"),
                    journal_path=Path(tmp) / "journal.jsonl",
                    reports_output_dir=Path(tmp) / "reports",
                    lock_path=Path(tmp) / "cycle.lock",
                    dry_run=True,
                    max_signals=2,
                )

        self.assertEqual(generate.call_args.kwargs["max_configs_scanned"], 2)


class ShadowSummaryTests(unittest.TestCase):
    def test_summary_calculates_profit_factor(self):
        rows = [
            shadow_row("win", outcome="WIN", pnl=2.0),
            shadow_row("loss", outcome="LOSS", pnl=-1.0),
            shadow_row("open", status="OPEN", outcome=None, pnl=None),
        ]

        summary = summarize_rows(rows)

        self.assertEqual(summary["open"], 1)
        self.assertEqual(summary["closed"], 2)
        self.assertEqual(summary["wins"], 1)
        self.assertEqual(summary["losses"], 1)
        self.assertEqual(summary["profit_factor"], 2.0)
        self.assertEqual(summary["avg_return"], 0.5)

    def test_drawdown_is_calculated(self):
        self.assertGreater(max_drawdown_pct([2.0, -5.0, 1.0]), 0)

    def test_summarizer_writes_markdown_and_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal_path = Path(tmp) / "shadow.jsonl"
            output_dir = Path(tmp) / "reports"
            journal = ShadowSignalJournal(journal_path)
            journal.append(shadow_row("win", outcome="WIN", pnl=2.0))
            journal.append(shadow_row("loss", outcome="LOSS", pnl=-1.0, config_id="cfg2"))

            report = summarize_shadow_signals(journal_path=journal_path, output_dir=output_dir)
            json_exists = Path(report["json_path"]).exists()
            markdown_exists = Path(report["markdown_path"]).exists()

        self.assertEqual(report["summary"]["profit_factor"], 2.0)
        self.assertTrue(json_exists)
        self.assertTrue(markdown_exists)

    def test_build_summary_groups_by_symbol_config_timeframe(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal_path = Path(tmp) / "shadow.jsonl"
            journal = ShadowSignalJournal(journal_path)
            journal.append(shadow_row("sol", symbol="SOL", config_id="cfg1", pnl=1.0))
            journal.append(shadow_row("btc", symbol="BTC", config_id="cfg2", pnl=-1.0, outcome="LOSS"))

            report = build_shadow_summary(journal_path)

        self.assertIn("SOL", report["by_symbol"])
        self.assertIn("BTC", report["by_symbol"])
        self.assertIn("cfg1", report["by_config"])
        self.assertIn("1h", report["by_timeframe"])

    def test_telegram_summary_does_not_leak_secrets(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal_path = Path(tmp) / "shadow.jsonl"
            journal = ShadowSignalJournal(journal_path)
            journal.append(shadow_row("win", outcome="WIN", pnl=1.0))
            with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "secret-token", "TELEGRAM_CHAT_ID": "secret-chat"}, clear=True):
                with patch("scripts.summarize_shadow_signals.send_telegram_message") as send:
                    summarize_shadow_signals(journal_path=journal_path, output_dir=Path(tmp) / "reports", notify_telegram=True)

        text = send.call_args.args[0]
        self.assertIn("Research only. No trading signal.", text)
        self.assertNotIn("secret-token", text)
        self.assertNotIn("secret-chat", text)

    def test_respects_tradeai_data_and_reports_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "custom_data"
            reports = Path(tmp) / "custom_reports"
            with patch.dict(os.environ, {"TRADEAI_DATA_DIR": str(data), "TRADEAI_REPORTS_DIR": str(reports)}, clear=True):
                self.assertEqual(default_journal_path(), data / "shadow_signal_journal.jsonl")
                self.assertEqual(default_shadow_reports_dir(), reports / "shadow")
                self.assertEqual(registry_path_for_choice("refined"), str(reports / "research_daemon" / "refined_registry.jsonl"))

    def test_shadow_cycle_cli_accepts_max_configs_scanned(self):
        args = build_shadow_cycle_parser().parse_args(["--max-signals", "1", "--max-configs-scanned", "8"])

        self.assertEqual(args.max_signals, 1)
        self.assertEqual(args.max_configs_scanned, 8)


class _noop_async_context:
    async def __aenter__(self):
        return None

    async def __aexit__(self, exc_type, exc, tb):
        return False


if __name__ == "__main__":
    unittest.main()
