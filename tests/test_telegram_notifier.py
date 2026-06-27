import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from research.telegram_notifier import (
    format_autopilot_summary_for_telegram,
    format_shadow_ops_cycle_brief,
    format_shadow_daily_summary,
    send_telegram_message,
    telegram_enabled,
)


def sample_rows():
    return [
        {
            "classification": "research_watchlist",
            "config": {
                "symbol": "SOL",
                "timeframe": "1h",
                "horizon_candles": 24,
                "risk_reward": 2.0,
                "atr_stop_multiplier": 1.5,
                "cost_mode": "low_costs",
            },
            "validation_metrics": {"profit_factor": 1.1128, "avg_return_pct": 0.0697, "max_drawdown_pct": 30.5},
            "test_metrics": {"profit_factor": 1.5, "avg_return_pct": 0.12, "max_drawdown_pct": 8.0},
        },
        {
            "classification": "hard_reject",
            "config": {
                "symbol": "ETH",
                "timeframe": "1h",
                "horizon_candles": 16,
                "risk_reward": 2.0,
                "atr_stop_multiplier": 1.25,
                "cost_mode": "medium_costs_current",
            },
            "validation_metrics": {"profit_factor": 0.8, "avg_return_pct": -0.1, "max_drawdown_pct": 20.0},
            "test_metrics": {"profit_factor": 2.0, "avg_return_pct": 0.3, "max_drawdown_pct": 5.0},
        },
    ]


class TelegramNotifierTests(unittest.TestCase):
    def test_telegram_disabled_without_env_vars(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(telegram_enabled())
            self.assertFalse(send_telegram_message("hello"))

    def test_format_summary_includes_counts_candidates_and_watchlist(self):
        text = format_autopilot_summary_for_telegram(
            sample_rows(),
            markdown_path="reports/research_autopilot/summary.md",
            jsonl_path="reports/research_autopilot/results.jsonl",
        )

        self.assertIn("Research Autopilot finished", text)
        self.assertIn("Research only. No trading signal.", text)
        self.assertIn("Experiments: 2", text)
        self.assertIn("Watchlist: 1", text)
        self.assertIn("hard_reject=1", text)
        self.assertIn("research_watchlist=1", text)
        self.assertIn("Best validation PF", text)
        self.assertIn("Best test PF (diagnostic only, not selectable)", text)
        self.assertNotIn("BUY", text)
        self.assertNotIn("SELL", text)

    def test_send_failure_returns_false(self):
        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "token", "TELEGRAM_CHAT_ID": "chat"}, clear=True):
            with patch("research.telegram_notifier.httpx.post", side_effect=RuntimeError("network down")):
                self.assertFalse(send_telegram_message("hello"))

    def test_shadow_ops_cycle_brief_explains_research_status(self):
        text = format_shadow_ops_cycle_brief({
            "health_before": {"health_status": "HEALTH_WARNING"},
            "evaluation": {"closed": 0, "errors": [{}]},
            "evaluation_error_summary": {"count": 1},
            "generation_cycle": {
                "generation_summary": {
                    "configs_scanned": 18,
                    "opened_signals": 0,
                    "skipped_hold": 11,
                    "skipped_errors": 1,
                    "status_counts": {"skipped_no_price": 0},
                },
            },
            "final_summary": {
                "open": 0,
                "closed": 25,
                "wins": 5,
                "losses": 19,
                "expired": 1,
                "profit_factor": 0.4148,
                "avg_return": -0.3861,
                "max_drawdown": 9.49,
            },
            "supabase_sync": {"ok": True, "reason": "supabase_first"},
            "cycles_sync": {"ok": True, "reason": None},
        })

        self.assertIn("TRADEAI Shadow Ops", text)
        self.assertIn("Research only. No trading signal.", text)
        self.assertIn("PF menor a 1.0", text)
        self.assertIn("No operar dinero real", text)
        self.assertNotIn("secret", text.lower())

    def test_shadow_daily_summary_is_human_readable(self):
        text = format_shadow_daily_summary([
            {"outcome": "WIN", "pnl_pct": 1.0},
            {"outcome": "LOSS", "pnl_pct": -2.0},
        ])

        self.assertIn("TRADEAI Shadow Summary", text)
        self.assertIn("WIN/LOSS/EXPIRED: 1/1/0", text)
        self.assertIn("Research only. No trading signal.", text)


class TelegramAutopilotIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_autopilot_notification_failure_does_not_break_run(self):
        import research.autopilot as autopilot

        with tempfile.TemporaryDirectory() as tmp:
            fake_result = {
                "experiment_id": "x",
                "classification": "research_watchlist",
                "config": {"symbol": "SOL", "timeframe": "1h"},
                "validation_metrics": {},
                "test_metrics": {},
                "baselines": {},
            }
            with patch.object(autopilot, "build_experiment_grid", return_value=[{"experiment_id": "x", "symbol": "SOL", "timeframe": "1h", "horizon_candles": 24, "risk_reward": 2.0, "atr_stop_multiplier": 1.5, "cost_mode": "low_costs"}]):
                with patch.object(autopilot, "run_experiment", new=AsyncMock(return_value=fake_result)):
                    with patch.object(autopilot, "send_telegram_message", side_effect=RuntimeError("telegram down")):
                        result = await autopilot.run_autopilot(
                            resume=False,
                            max_experiments=1,
                            reports_dir=tmp,
                            checkpoint_path=Path(tmp) / "checkpoint.json",
                            notify_telegram=True,
                        )

            self.assertFalse(result["interrupted"])
            self.assertFalse(result["telegram_sent"])

    async def test_autopilot_does_not_notify_by_default(self):
        import research.autopilot as autopilot

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(autopilot, "build_experiment_grid", return_value=[]):
                with patch.object(autopilot, "send_telegram_message") as send:
                    result = await autopilot.run_autopilot(
                        resume=False,
                        max_experiments=0,
                        reports_dir=tmp,
                        checkpoint_path=Path(tmp) / "checkpoint.json",
                    )

            send.assert_not_called()
            self.assertFalse(result["telegram_sent"])

    def test_cli_accepts_notify_telegram_flag(self):
        from scripts.run_autopilot import build_parser

        args = build_parser().parse_args(["--notify-telegram", "--max-experiments", "2"])

        self.assertTrue(args.notify_telegram)
        self.assertEqual(args.max_experiments, 2)

    def test_cli_default_without_flags_disables_telegram(self):
        from scripts.run_autopilot import build_parser

        args = build_parser().parse_args([])

        self.assertFalse(args.notify_telegram)

    def test_cli_no_notify_flag_disables_telegram(self):
        from scripts.run_autopilot import build_parser

        args = build_parser().parse_args(["--no-notify-telegram"])

        self.assertFalse(args.notify_telegram)

    async def test_cli_without_notify_does_not_print_telegram_disabled(self):
        import scripts.run_autopilot as script

        fake_result = {
            "interrupted": False,
            "ran": 0,
            "completed": 0,
            "total": 16,
            "candidates": 0,
            "watchlist": 0,
            "telegram_sent": False,
            "jsonl_path": "reports/results.jsonl",
            "markdown_path": "reports/summary.md",
            "checkpoint_path": "reports/checkpoint.json",
        }
        with patch.object(script, "run_autopilot", new=AsyncMock(return_value=fake_result)) as run:
            with patch("sys.argv", ["run_autopilot.py", "--no-resume", "--max-experiments", "2"]):
                with patch("builtins.print") as printed:
                    await script.main()

        self.assertFalse(run.call_args.kwargs["notify_telegram"])
        output = "\n".join(str(call.args[0]) for call in printed.call_args_list if call.args)
        self.assertNotIn("Telegram disabled", output)
        self.assertNotIn("telegram_sent:", output)

    async def test_cli_with_notify_prints_disabled_when_env_missing(self):
        import scripts.run_autopilot as script

        fake_result = {
            "interrupted": False,
            "ran": 0,
            "completed": 0,
            "total": 16,
            "candidates": 0,
            "watchlist": 0,
            "telegram_sent": False,
            "jsonl_path": "reports/results.jsonl",
            "markdown_path": "reports/summary.md",
            "checkpoint_path": "reports/checkpoint.json",
        }

        async def fake_run_autopilot(**kwargs):
            from research.telegram_notifier import send_telegram_message

            send_telegram_message("summary")
            return fake_result

        with patch.dict(os.environ, {}, clear=True):
            with patch.object(script, "run_autopilot", new=fake_run_autopilot):
                with patch("sys.argv", ["run_autopilot.py", "--no-resume", "--max-experiments", "2", "--notify-telegram"]):
                    with patch("builtins.print") as printed:
                        await script.main()

        output = "\n".join(str(call.args[0]) for call in printed.call_args_list if call.args)
        self.assertIn("Telegram disabled: missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID", output)
        self.assertIn("telegram_sent: False", output)


if __name__ == "__main__":
    unittest.main()
