import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

from research.research_daemon import (
    build_daemon_grid,
    render_cycle_markdown,
    run_research_cycle,
    summarize_cycle,
)
from scripts.run_research_daemon import build_parser


def fake_multi_window_result(classification="multi_window_reject", median_pf=0.8, median_avg=-0.1):
    setup_result = {
        "setup": {"symbol": "SOL", "timeframe": "1h", "horizon_candles": 16, "risk_reward": 2.0, "atr_stop_multiplier": 1.25, "cost_mode": "low_costs"},
        "classification": classification,
        "aggregate": {
            "total_windows": 4,
            "valid_windows": 4,
            "insufficient_windows": 0,
            "validation_positive_rate": 0.25,
            "beats_random_rate": 0.25,
            "beats_deterministic_rate": 0.25,
            "test_confirm_rate": 0.25,
            "median_validation_pf": median_pf,
            "median_validation_avg_return": median_avg,
            "worst_validation_drawdown": 12,
            "median_test_pf": 0.9,
            "median_test_avg_return": -0.05,
        },
        "windows": [],
    }
    return {
        "setups": [setup_result],
        "json_path": "result.json",
        "markdown_path": "result.md",
    }


class ResearchDaemonTests(unittest.IsolatedAsyncioTestCase):
    def test_grid_generates_96_configs(self):
        grid = build_daemon_grid()

        self.assertEqual(len(grid), 96)
        self.assertEqual({row["symbol"] for row in grid}, {"SOL"})
        self.assertEqual({row["timeframe"] for row in grid}, {"1h"})
        self.assertEqual({row["max_candles"] for row in grid}, {5000})
        self.assertEqual({row["horizon_candles"] for row in grid}, {12, 16, 20, 24})
        self.assertEqual({row["risk_reward"] for row in grid}, {1.5, 2.0, 2.5})
        self.assertEqual({row["atr_stop_multiplier"] for row in grid}, {1.0, 1.25, 1.5, 1.75})
        self.assertEqual({row["cost_mode"] for row in grid}, {"low_costs", "medium_costs_current"})
        self.assertEqual({row["strategy_mode"] for row in grid}, {"xgboost"})

    async def test_max_configs_per_cycle_limits_run(self):
        import research.research_daemon as daemon

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(daemon, "run_multi_window_validation_for_setups", new=AsyncMock(return_value=fake_multi_window_result())) as run:
                result = await run_research_cycle(
                    max_configs_per_cycle=2,
                    registry_path=Path(tmp) / "registry.jsonl",
                    cycles_dir=Path(tmp) / "cycles",
                    results_dir=Path(tmp) / "results",
                    status_path=Path(tmp) / "current_status.json",
                )

        self.assertEqual(run.call_count, 2)
        self.assertEqual(result["selected_configs"], 2)
        self.assertEqual(result["summary"]["evaluated"], 2)

    async def test_filters_completed_configs_when_resuming(self):
        import research.research_daemon as daemon

        with tempfile.TemporaryDirectory() as tmp:
            registry_path = Path(tmp) / "registry.jsonl"
            with patch.object(daemon, "run_multi_window_validation_for_setups", new=AsyncMock(return_value=fake_multi_window_result())):
                first = await run_research_cycle(
                    max_configs_per_cycle=1,
                    registry_path=registry_path,
                    cycles_dir=Path(tmp) / "cycles",
                    results_dir=Path(tmp) / "results",
                    status_path=Path(tmp) / "current_status.json",
                )
                second = await run_research_cycle(
                    max_configs_per_cycle=1,
                    registry_path=registry_path,
                    cycles_dir=Path(tmp) / "cycles",
                    results_dir=Path(tmp) / "results",
                    status_path=Path(tmp) / "current_status.json",
                )

        self.assertNotEqual(first["results"][0]["config_id"], second["results"][0]["config_id"])

    async def test_saves_result_per_config_in_registry(self):
        import research.research_daemon as daemon

        with tempfile.TemporaryDirectory() as tmp:
            registry_path = Path(tmp) / "registry.jsonl"
            with patch.object(daemon, "run_multi_window_validation_for_setups", new=AsyncMock(return_value=fake_multi_window_result())):
                await run_research_cycle(
                    max_configs_per_cycle=1,
                    registry_path=registry_path,
                    cycles_dir=Path(tmp) / "cycles",
                    results_dir=Path(tmp) / "results",
                    status_path=Path(tmp) / "current_status.json",
                )
            text = registry_path.read_text(encoding="utf-8")

        self.assertIn('"status": "running"', text)
        self.assertIn('"status": "completed"', text)
        self.assertIn('"classification": "multi_window_reject"', text)

    async def test_research_daemon_reports_classification_counts_correctly(self):
        import research.research_daemon as daemon

        results = [
            fake_multi_window_result("multi_window_reject", median_pf=0.7, median_avg=-0.1),
            fake_multi_window_result("unstable_watchlist", median_pf=1.02, median_avg=0.02),
        ]

        async def side_effect(*args, **kwargs):
            return results.pop(0)

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(daemon, "run_multi_window_validation_for_setups", new=AsyncMock(side_effect=side_effect)):
                result = await run_research_cycle(
                    max_configs_per_cycle=2,
                    registry_path=Path(tmp) / "registry.jsonl",
                    cycles_dir=Path(tmp) / "cycles",
                    results_dir=Path(tmp) / "results",
                    status_path=Path(tmp) / "current_status.json",
                )
                markdown = Path(result["markdown_path"]).read_text(encoding="utf-8")

        self.assertEqual(result["summary"]["classification_counts"], {
            "multi_window_reject": 1,
            "unstable_watchlist": 1,
        })
        self.assertIn("## Unstable Watchlist", markdown)
        self.assertIn("## Rejects", markdown)

    async def test_failed_config_is_saved_and_cycle_continues(self):
        import research.research_daemon as daemon

        async def side_effect(*args, **kwargs):
            if side_effect.calls == 0:
                side_effect.calls += 1
                raise RuntimeError("boom")
            return fake_multi_window_result()
        side_effect.calls = 0

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(daemon, "run_multi_window_validation_for_setups", new=AsyncMock(side_effect=side_effect)):
                result = await run_research_cycle(
                    max_configs_per_cycle=2,
                    registry_path=Path(tmp) / "registry.jsonl",
                    cycles_dir=Path(tmp) / "cycles",
                    results_dir=Path(tmp) / "results",
                    status_path=Path(tmp) / "current_status.json",
                )

        self.assertEqual(result["summary"]["failed"], 1)
        self.assertEqual(result["summary"]["completed"], 1)

    async def test_status_file_is_created_and_updated(self):
        import research.research_daemon as daemon

        with tempfile.TemporaryDirectory() as tmp:
            status_path = Path(tmp) / "current_status.json"
            with patch.object(daemon, "run_multi_window_validation_for_setups", new=AsyncMock(return_value=fake_multi_window_result())):
                await run_research_cycle(
                    max_configs_per_cycle=2,
                    registry_path=Path(tmp) / "registry.jsonl",
                    cycles_dir=Path(tmp) / "cycles",
                    results_dir=Path(tmp) / "results",
                    status_path=status_path,
                )
            status = json.loads(status_path.read_text(encoding="utf-8"))

        self.assertEqual(status["completed_in_cycle"], 2)
        self.assertEqual(status["selected_configs"], 2)
        self.assertEqual(status["classification_counts"], {"multi_window_reject": 2})
        self.assertEqual(status["rejects_so_far"], 2)
        self.assertIn("last_result", status)
        self.assertIn("elapsed_seconds", status)

    async def test_progress_callback_receives_each_completed_config(self):
        import research.research_daemon as daemon

        events = []
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(daemon, "run_multi_window_validation_for_setups", new=AsyncMock(return_value=fake_multi_window_result())):
                await run_research_cycle(
                    max_configs_per_cycle=2,
                    registry_path=Path(tmp) / "registry.jsonl",
                    cycles_dir=Path(tmp) / "cycles",
                    results_dir=Path(tmp) / "results",
                    status_path=Path(tmp) / "current_status.json",
                    progress_callback=events.append,
                )

        completed_events = [event for event in events if event["completed_in_cycle"] > 0]
        self.assertEqual([event["completed_in_cycle"] for event in completed_events], [1, 2])
        self.assertEqual(completed_events[-1]["classification_counts"], {"multi_window_reject": 2})

    async def test_status_records_failed_config_and_continues(self):
        import research.research_daemon as daemon

        async def side_effect(*args, **kwargs):
            if side_effect.calls == 0:
                side_effect.calls += 1
                raise RuntimeError("boom")
            return fake_multi_window_result()
        side_effect.calls = 0

        with tempfile.TemporaryDirectory() as tmp:
            status_path = Path(tmp) / "current_status.json"
            with patch.object(daemon, "run_multi_window_validation_for_setups", new=AsyncMock(side_effect=side_effect)):
                await run_research_cycle(
                    max_configs_per_cycle=2,
                    registry_path=Path(tmp) / "registry.jsonl",
                    cycles_dir=Path(tmp) / "cycles",
                    results_dir=Path(tmp) / "results",
                    status_path=status_path,
                )
            status = json.loads(status_path.read_text(encoding="utf-8"))

        self.assertEqual(status["errors_so_far"], 1)
        self.assertEqual(status["classification_counts"], {"failed": 1, "multi_window_reject": 1})
        self.assertEqual(status["completed_in_cycle"], 2)

    async def test_keyboard_interrupt_leaves_registry_valid(self):
        import research.research_daemon as daemon

        with tempfile.TemporaryDirectory() as tmp:
            registry_path = Path(tmp) / "registry.jsonl"
            with patch.object(daemon, "run_multi_window_validation_for_setups", new=AsyncMock(side_effect=KeyboardInterrupt)):
                result = await run_research_cycle(
                    max_configs_per_cycle=1,
                    registry_path=registry_path,
                    cycles_dir=Path(tmp) / "cycles",
                    results_dir=Path(tmp) / "results",
                    status_path=Path(tmp) / "current_status.json",
                )

            text = registry_path.read_text(encoding="utf-8")

        self.assertTrue(result["interrupted"])
        self.assertIn('"status": "running"', text)
        self.assertIn('"status": "failed"', text)
        self.assertIn("KeyboardInterrupt", text)

    async def test_telegram_not_called_by_default(self):
        import research.research_daemon as daemon

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(daemon, "run_multi_window_validation_for_setups", new=AsyncMock(return_value=fake_multi_window_result())):
                with patch.object(daemon, "send_telegram_message") as send:
                    await run_research_cycle(
                        max_configs_per_cycle=1,
                        registry_path=Path(tmp) / "registry.jsonl",
                        cycles_dir=Path(tmp) / "cycles",
                        results_dir=Path(tmp) / "results",
                        status_path=Path(tmp) / "current_status.json",
                    )

        send.assert_not_called()

    async def test_telegram_only_called_at_end_when_requested(self):
        import research.research_daemon as daemon

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(daemon, "run_multi_window_validation_for_setups", new=AsyncMock(return_value=fake_multi_window_result())):
                with patch.object(daemon, "send_telegram_message", return_value=True) as send:
                    result = await run_research_cycle(
                        max_configs_per_cycle=2,
                        notify_telegram=True,
                        registry_path=Path(tmp) / "registry.jsonl",
                        cycles_dir=Path(tmp) / "cycles",
                        results_dir=Path(tmp) / "results",
                        status_path=Path(tmp) / "current_status.json",
                    )

        self.assertEqual(send.call_count, 1)
        self.assertTrue(result["telegram_sent"])
        self.assertIn("Research only. No trading signal.", send.call_args.args[0])

    def test_no_test_selection_guardrail_in_cycle_markdown(self):
        cycle = {
            "finished_at": "2026-01-01T00:00:00+00:00",
            "summary": summarize_cycle([{
                "status": "completed",
                "classification": "multi_window_reject",
                "config": build_daemon_grid()[0],
                "result": fake_multi_window_result()["setups"][0],
            }]),
            "results": [{
                "status": "completed",
                "classification": "multi_window_reject",
                "config": build_daemon_grid()[0],
                "result": fake_multi_window_result()["setups"][0],
            }],
        }

        markdown = render_cycle_markdown(cycle)

        self.assertIn("Validation selects; test is diagnostic confirmation only.", markdown)
        self.assertIn("Top By Median Test PF (Diagnostic Only, Not Selectable)", markdown)

    def test_cli_defaults(self):
        args = build_parser().parse_args([])

        self.assertTrue(args.once)
        self.assertEqual(args.max_configs_per_cycle, 5)
        self.assertTrue(args.resume)
        self.assertFalse(args.retry_failed)
        self.assertFalse(args.notify_telegram)
        self.assertTrue(args.progress)
        self.assertFalse(args.quiet)

    def test_cli_accepts_quiet(self):
        args = build_parser().parse_args(["--quiet"])

        self.assertTrue(args.quiet)


if __name__ == "__main__":
    unittest.main()
