import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from research.focused_research_grid import build_focused_research_grid
from research.research_registry import ResearchRegistry
from scripts.run_focused_research_loop import (
    acquire_lock,
    build_parser,
    count_runnable,
    release_lock,
    run_focused_research_loop,
)


def fake_cycle_result(selected=2, classification="unstable_watchlist"):
    return {
        "interrupted": False,
        "grid_size": 81,
        "runnable_before_cycle": 81,
        "selected_configs": selected,
        "summary": {
            "evaluated": selected,
            "classification_counts": {classification: selected},
            "stable_candidates": [],
            "unstable_watchlist": [{}] * selected if classification == "unstable_watchlist" else [],
            "rejects": [],
        },
        "results": [],
        "json_path": "cycle.json",
        "markdown_path": "cycle.md",
    }


class FocusedResearchLoopTests(unittest.IsolatedAsyncioTestCase):
    def test_cli_defaults(self):
        args = build_parser().parse_args([])

        self.assertEqual(args.max_cycles, 1)
        self.assertEqual(args.max_configs_per_cycle, 10)
        self.assertFalse(args.sync_supabase)
        self.assertFalse(args.notify_telegram)
        self.assertTrue(args.resume)

    def test_cli_accepts_loop_flags(self):
        args = build_parser().parse_args([
            "--max-cycles", "3",
            "--max-configs-per-cycle", "7",
            "--sync-supabase",
            "--notify-telegram",
            "--symbols", "ADA", "ETH",
        ])

        self.assertEqual(args.max_cycles, 3)
        self.assertEqual(args.max_configs_per_cycle, 7)
        self.assertTrue(args.sync_supabase)
        self.assertTrue(args.notify_telegram)
        self.assertEqual(args.symbols, ["ADA", "ETH"])

    def test_lock_prevents_concurrent_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock_path = Path(tmp) / "loop.lock"
            acquire_lock(lock_path)
            with self.assertRaises(RuntimeError):
                acquire_lock(lock_path)
            release_lock(lock_path)

        self.assertFalse(lock_path.exists())

    def test_count_runnable_respects_completed_registry(self):
        grid = build_focused_research_grid(symbols=["ADA"])[:2]
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = Path(tmp) / "registry.jsonl"
            ResearchRegistry(registry_path).mark_finished(
                grid[0],
                status="completed",
                classification="unstable_watchlist",
            )

            self.assertEqual(count_runnable(grid, registry_path=registry_path), 1)

    async def test_loop_runs_multiple_cycles_and_writes_status(self):
        grid = build_focused_research_grid(symbols=["ADA"])[:4]

        async def side_effect(*args, **kwargs):
            registry = ResearchRegistry(kwargs["registry_path"])
            runnable = registry.filter_runnable(grid)
            selected = runnable[: kwargs["max_configs_per_cycle"]]
            for config in selected:
                registry.mark_finished(config, status="completed", classification="unstable_watchlist")
            return fake_cycle_result(selected=len(selected))

        with tempfile.TemporaryDirectory() as tmp:
            status_path = Path(tmp) / "status.json"
            with patch("scripts.run_focused_research_loop.build_focused_research_grid", return_value=grid):
                with patch("scripts.run_focused_research_loop.run_research_cycle", new=AsyncMock(side_effect=side_effect)) as run:
                    result = await run_focused_research_loop(
                        max_cycles=3,
                        max_configs_per_cycle=2,
                        registry_path=Path(tmp) / "registry.jsonl",
                        cycles_dir=Path(tmp) / "cycles",
                        results_dir=Path(tmp) / "results",
                        daemon_status_path=Path(tmp) / "daemon_status.json",
                        loop_status_path=status_path,
                        lock_path=Path(tmp) / "loop.lock",
                        quiet=True,
                    )
            status = json.loads(status_path.read_text(encoding="utf-8"))

        self.assertEqual(run.call_count, 2)
        self.assertEqual(result["cycles_completed"], 2)
        self.assertEqual(result["remaining_configs"], 0)
        self.assertEqual(status["cycles_completed"], 2)
        self.assertEqual(status["remaining_configs"], 0)

    async def test_loop_syncs_after_each_cycle_when_enabled(self):
        grid = build_focused_research_grid(symbols=["ADA"])[:1]

        async def side_effect(*args, **kwargs):
            registry = ResearchRegistry(kwargs["registry_path"])
            registry.mark_finished(grid[0], status="completed", classification="unstable_watchlist")
            return fake_cycle_result(selected=1)

        with tempfile.TemporaryDirectory() as tmp:
            with patch("scripts.run_focused_research_loop.build_focused_research_grid", return_value=grid):
                with patch("scripts.run_focused_research_loop.run_research_cycle", new=AsyncMock(side_effect=side_effect)):
                    with patch(
                        "scripts.run_focused_research_loop.sync_focused_results_to_supabase",
                        return_value={"ok": True, "configs_upserted": 1, "reason": None},
                    ) as sync:
                        result = await run_focused_research_loop(
                            max_cycles=2,
                            max_configs_per_cycle=1,
                            sync_supabase=True,
                            registry_path=Path(tmp) / "registry.jsonl",
                            cycles_dir=Path(tmp) / "cycles",
                            results_dir=Path(tmp) / "results",
                            daemon_status_path=Path(tmp) / "daemon_status.json",
                            loop_status_path=Path(tmp) / "status.json",
                            lock_path=Path(tmp) / "loop.lock",
                            quiet=True,
                        )

        self.assertEqual(sync.call_count, 1)
        self.assertEqual(result["sync_results"][0]["ok"], True)

    async def test_loop_does_not_sync_when_disabled(self):
        grid = build_focused_research_grid(symbols=["ADA"])[:1]

        async def side_effect(*args, **kwargs):
            ResearchRegistry(kwargs["registry_path"]).mark_finished(
                grid[0],
                status="completed",
                classification="unstable_watchlist",
            )
            return fake_cycle_result(selected=1)

        with tempfile.TemporaryDirectory() as tmp:
            with patch("scripts.run_focused_research_loop.build_focused_research_grid", return_value=grid):
                with patch("scripts.run_focused_research_loop.run_research_cycle", new=AsyncMock(side_effect=side_effect)):
                    with patch("scripts.run_focused_research_loop.sync_focused_results_to_supabase") as sync:
                        await run_focused_research_loop(
                            max_cycles=1,
                            max_configs_per_cycle=1,
                            sync_supabase=False,
                            registry_path=Path(tmp) / "registry.jsonl",
                            cycles_dir=Path(tmp) / "cycles",
                            results_dir=Path(tmp) / "results",
                            daemon_status_path=Path(tmp) / "daemon_status.json",
                            loop_status_path=Path(tmp) / "status.json",
                            lock_path=Path(tmp) / "loop.lock",
                            quiet=True,
                        )

        sync.assert_not_called()

    async def test_loop_releases_lock_after_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock_path = Path(tmp) / "loop.lock"
            with patch("scripts.run_focused_research_loop.run_research_cycle", new=AsyncMock(side_effect=RuntimeError("boom"))):
                with self.assertRaises(RuntimeError):
                    await run_focused_research_loop(
                        max_cycles=1,
                        max_configs_per_cycle=1,
                        symbols=["ADA"],
                        registry_path=Path(tmp) / "registry.jsonl",
                        cycles_dir=Path(tmp) / "cycles",
                        results_dir=Path(tmp) / "results",
                        daemon_status_path=Path(tmp) / "daemon_status.json",
                        loop_status_path=Path(tmp) / "status.json",
                        lock_path=lock_path,
                        quiet=True,
                    )

        self.assertFalse(lock_path.exists())


if __name__ == "__main__":
    unittest.main()
