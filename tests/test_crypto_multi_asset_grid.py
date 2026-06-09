import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from research.crypto_multi_asset_grid import (
    CRYPTO_MULTI_ATR_STOP_MULTIPLIERS,
    CRYPTO_MULTI_HORIZON_CANDLES,
    CRYPTO_MULTI_RISK_REWARDS,
    CRYPTO_MULTI_SYMBOLS,
    build_crypto_multi_asset_grid,
)
from research.research_registry import ResearchRegistry
from research.research_daemon import run_research_cycle
from scripts.run_crypto_multi_asset_research import (
    CRYPTO_MULTI_CYCLES_DIR,
    CRYPTO_MULTI_REGISTRY_PATH,
    CRYPTO_MULTI_RESULTS_DIR,
    CRYPTO_MULTI_STATUS_PATH,
    build_parser,
    parse_symbols,
    running_records,
)
from scripts.run_refined_research import REFINED_REGISTRY_PATH


def fake_multi_window_result():
    return {
        "setups": [{
            "classification": "multi_window_reject",
            "aggregate": {
                "total_windows": 4,
                "valid_windows": 4,
                "median_validation_pf": 0.8,
                "median_validation_avg_return": -0.1,
                "median_test_pf": 1.0,
                "test_confirm_rate": 0.2,
            },
            "windows": [],
        }],
        "json_path": "result.json",
        "markdown_path": "result.md",
    }


class CryptoMultiAssetGridTests(unittest.IsolatedAsyncioTestCase):
    def test_grid_generates_64_configs_by_default(self):
        grid = build_crypto_multi_asset_grid()

        self.assertEqual(len(grid), 64)
        self.assertEqual({row["symbol"] for row in grid}, set(CRYPTO_MULTI_SYMBOLS))

    def test_grid_filters_symbols(self):
        grid = build_crypto_multi_asset_grid(symbols=["BTC", "ETH", "SOL"])

        self.assertEqual(len(grid), 24)
        self.assertEqual({row["symbol"] for row in grid}, {"BTC", "ETH", "SOL"})

    def test_grid_rejects_non_crypto_symbols(self):
        with self.assertRaises(ValueError):
            build_crypto_multi_asset_grid(symbols=["SOL", "AAPL"])

    def test_grid_only_1h_xgboost_low_costs(self):
        grid = build_crypto_multi_asset_grid()

        self.assertEqual({row["timeframe"] for row in grid}, {"1h"})
        self.assertEqual({row["strategy_mode"] for row in grid}, {"xgboost"})
        self.assertEqual({row["cost_mode"] for row in grid}, {"low_costs"})

    def test_grid_has_expected_horizon_rr_and_atr_values(self):
        grid = build_crypto_multi_asset_grid()

        self.assertEqual({row["horizon_candles"] for row in grid}, set(CRYPTO_MULTI_HORIZON_CANDLES))
        self.assertEqual({row["risk_reward"] for row in grid}, set(CRYPTO_MULTI_RISK_REWARDS))
        self.assertEqual({row["atr_stop_multiplier"] for row in grid}, set(CRYPTO_MULTI_ATR_STOP_MULTIPLIERS))

    def test_grid_config_ids_are_stable_and_symbol_specific(self):
        first = build_crypto_multi_asset_grid()
        second = build_crypto_multi_asset_grid()

        self.assertEqual([row["config_id"] for row in first], [row["config_id"] for row in second])
        self.assertEqual(len({row["config_id"] for row in first}), 64)

    def test_cli_parses_symbol_lists(self):
        self.assertEqual(parse_symbols(["BTC,ETH", "SOL"]), ["BTC", "ETH", "SOL"])

    def test_cli_defaults(self):
        args = build_parser().parse_args([])

        self.assertTrue(args.once)
        self.assertEqual(args.max_configs_per_cycle, 5)
        self.assertTrue(args.resume)
        self.assertFalse(args.retry_failed)
        self.assertFalse(args.retry_running)
        self.assertFalse(args.list_running)
        self.assertFalse(args.notify_telegram)
        self.assertTrue(args.progress)
        self.assertFalse(args.quiet)

    def test_cli_accepts_retry_running_and_list_running(self):
        args = build_parser().parse_args(["--retry-running", "--list-running"])

        self.assertTrue(args.retry_running)
        self.assertTrue(args.list_running)

    def test_script_uses_separate_paths(self):
        self.assertIn("crypto_multi_registry.jsonl", str(CRYPTO_MULTI_REGISTRY_PATH))
        self.assertIn("crypto_multi_cycles", str(CRYPTO_MULTI_CYCLES_DIR))
        self.assertIn("crypto_multi_results", str(CRYPTO_MULTI_RESULTS_DIR))
        self.assertIn("crypto_multi_current_status.json", str(CRYPTO_MULTI_STATUS_PATH))
        self.assertNotEqual(CRYPTO_MULTI_REGISTRY_PATH, REFINED_REGISTRY_PATH)

    async def test_run_research_cycle_accepts_crypto_multi_grid_without_refined_registry(self):
        import research.research_daemon as daemon

        grid = build_crypto_multi_asset_grid(symbols=["BTC"])
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = Path(tmp) / "crypto_multi_registry.jsonl"
            refined_path = Path(tmp) / "refined_registry.jsonl"
            with patch.object(daemon, "run_multi_window_validation_for_setups", new=AsyncMock(return_value=fake_multi_window_result())) as run:
                result = await run_research_cycle(
                    grid=grid,
                    max_configs_per_cycle=2,
                    registry_path=registry_path,
                    cycles_dir=Path(tmp) / "crypto_multi_cycles",
                    results_dir=Path(tmp) / "crypto_multi_results",
                    status_path=Path(tmp) / "crypto_multi_current_status.json",
                )
                registry_exists = registry_path.exists()
                refined_exists = refined_path.exists()

        self.assertEqual(result["grid_size"], 8)
        self.assertEqual(run.call_count, 2)
        self.assertTrue(registry_exists)
        self.assertFalse(refined_exists)
        self.assertEqual({call.kwargs["setups"][0]["symbol"] for call in run.call_args_list}, {"BTC"})

    async def test_running_config_can_be_retried_with_retry_running(self):
        import research.research_daemon as daemon

        grid = build_crypto_multi_asset_grid(symbols=["BTC"])[:1]
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = Path(tmp) / "crypto_multi_registry.jsonl"
            registry = ResearchRegistry(registry_path)
            registry.mark_running(grid[0])
            with patch.object(daemon, "run_multi_window_validation_for_setups", new=AsyncMock(return_value=fake_multi_window_result())) as run:
                result = await run_research_cycle(
                    grid=grid,
                    max_configs_per_cycle=1,
                    retry_running=True,
                    registry_path=registry_path,
                    cycles_dir=Path(tmp) / "crypto_multi_cycles",
                    results_dir=Path(tmp) / "crypto_multi_results",
                    status_path=Path(tmp) / "crypto_multi_current_status.json",
                )
            latest = ResearchRegistry(registry_path).latest_by_config()[grid[0]["config_id"]]

        self.assertEqual(run.call_count, 1)
        self.assertEqual(result["selected_configs"], 1)
        self.assertEqual(latest["status"], "completed")

    async def test_running_config_not_retried_without_retry_running(self):
        import research.research_daemon as daemon

        grid = build_crypto_multi_asset_grid(symbols=["BTC"])[:1]
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = Path(tmp) / "crypto_multi_registry.jsonl"
            ResearchRegistry(registry_path).mark_running(grid[0])
            with patch.object(daemon, "run_multi_window_validation_for_setups", new=AsyncMock(return_value=fake_multi_window_result())) as run:
                result = await run_research_cycle(
                    grid=grid,
                    max_configs_per_cycle=1,
                    retry_running=False,
                    registry_path=registry_path,
                    cycles_dir=Path(tmp) / "crypto_multi_cycles",
                    results_dir=Path(tmp) / "crypto_multi_results",
                    status_path=Path(tmp) / "crypto_multi_current_status.json",
                )

        self.assertEqual(run.call_count, 0)
        self.assertEqual(result["selected_configs"], 0)

    async def test_completed_config_not_retried_with_retry_running(self):
        import research.research_daemon as daemon

        grid = build_crypto_multi_asset_grid(symbols=["BTC"])[:1]
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = Path(tmp) / "crypto_multi_registry.jsonl"
            ResearchRegistry(registry_path).mark_finished(
                grid[0],
                status="completed",
                classification="multi_window_reject",
            )
            with patch.object(daemon, "run_multi_window_validation_for_setups", new=AsyncMock(return_value=fake_multi_window_result())) as run:
                result = await run_research_cycle(
                    grid=grid,
                    max_configs_per_cycle=1,
                    retry_running=True,
                    registry_path=registry_path,
                    cycles_dir=Path(tmp) / "crypto_multi_cycles",
                    results_dir=Path(tmp) / "crypto_multi_results",
                    status_path=Path(tmp) / "crypto_multi_current_status.json",
                )

        self.assertEqual(run.call_count, 0)
        self.assertEqual(result["selected_configs"], 0)

    def test_running_records_lists_only_running_crypto_registry_entries(self):
        grid = build_crypto_multi_asset_grid(symbols=["BTC", "ETH"])
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = Path(tmp) / "crypto_multi_registry.jsonl"
            registry = ResearchRegistry(registry_path)
            registry.mark_running(grid[0])
            registry.mark_finished(grid[8], status="completed", classification="multi_window_reject")

            rows = running_records(registry_path)
            btc_rows = running_records(registry_path, symbols=["BTC"])
            eth_rows = running_records(registry_path, symbols=["ETH"])

        self.assertEqual([row["config_id"] for row in rows], [grid[0]["config_id"]])
        self.assertEqual([row["config_id"] for row in btc_rows], [grid[0]["config_id"]])
        self.assertEqual(eth_rows, [])

    def test_grid_guardrails_do_not_include_test_selection_fields(self):
        grid = build_crypto_multi_asset_grid(symbols=["SOL"])

        self.assertTrue(all(row["research_phase"] == "crypto_multi_asset_grid_v1" for row in grid))
        self.assertTrue(all("test" not in row for row in grid))


if __name__ == "__main__":
    unittest.main()
