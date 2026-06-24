from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from scripts.run_shadow_ops_once import (
    run_shadow_ops_once,
    summarize_evaluation_errors,
    supabase_candidate_configs,
    supabase_research_config_diagnostics,
)
from scripts.shadow_ops_healthcheck import build_healthcheck_report


def empty_summary():
    return {
        "summary": {"open": 0, "closed": 0, "total": 0},
        "signals": [],
        "by_symbol": {},
        "by_config": {},
        "by_timeframe": {},
    }


class RailwayShadowOpsModeTests(unittest.IsolatedAsyncioTestCase):
    async def test_healthcheck_does_not_warn_journal_missing_on_railway(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing_journal = Path(tmp) / "missing.jsonl"
            with patch.dict("os.environ", {"RAILWAY_ENVIRONMENT": "production"}, clear=False):
                with patch("scripts.shadow_ops_healthcheck.build_supabase_client_from_env", return_value=None):
                    with patch("scripts.shadow_ops_healthcheck._check_market_connectivity", new=AsyncMock(return_value={
                        "ok": True,
                        "rows": 1,
                        "error": None,
                        "category": None,
                    })):
                        report = await build_healthcheck_report(journal_path=missing_journal)

        self.assertNotIn("shadow_journal_missing", report["warnings"])
        self.assertNotIn("crypto_multi_registry_missing", report["blockers"])

    async def test_railway_mode_uses_supabase_store_and_candidates(self):
        fake_supabase = Mock()
        candidate = {
            "config_id": "cfg1",
            "symbol": "ADA",
            "timeframe": "1h",
            "_source_classification": "unstable_watchlist",
        }

        async def fake_generate(**kwargs):
            self.assertIsNotNone(kwargs.get("signal_store"))
            self.assertEqual(kwargs.get("candidate_configs"), [candidate])
            return [{"status": "skipped_hold", "config_id": "cfg1"}]

        with tempfile.TemporaryDirectory() as tmp:
            journal = Path(tmp) / "journal.jsonl"
            with patch.dict("os.environ", {"RAILWAY_ENVIRONMENT": "production"}, clear=False):
                with patch("scripts.run_shadow_ops_once.build_supabase_client_from_env", return_value=fake_supabase):
                    with patch("scripts.run_shadow_ops_once.build_healthcheck_report", new=AsyncMock(return_value={
                        "health_status": "HEALTH_OK",
                        "shadow_journal": {"open": 0},
                    })):
                        with patch("scripts.run_shadow_ops_once.supabase_candidate_configs", return_value=[candidate]):
                            with patch("scripts.run_shadow_ops_once.supabase_shadow_summary", return_value=empty_summary()):
                                with patch("scripts.run_shadow_ops_once.evaluate_shadow_signals_once", new=AsyncMock(return_value={
                                    "found_open": 0,
                                    "closed": 0,
                                    "still_open": 0,
                                    "errors": [],
                                    "closed_signals": [],
                                })) as evaluate:
                                    with patch("scripts.run_shadow_ops_once.generate_shadow_signals_once", new=AsyncMock(side_effect=fake_generate)):
                                        result = await run_shadow_ops_once(
                                            dry_run=False,
                                            journal_path=journal,
                                            reports_output_dir=Path(tmp) / "reports",
                                        )

        self.assertTrue(result["railway_mode"])
        self.assertTrue(result["supabase_first"])
        self.assertFalse(journal.exists())
        self.assertIsNotNone(evaluate.call_args.kwargs.get("signal_store"))

    async def test_local_mode_still_uses_existing_shadow_cycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict("os.environ", {}, clear=True):
                with patch("scripts.run_shadow_ops_once.build_healthcheck_report", new=AsyncMock(return_value={
                    "health_status": "HEALTH_OK",
                    "shadow_journal": {"open": 0},
                })):
                    with patch("scripts.run_shadow_ops_once.summarize_shadow_signals", return_value=empty_summary()):
                        with patch("scripts.run_shadow_ops_once.run_shadow_cycle_once", new=AsyncMock(return_value={
                            "generation_summary": {"opened_signals": 0, "configs_scanned": 0},
                        })) as cycle:
                            result = await run_shadow_ops_once(
                                dry_run=False,
                                journal_path=Path(tmp) / "journal.jsonl",
                                reports_output_dir=Path(tmp) / "reports",
                            )

        self.assertFalse(result["railway_mode"])
        self.assertFalse(result["supabase_first"])
        self.assertTrue(cycle.called)

    async def test_railway_result_includes_config_diagnostics_when_no_candidates(self):
        fake_supabase = Mock()

        with tempfile.TemporaryDirectory() as tmp:
            journal = Path(tmp) / "journal.jsonl"
            with patch.dict("os.environ", {"RAILWAY_ENVIRONMENT": "production"}, clear=False):
                with patch("scripts.run_shadow_ops_once.build_supabase_client_from_env", return_value=fake_supabase):
                    with patch("scripts.run_shadow_ops_once.build_healthcheck_report", new=AsyncMock(return_value={
                        "health_status": "HEALTH_OK",
                        "shadow_journal": {"open": 0},
                    })):
                        with patch("scripts.run_shadow_ops_once.supabase_research_config_diagnostics", return_value={
                            "query_ok": True,
                            "source_rows": 64,
                            "completed_rows": 64,
                            "eligible_classification_rows": 21,
                            "symbol_filtered_rows": 0,
                            "classification_counts": {"unstable_watchlist": 21},
                        }):
                            with patch("scripts.run_shadow_ops_once.supabase_candidate_configs", return_value=[]):
                                with patch("scripts.run_shadow_ops_once.supabase_shadow_summary", return_value=empty_summary()):
                                    with patch("scripts.run_shadow_ops_once.evaluate_shadow_signals_once", new=AsyncMock(return_value={
                                        "found_open": 0,
                                        "closed": 0,
                                        "still_open": 0,
                                        "errors": [],
                                        "closed_signals": [],
                                    })):
                                        result = await run_shadow_ops_once(
                                            dry_run=False,
                                            journal_path=journal,
                                            reports_output_dir=Path(tmp) / "reports",
                                        )

        self.assertEqual(result["candidate_configs_count"], 0)
        self.assertEqual(result["research_config_diagnostics"]["source_rows"], 64)
        summary = result["generation_cycle"]["generation_summary"]
        self.assertEqual(summary["configs_scanned"], 0)
        self.assertEqual(summary["status_counts"]["no_selectable_configs"], 1)


class RailwayResearchConfigDiagnosticsTests(unittest.TestCase):
    def test_supabase_candidate_configs_filters_completed_watchlist_by_symbol(self):
        rows = [
            {"config_id": "ada", "source": "crypto_multi", "status": "completed", "classification": "unstable_watchlist", "symbol": "ADA", "config": {"symbol": "ADA"}},
            {"config_id": "btc", "source": "crypto_multi", "status": "completed", "classification": "unstable_watchlist", "symbol": "BTC", "config": {"symbol": "BTC"}},
            {"config_id": "reject", "source": "crypto_multi", "status": "completed", "classification": "multi_window_reject", "symbol": "ETH", "config": {"symbol": "ETH"}},
            {"config_id": "running", "source": "crypto_multi", "status": "running", "classification": "unstable_watchlist", "symbol": "SOL", "config": {"symbol": "SOL"}},
        ]
        with patch("scripts.run_shadow_ops_once._fetch_supabase_research_config_rows", return_value=rows):
            configs = supabase_candidate_configs(
                Mock(),
                symbols=["ADA", "ETH", "SOL"],
                allow_watchlist_shadow=True,
            )
            diagnostics = supabase_research_config_diagnostics(
                Mock(),
                symbols=["ADA", "ETH", "SOL"],
                allow_watchlist_shadow=True,
            )

        self.assertEqual([row["config_id"] for row in configs], ["ada"])
        self.assertEqual(diagnostics["source_rows"], 4)
        self.assertEqual(diagnostics["completed_rows"], 3)
        self.assertEqual(diagnostics["eligible_classification_rows"], 2)
        self.assertEqual(diagnostics["symbol_filtered_rows"], 1)

    def test_evaluation_error_summary_counts_codes_and_categories(self):
        summary = summarize_evaluation_errors([
            {
                "shadow_signal_id": "sig1",
                "symbol": "ADA",
                "timeframe": "1h",
                "error_type": "TimeoutError",
                "error_code": "network_error_retry_later",
                "error_category": "timeout",
                "signal_left_open": True,
                "error": "TimeoutError: request timed out",
            },
            {
                "shadow_signal_id": "sig2",
                "symbol": "ETH",
                "timeframe": "1h",
                "error_type": "ConnectError",
                "error_code": "network_error_retry_later",
                "error_category": "network",
                "signal_left_open": True,
                "error": "ConnectError: network down",
            },
        ])

        self.assertEqual(summary["count"], 2)
        self.assertEqual(summary["error_code_counts"]["network_error_retry_later"], 2)
        self.assertEqual(summary["error_category_counts"]["timeout"], 1)
        self.assertEqual(summary["samples"][0]["signal_left_open"], True)


if __name__ == "__main__":
    unittest.main()
