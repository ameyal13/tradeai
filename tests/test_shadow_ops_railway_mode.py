from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from scripts.run_shadow_ops_once import run_shadow_ops_once
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


if __name__ == "__main__":
    unittest.main()
