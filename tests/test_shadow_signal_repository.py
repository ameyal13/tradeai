import json
import tempfile
import unittest
from pathlib import Path

from tools.shadow_signal_journal import ShadowSignalJournal
from tools.shadow_signal_repository import (
    ShadowSignalRepository,
    normalize_shadow_event_for_store,
    normalize_shadow_signal_for_store,
)
from scripts.sync_shadow_journal_to_supabase import build_parser


def shadow_row(signal_id="sig1", status="OPEN", outcome=None, pnl=None):
    return {
        "shadow_signal_id": signal_id,
        "config_id": "cfg1",
        "source_registry": "crypto_multi_registry",
        "classification": "unstable_watchlist",
        "symbol": "ada",
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
        "status": status,
        "outcome": outcome,
        "pnl_pct": pnl,
        "input_features": {"probability_buy_win": 0.6},
        "agent_review": {"review_status": "APPROVE"},
        "news_context": {"item_count": 0},
        "market_context": {"context_status": "APPROVE"},
        "research_only": True,
        "watchlist_shadow": True,
    }


class FakeResult:
    def __init__(self, data=None):
        self.data = data or []


class FakeTable:
    def __init__(self, name, parent):
        self.name = name
        self.parent = parent
        self.rows = []

    def select(self, *args, **kwargs):
        return self

    def order(self, *args, **kwargs):
        return self

    def limit(self, *args, **kwargs):
        return self

    def eq(self, *args, **kwargs):
        return self

    def upsert(self, rows, **kwargs):
        self.rows.extend(rows)
        self.parent.upserts.append((self.name, rows, kwargs))
        return self

    def execute(self):
        return FakeResult(self.parent.data.get(self.name, self.rows))


class FakeSupabase:
    def __init__(self):
        self.upserts = []
        self.data = {}

    def table(self, name):
        return FakeTable(name, self)


class ShadowSignalRepositoryTests(unittest.TestCase):
    def test_normalize_shadow_signal_for_store(self):
        normalized = normalize_shadow_signal_for_store(shadow_row())

        self.assertEqual(normalized["shadow_signal_id"], "sig1")
        self.assertEqual(normalized["symbol"], "ADA")
        self.assertEqual(normalized["entry_price"], 1.0)
        self.assertEqual(normalized["input_features"]["probability_buy_win"], 0.6)
        self.assertEqual(normalized["raw"]["shadow_signal_id"], "sig1")

    def test_normalize_shadow_event_for_store(self):
        event = normalize_shadow_event_for_store(shadow_row(status="CLOSED", outcome="WIN", pnl=1.0), 2)

        self.assertEqual(event["shadow_signal_id"], "sig1")
        self.assertEqual(event["event_sequence"], 2)
        self.assertEqual(event["status"], "CLOSED")
        self.assertEqual(event["outcome"], "WIN")

    def test_local_list_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal_path = Path(tmp) / "shadow.jsonl"
            journal = ShadowSignalJournal(journal_path)
            journal.append(shadow_row("sig1", status="OPEN"))
            journal.append(shadow_row("sig2", status="CLOSED", outcome="LOSS", pnl=-1.0))
            repo = ShadowSignalRepository(journal_path=journal_path)

            open_rows = repo.list_signals(status="OPEN")
            summary = repo.summary(prefer_supabase=False)

        self.assertEqual(len(open_rows), 1)
        self.assertEqual(open_rows[0]["shadow_signal_id"], "sig1")
        self.assertEqual(summary["summary"]["open"], 1)
        self.assertEqual(summary["summary"]["closed"], 1)

    def test_sync_local_to_supabase_upserts_latest_and_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal_path = Path(tmp) / "shadow.jsonl"
            journal = ShadowSignalJournal(journal_path)
            journal.append(shadow_row("sig1", status="OPEN"))
            journal.append(shadow_row("sig1", status="CLOSED", outcome="WIN", pnl=1.0))
            supabase = FakeSupabase()
            repo = ShadowSignalRepository(supabase_client=supabase, journal_path=journal_path)

            result = repo.sync_local_to_supabase()

        self.assertTrue(result["ok"])
        self.assertEqual(result["signals_upserted"], 1)
        self.assertEqual(result["events_upserted"], 2)
        table_names = [item[0] for item in supabase.upserts]
        self.assertEqual(table_names, ["shadow_signals", "shadow_signal_events"])

    def test_sync_without_supabase_is_safe(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal_path = Path(tmp) / "shadow.jsonl"
            journal_path.write_text(json.dumps(shadow_row()) + "\n", encoding="utf-8")
            repo = ShadowSignalRepository(journal_path=journal_path)

            result = repo.sync_local_to_supabase()

        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "supabase_not_configured")

    def test_sync_cli_parser_and_env_client_are_safe_without_env(self):
        args = build_parser().parse_args(["--journal-path", "data/test.jsonl", "--dry-run"])

        self.assertEqual(args.journal_path, "data/test.jsonl")
        self.assertTrue(args.dry_run)


if __name__ == "__main__":
    unittest.main()
