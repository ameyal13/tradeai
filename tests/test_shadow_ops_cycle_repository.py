import tempfile
import unittest
from pathlib import Path

from tools.shadow_ops_cycle_repository import (
    ShadowOpsCycleRepository,
    normalize_shadow_ops_cycle_for_store,
)


def cycle_row(cycle_id="cycle1", opened=0, hold=3, finished_at="2026-01-01T00:01:00+00:00"):
    return {
        "cycle_id": cycle_id,
        "started_at": "2026-01-01T00:00:00+00:00",
        "finished_at": finished_at,
        "dry_run": False,
        "health_status": "HEALTH_OK",
        "evaluated_closed": 1,
        "evaluation_errors": 0,
        "open_after_evaluation": 0,
        "generation_skipped_reason": None,
        "opened_signals": opened,
        "configs_scanned": 5,
        "skipped_hold": hold,
        "skipped_duplicate_open": 0,
        "skipped_duplicate_similar": 0,
        "skipped_errors": 0,
        "status_counts": {"skipped_hold": hold, "OPEN": opened},
        "final_open": opened,
        "final_closed": 6,
        "sync_supabase": True,
        "supabase_sync_ok": True,
        "supabase_sync_reason": None,
        "research_only": True,
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


class ShadowOpsCycleRepositoryTests(unittest.TestCase):
    def test_normalize_cycle_for_store(self):
        row = normalize_shadow_ops_cycle_for_store(cycle_row(opened=1, hold=4))

        self.assertEqual(row["cycle_id"], "cycle1")
        self.assertEqual(row["opened_signals"], 1)
        self.assertEqual(row["skipped_hold"], 4)
        self.assertTrue(row["research_only"])

    def test_append_and_list_local_cycles(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cycles.jsonl"
            repo = ShadowOpsCycleRepository(path=path)
            repo.append_cycle(cycle_row("cycle1"))
            repo.append_cycle(cycle_row("cycle2", opened=1, hold=0, finished_at="2026-01-01T00:02:00+00:00"))

            rows = repo.list_local_cycles(limit=10)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["cycle_id"], "cycle2")

    def test_sync_local_to_supabase_upserts_cycles(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cycles.jsonl"
            supabase = FakeSupabase()
            repo = ShadowOpsCycleRepository(supabase_client=supabase, path=path)
            repo.append_cycle(cycle_row("cycle1"))

            result = repo.sync_local_to_supabase()

        self.assertTrue(result["ok"])
        self.assertEqual(result["cycles_upserted"], 1)
        self.assertEqual(supabase.upserts[0][0], "shadow_ops_cycles")

    def test_sync_without_supabase_is_safe(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cycles.jsonl"
            repo = ShadowOpsCycleRepository(path=path)
            repo.append_cycle(cycle_row())

            result = repo.sync_local_to_supabase()

        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "supabase_not_configured")


if __name__ == "__main__":
    unittest.main()
