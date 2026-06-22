from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from scripts.run_shadow_ops_once import (
    SHADOW_OPS_LOCK_NAME,
    acquire_shadow_ops_lock,
    release_shadow_ops_lock,
)


class FakeResult:
    def __init__(self, data=None):
        self.data = data or []


class FakeTable:
    def __init__(self, store):
        self.store = store
        self.filters = {}
        self.action = None
        self.payload = None

    def insert(self, payload):
        self.action = "insert"
        self.payload = payload
        return self

    def upsert(self, payload, on_conflict=None):
        self.action = "upsert"
        self.payload = payload
        return self

    def select(self, columns):
        self.action = "select"
        return self

    def delete(self):
        self.action = "delete"
        return self

    def eq(self, key, value):
        self.filters[key] = value
        return self

    def limit(self, value):
        return self

    def execute(self):
        if self.action == "insert":
            lock_name = self.payload["lock_name"]
            if lock_name in self.store:
                raise RuntimeError("duplicate key")
            self.store[lock_name] = dict(self.payload)
            return FakeResult([self.store[lock_name]])
        if self.action == "upsert":
            self.store[self.payload["lock_name"]] = dict(self.payload)
            return FakeResult([self.store[self.payload["lock_name"]]])
        if self.action == "select":
            row = self.store.get(self.filters.get("lock_name"))
            return FakeResult([row] if row else [])
        if self.action == "delete":
            row = self.store.get(self.filters.get("lock_name"))
            if row and row.get("owner_id") == self.filters.get("owner_id"):
                del self.store[self.filters["lock_name"]]
            return FakeResult([])
        return FakeResult([])


class FakeSupabase:
    def __init__(self):
        self.store = {}

    def table(self, name):
        assert name == "shadow_ops_locks"
        return FakeTable(self.store)


class ShadowOpsLockTests(unittest.TestCase):
    def test_lock_is_acquired(self):
        supabase = FakeSupabase()

        result = acquire_shadow_ops_lock(supabase, owner_id="host_1", cycle_id="cycle_1")

        self.assertTrue(result["acquired"])
        self.assertEqual(result["reason"], "inserted")
        self.assertIn(SHADOW_OPS_LOCK_NAME, supabase.store)

    def test_lock_is_released(self):
        supabase = FakeSupabase()
        acquire_shadow_ops_lock(supabase, owner_id="host_1", cycle_id="cycle_1")

        result = release_shadow_ops_lock(supabase, owner_id="host_1")

        self.assertTrue(result["released"])
        self.assertNotIn(SHADOW_OPS_LOCK_NAME, supabase.store)

    def test_orphan_lock_is_overwritten(self):
        supabase = FakeSupabase()
        old = datetime.now(timezone.utc) - timedelta(minutes=20)
        supabase.store[SHADOW_OPS_LOCK_NAME] = {
            "lock_name": SHADOW_OPS_LOCK_NAME,
            "owner_id": "dead_1",
            "acquired_at": old.isoformat(),
            "expires_at": (old + timedelta(minutes=10)).isoformat(),
            "heartbeat_at": old.isoformat(),
            "cycle_id": "old_cycle",
            "metadata": {},
        }

        result = acquire_shadow_ops_lock(supabase, owner_id="host_2", cycle_id="cycle_2")

        self.assertTrue(result["acquired"])
        self.assertEqual(result["reason"], "stale_lock_overwritten")
        self.assertEqual(supabase.store[SHADOW_OPS_LOCK_NAME]["owner_id"], "host_2")

    def test_two_processes_cannot_hold_lock(self):
        supabase = FakeSupabase()

        first = acquire_shadow_ops_lock(supabase, owner_id="host_1", cycle_id="cycle_1")
        second = acquire_shadow_ops_lock(supabase, owner_id="host_2", cycle_id="cycle_2")

        self.assertTrue(first["acquired"])
        self.assertFalse(second["acquired"])
        self.assertEqual(second["reason"], "active_lock")
        self.assertEqual(supabase.store[SHADOW_OPS_LOCK_NAME]["owner_id"], "host_1")


if __name__ == "__main__":
    unittest.main()
