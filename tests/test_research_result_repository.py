import json
import tempfile
import unittest
from pathlib import Path

from tools.research_result_repository import ResearchResultRepository, normalize_research_config_for_store


def registry_row(config_id="cfg1", classification="unstable_watchlist"):
    return {
        "config_id": config_id,
        "status": "completed",
        "classification": classification,
        "config": {
            "symbol": "ADA",
            "timeframe": "1h",
            "strategy_mode": "xgboost",
            "horizon_candles": 12,
            "risk_reward": 2.0,
            "atr_stop_multiplier": 1.25,
            "cost_mode": "low_costs",
        },
        "aggregate": {
            "median_validation_pf": 1.1,
            "median_validation_avg_return": 0.05,
            "median_test_pf": 0.9,
            "validation_positive_rate": 0.5,
            "beats_random_rate": 0.5,
            "beats_deterministic_rate": 0.4,
            "valid_windows": 5,
        },
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

    def eq(self, *args, **kwargs):
        return self

    def order(self, *args, **kwargs):
        return self

    def limit(self, *args, **kwargs):
        return self

    def upsert(self, rows, **kwargs):
        self.parent.upserts.append((self.name, rows, kwargs))
        self.rows.extend(rows)
        return self

    def execute(self):
        return FakeResult(self.parent.data.get(self.name, self.rows))


class FakeSupabase:
    def __init__(self):
        self.upserts = []
        self.data = {}

    def table(self, name):
        return FakeTable(name, self)


class ResearchResultRepositoryTests(unittest.TestCase):
    def test_normalize_research_config_for_store(self):
        row = normalize_research_config_for_store(registry_row(), source="crypto_multi")

        self.assertEqual(row["config_id"], "cfg1")
        self.assertEqual(row["symbol"], "ADA")
        self.assertEqual(row["classification"], "unstable_watchlist")
        self.assertEqual(row["median_validation_pf"], 1.1)
        self.assertIn("ADA 1h h12", row["label"])

    def test_sync_local_to_supabase_upserts_configs(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "registry.jsonl"
            row = registry_row()
            registry.write_text(json.dumps(row) + "\n", encoding="utf-8")
            supabase = FakeSupabase()
            repo = ResearchResultRepository(supabase_client=supabase, registry_path=registry)

            result = repo.sync_local_to_supabase()

        self.assertTrue(result["ok"])
        self.assertEqual(result["configs_upserted"], 1)
        self.assertEqual(supabase.upserts[0][0], "research_configs")

    def test_summary_uses_supabase_rows_when_available(self):
        supabase = FakeSupabase()
        supabase.data["research_configs"] = [normalize_research_config_for_store(registry_row())]
        repo = ResearchResultRepository(supabase_client=supabase)

        summary = repo.summary()

        self.assertEqual(summary["summary"]["total_configs"], 1)
        self.assertEqual(summary["records"][0]["symbol"], "ADA")


if __name__ == "__main__":
    unittest.main()
