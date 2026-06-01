import tempfile
import unittest
from pathlib import Path

from research.research_registry import ResearchRegistry, config_id


def config(**overrides):
    base = {
        "symbol": "SOL",
        "timeframe": "1h",
        "max_candles": 5000,
        "window_size_candles": 600,
        "step_size_candles": 250,
        "horizon_candles": 16,
        "risk_reward": 2.0,
        "atr_stop_multiplier": 1.25,
        "cost_mode": "low_costs",
        "strategy_mode": "xgboost",
    }
    base.update(overrides)
    base["config_id"] = config_id(base)
    return base


class ResearchRegistryTests(unittest.TestCase):
    def test_config_id_is_stable(self):
        first = config()
        shuffled = dict(reversed(list(first.items())))

        self.assertEqual(config_id(first), config_id(shuffled))

    def test_config_id_changes_when_methodological_param_changes(self):
        first = config()
        second = config(horizon_candles=24)

        self.assertNotEqual(config_id(first), config_id(second))

    def test_registry_is_append_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = ResearchRegistry(Path(tmp) / "registry.jsonl")
            row = config()
            registry.mark_running(row)
            registry.mark_finished(row, status="completed", classification="multi_window_reject")
            rows = registry.load_all()

        self.assertEqual(len(rows), 2)
        self.assertEqual([item["status"] for item in rows], ["running", "completed"])

    def test_load_completed_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = ResearchRegistry(Path(tmp) / "registry.jsonl")
            row = config()
            registry.mark_finished(row, status="completed", classification="multi_window_reject")

            self.assertEqual(registry.load_completed_ids(), {row["config_id"]})

    def test_filter_does_not_repeat_completed(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = ResearchRegistry(Path(tmp) / "registry.jsonl")
            row = config()
            registry.mark_finished(row, status="completed", classification="multi_window_reject")

            self.assertEqual(registry.filter_runnable([row]), [])

    def test_retry_failed_only_with_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = ResearchRegistry(Path(tmp) / "registry.jsonl")
            row = config()
            registry.mark_finished(row, status="failed", error="boom")

            self.assertEqual(registry.filter_runnable([row], retry_failed=False), [])
            self.assertEqual(registry.filter_runnable([row], retry_failed=True), [row])


if __name__ == "__main__":
    unittest.main()
