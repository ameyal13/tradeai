"""Append-only local registry for Research Daemon runs."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VALID_STATUSES = {"pending", "running", "completed", "failed", "insufficient_data"}
FINAL_NON_RETRY_STATUSES = {"completed", "insufficient_data"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def config_id(config: dict[str, Any]) -> str:
    """Stable id for methodological config identity."""
    payload = dict(config)
    payload.pop("config_id", None)
    payload.pop("experiment_id", None)
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


class ResearchRegistry:
    """JSONL append-only registry.

    The latest row for a config_id is the current state. Older rows are kept as
    audit history.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append_record(self, record: dict[str, Any]) -> dict[str, Any]:
        status = record.get("status")
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid registry status: {status}")
        row = dict(record)
        row.setdefault("recorded_at", utc_now())
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        return row

    def load_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows

    def latest_by_config(self) -> dict[str, dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        for row in self.load_all():
            cid = str(row.get("config_id"))
            latest[cid] = row
        return latest

    def load_completed_ids(self) -> set[str]:
        return {
            cid for cid, row in self.latest_by_config().items()
            if row.get("status") == "completed"
        }

    def should_run(self, config: dict[str, Any], retry_failed: bool = False) -> bool:
        cid = config.get("config_id") or config_id(config)
        row = self.latest_by_config().get(str(cid))
        if row is None:
            return True
        status = row.get("status")
        if status in FINAL_NON_RETRY_STATUSES:
            return False
        if status == "failed":
            return bool(retry_failed)
        return False

    def filter_runnable(self, configs: list[dict[str, Any]], retry_failed: bool = False) -> list[dict[str, Any]]:
        return [config for config in configs if self.should_run(config, retry_failed=retry_failed)]

    def mark_running(self, config: dict[str, Any]) -> dict[str, Any]:
        cid = config.get("config_id") or config_id(config)
        return self.append_record({
            "config_id": cid,
            "status": "running",
            "config": config,
            "started_at": utc_now(),
            "finished_at": None,
            "classification": None,
            "json_path": None,
            "markdown_path": None,
            "error": None,
        })

    def mark_finished(
        self,
        config: dict[str, Any],
        status: str,
        classification: str | None = None,
        json_path: str | None = None,
        markdown_path: str | None = None,
        error: str | None = None,
        started_at: str | None = None,
    ) -> dict[str, Any]:
        cid = config.get("config_id") or config_id(config)
        return self.append_record({
            "config_id": cid,
            "status": status,
            "config": config,
            "started_at": started_at,
            "finished_at": utc_now(),
            "classification": classification,
            "json_path": json_path,
            "markdown_path": markdown_path,
            "error": error,
        })
