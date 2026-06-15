"""Shadow ops cycle repository for dashboard diagnostics.

This stores operational cycle summaries: how many configs were scanned, why
signals were skipped, whether Supabase sync worked, and final open/closed
counts. It never generates signals or places orders.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def data_dir() -> Path:
    return Path(os.getenv("TRADEAI_DATA_DIR", "data"))


DEFAULT_SHADOW_OPS_CYCLES_PATH = data_dir() / "shadow_ops_cycles.jsonl"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _bool(value: Any) -> bool:
    return bool(value)


def normalize_shadow_ops_cycle_for_store(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "cycle_id": str(row.get("cycle_id")),
        "started_at": row.get("started_at"),
        "finished_at": row.get("finished_at") or utc_now_iso(),
        "dry_run": _bool(row.get("dry_run")),
        "health_status": row.get("health_status"),
        "evaluated_closed": _int(row.get("evaluated_closed")),
        "evaluation_errors": _int(row.get("evaluation_errors")),
        "open_after_evaluation": _int(row.get("open_after_evaluation")),
        "generation_skipped_reason": row.get("generation_skipped_reason"),
        "opened_signals": _int(row.get("opened_signals")),
        "configs_scanned": _int(row.get("configs_scanned")),
        "skipped_hold": _int(row.get("skipped_hold")),
        "skipped_duplicate_open": _int(row.get("skipped_duplicate_open")),
        "skipped_duplicate_similar": _int(row.get("skipped_duplicate_similar")),
        "skipped_errors": _int(row.get("skipped_errors")),
        "status_counts": row.get("status_counts") if isinstance(row.get("status_counts"), dict) else {},
        "final_open": _int(row.get("final_open")),
        "final_closed": _int(row.get("final_closed")),
        "sync_supabase": _bool(row.get("sync_supabase")),
        "supabase_sync_ok": _bool(row.get("supabase_sync_ok")),
        "supabase_sync_reason": row.get("supabase_sync_reason"),
        "research_only": True,
        "raw": row.get("raw") if isinstance(row.get("raw"), dict) else row,
    }


class ShadowOpsCycleRepository:
    def __init__(self, supabase_client: Any = None, path: str | Path = DEFAULT_SHADOW_OPS_CYCLES_PATH):
        self.supabase = supabase_client
        self.path = Path(path)

    def append_cycle(self, row: dict[str, Any]) -> dict[str, Any]:
        payload = normalize_shadow_ops_cycle_for_store(row)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
        return payload

    def list_local_cycles(self, limit: int = 20) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        rows.sort(key=lambda item: str(item.get("finished_at") or item.get("started_at") or ""), reverse=True)
        return rows[: max(0, int(limit))]

    def list_cycles(self, limit: int = 20) -> list[dict[str, Any]]:
        if self.supabase is not None:
            try:
                return (
                    self.supabase.table("shadow_ops_cycles")
                    .select("*")
                    .order("finished_at", desc=True)
                    .limit(max(1, int(limit)))
                    .execute()
                    .data
                    or []
                )
            except Exception:
                pass
        return self.list_local_cycles(limit=limit)

    def sync_local_to_supabase(self) -> dict[str, Any]:
        if self.supabase is None:
            return {"ok": False, "reason": "supabase_not_configured", "cycles_upserted": 0, "path": str(self.path)}
        if not self.path.exists():
            return {"ok": False, "reason": "cycles_missing", "cycles_upserted": 0, "path": str(self.path)}
        rows = self.list_local_cycles(limit=10_000)
        payload = [normalize_shadow_ops_cycle_for_store(row) for row in rows if row.get("cycle_id")]
        if payload:
            self.supabase.table("shadow_ops_cycles").upsert(payload, on_conflict="cycle_id").execute()
        return {"ok": True, "reason": None, "cycles_upserted": len(payload), "path": str(self.path)}
