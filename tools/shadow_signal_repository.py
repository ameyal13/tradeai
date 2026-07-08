"""Shadow signal repository with local JSONL fallback and Supabase sync.

The shadow journal remains append-only locally. Supabase is optional and used
only when explicitly configured by the backend/service role. This module never
places orders and never changes strategy selection.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.summarize_shadow_signals import build_shadow_summary, group_summaries, load_latest_shadow_rows, summarize_rows
from tools.shadow_signal_journal import DEFAULT_SHADOW_JOURNAL_PATH, OPEN, shadow_signals_are_similar


SHADOW_SIGNAL_COLUMNS = {
    "shadow_signal_id",
    "config_id",
    "source_registry",
    "classification",
    "symbol",
    "timeframe",
    "strategy_mode",
    "side",
    "entry_price",
    "stop_loss",
    "take_profit",
    "risk_reward",
    "horizon_candles",
    "horizon_minutes",
    "confidence",
    "generated_at",
    "expires_at",
    "status",
    "outcome",
    "exit_price",
    "exit_reason",
    "pnl_pct",
    "pnl_amount",
    "commission_pct",
    "slippage_pct",
    "spread_pct",
    "mfe_pct",
    "mae_pct",
    "notes",
    "input_features",
    "agent_review",
    "news_context",
    "market_context",
    "model_provider",
    "model_name",
    "research_only",
    "watchlist_shadow",
    "updated_at",
    "raw",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _finite_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _int_or_none(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _json_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def is_strategy_eligible_shadow_signal(row: dict[str, Any]) -> bool:
    """Return True for outcomes that represent actual strategy evidence.

    Technical expirations such as historical evaluation HTTP errors are kept in
    the operational journal, but they should not be mixed into strategy metrics.
    """
    outcome = str(row.get("outcome") or "").upper()
    exit_reason = str(row.get("exit_reason") or "")
    return outcome not in {"EXPIRED", "INVALID", ""} and exit_reason != "evaluation_http_error"


def shadow_strategy_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    closed_rows = [row for row in rows if row.get("status") in {"CLOSED", "EXPIRED"}]
    eligible_rows = [row for row in closed_rows if is_strategy_eligible_shadow_signal(row)]
    technical_exclusions = [row for row in closed_rows if not is_strategy_eligible_shadow_signal(row)]
    by_exit_reason: dict[str, int] = {}
    for row in technical_exclusions:
        reason = str(row.get("exit_reason") or row.get("outcome") or "UNKNOWN")
        by_exit_reason[reason] = by_exit_reason.get(reason, 0) + 1
    return {
        "summary": summarize_rows(eligible_rows),
        "technical_exclusions": len(technical_exclusions),
        "technical_exclusions_by_exit_reason": dict(sorted(by_exit_reason.items())),
    }


def normalize_shadow_signal_for_store(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize one latest shadow row for the ``shadow_signals`` table."""
    payload = {
        "shadow_signal_id": str(row.get("shadow_signal_id")),
        "config_id": row.get("config_id"),
        "source_registry": row.get("source_registry"),
        "classification": row.get("classification"),
        "symbol": str(row.get("symbol", "")).upper() or None,
        "timeframe": row.get("timeframe"),
        "strategy_mode": row.get("strategy_mode"),
        "side": row.get("side"),
        "entry_price": _finite_float(row.get("entry_price")),
        "stop_loss": _finite_float(row.get("stop_loss")),
        "take_profit": _finite_float(row.get("take_profit")),
        "risk_reward": _finite_float(row.get("risk_reward")),
        "horizon_candles": _int_or_none(row.get("horizon_candles")),
        "horizon_minutes": _int_or_none(row.get("horizon_minutes")),
        "confidence": _finite_float(row.get("confidence")),
        "generated_at": row.get("generated_at"),
        "expires_at": row.get("expires_at"),
        "status": row.get("status"),
        "outcome": row.get("outcome"),
        "exit_price": _finite_float(row.get("exit_price")),
        "exit_reason": row.get("exit_reason"),
        "pnl_pct": _finite_float(row.get("pnl_pct")),
        "pnl_amount": _finite_float(row.get("pnl_amount")),
        "commission_pct": _finite_float(row.get("commission_pct")),
        "slippage_pct": _finite_float(row.get("slippage_pct")),
        "spread_pct": _finite_float(row.get("spread_pct")),
        "mfe_pct": _finite_float(row.get("mfe_pct")),
        "mae_pct": _finite_float(row.get("mae_pct")),
        "notes": row.get("notes"),
        "input_features": _json_dict(row.get("input_features")),
        "agent_review": _json_dict(row.get("agent_review")),
        "news_context": _json_dict(row.get("news_context")),
        "market_context": _json_dict(row.get("market_context")),
        "model_provider": row.get("model_provider"),
        "model_name": row.get("model_name"),
        "research_only": bool(row.get("research_only", True)),
        "watchlist_shadow": bool(row.get("watchlist_shadow", False)),
        "updated_at": row.get("recorded_at") or utc_now_iso(),
        "raw": row,
    }
    return {key: value for key, value in payload.items() if key in SHADOW_SIGNAL_COLUMNS}


def normalize_shadow_event_for_store(row: dict[str, Any], sequence: int) -> dict[str, Any]:
    return {
        "shadow_signal_id": str(row.get("shadow_signal_id")),
        "event_sequence": int(sequence),
        "event_type": row.get("exit_reason") or row.get("status") or "record",
        "status": row.get("status"),
        "outcome": row.get("outcome"),
        "recorded_at": row.get("recorded_at") or utc_now_iso(),
        "payload": row,
    }


class ShadowSignalRepository:
    """Read local shadow state and optionally sync it to Supabase."""

    def __init__(
        self,
        supabase_client: Any = None,
        journal_path: str | Path = DEFAULT_SHADOW_JOURNAL_PATH,
    ):
        self.supabase = supabase_client
        self.journal_path = Path(journal_path)

    def list_local_signals(
        self,
        status: str | None = None,
        symbol: str | None = None,
        limit: int = 100,
        newest_first: bool = True,
    ) -> list[dict[str, Any]]:
        rows = load_latest_shadow_rows(self.journal_path)
        if status:
            rows = [row for row in rows if str(row.get("status")) == status]
        if symbol:
            rows = [row for row in rows if str(row.get("symbol", "")).upper() == symbol.upper()]
        rows = sorted(rows, key=lambda row: str(row.get("generated_at", "")), reverse=newest_first)
        return rows[: max(0, int(limit))]

    def list_signals(
        self,
        status: str | None = None,
        symbol: str | None = None,
        limit: int = 100,
        newest_first: bool = True,
    ) -> list[dict[str, Any]]:
        if self.supabase is not None:
            try:
                query = (
                    self.supabase.table("shadow_signals")
                    .select("*")
                    .order("generated_at", desc=newest_first)
                    .limit(limit)
                )
                if status:
                    query = query.eq("status", status)
                if symbol:
                    query = query.eq("symbol", symbol.upper())
                return query.execute().data or []
            except Exception:
                pass
        return self.list_local_signals(status=status, symbol=symbol, limit=limit, newest_first=newest_first)

    def summary(self, prefer_supabase: bool = True) -> dict[str, Any]:
        if prefer_supabase and self.supabase is not None:
            try:
                rows = self.supabase.table("shadow_signals").select("*").execute().data or []
                strategy_summary = shadow_strategy_summary(rows)
                return {
                    "source": "supabase",
                    "journal_path": str(self.journal_path),
                    "summary": summarize_rows(rows),
                    "strategy_eligible": strategy_summary["summary"],
                    "technical_exclusions": strategy_summary["technical_exclusions"],
                    "technical_exclusions_by_exit_reason": strategy_summary["technical_exclusions_by_exit_reason"],
                    "by_symbol": group_summaries(rows, "symbol"),
                    "by_config": group_summaries(rows, "config_id"),
                    "by_timeframe": group_summaries(rows, "timeframe"),
                    "signals": rows,
                }
            except Exception:
                pass
        report = build_shadow_summary(self.journal_path)
        strategy_summary = shadow_strategy_summary(report.get("signals") or [])
        report["strategy_eligible"] = strategy_summary["summary"]
        report["technical_exclusions"] = strategy_summary["technical_exclusions"]
        report["technical_exclusions_by_exit_reason"] = strategy_summary["technical_exclusions_by_exit_reason"]
        report["source"] = "local_jsonl"
        return report

    def sync_local_to_supabase(self) -> dict[str, Any]:
        """Upsert latest signal state and append raw journal events to Supabase."""
        if self.supabase is None:
            return {
                "ok": False,
                "reason": "supabase_not_configured",
                "signals_upserted": 0,
                "events_upserted": 0,
                "journal_path": str(self.journal_path),
            }
        if not self.journal_path.exists():
            return {
                "ok": False,
                "reason": "journal_missing",
                "signals_upserted": 0,
                "events_upserted": 0,
                "journal_path": str(self.journal_path),
            }

        all_records = []
        for line in self.journal_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                all_records.append(json.loads(line))
        latest = load_latest_shadow_rows(self.journal_path)
        signals = [normalize_shadow_signal_for_store(row) for row in latest if row.get("shadow_signal_id")]
        events = [
            normalize_shadow_event_for_store(row, index + 1)
            for index, row in enumerate(all_records)
            if row.get("shadow_signal_id")
        ]
        if signals:
            self.supabase.table("shadow_signals").upsert(signals, on_conflict="shadow_signal_id").execute()
        if events:
            self.supabase.table("shadow_signal_events").upsert(
                events,
                on_conflict="shadow_signal_id,event_sequence",
            ).execute()
        return {
            "ok": True,
            "signals_upserted": len(signals),
            "events_upserted": len(events),
            "journal_path": str(self.journal_path),
        }


class SupabaseShadowSignalStore:
    """Supabase-backed shadow signal store with the local journal interface.

    This is used by Railway/stateless workers. Local mode continues to use the
    append-only JSONL journal.
    """

    def __init__(self, supabase_client: Any):
        if supabase_client is None:
            raise ValueError("supabase_client is required for SupabaseShadowSignalStore")
        self.supabase = supabase_client

    def list_signals(self, status: str | None = None) -> list[dict[str, Any]]:
        query = self.supabase.table("shadow_signals").select("*").order("generated_at", desc=False)
        if status:
            query = query.eq("status", status)
        return query.execute().data or []

    def has_open_signal(self, config_id: str, symbol: str, timeframe: str) -> bool:
        rows = (
            self.supabase.table("shadow_signals")
            .select("shadow_signal_id")
            .eq("status", OPEN)
            .eq("config_id", config_id)
            .eq("symbol", symbol.upper())
            .eq("timeframe", timeframe)
            .limit(1)
            .execute()
            .data
            or []
        )
        return bool(rows)

    def find_open_similar_signal(self, signal: dict[str, Any]) -> dict[str, Any] | None:
        rows = (
            self.supabase.table("shadow_signals")
            .select("*")
            .eq("status", OPEN)
            .eq("symbol", str(signal.get("symbol", "")).upper())
            .eq("timeframe", signal.get("timeframe"))
            .eq("side", signal.get("side"))
            .execute()
            .data
            or []
        )
        for row in rows:
            if shadow_signals_are_similar(row, signal):
                return row
        return None

    def _next_event_sequence(self, shadow_signal_id: str) -> int:
        rows = (
            self.supabase.table("shadow_signal_events")
            .select("event_sequence")
            .eq("shadow_signal_id", shadow_signal_id)
            .order("event_sequence", desc=True)
            .limit(1)
            .execute()
            .data
            or []
        )
        if not rows:
            return 1
        return int(rows[0].get("event_sequence") or 0) + 1

    def _upsert_signal_and_event(self, row: dict[str, Any]) -> dict[str, Any]:
        payload = normalize_shadow_signal_for_store(row)
        self.supabase.table("shadow_signals").upsert(payload, on_conflict="shadow_signal_id").execute()
        sequence = self._next_event_sequence(str(row.get("shadow_signal_id")))
        event = normalize_shadow_event_for_store(row, sequence)
        self.supabase.table("shadow_signal_events").upsert(
            event,
            on_conflict="shadow_signal_id,event_sequence",
        ).execute()
        return row

    def create_signal(self, payload: dict[str, Any]) -> dict[str, Any]:
        row = dict(payload)
        row.setdefault("status", OPEN)
        row.setdefault("outcome", None)
        row.setdefault("research_only", True)
        row.setdefault("recorded_at", utc_now_iso())
        return self._upsert_signal_and_event(row)

    def update_signal(self, signal: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
        row = dict(signal)
        row.update(updates)
        row.setdefault("research_only", True)
        row["recorded_at"] = utc_now_iso()
        return self._upsert_signal_and_event(row)
