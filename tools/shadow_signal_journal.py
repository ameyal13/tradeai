"""Append-only local journal for shadow/paper signals.

This module is research-only. It never places exchange orders and delegates
outcome evaluation to ``tools.prediction_journal.evaluate_prediction_against_candles``
so WIN/LOSS/cost/TP/SL semantics stay unified.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from tools.prediction_journal import evaluate_prediction_against_candles, parse_dt, utc_now
from tools.trade_opportunity_research import DEFAULT_COST_PROFILES


DEFAULT_SHADOW_JOURNAL_PATH = Path("data") / "shadow_signal_journal.jsonl"
OPEN = "OPEN"
CLOSED = "CLOSED"
EXPIRED = "EXPIRED"
BLOCKED = "BLOCKED"
DEFAULT_SIMILAR_PRICE_REL_TOL = 0.001


def timeframe_to_minutes(timeframe: str) -> int:
    """Convert common Binance-style timeframes to minutes."""
    value = str(timeframe).strip().lower()
    if value.endswith("m"):
        return int(value[:-1])
    if value.endswith("h"):
        return int(value[:-1]) * 60
    if value.endswith("d"):
        return int(value[:-1]) * 60 * 24
    if value.endswith("w"):
        return int(value[:-1]) * 60 * 24 * 7
    raise ValueError(f"Unsupported timeframe for horizon conversion: {timeframe}")


def horizon_minutes_from_candles(horizon_candles: int, timeframe: str) -> int:
    return int(horizon_candles) * timeframe_to_minutes(timeframe)


def normalize_shadow_outcome(outcome: str | None) -> str | None:
    if outcome == "INVALID_DATA":
        return "INVALID"
    return outcome


def _json_default(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _float_or_none(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric


def _prices_very_similar(left: Any, right: Any, rel_tol: float = DEFAULT_SIMILAR_PRICE_REL_TOL) -> bool:
    left_num = _float_or_none(left)
    right_num = _float_or_none(right)
    if left_num is None or right_num is None:
        return False
    scale = max(abs(left_num), abs(right_num), 1e-12)
    return abs(left_num - right_num) <= abs(float(rel_tol)) * scale


def shadow_signals_are_similar(
    left: dict[str, Any],
    right: dict[str, Any],
    rel_tol: float = DEFAULT_SIMILAR_PRICE_REL_TOL,
) -> bool:
    """Return True when two open shadow signals represent the same exposure."""
    return (
        str(left.get("symbol", "")).upper() == str(right.get("symbol", "")).upper()
        and str(left.get("timeframe", "")) == str(right.get("timeframe", ""))
        and str(left.get("side", "")).upper() == str(right.get("side", "")).upper()
        and _prices_very_similar(left.get("entry_price"), right.get("entry_price"), rel_tol=rel_tol)
        and _prices_very_similar(left.get("stop_loss"), right.get("stop_loss"), rel_tol=rel_tol)
        and _prices_very_similar(left.get("take_profit"), right.get("take_profit"), rel_tol=rel_tol)
    )


class ShadowSignalJournal:
    """Append-only JSONL journal with latest-state reads by shadow_signal_id."""

    def __init__(self, path: str | Path = DEFAULT_SHADOW_JOURNAL_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, row: dict[str, Any]) -> dict[str, Any]:
        payload = dict(row)
        payload.setdefault("recorded_at", utc_now().isoformat())
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, default=_json_default) + "\n")
        return payload

    def load_all_records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows

    def latest_by_id(self) -> dict[str, dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        for row in self.load_all_records():
            sid = str(row.get("shadow_signal_id"))
            latest[sid] = row
        return latest

    def list_signals(self, status: str | None = None) -> list[dict[str, Any]]:
        rows = list(self.latest_by_id().values())
        if status:
            rows = [row for row in rows if row.get("status") == status]
        return sorted(rows, key=lambda row: str(row.get("generated_at", "")))

    def has_open_signal(self, config_id: str, symbol: str, timeframe: str) -> bool:
        for row in self.list_signals(status=OPEN):
            if row.get("config_id") == config_id and row.get("symbol") == symbol.upper() and row.get("timeframe") == timeframe:
                return True
        return False

    def find_open_similar_signal(
        self,
        signal: dict[str, Any],
        rel_tol: float = DEFAULT_SIMILAR_PRICE_REL_TOL,
    ) -> dict[str, Any] | None:
        for row in self.list_signals(status=OPEN):
            if shadow_signals_are_similar(row, signal, rel_tol=rel_tol):
                return row
        return None

    def create_signal(self, payload: dict[str, Any]) -> dict[str, Any]:
        row = dict(payload)
        row.setdefault("shadow_signal_id", str(uuid4()))
        row.setdefault("status", OPEN)
        row.setdefault("outcome", None)
        row.setdefault("research_only", True)
        row.setdefault("generated_at", utc_now().isoformat())
        return self.append(row)

    def update_signal(self, signal: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
        row = dict(signal)
        row.update(updates)
        return self.append(row)


def shadow_signal_to_prediction_payload(signal: dict[str, Any]) -> dict[str, Any]:
    side = str(signal.get("side", "")).upper()
    prediction_signal = "BUY" if side == "LONG" else "SELL" if side == "SHORT" else "HOLD"
    return {
        "id": signal.get("shadow_signal_id"),
        "symbol": signal.get("symbol"),
        "timeframe": signal.get("timeframe"),
        "strategy_mode": signal.get("strategy_mode"),
        "strategy_name": signal.get("strategy_name") or "shadow_signal",
        "strategy_version": signal.get("strategy_version") or "v1",
        "signal": prediction_signal,
        "confidence": signal.get("confidence", 0),
        "entry_price": signal.get("entry_price"),
        "stop_loss": signal.get("stop_loss"),
        "take_profit": signal.get("take_profit"),
        "risk_reward_ratio": signal.get("risk_reward"),
        "horizon_minutes": signal.get("horizon_minutes"),
        "input_features": signal.get("input_features") or {},
        "reasoning": signal.get("notes") or "",
        "model_provider": signal.get("model_provider"),
        "model_name": signal.get("model_name"),
        "created_at": signal.get("generated_at"),
    }


def build_shadow_signal_from_strategy(
    *,
    config: dict[str, Any],
    source_registry: str,
    classification: str,
    signal: dict[str, Any],
    costs: dict[str, float],
    watchlist_shadow: bool = False,
    notes: str = "",
) -> dict[str, Any] | None:
    """Build a journal row from a generated StrategySignal dict."""
    raw_signal = str(signal.get("signal", "HOLD")).upper()
    if raw_signal == "HOLD":
        return None
    if raw_signal not in {"BUY", "SELL"}:
        return None
    timeframe = str(config.get("timeframe") or "1h")
    horizon_candles = int(config.get("horizon_candles") or 1)
    horizon_minutes = horizon_minutes_from_candles(horizon_candles, timeframe)
    generated_at = utc_now()
    expires_at = generated_at + timedelta(minutes=horizon_minutes)
    side = "LONG" if raw_signal == "BUY" else "SHORT"
    warning = "WATCHLIST SHADOW ONLY. NO TRADING SIGNAL." if watchlist_shadow else "STABLE SHADOW. NO TRADING SIGNAL."
    return {
        "shadow_signal_id": str(uuid4()),
        "config_id": config.get("config_id") or config.get("experiment_id"),
        "source_registry": source_registry,
        "classification": classification,
        "symbol": str(config.get("symbol", "")).upper(),
        "timeframe": timeframe,
        "strategy_mode": signal.get("strategy_mode") or config.get("strategy_mode"),
        "strategy_name": signal.get("strategy_name"),
        "strategy_version": signal.get("strategy_version"),
        "side": side,
        "entry_price": signal.get("entry_price"),
        "stop_loss": signal.get("stop_loss"),
        "take_profit": signal.get("take_profit"),
        "risk_reward": signal.get("risk_reward_ratio") or config.get("risk_reward"),
        "horizon_candles": horizon_candles,
        "horizon_minutes": horizon_minutes,
        "confidence": signal.get("confidence"),
        "generated_at": generated_at.isoformat(),
        "expires_at": expires_at.isoformat(),
        "status": OPEN,
        "outcome": None,
        "exit_price": None,
        "exit_reason": None,
        "pnl_pct": None,
        "pnl_amount": None,
        "fees": None,
        "slippage": None,
        "spread": None,
        "commission_pct": float(costs.get("commission_pct", 0)),
        "slippage_pct": float(costs.get("slippage_pct", 0)),
        "spread_pct": float(costs.get("spread_pct", 0)),
        "mfe_pct": None,
        "mae_pct": None,
        "notes": " | ".join(part for part in [warning, notes, signal.get("reasoning")] if part),
        "input_features": signal.get("input_features") or {},
        "model_provider": signal.get("model_provider"),
        "model_name": signal.get("model_name"),
        "research_only": True,
        "watchlist_shadow": bool(watchlist_shadow),
    }


def evaluate_shadow_signal_with_candles(
    signal: dict[str, Any],
    candles: pd.DataFrame,
    now: datetime | None = None,
) -> dict[str, Any] | None:
    """Evaluate a shadow signal and return close updates, or None if still open."""
    now_dt = now or utc_now()
    due = now_dt >= parse_dt(signal["expires_at"])
    outcome = evaluate_prediction_against_candles(
        shadow_signal_to_prediction_payload(signal),
        candles,
        commission_pct=float(signal.get("commission_pct", 0) or 0),
        slippage_pct=float(signal.get("slippage_pct", 0) or 0),
        spread_pct=float(signal.get("spread_pct", 0) or 0),
    )
    raw_path = outcome.get("raw_path") if isinstance(outcome.get("raw_path"), dict) else {}
    exit_reason = raw_path.get("exit_reason")
    hit_exit = exit_reason in {"take_profit", "stop_loss", "ambiguous_intrabar_conservative_loss"}
    if not due and not hit_exit:
        return None
    normalized_outcome = normalize_shadow_outcome(outcome.get("outcome"))
    status = EXPIRED if normalized_outcome == "EXPIRED" else CLOSED
    if normalized_outcome == "INVALID":
        status = CLOSED
    return {
        "status": status,
        "outcome": normalized_outcome,
        "exit_price": outcome.get("exit_price"),
        "exit_reason": exit_reason or "invalid",
        "pnl_pct": outcome.get("return_pct"),
        "fees": outcome.get("fees_paid"),
        "slippage": outcome.get("slippage_cost"),
        "spread": signal.get("spread_pct"),
        "mfe_pct": outcome.get("max_favorable_excursion_pct"),
        "mae_pct": outcome.get("max_adverse_excursion_pct"),
        "evaluated_at": outcome.get("evaluated_at"),
        "evaluation_raw_path": outcome.get("raw_path"),
    }


def cost_profile_for_config(config: dict[str, Any]) -> dict[str, float]:
    cost_mode = str(config.get("cost_mode") or "medium_costs_current")
    profile = DEFAULT_COST_PROFILES.get(cost_mode)
    if profile is None:
        raise ValueError(f"Unknown cost_mode: {cost_mode}")
    return {key: float(value) for key, value in profile.items()}
