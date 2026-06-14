"""Offline replay of the local shadow-ops policy.

This is research-only. It simulates the current shadow scheduler behavior over
historical candles without writing the live shadow journal and without placing
orders.
"""
from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import pandas as pd

from scripts.generate_shadow_signals_once import (
    classify_signal_skip,
    load_candidate_configs,
    registry_path_from_choice,
    strategy_params_from_config,
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


from tools.shadow_signal_journal import (
    CLOSED,
    OPEN,
    build_shadow_signal_from_strategy,
    cost_profile_for_config,
    evaluate_shadow_signal_with_candles,
    horizon_minutes_from_candles,
    shadow_signals_are_similar,
)
from tools.strategy_signals import generate_strategy_signal_from_df


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _timeframe_minutes(timeframe: str) -> int:
    value = str(timeframe).strip().lower()
    if value.endswith("m"):
        return int(value[:-1])
    if value.endswith("h"):
        return int(value[:-1]) * 60
    if value.endswith("d"):
        return int(value[:-1]) * 1440
    raise ValueError(f"Unsupported timeframe: {timeframe}")


def _normalize_candles(candles: pd.DataFrame) -> pd.DataFrame:
    df = candles.copy()
    if "timestamp" not in df.columns:
        df = df.reset_index().rename(columns={"index": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)


def _prediction_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    closed = [row for row in rows if row.get("status") in {CLOSED, "EXPIRED"}]
    wins = [row for row in closed if row.get("outcome") == "WIN"]
    losses = [row for row in closed if row.get("outcome") == "LOSS"]
    expired = [row for row in closed if row.get("outcome") == "EXPIRED" or row.get("status") == "EXPIRED"]
    returns = [_safe_float(row.get("pnl_pct")) or 0.0 for row in closed]
    profits = sum(value for value in returns if value > 0)
    loss_sum = abs(sum(value for value in returns if value < 0))
    equity = 100.0
    peak = equity
    max_drawdown = 0.0
    for value in returns:
        equity *= 1 + value / 100
        peak = max(peak, equity)
        if peak:
            max_drawdown = max(max_drawdown, (peak - equity) / peak * 100)
    return {
        "total": len(rows),
        "open": sum(1 for row in rows if row.get("status") == OPEN),
        "closed": len(closed),
        "wins": len(wins),
        "losses": len(losses),
        "expired": len(expired),
        "win_rate": round(len(wins) / len(closed) * 100, 6) if closed else 0.0,
        "profit_factor": round(profits / loss_sum, 6) if loss_sum else None,
        "avg_return": round(sum(returns) / len(returns), 6) if returns else 0.0,
        "total_return_pct": round(sum(returns), 6),
        "max_drawdown": round(max_drawdown, 6),
    }


def _group_metrics(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key) or "unknown")].append(row)
    return {name: _prediction_metrics(group) for name, group in sorted(grouped.items())}


def _status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(row.get("status")) for row in rows))


def _as_signal_row(
    config: dict[str, Any],
    classification: str,
    signal: dict[str, Any],
    generated_at: pd.Timestamp,
) -> dict[str, Any] | None:
    shadow = build_shadow_signal_from_strategy(
        config=config,
        source_registry="shadow_replay",
        classification=classification,
        signal=signal,
        costs=cost_profile_for_config(config),
        watchlist_shadow=classification == "unstable_watchlist",
        notes="shadow_replay_simulated",
    )
    if shadow is None:
        return None
    horizon_minutes = horizon_minutes_from_candles(int(config["horizon_candles"]), str(config["timeframe"]))
    generated = generated_at.to_pydatetime()
    shadow["shadow_signal_id"] = str(uuid4())
    shadow["generated_at"] = generated.isoformat()
    shadow["expires_at"] = (generated + timedelta(minutes=horizon_minutes)).isoformat()
    shadow["replay_generated_at"] = generated.isoformat()
    return shadow


def _evaluate_open_signals(
    open_signals: list[dict[str, Any]],
    candles_until_now: pd.DataFrame,
    now: pd.Timestamp,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    still_open: list[dict[str, Any]] = []
    closed: list[dict[str, Any]] = []
    for signal in open_signals:
        updates = evaluate_shadow_signal_with_candles(signal, candles_until_now, now=now.to_pydatetime())
        if updates is None:
            still_open.append(signal)
            continue
        row = dict(signal)
        row.update(updates)
        closed.append(row)
    return still_open, closed


def _config_label(config: dict[str, Any]) -> str:
    return (
        f"{config.get('symbol')} {config.get('timeframe')} "
        f"h{config.get('horizon_candles')} RR{config.get('risk_reward')} "
        f"ATR{config.get('atr_stop_multiplier')} {config.get('cost_mode')}"
    )


def run_shadow_replay_for_candles(
    *,
    candles: pd.DataFrame,
    configs: list[dict[str, Any]],
    symbol: str,
    timeframe: str,
    days: int = 60,
    max_signals: int = 1,
    max_configs_scanned: int = 21,
    cycle_step_candles: int = 1,
    min_history_candles: int = 300,
    use_sentiment: bool = False,
    max_cycles: int | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    max_runtime_seconds: int | None = None,
) -> dict[str, Any]:
    """Replay current shadow policy on one symbol/timeframe candle set."""
    df = _normalize_candles(candles)
    interval_minutes = _timeframe_minutes(timeframe)
    requested_cycle_count = max(1, int(days * 1440 / interval_minutes))
    start_index = max(min_history_candles, len(df) - requested_cycle_count)
    cycle_indices = list(range(start_index, max(start_index, len(df) - 1), max(1, int(cycle_step_candles))))
    if max_cycles is not None:
        cycle_indices = cycle_indices[: max(0, int(max_cycles))]

    selected_configs = [
        config for config in configs
        if str(config.get("symbol", "")).upper() == symbol.upper()
        and str(config.get("timeframe", timeframe)) == timeframe
    ]
    events: list[dict[str, Any]] = []
    signals: list[dict[str, Any]] = []
    open_signals: list[dict[str, Any]] = []
    skipped_because_open = 0
    cycles_with_generation_attempt = 0
    completed_cycles = 0
    stop_reason: str | None = None
    started_monotonic = time.monotonic()

    def emit_progress(cycle_number: int, now: pd.Timestamp) -> None:
        if progress_callback is None:
            return
        progress_callback({
            "symbol": symbol.upper(),
            "timeframe": timeframe,
            "cycle": cycle_number,
            "total_cycles": len(cycle_indices),
            "timestamp": now.isoformat(),
            "elapsed_seconds": round(time.monotonic() - started_monotonic, 3),
            "open_count": len(open_signals),
            "summary": _prediction_metrics(signals + open_signals),
            "event_status_counts": _status_counts(events),
            "last_event_status": events[-1].get("status") if events else None,
            "stop_reason": stop_reason,
        })

    for cycle_number, idx in enumerate(cycle_indices, start=1):
        if max_runtime_seconds is not None and time.monotonic() - started_monotonic >= max_runtime_seconds:
            stop_reason = "max_runtime_seconds"
            break
        now = pd.Timestamp(df.loc[idx, "timestamp"])
        candles_until_now = df.iloc[: idx + 1].copy()
        open_signals, newly_closed = _evaluate_open_signals(open_signals, candles_until_now, now)
        signals.extend(newly_closed)
        completed_cycles = cycle_number
        if open_signals:
            skipped_because_open += 1
            events.append({
                "cycle": cycle_number,
                "timestamp": now.isoformat(),
                "status": "skipped_open_exists",
                "open_count": len(open_signals),
            })
            emit_progress(cycle_number, now)
            continue
        if max_signals <= 0:
            emit_progress(cycle_number, now)
            continue
        cycles_with_generation_attempt += 1
        opened_this_cycle: list[dict[str, Any]] = []
        attempted = 0
        for config in selected_configs:
            if len(opened_this_cycle) >= max_signals:
                break
            if attempted >= max_configs_scanned:
                break
            attempted += 1
            base = {
                "cycle": cycle_number,
                "timestamp": now.isoformat(),
                "config_id": config.get("config_id"),
                "config_label": _config_label(config),
                "symbol": symbol.upper(),
                "timeframe": timeframe,
                "classification": config.get("_source_classification"),
            }
            try:
                horizon_minutes = horizon_minutes_from_candles(int(config["horizon_candles"]), timeframe)
                params = strategy_params_from_config(config)
                params["use_sentiment"] = bool(use_sentiment)
                signal = generate_strategy_signal_from_df(
                    candles_until_now,
                    strategy_mode=str(config.get("strategy_mode", "xgboost")),
                    provider="none",
                    horizon_minutes=horizon_minutes,
                    strategy_params=params,
                ).to_dict()
                skip = classify_signal_skip(signal)
                if skip:
                    events.append({**base, **skip})
                    continue
                shadow = _as_signal_row(config, str(config.get("_source_classification")), signal, now)
                if shadow is None:
                    events.append({**base, "status": "skipped_invalid_levels", "skip_reason": "entry/SL/TP invalidos"})
                    continue
                if any(shadow_signals_are_similar(existing, shadow) for existing in opened_this_cycle):
                    events.append({**base, "status": "skipped_duplicate_open_similar"})
                    continue
                shadow["status"] = OPEN
                shadow["dry_run"] = True
                opened_this_cycle.append(shadow)
                open_signals.append(shadow)
                events.append({**base, "status": OPEN, "side": shadow.get("side"), "confidence": shadow.get("confidence")})
            except Exception as exc:  # noqa: BLE001 - one config should not stop replay.
                events.append({
                    **base,
                    "status": "skipped_error",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)[:180],
                })
        if opened_this_cycle:
            emit_progress(cycle_number, now)
            continue
        emit_progress(cycle_number, now)

    if cycle_indices:
        final_index = cycle_indices[-1]
        final_now = pd.Timestamp(df.loc[final_index, "timestamp"])
        final_candles = df.iloc[: final_index + 1].copy()
        open_signals, newly_closed = _evaluate_open_signals(open_signals, final_candles, final_now)
        signals.extend(newly_closed)
    signals.extend(open_signals)

    closed_signals = [row for row in signals if row.get("status") in {CLOSED, "EXPIRED"}]
    return {
        "symbol": symbol.upper(),
        "timeframe": timeframe,
        "days": days,
        "rows": len(df),
        "start_index": start_index,
        "cycles": completed_cycles,
        "planned_cycles": len(cycle_indices),
        "stop_reason": stop_reason,
        "cycle_step_candles": cycle_step_candles,
        "min_history_candles": min_history_candles,
        "max_signals": max_signals,
        "max_configs_scanned": max_configs_scanned,
        "selected_config_count": len(selected_configs),
        "cycles_with_generation_attempt": cycles_with_generation_attempt,
        "cycles_skipped_open_exists": skipped_because_open,
        "event_status_counts": _status_counts(events),
        "summary": _prediction_metrics(signals),
        "by_symbol": _group_metrics(signals, "symbol"),
        "by_config": _group_metrics(signals, "config_id"),
        "by_side": _group_metrics(signals, "side"),
        "signals": signals,
        "closed_signals": closed_signals,
        "events": events,
    }


def combine_replay_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    signals = [signal for report in reports for signal in report.get("signals", [])]
    events = [event for report in reports for event in report.get("events", [])]
    return {
        "summary": _prediction_metrics(signals),
        "by_symbol": _group_metrics(signals, "symbol"),
        "by_config": _group_metrics(signals, "config_id"),
        "by_side": _group_metrics(signals, "side"),
        "event_status_counts": _status_counts(events),
        "signals": signals,
        "events": events,
    }


def load_shadow_replay_configs(
    registry: str = "crypto_multi",
    symbols: list[str] | None = None,
    allow_watchlist_shadow: bool = True,
) -> list[dict[str, Any]]:
    registry_path = PROJECT_ROOT / "reports" / "research_daemon" / "crypto_multi_registry.jsonl" if registry == "crypto_multi" else registry_path_from_choice(registry)
    return load_candidate_configs(
        registry_path,
        symbols=symbols,
        allow_watchlist_shadow=allow_watchlist_shadow,
        min_classification="stable_research_candidate",
        include_not_allowed=True,
    )


def render_shadow_replay_markdown(report: dict[str, Any]) -> str:
    summary = report["combined"]["summary"]
    lines = [
        "# Shadow Replay Audit",
        "",
        "Research only. No trading signal. No exchange orders.",
        "",
        "## Overall",
        "",
        f"- total simulated signals: `{summary.get('total')}`",
        f"- open at replay end: `{summary.get('open')}`",
        f"- closed: `{summary.get('closed')}`",
        f"- wins/losses/expired: `{summary.get('wins')}` / `{summary.get('losses')}` / `{summary.get('expired')}`",
        f"- win rate: `{summary.get('win_rate')}`",
        f"- profit factor: `{summary.get('profit_factor')}`",
        f"- avg return: `{summary.get('avg_return')}`",
        f"- total return pct: `{summary.get('total_return_pct')}`",
        f"- max drawdown: `{summary.get('max_drawdown')}`",
        "",
        "## Policy",
        "",
        f"- registry: `{report['config'].get('registry')}`",
        f"- symbols: `{', '.join(report['config'].get('symbols') or [])}`",
        f"- timeframe: `{report['config'].get('timeframe')}`",
        f"- days: `{report['config'].get('days')}`",
        f"- max_signals: `{report['config'].get('max_signals')}`",
        f"- max_configs_scanned: `{report['config'].get('max_configs_scanned')}`",
        f"- use_sentiment: `{report['config'].get('use_sentiment')}`",
        "",
        "## By Symbol",
        "",
    ]
    for key, value in report["combined"].get("by_symbol", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## By Side", ""])
    for key, value in report["combined"].get("by_side", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Event Status Counts", ""])
    for key, value in sorted(report["combined"].get("event_status_counts", {}).items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Per Symbol Replay", ""])
    for item in report.get("runs", []):
        lines.append(
            f"- `{item.get('symbol')} {item.get('timeframe')}`: "
            f"cycles=`{item.get('cycles')}`, configs=`{item.get('selected_config_count')}`, "
            f"summary=`{item.get('summary')}`"
        )
    return "\n".join(lines).rstrip() + "\n"


def save_shadow_replay_report(report: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    json_path = target / f"shadow_replay_{stamp}.json"
    markdown_path = target / f"shadow_replay_{stamp}.md"
    report["json_path"] = str(json_path)
    report["markdown_path"] = str(markdown_path)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    markdown_path.write_text(render_shadow_replay_markdown(report), encoding="utf-8")
    return {"json_path": str(json_path), "markdown_path": str(markdown_path)}
