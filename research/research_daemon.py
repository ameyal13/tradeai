"""Local Research Daemon v1.

Runs controlled research cycles over a fixed grid. This module is research-only:
no trading, paper trading, Supabase writes, endpoints, or operational signals.
"""
from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from research.experiment_grid import ExperimentConfig
from research.multi_window_validator import run_multi_window_validation_for_setups
from research.research_registry import ResearchRegistry, config_id, utc_now
from research.telegram_notifier import send_telegram_message


DEFAULT_DAEMON_DIR = Path("reports") / "research_daemon"
DEFAULT_REGISTRY_PATH = DEFAULT_DAEMON_DIR / "registry.jsonl"
DEFAULT_CYCLES_DIR = DEFAULT_DAEMON_DIR / "cycles"
DEFAULT_RESULTS_DIR = DEFAULT_DAEMON_DIR / "results"
DEFAULT_STATUS_PATH = DEFAULT_DAEMON_DIR / "current_status.json"

DEFAULT_SYMBOLS = ["SOL"]
DEFAULT_TIMEFRAMES = ["1h"]
DEFAULT_MAX_CANDLES = 5000
DEFAULT_WINDOW_SIZE_CANDLES = 600
DEFAULT_STEP_SIZE_CANDLES = 250
DEFAULT_HORIZON_CANDLES = [12, 16, 20, 24]
DEFAULT_RISK_REWARDS = [1.5, 2.0, 2.5]
DEFAULT_ATR_STOP_MULTIPLIERS = [1.0, 1.25, 1.5, 1.75]
DEFAULT_COST_MODES = ["low_costs", "medium_costs_current"]
DEFAULT_STRATEGY_MODES = ["xgboost"]


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def build_daemon_grid(
    symbols: list[str] | None = None,
    timeframes: list[str] | None = None,
    max_candles: int = DEFAULT_MAX_CANDLES,
    window_size_candles: int = DEFAULT_WINDOW_SIZE_CANDLES,
    step_size_candles: int = DEFAULT_STEP_SIZE_CANDLES,
    horizon_candles: list[int] | None = None,
    risk_rewards: list[float] | None = None,
    atr_stop_multipliers: list[float] | None = None,
    cost_modes: list[str] | None = None,
    strategy_modes: list[str] | None = None,
    min_train_rows: int = 120,
) -> list[dict[str, Any]]:
    """Build the approved Research Daemon v1 grid."""
    rows: list[dict[str, Any]] = []
    for symbol in symbols or DEFAULT_SYMBOLS:
        for timeframe in timeframes or DEFAULT_TIMEFRAMES:
            for horizon in horizon_candles or DEFAULT_HORIZON_CANDLES:
                for rr in risk_rewards or DEFAULT_RISK_REWARDS:
                    for atr_stop in atr_stop_multipliers or DEFAULT_ATR_STOP_MULTIPLIERS:
                        for cost_mode in cost_modes or DEFAULT_COST_MODES:
                            for strategy_mode in strategy_modes or DEFAULT_STRATEGY_MODES:
                                setup = ExperimentConfig(
                                    symbol=symbol.upper(),
                                    timeframe=timeframe,
                                    horizon_candles=int(horizon),
                                    risk_reward=float(rr),
                                    atr_stop_multiplier=float(atr_stop),
                                    cost_mode=cost_mode,
                                    strategy_mode=strategy_mode,
                                    max_candles=int(max_candles),
                                    min_train_rows=int(min_train_rows),
                                ).to_dict()
                                row = {
                                    **setup,
                                    "window_size_candles": int(window_size_candles),
                                    "step_size_candles": int(step_size_candles),
                                }
                                row["config_id"] = config_id(row)
                                rows.append(row)
    return rows


def _setup_from_daemon_config(config: dict[str, Any]) -> dict[str, Any]:
    keys = {
        "symbol", "timeframe", "horizon_candles", "risk_reward",
        "atr_stop_multiplier", "cost_mode", "strategy_mode", "max_candles",
        "min_train_rows", "buy_threshold", "sell_threshold", "trade_label_scheme",
        "experiment_id",
    }
    return {key: value for key, value in config.items() if key in keys}


def _classification_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        classification = str(row.get("classification", row.get("status", "unknown")))
        counts[classification] = counts.get(classification, 0) + 1
    return counts


def _safe_metric(row: dict[str, Any], name: str) -> float:
    value = ((row.get("result") or {}).get("aggregate") or {}).get(name)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float("-inf")
    return numeric if math.isfinite(numeric) else float("-inf")


def summarize_cycle(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize one daemon cycle."""
    completed = [row for row in rows if row.get("status") in {"completed", "insufficient_data"}]
    failed = [row for row in rows if row.get("status") == "failed"]
    counts = _classification_counts(completed + failed)
    return {
        "evaluated": len(rows),
        "completed": len(completed),
        "failed": len(failed),
        "classification_counts": counts,
        "stable_candidates": [row for row in completed if row.get("classification") == "stable_research_candidate"],
        "unstable_watchlist": [row for row in completed if row.get("classification") == "unstable_watchlist"],
        "rejects": [row for row in completed if row.get("classification") == "multi_window_reject"],
        "needs_more_data": [row for row in completed if row.get("classification") == "needs_more_data"],
        "top_validation_pf": sorted(completed, key=lambda row: _safe_metric(row, "median_validation_pf"), reverse=True)[:5],
        "top_validation_avg": sorted(completed, key=lambda row: _safe_metric(row, "median_validation_avg_return"), reverse=True)[:5],
        "top_test_pf_diagnostic": sorted(completed, key=lambda row: _safe_metric(row, "median_test_pf"), reverse=True)[:5],
    }


def _config_label(config: dict[str, Any]) -> str:
    return (
        f"{config.get('symbol')} {config.get('timeframe')} h{config.get('horizon_candles')} "
        f"RR{config.get('risk_reward')} ATR{config.get('atr_stop_multiplier')} "
        f"{config.get('cost_mode')} {config.get('strategy_mode')}"
    )


def _result_line(row: dict[str, Any]) -> str:
    aggregate = ((row.get("result") or {}).get("aggregate") or {})
    return (
        f"- `{row.get('classification', row.get('status'))}: {_config_label(row.get('config', {}))}` "
        f"val PF `{aggregate.get('median_validation_pf')}`, "
        f"val avg `{aggregate.get('median_validation_avg_return')}`, "
        f"test PF `{aggregate.get('median_test_pf')}`, "
        f"valid windows `{aggregate.get('valid_windows')}`"
    )


def _result_group(rows: list[dict[str, Any]], empty_message: str) -> list[str]:
    if not rows:
        return [empty_message]
    return [_result_line(row) for row in rows]


def render_cycle_markdown(cycle: dict[str, Any]) -> str:
    summary = cycle.get("summary") or {}
    rows = cycle.get("results") or []
    lines = [
        "# Research Daemon Cycle Summary",
        "",
        f"Generated at: `{cycle.get('finished_at')}`",
        "",
        "## Guardrails",
        "",
        "- Research only. No trading signal.",
        "- No trading, paper trading, Supabase, scheduler, endpoints, or frontend.",
        "- Multi-window validation is mandatory for each configuration.",
        "- Validation selects; test is diagnostic confirmation only.",
        "- Accuracy is not used.",
        "",
        "## Cycle",
        "",
        f"- evaluated: `{summary.get('evaluated')}`",
        f"- completed: `{summary.get('completed')}`",
        f"- failed: `{summary.get('failed')}`",
        f"- classification counts: `{summary.get('classification_counts')}`",
        "",
        "## Stable Candidates",
        "",
    ]
    lines.extend(_result_group(summary.get("stable_candidates", []), "No stable candidates."))
    lines.extend(["", "## Unstable Watchlist", ""])
    lines.extend(_result_group(summary.get("unstable_watchlist", []), "No unstable watchlist items."))
    lines.extend(["", "## Rejects", ""])
    lines.extend(_result_group(summary.get("rejects", []), "No rejects."))
    lines.extend(["", "## Failed", ""])
    failed = [row for row in rows if row.get("status") == "failed"]
    if failed:
        for row in failed:
            lines.append(f"- `{_config_label(row.get('config', {}))}` error `{row.get('error')}`")
    else:
        lines.append("No failed configs.")
    lines.extend(["", "## Top By Median Validation PF", ""])
    lines.extend(_result_line(row) for row in summary.get("top_validation_pf", []))
    lines.extend(["", "## Top By Median Validation Avg Return", ""])
    lines.extend(_result_line(row) for row in summary.get("top_validation_avg", []))
    lines.extend(["", "## Top By Median Test PF (Diagnostic Only, Not Selectable)", ""])
    lines.extend(_result_line(row) for row in summary.get("top_test_pf_diagnostic", []))
    lines.extend(["", "## All Results", ""])
    lines.extend(_result_line(row) for row in rows if row.get("status") != "failed")
    return "\n".join(lines).rstrip() + "\n"


def _json_sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_sanitize(item) for item in value]
    if isinstance(value, tuple):
        return [_json_sanitize(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
    return value


def build_status_payload(
    cycle_started_at: str,
    current_config: dict[str, Any] | None,
    completed_in_cycle: int,
    selected_configs: int,
    results: list[dict[str, Any]],
    started_monotonic: float,
) -> dict[str, Any]:
    summary = summarize_cycle(results)
    elapsed = max(0.0, time.monotonic() - started_monotonic)
    eta_seconds = None
    if completed_in_cycle > 0 and selected_configs > completed_in_cycle:
        avg_seconds = elapsed / completed_in_cycle
        eta_seconds = round(avg_seconds * (selected_configs - completed_in_cycle), 2)
    errors = [row for row in results if row.get("status") == "failed"]
    return {
        "cycle_started_at": cycle_started_at,
        "current_config": current_config,
        "completed_in_cycle": int(completed_in_cycle),
        "selected_configs": int(selected_configs),
        "classification_counts": summary["classification_counts"],
        "last_result": results[-1] if results else None,
        "stable_candidates_so_far": len(summary["stable_candidates"]),
        "unstable_watchlist_so_far": len(summary["unstable_watchlist"]),
        "rejects_so_far": len(summary["rejects"]),
        "errors_so_far": len(errors),
        "elapsed_seconds": round(elapsed, 2),
        "eta_seconds": eta_seconds,
        "updated_at": utc_now(),
    }


def write_status_file(status_path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(status_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_sanitize(payload), indent=2, sort_keys=True), encoding="utf-8")
    return path


def _emit_progress(progress_callback: Callable[[dict[str, Any]], None] | None, payload: dict[str, Any]) -> None:
    if progress_callback is None:
        return
    try:
        progress_callback(payload)
    except Exception:  # noqa: BLE001 - progress display must never break research.
        return


def save_cycle_report(cycle: dict[str, Any], cycles_dir: str | Path = DEFAULT_CYCLES_DIR) -> dict[str, Path]:
    target = Path(cycles_dir)
    target.mkdir(parents=True, exist_ok=True)
    stamp = utc_stamp()
    json_path = target / f"research_cycle_{stamp}.json"
    markdown_path = target / f"research_cycle_{stamp}.md"
    cycle["json_path"] = str(json_path)
    cycle["markdown_path"] = str(markdown_path)
    json_path.write_text(json.dumps(_json_sanitize(cycle), indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(render_cycle_markdown(cycle), encoding="utf-8")
    return {"json_path": json_path, "markdown_path": markdown_path}


def format_daemon_cycle_for_telegram(cycle: dict[str, Any]) -> str:
    summary = cycle.get("summary") or {}
    counts = summary.get("classification_counts") or {}
    count_text = ", ".join(f"{key}={value}" for key, value in sorted(counts.items())) or "none"
    lines = [
        "Research Daemon cycle finished",
        "Research only. No trading signal.",
        "",
        f"Evaluated: {summary.get('evaluated')}",
        f"Stable candidates: {len(summary.get('stable_candidates') or [])}",
        f"Unstable watchlist: {len(summary.get('unstable_watchlist') or [])}",
        f"Counts: {count_text}",
        "",
        "Best median validation PF:",
    ]
    for index, row in enumerate((summary.get("top_validation_pf") or [])[:3], start=1):
        aggregate = ((row.get("result") or {}).get("aggregate") or {})
        lines.append(
            f"{index}. {_config_label(row.get('config', {}))} | PF {aggregate.get('median_validation_pf')} "
            f"| avg {aggregate.get('median_validation_avg_return')}"
        )
    lines.extend([
        "",
        "Best test PF is diagnostic only, not selectable.",
        "",
        f"Markdown: {cycle.get('markdown_path')}",
        f"JSON: {cycle.get('json_path')}",
    ])
    return "\n".join(lines)[:3900]


async def run_research_cycle(
    max_configs_per_cycle: int = 5,
    resume: bool = True,
    retry_failed: bool = False,
    retry_running: bool = False,
    notify_telegram: bool = False,
    grid: list[dict[str, Any]] | None = None,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    cycles_dir: str | Path = DEFAULT_CYCLES_DIR,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    status_path: str | Path = DEFAULT_STATUS_PATH,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run one controlled daemon cycle."""
    registry = ResearchRegistry(registry_path)
    active_grid = grid if grid is not None else build_daemon_grid()
    runnable = (
        registry.filter_runnable(active_grid, retry_failed=retry_failed, retry_running=retry_running)
        if resume
        else active_grid
    )
    selected = runnable[: max(0, int(max_configs_per_cycle))]
    results: list[dict[str, Any]] = []
    interrupted = False
    cycle_started_at = utc_now()
    started_monotonic = time.monotonic()

    Path(results_dir).mkdir(parents=True, exist_ok=True)
    initial_status = build_status_payload(
        cycle_started_at=cycle_started_at,
        current_config=selected[0] if selected else None,
        completed_in_cycle=0,
        selected_configs=len(selected),
        results=results,
        started_monotonic=started_monotonic,
    )
    write_status_file(status_path, initial_status)
    _emit_progress(progress_callback, initial_status)

    for index, config in enumerate(selected, start=1):
        running = registry.mark_running(config)
        started_at = running.get("started_at")
        try:
            setup = _setup_from_daemon_config(config)
            result = await run_multi_window_validation_for_setups(
                setups=[setup],
                symbol=str(config["symbol"]),
                timeframe=str(config["timeframe"]),
                max_candles=int(config["max_candles"]),
                window_size_candles=int(config["window_size_candles"]),
                step_size_candles=int(config["step_size_candles"]),
                refresh_cache=False,
                output_dir=results_dir,
            )
            setup_result = result["setups"][0]
            classification = str(setup_result.get("classification"))
            status = "insufficient_data" if classification == "needs_more_data" else "completed"
            registry.mark_finished(
                config,
                status=status,
                classification=classification,
                json_path=result.get("json_path"),
                markdown_path=result.get("markdown_path"),
                started_at=started_at,
            )
            results.append({
                "config_id": config["config_id"],
                "status": status,
                "classification": classification,
                "config": config,
                "result": setup_result,
                "json_path": result.get("json_path"),
                "markdown_path": result.get("markdown_path"),
                "started_at": started_at,
                "finished_at": utc_now(),
            })
        except KeyboardInterrupt:
            registry.mark_finished(
                config,
                status="failed",
                classification=None,
                error="KeyboardInterrupt",
                started_at=started_at,
            )
            results.append({
                "config_id": config["config_id"],
                "status": "failed",
                "classification": "failed",
                "config": config,
                "result": None,
                "error": "KeyboardInterrupt",
                "started_at": started_at,
                "finished_at": utc_now(),
            })
            interrupted = True
            status_payload = build_status_payload(
                cycle_started_at=cycle_started_at,
                current_config=config,
                completed_in_cycle=len(results),
                selected_configs=len(selected),
                results=results,
                started_monotonic=started_monotonic,
            )
            write_status_file(status_path, status_payload)
            _emit_progress(progress_callback, status_payload)
            break
        except Exception as exc:  # noqa: BLE001 - one failed config must not stop the cycle.
            error = f"{type(exc).__name__}: {exc}"
            registry.mark_finished(
                config,
                status="failed",
                classification=None,
                error=error,
                started_at=started_at,
            )
            results.append({
                "config_id": config["config_id"],
                "status": "failed",
                "classification": "failed",
                "config": config,
                "result": None,
                "error": error,
                "started_at": started_at,
                "finished_at": utc_now(),
            })
        status_payload = build_status_payload(
            cycle_started_at=cycle_started_at,
            current_config=config,
            completed_in_cycle=len(results),
            selected_configs=len(selected),
            results=results,
            started_monotonic=started_monotonic,
        )
        status_payload["current_index"] = index
        write_status_file(status_path, status_payload)
        _emit_progress(progress_callback, status_payload)

    cycle = {
        "created_at": cycle_started_at,
        "finished_at": utc_now(),
        "interrupted": interrupted,
        "requested_max_configs": int(max_configs_per_cycle),
        "grid_size": len(active_grid),
        "runnable_before_cycle": len(runnable),
        "selected_configs": len(selected),
        "results": results,
        "summary": summarize_cycle(results),
        "status_path": str(status_path),
        "guardrails": {
            "research_only": True,
            "no_trading": True,
            "no_paper_trading": True,
            "no_supabase": True,
            "test_not_used_for_selection": True,
            "accuracy_not_used": True,
        },
    }
    paths = save_cycle_report(cycle, cycles_dir=cycles_dir)
    cycle["json_path"] = str(paths["json_path"])
    cycle["markdown_path"] = str(paths["markdown_path"])

    telegram_sent = False
    if notify_telegram:
        telegram_sent = send_telegram_message(format_daemon_cycle_for_telegram(cycle))
    cycle["telegram_sent"] = telegram_sent
    return cycle
