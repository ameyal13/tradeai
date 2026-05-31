"""Multi-window validation for Research Autopilot watchlist setups.

This module validates fixed research watchlist setups across rolling historical
windows. It does not generate operational signals and does not write to
Supabase or the prediction journal.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research.experiment_grid import ExperimentConfig
from research.experiment_runner import run_experiment_on_candles
from scripts.run_historical_experiments import load_experiment_candles


DEFAULT_OUTPUT_DIR = Path("reports") / "research_autopilot" / "multi_window"
FINAL_CLASSIFICATIONS = {
    "needs_more_data",
    "multi_window_reject",
    "unstable_watchlist",
    "stable_research_candidate",
}


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def build_watchlist_setups(
    symbol: str = "SOL",
    timeframe: str = "1h",
    horizon_candles: int = 16,
    risk_reward: float = 2.0,
    atr_stop_multipliers: list[float] | None = None,
    cost_mode: str = "low_costs",
    strategy_mode: str = "xgboost",
    max_candles: int = 1500,
    min_train_rows: int = 120,
) -> list[dict[str, Any]]:
    """Return the fixed watchlist setups approved for this phase."""
    configs: list[dict[str, Any]] = []
    for atr_stop in atr_stop_multipliers or [1.25, 1.5]:
        configs.append(ExperimentConfig(
            symbol=symbol.upper(),
            timeframe=timeframe,
            horizon_candles=int(horizon_candles),
            risk_reward=float(risk_reward),
            atr_stop_multiplier=float(atr_stop),
            cost_mode=cost_mode,
            strategy_mode=strategy_mode,
            max_candles=int(max_candles),
            min_train_rows=int(min_train_rows),
        ).to_dict())
    return configs


def _time_at(candles: pd.DataFrame, index: int) -> str | None:
    if "timestamp" not in candles.columns or len(candles) == 0:
        return None
    value = candles.iloc[index]["timestamp"]
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def generate_rolling_windows(
    candles: pd.DataFrame,
    window_size_candles: int,
    step_size_candles: int,
    horizon_candles: int,
) -> list[dict[str, Any]]:
    """Generate rolling candle windows using end-exclusive indexes."""
    if window_size_candles <= 0:
        raise ValueError("window_size_candles must be positive")
    if step_size_candles <= 0:
        raise ValueError("step_size_candles must be positive")

    total = int(len(candles))
    if total == 0:
        return []

    windows: list[dict[str, Any]] = []
    if total <= window_size_candles:
        starts = [0]
    else:
        starts = list(range(0, total - window_size_candles + 1, step_size_candles))
        last_start = total - window_size_candles
        if starts[-1] != last_start:
            starts.append(last_start)

    for window_index, start in enumerate(starts):
        end = min(start + window_size_candles, total)
        windows.append({
            "window_index": window_index,
            "start_index": int(start),
            "end_index": int(end),
            "size_candles": int(end - start),
            "purge_candles": int(horizon_candles),
            "start_time": _time_at(candles, start),
            "end_time": _time_at(candles, end - 1),
        })
    return windows


def _safe_get(payload: dict[str, Any], path: list[str], default: Any = None) -> Any:
    value: Any = payload
    for key in path:
        if not isinstance(value, dict):
            return default
        value = value.get(key)
    return default if value is None else value


def _window_status(result: dict[str, Any]) -> str:
    reasons = result.get("reasons") or []
    split = result.get("split") or {}
    if "insufficient_data_for_train_validation_test_split" in reasons:
        return "insufficient_data"
    if not split.get("train_rows") or not split.get("validation_rows") or not split.get("test_rows"):
        return "insufficient_data"
    return "valid"


def _window_summary(window: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    validation = result.get("validation_metrics") or {}
    test = result.get("test_metrics") or {}
    diagnostics = result.get("diagnostics") or {}
    baselines = result.get("baselines") or {}
    exposure = diagnostics.get("validation_directional_exposure") or {}
    split = result.get("split") or {}
    return {
        **window,
        "window_status": _window_status(result),
        "classification": result.get("classification"),
        "reasons": result.get("reasons") or [],
        "validation_avg_return": validation.get("avg_return_pct"),
        "validation_profit_factor": validation.get("profit_factor"),
        "validation_drawdown": validation.get("max_drawdown_pct"),
        "test_avg_return": test.get("avg_return_pct"),
        "test_profit_factor": test.get("profit_factor"),
        "test_drawdown": test.get("max_drawdown_pct"),
        "beats_random_validation": diagnostics.get("beats_random_validation", False),
        "beats_deterministic_validation": diagnostics.get("beats_deterministic_validation", False),
        "validation_positive": diagnostics.get("validation_positive", False),
        "test_confirms": diagnostics.get("test_confirms", False),
        "directional_bias": exposure.get("directional_bias", "unknown"),
        "n_trades": validation.get("n_trades", 0),
        "random_validation_avg": _safe_get(baselines, ["validation", "random_same_count", "avg_return_pct"]),
        "random_validation_pf": _safe_get(baselines, ["validation", "random_same_count", "profit_factor"]),
        "deterministic_validation_avg": _safe_get(baselines, ["validation", "deterministic", "avg_return_pct"]),
        "deterministic_validation_pf": _safe_get(baselines, ["validation", "deterministic", "profit_factor"]),
        "split": split,
        "result": result,
    }


def run_setup_across_windows(
    setup: dict[str, Any],
    candles: pd.DataFrame,
    window_size_candles: int = 600,
    step_size_candles: int = 250,
) -> dict[str, Any]:
    """Run one fixed setup across all rolling windows."""
    windows = generate_rolling_windows(
        candles,
        window_size_candles=window_size_candles,
        step_size_candles=step_size_candles,
        horizon_candles=int(setup["horizon_candles"]),
    )
    window_results: list[dict[str, Any]] = []
    base_id = str(setup["experiment_id"])

    for window in windows:
        start = int(window["start_index"])
        end = int(window["end_index"])
        window_candles = candles.iloc[start:end].reset_index(drop=True).copy()
        window_config = dict(setup)
        window_config["experiment_id"] = f"{base_id}_w{window['window_index']:03d}"
        window_config["max_candles"] = int(len(window_candles))
        try:
            result = run_experiment_on_candles(
                window_config,
                window_candles,
                {
                    "data_source": "multi_window_local_cache",
                    "data_cache_path": None,
                    "data_warning": None,
                },
            )
            summary = _window_summary(window, result)
        except Exception as exc:  # pragma: no cover - defensive boundary
            summary = {
                **window,
                "window_status": "error",
                "classification": "error",
                "reasons": [f"{type(exc).__name__}: {exc}"],
                "result": None,
            }
        window_results.append(summary)

    aggregate = aggregate_multi_window_results(window_results)
    return {
        "setup": setup,
        "aggregate": aggregate,
        "classification": classify_multi_window_setup(aggregate),
        "windows": window_results,
    }


def _numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            values.append(numeric)
    return values


def _median(rows: list[dict[str, Any]], key: str) -> float | None:
    values = _numeric_values(rows, key)
    return round(float(np.median(values)), 6) if values else None


def _rate(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return round(sum(1 for row in rows if bool(row.get(key))) / len(rows), 6)


def aggregate_multi_window_results(window_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-window results for one setup."""
    valid = [row for row in window_results if row.get("window_status") == "valid"]
    insufficient = [row for row in window_results if row.get("window_status") == "insufficient_data"]
    drawdowns = _numeric_values(valid, "validation_drawdown")
    bias_counts = Counter(str(row.get("directional_bias", "unknown")) for row in valid)
    dominant_bias, dominant_count = bias_counts.most_common(1)[0] if bias_counts else ("none", 0)
    directional_stability = {
        "dominant_bias": dominant_bias,
        "dominant_bias_rate": round(dominant_count / len(valid), 6) if valid else 0.0,
        "counts": dict(bias_counts),
    }
    test_contradiction_rate = 0.0
    if valid:
        test_contradictions = [
            row for row in valid
            if (row.get("test_avg_return") is not None and row.get("test_profit_factor") is not None)
            and float(row.get("test_avg_return") or 0) <= 0
            and float(row.get("test_profit_factor") or 0) < 1.0
        ]
        test_contradiction_rate = round(len(test_contradictions) / len(valid), 6)

    return {
        "total_windows": int(len(window_results)),
        "valid_windows": int(len(valid)),
        "insufficient_windows": int(len(insufficient)),
        "error_windows": int(sum(1 for row in window_results if row.get("window_status") == "error")),
        "validation_positive_rate": _rate(valid, "validation_positive"),
        "beats_random_rate": _rate(valid, "beats_random_validation"),
        "beats_deterministic_rate": _rate(valid, "beats_deterministic_validation"),
        "test_confirm_rate": _rate(valid, "test_confirms"),
        "test_contradiction_rate": test_contradiction_rate,
        "median_validation_pf": _median(valid, "validation_profit_factor"),
        "median_validation_avg_return": _median(valid, "validation_avg_return"),
        "worst_validation_drawdown": round(max(drawdowns), 6) if drawdowns else None,
        "median_test_pf": _median(valid, "test_profit_factor"),
        "median_test_avg_return": _median(valid, "test_avg_return"),
        "directional_stability": directional_stability,
    }


def classify_multi_window_setup(aggregate: dict[str, Any]) -> str:
    """Classify a fixed setup using validation-first multi-window criteria."""
    valid_windows = int(aggregate.get("valid_windows") or 0)
    if valid_windows < 3:
        return "needs_more_data"

    validation_rate = float(aggregate.get("validation_positive_rate") or 0)
    beats_random_rate = float(aggregate.get("beats_random_rate") or 0)
    beats_deterministic_rate = float(aggregate.get("beats_deterministic_rate") or 0)
    median_pf = float(aggregate.get("median_validation_pf") or 0)
    median_avg = float(aggregate.get("median_validation_avg_return") or 0)
    test_contradiction_rate = float(aggregate.get("test_contradiction_rate") or 0)

    if (
        validation_rate >= 0.60
        and beats_random_rate >= 0.60
        and beats_deterministic_rate >= 0.50
        and median_pf > 1.05
        and median_avg > 0
        and test_contradiction_rate <= 0.40
    ):
        return "stable_research_candidate"

    if validation_rate >= 0.30 or beats_random_rate >= 0.30 or median_avg > 0 or median_pf > 1.0:
        return "unstable_watchlist"
    return "multi_window_reject"


def _setup_title(setup: dict[str, Any]) -> str:
    return (
        f"{setup.get('symbol')} {setup.get('timeframe')} "
        f"h{setup.get('horizon_candles')} RR{setup.get('risk_reward')} "
        f"ATR{setup.get('atr_stop_multiplier')} {setup.get('cost_mode')} "
        f"{setup.get('strategy_mode')}"
    )


def render_multi_window_markdown(summary: dict[str, Any]) -> str:
    """Render a readable Markdown report for multi-window validation."""
    lines = [
        "# Multi-Window Validation Summary",
        "",
        f"Generated at: `{summary.get('created_at')}`",
        "",
        "## Guardrails",
        "",
        "- Research only; no trading, paper trading, Supabase, scheduler, or frontend.",
        "- Validation selects; test only confirms or contradicts.",
        "- Fixed watchlist setups only; no threshold, feature, model, or cost changes.",
        "",
        "## Classification Counts",
        "",
        f"`{summary.get('classification_counts', {})}`",
        "",
        "## Setup Results",
        "",
    ]
    for setup_result in summary.get("setups", []):
        setup = setup_result.get("setup", {})
        aggregate = setup_result.get("aggregate", {})
        classification = setup_result.get("classification")
        lines.extend([
            f"### {_setup_title(setup)}",
            "",
            f"Classification: `{classification}`",
            "",
            (
                f"- windows valid/total: `{aggregate.get('valid_windows')}` / "
                f"`{aggregate.get('total_windows')}`"
            ),
            f"- validation positive rate: `{aggregate.get('validation_positive_rate')}`",
            f"- beats random rate: `{aggregate.get('beats_random_rate')}`",
            f"- beats deterministic rate: `{aggregate.get('beats_deterministic_rate')}`",
            f"- test confirm rate: `{aggregate.get('test_confirm_rate')}`",
            f"- median validation PF: `{aggregate.get('median_validation_pf')}`",
            f"- median validation avg return: `{aggregate.get('median_validation_avg_return')}`",
            f"- worst validation drawdown: `{aggregate.get('worst_validation_drawdown')}`",
            f"- directional stability: `{aggregate.get('directional_stability')}`",
            "",
            "| Window | Status | Val Avg | Val PF | Test Avg | Test PF | Beats Random | Beats Det | Bias | Trades |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | ---: |",
        ])
        for row in setup_result.get("windows", []):
            lines.append(
                f"| {row.get('window_index')} | {row.get('window_status')} | "
                f"{row.get('validation_avg_return')} | {row.get('validation_profit_factor')} | "
                f"{row.get('test_avg_return')} | {row.get('test_profit_factor')} | "
                f"{row.get('beats_random_validation')} | {row.get('beats_deterministic_validation')} | "
                f"{row.get('directional_bias')} | {row.get('n_trades')} |"
            )
        lines.append("")
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
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def save_multi_window_report(summary: dict[str, Any], output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Path]:
    """Save JSON and Markdown reports."""
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    stamp = utc_stamp()
    json_path = target_dir / f"multi_window_results_{stamp}.json"
    markdown_path = target_dir / f"multi_window_summary_{stamp}.md"
    summary["json_path"] = str(json_path)
    summary["markdown_path"] = str(markdown_path)
    json_path.write_text(json.dumps(_json_sanitize(summary), indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(render_multi_window_markdown(summary), encoding="utf-8")
    return {"json_path": json_path, "markdown_path": markdown_path}


async def run_multi_window_validation(
    symbol: str = "SOL",
    timeframe: str = "1h",
    max_candles: int = 1500,
    window_size_candles: int = 600,
    step_size_candles: int = 250,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Run the approved SOL watchlist setups across rolling windows."""
    loaded = await load_experiment_candles(
        symbol.upper(),
        timeframe,
        max_candles=int(max_candles),
        use_cache=True,
        refresh_cache=False,
    )
    candles = loaded["candles"]
    setups = build_watchlist_setups(
        symbol=symbol,
        timeframe=timeframe,
        max_candles=int(max_candles),
    )
    setup_results = [
        run_setup_across_windows(
            setup,
            candles,
            window_size_candles=int(window_size_candles),
            step_size_candles=int(step_size_candles),
        )
        for setup in setups
    ]
    counts: dict[str, int] = {}
    for row in setup_results:
        classification = str(row.get("classification"))
        counts[classification] = counts.get(classification, 0) + 1
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "methodology": {
            "window_type": "rolling",
            "window_size_candles": int(window_size_candles),
            "step_size_candles": int(step_size_candles),
            "test_not_used_for_selection": True,
            "purge_candles": 16,
            "accuracy_not_used": True,
            "no_trading": True,
        },
        "data": {
            "symbol": symbol.upper(),
            "timeframe": timeframe,
            "max_candles": int(max_candles),
            "rows": int(len(candles)),
            "data_source": loaded.get("data_source"),
            "data_cache_path": loaded.get("data_cache_path"),
            "data_warning": loaded.get("data_warning"),
        },
        "classification_counts": counts,
        "setups": setup_results,
    }
    paths = save_multi_window_report(summary, output_dir=output_dir)
    summary["json_path"] = str(paths["json_path"])
    summary["markdown_path"] = str(paths["markdown_path"])
    return summary
