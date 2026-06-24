"""Run Feature Expansion Grid v1 locally.

Research only: no trading, no paper trading, no Supabase writes, no endpoints,
and no operational signals. This phase freezes sizing/model/costs and varies
only research feature families, including time_only as a reference baseline.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.feature_expansion_grid import (  # noqa: E402
    FEATURE_EXPANSION_FEATURE_SETS,
    OBJECTIVE_PROFIT_FACTOR,
    TIME_ONLY_BASELINE_PF,
    build_feature_expansion_grid,
)
from research.multi_window_validator import generate_rolling_windows  # noqa: E402
from research.research_daemon import DEFAULT_DAEMON_DIR  # noqa: E402
from research.research_registry import ResearchRegistry, utc_now  # noqa: E402
from scripts.run_crypto_multi_asset_research import parse_symbols, print_running_records, running_records  # noqa: E402
from scripts.run_historical_experiments import load_experiment_candles  # noqa: E402
from scripts.run_research_daemon import print_progress  # noqa: E402
from tools.feature_research import (  # noqa: E402
    BTC_CONTEXT_FEATURE_COLS,
    FUNDING_FEATURE_COLS,
    MTF_FEATURE_COLS,
    TIME_COLS,
    add_research_features,
    build_directional_net_returns,
    evaluate_feature_set,
)
from tools.funding_features import (  # noqa: E402
    compute_funding_features,
    fetch_funding_rate_history,
    fetch_open_interest_history,
)
from tools.historical_data import fetch_binance_klines  # noqa: E402
from tools.ml_engine import FEATURE_COLS, _atr_label_pcts, build_trade_outcome_labels  # noqa: E402
from tools.multitimeframe_features import (  # noqa: E402
    compute_4h_features,
    compute_btc_context_features,
    fetch_4h_candles_aligned,
)
from tools.trade_opportunity_research import DEFAULT_COST_PROFILES  # noqa: E402


FEATURE_EXPANSION_REGISTRY_PATH = DEFAULT_DAEMON_DIR / "feature_expansion_v1_registry.jsonl"
FEATURE_EXPANSION_CYCLES_DIR = DEFAULT_DAEMON_DIR / "feature_expansion_v1_cycles"
FEATURE_EXPANSION_RESULTS_DIR = DEFAULT_DAEMON_DIR / "feature_expansion_v1_results"
FEATURE_EXPANSION_STATUS_PATH = DEFAULT_DAEMON_DIR / "feature_expansion_v1_current_status.json"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


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
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _time_at(candles: pd.DataFrame, index: int) -> str | None:
    if "timestamp" not in candles.columns or len(candles) == 0:
        return None
    value = candles.iloc[index]["timestamp"]
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def _metric(result: dict[str, Any], key: str) -> float | None:
    try:
        value = float(result.get(key))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _median(values: list[float]) -> float | None:
    return round(float(np.median(values)), 6) if values else None


def metrics_summary(rows: list[dict[str, Any]], prefix: str = "") -> dict[str, Any]:
    valid = [row for row in rows if row.get("window_status") == "valid"]

    def values(key: str) -> list[float]:
        out: list[float] = []
        for row in valid:
            value = row.get(key)
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                out.append(number)
        return out

    label = f"{prefix}_" if prefix else ""
    return {
        f"{label}valid_windows": len(valid),
        f"{label}median_pf": _median(values("profit_factor")),
        f"{label}median_avg_return": _median(values("average_return")),
        f"{label}median_trades": _median(values("trades")),
    }


def _feature_cols_for_set(feature_set: str) -> list[str]:
    baseline = list(FEATURE_COLS)
    if feature_set == "baseline":
        return baseline
    if feature_set == "time_only":
        return list(TIME_COLS)
    if feature_set == "baseline_plus_funding":
        return baseline + FUNDING_FEATURE_COLS
    if feature_set == "baseline_plus_4h":
        return baseline + MTF_FEATURE_COLS
    if feature_set == "baseline_plus_all_new":
        return baseline + FUNDING_FEATURE_COLS + MTF_FEATURE_COLS
    if feature_set == "funding_only":
        return list(FUNDING_FEATURE_COLS)
    if feature_set == "4h_only":
        return list(MTF_FEATURE_COLS)
    if feature_set == "baseline_plus_btc_context":
        return baseline + BTC_CONTEXT_FEATURE_COLS
    if feature_set == "baseline_plus_all_new_plus_btc":
        return baseline + FUNDING_FEATURE_COLS + MTF_FEATURE_COLS + BTC_CONTEXT_FEATURE_COLS
    raise ValueError(f"Unknown feature_set: {feature_set}")


async def build_expanded_feature_frame(symbol: str, candles: pd.DataFrame) -> pd.DataFrame:
    """Build base + funding + 4h + BTC context features for one symbol."""
    base = add_research_features(candles, include_market_context=False)
    times = pd.to_datetime(candles["timestamp"], utc=True)
    start_time = times.min()
    end_time = times.max()

    funding_df = await fetch_funding_rate_history(symbol, start_time, end_time, limit=max(1000, int(len(candles) / 4) + 100))
    oi_df = await fetch_open_interest_history(symbol, start_time, end_time, period="1h", limit=len(candles) + 100)
    funding_features = compute_funding_features(candles, funding_df, oi_df)
    funding_features.index = base.index

    candles_4h = await fetch_4h_candles_aligned(symbol, candles)
    mtf_features = compute_4h_features(candles, candles_4h)
    mtf_features.index = base.index

    btc = await fetch_binance_klines("BTC", "1h", start_time=start_time, end_time=end_time, limit=len(candles) + 100)
    btc_features = compute_btc_context_features(candles, btc)
    btc_features.index = base.index

    return pd.concat([base, funding_features, mtf_features, btc_features], axis=1)


def _window_evaluation(
    features: pd.DataFrame,
    feature_cols: list[str],
    config: dict[str, Any],
    dummy_random: bool = False,
) -> dict[str, Any]:
    costs = DEFAULT_COST_PROFILES[str(config["cost_mode"])]
    stop_pcts, take_pcts = _atr_label_pcts(
        features,
        atr_stop_multiplier=float(config["atr_stop_multiplier"]),
        min_rr=float(config["risk_reward"]),
        atr_take_profit_multiplier=None,
        fallback_stop_loss_pct=0.03,
        fallback_take_profit_pct=0.03 * float(config["risk_reward"]),
    )
    labels = build_trade_outcome_labels(
        features,
        horizon_candles=int(config["horizon_candles"]),
        stop_loss_pct=0.03,
        take_profit_pct=0.03 * float(config["risk_reward"]),
        commission_pct=float(costs["commission_pct"]),
        slippage_pct=float(costs["slippage_pct"]),
        spread_pct=float(costs["spread_pct"]),
        stop_loss_pcts=stop_pcts,
        take_profit_pcts=take_pcts,
        label_scheme=str(config.get("trade_label_scheme", "expected_value_classification")),
    )
    returns = build_directional_net_returns(
        features,
        horizon_candles=int(config["horizon_candles"]),
        stop_loss_pct=0.03,
        take_profit_pct=0.03 * float(config["risk_reward"]),
        commission_pct=float(costs["commission_pct"]),
        slippage_pct=float(costs["slippage_pct"]),
        spread_pct=float(costs["spread_pct"]),
        stop_loss_pcts=stop_pcts,
        take_profit_pcts=take_pcts,
    )
    return evaluate_feature_set(
        features,
        labels,
        returns,
        feature_cols,
        n_splits=4,
        min_train_rows=int(config.get("min_train_rows", 120)),
        horizon_candles=int(config["horizon_candles"]),
        buy_threshold=float(config.get("buy_threshold", 0.58)),
        sell_threshold=float(config.get("sell_threshold", 0.58)),
        dummy_random=dummy_random,
        compute_permutation_importance=False,
    )


def aggregate_feature_expansion_windows(windows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in windows if row.get("window_status") == "valid"]
    selected = metrics_summary(valid)
    time_only = metrics_summary([row.get("time_only_result", {}) | {"window_status": row.get("window_status")} for row in valid], "time_only")
    dummy = metrics_summary([row.get("dummy_random_result", {}) | {"window_status": row.get("window_status")} for row in valid], "dummy_random")
    pf_values = [float(row["profit_factor"]) for row in valid if _metric(row, "profit_factor") is not None]
    avg_values = [float(row["average_return"]) for row in valid if _metric(row, "average_return") is not None]
    beats_time = [
        row for row in valid
        if (_metric(row, "profit_factor") is not None and _metric(row.get("time_only_result", {}), "profit_factor") is not None)
        and float(row["profit_factor"]) > float(row["time_only_result"]["profit_factor"])
    ]
    beats_dummy = [
        row for row in valid
        if (_metric(row, "average_return") is not None and _metric(row.get("dummy_random_result", {}), "average_return") is not None)
        and float(row["average_return"]) > float(row["dummy_random_result"]["average_return"])
    ]
    aggregate = {
        "total_windows": len(windows),
        "valid_windows": len(valid),
        "insufficient_windows": sum(1 for row in windows if row.get("window_status") == "insufficient_data"),
        "error_windows": sum(1 for row in windows if row.get("window_status") == "error"),
        "median_validation_pf": selected.get("median_pf"),
        "median_validation_avg_return": selected.get("median_avg_return"),
        "median_validation_trades": selected.get("median_trades"),
        "time_only_median_pf": time_only.get("time_only_median_pf"),
        "time_only_median_avg_return": time_only.get("time_only_median_avg_return"),
        "dummy_random_median_pf": dummy.get("dummy_random_median_pf"),
        "dummy_random_median_avg_return": dummy.get("dummy_random_median_avg_return"),
        "beats_time_only_rate": round(len(beats_time) / len(valid), 6) if valid else 0.0,
        "beats_dummy_random_rate": round(len(beats_dummy) / len(valid), 6) if valid else 0.0,
        "objective_pf_rate": round(sum(1 for value in pf_values if value > OBJECTIVE_PROFIT_FACTOR) / len(pf_values), 6) if pf_values else 0.0,
        "validation_positive_rate": round(sum(1 for idx, value in enumerate(pf_values) if value > 1.0 and avg_values[idx] > 0) / len(pf_values), 6) if pf_values and len(pf_values) == len(avg_values) else 0.0,
    }
    return aggregate


def classify_feature_expansion_setup(aggregate: dict[str, Any]) -> str:
    valid_windows = int(aggregate.get("valid_windows") or 0)
    if valid_windows < 3:
        return "needs_more_data"
    median_pf = float(aggregate.get("median_validation_pf") or 0)
    median_avg = float(aggregate.get("median_validation_avg_return") or 0)
    beats_time = float(aggregate.get("beats_time_only_rate") or 0)
    beats_dummy = float(aggregate.get("beats_dummy_random_rate") or 0)
    if median_pf > 1.0 and median_avg > 0 and beats_time >= 0.60 and beats_dummy >= 0.60:
        return "feature_expansion_candidate"
    if median_pf > TIME_ONLY_BASELINE_PF and median_avg > 0 and beats_dummy >= 0.50:
        return "feature_expansion_watchlist"
    return "feature_expansion_reject"


async def run_feature_expansion_config(
    config: dict[str, Any],
    candles: pd.DataFrame,
    expanded_features: pd.DataFrame,
) -> dict[str, Any]:
    windows = generate_rolling_windows(
        candles,
        window_size_candles=int(config["window_size_candles"]),
        step_size_candles=int(config["step_size_candles"]),
        horizon_candles=int(config["horizon_candles"]),
    )
    feature_cols = _feature_cols_for_set(str(config["feature_set"]))
    window_rows: list[dict[str, Any]] = []
    for window in windows:
        start = int(window["start_index"])
        end = int(window["end_index"])
        frame = expanded_features.iloc[start:end].copy()
        try:
            result = _window_evaluation(frame, feature_cols, config)
            time_result = _window_evaluation(frame, list(TIME_COLS), config)
            dummy_result = _window_evaluation(frame, list(FEATURE_COLS), config, dummy_random=True)
            status = "valid" if result.get("status") == "ok" else "insufficient_data"
            window_rows.append({
                **window,
                "start_time": _time_at(candles, start),
                "end_time": _time_at(candles, end - 1),
                "window_status": status,
                "feature_set": config["feature_set"],
                "feature_cols": feature_cols,
                "profit_factor": result.get("profit_factor"),
                "average_return": result.get("average_return"),
                "trades": result.get("trades"),
                "buy_trades": result.get("buy_trades"),
                "sell_trades": result.get("sell_trades"),
                "hold_count": result.get("hold_count"),
                "status": result.get("status"),
                "time_only_result": {
                    "profit_factor": time_result.get("profit_factor"),
                    "average_return": time_result.get("average_return"),
                    "trades": time_result.get("trades"),
                    "status": time_result.get("status"),
                },
                "dummy_random_result": {
                    "profit_factor": dummy_result.get("profit_factor"),
                    "average_return": dummy_result.get("average_return"),
                    "trades": dummy_result.get("trades"),
                    "status": dummy_result.get("status"),
                },
            })
        except Exception as exc:  # noqa: BLE001
            window_rows.append({
                **window,
                "window_status": "error",
                "error": f"{type(exc).__name__}: {str(exc)[:300]}",
            })
    aggregate = aggregate_feature_expansion_windows(window_rows)
    classification = classify_feature_expansion_setup(aggregate)
    return {
        "setup": config,
        "classification": classification,
        "aggregate": aggregate,
        "windows": window_rows,
    }


def render_feature_expansion_markdown(summary: dict[str, Any]) -> str:
    setup = summary["setups"][0]["setup"]
    result = summary["setups"][0]
    aggregate = result["aggregate"]
    lines = [
        "# Feature Expansion v1 Result",
        "",
        f"Generated at: `{summary['created_at']}`",
        "",
        "## Guardrails",
        "",
        "- Research only. No trading signal.",
        "- No live/shadow feature changes.",
        "- No Supabase writes.",
        "- Validation is purged and test metrics are not used here.",
        "",
        "## Setup",
        "",
        f"- symbol: `{setup.get('symbol')}`",
        f"- timeframe: `{setup.get('timeframe')}`",
        f"- feature_set: `{setup.get('feature_set')}`",
        f"- horizon_candles: `{setup.get('horizon_candles')}`",
        f"- risk_reward: `{setup.get('risk_reward')}`",
        f"- atr_stop_multiplier: `{setup.get('atr_stop_multiplier')}`",
        f"- cost_mode: `{setup.get('cost_mode')}`",
        "",
        "## Aggregate",
        "",
        f"- classification: `{result.get('classification')}`",
        f"- median validation PF: `{aggregate.get('median_validation_pf')}`",
        f"- median validation avg return: `{aggregate.get('median_validation_avg_return')}`",
        f"- beats time_only rate: `{aggregate.get('beats_time_only_rate')}`",
        f"- beats dummy_random rate: `{aggregate.get('beats_dummy_random_rate')}`",
        f"- time_only reference PF from audit: `{TIME_ONLY_BASELINE_PF}`",
        f"- objective PF: `{OBJECTIVE_PROFIT_FACTOR}`",
        "",
        "## Windows",
        "",
        "| Window | Status | PF | Avg | Trades | Time PF | Dummy PF |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["windows"]:
        lines.append(
            f"| {row.get('window_index')} | {row.get('window_status')} | {row.get('profit_factor')} | "
            f"{row.get('average_return')} | {row.get('trades')} | "
            f"{(row.get('time_only_result') or {}).get('profit_factor')} | "
            f"{(row.get('dummy_random_result') or {}).get('profit_factor')} |"
        )
    return "\n".join(lines).rstrip() + "\n"


def save_config_result(result: dict[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = result["setup"]
    stamp = utc_stamp()
    name = f"feature_expansion_{config['symbol']}_{config['feature_set']}_{config['config_id']}_{stamp}"
    json_path = output_dir / f"{name}.json"
    markdown_path = output_dir / f"{name}.md"
    summary = {
        "created_at": utc_now(),
        "methodology": {
            "phase": "feature_expansion_v1",
            "rolling_windows": True,
            "purge_candles": int(config["horizon_candles"]),
            "research_only": True,
            "no_trading_signal": True,
            "time_only_reference_pf": TIME_ONLY_BASELINE_PF,
            "objective_profit_factor": OBJECTIVE_PROFIT_FACTOR,
        },
        "setups": [result],
    }
    summary["json_path"] = str(json_path)
    summary["markdown_path"] = str(markdown_path)
    json_path.write_text(json.dumps(_json_sanitize(summary), indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(render_feature_expansion_markdown(summary), encoding="utf-8")
    return {"json_path": str(json_path), "markdown_path": str(markdown_path)}


def write_status(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_sanitize(payload), indent=2, sort_keys=True), encoding="utf-8")


async def run_feature_expansion_cycle(
    *,
    grid: list[dict[str, Any]],
    max_configs_per_cycle: int,
    resume: bool,
    retry_failed: bool,
    retry_running: bool,
    registry_path: Path,
    cycles_dir: Path,
    results_dir: Path,
    status_path: Path,
    progress_callback: Any = None,
) -> dict[str, Any]:
    registry = ResearchRegistry(registry_path)
    runnable = registry.filter_runnable(grid, retry_failed=retry_failed, retry_running=retry_running) if resume else grid
    selected = runnable[: max(0, int(max_configs_per_cycle))]
    feature_cache: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    results: list[dict[str, Any]] = []
    start = time.monotonic()
    cycle_started_at = utc_now()

    write_status(status_path, {
        "cycle_started_at": cycle_started_at,
        "selected_configs": len(selected),
        "completed_in_cycle": 0,
        "classification_counts": {},
        "updated_at": utc_now(),
    })

    for index, config in enumerate(selected, start=1):
        running = registry.mark_running(config)
        started_at = running.get("started_at")
        try:
            symbol = str(config["symbol"]).upper()
            if symbol not in feature_cache:
                loaded = await load_experiment_candles(
                    symbol,
                    str(config["timeframe"]),
                    max_candles=int(config["max_candles"]),
                    use_cache=True,
                    refresh_cache=False,
                )
                candles = loaded["candles"]
                feature_cache[symbol] = (candles, await build_expanded_feature_frame(symbol, candles))
            candles, features = feature_cache[symbol]
            setup_result = await run_feature_expansion_config(config, candles, features)
            paths = save_config_result(setup_result, results_dir)
            classification = str(setup_result["classification"])
            status = "insufficient_data" if classification == "needs_more_data" else "completed"
            registry.mark_finished(
                config,
                status=status,
                classification=classification,
                json_path=paths["json_path"],
                markdown_path=paths["markdown_path"],
                started_at=started_at,
            )
            row = {
                "config_id": config["config_id"],
                "status": status,
                "classification": classification,
                "config": config,
                "result": setup_result,
                **paths,
            }
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {str(exc)[:300]}"
            registry.mark_finished(config, status="failed", classification=None, error=error, started_at=started_at)
            row = {
                "config_id": config["config_id"],
                "status": "failed",
                "classification": "failed",
                "config": config,
                "error": error,
            }
        results.append(row)
        counts: dict[str, int] = {}
        for item in results:
            label = str(item.get("classification") or item.get("status"))
            counts[label] = counts.get(label, 0) + 1
        elapsed = time.monotonic() - start
        status_payload = {
            "cycle_started_at": cycle_started_at,
            "current_index": index,
            "current_config": config,
            "completed_in_cycle": len(results),
            "selected_configs": len(selected),
            "classification_counts": counts,
            "last_result": row,
            "elapsed_seconds": round(elapsed, 2),
            "updated_at": utc_now(),
        }
        write_status(status_path, status_payload)
        if progress_callback:
            progress_callback(status_payload)

    cycle = {
        "created_at": cycle_started_at,
        "finished_at": utc_now(),
        "grid_size": len(grid),
        "runnable_before_cycle": len(runnable),
        "selected_configs": len(selected),
        "results": results,
        "summary": {
            "evaluated": len(results),
            "classification_counts": {
                label: sum(1 for row in results if str(row.get("classification")) == label)
                for label in sorted({str(row.get("classification")) for row in results})
            },
        },
        "guardrails": {
            "research_only": True,
            "no_trading_signal": True,
            "no_supabase": True,
            "accuracy_not_used": True,
        },
    }
    cycles_dir.mkdir(parents=True, exist_ok=True)
    stamp = utc_stamp()
    json_path = cycles_dir / f"feature_expansion_cycle_{stamp}.json"
    markdown_path = cycles_dir / f"feature_expansion_cycle_{stamp}.md"
    cycle["json_path"] = str(json_path)
    cycle["markdown_path"] = str(markdown_path)
    json_path.write_text(json.dumps(_json_sanitize(cycle), indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(render_cycle_markdown(cycle), encoding="utf-8")
    return cycle


def render_cycle_markdown(cycle: dict[str, Any]) -> str:
    lines = [
        "# Feature Expansion v1 Cycle",
        "",
        "Research only. No trading signal.",
        "",
        f"- grid size: `{cycle.get('grid_size')}`",
        f"- selected configs: `{cycle.get('selected_configs')}`",
        f"- classification counts: `{(cycle.get('summary') or {}).get('classification_counts')}`",
        "",
        "## Results",
        "",
    ]
    for row in cycle.get("results", []):
        config = row.get("config") or {}
        aggregate = ((row.get("result") or {}).get("aggregate") or {})
        lines.append(
            f"- `{row.get('classification')}: {config.get('symbol')} {config.get('feature_set')}` "
            f"PF `{aggregate.get('median_validation_pf')}`, avg `{aggregate.get('median_validation_avg_return')}`, "
            f"beats_time `{aggregate.get('beats_time_only_rate')}`"
        )
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Feature Expansion Grid v1.")
    parser.add_argument("--once", action="store_true", default=True)
    parser.add_argument("--max-configs-per-cycle", type=int, default=5)
    parser.add_argument("--symbols", nargs="*", default=None, help="Subset of ADA, ETH.")
    parser.add_argument("--feature-sets", nargs="*", default=None, help=f"Subset of {FEATURE_EXPANSION_FEATURE_SETS}.")
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", action="store_true", dest="resume", default=True)
    resume_group.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--retry-failed", action="store_true", default=False)
    parser.add_argument("--retry-running", action="store_true", default=False)
    parser.add_argument("--list-running", action="store_true", default=False)
    parser.add_argument("--progress", action="store_true", default=True)
    parser.add_argument("--quiet", action="store_true", default=False)
    return parser


async def main() -> None:
    args = build_parser().parse_args()
    symbols = parse_symbols(args.symbols)
    if args.list_running:
        print_running_records(running_records(FEATURE_EXPANSION_REGISTRY_PATH, symbols=symbols))
        return
    grid = build_feature_expansion_grid(symbols=symbols, feature_sets=args.feature_sets)
    result = await run_feature_expansion_cycle(
        grid=grid,
        max_configs_per_cycle=args.max_configs_per_cycle,
        resume=args.resume,
        retry_failed=args.retry_failed,
        retry_running=args.retry_running,
        registry_path=FEATURE_EXPANSION_REGISTRY_PATH,
        cycles_dir=FEATURE_EXPANSION_CYCLES_DIR,
        results_dir=FEATURE_EXPANSION_RESULTS_DIR,
        status_path=FEATURE_EXPANSION_STATUS_PATH,
        progress_callback=print_progress if args.progress and not args.quiet else None,
    )
    print("Feature Expansion Grid v1 research cycle finished")
    print("Research only. No trading signal.")
    print(f"grid_size: {result['grid_size']}")
    print(f"runnable_before_cycle: {result['runnable_before_cycle']}")
    print(f"selected_configs: {result['selected_configs']}")
    print(f"evaluated: {result['summary']['evaluated']}")
    print(f"classification_counts: {result['summary']['classification_counts']}")
    print(f"registry: {FEATURE_EXPANSION_REGISTRY_PATH}")
    print(f"status: {FEATURE_EXPANSION_STATUS_PATH}")
    print(f"json: {result['json_path']}")
    print(f"markdown: {result['markdown_path']}")


if __name__ == "__main__":
    asyncio.run(main())
