"""Walk-forward optimizer for simple strategy parameters."""
from __future__ import annotations

from itertools import product
from typing import Any

import pandas as pd

from tools.historical_data import normalize_ohlcv
from tools.historical_replay import run_historical_replay


BASELINE_PARAMS = {
    "rsi_buy_threshold": 35,
    "rsi_sell_threshold": 65,
    "atr_stop_multiplier": 1.5,
    "atr_take_profit_multiplier": None,
    "min_volume_ratio": 1.2,
    "min_risk_reward": 1.5,
    "probability_buy_threshold": 0.58,
    "probability_sell_threshold": 0.42,
}


DEFAULT_PARAM_GRID = {
    "rsi_buy_threshold": [30, 35],
    "rsi_sell_threshold": [65, 70],
    "min_risk_reward": [1.5, 2.0],
    "probability_buy_threshold": [0.56, 0.60],
    "probability_sell_threshold": [0.44, 0.40],
}


def expand_param_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(grid.keys())
    return [dict(zip(keys, values)) for values in product(*[grid[key] for key in keys])]


def first_metric(result: dict[str, Any]) -> dict[str, Any]:
    return result.get("metrics", [{}])[0] if result.get("metrics") else {}


def objective_score(metrics: dict[str, Any]) -> float:
    evaluated = float(metrics.get("evaluated_predictions", 0) or 0)
    if evaluated <= 0:
        return -1e9
    profit_factor = float(metrics.get("profit_factor", 0) or 0)
    average_return = float(metrics.get("average_return_pct", metrics.get("average_return", 0)) or 0)
    max_drawdown = float(metrics.get("max_drawdown", 0) or 0)
    sharpe = float(metrics.get("sharpe", 0) or 0)
    win_rate = float(metrics.get("win_rate", 0) or 0)
    return profit_factor + average_return + sharpe + (win_rate / 100) - (max_drawdown / 10)


def run_walk_forward_optimizer(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    strategy_mode: str = "deterministic",
    train_size: int = 90,
    validation_size: int = 45,
    horizon_candles: int = 3,
    min_history: int = 40,
    step_size: int = 5,
    parameter_grid: dict[str, list[Any]] | None = None,
    baseline_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    candles = normalize_ohlcv(df)
    baseline = {**BASELINE_PARAMS, **(baseline_params or {})}
    grid = parameter_grid or DEFAULT_PARAM_GRID
    warnings = []
    if len(candles) < train_size + validation_size:
        return {
            "baseline_params": baseline,
            "candidate_params": None,
            "train_metrics": {},
            "validation_metrics": {},
            "baseline_validation_metrics": {},
            "candidate_beats_baseline": False,
            "objective_metric": "profit_factor+average_return+sharpe+win_rate-max_drawdown",
            "warnings": ["insufficient_data"],
        }

    train = candles.iloc[:train_size]
    validation = candles.iloc[train_size:train_size + validation_size]
    candidates = expand_param_grid(grid)

    best_params = None
    best_train_metrics = {}
    best_train_score = -1e18
    for params in candidates:
        merged = {**baseline, **params}
        result = run_historical_replay(
            train,
            symbol=symbol,
            timeframe=timeframe,
            strategy_mode=strategy_mode,
            horizon_candles=horizon_candles,
            min_history=min_history,
            step_size=step_size,
            strategy_params=merged,
        )
        metrics = first_metric(result)
        score = objective_score(metrics)
        if score > best_train_score:
            best_train_score = score
            best_train_metrics = metrics
            best_params = merged

    baseline_validation = run_historical_replay(
        validation,
        symbol=symbol,
        timeframe=timeframe,
        strategy_mode=strategy_mode,
        horizon_candles=horizon_candles,
        min_history=min(min_history, max(5, len(validation) // 3)),
        step_size=step_size,
        strategy_params=baseline,
    )
    candidate_validation = run_historical_replay(
        validation,
        symbol=symbol,
        timeframe=timeframe,
        strategy_mode=strategy_mode,
        horizon_candles=horizon_candles,
        min_history=min(min_history, max(5, len(validation) // 3)),
        step_size=step_size,
        strategy_params=best_params or baseline,
    )
    baseline_metrics = first_metric(baseline_validation)
    candidate_metrics = first_metric(candidate_validation)
    candidate_score = objective_score(candidate_metrics)
    baseline_score = objective_score(baseline_metrics)
    candidate_beats = bool(candidate_score > baseline_score and float(candidate_metrics.get("evaluated_predictions", 0) or 0) > 0)
    if best_params is None:
        warnings.append("no_candidate_params_generated")

    return {
        "baseline_params": baseline,
        "candidate_params": best_params,
        "train_metrics": best_train_metrics,
        "validation_metrics": candidate_metrics,
        "baseline_validation_metrics": baseline_metrics,
        "candidate_beats_baseline": candidate_beats,
        "objective_metric": "profit_factor+average_return+sharpe+win_rate-max_drawdown",
        "warnings": warnings,
        "windows": {
            "train": {"start_index": 0, "end_index": train_size - 1},
            "validation": {"start_index": train_size, "end_index": train_size + validation_size - 1},
        },
    }
