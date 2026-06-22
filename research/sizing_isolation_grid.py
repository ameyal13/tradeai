"""Sizing Isolation Research grid.

This phase freezes the baseline focused_v2A model/features and varies only
horizon, risk/reward, and ATR stop multiplier. It is research-only and produces
no operational signals.
"""
from __future__ import annotations

from typing import Any

from research.asset_universe import CRYPTO_UNIVERSE, normalize_crypto_symbols
from research.experiment_grid import ExperimentConfig
from research.research_registry import config_id


SIZING_SYMBOLS = ["ADA", "ETH"]
SIZING_TIMEFRAMES = ["1h"]
SIZING_HORIZON_CANDLES = [8, 10, 12, 14, 16]
SIZING_RISK_REWARDS = [1.5, 2.0, 2.5, 3.0]
SIZING_ATR_STOP_MULTIPLIERS = [1.0, 1.5, 2.0, 2.5]
SIZING_COST_MODES = ["low_costs"]
SIZING_STRATEGY_MODES = ["xgboost"]
SIZING_MAX_CANDLES = 5000
SIZING_WINDOW_SIZE_CANDLES = 600
SIZING_STEP_SIZE_CANDLES = 250
SIZING_PHASE = "sizing_isolation_v1"
SIZING_FEATURE_FAMILY = "current_xgboost_features"


def _validate_symbols(symbols: list[str]) -> list[str]:
    cleaned = normalize_crypto_symbols(symbols)
    invalid = sorted(set(cleaned) - set(CRYPTO_UNIVERSE))
    if invalid:
        raise ValueError(f"Only crypto symbols are supported in this phase: {invalid}")
    unsupported = sorted(set(cleaned) - set(SIZING_SYMBOLS))
    if unsupported:
        raise ValueError(f"Sizing Isolation v1 only supports: {unsupported}")
    return cleaned


def build_sizing_isolation_grid(
    symbols: list[str] | str | None = None,
    horizons: list[int] | None = None,
    max_candles: int = SIZING_MAX_CANDLES,
    window_size_candles: int = SIZING_WINDOW_SIZE_CANDLES,
    step_size_candles: int = SIZING_STEP_SIZE_CANDLES,
    min_train_rows: int = 120,
) -> list[dict[str, Any]]:
    """Build the approved sizing-isolation grid.

    Only horizon_candles, risk_reward, and atr_stop_multiplier vary. Model,
    feature family, thresholds, cost mode, strategy mode, timeframe, and
    validation process remain aligned with focused_v2A baseline.
    """
    active_symbols = _validate_symbols(normalize_crypto_symbols(symbols) if symbols else SIZING_SYMBOLS)
    active_horizons = [int(value) for value in (horizons or SIZING_HORIZON_CANDLES)]
    invalid_horizons = sorted(set(active_horizons) - set(SIZING_HORIZON_CANDLES))
    if invalid_horizons:
        raise ValueError(f"Unsupported sizing horizons: {invalid_horizons}")

    rows: list[dict[str, Any]] = []
    for symbol in active_symbols:
        for timeframe in SIZING_TIMEFRAMES:
            for horizon in active_horizons:
                for rr in SIZING_RISK_REWARDS:
                    for atr_stop in SIZING_ATR_STOP_MULTIPLIERS:
                        for cost_mode in SIZING_COST_MODES:
                            for strategy_mode in SIZING_STRATEGY_MODES:
                                setup = ExperimentConfig(
                                    symbol=symbol,
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
                                    "research_phase": SIZING_PHASE,
                                    "feature_family": SIZING_FEATURE_FAMILY,
                                    "use_market_context_features": False,
                                }
                                row["config_id"] = config_id(row)
                                rows.append(row)
    return rows
