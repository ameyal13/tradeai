"""Focused Research Grid v2A around crypto_multi watchlist assets.

This phase narrows research to ADA/ETH/SOL after crypto_multi found watchlist
activity there. It is research-only and produces no operational signals.
"""
from __future__ import annotations

from typing import Any

from research.asset_universe import CRYPTO_UNIVERSE, normalize_crypto_symbols
from research.experiment_grid import ExperimentConfig
from research.research_registry import config_id


FOCUSED_SYMBOLS = ["ADA", "ETH", "SOL"]
FOCUSED_TIMEFRAMES = ["1h"]
FOCUSED_HORIZON_CANDLES = [10, 12, 14]
FOCUSED_RISK_REWARDS = [2.0, 2.5, 2.8]
FOCUSED_ATR_STOP_MULTIPLIERS = [1.0, 1.25, 1.5]
FOCUSED_COST_MODES = ["low_costs"]
FOCUSED_STRATEGY_MODES = ["xgboost"]
FOCUSED_MAX_CANDLES = 5000
FOCUSED_WINDOW_SIZE_CANDLES = 600
FOCUSED_STEP_SIZE_CANDLES = 250
FOCUSED_PHASE = "focused_crypto_watchlist_v2a"


def _validate_symbols(symbols: list[str]) -> list[str]:
    allowed = set(CRYPTO_UNIVERSE)
    cleaned = normalize_crypto_symbols(symbols)
    invalid = sorted(set(cleaned) - allowed)
    if invalid:
        raise ValueError(f"Only crypto symbols are supported in this phase: {invalid}")
    unsupported = sorted(set(cleaned) - set(FOCUSED_SYMBOLS))
    if unsupported:
        raise ValueError(f"Focused v2A only supports watchlist symbols: {unsupported}")
    return cleaned


def build_focused_research_grid(
    symbols: list[str] | str | None = None,
    max_candles: int = FOCUSED_MAX_CANDLES,
    window_size_candles: int = FOCUSED_WINDOW_SIZE_CANDLES,
    step_size_candles: int = FOCUSED_STEP_SIZE_CANDLES,
    min_train_rows: int = 120,
) -> list[dict[str, Any]]:
    active_symbols = _validate_symbols(normalize_crypto_symbols(symbols) if symbols else FOCUSED_SYMBOLS)
    rows: list[dict[str, Any]] = []
    for symbol in active_symbols:
        for timeframe in FOCUSED_TIMEFRAMES:
            for horizon in FOCUSED_HORIZON_CANDLES:
                for rr in FOCUSED_RISK_REWARDS:
                    for atr_stop in FOCUSED_ATR_STOP_MULTIPLIERS:
                        for cost_mode in FOCUSED_COST_MODES:
                            for strategy_mode in FOCUSED_STRATEGY_MODES:
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
                                    "research_phase": FOCUSED_PHASE,
                                }
                                row["config_id"] = config_id(row)
                                rows.append(row)
    return rows
