"""Small opt-in research grid for Market Context Features v1.

This grid is intentionally separate from crypto_multi and focused_v2a so the
effect of the added features can be compared without rewriting prior evidence.
It is research-only and produces no operational signals.
"""
from __future__ import annotations

from typing import Any

from research.asset_universe import CRYPTO_UNIVERSE, normalize_crypto_symbols
from research.experiment_grid import ExperimentConfig
from research.research_registry import config_id


MARKET_CONTEXT_SYMBOLS = ["ADA", "ETH"]
MARKET_CONTEXT_TIMEFRAMES = ["1h"]
MARKET_CONTEXT_HORIZON_CANDLES = [10, 12]
MARKET_CONTEXT_RISK_REWARDS = [2.0, 2.5]
MARKET_CONTEXT_ATR_STOP_MULTIPLIERS = [1.25, 1.5]
MARKET_CONTEXT_COST_MODES = ["low_costs"]
MARKET_CONTEXT_STRATEGY_MODES = ["xgboost"]
MARKET_CONTEXT_MAX_CANDLES = 5000
MARKET_CONTEXT_WINDOW_SIZE_CANDLES = 600
MARKET_CONTEXT_STEP_SIZE_CANDLES = 250
MARKET_CONTEXT_PHASE = "market_context_features_v1"


def _validate_symbols(symbols: list[str]) -> list[str]:
    cleaned = normalize_crypto_symbols(symbols)
    invalid = sorted(set(cleaned) - set(CRYPTO_UNIVERSE))
    if invalid:
        raise ValueError(f"Only crypto symbols are supported in this phase: {invalid}")
    unsupported = sorted(set(cleaned) - set(MARKET_CONTEXT_SYMBOLS))
    if unsupported:
        raise ValueError(f"Market Context v1 only supports: {unsupported}")
    return cleaned


def build_market_context_research_grid(
    symbols: list[str] | str | None = None,
    max_candles: int = MARKET_CONTEXT_MAX_CANDLES,
    window_size_candles: int = MARKET_CONTEXT_WINDOW_SIZE_CANDLES,
    step_size_candles: int = MARKET_CONTEXT_STEP_SIZE_CANDLES,
    min_train_rows: int = 120,
) -> list[dict[str, Any]]:
    active_symbols = _validate_symbols(normalize_crypto_symbols(symbols) if symbols else MARKET_CONTEXT_SYMBOLS)
    rows: list[dict[str, Any]] = []
    for symbol in active_symbols:
        for timeframe in MARKET_CONTEXT_TIMEFRAMES:
            for horizon in MARKET_CONTEXT_HORIZON_CANDLES:
                for rr in MARKET_CONTEXT_RISK_REWARDS:
                    for atr_stop in MARKET_CONTEXT_ATR_STOP_MULTIPLIERS:
                        for cost_mode in MARKET_CONTEXT_COST_MODES:
                            for strategy_mode in MARKET_CONTEXT_STRATEGY_MODES:
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
                                    "research_phase": MARKET_CONTEXT_PHASE,
                                    "use_market_context_features": True,
                                    "feature_family": "current_plus_market_context_v1",
                                }
                                row["config_id"] = config_id(row)
                                rows.append(row)
    return rows
