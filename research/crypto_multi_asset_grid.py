"""Crypto Multi-Asset Research Grid v1.

This grid is separate from the general daemon and refined SOL registry. It is
research-only and produces no operational signals.
"""
from __future__ import annotations

from typing import Any

from research.asset_universe import CRYPTO_UNIVERSE, normalize_crypto_symbols
from research.experiment_grid import ExperimentConfig
from research.research_registry import config_id


CRYPTO_MULTI_SYMBOLS = ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "AVAX", "LINK"]
CRYPTO_MULTI_TIMEFRAMES = ["1h"]
CRYPTO_MULTI_HORIZON_CANDLES = [12, 14]
CRYPTO_MULTI_RISK_REWARDS = [2.0, 2.5]
CRYPTO_MULTI_ATR_STOP_MULTIPLIERS = [1.0, 1.25]
CRYPTO_MULTI_COST_MODES = ["low_costs"]
CRYPTO_MULTI_STRATEGY_MODES = ["xgboost"]
CRYPTO_MULTI_MAX_CANDLES = 5000
CRYPTO_MULTI_WINDOW_SIZE_CANDLES = 600
CRYPTO_MULTI_STEP_SIZE_CANDLES = 250


def _validate_crypto_symbols(symbols: list[str]) -> list[str]:
    allowed = set(CRYPTO_UNIVERSE)
    cleaned = normalize_crypto_symbols(symbols)
    invalid = sorted(set(cleaned) - allowed)
    if invalid:
        raise ValueError(f"Only crypto symbols are supported in this phase: {invalid}")
    return cleaned


def build_crypto_multi_asset_grid(
    symbols: list[str] | str | None = None,
    max_candles: int = CRYPTO_MULTI_MAX_CANDLES,
    window_size_candles: int = CRYPTO_MULTI_WINDOW_SIZE_CANDLES,
    step_size_candles: int = CRYPTO_MULTI_STEP_SIZE_CANDLES,
    min_train_rows: int = 120,
) -> list[dict[str, Any]]:
    """Build the approved Crypto Multi-Asset Research Grid v1."""
    active_symbols = _validate_crypto_symbols(
        normalize_crypto_symbols(symbols) if symbols else CRYPTO_MULTI_SYMBOLS
    )
    rows: list[dict[str, Any]] = []
    for symbol in active_symbols:
        for timeframe in CRYPTO_MULTI_TIMEFRAMES:
            for horizon in CRYPTO_MULTI_HORIZON_CANDLES:
                for rr in CRYPTO_MULTI_RISK_REWARDS:
                    for atr_stop in CRYPTO_MULTI_ATR_STOP_MULTIPLIERS:
                        for cost_mode in CRYPTO_MULTI_COST_MODES:
                            for strategy_mode in CRYPTO_MULTI_STRATEGY_MODES:
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
                                    "research_phase": "crypto_multi_asset_grid_v1",
                                }
                                row["config_id"] = config_id(row)
                                rows.append(row)
    return rows
