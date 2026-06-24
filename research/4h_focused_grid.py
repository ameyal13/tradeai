"""4h Focused Research Grid.

Research-only grid to compare 4h_only features against BTC-context features
using the already validated sizing setup: h10, RR2.0, ATR1.5, low_costs.
"""
from __future__ import annotations

from typing import Any

from research.asset_universe import CRYPTO_UNIVERSE, normalize_crypto_symbols
from research.experiment_grid import ExperimentConfig
from research.research_registry import config_id


FOUR_H_FOCUSED_SYMBOLS = ["ADA", "ETH", "SOL"]
FOUR_H_FOCUSED_TIMEFRAMES = ["1h"]
FOUR_H_FOCUSED_HORIZON_CANDLES = [10]
FOUR_H_FOCUSED_RISK_REWARDS = [2.0]
FOUR_H_FOCUSED_ATR_STOP_MULTIPLIERS = [1.5]
FOUR_H_FOCUSED_COST_MODES = ["low_costs"]
FOUR_H_FOCUSED_STRATEGY_MODES = ["xgboost"]
FOUR_H_FOCUSED_FEATURE_SETS = ["4h_only", "baseline_plus_btc_context"]
FOUR_H_FOCUSED_MAX_CANDLES = 5000
FOUR_H_FOCUSED_WINDOW_SIZE_CANDLES = 600
FOUR_H_FOCUSED_STEP_SIZE_CANDLES = 250
FOUR_H_FOCUSED_PHASE = "4h_focused_v1"


def _normalize_symbol(symbol: str) -> str:
    value = str(symbol).upper().replace("/", "").replace("-", "").strip()
    if value.endswith("USDT"):
        value = value[:-4]
    return value


def _validate_symbols(symbols: list[str]) -> list[str]:
    cleaned = [_normalize_symbol(symbol) for symbol in normalize_crypto_symbols(symbols)]
    invalid = sorted(set(cleaned) - set(CRYPTO_UNIVERSE))
    if invalid:
        raise ValueError(f"Only crypto symbols are supported in this phase: {invalid}")
    unsupported = sorted(set(cleaned) - set(FOUR_H_FOCUSED_SYMBOLS))
    if unsupported:
        raise ValueError(f"4h Focused v1 only supports: {unsupported}")
    return cleaned


def build_four_h_focused_grid(
    symbols: list[str] | str | None = None,
    feature_sets: list[str] | None = None,
    max_candles: int = FOUR_H_FOCUSED_MAX_CANDLES,
    window_size_candles: int = FOUR_H_FOCUSED_WINDOW_SIZE_CANDLES,
    step_size_candles: int = FOUR_H_FOCUSED_STEP_SIZE_CANDLES,
    min_train_rows: int = 120,
) -> list[dict[str, Any]]:
    """Build the controlled 4h-focused grid."""
    active_symbols = _validate_symbols(normalize_crypto_symbols(symbols) if symbols else FOUR_H_FOCUSED_SYMBOLS)
    active_feature_sets = list(feature_sets or FOUR_H_FOCUSED_FEATURE_SETS)
    invalid_sets = sorted(set(active_feature_sets) - set(FOUR_H_FOCUSED_FEATURE_SETS))
    if invalid_sets:
        raise ValueError(f"Unsupported feature sets: {invalid_sets}")

    rows: list[dict[str, Any]] = []
    for symbol in active_symbols:
        for timeframe in FOUR_H_FOCUSED_TIMEFRAMES:
            for horizon in FOUR_H_FOCUSED_HORIZON_CANDLES:
                for rr in FOUR_H_FOCUSED_RISK_REWARDS:
                    for atr_stop in FOUR_H_FOCUSED_ATR_STOP_MULTIPLIERS:
                        for cost_mode in FOUR_H_FOCUSED_COST_MODES:
                            for strategy_mode in FOUR_H_FOCUSED_STRATEGY_MODES:
                                for feature_set in active_feature_sets:
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
                                        "research_phase": FOUR_H_FOCUSED_PHASE,
                                        "feature_set": feature_set,
                                        "feature_family": feature_set,
                                        "use_market_context_features": False,
                                    }
                                    row["config_id"] = config_id(row)
                                    rows.append(row)
    return rows
