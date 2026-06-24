"""Feature Expansion Grid v1.

This phase freezes sizing/model/costs and varies only opt-in research feature
families. It is research-only and produces no operational signals.
"""
from __future__ import annotations

from typing import Any

from research.asset_universe import CRYPTO_UNIVERSE, normalize_crypto_symbols
from research.experiment_grid import ExperimentConfig
from research.research_registry import config_id


FEATURE_EXPANSION_SYMBOLS = ["ADA", "ETH"]
FEATURE_EXPANSION_TIMEFRAMES = ["1h"]
FEATURE_EXPANSION_HORIZON_CANDLES = [10]
FEATURE_EXPANSION_RISK_REWARDS = [2.0]
FEATURE_EXPANSION_ATR_STOP_MULTIPLIERS = [1.5]
FEATURE_EXPANSION_COST_MODES = ["low_costs"]
FEATURE_EXPANSION_STRATEGY_MODES = ["xgboost"]
FEATURE_EXPANSION_FEATURE_SETS = [
    "baseline",
    "time_only",
    "baseline_plus_funding",
    "baseline_plus_4h",
    "baseline_plus_all_new",
    "funding_only",
    "4h_only",
    "baseline_plus_btc_context",
    "baseline_plus_all_new_plus_btc",
]
FEATURE_EXPANSION_MAX_CANDLES = 5000
FEATURE_EXPANSION_WINDOW_SIZE_CANDLES = 600
FEATURE_EXPANSION_STEP_SIZE_CANDLES = 250
FEATURE_EXPANSION_PHASE = "feature_expansion_v1"
TIME_ONLY_BASELINE_PF = 0.81
OBJECTIVE_PROFIT_FACTOR = 1.0


def _validate_symbols(symbols: list[str]) -> list[str]:
    cleaned = [
        symbol[:-4] if symbol.upper().endswith("USDT") else symbol
        for symbol in normalize_crypto_symbols(symbols)
    ]
    invalid = sorted(set(cleaned) - set(CRYPTO_UNIVERSE))
    if invalid:
        raise ValueError(f"Only crypto symbols are supported in this phase: {invalid}")
    unsupported = sorted(set(cleaned) - set(FEATURE_EXPANSION_SYMBOLS))
    if unsupported:
        raise ValueError(f"Feature Expansion v1 only supports: {unsupported}")
    return cleaned


def build_feature_expansion_grid(
    symbols: list[str] | str | None = None,
    feature_sets: list[str] | None = None,
    max_candles: int = FEATURE_EXPANSION_MAX_CANDLES,
    window_size_candles: int = FEATURE_EXPANSION_WINDOW_SIZE_CANDLES,
    step_size_candles: int = FEATURE_EXPANSION_STEP_SIZE_CANDLES,
    min_train_rows: int = 120,
) -> list[dict[str, Any]]:
    """Build the controlled Feature Expansion v1 grid."""
    active_symbols = _validate_symbols(normalize_crypto_symbols(symbols) if symbols else FEATURE_EXPANSION_SYMBOLS)
    active_feature_sets = list(feature_sets or FEATURE_EXPANSION_FEATURE_SETS)
    invalid_sets = sorted(set(active_feature_sets) - set(FEATURE_EXPANSION_FEATURE_SETS))
    if invalid_sets:
        raise ValueError(f"Unsupported feature sets: {invalid_sets}")

    rows: list[dict[str, Any]] = []
    for symbol in active_symbols:
        for timeframe in FEATURE_EXPANSION_TIMEFRAMES:
            for horizon in FEATURE_EXPANSION_HORIZON_CANDLES:
                for rr in FEATURE_EXPANSION_RISK_REWARDS:
                    for atr_stop in FEATURE_EXPANSION_ATR_STOP_MULTIPLIERS:
                        for cost_mode in FEATURE_EXPANSION_COST_MODES:
                            for strategy_mode in FEATURE_EXPANSION_STRATEGY_MODES:
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
                                        "research_phase": FEATURE_EXPANSION_PHASE,
                                        "feature_set": feature_set,
                                        "feature_family": feature_set,
                                        "time_only_baseline_pf": TIME_ONLY_BASELINE_PF,
                                        "objective_profit_factor": OBJECTIVE_PROFIT_FACTOR,
                                        "use_market_context_features": False,
                                    }
                                    row["config_id"] = config_id(row)
                                    rows.append(row)
    return rows
