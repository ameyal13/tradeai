"""Feature Policy Confirmation Grid v1.

This grid compares candidate feature policies against controls before any
promotion to live/shadow feature generation. It is research-only.
"""
from __future__ import annotations

from typing import Any

from research.asset_universe import CRYPTO_UNIVERSE, normalize_crypto_symbols
from research.experiment_grid import ExperimentConfig
from research.research_registry import config_id


FEATURE_POLICY_SYMBOLS = ["ADA", "ETH", "SOL"]
FEATURE_POLICY_TIMEFRAMES = ["1h"]
FEATURE_POLICY_HORIZON_CANDLES = [10]
FEATURE_POLICY_RISK_REWARDS = [2.0]
FEATURE_POLICY_ATR_STOP_MULTIPLIERS = [1.5]
FEATURE_POLICY_COST_MODES = ["low_costs"]
FEATURE_POLICY_STRATEGY_MODES = ["xgboost"]
FEATURE_POLICY_PHASE = "feature_policy_confirmation_v1"
FEATURE_POLICY_MAX_CANDLES = 5000
FEATURE_POLICY_WINDOW_SIZE_CANDLES = 600
FEATURE_POLICY_STEP_SIZE_CANDLES = 250

FEATURE_POLICY_FEATURE_SETS_BY_SYMBOL = {
    "ADA": ["baseline", "time_only", "baseline_plus_btc_context"],
    "ETH": ["baseline", "time_only", "baseline_plus_btc_context"],
    "SOL": ["baseline", "time_only", "4h_only", "baseline_plus_btc_context"],
}


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
    unsupported = sorted(set(cleaned) - set(FEATURE_POLICY_SYMBOLS))
    if unsupported:
        raise ValueError(f"Feature Policy Confirmation v1 only supports: {unsupported}")
    return cleaned


def build_feature_policy_confirmation_grid(
    symbols: list[str] | str | None = None,
    max_candles: int = FEATURE_POLICY_MAX_CANDLES,
    window_size_candles: int = FEATURE_POLICY_WINDOW_SIZE_CANDLES,
    step_size_candles: int = FEATURE_POLICY_STEP_SIZE_CANDLES,
    min_train_rows: int = 120,
) -> list[dict[str, Any]]:
    """Build the small control grid for symbol-specific feature policy."""
    active_symbols = _validate_symbols(normalize_crypto_symbols(symbols) if symbols else FEATURE_POLICY_SYMBOLS)
    rows: list[dict[str, Any]] = []
    for symbol in active_symbols:
        for timeframe in FEATURE_POLICY_TIMEFRAMES:
            for horizon in FEATURE_POLICY_HORIZON_CANDLES:
                for rr in FEATURE_POLICY_RISK_REWARDS:
                    for atr_stop in FEATURE_POLICY_ATR_STOP_MULTIPLIERS:
                        for cost_mode in FEATURE_POLICY_COST_MODES:
                            for strategy_mode in FEATURE_POLICY_STRATEGY_MODES:
                                for feature_set in FEATURE_POLICY_FEATURE_SETS_BY_SYMBOL[symbol]:
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
                                        "research_phase": FEATURE_POLICY_PHASE,
                                        "feature_set": feature_set,
                                        "feature_family": feature_set,
                                        "use_market_context_features": False,
                                    }
                                    row["config_id"] = config_id(row)
                                    rows.append(row)
    return rows
