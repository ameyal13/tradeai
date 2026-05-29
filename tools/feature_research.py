"""Offline feature research utilities for XGBoost strategy audits.

This module does not generate live signals. It measures whether current
features have out-of-sample signal under purged temporal validation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from tools.ml_engine import (
    FEATURE_COLS,
    _atr_label_pcts,
    build_trade_outcome_labels,
    predict_proba_xgboost,
    train_xgboost_model,
)
from tools.prediction_journal import calculate_profit_factor
from tools.strategy_signals import add_features


TECHNICAL_INDICATOR_COLS = ["rsi", "macd_hist", "ema_fast", "ema_slow"]
RETURN_COLS = ["return_1", "return_3"]
VOLATILITY_COLS = ["volatility_10", "atr", "atr_pct"]
TREND_COLS = ["ema_fast", "ema_slow", "macd_hist", "ema_distance_pct"]
VOLUME_COLS = ["relative_volume"]
TIME_COLS = ["hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos"]
REGIME_COLS = ["ema_distance_pct", "atr_pct"]


@dataclass(frozen=True)
class PurgedFold:
    train_positions: np.ndarray
    validation_positions: np.ndarray


def add_research_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add current XGBoost features plus research-only time/regime features."""
    data = add_features(df)
    index = pd.to_datetime(data.index, utc=True)
    hour = index.hour.to_numpy(dtype=float)
    day = index.dayofweek.to_numpy(dtype=float)

    data["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    data["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    data["day_of_week_sin"] = np.sin(2 * np.pi * day / 7)
    data["day_of_week_cos"] = np.cos(2 * np.pi * day / 7)
    close = data["close"].astype(float).replace(0, np.nan)
    data["ema_distance_pct"] = (data["close"].astype(float) - data["ema_slow"].astype(float)) / close
    data["atr_pct"] = data["atr"].astype(float) / close
    return data


def feature_families() -> dict[str, list[str]]:
    """Feature sets used by the audit. Strategy code remains unchanged."""
    current = list(FEATURE_COLS)
    return {
        "all_current": current,
        "all_current_plus_time_regime": current + TIME_COLS + REGIME_COLS,
        "no_technical_indicators": [col for col in current if col not in TECHNICAL_INDICATOR_COLS],
        "returns_only": list(RETURN_COLS),
        "volatility_atr_only": list(VOLATILITY_COLS),
        "trend_only": list(TREND_COLS),
        "volume_only": list(VOLUME_COLS),
        "time_only": list(TIME_COLS),
        "dummy_random": current,
    }


def feature_nan_summary(features: pd.DataFrame, feature_cols: list[str]) -> dict[str, int]:
    return {column: int(features[column].isna().sum()) for column in feature_cols if column in features.columns}


def future_close_return_pct(features: pd.DataFrame, horizon_candles: int) -> pd.Series:
    future = features["close"].shift(-horizon_candles)
    return (future / features["close"] - 1) * 100


def feature_correlations_to_future_return(
    features: pd.DataFrame,
    feature_cols: list[str],
    horizon_candles: int = 4,
) -> dict[str, float | None]:
    """Pearson correlation of each feature against future close return."""
    future_return = future_close_return_pct(features, horizon_candles)
    correlations: dict[str, float | None] = {}
    for column in feature_cols:
        if column not in features.columns:
            correlations[column] = None
            continue
        valid = features[column].notna() & future_return.notna()
        if int(valid.sum()) < 5 or features.loc[valid, column].nunique() < 2:
            correlations[column] = None
            continue
        value = features.loc[valid, column].corr(future_return.loc[valid])
        correlations[column] = None if pd.isna(value) else round(float(value), 6)
    return correlations


def build_directional_net_returns(
    df: pd.DataFrame,
    horizon_candles: int,
    stop_loss_pct: float,
    take_profit_pct: float,
    commission_pct: float,
    slippage_pct: float,
    spread_pct: float = 0.0003,
    stop_loss_pcts: pd.Series | np.ndarray | None = None,
    take_profit_pcts: pd.Series | np.ndarray | None = None,
) -> pd.DataFrame:
    """Net BUY/SELL returns using the same delayed entry and TP/SL path rules."""
    if horizon_candles <= 0:
        return pd.DataFrame(index=df.index, data={"buy_return_pct": np.nan, "sell_return_pct": np.nan})

    data = df.copy()
    for column in ["open", "high", "low", "close"]:
        if column not in data.columns:
            raise ValueError(f"Missing required OHLC column: {column}")
        data[column] = data[column].astype(float)

    opens = data["open"].to_numpy(dtype=float)
    highs = data["high"].to_numpy(dtype=float)
    lows = data["low"].to_numpy(dtype=float)
    closes = data["close"].to_numpy(dtype=float)
    stop_pcts = (
        np.asarray(stop_loss_pcts, dtype=float)
        if stop_loss_pcts is not None
        else np.full(len(data), float(stop_loss_pct), dtype=float)
    )
    take_pcts = (
        np.asarray(take_profit_pcts, dtype=float)
        if take_profit_pcts is not None
        else np.full(len(data), float(take_profit_pct), dtype=float)
    )
    buy_returns = np.full(len(data), np.nan, dtype=float)
    sell_returns = np.full(len(data), np.nan, dtype=float)

    for index_n in range(len(data)):
        entry_idx = index_n + 1
        end_idx = entry_idx + horizon_candles
        if entry_idx >= len(data) or end_idx > len(data):
            continue

        entry = opens[entry_idx]
        if not np.isfinite(entry) or entry <= 0:
            continue
        row_stop_pct = stop_pcts[index_n] if index_n < len(stop_pcts) else stop_loss_pct
        row_take_pct = take_pcts[index_n] if index_n < len(take_pcts) else take_profit_pct
        if not np.isfinite(row_stop_pct) or row_stop_pct <= 0:
            row_stop_pct = stop_loss_pct
        if not np.isfinite(row_take_pct) or row_take_pct <= 0:
            row_take_pct = take_profit_pct

        buy_sl = entry * (1 - row_stop_pct)
        buy_tp = entry * (1 + row_take_pct)
        sell_sl = entry * (1 + row_stop_pct)
        sell_tp = entry * (1 - row_take_pct)
        buy_exit = closes[end_idx - 1]
        sell_exit = closes[end_idx - 1]

        for high, low in zip(highs[entry_idx:end_idx], lows[entry_idx:end_idx]):
            if low <= buy_sl and high >= buy_tp:
                buy_exit = buy_sl
                break
            if low <= buy_sl:
                buy_exit = buy_sl
                break
            if high >= buy_tp:
                buy_exit = buy_tp
                break

        for high, low in zip(highs[entry_idx:end_idx], lows[entry_idx:end_idx]):
            if high >= sell_sl and low <= sell_tp:
                sell_exit = sell_sl
                break
            if high >= sell_sl:
                sell_exit = sell_sl
                break
            if low <= sell_tp:
                sell_exit = sell_tp
                break

        buy_costs = abs(entry + buy_exit) * (commission_pct + slippage_pct + spread_pct / 2)
        sell_costs = abs(entry + sell_exit) * (commission_pct + slippage_pct + spread_pct / 2)
        buy_returns[index_n] = (buy_exit - entry) / entry * 100 - (buy_costs / entry * 100)
        sell_returns[index_n] = (entry - sell_exit) / entry * 100 - (sell_costs / entry * 100)

    return pd.DataFrame(index=df.index, data={"buy_return_pct": buy_returns, "sell_return_pct": sell_returns})


def purged_walk_forward_splits(
    valid_positions: np.ndarray,
    n_splits: int,
    min_train_rows: int,
    horizon_candles: int,
) -> list[PurgedFold]:
    """Create temporal folds with an embargo so train labels cannot see validation candles."""
    positions = np.asarray(sorted(set(int(pos) for pos in valid_positions)), dtype=int)
    if len(positions) < min_train_rows + horizon_candles + n_splits:
        return []

    remaining = len(positions) - min_train_rows
    fold_size = max(1, remaining // (n_splits + 1))
    folds: list[PurgedFold] = []
    for fold in range(n_splits):
        val_start_offset = min_train_rows + fold * fold_size
        val_end_offset = min(val_start_offset + fold_size, len(positions))
        if val_end_offset <= val_start_offset:
            break
        validation = positions[val_start_offset:val_end_offset]
        if len(validation) == 0:
            continue
        train_cutoff = int(validation[0]) - horizon_candles
        train = positions[positions < train_cutoff]
        if len(train) < min_train_rows:
            continue
        folds.append(PurgedFold(train_positions=train, validation_positions=validation))
    return folds


def _profit_metrics(returns: list[float]) -> dict[str, Any]:
    outcomes = [{"return_pct": value} for value in returns]
    wins = [value for value in returns if value > 0]
    return {
        "trades": len(returns),
        "win_rate": round(len(wins) / len(returns) * 100, 6) if returns else 0,
        "average_return": round(float(np.mean(returns)), 6) if returns else 0,
        "total_return_pct": round(float(np.sum(returns)), 6) if returns else 0,
        "profit_factor": calculate_profit_factor(outcomes) if returns else 0,
    }


def _model_importance(model: Any, feature_cols: list[str]) -> dict[str, float]:
    values = getattr(model, "feature_importances_", None)
    if values is None:
        return {column: 0.0 for column in feature_cols}
    values = np.asarray(values, dtype=float)
    if values.sum() > 0:
        values = values / values.sum()
    return {column: round(float(values[idx]), 6) for idx, column in enumerate(feature_cols)}


def _safe_mean_importances(rows: list[dict[str, float]], feature_cols: list[str]) -> dict[str, float]:
    if not rows:
        return {column: 0.0 for column in feature_cols}
    return {
        column: round(float(np.mean([row.get(column, 0.0) for row in rows])), 6)
        for column in feature_cols
    }


def _select_returns(
    buy_prob: float,
    sell_prob: float,
    buy_return: float,
    sell_return: float,
    buy_threshold: float,
    sell_threshold: float,
) -> tuple[str, float | None]:
    buy_edge = buy_prob - buy_threshold
    sell_edge = sell_prob - sell_threshold
    if buy_prob >= buy_threshold and buy_edge >= sell_edge:
        return "BUY", buy_return
    if sell_prob >= sell_threshold and sell_edge > buy_edge:
        return "SELL", sell_return
    return "HOLD", None


def evaluate_feature_set(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    returns: pd.DataFrame,
    feature_cols: list[str],
    n_splits: int = 4,
    min_train_rows: int = 120,
    horizon_candles: int = 4,
    buy_threshold: float = 0.58,
    sell_threshold: float = 0.58,
    random_seed: int = 42,
    dummy_random: bool = False,
) -> dict[str, Any]:
    """Evaluate a feature set with purged temporal folds and EV metrics."""
    available_cols = [column for column in feature_cols if column in features.columns]
    if not available_cols:
        return {"status": "no_features", "feature_cols": feature_cols, "folds": 0}

    feature_valid = features[available_cols].notna().all(axis=1)
    return_valid = returns[["buy_return_pct", "sell_return_pct"]].notna().all(axis=1)
    positions = np.flatnonzero((feature_valid & return_valid).to_numpy())
    folds = purged_walk_forward_splits(positions, n_splits, min_train_rows, horizon_candles)
    rng = np.random.default_rng(random_seed)
    trade_returns: list[float] = []
    buy_trades = 0
    sell_trades = 0
    hold_count = 0
    fold_rows: list[dict[str, Any]] = []
    buy_importances: list[dict[str, float]] = []
    sell_importances: list[dict[str, float]] = []
    permutation_deltas = {column: [] for column in available_cols}

    for fold_index, fold in enumerate(folds):
        train = fold.train_positions
        validation = fold.validation_positions
        train_buy_mask = labels.iloc[train]["buy_win"].notna().to_numpy()
        train_sell_mask = labels.iloc[train]["sell_win"].notna().to_numpy()
        buy_train = train[train_buy_mask]
        sell_train = train[train_sell_mask]
        if not dummy_random and (len(buy_train) < min_train_rows or len(sell_train) < min_train_rows):
            fold_rows.append({
                "fold": fold_index,
                "status": "insufficient_labels",
                "buy_train_rows": int(len(buy_train)),
                "sell_train_rows": int(len(sell_train)),
                "validation_rows": int(len(validation)),
            })
            continue

        if dummy_random:
            buy_probs = rng.random(len(validation))
            sell_probs = rng.random(len(validation))
        else:
            buy_model = train_xgboost_model(
                features.iloc[buy_train][available_cols].to_numpy(dtype=float),
                labels.iloc[buy_train]["buy_win"].astype(int).to_numpy(),
            )
            sell_model = train_xgboost_model(
                features.iloc[sell_train][available_cols].to_numpy(dtype=float),
                labels.iloc[sell_train]["sell_win"].astype(int).to_numpy(),
            )
            validation_x = features.iloc[validation][available_cols].to_numpy(dtype=float)
            buy_probs = np.array([predict_proba_xgboost(buy_model, row) for row in validation_x])
            sell_probs = np.array([predict_proba_xgboost(sell_model, row) for row in validation_x])
            buy_importances.append(_model_importance(buy_model, available_cols))
            sell_importances.append(_model_importance(sell_model, available_cols))

        fold_returns: list[float] = []
        fold_buy = 0
        fold_sell = 0
        fold_hold = 0
        buy_return_values = returns.iloc[validation]["buy_return_pct"].to_numpy(dtype=float)
        sell_return_values = returns.iloc[validation]["sell_return_pct"].to_numpy(dtype=float)
        for row_idx, buy_prob in enumerate(buy_probs):
            side, value = _select_returns(
                float(buy_prob),
                float(sell_probs[row_idx]),
                float(buy_return_values[row_idx]),
                float(sell_return_values[row_idx]),
                buy_threshold,
                sell_threshold,
            )
            if side == "BUY":
                fold_buy += 1
            elif side == "SELL":
                fold_sell += 1
            else:
                fold_hold += 1
            if value is not None:
                fold_returns.append(float(value))

        if not dummy_random and fold_returns:
            base_average = float(np.mean(fold_returns))
            validation_frame = features.iloc[validation][available_cols].copy()
            for column in available_cols:
                permuted_frame = validation_frame.copy()
                permuted_frame[column] = rng.permutation(permuted_frame[column].to_numpy())
                perm_x = permuted_frame[available_cols].to_numpy(dtype=float)
                perm_buy_probs = np.array([predict_proba_xgboost(buy_model, row) for row in perm_x])
                perm_sell_probs = np.array([predict_proba_xgboost(sell_model, row) for row in perm_x])
                perm_returns = []
                for row_idx, buy_prob in enumerate(perm_buy_probs):
                    _, value = _select_returns(
                        float(buy_prob),
                        float(perm_sell_probs[row_idx]),
                        float(buy_return_values[row_idx]),
                        float(sell_return_values[row_idx]),
                        buy_threshold,
                        sell_threshold,
                    )
                    if value is not None:
                        perm_returns.append(float(value))
                perm_average = float(np.mean(perm_returns)) if perm_returns else 0.0
                permutation_deltas[column].append(base_average - perm_average)

        buy_trades += fold_buy
        sell_trades += fold_sell
        hold_count += fold_hold
        trade_returns.extend(fold_returns)
        fold_rows.append({
            "fold": fold_index,
            "status": "ok",
            "buy_train_rows": int(len(buy_train)),
            "sell_train_rows": int(len(sell_train)),
            "validation_rows": int(len(validation)),
            "trades": len(fold_returns),
            "buy_trades": fold_buy,
            "sell_trades": fold_sell,
            "hold_count": fold_hold,
            **_profit_metrics(fold_returns),
        })

    model_importance = {
        column: round((_safe_mean_importances(buy_importances, available_cols).get(column, 0.0)
                       + _safe_mean_importances(sell_importances, available_cols).get(column, 0.0)) / 2, 6)
        for column in available_cols
    }
    permutation_importance = {
        column: round(float(np.mean(values)), 6) if values else 0.0
        for column, values in permutation_deltas.items()
    }
    return {
        "status": "ok" if folds else "insufficient_data",
        "feature_cols": available_cols,
        "folds": len(folds),
        "buy_trades": buy_trades,
        "sell_trades": sell_trades,
        "hold_count": hold_count,
        "validation_rows": int(sum(len(fold.validation_positions) for fold in folds)),
        "folds_detail": fold_rows,
        "model_importance": model_importance,
        "permutation_importance": permutation_importance,
        **_profit_metrics(trade_returns),
    }


def run_feature_audit(
    df: pd.DataFrame,
    horizon_candles: int = 4,
    n_splits: int = 4,
    min_train_rows: int = 120,
    trade_label_scheme: str = "expected_value_classification",
    label_level_mode: str = "atr",
    stop_loss_pct: float = 0.03,
    take_profit_pct: float = 0.045,
    commission_pct: float = 0.001,
    slippage_pct: float = 0.0005,
    spread_pct: float = 0.0003,
    atr_stop_multiplier: float = 1.5,
    atr_take_profit_multiplier: float | None = None,
    min_risk_reward: float = 1.5,
    expected_value_threshold_pct: float = 0.05,
    buy_threshold: float = 0.58,
    sell_threshold: float = 0.58,
) -> dict[str, Any]:
    """Run feature importance, permutation importance, and ablation audit."""
    features = add_research_features(df)
    stop_pcts = None
    take_pcts = None
    if label_level_mode == "atr":
        stop_pcts, take_pcts = _atr_label_pcts(
            features,
            atr_stop_multiplier=atr_stop_multiplier,
            min_rr=min_risk_reward,
            atr_take_profit_multiplier=atr_take_profit_multiplier,
            fallback_stop_loss_pct=stop_loss_pct,
            fallback_take_profit_pct=take_profit_pct,
        )
    labels = build_trade_outcome_labels(
        features,
        horizon_candles=horizon_candles,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        commission_pct=commission_pct,
        slippage_pct=slippage_pct,
        spread_pct=spread_pct,
        stop_loss_pcts=stop_pcts,
        take_profit_pcts=take_pcts,
        label_scheme=trade_label_scheme,
        expected_value_threshold_pct=expected_value_threshold_pct,
    )
    returns = build_directional_net_returns(
        features,
        horizon_candles=horizon_candles,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        commission_pct=commission_pct,
        slippage_pct=slippage_pct,
        spread_pct=spread_pct,
        stop_loss_pcts=stop_pcts,
        take_profit_pcts=take_pcts,
    )

    families = feature_families()
    ablations = {
        name: evaluate_feature_set(
            features,
            labels,
            returns,
            cols,
            n_splits=n_splits,
            min_train_rows=min_train_rows,
            horizon_candles=horizon_candles,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
            dummy_random=name == "dummy_random",
        )
        for name, cols in families.items()
    }
    all_current = ablations.get("all_current", {})
    correlations = feature_correlations_to_future_return(features, FEATURE_COLS, horizon_candles=horizon_candles)
    importances = all_current.get("model_importance", {})
    removal_candidates = [
        column
        for column in FEATURE_COLS
        if (importances.get(column, 0.0) < 0.05)
        or (correlations.get(column) is None)
        or (abs(float(correlations.get(column) or 0.0)) < 0.03)
    ]
    return {
        "rows": len(features),
        "current_feature_cols": list(FEATURE_COLS),
        "research_feature_cols": sorted({col for cols in families.values() for col in cols}),
        "label_scheme": trade_label_scheme,
        "label_level_mode": label_level_mode,
        "horizon_candles": horizon_candles,
        "feature_nan_summary": feature_nan_summary(features, sorted({col for cols in families.values() for col in cols})),
        "raw_buy_label_count": int(labels["buy_win"].notna().sum()),
        "raw_sell_label_count": int(labels["sell_win"].notna().sum()),
        "raw_buy_positive_count": int((labels["buy_win"] == 1).sum()),
        "raw_sell_positive_count": int((labels["sell_win"] == 1).sum()),
        "feature_correlations_to_future_return": correlations,
        "removal_candidates": removal_candidates,
        "ablation_results": ablations,
        "methodology": {
            "split": "purged_walk_forward",
            "purge_candles": horizon_candles,
            "train_before_validation": True,
            "sentiment_disabled": True,
            "leakage_notes": [
                "Research folds purge horizon_candles before validation so train labels cannot consume validation candles.",
                "Current strategy features use rolling indicators on closed candles; no centered rolling windows were found.",
                "Raw EMA/ATR price-level features are not normalized in the live XGBoost feature set and may overfit symbol/price regime.",
            ],
        },
    }
