"""
ML Engine para TradeAI.
Reemplaza la regresión logística numpy con XGBoost + walk-forward validation.
Compatible con la interfaz existente de strategy_signals.py.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from tools.sentiment_engine import fear_greed_to_feature

FEATURE_COLS = [
    "rsi", "macd_hist", "ema_fast", "ema_slow",
    "relative_volume", "return_1", "return_3",
    "volatility_10", "atr",
]


class ConstantProbabilityModel:
    def __init__(self, probability: float):
        self.probability = float(probability)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        rows = len(np.atleast_2d(x))
        prob = min(0.99, max(0.01, self.probability))
        return np.tile(np.array([[1 - prob, prob]], dtype=float), (rows, 1))


def _xgboost_classifier():
    try:
        from xgboost import XGBClassifier
    except Exception as exc:
        raise ImportError("xgboost is not installed") from exc
    return XGBClassifier


def _feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    for col in FEATURE_COLS:
        if col not in data.columns:
            data[col] = np.nan
    return data


def _feature_cols(sentiment_features: dict[str, Any] | None = None) -> list[str]:
    cols = list(FEATURE_COLS)
    if sentiment_features is not None:
        cols.append("fear_greed_feature")
    return cols


def _apply_sentiment_features(df: pd.DataFrame, sentiment_features: dict[str, Any] | None = None) -> pd.DataFrame:
    data = df.copy()
    if sentiment_features is None:
        return data
    value = sentiment_features.get("value", 50)
    data["fear_greed_feature"] = fear_greed_to_feature(int(value))
    return data


def build_labels(df: pd.DataFrame, horizon_candles: int = 1, threshold_pct: float = 0.003) -> pd.Series:
    """
    Genera etiquetas binarias: 1 si el precio sube >= threshold_pct en los
    próximos horizon_candles, 0 si no.
    La última fila siempre es NaN (no hay futuro para labelear).
    """
    if "close" not in df.columns:
        raise ValueError("df must include a close column")
    future_return = df["close"].shift(-horizon_candles) / df["close"] - 1
    labels = (future_return >= threshold_pct).astype(float)
    labels[future_return.isna()] = np.nan
    labels.iloc[-1] = np.nan
    return labels


def build_trade_outcome_labels(
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
    """Fast directional trade labels equivalent to TP/SL touch outcomes.

    Costs stay in the signature for experiment metadata parity, but this
    primary directional label is based on TP/SL touches. EXPIRED, BREAKEVEN,
    and INVALID_DATA remain NaN. Optional per-row stop/take-profit pct arrays
    let labels match ATR-based strategy risk levels without using future data.
    """
    if horizon_candles <= 0:
        return pd.DataFrame(index=df.index, data={"buy_win": np.nan, "sell_win": np.nan})

    data = df.copy()
    if "timestamp" in data.columns:
        data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    for column in ["open", "high", "low", "close"]:
        if column not in data.columns:
            raise ValueError(f"Missing required OHLC column: {column}")
        data[column] = data[column].astype(float)

    open_prices = data["open"].to_numpy(dtype=float)
    highs = data["high"].to_numpy(dtype=float)
    lows = data["low"].to_numpy(dtype=float)
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
    buy_labels = np.full(len(data), np.nan, dtype=float)
    sell_labels = np.full(len(data), np.nan, dtype=float)

    for index_n in range(len(data)):
        entry_idx = index_n + 1
        if entry_idx >= len(data):
            continue
        end_idx = entry_idx + horizon_candles
        if end_idx > len(data):
            continue

        entry = open_prices[entry_idx]
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

        path_highs = highs[entry_idx:end_idx]
        path_lows = lows[entry_idx:end_idx]

        for high, low in zip(path_highs, path_lows):
            buy_hit_sl = low <= buy_sl
            buy_hit_tp = high >= buy_tp
            if buy_hit_sl and buy_hit_tp:
                buy_labels[index_n] = 0.0
                break
            if buy_hit_sl:
                buy_labels[index_n] = 0.0
                break
            if buy_hit_tp:
                buy_labels[index_n] = 1.0
                break

        for high, low in zip(path_highs, path_lows):
            sell_hit_sl = high >= sell_sl
            sell_hit_tp = low <= sell_tp
            if sell_hit_sl and sell_hit_tp:
                sell_labels[index_n] = 0.0
                break
            if sell_hit_sl:
                sell_labels[index_n] = 0.0
                break
            if sell_hit_tp:
                sell_labels[index_n] = 1.0
                break

    return pd.DataFrame(index=df.index, data={"buy_win": buy_labels, "sell_win": sell_labels})


def train_xgboost_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
) -> Any:
    """
    Entrena un XGBoostClassifier con los parámetros optimizados para señales
    de trading (class_weight balanceado, n_estimators=200, max_depth=4,
    learning_rate=0.05, subsample=0.8).
    Usa early stopping si se pasa validación.
    Retorna el modelo entrenado.
    """
    y_train = np.asarray(y_train, dtype=int)
    unique = np.unique(y_train)
    if len(unique) < 2:
        return ConstantProbabilityModel(float(unique[0]) if len(unique) else 0.5)

    XGBClassifier = _xgboost_classifier()
    positives = max(1, int((y_train == 1).sum()))
    negatives = max(1, int((y_train == 0).sum()))
    scale_pos_weight = negatives / positives
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=1,
    )
    fit_kwargs: dict[str, Any] = {}
    if x_val is not None and y_val is not None and len(x_val) and len(np.unique(y_val)) >= 2:
        fit_kwargs["eval_set"] = [(x_val, y_val)]
        fit_kwargs["verbose"] = False
    model.fit(x_train, y_train, **fit_kwargs)
    return model


def predict_proba_xgboost(model: Any, x: np.ndarray) -> float:
    """
    Retorna la probabilidad de subida (clase 1) para una sola fila de features.
    """
    row = np.asarray(x, dtype=float).reshape(1, -1)
    proba = model.predict_proba(row)
    if proba.shape[1] == 1:
        return float(proba[0, 0])
    return float(proba[0, 1])


def _prepared_dataset(
    df_features: pd.DataFrame,
    horizon_candles: int = 1,
    threshold_pct: float = 0.003,
    feature_cols: list[str] | None = None,
    labels: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    data = _feature_frame(df_features)
    cols = feature_cols or FEATURE_COLS
    labels = labels if labels is not None else build_labels(data, horizon_candles=horizon_candles, threshold_pct=threshold_pct)
    valid = data[cols].notna().all(axis=1) & labels.notna()
    return data.loc[valid, cols], labels.loc[valid].astype(int)


def _prepared_trade_dataset(
    df_features: pd.DataFrame,
    trade_labels: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    data = _feature_frame(df_features)
    features_valid = data[feature_cols].notna().all(axis=1)
    buy_valid = features_valid & trade_labels["buy_win"].notna()
    sell_valid = features_valid & trade_labels["sell_win"].notna()
    return (
        data.loc[buy_valid, feature_cols],
        trade_labels.loc[buy_valid, "buy_win"].astype(int),
        data.loc[sell_valid, feature_cols],
        trade_labels.loc[sell_valid, "sell_win"].astype(int),
    )


def _feature_nan_summary(data: pd.DataFrame, feature_cols: list[str]) -> dict[str, int]:
    return {column: int(data[column].isna().sum()) for column in feature_cols}


def _atr_label_pcts(
    df_features: pd.DataFrame,
    atr_stop_multiplier: float,
    min_rr: float,
    atr_take_profit_multiplier: float | None,
    fallback_stop_loss_pct: float,
    fallback_take_profit_pct: float,
) -> tuple[np.ndarray, np.ndarray]:
    data = _feature_frame(df_features)
    stop_distance = data["atr"].astype(float) * atr_stop_multiplier
    fallback_stop_distance = data["close"].astype(float) * fallback_stop_loss_pct
    stop_distance = stop_distance.where(np.isfinite(stop_distance) & (stop_distance > 0), fallback_stop_distance)
    take_distance = (
        data["atr"].astype(float) * atr_take_profit_multiplier
        if atr_take_profit_multiplier is not None
        else stop_distance * min_rr
    )
    fallback_take_distance = data["close"].astype(float) * fallback_take_profit_pct
    take_distance = take_distance.where(np.isfinite(take_distance) & (take_distance > 0), fallback_take_distance)
    close = data["close"].astype(float).replace(0, np.nan)
    return (stop_distance / close).to_numpy(dtype=float), (take_distance / close).to_numpy(dtype=float)


def _positive_count(labels: pd.Series | None) -> int:
    if labels is None or len(labels) == 0:
        return 0
    return int((labels == 1).sum())


def _positive_rate(labels: pd.Series | None) -> float | None:
    if labels is None or len(labels) == 0:
        return None
    return round(_positive_count(labels) / len(labels), 6)


def walk_forward_accuracy(
    df_features: pd.DataFrame,
    n_splits: int = 5,
    horizon_candles: int = 1,
    feature_cols: list[str] | None = None,
) -> float | None:
    """
    Calcula accuracy promedio con walk-forward cross-validation.
    Divide el dataset en n_splits folds temporales.
    En cada fold: entrena en el pasado, valida en el futuro inmediato.
    Retorna accuracy promedio (float 0-1) o None si hay datos insuficientes.
    """
    x, y = _prepared_dataset(df_features, horizon_candles=horizon_candles, feature_cols=feature_cols)
    if len(x) < max(60, n_splits * 20):
        return None

    fold_size = len(x) // (n_splits + 1)
    if fold_size < 10:
        return None

    scores = []
    for fold in range(n_splits):
        train_end = fold_size * (fold + 1)
        val_end = min(train_end + fold_size, len(x))
        if val_end <= train_end:
            break
        x_train = x.iloc[:train_end].to_numpy(dtype=float)
        y_train = y.iloc[:train_end].to_numpy(dtype=int)
        x_val = x.iloc[train_end:val_end].to_numpy(dtype=float)
        y_val = y.iloc[train_end:val_end].to_numpy(dtype=int)
        model = train_xgboost_model(x_train, y_train, x_val, y_val)
        probs = np.array([predict_proba_xgboost(model, row) for row in x_val])
        preds = (probs >= 0.5).astype(int)
        scores.append(float((preds == y_val).mean()))

    if not scores:
        return None
    raw_score = float(np.mean(scores))
    if len(np.unique(y)) < 2:
        return 0.5
    return round(raw_score, 6)


def xgboost_signal(
    df: pd.DataFrame,
    horizon_minutes: int = 60,
    min_train_rows: int = 200,
    strategy_params: dict[str, Any] | None = None,
    sentiment_features: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Función principal. Recibe un DataFrame con OHLCV, retorna un dict con:
    {
      "model_available": bool,
      "probability_up": float | None,
      "signal": "BUY" | "SELL" | "HOLD",
      "confidence": float,       # 0-100
      "validation_accuracy": float | None,
      "walk_forward_accuracy": float | None,
      "train_rows": int,
      "reason": str,             # descripción para logging
    }

    Lógica:
    - Si hay < min_train_rows filas con features completas → model_available=False, signal="HOLD"
    - Split temporal 70/30 (nunca mezclar futuro con pasado)
    - Entrenar XGBoost en 70%
    - Validar en 30%
    - Predecir en la última fila cerrada (iloc[-2] para evitar lookahead)
    - buy_threshold=0.58, sell_threshold=0.42 (configurable vía strategy_params)
    """
    strategy_params = strategy_params or {}
    min_train_rows = int(strategy_params.get("min_train_rows", min_train_rows))
    threshold_pct = float(strategy_params.get("label_threshold_pct", 0.003))
    horizon_candles = int(strategy_params.get("horizon_candles", 1))
    buy_threshold = float(strategy_params.get("probability_buy_threshold", 0.58))
    sell_threshold = float(strategy_params.get("probability_sell_threshold", 0.42))
    use_trade_labels = bool(strategy_params.get("use_trade_labels", False))
    label_type = "trade_outcome_directional" if use_trade_labels else "price_return"
    buy_win_threshold = float(strategy_params.get("buy_win_threshold", buy_threshold))
    sell_win_threshold = float(strategy_params.get("sell_win_threshold", 1 - sell_threshold))
    label_stop_loss_pct = float(strategy_params.get("stop_loss_pct", 0.03))
    label_take_profit_pct = float(strategy_params.get("take_profit_pct", 0.045))
    label_commission_pct = float(strategy_params.get("commission_pct", 0.001))
    label_slippage_pct = float(strategy_params.get("slippage_pct", 0.0005))
    label_spread_pct = float(strategy_params.get("spread_pct", 0.0003))
    min_rr = float(strategy_params.get("min_risk_reward", 1.5))
    atr_stop_multiplier = float(strategy_params.get("atr_stop_multiplier", 1.5))
    atr_take_profit_multiplier_value = strategy_params.get("atr_take_profit_multiplier")
    atr_take_profit_multiplier = (
        float(atr_take_profit_multiplier_value)
        if atr_take_profit_multiplier_value is not None
        else None
    )
    explicit_fixed_label_levels = "stop_loss_pct" in strategy_params or "take_profit_pct" in strategy_params
    label_level_mode = str(
        strategy_params.get(
            "label_level_mode",
            "fixed_pct" if explicit_fixed_label_levels else "atr",
        )
    )
    label_params = {
        "label_type": label_type,
        "min_train_rows": min_train_rows if use_trade_labels else None,
        "label_level_mode": label_level_mode if use_trade_labels else None,
        "label_stop_loss_pct": label_stop_loss_pct if use_trade_labels else None,
        "label_take_profit_pct": label_take_profit_pct if use_trade_labels else None,
        "label_horizon_candles": horizon_candles if use_trade_labels else None,
        "label_atr_stop_multiplier": atr_stop_multiplier if use_trade_labels and label_level_mode == "atr" else None,
        "label_atr_take_profit_multiplier": atr_take_profit_multiplier if use_trade_labels and label_level_mode == "atr" else None,
        "label_min_risk_reward": min_rr if use_trade_labels and label_level_mode == "atr" else None,
        "label_costs": {
            "commission_pct": label_commission_pct,
            "slippage_pct": label_slippage_pct,
            "spread_pct": label_spread_pct,
        } if use_trade_labels else None,
        "label_level_note": (
            "atr_aligned_trade_labels; costs_recorded_but_primary_label_is_tp_sl_touch"
            if use_trade_labels and label_level_mode == "atr"
            else "fixed_pct_trade_labels; costs_recorded_but_primary_label_is_tp_sl_touch"
            if use_trade_labels
            else None
        ),
    }
    trade_diag = {
        "raw_buy_label_count": 0,
        "raw_sell_label_count": 0,
        "raw_buy_positive_count": 0,
        "raw_sell_positive_count": 0,
        "feature_valid_count": 0,
        "feature_nan_summary": {},
        "buy_label_count": 0,
        "sell_label_count": 0,
        "buy_positive_count": 0,
        "sell_positive_count": 0,
        "buy_positive_rate": None,
        "sell_positive_rate": None,
        "probability_buy_win": None,
        "probability_sell_win": None,
        "buy_threshold": buy_win_threshold if use_trade_labels else buy_threshold,
        "sell_threshold": sell_win_threshold if use_trade_labels else sell_threshold,
        "decision_margin": None,
        "hold_reason": None,
    } if use_trade_labels else {}

    data = _apply_sentiment_features(_feature_frame(df), sentiment_features)
    feature_cols = _feature_cols(sentiment_features)
    if use_trade_labels:
        trade_diag["feature_nan_summary"] = _feature_nan_summary(data, feature_cols)
        trade_diag["feature_valid_count"] = int(data[feature_cols].notna().all(axis=1).sum())
    if len(data) < 2:
        return {
            "model_available": False,
            "probability_up": None,
            "signal": "HOLD",
            "confidence": 20.0,
            "validation_accuracy": None,
            "walk_forward_accuracy": None,
            "train_rows": 0,
            "reason": "insufficient_rows_for_closed_candle_prediction",
            "label_type": label_type,
            **trade_diag,
            "hold_reason": "model_unavailable" if use_trade_labels else None,
            **label_params,
        }

    if use_trade_labels:
        label_stop_pcts = None
        label_take_pcts = None
        if label_level_mode == "atr":
            label_stop_pcts, label_take_pcts = _atr_label_pcts(
                data.iloc[:-1],
                atr_stop_multiplier=atr_stop_multiplier,
                min_rr=min_rr,
                atr_take_profit_multiplier=atr_take_profit_multiplier,
                fallback_stop_loss_pct=label_stop_loss_pct,
                fallback_take_profit_pct=label_take_profit_pct,
            )
        trade_labels = build_trade_outcome_labels(
            data.iloc[:-1],
            horizon_candles=horizon_candles,
            stop_loss_pct=label_stop_loss_pct,
            take_profit_pct=label_take_profit_pct,
            commission_pct=label_commission_pct,
            slippage_pct=label_slippage_pct,
            spread_pct=label_spread_pct,
            stop_loss_pcts=label_stop_pcts,
            take_profit_pcts=label_take_pcts,
        )
        buy_x, buy_y, sell_x, sell_y = _prepared_trade_dataset(data.iloc[:-1], trade_labels, feature_cols)
        x = buy_x
        y = buy_y
        trade_diag.update({
            "raw_buy_label_count": int(trade_labels["buy_win"].notna().sum()),
            "raw_sell_label_count": int(trade_labels["sell_win"].notna().sum()),
            "raw_buy_positive_count": int((trade_labels["buy_win"] == 1).sum()),
            "raw_sell_positive_count": int((trade_labels["sell_win"] == 1).sum()),
            "buy_label_count": int(len(buy_y)),
            "sell_label_count": int(len(sell_y)),
            "buy_positive_count": _positive_count(buy_y),
            "sell_positive_count": _positive_count(sell_y),
            "buy_positive_rate": _positive_rate(buy_y),
            "sell_positive_rate": _positive_rate(sell_y),
        })
    else:
        x, y = _prepared_dataset(
            data.iloc[:-1],
            horizon_candles=horizon_candles,
            threshold_pct=threshold_pct,
            feature_cols=feature_cols,
        )
    prediction_row = data.iloc[-2][feature_cols]
    if prediction_row.isna().any():
        return {
            "model_available": False,
            "probability_up": None,
            "signal": "HOLD",
            "confidence": 20.0,
            "validation_accuracy": None,
            "walk_forward_accuracy": None,
            "train_rows": len(x),
            "reason": "prediction_row_features_incomplete",
            "label_type": label_type,
            **trade_diag,
            "hold_reason": "model_unavailable" if use_trade_labels else None,
            **label_params,
        }
    if use_trade_labels and int(trade_diag["raw_buy_label_count"]) < min_train_rows:
        return {
            "model_available": False,
            "probability_up": None,
            "signal": "HOLD",
            "confidence": 25.0,
            "validation_accuracy": None,
            "walk_forward_accuracy": None,
            "train_rows": int(len(buy_y)),
            "reason": "insufficient_raw_buy_labels",
            "label_type": label_type,
            **trade_diag,
            "hold_reason": "insufficient_raw_buy_labels",
            **label_params,
        }
    if use_trade_labels and int(trade_diag["raw_sell_label_count"]) < min_train_rows:
        return {
            "model_available": False,
            "probability_up": None,
            "signal": "HOLD",
            "confidence": 25.0,
            "validation_accuracy": None,
            "walk_forward_accuracy": None,
            "train_rows": int(len(sell_y)),
            "reason": "insufficient_raw_sell_labels",
            "label_type": label_type,
            **trade_diag,
            "hold_reason": "insufficient_raw_sell_labels",
            **label_params,
        }
    if use_trade_labels and len(buy_y) < min_train_rows:
        return {
            "model_available": False,
            "probability_up": None,
            "signal": "HOLD",
            "confidence": 25.0,
            "validation_accuracy": None,
            "walk_forward_accuracy": None,
            "train_rows": int(len(buy_y)),
            "reason": "insufficient_buy_labels_after_feature_filter",
            "label_type": label_type,
            **trade_diag,
            "hold_reason": "insufficient_buy_labels",
            **label_params,
        }
    if use_trade_labels and len(sell_y) < min_train_rows:
        return {
            "model_available": False,
            "probability_up": None,
            "signal": "HOLD",
            "confidence": 25.0,
            "validation_accuracy": None,
            "walk_forward_accuracy": None,
            "train_rows": int(len(sell_y)),
            "reason": "insufficient_sell_labels_after_feature_filter",
            "label_type": label_type,
            **trade_diag,
            "hold_reason": "insufficient_sell_labels",
            **label_params,
        }
    if len(x) < min_train_rows:
        return {
            "model_available": False,
            "probability_up": None,
            "signal": "HOLD",
            "confidence": 25.0,
            "validation_accuracy": None,
            "walk_forward_accuracy": None,
            "train_rows": len(x),
            "reason": "insufficient_training_rows",
            "label_type": label_type,
            **trade_diag,
            "hold_reason": "insufficient_train_rows" if use_trade_labels else None,
            **label_params,
        }

    validation_accuracy = None
    probability_sell_win = None
    if use_trade_labels:
        buy_split = max(1, int(len(buy_x) * 0.7))
        sell_split = max(1, int(len(sell_x) * 0.7))
        if buy_split >= len(buy_x):
            buy_split = len(buy_x) - 1
        if sell_split >= len(sell_x):
            sell_split = len(sell_x) - 1
        x_train = buy_x.iloc[:buy_split].to_numpy(dtype=float)
        y_train = buy_y.iloc[:buy_split].to_numpy(dtype=int)
        x_val = buy_x.iloc[buy_split:].to_numpy(dtype=float)
        y_val = buy_y.iloc[buy_split:].to_numpy(dtype=int)
        model = train_xgboost_model(x_train, y_train, x_val, y_val)
        sell_train_x = sell_x.iloc[:sell_split].to_numpy(dtype=float)
        sell_train_y = sell_y.iloc[:sell_split].to_numpy(dtype=int)
        sell_val_x = sell_x.iloc[sell_split:].to_numpy(dtype=float)
        sell_val_y = sell_y.iloc[sell_split:].to_numpy(dtype=int)
        sell_model = train_xgboost_model(sell_train_x, sell_train_y, sell_val_x, sell_val_y)
        if len(x_val) and len(sell_val_x):
            buy_probs = np.array([predict_proba_xgboost(model, row) for row in x_val])
            sell_probs = np.array([predict_proba_xgboost(sell_model, row) for row in sell_val_x])
            buy_preds = (buy_probs >= buy_win_threshold).astype(int)
            sell_preds = (sell_probs >= sell_win_threshold).astype(int)
            buy_accuracy = float((buy_preds == y_val).mean())
            sell_accuracy = float((sell_preds == sell_val_y).mean())
            validation_accuracy = round((buy_accuracy + sell_accuracy) / 2, 6)
        probability = predict_proba_xgboost(model, prediction_row.to_numpy(dtype=float))
        probability_sell_win = predict_proba_xgboost(sell_model, prediction_row.to_numpy(dtype=float))
        buy_edge = probability - buy_win_threshold
        sell_edge = probability_sell_win - sell_win_threshold
        decision_margin = max(buy_edge, sell_edge)
        trade_diag.update({
            "probability_buy_win": round(float(probability), 6),
            "probability_sell_win": round(float(probability_sell_win), 6),
            "decision_margin": round(float(decision_margin), 6),
        })
        if probability >= buy_win_threshold and buy_edge >= sell_edge:
            signal = "BUY"
            confidence = min(95.0, max(5.0, probability * 100))
        elif probability_sell_win >= sell_win_threshold and sell_edge > buy_edge:
            signal = "SELL"
            confidence = min(95.0, max(5.0, probability_sell_win * 100))
        else:
            signal = "HOLD"
            confidence = min(95.0, max(5.0, max(probability, probability_sell_win) * 100))
            trade_diag["hold_reason"] = "probabilities_below_threshold" if probability < buy_win_threshold and probability_sell_win < sell_win_threshold else "no_directional_edge"
    else:
        split = max(1, int(len(x) * 0.7))
        if split >= len(x):
            split = len(x) - 1
        x_train = x.iloc[:split].to_numpy(dtype=float)
        y_train = y.iloc[:split].to_numpy(dtype=int)
        x_val = x.iloc[split:].to_numpy(dtype=float)
        y_val = y.iloc[split:].to_numpy(dtype=int)
        model = train_xgboost_model(x_train, y_train, x_val, y_val)
        if len(x_val):
            probs = np.array([predict_proba_xgboost(model, row) for row in x_val])
            preds = (probs >= 0.5).astype(int)
            validation_accuracy = round(float((preds == y_val).mean()), 6)
            if len(np.unique(y)) < 2:
                validation_accuracy = 0.5

    if not use_trade_labels:
        probability = predict_proba_xgboost(model, prediction_row.to_numpy(dtype=float))
        if probability >= buy_threshold:
            signal = "BUY"
        elif probability <= sell_threshold:
            signal = "SELL"
        else:
            signal = "HOLD"
        confidence = min(95.0, max(5.0, abs(probability - 0.5) * 200))

    wf_accuracy = None if use_trade_labels else walk_forward_accuracy(data.iloc[:-1], horizon_candles=horizon_candles, feature_cols=feature_cols)
    return {
        "model_available": True,
        "probability_up": round(float(probability), 6),
        "probability_buy_win": round(float(probability), 6) if use_trade_labels else None,
        "probability_sell_win": round(float(probability_sell_win), 6) if probability_sell_win is not None else None,
        "signal": signal,
        "confidence": round(float(confidence), 6),
        "validation_accuracy": validation_accuracy,
        "walk_forward_accuracy": wf_accuracy,
        "train_rows": int(len(x_train)),
        "reason": "xgboost temporal split using iloc[-2] closed candle",
        "sentiment_features": sentiment_features or {},
        "label_type": label_type,
        **trade_diag,
        **label_params,
    }
