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
) -> pd.DataFrame:
    """Fast directional trade labels equivalent to TP/SL touch outcomes.

    Costs stay in the signature for experiment metadata parity, but this
    primary directional label is based on TP/SL touches. EXPIRED, BREAKEVEN,
    and INVALID_DATA remain NaN.
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
    buy_labels = np.full(len(data), np.nan, dtype=float)
    sell_labels = np.full(len(data), np.nan, dtype=float)

    for index_n in range(len(data)):
        entry_idx = index_n + 1
        if entry_idx >= len(data):
            continue
        end_idx = min(len(data), entry_idx + horizon_candles)
        if end_idx <= entry_idx:
            continue

        entry = open_prices[entry_idx]
        buy_sl = entry * (1 - stop_loss_pct)
        buy_tp = entry * (1 + take_profit_pct)
        sell_sl = entry * (1 + stop_loss_pct)
        sell_tp = entry * (1 - take_profit_pct)

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
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    data = _feature_frame(df_features)
    valid = (
        data[feature_cols].notna().all(axis=1)
        & trade_labels["buy_win"].notna()
        & trade_labels["sell_win"].notna()
    )
    return (
        data.loc[valid, feature_cols],
        trade_labels.loc[valid, "buy_win"].astype(int),
        trade_labels.loc[valid, "sell_win"].astype(int),
    )


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
    threshold_pct = float(strategy_params.get("label_threshold_pct", 0.003))
    horizon_candles = int(strategy_params.get("horizon_candles", 1))
    buy_threshold = float(strategy_params.get("probability_buy_threshold", 0.58))
    sell_threshold = float(strategy_params.get("probability_sell_threshold", 0.42))
    use_trade_labels = bool(strategy_params.get("use_trade_labels", False))
    label_type = "trade_outcome_directional" if use_trade_labels else "price_return"
    label_stop_loss_pct = float(strategy_params.get("stop_loss_pct", 0.03))
    label_take_profit_pct = float(strategy_params.get("take_profit_pct", 0.045))
    label_commission_pct = float(strategy_params.get("commission_pct", 0.001))
    label_slippage_pct = float(strategy_params.get("slippage_pct", 0.0005))
    label_spread_pct = float(strategy_params.get("spread_pct", 0.0003))
    label_params = {
        "label_type": label_type,
        "label_stop_loss_pct": label_stop_loss_pct if use_trade_labels else None,
        "label_take_profit_pct": label_take_profit_pct if use_trade_labels else None,
        "label_horizon_candles": horizon_candles if use_trade_labels else None,
        "label_costs": {
            "commission_pct": label_commission_pct,
            "slippage_pct": label_slippage_pct,
            "spread_pct": label_spread_pct,
        } if use_trade_labels else None,
        "label_level_note": "fixed_pct_trade_labels_not_atr; costs_recorded_but_primary_label_is_tp_sl_touch" if use_trade_labels else None,
    }

    data = _apply_sentiment_features(_feature_frame(df), sentiment_features)
    feature_cols = _feature_cols(sentiment_features)
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
            **label_params,
        }

    if use_trade_labels:
        trade_labels = build_trade_outcome_labels(
            data.iloc[:-1],
            horizon_candles=horizon_candles,
            stop_loss_pct=label_stop_loss_pct,
            take_profit_pct=label_take_profit_pct,
            commission_pct=label_commission_pct,
            slippage_pct=label_slippage_pct,
            spread_pct=label_spread_pct,
        )
        x, buy_y, sell_y = _prepared_trade_dataset(data.iloc[:-1], trade_labels, feature_cols)
        y = buy_y
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
            **label_params,
        }

    split = max(1, int(len(x) * 0.7))
    if split >= len(x):
        split = len(x) - 1
    x_train = x.iloc[:split].to_numpy(dtype=float)
    y_train = y.iloc[:split].to_numpy(dtype=int)
    x_val = x.iloc[split:].to_numpy(dtype=float)
    y_val = y.iloc[split:].to_numpy(dtype=int)
    model = train_xgboost_model(x_train, y_train, x_val, y_val)

    validation_accuracy = None
    probability_sell_win = None
    if use_trade_labels:
        sell_train_y = sell_y.iloc[:split].to_numpy(dtype=int)
        sell_val_y = sell_y.iloc[split:].to_numpy(dtype=int)
        sell_model = train_xgboost_model(x_train, sell_train_y, x_val, sell_val_y)
        if len(x_val):
            buy_probs = np.array([predict_proba_xgboost(model, row) for row in x_val])
            sell_probs = np.array([predict_proba_xgboost(sell_model, row) for row in x_val])
            buy_preds = (buy_probs >= buy_threshold).astype(int)
            sell_preds = (sell_probs >= (1 - sell_threshold)).astype(int)
            validation_accuracy = round(float(((buy_preds == y_val) & (sell_preds == sell_val_y)).mean()), 6)
        probability = predict_proba_xgboost(model, prediction_row.to_numpy(dtype=float))
        probability_sell_win = predict_proba_xgboost(sell_model, prediction_row.to_numpy(dtype=float))
        if probability >= buy_threshold and probability >= probability_sell_win:
            signal = "BUY"
            confidence = min(95.0, max(5.0, probability * 100))
        elif probability_sell_win >= (1 - sell_threshold) and probability_sell_win > probability:
            signal = "SELL"
            confidence = min(95.0, max(5.0, probability_sell_win * 100))
        else:
            signal = "HOLD"
            confidence = min(95.0, max(5.0, max(probability, probability_sell_win) * 100))
    elif len(x_val):
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
        **label_params,
    }
