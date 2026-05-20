"""
ML Engine para TradeAI.
Reemplaza la regresión logística numpy con XGBoost + walk-forward validation.
Compatible con la interfaz existente de strategy_signals.py.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


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
) -> tuple[pd.DataFrame, pd.Series]:
    data = _feature_frame(df_features)
    labels = build_labels(data, horizon_candles=horizon_candles, threshold_pct=threshold_pct)
    valid = data[FEATURE_COLS].notna().all(axis=1) & labels.notna()
    return data.loc[valid, FEATURE_COLS], labels.loc[valid].astype(int)


def walk_forward_accuracy(
    df_features: pd.DataFrame,
    n_splits: int = 5,
    horizon_candles: int = 1,
) -> float | None:
    """
    Calcula accuracy promedio con walk-forward cross-validation.
    Divide el dataset en n_splits folds temporales.
    En cada fold: entrena en el pasado, valida en el futuro inmediato.
    Retorna accuracy promedio (float 0-1) o None si hay datos insuficientes.
    """
    x, y = _prepared_dataset(df_features, horizon_candles=horizon_candles)
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

    data = _feature_frame(df)
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
        }

    x, y = _prepared_dataset(data.iloc[:-1], horizon_candles=horizon_candles, threshold_pct=threshold_pct)
    prediction_row = data.iloc[-2][FEATURE_COLS]
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
    if len(x_val):
        probs = np.array([predict_proba_xgboost(model, row) for row in x_val])
        preds = (probs >= 0.5).astype(int)
        validation_accuracy = round(float((preds == y_val).mean()), 6)
        if len(np.unique(y)) < 2:
            validation_accuracy = 0.5

    probability = predict_proba_xgboost(model, prediction_row.to_numpy(dtype=float))
    if probability >= buy_threshold:
        signal = "BUY"
    elif probability <= sell_threshold:
        signal = "SELL"
    else:
        signal = "HOLD"

    confidence = min(95.0, max(5.0, abs(probability - 0.5) * 200))
    wf_accuracy = walk_forward_accuracy(data.iloc[:-1], horizon_candles=horizon_candles)
    return {
        "model_available": True,
        "probability_up": round(float(probability), 6),
        "signal": signal,
        "confidence": round(float(confidence), 6),
        "validation_accuracy": validation_accuracy,
        "walk_forward_accuracy": wf_accuracy,
        "train_rows": int(len(x_train)),
        "reason": "xgboost temporal split using iloc[-2] closed candle",
    }
