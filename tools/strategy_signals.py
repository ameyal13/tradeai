"""Strategy signal generation for deterministic, model_based, and hybrid modes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import AverageTrueRange, BollingerBands

from tools.ml_engine import xgboost_signal
from tools.sentiment_engine import get_fear_greed_index


STRATEGY_MODES = {"deterministic", "model_based", "hybrid", "xgboost"}


@dataclass
class StrategySignal:
    strategy_mode: str
    strategy_name: str
    strategy_version: str
    signal: str
    confidence: float
    entry_price: float
    stop_loss: float | None
    take_profit: float | None
    risk_reward_ratio: float | None
    horizon_minutes: int
    input_features: dict[str, Any]
    reasoning: str
    model_provider: str | None = None
    model_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_mode": self.strategy_mode,
            "strategy_name": self.strategy_name,
            "strategy_version": self.strategy_version,
            "signal": self.signal,
            "confidence": round(float(self.confidence), 6),
            "entry_price": round(float(self.entry_price), 8),
            "stop_loss": round(float(self.stop_loss), 8) if self.stop_loss is not None else None,
            "take_profit": round(float(self.take_profit), 8) if self.take_profit is not None else None,
            "risk_reward_ratio": round(float(self.risk_reward_ratio), 6) if self.risk_reward_ratio is not None else None,
            "horizon_minutes": self.horizon_minutes,
            "input_features": self.input_features,
            "reasoning": self.reasoning,
            "model_provider": self.model_provider,
            "model_name": self.model_name,
        }


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df.set_index("timestamp", inplace=True)
    elif "open_time" in df.columns:
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
        df.set_index("open_time", inplace=True)
    df.index = pd.to_datetime(df.index, utc=True)
    df.sort_index(inplace=True)
    for column in ["open", "high", "low", "close"]:
        df[column] = df[column].astype(float)
    if "volume" in df.columns:
        df["volume"] = df["volume"].astype(float)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_ohlcv(df)
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"] if "volume" in df else pd.Series(index=df.index, dtype=float)

    if len(df) >= 14:
        df["rsi"] = RSIIndicator(close=close, window=14).rsi()
        df["atr"] = AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()
    else:
        df["rsi"] = np.nan
        df["atr"] = np.nan

    if len(df) >= 26:
        macd = MACD(close=close)
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_hist"] = macd.macd_diff()
        df["ema_fast"] = EMAIndicator(close=close, window=12).ema_indicator()
        df["ema_slow"] = EMAIndicator(close=close, window=26).ema_indicator()
    else:
        df["macd"] = np.nan
        df["macd_signal"] = np.nan
        df["macd_hist"] = np.nan
        df["ema_fast"] = np.nan
        df["ema_slow"] = np.nan

    if len(df) >= 20:
        bb = BollingerBands(close=close, window=20, window_dev=2)
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_mid"] = bb.bollinger_mavg()
        df["volume_sma"] = volume.rolling(20).mean() if "volume" in df else np.nan
    else:
        df["bb_upper"] = np.nan
        df["bb_lower"] = np.nan
        df["bb_mid"] = np.nan
        df["volume_sma"] = np.nan

    df["return_1"] = close.pct_change()
    df["return_3"] = close.pct_change(3)
    df["volatility_10"] = df["return_1"].rolling(10).std()
    df["relative_volume"] = df["volume"] / df["volume_sma"] if "volume" in df else np.nan
    return df


def risk_levels(
    price: float,
    atr: float | None,
    signal: str,
    min_rr: float = 1.5,
    atr_stop_multiplier: float = 1.5,
    atr_take_profit_multiplier: float | None = None,
) -> tuple[float | None, float | None, float | None]:
    if signal == "HOLD" or price <= 0:
        return None, None, None
    stop_distance = atr * atr_stop_multiplier if atr is not None and not np.isnan(atr) and atr > 0 else price * 0.03
    target_distance = (atr * atr_take_profit_multiplier) if atr_take_profit_multiplier and atr is not None and not np.isnan(atr) and atr > 0 else stop_distance * min_rr
    if signal == "BUY":
        return price - stop_distance, price + target_distance, min_rr
    return price + stop_distance, price - target_distance, min_rr


def deterministic_signal_from_df(
    df: pd.DataFrame,
    horizon_minutes: int = 60,
    min_rr: float = 1.5,
    strategy_params: dict[str, Any] | None = None,
) -> StrategySignal:
    strategy_params = strategy_params or {}
    rsi_buy_threshold = float(strategy_params.get("rsi_buy_threshold", 35))
    rsi_sell_threshold = float(strategy_params.get("rsi_sell_threshold", 65))
    min_rr = float(strategy_params.get("min_risk_reward", min_rr))
    atr_stop_multiplier = float(strategy_params.get("atr_stop_multiplier", 1.5))
    atr_take_profit_multiplier = strategy_params.get("atr_take_profit_multiplier")
    atr_take_profit_multiplier = float(atr_take_profit_multiplier) if atr_take_profit_multiplier is not None else None
    features = add_features(df)
    row = features.iloc[-1]
    price = float(row["close"])
    score = 0
    reasons = []

    rsi = row.get("rsi")
    if pd.notna(rsi):
        if rsi < rsi_buy_threshold:
            score += 2
            reasons.append(f"RSI {rsi:.2f} oversold")
        elif rsi > rsi_sell_threshold:
            score -= 2
            reasons.append(f"RSI {rsi:.2f} overbought")
        else:
            reasons.append(f"RSI {rsi:.2f} neutral")

    macd_hist = row.get("macd_hist")
    if pd.notna(macd_hist):
        score += 1 if macd_hist > 0 else -1
        reasons.append("MACD histogram bullish" if macd_hist > 0 else "MACD histogram bearish")

    ema_fast = row.get("ema_fast")
    ema_slow = row.get("ema_slow")
    if pd.notna(ema_fast) and pd.notna(ema_slow):
        score += 1 if ema_fast > ema_slow else -1
        reasons.append("EMA trend filter bullish" if ema_fast > ema_slow else "EMA trend filter bearish")

    bb_lower = row.get("bb_lower")
    bb_upper = row.get("bb_upper")
    if pd.notna(bb_lower) and price < bb_lower:
        score += 1
        reasons.append("Price below lower Bollinger Band")
    elif pd.notna(bb_upper) and price > bb_upper:
        score -= 1
        reasons.append("Price above upper Bollinger Band")

    relative_volume = row.get("relative_volume")
    if pd.notna(relative_volume):
        min_volume_ratio = float(strategy_params.get("min_volume_ratio", 1.2))
        if relative_volume >= min_volume_ratio:
            score += 0.5 if score > 0 else -0.5 if score < 0 else 0
            reasons.append(f"Relative volume confirms move ({relative_volume:.2f}x)")
        else:
            reasons.append(f"Relative volume weak ({relative_volume:.2f}x)")

    signal = "BUY" if score >= 2 else "SELL" if score <= -2 else "HOLD"
    confidence = min(90, 45 + abs(score) * 10) if signal != "HOLD" else 40
    stop_loss, take_profit, rr = risk_levels(price, row.get("atr"), signal, min_rr, atr_stop_multiplier, atr_take_profit_multiplier)

    if signal != "HOLD" and (rr is None or rr < min_rr):
        signal = "HOLD"
        confidence = 35
        reasons.append("Risk/reward below minimum")

    input_features = latest_feature_snapshot(row)
    input_features["score"] = score
    return StrategySignal(
        strategy_mode="deterministic",
        strategy_name="technical_rsi_macd_ema_bb_volume_atr",
        strategy_version="v1",
        signal=signal,
        confidence=confidence,
        entry_price=price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        risk_reward_ratio=rr,
        horizon_minutes=horizon_minutes,
        input_features=input_features,
        reasoning=" | ".join(reasons) or "Insufficient feature confirmation",
    )


def latest_feature_snapshot(row: pd.Series) -> dict[str, Any]:
    keys = [
        "close", "rsi", "macd_hist", "ema_fast", "ema_slow", "bb_upper", "bb_lower",
        "atr", "relative_volume", "return_1", "return_3", "volatility_10",
    ]
    snapshot = {}
    for key in keys:
        value = row.get(key)
        snapshot[key] = None if value is None or pd.isna(value) else float(value)
    return snapshot


def sigmoid(value: float) -> float:
    return 1 / (1 + np.exp(-value))


def train_numpy_logistic_regression(x: np.ndarray, y: np.ndarray, epochs: int = 350, lr: float = 0.1) -> tuple[np.ndarray, float]:
    if len(np.unique(y)) < 2:
        raise ValueError("Need both positive and negative target classes")
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std == 0] = 1
    z = (x - mean) / std
    weights = np.zeros(z.shape[1])
    bias = 0.0
    for _ in range(epochs):
        probs = sigmoid(z @ weights + bias)
        error = probs - y
        weights -= lr * (z.T @ error / len(y))
        bias -= lr * float(error.mean())
    packed = np.concatenate([weights, mean, std])
    return packed, bias


def predict_numpy_logistic(packed: np.ndarray, bias: float, x: np.ndarray) -> float:
    n_features = len(x)
    weights = packed[:n_features]
    mean = packed[n_features:n_features * 2]
    std = packed[n_features * 2:]
    z = (x - mean) / std
    return float(sigmoid(z @ weights + bias))


def build_model_dataset(df: pd.DataFrame, horizon_candles: int = 3) -> tuple[pd.DataFrame, pd.Series]:
    features = add_features(df)
    feature_cols = ["rsi", "macd_hist", "ema_fast", "ema_slow", "relative_volume", "return_1", "return_3", "volatility_10", "atr"]
    data = features[feature_cols].copy()
    future_return = features["close"].shift(-horizon_candles) / features["close"] - 1
    target = (future_return > 0).astype(int)
    valid = data.notna().all(axis=1) & future_return.notna()
    return data[valid], target[valid]


def model_based_signal_from_df(
    df: pd.DataFrame,
    horizon_minutes: int = 60,
    min_train_rows: int = 40,
    strategy_params: dict[str, Any] | None = None,
) -> StrategySignal:
    strategy_params = strategy_params or {}
    features = add_features(df)
    latest_row = features.iloc[-1]
    price = float(latest_row["close"])

    x, y = build_model_dataset(features)
    feature_cols = ["rsi", "macd_hist", "ema_fast", "ema_slow", "relative_volume", "return_1", "return_3", "volatility_10", "atr"]
    latest_features = latest_row[feature_cols]
    if latest_features.isna().any():
        return StrategySignal(
            strategy_mode="model_based",
            strategy_name="numpy_logistic_temporal_split",
            strategy_version="interface_v1",
            signal="HOLD",
            confidence=25,
            entry_price=price,
            stop_loss=None,
            take_profit=None,
            risk_reward_ratio=None,
            horizon_minutes=horizon_minutes,
            input_features={"model_available": False, "reason": "latest_features_incomplete"},
            reasoning="Model interface ready, but latest closed candle lacks complete features.",
            model_provider="local_numpy",
            model_name="logistic_regression_no_sklearn",
        )

    if len(x) < min_train_rows:
        return StrategySignal(
            strategy_mode="model_based",
            strategy_name="numpy_logistic_temporal_split",
            strategy_version="interface_v1",
            signal="HOLD",
            confidence=30,
            entry_price=price,
            stop_loss=None,
            take_profit=None,
            risk_reward_ratio=None,
            horizon_minutes=horizon_minutes,
            input_features={"model_available": False, "reason": "insufficient_training_rows", "rows": len(x)},
            reasoning="Model interface ready, but not enough historical rows for temporal split training.",
            model_provider="local_numpy",
            model_name="logistic_regression_no_sklearn",
        )

    split = int(len(x) * 0.7)
    train_x = x.iloc[:split].to_numpy(dtype=float)
    train_y = y.iloc[:split].to_numpy(dtype=float)
    valid_x = x.iloc[split:-1].to_numpy(dtype=float)
    valid_y = y.iloc[split:-1].to_numpy(dtype=float)
    latest_x = latest_features.to_numpy(dtype=float)

    try:
        packed, bias = train_numpy_logistic_regression(train_x, train_y)
        probability = predict_numpy_logistic(packed, bias, latest_x)
    except ValueError as exc:
        return StrategySignal(
            strategy_mode="model_based",
            strategy_name="numpy_logistic_temporal_split",
            strategy_version="interface_v1",
            signal="HOLD",
            confidence=30,
            entry_price=price,
            stop_loss=None,
            take_profit=None,
            risk_reward_ratio=None,
            horizon_minutes=horizon_minutes,
            input_features={"model_available": False, "reason": str(exc)},
            reasoning="Model interface ready, but training target is not usable yet.",
            model_provider="local_numpy",
            model_name="logistic_regression_no_sklearn",
        )

    validation_accuracy = None
    if len(valid_x):
        preds = np.array([predict_numpy_logistic(packed, bias, row) >= 0.5 for row in valid_x])
        validation_accuracy = float((preds == valid_y.astype(bool)).mean())

    buy_threshold = float(strategy_params.get("probability_buy_threshold", 0.58))
    sell_threshold = float(strategy_params.get("probability_sell_threshold", 0.42))
    min_rr = float(strategy_params.get("min_risk_reward", 1.5))
    atr_stop_multiplier = float(strategy_params.get("atr_stop_multiplier", 1.5))
    atr_take_profit_multiplier = strategy_params.get("atr_take_profit_multiplier")
    atr_take_profit_multiplier = float(atr_take_profit_multiplier) if atr_take_profit_multiplier is not None else None
    if probability >= buy_threshold:
        signal = "BUY"
        sl, tp, rr = risk_levels(price, latest_row.get("atr"), "BUY", min_rr, atr_stop_multiplier, atr_take_profit_multiplier)
    elif probability <= sell_threshold:
        signal = "SELL"
        sl, tp, rr = risk_levels(price, latest_row.get("atr"), "SELL", min_rr, atr_stop_multiplier, atr_take_profit_multiplier)
    else:
        signal = "HOLD"
        sl, tp, rr = None, None, None

    confidence = 45 + abs(probability - 0.5) * 100
    if validation_accuracy is not None:
        confidence = (confidence + validation_accuracy * 100) / 2

    return StrategySignal(
        strategy_mode="model_based",
        strategy_name="numpy_logistic_temporal_split",
        strategy_version="interface_v1",
        signal=signal,
        confidence=min(85, max(25, confidence)),
        entry_price=price,
        stop_loss=sl,
        take_profit=tp,
        risk_reward_ratio=rr,
        horizon_minutes=horizon_minutes,
        input_features={
            "model_available": True,
            "probability_up": probability,
            "temporal_split": {"train_rows": len(train_x), "validation_rows": len(valid_x)},
            "prediction_row": "latest_closed_candle",
            "validation_accuracy": validation_accuracy,
            "latest_features": latest_feature_snapshot(latest_row),
            "dependency_note": "scikit-learn not required; numpy fallback model used",
        },
        reasoning=f"Temporal split numpy logistic model probability_up={probability:.3f}.",
        model_provider="local_numpy",
        model_name="logistic_regression_no_sklearn",
    )


def hybrid_signal_from_df(
    df: pd.DataFrame,
    provider: str = "none",
    horizon_minutes: int = 60,
    strategy_params: dict[str, Any] | None = None,
) -> StrategySignal:
    deterministic = deterministic_signal_from_df(df, horizon_minutes=horizon_minutes, strategy_params=strategy_params)
    model = model_based_signal_from_df(df, horizon_minutes=horizon_minutes, strategy_params=strategy_params)
    signal = deterministic.signal
    confidence = deterministic.confidence
    reasons = [deterministic.reasoning]
    adjustments = []

    probability = model.input_features.get("probability_up") if model.input_features.get("model_available") else None
    if probability is not None and signal in {"BUY", "SELL"}:
        supports_base = (signal == "BUY" and probability >= 0.5) or (signal == "SELL" and probability <= 0.5)
        delta = min(10, abs(probability - 0.5) * 40)
        confidence += delta if supports_base else -delta
        adjustments.append({
            "source": "model_based",
            "probability_up": probability,
            "supports_base_signal": supports_base,
            "confidence_delta": delta if supports_base else -delta,
        })
    elif model.signal != "HOLD" and signal == "HOLD":
        confidence = max(confidence, min(55, model.confidence))
        adjustments.append({"source": "model_based", "note": "model signal noted but HOLD preserved without deterministic confirmation"})

    llm_delta = 0
    if provider and provider != "none":
        llm_delta = 3
        confidence += llm_delta
        adjustments.append({
            "source": "llm_context",
            "provider": provider,
            "confidence_delta": llm_delta,
            "note": "Context adjustment bounded; LLM did not change signal direction.",
        })

    confidence = min(90, max(20, confidence))
    features = {
        "base_deterministic": deterministic.input_features,
        "model_based": model.input_features,
        "adjustments": adjustments,
        "llm_can_change_direction": False,
    }
    reasons.append("Hybrid kept deterministic direction; model/LLM only adjusted confidence within bounds.")
    return StrategySignal(
        strategy_mode="hybrid",
        strategy_name="deterministic_plus_model_context",
        strategy_version="v1",
        signal=signal,
        confidence=confidence,
        entry_price=deterministic.entry_price,
        stop_loss=deterministic.stop_loss,
        take_profit=deterministic.take_profit,
        risk_reward_ratio=deterministic.risk_reward_ratio,
        horizon_minutes=horizon_minutes,
        input_features=features,
        reasoning=" | ".join(reasons),
        model_provider=provider if provider != "none" else model.model_provider,
        model_name=model.model_name,
    )


def xgboost_signal_from_df(
    df: pd.DataFrame,
    horizon_minutes: int = 60,
    strategy_params: dict[str, Any] | None = None,
    use_sentiment: bool = True,
) -> StrategySignal:
    try:
        features = add_features(df)
        fear_greed = get_fear_greed_index() if use_sentiment else None
        result = xgboost_signal(
            features,
            horizon_minutes=horizon_minutes,
            strategy_params=strategy_params,
            sentiment_features=fear_greed,
        )
        if fear_greed:
            result["reason"] = f"{result['reason']} | Fear & Greed: {fear_greed['classification']}"
    except ImportError as exc:
        import logging

        logging.getLogger(__name__).warning("XGBoost unavailable; falling back to model_based mode: %s", exc)
        return model_based_signal_from_df(df, horizon_minutes=horizon_minutes, strategy_params=strategy_params)

    row = features.iloc[-2] if len(features) >= 2 else features.iloc[-1]
    price = float(row["close"])
    min_rr = float((strategy_params or {}).get("min_risk_reward", 1.5))
    atr_stop_multiplier = float((strategy_params or {}).get("atr_stop_multiplier", 1.5))
    atr_take_profit_multiplier = (strategy_params or {}).get("atr_take_profit_multiplier")
    atr_take_profit_multiplier = float(atr_take_profit_multiplier) if atr_take_profit_multiplier is not None else None
    stop_loss, take_profit, rr = risk_levels(
        price,
        row.get("atr"),
        result["signal"],
        min_rr,
        atr_stop_multiplier,
        atr_take_profit_multiplier,
    )
    input_features = latest_feature_snapshot(row)
    input_features.update(result)
    input_features["prediction_row"] = "iloc[-2]"

    return StrategySignal(
        strategy_mode="xgboost",
        strategy_name="xgboost_temporal_split",
        strategy_version="v1",
        signal=result["signal"],
        confidence=result["confidence"],
        entry_price=price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        risk_reward_ratio=rr,
        horizon_minutes=horizon_minutes,
        input_features=input_features,
        reasoning=result["reason"],
        model_provider="local_xgboost",
        model_name="xgboost_classifier_v1",
    )


def generate_strategy_signal_from_df(
    df: pd.DataFrame,
    strategy_mode: str = "deterministic",
    provider: str = "none",
    horizon_minutes: int = 60,
    strategy_params: dict[str, Any] | None = None,
) -> StrategySignal:
    mode = strategy_mode.lower()
    if mode not in STRATEGY_MODES:
        raise ValueError("strategy_mode must be deterministic, model_based, hybrid, or xgboost")
    if mode == "deterministic":
        return deterministic_signal_from_df(df, horizon_minutes=horizon_minutes, strategy_params=strategy_params)
    if mode == "model_based":
        return model_based_signal_from_df(df, horizon_minutes=horizon_minutes, strategy_params=strategy_params)
    if mode == "xgboost":
        use_sentiment = bool((strategy_params or {}).get("use_sentiment", True))
        return xgboost_signal_from_df(
            df,
            horizon_minutes=horizon_minutes,
            strategy_params=strategy_params,
            use_sentiment=use_sentiment,
        )
    return hybrid_signal_from_df(df, provider=provider, horizon_minutes=horizon_minutes, strategy_params=strategy_params)
