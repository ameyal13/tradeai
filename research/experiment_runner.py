"""Controlled experiment runner for the local Research Autopilot."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from scripts.run_historical_experiments import load_experiment_candles
from tools.feature_research import MARKET_CONTEXT_FEATURE_COLS, add_research_features, build_directional_net_returns
from tools.ml_engine import FEATURE_COLS, _atr_label_pcts, build_trade_outcome_labels, predict_proba_xgboost, train_xgboost_model
from tools.strategy_signals import generate_strategy_signal_from_df
from tools.trade_opportunity_research import DEFAULT_COST_PROFILES, oracle_top_k_returns, profit_metrics, random_same_count_returns


def max_drawdown_pct(returns: list[float]) -> float:
    equity = 100.0
    peak = equity
    max_drawdown = 0.0
    for ret in returns:
        equity *= 1 + ret / 100
        peak = max(peak, equity)
        if peak > 0:
            max_drawdown = max(max_drawdown, (peak - equity) / peak * 100)
    return round(max_drawdown, 6)


def metrics_from_returns(returns: list[float]) -> dict[str, Any]:
    metrics = profit_metrics(returns)
    return {
        "n_trades": metrics["trades"],
        "win_rate": metrics["win_rate"],
        "avg_return_pct": metrics["average_return"],
        "profit_factor": metrics["profit_factor"],
        "max_drawdown_pct": max_drawdown_pct(returns),
        "total_return_pct": metrics["total_return_pct"],
    }


def purged_train_validation_test_split(
    valid_positions: np.ndarray,
    horizon_candles: int,
    train_ratio: float = 0.6,
    validation_ratio: float = 0.2,
    min_train_rows: int = 120,
) -> dict[str, np.ndarray]:
    """Temporal train/validation/test split with purge before validation/test."""
    positions = np.asarray(sorted(set(int(pos) for pos in valid_positions)), dtype=int)
    if len(positions) < min_train_rows + horizon_candles * 2 + 20:
        return {"train": np.array([], dtype=int), "validation": np.array([], dtype=int), "test": np.array([], dtype=int)}

    train_end = max(min_train_rows, int(len(positions) * train_ratio))
    validation_end = max(train_end + 1, int(len(positions) * (train_ratio + validation_ratio)))
    train = positions[:train_end]
    validation = positions[train_end:validation_end]
    test = positions[validation_end:]

    if len(validation):
        validation = validation[validation > int(train[-1]) + horizon_candles]
    if len(test) and len(validation):
        test = test[test > int(validation[-1]) + horizon_candles]
    if len(test) and not len(validation):
        test = test[test > int(train[-1]) + horizon_candles]
    return {"train": train, "validation": validation, "test": test}


def _valid_positions(features: pd.DataFrame, labels: pd.DataFrame, returns: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    valid = (
        features[feature_cols].notna().all(axis=1)
        & labels["buy_win"].notna()
        & labels["sell_win"].notna()
        & returns["buy_return_pct"].notna()
        & returns["sell_return_pct"].notna()
    )
    return np.flatnonzero(valid.to_numpy())


def _fit_direction_model(features: pd.DataFrame, labels: pd.Series, train_positions: np.ndarray, feature_cols: list[str]):
    x_train = features.iloc[train_positions][feature_cols].to_numpy(dtype=float)
    y_train = labels.iloc[train_positions].astype(int).to_numpy()
    return train_xgboost_model(x_train, y_train)


def _evaluate_model(
    buy_model: Any,
    sell_model: Any,
    features: pd.DataFrame,
    returns: pd.DataFrame,
    positions: np.ndarray,
    feature_cols: list[str],
    buy_threshold: float,
    sell_threshold: float,
) -> dict[str, Any]:
    trade_returns: list[float] = []
    buy_trades = 0
    sell_trades = 0
    hold_count = 0
    if len(positions) == 0:
        return {"positions": 0, "buy_trades": 0, "sell_trades": 0, "hold_count": 0, **metrics_from_returns([])}

    x = features.iloc[positions][feature_cols].to_numpy(dtype=float)
    buy_probs = np.array([predict_proba_xgboost(buy_model, row) for row in x])
    sell_probs = np.array([predict_proba_xgboost(sell_model, row) for row in x])
    buy_returns = returns.iloc[positions]["buy_return_pct"].to_numpy(dtype=float)
    sell_returns = returns.iloc[positions]["sell_return_pct"].to_numpy(dtype=float)
    for idx, buy_prob in enumerate(buy_probs):
        sell_prob = float(sell_probs[idx])
        buy_edge = float(buy_prob) - buy_threshold
        sell_edge = sell_prob - sell_threshold
        if buy_prob >= buy_threshold and buy_edge >= sell_edge:
            buy_trades += 1
            trade_returns.append(float(buy_returns[idx]))
        elif sell_prob >= sell_threshold and sell_edge > buy_edge:
            sell_trades += 1
            trade_returns.append(float(sell_returns[idx]))
        else:
            hold_count += 1
    return {
        "positions": int(len(positions)),
        "buy_trades": buy_trades,
        "sell_trades": sell_trades,
        "hold_count": hold_count,
        **metrics_from_returns(trade_returns),
    }


def _always_baseline(returns: pd.DataFrame, positions: np.ndarray, side: str) -> dict[str, Any]:
    column = "buy_return_pct" if side == "BUY" else "sell_return_pct"
    values = [float(value) for value in returns.iloc[positions][column].dropna().to_list()]
    return metrics_from_returns(values)


def _oracle_baseline(returns: pd.DataFrame, positions: np.ndarray, trade_count: int) -> dict[str, Any]:
    buy = returns.iloc[positions]["buy_return_pct"].to_numpy(dtype=float)
    sell = returns.iloc[positions]["sell_return_pct"].to_numpy(dtype=float)
    return metrics_from_returns(oracle_top_k_returns(buy, sell, top_k=trade_count))


def _random_baseline(returns: pd.DataFrame, positions: np.ndarray, trade_count: int) -> dict[str, Any]:
    buy = returns.iloc[positions]["buy_return_pct"].to_numpy(dtype=float)
    sell = returns.iloc[positions]["sell_return_pct"].to_numpy(dtype=float)
    raw = random_same_count_returns(buy, sell, trade_count=trade_count)
    return {
        "n_trades": raw["trades"],
        "win_rate": raw["win_rate"],
        "avg_return_pct": raw["average_return"],
        "profit_factor": raw["profit_factor"],
        "max_drawdown_pct": None,
        "total_return_pct": raw["total_return_pct"],
    }


def _deterministic_baseline(
    candles: pd.DataFrame,
    returns: pd.DataFrame,
    positions: np.ndarray,
    horizon_candles: int,
    risk_reward: float,
    atr_stop_multiplier: float,
) -> dict[str, Any]:
    values: list[float] = []
    buy_trades = 0
    sell_trades = 0
    for pos in positions:
        history = candles.iloc[: int(pos) + 1]
        signal = generate_strategy_signal_from_df(
            history,
            strategy_mode="deterministic",
            horizon_minutes=horizon_candles * 60,
            strategy_params={"min_risk_reward": risk_reward, "atr_stop_multiplier": atr_stop_multiplier},
        ).to_dict()
        if signal["signal"] == "BUY":
            buy_trades += 1
            values.append(float(returns.iloc[int(pos)]["buy_return_pct"]))
        elif signal["signal"] == "SELL":
            sell_trades += 1
            values.append(float(returns.iloc[int(pos)]["sell_return_pct"]))
    return {
        "buy_trades": buy_trades,
        "sell_trades": sell_trades,
        **metrics_from_returns(values),
    }


def _baselines(
    candles: pd.DataFrame,
    returns: pd.DataFrame,
    positions: np.ndarray,
    model_trade_count: int,
    horizon_candles: int,
    risk_reward: float,
    atr_stop_multiplier: float,
) -> dict[str, Any]:
    return {
        "random_same_count": _random_baseline(returns, positions, model_trade_count),
        "always_buy": _always_baseline(returns, positions, "BUY"),
        "always_sell": _always_baseline(returns, positions, "SELL"),
        "oracle_top_k": _oracle_baseline(returns, positions, model_trade_count),
        "deterministic": _deterministic_baseline(
            candles,
            returns,
            positions,
            horizon_candles=horizon_candles,
            risk_reward=risk_reward,
            atr_stop_multiplier=atr_stop_multiplier,
        ),
    }


def diagnostic_flags(validation_metrics: dict[str, Any], test_metrics: dict[str, Any], baselines: dict[str, Any]) -> dict[str, bool]:
    random_avg = baselines["validation"]["random_same_count"]["avg_return_pct"]
    deterministic_avg = baselines["validation"]["deterministic"]["avg_return_pct"]
    return {
        "beats_random_validation": validation_metrics["avg_return_pct"] > random_avg,
        "beats_deterministic_validation": validation_metrics["avg_return_pct"] > deterministic_avg,
        "validation_positive": validation_metrics["avg_return_pct"] > 0 and validation_metrics["profit_factor"] > 1.0,
        "test_confirms": (
            test_metrics["profit_factor"] >= 1.0
            and test_metrics["avg_return_pct"] > 0
            and test_metrics["max_drawdown_pct"] < 20
        ),
        "high_drawdown_flag": validation_metrics["max_drawdown_pct"] >= 15,
    }


def directional_exposure(metrics: dict[str, Any]) -> dict[str, Any]:
    buy_trades = int(metrics.get("buy_trades") or 0)
    sell_trades = int(metrics.get("sell_trades") or 0)
    total = buy_trades + sell_trades
    buy_ratio = round(buy_trades / total, 6) if total else 0
    sell_ratio = round(sell_trades / total, 6) if total else 0
    if total == 0:
        bias = "no_trades"
    elif buy_ratio >= 0.85:
        bias = "buy_heavy"
    elif sell_ratio >= 0.85:
        bias = "sell_heavy"
    else:
        bias = "balanced"
    return {
        "buy_trades": buy_trades,
        "sell_trades": sell_trades,
        "buy_ratio": buy_ratio,
        "sell_ratio": sell_ratio,
        "directional_bias": bias,
    }


def classify_result(validation_metrics: dict[str, Any], test_metrics: dict[str, Any], baselines: dict[str, Any]) -> tuple[str, list[str]]:
    reasons: list[str] = []
    flags = diagnostic_flags(validation_metrics, test_metrics, baselines)
    validation_passes_random = flags["beats_random_validation"]
    validation_passes_deterministic = flags["beats_deterministic_validation"]

    if validation_metrics["profit_factor"] < 1.0:
        reasons.append("validation_profit_factor_below_1")
    if validation_metrics["avg_return_pct"] <= 0:
        reasons.append("validation_average_return_nonpositive")
    if not validation_passes_random:
        reasons.append("validation_does_not_beat_random_same_count")
    if reasons:
        if test_metrics["avg_return_pct"] > 0 and test_metrics["profit_factor"] > 1.0:
            reasons.append("test_positive_but_validation_failed_do_not_select")
        if validation_metrics["avg_return_pct"] <= 0 and validation_metrics["profit_factor"] < 1.0:
            return "hard_reject", reasons
        return "reject", reasons

    if flags["validation_positive"] and flags["high_drawdown_flag"]:
        return "research_watchlist", ["validation_positive_but_high_drawdown"]

    validation_candidate = (
        validation_metrics["profit_factor"] > 1.1
        and validation_metrics["avg_return_pct"] > 0
        and validation_metrics["max_drawdown_pct"] < 15
        and validation_passes_random
        and validation_passes_deterministic
    )
    weak_candidate = (
        1.0 <= validation_metrics["profit_factor"] <= 1.1
        and validation_metrics["avg_return_pct"] > 0
        and validation_passes_random
        and not validation_passes_deterministic
    )
    if validation_candidate:
        if not flags["test_confirms"]:
            return "validation_candidate_test_failed", ["test_holdout_failed_confirmation"]
        return "candidate_for_further_validation", ["validation_and_test_confirmed"]
    if weak_candidate:
        return "weak_candidate", ["validation_weak_candidate"]
    if flags["validation_positive"]:
        return "research_watchlist", ["validation_positive_but_does_not_match_candidate_rules"]
    return "reject", ["validation_positive_but_does_not_match_candidate_rules"]


def run_experiment_on_candles(
    config: dict[str, Any],
    candles: pd.DataFrame,
    data_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one controlled experiment on a preselected candle window.

    This keeps train/validation/test ownership with callers that need exact
    temporal windows, while preserving the same modeling path used by
    ``run_experiment``.
    """
    cost_mode = config["cost_mode"]
    if cost_mode not in DEFAULT_COST_PROFILES:
        raise ValueError(f"Unknown cost_mode: {cost_mode}")
    costs = DEFAULT_COST_PROFILES[cost_mode]
    data_metadata = data_metadata or {}
    use_market_context_features = bool(config.get("use_market_context_features", False))
    features = add_research_features(candles, include_market_context=use_market_context_features)
    feature_cols = list(FEATURE_COLS)
    feature_family = "current_xgboost_features"
    if use_market_context_features:
        feature_cols = feature_cols + MARKET_CONTEXT_FEATURE_COLS
        feature_family = "current_plus_market_context_v1"
    stop_pcts, take_pcts = _atr_label_pcts(
        features,
        atr_stop_multiplier=float(config["atr_stop_multiplier"]),
        min_rr=float(config["risk_reward"]),
        atr_take_profit_multiplier=None,
        fallback_stop_loss_pct=0.03,
        fallback_take_profit_pct=0.03 * float(config["risk_reward"]),
    )
    labels = build_trade_outcome_labels(
        features,
        horizon_candles=int(config["horizon_candles"]),
        stop_loss_pct=0.03,
        take_profit_pct=0.03 * float(config["risk_reward"]),
        commission_pct=float(costs["commission_pct"]),
        slippage_pct=float(costs["slippage_pct"]),
        spread_pct=float(costs["spread_pct"]),
        stop_loss_pcts=stop_pcts,
        take_profit_pcts=take_pcts,
        label_scheme=str(config.get("trade_label_scheme", "expected_value_classification")),
    )
    returns = build_directional_net_returns(
        features,
        horizon_candles=int(config["horizon_candles"]),
        stop_loss_pct=0.03,
        take_profit_pct=0.03 * float(config["risk_reward"]),
        commission_pct=float(costs["commission_pct"]),
        slippage_pct=float(costs["slippage_pct"]),
        spread_pct=float(costs["spread_pct"]),
        stop_loss_pcts=stop_pcts,
        take_profit_pcts=take_pcts,
    )
    valid_positions = _valid_positions(features, labels, returns, feature_cols)
    splits = purged_train_validation_test_split(
        valid_positions,
        horizon_candles=int(config["horizon_candles"]),
        min_train_rows=int(config.get("min_train_rows", 120)),
    )
    if len(splits["train"]) == 0 or len(splits["validation"]) == 0 or len(splits["test"]) == 0:
        validation_metrics = metrics_from_returns([])
        test_metrics = metrics_from_returns([])
        baselines = {"validation": {}, "test": {}}
        classification = "reject"
        reasons = ["insufficient_data_for_train_validation_test_split"]
    else:
        buy_model = _fit_direction_model(features, labels["buy_win"], splits["train"], feature_cols)
        sell_model = _fit_direction_model(features, labels["sell_win"], splits["train"], feature_cols)
        validation_metrics = _evaluate_model(
            buy_model,
            sell_model,
            features,
            returns,
            splits["validation"],
            feature_cols,
            buy_threshold=float(config.get("buy_threshold", 0.58)),
            sell_threshold=float(config.get("sell_threshold", 0.58)),
        )
        test_metrics = _evaluate_model(
            buy_model,
            sell_model,
            features,
            returns,
            splits["test"],
            feature_cols,
            buy_threshold=float(config.get("buy_threshold", 0.58)),
            sell_threshold=float(config.get("sell_threshold", 0.58)),
        )
        baselines = {
            "validation": _baselines(
                candles,
                returns,
                splits["validation"],
                model_trade_count=int(validation_metrics["n_trades"]),
                horizon_candles=int(config["horizon_candles"]),
                risk_reward=float(config["risk_reward"]),
                atr_stop_multiplier=float(config["atr_stop_multiplier"]),
            ),
            "test": _baselines(
                candles,
                returns,
                splits["test"],
                model_trade_count=int(test_metrics["n_trades"]),
                horizon_candles=int(config["horizon_candles"]),
                risk_reward=float(config["risk_reward"]),
                atr_stop_multiplier=float(config["atr_stop_multiplier"]),
            ),
        }
        classification, reasons = classify_result(validation_metrics, test_metrics, baselines)
        flags = diagnostic_flags(validation_metrics, test_metrics, baselines)

    return {
        "experiment_id": config["experiment_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "data_source": data_metadata.get("data_source"),
        "data_cache_path": data_metadata.get("data_cache_path"),
        "data_warning": data_metadata.get("data_warning"),
        "split": {
            "method": "temporal_train_validation_test_with_purge",
            "purge_candles": int(config["horizon_candles"]),
            "train_rows": int(len(splits["train"])),
            "validation_rows": int(len(splits["validation"])),
            "test_rows": int(len(splits["test"])),
            "train_end": int(splits["train"][-1]) if len(splits["train"]) else None,
            "validation_start": int(splits["validation"][0]) if len(splits["validation"]) else None,
            "validation_end": int(splits["validation"][-1]) if len(splits["validation"]) else None,
            "test_start": int(splits["test"][0]) if len(splits["test"]) else None,
            "test_is_holdout_only": True,
        },
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "baselines": baselines,
        "classification": classification,
        "reasons": reasons,
        "diagnostics": {
            **(flags if "flags" in locals() else {
                "beats_random_validation": False,
                "beats_deterministic_validation": False,
                "validation_positive": False,
                "test_confirms": False,
                "high_drawdown_flag": False,
            }),
            "validation_directional_exposure": directional_exposure(validation_metrics),
            "test_directional_exposure": directional_exposure(test_metrics),
            "feature_family": feature_family,
            "feature_cols": feature_cols,
            "use_market_context_features": use_market_context_features,
        },
        "guardrails": {
            "no_trading": True,
            "no_prediction_journal_writes": True,
            "accuracy_not_used": True,
            "test_not_used_for_selection": True,
        },
    }


async def run_experiment(config: dict[str, Any]) -> dict[str, Any]:
    """Run one controlled experiment. No operational signal is generated."""
    loaded = await load_experiment_candles(
        config["symbol"],
        config["timeframe"],
        max_candles=int(config.get("max_candles", 1500)),
        use_cache=True,
        refresh_cache=False,
    )
    return run_experiment_on_candles(config, loaded["candles"], loaded)
