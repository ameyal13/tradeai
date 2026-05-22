"""Run historical replay experiments across symbols, timeframes, and strategies.

This script is for offline/historical research. It does not place trades and
does not persist to Supabase unless --persist is explicitly passed.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.historical_data import fetch_binance_klines
from tools.historical_replay import run_historical_replay
from tools.prediction_journal import PredictionStore, calculate_profit_factor


DEFAULT_SYMBOLS = ["BTC", "ETH", "SOL"]
DEFAULT_TIMEFRAMES = ["15m", "1h"]
DEFAULT_STRATEGY_MODES = ["deterministic"]
ALLOWED_STRATEGY_MODES = {"deterministic", "model_based", "hybrid", "xgboost"}
DEFAULT_HORIZON_MINUTES = 60
DEFAULT_MAX_CANDLES = 500
DEFAULT_MAX_PREDICTIONS = 100
DEFAULT_TRADE_LABEL_MIN_TRAIN_ROWS = 150
DEFAULT_TRADE_LABEL_SCHEME = "touch_only"
ALLOWED_TRADE_LABEL_SCHEMES = {"touch_only", "hybrid_touch_or_expiry"}
HISTORICAL_SENTIMENT_USED = False
HISTORICAL_SENTIMENT_NOTE = "Fear & Greed current value disabled for historical replay to avoid time leakage."
MIN_EVALUATION_CANDLES_BY_INTERVAL_MINUTES = {
    60: 4,
    240: 4,
}


def format_historical_data_error(exc: Exception) -> str:
    exc_type = exc.__class__.__name__
    status_code = getattr(getattr(exc, "response", None), "status_code", None)
    if status_code is not None:
        return f"historical_data_error: {exc_type}: HTTP {status_code}: {exc}"
    return f"historical_data_error: {exc_type}: {exc}"


def interval_to_minutes(interval: str) -> int:
    unit = interval[-1].lower()
    value = int(interval[:-1])
    if unit == "m":
        return value
    if unit == "h":
        return value * 60
    if unit == "d":
        return value * 1440
    raise ValueError(f"Unsupported interval: {interval}")


def horizon_candles_for_interval(horizon_minutes: int, interval: str) -> int:
    interval_minutes = interval_to_minutes(interval)
    requested = max(1, int((horizon_minutes + interval_minutes - 1) / interval_minutes))
    minimum = MIN_EVALUATION_CANDLES_BY_INTERVAL_MINUTES.get(interval_minutes, 1)
    return max(requested, minimum)


def effective_horizon_minutes(horizon_minutes: int, interval: str) -> int:
    return horizon_candles_for_interval(horizon_minutes, interval) * interval_to_minutes(interval)


def validate_strategy_modes(strategy_modes: list[str]) -> list[str]:
    invalid = [mode for mode in strategy_modes if mode not in ALLOWED_STRATEGY_MODES]
    if invalid:
        raise ValueError(f"Unsupported strategy_mode(s): {', '.join(invalid)}")
    return strategy_modes


def first_metric(result: dict[str, Any]) -> dict[str, Any]:
    metrics = result.get("metrics") or []
    return metrics[0] if metrics else {}


def safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def average(values: list[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0


def win_rate(outcomes: list[dict[str, Any]]) -> float:
    if not outcomes:
        return 0
    wins = sum(1 for outcome in outcomes if outcome.get("outcome") == "WIN")
    return round(wins / len(outcomes) * 100, 6)


def average_return(outcomes: list[dict[str, Any]]) -> float:
    returns = [safe_float(outcome.get("return_pct")) or 0 for outcome in outcomes]
    return average(returns)


def average_feature_value(feature_rows: list[dict[str, Any]], key: str) -> float:
    values = [value for value in (safe_float(row.get(key)) for row in feature_rows) if value is not None]
    return average(values)


def max_feature_value(feature_rows: list[dict[str, Any]], key: str) -> float:
    values = [value for value in (safe_float(row.get(key)) for row in feature_rows) if value is not None]
    return round(max(values), 6) if values else 0


def hold_reasons(feature_rows: list[dict[str, Any]]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for row in feature_rows:
        reason = row.get("hold_reason")
        if reason:
            summary[str(reason)] = summary.get(str(reason), 0) + 1
    return summary


def combined_feature_nan_summary(feature_rows: list[dict[str, Any]]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for row in feature_rows:
        values = row.get("feature_nan_summary") or {}
        if not isinstance(values, dict):
            continue
        for key, value in values.items():
            try:
                summary[str(key)] = summary.get(str(key), 0) + int(value)
            except (TypeError, ValueError):
                continue
    return summary


def first_feature_value(feature_rows: list[dict[str, Any]], key: str) -> Any:
    for row in feature_rows:
        value = row.get(key)
        if value is not None:
            return value
    return None


def performance_for_predictions(predictions: list[dict[str, Any]], outcomes_by_prediction: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = [outcomes_by_prediction[prediction["id"]] for prediction in predictions if prediction.get("id") in outcomes_by_prediction]
    return {
        "count": len(predictions),
        "evaluated_predictions": len(rows),
        "win_rate": win_rate(rows),
        "average_return": average_return(rows),
        "total_return_pct": round(sum(safe_float(row.get("return_pct")) or 0 for row in rows), 6) if rows else 0,
        "profit_factor": calculate_profit_factor(rows),
    }


def prediction_hour_utc(prediction: dict[str, Any]) -> str:
    value = prediction.get("created_at")
    if not value:
        return "unknown"
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return f"{dt.astimezone(timezone.utc).hour:02d}:00"
    except (TypeError, ValueError):
        return "unknown"


def performance_by_hour_utc(
    predictions: list[dict[str, Any]],
    outcomes_by_prediction: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for prediction in predictions:
        buckets.setdefault(prediction_hour_utc(prediction), []).append(prediction)
    return {
        hour: performance_for_predictions(group, outcomes_by_prediction)
        for hour, group in sorted(buckets.items())
    }


def best_hour_by_average_return(hourly: dict[str, dict[str, Any]]) -> str | None:
    evaluated = [
        (hour, metrics)
        for hour, metrics in hourly.items()
        if metrics.get("evaluated_predictions", 0) > 0
    ]
    if not evaluated:
        return None
    return max(evaluated, key=lambda item: (item[1].get("average_return", 0), item[1].get("profit_factor", 0)))[0]


def worst_hour_by_average_return(hourly: dict[str, dict[str, Any]]) -> str | None:
    evaluated = [
        (hour, metrics)
        for hour, metrics in hourly.items()
        if metrics.get("evaluated_predictions", 0) > 0
    ]
    if not evaluated:
        return None
    return min(evaluated, key=lambda item: (item[1].get("average_return", 0), item[1].get("profit_factor", 0)))[0]


def confidence_bucket(confidence: Any) -> str:
    value = safe_float(confidence)
    if value is None:
        return "unknown"
    if value < 40:
        return "0-40"
    if value < 60:
        return "40-60"
    if value < 70:
        return "60-70"
    if value < 80:
        return "70-80"
    return "80-100"


def diagnostic_metrics(result: dict[str, Any] | None) -> dict[str, Any]:
    predictions = (result or {}).get("predictions") or []
    outcomes = (result or {}).get("outcomes") or []
    outcomes_by_prediction = {outcome.get("prediction_id"): outcome for outcome in outcomes if outcome.get("prediction_id")}
    feature_rows = [prediction.get("input_features") or {} for prediction in predictions]

    buy_predictions = [prediction for prediction in predictions if prediction.get("signal") == "BUY"]
    sell_predictions = [prediction for prediction in predictions if prediction.get("signal") == "SELL"]
    hold_predictions = [prediction for prediction in predictions if prediction.get("signal") == "HOLD"]
    buy_perf = performance_for_predictions(buy_predictions, outcomes_by_prediction)
    sell_perf = performance_for_predictions(sell_predictions, outcomes_by_prediction)

    risk_rewards = [value for value in (safe_float(prediction.get("risk_reward_ratio")) for prediction in predictions) if value is not None]
    confidences = [value for value in (safe_float(prediction.get("confidence")) for prediction in predictions) if value is not None]
    stop_distances = []
    take_profit_distances = []
    for prediction in predictions:
        entry = safe_float(prediction.get("entry_price"))
        stop_loss = safe_float(prediction.get("stop_loss"))
        take_profit = safe_float(prediction.get("take_profit"))
        if entry and entry > 0 and stop_loss is not None:
            stop_distances.append(abs(entry - stop_loss) / entry * 100)
        if entry and entry > 0 and take_profit is not None:
            take_profit_distances.append(abs(take_profit - entry) / entry * 100)

    buckets: dict[str, list[dict[str, Any]]] = {name: [] for name in ["0-40", "40-60", "60-70", "70-80", "80-100", "unknown"]}
    for prediction in predictions:
        buckets.setdefault(confidence_bucket(prediction.get("confidence")), []).append(prediction)
    bucket_metrics = {
        name: performance_for_predictions(group, outcomes_by_prediction)
        for name, group in buckets.items()
        if group
    }
    hourly_metrics = performance_by_hour_utc(predictions, outcomes_by_prediction)

    return {
        "buy_count": len(buy_predictions),
        "sell_count": len(sell_predictions),
        "hold_count": len(hold_predictions),
        "buy_win_rate": buy_perf["win_rate"],
        "sell_win_rate": sell_perf["win_rate"],
        "buy_average_return": buy_perf["average_return"],
        "sell_average_return": sell_perf["average_return"],
        "avg_confidence": average(confidences),
        "avg_risk_reward": average(risk_rewards),
        "avg_stop_distance_pct": average(stop_distances),
        "avg_take_profit_distance_pct": average(take_profit_distances),
        "tp_hit_count": sum(1 for outcome in outcomes if outcome.get("hit_take_profit")),
        "sl_hit_count": sum(1 for outcome in outcomes if outcome.get("hit_stop_loss")),
        "expired_count": sum(1 for outcome in outcomes if outcome.get("outcome") == "EXPIRED"),
        "confidence_buckets": bucket_metrics,
        "hourly_performance_utc": hourly_metrics,
        "best_hour_utc": best_hour_by_average_return(hourly_metrics),
        "worst_hour_utc": worst_hour_by_average_return(hourly_metrics),
        "avg_probability_buy_win": average_feature_value(feature_rows, "probability_buy_win"),
        "avg_probability_sell_win": average_feature_value(feature_rows, "probability_sell_win"),
        "max_probability_buy_win": max_feature_value(feature_rows, "probability_buy_win"),
        "max_probability_sell_win": max_feature_value(feature_rows, "probability_sell_win"),
        "raw_buy_label_count": average_feature_value(feature_rows, "raw_buy_label_count"),
        "raw_sell_label_count": average_feature_value(feature_rows, "raw_sell_label_count"),
        "raw_buy_positive_count": average_feature_value(feature_rows, "raw_buy_positive_count"),
        "raw_sell_positive_count": average_feature_value(feature_rows, "raw_sell_positive_count"),
        "feature_valid_count": average_feature_value(feature_rows, "feature_valid_count"),
        "avg_buy_label_count": average_feature_value(feature_rows, "buy_label_count"),
        "avg_sell_label_count": average_feature_value(feature_rows, "sell_label_count"),
        "avg_buy_positive_rate": average_feature_value(feature_rows, "buy_positive_rate"),
        "avg_sell_positive_rate": average_feature_value(feature_rows, "sell_positive_rate"),
        "hold_reasons_summary": hold_reasons(feature_rows),
        "feature_nan_summary": combined_feature_nan_summary(feature_rows),
        "label_level_mode": first_feature_value(feature_rows, "label_level_mode"),
        "trade_label_scheme": first_feature_value(feature_rows, "trade_label_scheme"),
        "label_params": {
            "min_train_rows": first_feature_value(feature_rows, "min_train_rows"),
            "trade_label_scheme": first_feature_value(feature_rows, "trade_label_scheme"),
            "label_stop_loss_pct": first_feature_value(feature_rows, "label_stop_loss_pct"),
            "label_take_profit_pct": first_feature_value(feature_rows, "label_take_profit_pct"),
            "label_horizon_candles": first_feature_value(feature_rows, "label_horizon_candles"),
            "expiry_return_threshold_pct": first_feature_value(feature_rows, "expiry_return_threshold_pct"),
            "label_atr_stop_multiplier": first_feature_value(feature_rows, "label_atr_stop_multiplier"),
            "label_atr_take_profit_multiplier": first_feature_value(feature_rows, "label_atr_take_profit_multiplier"),
            "label_min_risk_reward": first_feature_value(feature_rows, "label_min_risk_reward"),
            "label_costs": first_feature_value(feature_rows, "label_costs"),
        },
        "label_horizon_candles": first_feature_value(feature_rows, "label_horizon_candles"),
        "label_type": first_feature_value(feature_rows, "label_type"),
    }


def flatten_trade_rows(
    result: dict[str, Any],
    symbol: str,
    timeframe: str,
    strategy_mode: str,
    use_trade_labels: bool,
    effective_horizon_minutes_value: int,
    evaluation_horizon_candles: int,
) -> list[dict[str, Any]]:
    predictions = result.get("predictions") or []
    outcomes_by_prediction = {
        outcome.get("prediction_id"): outcome
        for outcome in (result.get("outcomes") or [])
        if outcome.get("prediction_id")
    }
    rows = []
    for prediction in predictions:
        features = prediction.get("input_features") or {}
        outcome = outcomes_by_prediction.get(prediction.get("id"), {})
        entry = safe_float(prediction.get("entry_price"))
        stop_loss = safe_float(prediction.get("stop_loss"))
        take_profit = safe_float(prediction.get("take_profit"))
        outcome_name = outcome.get("outcome")
        rows.append({
            "symbol": symbol,
            "timeframe": timeframe,
            "strategy_mode": strategy_mode,
            "use_trade_labels": bool(use_trade_labels),
            "label_type": features.get("label_type"),
            "trade_label_scheme": features.get("trade_label_scheme"),
            "label_level_mode": features.get("label_level_mode"),
            "label_horizon_candles": features.get("label_horizon_candles"),
            "created_at": prediction.get("created_at"),
            "signal": prediction.get("signal"),
            "confidence": prediction.get("confidence"),
            "probability_buy_win": features.get("probability_buy_win"),
            "probability_sell_win": features.get("probability_sell_win"),
            "entry_price": prediction.get("entry_price"),
            "exit_price": outcome.get("exit_price"),
            "outcome": outcome_name,
            "return_pct": outcome.get("return_pct"),
            "tp_hit": bool(outcome.get("hit_take_profit")),
            "sl_hit": bool(outcome.get("hit_stop_loss")),
            "expired": outcome_name == "EXPIRED",
            "stop_distance_pct": abs(entry - stop_loss) / entry * 100 if entry and stop_loss is not None else None,
            "take_profit_distance_pct": abs(take_profit - entry) / entry * 100 if entry and take_profit is not None else None,
            "hold_reason": features.get("hold_reason"),
            "risk_reward": prediction.get("risk_reward_ratio"),
            "effective_horizon_minutes": effective_horizon_minutes_value,
            "evaluation_horizon_candles": evaluation_horizon_candles,
        })
    return rows


def summarize_run(
    symbol: str,
    timeframe: str,
    strategy_mode: str,
    result: dict[str, Any] | None = None,
    error: str | None = None,
    requested_horizon_minutes: int | None = None,
    effective_horizon_minutes_value: int | None = None,
    evaluation_horizon_candles: int | None = None,
    sentiment_used: bool = HISTORICAL_SENTIMENT_USED,
    sentiment_note: str = HISTORICAL_SENTIMENT_NOTE,
    use_trade_labels: bool = False,
) -> dict[str, Any]:
    metric = first_metric(result or {})
    outcomes = (result or {}).get("outcomes") or []
    invalid_count = sum(1 for outcome in outcomes if outcome.get("outcome") == "INVALID_DATA")
    warnings = []
    assumptions = (result or {}).get("assumptions") or {}
    if assumptions.get("error"):
        warnings.append(str(assumptions["error"]))
    if error:
        warnings.append(error)

    row = {
        "symbol": symbol,
        "timeframe": timeframe,
        "strategy_mode": strategy_mode,
        "requested_horizon_minutes": requested_horizon_minutes,
        "effective_horizon_minutes": effective_horizon_minutes_value,
        "evaluation_horizon_candles": evaluation_horizon_candles,
        "sentiment_used": sentiment_used,
        "sentiment_note": sentiment_note,
        "use_trade_labels": bool(use_trade_labels),
        "total_predictions": len((result or {}).get("predictions") or []),
        "evaluated_predictions": metric.get("evaluated_predictions", len(outcomes)),
        "win_rate": metric.get("win_rate", 0),
        "average_return": metric.get("average_return", 0),
        "total_return_pct": metric.get("total_return_pct", 0),
        "profit_factor": metric.get("profit_factor", 0),
        "max_drawdown": metric.get("max_drawdown", 0),
        "sharpe": metric.get("sharpe", 0),
        "invalid_count": invalid_count,
        "warnings": "; ".join(warnings),
    }
    row.update(diagnostic_metrics(result))
    return row


async def run_experiments(
    symbols: list[str],
    timeframes: list[str],
    strategy_modes: list[str],
    horizon_minutes: int = DEFAULT_HORIZON_MINUTES,
    max_candles: int = DEFAULT_MAX_CANDLES,
    max_predictions: int = DEFAULT_MAX_PREDICTIONS,
    persist: bool = False,
    reports_dir: str | Path = "reports",
    use_trade_labels: bool = False,
    save_trades: bool = False,
    trade_label_scheme: str = DEFAULT_TRADE_LABEL_SCHEME,
) -> dict[str, Any]:
    if trade_label_scheme not in ALLOWED_TRADE_LABEL_SCHEMES:
        raise ValueError(f"Unsupported trade_label_scheme: {trade_label_scheme}")
    strategy_modes = validate_strategy_modes(strategy_modes)
    store = PredictionStore() if persist else None
    summaries: list[dict[str, Any]] = []
    raw_runs: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []

    strategy_params = {"use_sentiment": HISTORICAL_SENTIMENT_USED}

    for symbol in symbols:
        for timeframe in timeframes:
            replay_horizon_candles = horizon_candles_for_interval(horizon_minutes, timeframe)
            replay_horizon_minutes = effective_horizon_minutes(horizon_minutes, timeframe)
            run_strategy_params = dict(strategy_params)
            replay_min_history = 50
            if use_trade_labels:
                run_strategy_params.update({
                    "use_trade_labels": True,
                    "trade_label_scheme": trade_label_scheme,
                    "horizon_candles": replay_horizon_candles,
                    "min_train_rows": DEFAULT_TRADE_LABEL_MIN_TRAIN_ROWS,
                    "commission_pct": 0.001,
                    "slippage_pct": 0.0005,
                    "spread_pct": 0.0003,
                })
                min_train_rows = int(run_strategy_params.get("min_train_rows", 200))
                replay_min_history = max(replay_min_history, min_train_rows + 35 + replay_horizon_candles + 2)
            if replay_horizon_minutes != horizon_minutes:
                print(
                    f"WARNING: {timeframe} requested_horizon_minutes={horizon_minutes} "
                    f"effective_horizon_minutes={replay_horizon_minutes} "
                    f"evaluation_horizon_candles={replay_horizon_candles}"
                )
            candles = None
            try:
                candles = await fetch_binance_klines(symbol, timeframe, limit=max_candles)
            except Exception as exc:
                error = format_historical_data_error(exc)
                for strategy_mode in strategy_modes:
                    summaries.append(
                        summarize_run(
                            symbol,
                            timeframe,
                            strategy_mode,
                            error=error,
                            requested_horizon_minutes=horizon_minutes,
                            effective_horizon_minutes_value=replay_horizon_minutes,
                            evaluation_horizon_candles=replay_horizon_candles,
                            use_trade_labels=use_trade_labels,
                        )
                    )
                continue

            if use_trade_labels and max_predictions is not None:
                last_n = len(candles) - replay_horizon_candles - 2
                latest_window_start_n = max(0, last_n - max_predictions + 1)
                replay_min_history = max(replay_min_history, latest_window_start_n + 1)

            for strategy_mode in strategy_modes:
                try:
                    result = run_historical_replay(
                        candles,
                        symbol=symbol,
                        timeframe=timeframe,
                        strategy_mode=strategy_mode,
                        horizon_candles=replay_horizon_candles,
                        horizon_minutes=replay_horizon_minutes,
                        max_predictions=max_predictions,
                        min_history=replay_min_history,
                        strategy_params=run_strategy_params,
                        store=store,
                    )
                    summaries.append(
                        summarize_run(
                            symbol,
                            timeframe,
                            strategy_mode,
                            result=result,
                            requested_horizon_minutes=horizon_minutes,
                            effective_horizon_minutes_value=replay_horizon_minutes,
                            evaluation_horizon_candles=replay_horizon_candles,
                            use_trade_labels=use_trade_labels,
                        )
                    )
                    raw_runs.append({
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "strategy_mode": strategy_mode,
                        "requested_horizon_minutes": horizon_minutes,
                        "effective_horizon_minutes": replay_horizon_minutes,
                        "evaluation_horizon_candles": replay_horizon_candles,
                        "sentiment_used": HISTORICAL_SENTIMENT_USED,
                        "sentiment_note": HISTORICAL_SENTIMENT_NOTE,
                        "use_trade_labels": use_trade_labels,
                        "min_history": replay_min_history,
                        "assumptions": result.get("assumptions", {}),
                        "metrics": result.get("metrics", []),
                    })
                    if save_trades:
                        trade_rows.extend(flatten_trade_rows(
                            result,
                            symbol=symbol,
                            timeframe=timeframe,
                            strategy_mode=strategy_mode,
                            use_trade_labels=use_trade_labels,
                            effective_horizon_minutes_value=replay_horizon_minutes,
                            evaluation_horizon_candles=replay_horizon_candles,
                        ))
                except Exception as exc:
                    summaries.append(
                        summarize_run(
                            symbol,
                            timeframe,
                            strategy_mode,
                            error=f"replay_error: {exc}",
                            requested_horizon_minutes=horizon_minutes,
                            effective_horizon_minutes_value=replay_horizon_minutes,
                            evaluation_horizon_candles=replay_horizon_candles,
                            use_trade_labels=use_trade_labels,
                        )
                    )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "symbols": symbols,
            "timeframes": timeframes,
            "strategy_modes": strategy_modes,
            "horizon_minutes": horizon_minutes,
            "max_candles": max_candles,
            "max_predictions_per_run": max_predictions,
            "evaluation_horizon_note": "1h and 4h runs use at least 4 future candles so TP/SL has time to resolve.",
            "sentiment_used": HISTORICAL_SENTIMENT_USED,
            "sentiment_note": HISTORICAL_SENTIMENT_NOTE,
            "use_trade_labels": use_trade_labels,
            "trade_label_scheme": trade_label_scheme if use_trade_labels else None,
            "trade_label_min_train_rows": DEFAULT_TRADE_LABEL_MIN_TRAIN_ROWS if use_trade_labels else None,
            "save_trades": save_trades,
            "persist": persist,
        },
        "summary": summaries,
        "runs": raw_runs,
        "trades": trade_rows if save_trades else [],
    }
    paths = write_report(report, reports_dir, save_trades=save_trades)
    report["report_paths"] = paths
    return report


def write_report(report: dict[str, Any], reports_dir: str | Path = "reports", save_trades: bool = False) -> dict[str, str]:
    target = Path(reports_dir)
    target.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = target / f"historical_experiments_{stamp}.json"
    csv_path = target / f"historical_experiments_{stamp}.csv"
    trades_path = target / f"historical_trades_{stamp}.csv"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    rows = report["summary"]
    fieldnames = [
        "symbol", "timeframe", "strategy_mode", "total_predictions",
        "requested_horizon_minutes", "effective_horizon_minutes",
        "evaluation_horizon_candles", "sentiment_used", "sentiment_note",
        "use_trade_labels",
        "evaluated_predictions", "win_rate", "average_return",
        "total_return_pct", "profit_factor", "max_drawdown", "sharpe",
        "invalid_count", "buy_count", "sell_count", "hold_count",
        "buy_win_rate", "sell_win_rate", "buy_average_return",
        "sell_average_return", "avg_confidence", "avg_risk_reward",
        "avg_stop_distance_pct", "avg_take_profit_distance_pct",
        "tp_hit_count", "sl_hit_count", "expired_count",
        "avg_probability_buy_win", "avg_probability_sell_win",
        "max_probability_buy_win", "max_probability_sell_win",
        "raw_buy_label_count", "raw_sell_label_count",
        "raw_buy_positive_count", "raw_sell_positive_count",
        "feature_valid_count",
        "avg_buy_label_count", "avg_sell_label_count",
        "avg_buy_positive_rate", "avg_sell_positive_rate",
        "hold_reasons_summary", "feature_nan_summary", "label_level_mode",
        "trade_label_scheme", "label_params", "label_horizon_candles", "label_type",
        "confidence_buckets", "hourly_performance_utc", "best_hour_utc",
        "worst_hour_utc", "warnings",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["confidence_buckets"] = json.dumps(csv_row.get("confidence_buckets", {}), sort_keys=True)
            csv_row["hourly_performance_utc"] = json.dumps(csv_row.get("hourly_performance_utc", {}), sort_keys=True)
            csv_row["hold_reasons_summary"] = json.dumps(csv_row.get("hold_reasons_summary", {}), sort_keys=True)
            csv_row["feature_nan_summary"] = json.dumps(csv_row.get("feature_nan_summary", {}), sort_keys=True)
            csv_row["label_params"] = json.dumps(csv_row.get("label_params", {}), sort_keys=True)
            csv_row["use_trade_labels"] = "true" if csv_row.get("use_trade_labels") else "false"
            writer.writerow(csv_row)
    paths = {"json": str(json_path), "csv": str(csv_path)}
    if save_trades:
        trade_fieldnames = [
            "symbol", "timeframe", "strategy_mode", "use_trade_labels",
            "label_type", "label_level_mode", "label_horizon_candles",
            "trade_label_scheme",
            "created_at", "signal", "confidence", "probability_buy_win",
            "probability_sell_win", "entry_price", "exit_price", "outcome",
            "return_pct", "tp_hit", "sl_hit", "expired", "stop_distance_pct",
            "take_profit_distance_pct", "hold_reason", "risk_reward",
            "effective_horizon_minutes", "evaluation_horizon_candles",
        ]
        with trades_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=trade_fieldnames)
            writer.writeheader()
            for row in report.get("trades", []):
                csv_row = dict(row)
                csv_row["use_trade_labels"] = "true" if csv_row.get("use_trade_labels") else "false"
                csv_row["tp_hit"] = "true" if csv_row.get("tp_hit") else "false"
                csv_row["sl_hit"] = "true" if csv_row.get("sl_hit") else "false"
                csv_row["expired"] = "true" if csv_row.get("expired") else "false"
                writer.writerow(csv_row)
        paths["trades_csv"] = str(trades_path)
    return paths


def print_summary(rows: list[dict[str, Any]]) -> None:
    headers = [
        "symbol", "timeframe", "strategy_mode", "total_predictions",
        "requested_horizon_minutes", "effective_horizon_minutes",
        "evaluation_horizon_candles", "sentiment_used", "use_trade_labels",
        "evaluated_predictions", "win_rate", "average_return",
        "total_return_pct", "profit_factor", "max_drawdown", "sharpe",
        "invalid_count", "buy_count", "sell_count", "hold_count",
        "buy_win_rate", "sell_win_rate", "buy_average_return",
        "sell_average_return", "avg_confidence", "avg_risk_reward",
        "avg_stop_distance_pct", "avg_take_profit_distance_pct",
        "tp_hit_count", "sl_hit_count", "expired_count",
        "avg_probability_buy_win", "avg_probability_sell_win",
        "max_probability_buy_win", "max_probability_sell_win",
        "raw_buy_label_count", "raw_sell_label_count",
        "raw_buy_positive_count", "raw_sell_positive_count",
        "feature_valid_count",
        "avg_buy_label_count", "avg_sell_label_count",
        "avg_buy_positive_rate", "avg_sell_positive_rate",
        "hold_reasons_summary", "label_level_mode", "label_horizon_candles",
        "trade_label_scheme", "label_type", "best_hour_utc", "worst_hour_utc", "warnings",
    ]
    widths = {header: max(len(header), *(len(str(row.get(header, ""))) for row in rows)) for header in headers}
    print(" | ".join(header.ljust(widths[header]) for header in headers))
    print("-+-".join("-" * widths[header] for header in headers))
    for row in rows:
        print(" | ".join(str(row.get(header, "")).ljust(widths[header]) for header in headers))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run historical replay experiments.")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--timeframes", nargs="+", default=DEFAULT_TIMEFRAMES)
    parser.add_argument("--strategy-modes", nargs="+", default=DEFAULT_STRATEGY_MODES)
    parser.add_argument("--max-candles", type=int, default=DEFAULT_MAX_CANDLES)
    parser.add_argument("--max-predictions", type=int, default=DEFAULT_MAX_PREDICTIONS)
    parser.add_argument("--horizon-minutes", type=int, default=DEFAULT_HORIZON_MINUTES)
    parser.add_argument("--persist", action="store_true")
    parser.add_argument("--use-trade-labels", action="store_true")
    parser.add_argument("--save-trades", action="store_true")
    parser.add_argument("--trade-label-scheme", choices=sorted(ALLOWED_TRADE_LABEL_SCHEMES), default=DEFAULT_TRADE_LABEL_SCHEME)
    return parser


async def main() -> None:
    args = build_parser().parse_args()
    report = await run_experiments(
        symbols=[symbol.upper() for symbol in args.symbols],
        timeframes=args.timeframes,
        strategy_modes=args.strategy_modes,
        horizon_minutes=args.horizon_minutes,
        max_candles=args.max_candles,
        max_predictions=args.max_predictions,
        persist=args.persist,
        use_trade_labels=args.use_trade_labels,
        save_trades=args.save_trades,
        trade_label_scheme=args.trade_label_scheme,
    )
    print_summary(report["summary"])
    print(f"JSON report: {report['report_paths']['json']}")
    print(f"CSV report: {report['report_paths']['csv']}")
    if "trades_csv" in report["report_paths"]:
        print(f"Trades CSV: {report['report_paths']['trades_csv']}")


if __name__ == "__main__":
    asyncio.run(main())
