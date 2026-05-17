"""Prediction journal storage and outcome evaluation.

Supabase is used when configured. For local development without Supabase, the
store falls back to a small JSON file under data/.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd


VALID_SIGNALS = {"BUY", "SELL", "HOLD"}
VALID_MODES = {"deterministic", "model_based", "hybrid"}
PENDING = "pending"
EVALUATED = "evaluated"
INVALID = "invalid"

BINANCE_BASE = "https://api.binance.com/api/v3"
SYMBOL_MAP = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "BNB": "BNBUSDT",
    "SOL": "SOLUSDT",
    "XRP": "XRPUSDT",
    "ADA": "ADAUSDT",
    "DOGE": "DOGEUSDT",
    "AVAX": "AVAXUSDT",
    "LINK": "LINKUSDT",
    "DOT": "DOTUSDT",
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_dt(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def normalize_prediction(payload: dict[str, Any]) -> dict[str, Any]:
    signal = str(payload.get("signal", "HOLD")).upper()
    strategy_mode = str(payload.get("strategy_mode", "deterministic"))
    created_at = payload.get("created_at") or utc_now().isoformat()

    if signal not in VALID_SIGNALS:
        raise ValueError("signal must be BUY, SELL, or HOLD")
    if strategy_mode not in VALID_MODES:
        raise ValueError("strategy_mode must be deterministic, model_based, or hybrid")

    entry_price = float(payload.get("entry_price", 0) or 0)
    stop_loss = payload.get("stop_loss")
    take_profit = payload.get("take_profit")
    risk_reward_ratio = payload.get("risk_reward_ratio")

    if risk_reward_ratio is None and entry_price > 0 and stop_loss and take_profit:
        risk = abs(entry_price - float(stop_loss))
        reward = abs(float(take_profit) - entry_price)
        risk_reward_ratio = reward / risk if risk > 0 else None

    return {
        "id": str(payload.get("id") or uuid4()),
        "user_id": payload.get("user_id"),
        "symbol": str(payload.get("symbol", "")).upper(),
        "timeframe": payload.get("timeframe") or payload.get("interval") or "1h",
        "strategy_mode": strategy_mode,
        "strategy_name": payload.get("strategy_name") or "deterministic_signal",
        "strategy_version": payload.get("strategy_version") or "v1",
        "signal": signal,
        "confidence": float(payload.get("confidence", 0) or 0),
        "entry_price": entry_price,
        "stop_loss": float(stop_loss) if stop_loss is not None else None,
        "take_profit": float(take_profit) if take_profit is not None else None,
        "risk_reward_ratio": float(risk_reward_ratio) if risk_reward_ratio is not None else None,
        "horizon_minutes": int(payload.get("horizon_minutes") or 60),
        "input_features": payload.get("input_features") or {},
        "reasoning": payload.get("reasoning") or "",
        "model_provider": payload.get("model_provider"),
        "model_name": payload.get("model_name"),
        "status": payload.get("status") or PENDING,
        "created_at": parse_dt(created_at).isoformat(),
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
        if column not in df.columns:
            raise ValueError(f"Missing required OHLC column: {column}")
        df[column] = df[column].astype(float)
    if "volume" in df.columns:
        df["volume"] = df["volume"].astype(float)
    return df


def evaluate_prediction_against_candles(
    prediction: dict[str, Any],
    candles: pd.DataFrame,
    commission_pct: float = 0.001,
    slippage_pct: float = 0.0005,
) -> dict[str, Any]:
    """Evaluate one prediction using candles after its creation timestamp."""
    pred = normalize_prediction(prediction)
    signal = pred["signal"]
    created_at = parse_dt(pred["created_at"])
    horizon_end = created_at + timedelta(minutes=pred["horizon_minutes"])
    df = normalize_ohlcv(candles)
    path = df[(df.index > created_at) & (df.index <= horizon_end)]

    if path.empty or pred["entry_price"] <= 0 or signal == "HOLD":
        return build_invalid_or_expired_outcome(pred, path, signal)

    entry = pred["entry_price"]
    exit_price = float(path.iloc[-1]["close"])
    highs = path["high"].astype(float)
    lows = path["low"].astype(float)

    if signal == "BUY":
        best_price = float(highs.max())
        worst_price = float(lows.min())
        return_pct = (exit_price - entry) / entry * 100
        mfe = (best_price - entry) / entry * 100
        mae = (entry - worst_price) / entry * 100
        hit_sl = pred["stop_loss"] is not None and bool((lows <= pred["stop_loss"]).any())
        hit_tp = pred["take_profit"] is not None and bool((highs >= pred["take_profit"]).any())
    elif signal == "SELL":
        best_price = float(lows.min())
        worst_price = float(highs.max())
        return_pct = (entry - exit_price) / entry * 100
        mfe = (entry - best_price) / entry * 100
        mae = (worst_price - entry) / entry * 100
        hit_sl = pred["stop_loss"] is not None and bool((highs >= pred["stop_loss"]).any())
        hit_tp = pred["take_profit"] is not None and bool((lows <= pred["take_profit"]).any())
    else:
        return build_invalid_or_expired_outcome(pred, path, signal)

    fees_paid = abs(entry + exit_price) * commission_pct
    slippage_cost = abs(entry + exit_price) * slippage_pct
    net_return_pct = return_pct - ((fees_paid + slippage_cost) / entry * 100)

    if hit_tp and not hit_sl:
        outcome = "WIN"
    elif hit_sl and not hit_tp:
        outcome = "LOSS"
    elif path.index[-1] >= horizon_end:
        outcome = "EXPIRED"
    elif net_return_pct > 0.05:
        outcome = "WIN"
    elif net_return_pct < -0.05:
        outcome = "LOSS"
    else:
        outcome = "BREAKEVEN"

    return {
        "id": str(uuid4()),
        "prediction_id": pred["id"],
        "evaluated_at": utc_now().isoformat(),
        "exit_price": round(exit_price, 8),
        "return_pct": round(net_return_pct, 8),
        "max_favorable_excursion_pct": round(max(0.0, float(mfe)), 8),
        "max_adverse_excursion_pct": round(max(0.0, float(mae)), 8),
        "hit_stop_loss": hit_sl,
        "hit_take_profit": hit_tp,
        "outcome": outcome,
        "fees_paid": round(fees_paid, 8),
        "slippage_cost": round(slippage_cost, 8),
        "raw_path": candles_to_raw_path(path),
    }


def build_invalid_or_expired_outcome(prediction: dict[str, Any], path: pd.DataFrame, signal: str) -> dict[str, Any]:
    if signal == "HOLD" and not path.empty:
        outcome = "EXPIRED"
    else:
        outcome = "INVALID_DATA"
    exit_price = float(path.iloc[-1]["close"]) if not path.empty else None
    return {
        "id": str(uuid4()),
        "prediction_id": prediction["id"],
        "evaluated_at": utc_now().isoformat(),
        "exit_price": exit_price,
        "return_pct": 0.0,
        "max_favorable_excursion_pct": 0.0,
        "max_adverse_excursion_pct": 0.0,
        "hit_stop_loss": False,
        "hit_take_profit": False,
        "outcome": outcome,
        "fees_paid": 0.0,
        "slippage_cost": 0.0,
        "raw_path": candles_to_raw_path(path),
    }


def candles_to_raw_path(path: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for ts, row in path.iterrows():
        rows.append({
            "time": ts.isoformat(),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]) if "volume" in row and not pd.isna(row["volume"]) else None,
        })
    return rows


async def fetch_future_klines(symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch Binance candles after prediction time for outcome evaluation."""
    import httpx

    ticker = SYMBOL_MAP.get(symbol.upper(), symbol.upper() + "USDT")
    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.get(
            f"{BINANCE_BASE}/klines",
            params={
                "symbol": ticker,
                "interval": timeframe,
                "startTime": int(start.timestamp() * 1000),
                "endTime": int(end.timestamp() * 1000),
                "limit": 1000,
            },
        )
        response.raise_for_status()
        raw = response.json()

    df = pd.DataFrame(raw, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore",
    ])
    if df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return normalize_ohlcv(df[["open_time", "open", "high", "low", "close", "volume"]])


class PredictionStore:
    def __init__(self, supabase_client: Any = None, file_path: str | Path = "data/prediction_journal.json"):
        self.supabase = supabase_client if self._supabase_configured() else None
        self.file_path = Path(file_path)

    def _supabase_configured(self) -> bool:
        url = os.getenv("SUPABASE_URL", "")
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
        return bool(url and key and "xxxx" not in url and "your_" not in key)

    def _read_file(self) -> dict[str, list[dict[str, Any]]]:
        if not self.file_path.exists():
            return {"predictions": [], "outcomes": []}
        return json.loads(self.file_path.read_text(encoding="utf-8"))

    def _write_file(self, data: dict[str, list[dict[str, Any]]]) -> None:
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.file_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def create_prediction(self, payload: dict[str, Any]) -> dict[str, Any]:
        prediction = normalize_prediction(payload)
        if self.supabase is not None:
            try:
                result = self.supabase.table("prediction_journal").insert(prediction).execute()
                return result.data[0] if result.data else prediction
            except Exception:
                pass
        data = self._read_file()
        data["predictions"].append(prediction)
        self._write_file(data)
        return prediction

    def list_predictions(self, status: str | None = None, symbol: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        if self.supabase is not None:
            try:
                query = self.supabase.table("prediction_journal").select("*").order("created_at", desc=True).limit(limit)
                if status:
                    query = query.eq("status", status)
                if symbol:
                    query = query.eq("symbol", symbol.upper())
                return query.execute().data or []
            except Exception:
                pass
        predictions = self._read_file()["predictions"]
        if status:
            predictions = [item for item in predictions if item.get("status") == status]
        if symbol:
            predictions = [item for item in predictions if item.get("symbol") == symbol.upper()]
        return sorted(predictions, key=lambda item: item.get("created_at", ""), reverse=True)[:limit]

    def get_prediction(self, prediction_id: str) -> dict[str, Any] | None:
        if self.supabase is not None:
            try:
                result = self.supabase.table("prediction_journal").select("*").eq("id", prediction_id).single().execute()
                if result.data:
                    return result.data
            except Exception:
                pass
        return next((item for item in self._read_file()["predictions"] if item["id"] == prediction_id), None)

    def update_prediction_status(self, prediction_id: str, status: str) -> None:
        if self.supabase is not None:
            try:
                self.supabase.table("prediction_journal").update({"status": status}).eq("id", prediction_id).execute()
                return
            except Exception:
                pass
        data = self._read_file()
        for item in data["predictions"]:
            if item["id"] == prediction_id:
                item["status"] = status
        self._write_file(data)

    def create_outcome(self, outcome: dict[str, Any]) -> dict[str, Any]:
        if self.supabase is not None:
            try:
                result = self.supabase.table("prediction_outcomes").insert(outcome).execute()
                return result.data[0] if result.data else outcome
            except Exception:
                pass
        data = self._read_file()
        data["outcomes"].append(outcome)
        self._write_file(data)
        return outcome

    def list_outcomes(self) -> list[dict[str, Any]]:
        if self.supabase is not None:
            try:
                return self.supabase.table("prediction_outcomes").select("*").execute().data or []
            except Exception:
                pass
        return self._read_file()["outcomes"]

    def due_predictions(self, now: datetime | None = None, limit: int = 100) -> list[dict[str, Any]]:
        now = now or utc_now()
        pending = self.list_predictions(status=PENDING, limit=limit)
        return [
            prediction for prediction in pending
            if parse_dt(prediction["created_at"]) + timedelta(minutes=int(prediction["horizon_minutes"])) <= now
        ]


def metrics_by_signal(predictions: list[dict[str, Any]], outcomes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return aggregate_metrics(predictions, outcomes, "signal")


def metrics_by_strategy(predictions: list[dict[str, Any]], outcomes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return aggregate_metrics(predictions, outcomes, "strategy_name")


def metrics_by_strategy_mode(predictions: list[dict[str, Any]], outcomes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return aggregate_metrics(predictions, outcomes, "strategy_mode")


def metrics_by_symbol_timeframe(predictions: list[dict[str, Any]], outcomes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched = []
    for prediction in predictions:
        item = dict(prediction)
        item["symbol_timeframe"] = f"{prediction.get('symbol', 'UNKNOWN')}:{prediction.get('timeframe', 'unknown')}"
        enriched.append(item)
    return aggregate_metrics(enriched, outcomes, "symbol_timeframe")


def aggregate_metrics(predictions: list[dict[str, Any]], outcomes: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    prediction_map = {item["id"]: item for item in predictions}
    groups: dict[str, list[dict[str, Any]]] = {}
    for outcome in outcomes:
        prediction = prediction_map.get(outcome.get("prediction_id"))
        if not prediction:
            continue
        groups.setdefault(str(prediction.get(key, "unknown")), []).append(outcome)

    result = []
    for name, rows in groups.items():
        returns = [float(row.get("return_pct") or 0) for row in rows]
        wins = [row for row in rows if row.get("outcome") == "WIN"]
        losses = [row for row in rows if row.get("outcome") == "LOSS"]
        result.append({
            key: name,
            "evaluated_predictions": len(rows),
            "total_signals": len(rows),
            "win_rate": round(len(wins) / len(rows) * 100, 6) if rows else 0,
            "loss_rate": round(len(losses) / len(rows) * 100, 6) if rows else 0,
            "average_return": round(float(np.mean(returns)), 6) if returns else 0,
            "average_return_pct": round(float(np.mean(returns)), 6) if returns else 0,
            "total_return_pct": round(float(np.sum(returns)), 6) if returns else 0,
            "profit_factor": calculate_profit_factor(rows),
            "max_drawdown": calculate_return_drawdown(returns),
            "sharpe": calculate_sharpe(returns),
        })
    return sorted(result, key=lambda item: item["evaluated_predictions"], reverse=True)


def calculate_profit_factor(outcomes: list[dict[str, Any]]) -> float:
    profits = sum(float(row.get("return_pct") or 0) for row in outcomes if float(row.get("return_pct") or 0) > 0)
    losses = abs(sum(float(row.get("return_pct") or 0) for row in outcomes if float(row.get("return_pct") or 0) < 0))
    if losses == 0:
        return round(profits, 6) if profits > 0 else 0
    return round(profits / losses, 6)


def calculate_return_drawdown(returns: list[float]) -> float:
    equity = 100.0
    peak = equity
    max_drawdown = 0.0
    for ret in returns:
        equity *= 1 + ret / 100
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, (peak - equity) / peak * 100 if peak else 0)
    return round(max_drawdown, 6)


def calculate_sharpe(returns: list[float]) -> float:
    if len(returns) < 2 or np.std(returns) == 0:
        return 0
    return round(float(np.mean(returns) / np.std(returns) * np.sqrt(len(returns))), 6)
