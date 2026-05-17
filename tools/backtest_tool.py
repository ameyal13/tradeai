"""Backtest Engine V2 for Trading Copilot.

The engine is deterministic and candle-based:
- signals are generated from a closed candle;
- entries execute on the next candle open;
- exits use high/low for stop-loss and take-profit checks.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import AverageTrueRange, BollingerBands


COINGECKO_BASE = "https://api.coingecko.com/api/v3"

COINGECKO_IDS = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "BNB": "binancecoin",
    "SOL": "solana",
    "XRP": "ripple",
    "ADA": "cardano",
    "DOGE": "dogecoin",
    "AVAX": "avalanche-2",
}

LONG = "LONG"
SHORT = "SHORT"


@dataclass
class BacktestConfig:
    initial_capital: float = 1000.0
    commission_pct: float = 0.001
    slippage_pct: float = 0.0005
    spread_pct: float = 0.0003
    risk_per_trade_pct: float = 0.01
    min_volume: float | None = None


async def fetch_historical_ohlcv(symbol: str, days: int = 365) -> pd.DataFrame:
    """Fetch historical OHLC data from CoinGecko's free public endpoint."""
    import httpx

    coin_id = COINGECKO_IDS.get(symbol.upper(), symbol.lower())
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(
            f"{COINGECKO_BASE}/coins/{coin_id}/ohlc",
            params={"vs_currency": "usd", "days": str(days)},
        )
        response.raise_for_status()
        raw = response.json()

    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    df = df.astype(float)
    df.sort_index(inplace=True)
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add indicators using only rolling historical data up to each candle."""
    df = normalize_ohlcv(df)
    close = df["close"]
    high = df["high"]
    low = df["low"]

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
    else:
        df["macd"] = np.nan
        df["macd_signal"] = np.nan
        df["macd_hist"] = np.nan

    if len(df) >= 20:
        bb = BollingerBands(close=close, window=20, window_dev=2)
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_mid"] = bb.bollinger_mavg()
    else:
        df["bb_upper"] = np.nan
        df["bb_lower"] = np.nan
        df["bb_mid"] = np.nan

    return df


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Return a sorted OHLCV DataFrame with required numeric columns."""
    df = df.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    for column in ["open", "high", "low", "close"]:
        if column not in df.columns:
            raise ValueError(f"Missing required OHLC column: {column}")
        df[column] = df[column].astype(float)

    if "volume" in df.columns:
        df["volume"] = df["volume"].astype(float)

    return df


def normalize_pct(value: Any, default: float | None = None) -> float | None:
    """Accept either decimal fractions (0.05) or human pct values (5)."""
    if value is None:
        return default
    pct = float(value)
    return pct / 100 if pct > 1 else pct


def evaluate_condition(row: pd.Series, condition: dict) -> bool:
    """Evaluate one strategy condition against a closed candle."""
    indicator = condition.get("indicator", "").lower()
    operator = condition.get("operator", "")
    expected = float(condition.get("value", 0))

    aliases = {
        "price": "close",
        "close": "close",
        "open": "open",
        "high": "high",
        "low": "low",
        "volume": "volume",
        "rsi": "rsi",
        "atr": "atr",
        "macd": "macd",
        "macd_signal": "macd_signal",
        "macd_hist": "macd_hist",
        "bb_lower": "bb_lower",
        "bb_upper": "bb_upper",
        "bb_mid": "bb_mid",
    }
    column = aliases.get(indicator)
    if not column or column not in row:
        return False

    actual = row.get(column)
    if actual is None or pd.isna(actual):
        return False

    operations = {
        "<": actual < expected,
        ">": actual > expected,
        "<=": actual <= expected,
        ">=": actual >= expected,
        "==": actual == expected,
        "!=": actual != expected,
    }
    return operations.get(operator, False)


def conditions_met(row: pd.Series, conditions: list[dict]) -> bool:
    return bool(conditions) and all(evaluate_condition(row, condition) for condition in conditions)


def passes_liquidity_filter(row: pd.Series, strategy: dict, config: BacktestConfig) -> bool:
    min_volume = strategy.get("min_volume", config.min_volume)
    if min_volume is None or "volume" not in row or pd.isna(row.get("volume")):
        return True
    return float(row["volume"]) >= float(min_volume)


def signal_from_candle(row: pd.Series, strategy: dict, config: BacktestConfig) -> str | None:
    """Generate a signal using only the current closed candle."""
    if not passes_liquidity_filter(row, strategy, config):
        return None

    mode = str(strategy.get("side", strategy.get("direction", LONG))).upper()
    long_conditions = strategy.get("long_entry_conditions") or strategy.get("entry_conditions", [])
    short_conditions = strategy.get("short_entry_conditions", [])

    if mode in {"LONG", "BOTH"} and conditions_met(row, long_conditions):
        return LONG
    if mode in {"SHORT", "BOTH"}:
        if short_conditions and conditions_met(row, short_conditions):
            return SHORT
        if mode == "SHORT" and conditions_met(row, long_conditions):
            return SHORT
    return None


def stop_take_for_entry(entry_price: float, side: str, signal_row: pd.Series, strategy: dict) -> tuple[float, float | None]:
    """Calculate stop-loss and take-profit from percent or ATR settings."""
    sl_pct = normalize_pct(strategy.get("stop_loss_pct"), 0.02)
    tp_pct = normalize_pct(strategy.get("take_profit_pct"), None)
    atr = float(signal_row.get("atr", np.nan)) if "atr" in signal_row else np.nan

    atr_stop_multiplier = strategy.get("atr_stop_multiplier")
    atr_take_profit_multiplier = strategy.get("atr_take_profit_multiplier")

    if atr_stop_multiplier is not None and not np.isnan(atr) and atr > 0:
        stop_distance = atr * float(atr_stop_multiplier)
    else:
        stop_distance = entry_price * float(sl_pct)

    if atr_take_profit_multiplier is not None and not np.isnan(atr) and atr > 0:
        take_distance = atr * float(atr_take_profit_multiplier)
    elif tp_pct is not None:
        take_distance = entry_price * float(tp_pct)
    else:
        take_distance = None

    if side == LONG:
        stop_loss = entry_price - stop_distance
        take_profit = entry_price + take_distance if take_distance is not None else None
    else:
        stop_loss = entry_price + stop_distance
        take_profit = entry_price - take_distance if take_distance is not None else None

    return stop_loss, take_profit


def execution_price(raw_price: float, side: str, action: str, config: BacktestConfig) -> float:
    """Apply slippage and half-spread to entry/exit prices."""
    half_spread = config.spread_pct / 2
    if (side == LONG and action == "entry") or (side == SHORT and action == "exit"):
        return raw_price * (1 + config.slippage_pct + half_spread)
    return raw_price * (1 - config.slippage_pct - half_spread)


def calculate_position_size(capital: float, entry_price: float, stop_loss: float, config: BacktestConfig) -> float:
    """Size position by risk and cap notional exposure at available equity."""
    stop_distance = abs(entry_price - stop_loss)
    if capital <= 0 or entry_price <= 0 or stop_distance <= 0:
        return 0.0

    risk_amount = capital * config.risk_per_trade_pct
    quantity_by_risk = risk_amount / stop_distance
    quantity_by_capital = capital / entry_price
    return max(0.0, min(quantity_by_risk, quantity_by_capital))


def estimate_slippage_cost(quantity: float, reference_price: float, config: BacktestConfig) -> float:
    return abs(quantity * reference_price * config.slippage_pct)


def entry_position(
    ts: pd.Timestamp,
    candle: pd.Series,
    signal_row: pd.Series,
    side: str,
    strategy: dict,
    capital: float,
    config: BacktestConfig,
) -> dict | None:
    raw_entry = float(candle["open"])
    entry = execution_price(raw_entry, side, "entry", config)
    stop_loss, take_profit = stop_take_for_entry(entry, side, signal_row, strategy)
    quantity = calculate_position_size(capital, entry, stop_loss, config)
    if quantity <= 0:
        return None

    notional = quantity * entry
    entry_fee = notional * config.commission_pct
    entry_slippage = estimate_slippage_cost(quantity, raw_entry, config)

    return {
        "entry_time": ts,
        "side": side,
        "entry_price": entry,
        "raw_entry_price": raw_entry,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "quantity": quantity,
        "entry_fee": entry_fee,
        "slippage_cost": entry_slippage,
        "signal_time": signal_row.name,
    }


def check_exit(candle: pd.Series, position: dict, strategy: dict) -> tuple[str, float] | None:
    """Check stop/take-profit with high/low. If both hit, assume stop first."""
    side = position["side"]
    high = float(candle["high"])
    low = float(candle["low"])
    stop_loss = position["stop_loss"]
    take_profit = position.get("take_profit")

    if side == LONG:
        if low <= stop_loss:
            return "STOP_LOSS", stop_loss
        if take_profit is not None and high >= take_profit:
            return "TAKE_PROFIT", take_profit
    else:
        if high >= stop_loss:
            return "STOP_LOSS", stop_loss
        if take_profit is not None and low <= take_profit:
            return "TAKE_PROFIT", take_profit

    exit_conditions = strategy.get("exit_conditions", [])
    if exit_conditions and conditions_met(candle, exit_conditions):
        return "EXIT_SIGNAL", float(candle["close"])

    return None


def close_position(
    ts: pd.Timestamp,
    raw_exit_price: float,
    reason: str,
    position: dict,
    config: BacktestConfig,
) -> dict:
    side = position["side"]
    exit_price = execution_price(raw_exit_price, side, "exit", config)
    quantity = position["quantity"]
    entry_price = position["entry_price"]

    if side == LONG:
        gross_pnl = (exit_price - entry_price) * quantity
        pnl_pct = (exit_price - entry_price) / entry_price * 100
    else:
        gross_pnl = (entry_price - exit_price) * quantity
        pnl_pct = (entry_price - exit_price) / entry_price * 100

    exit_fee = quantity * exit_price * config.commission_pct
    fees = position["entry_fee"] + exit_fee
    exit_slippage = estimate_slippage_cost(quantity, raw_exit_price, config)
    slippage_cost = position["slippage_cost"] + exit_slippage
    pnl = gross_pnl - fees

    return {
        "entry_time": position["entry_time"].isoformat(),
        "exit_time": ts.isoformat(),
        "side": side,
        "entry_price": round(entry_price, 8),
        "exit_price": round(exit_price, 8),
        "stop_loss": round(position["stop_loss"], 8),
        "take_profit": round(position["take_profit"], 8) if position.get("take_profit") is not None else None,
        "quantity": round(quantity, 10),
        "fees": round(fees, 8),
        "slippage_cost": round(slippage_cost, 8),
        "pnl": round(pnl, 8),
        "pnl_pct": round(pnl_pct, 6),
        "exit_reason": reason,
    }


def calculate_max_drawdown(equity_values: list[float]) -> float:
    peak = equity_values[0] if equity_values else 0
    max_drawdown = 0.0
    for equity in equity_values:
        peak = max(peak, equity)
        if peak > 0:
            max_drawdown = max(max_drawdown, (peak - equity) / peak * 100)
    return max_drawdown


def calculate_metrics(
    trades: list[dict],
    equity_curve: list[dict],
    initial_capital: float,
    df: pd.DataFrame,
) -> dict:
    final_capital = equity_curve[-1]["capital"] if equity_curve else initial_capital
    returns = [trade["pnl_pct"] for trade in trades]
    wins = [trade for trade in trades if trade["pnl"] > 0]
    losses = [trade for trade in trades if trade["pnl"] < 0]
    gross_profit = sum(trade["pnl"] for trade in wins)
    gross_loss = abs(sum(trade["pnl"] for trade in losses))
    equity_values = [point["capital"] for point in equity_curve] or [initial_capital]

    number_of_trades = len(trades)
    average_trade_return = float(np.mean(returns)) if returns else 0.0
    average_win = float(np.mean([trade["pnl_pct"] for trade in wins])) if wins else 0.0
    average_loss = float(np.mean([trade["pnl_pct"] for trade in losses])) if losses else 0.0
    win_rate = len(wins) / number_of_trades * 100 if number_of_trades else 0.0
    loss_rate = len(losses) / number_of_trades * 100 if number_of_trades else 0.0

    buy_and_hold = (df.iloc[-1]["close"] - df.iloc[0]["close"]) / df.iloc[0]["close"] * 100 if len(df) > 1 else 0.0
    sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252)) if len(returns) > 1 and np.std(returns) > 0 else 0.0
    expectancy = (win_rate / 100 * average_win) + (loss_rate / 100 * average_loss)

    return {
        "final_capital": round(final_capital, 8),
        "total_return_pct": round((final_capital - initial_capital) / initial_capital * 100, 6),
        "buy_and_hold_return_pct": round(float(buy_and_hold), 6),
        "bh_return_pct": round(float(buy_and_hold), 6),
        "number_of_trades": number_of_trades,
        "total_trades": number_of_trades,
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "win_rate": round(win_rate, 6),
        "average_trade_return": round(average_trade_return, 6),
        "average_win": round(average_win, 6),
        "average_loss": round(average_loss, 6),
        "profit_factor": round(gross_profit / gross_loss, 6) if gross_loss > 0 else (round(gross_profit, 6) if gross_profit > 0 else 0),
        "max_drawdown": round(calculate_max_drawdown(equity_values), 6),
        "sharpe_ratio": round(sharpe, 6),
        "expectancy": round(expectancy, 6),
        "fees_total": round(sum(trade["fees"] for trade in trades), 8),
        "slippage_total": round(sum(trade["slippage_cost"] for trade in trades), 8),
    }


def simulate_backtest(
    df: pd.DataFrame,
    strategy: dict,
    initial_capital: float = 1000.0,
) -> dict:
    """Run a deterministic V2 backtest against an already loaded OHLCV DataFrame."""
    commission_pct = normalize_pct(strategy.get("commission_pct"), 0.001)
    slippage_pct = normalize_pct(strategy.get("slippage_pct"), 0.0005)
    spread_pct = normalize_pct(strategy.get("spread_pct"), 0.0003)
    risk_per_trade_pct = normalize_pct(strategy.get("risk_per_trade_pct"), 0.01)
    config = BacktestConfig(
        initial_capital=initial_capital,
        commission_pct=0.001 if commission_pct is None else commission_pct,
        slippage_pct=0.0005 if slippage_pct is None else slippage_pct,
        spread_pct=0.0003 if spread_pct is None else spread_pct,
        risk_per_trade_pct=0.01 if risk_per_trade_pct is None else risk_per_trade_pct,
        min_volume=strategy.get("min_volume"),
    )
    df = add_indicators(df)
    if df.empty or len(df) < 2:
        return {"error": "Not enough data for backtest"}

    capital = initial_capital
    position = None
    pending_signal: dict | None = None
    trades: list[dict] = []
    equity_curve: list[dict] = []

    for ts, row in df.iterrows():
        if position is None and pending_signal is not None:
            position = entry_position(
                ts=ts,
                candle=row,
                signal_row=pending_signal["row"],
                side=pending_signal["side"],
                strategy=strategy,
                capital=capital,
                config=config,
            )
            pending_signal = None

        if position is not None:
            exit_check = check_exit(row, position, strategy)
            if exit_check:
                reason, raw_exit = exit_check
                trade = close_position(ts, raw_exit, reason, position, config)
                trades.append(trade)
                capital += trade["pnl"]
                position = None

        market_equity = capital
        if position is not None:
            close_price = float(row["close"])
            mark_exit = execution_price(close_price, position["side"], "exit", config)
            if position["side"] == LONG:
                market_equity += (mark_exit - position["entry_price"]) * position["quantity"] - position["entry_fee"]
            else:
                market_equity += (position["entry_price"] - mark_exit) * position["quantity"] - position["entry_fee"]

        equity_curve.append({"date": ts.isoformat(), "capital": round(market_equity, 8)})

        if position is None and pending_signal is None:
            side = signal_from_candle(row, strategy, config)
            if side:
                pending_signal = {"side": side, "row": row}

    if position is not None:
        ts = df.index[-1]
        row = df.iloc[-1]
        trade = close_position(ts, float(row["close"]), "END_OF_PERIOD", position, config)
        trades.append(trade)
        capital += trade["pnl"]
        equity_curve.append({"date": ts.isoformat(), "capital": round(capital, 8)})

    metrics = calculate_metrics(trades, equity_curve, initial_capital, df)
    return {
        **metrics,
        "initial_capital": initial_capital,
        "trades": trades,
        "equity_curve": equity_curve,
        "assumptions": {
            "execution_delay_candles": 1,
            "same_candle_tp_sl_order": "stop_loss_first_when_both_touched",
            "commission_pct": config.commission_pct,
            "slippage_pct": config.slippage_pct,
            "spread_pct": config.spread_pct,
            "risk_per_trade_pct": config.risk_per_trade_pct,
            "liquidity_filter": config.min_volume,
        },
    }


async def run_backtest(
    symbol: str,
    strategy: dict,
    date_from: str,
    date_to: str,
    initial_capital: float = 1000.0,
    timeframe: str = "1d",
) -> dict:
    """Fetch data, run Backtest Engine V2, and return metrics plus trade log."""
    days = (datetime.fromisoformat(date_to) - datetime.fromisoformat(date_from)).days + 30
    df = await fetch_historical_ohlcv(symbol, days=min(max(days, 2), 365))
    df = normalize_ohlcv(df)
    df = df[date_from:date_to]

    if df.empty:
        return {"error": "No data for the selected range"}

    result = simulate_backtest(df, strategy, initial_capital)
    result.update({
        "symbol": symbol.upper(),
        "date_from": date_from,
        "date_to": date_to,
        "timeframe": timeframe,
        "engine_version": "v2",
    })
    return result
