# backend/tools/backtest_tool.py
"""
Motor de backtesting — aplica estrategias a datos históricos de CoinGecko.
Evita look-ahead bias procesando vela a vela en orden cronológico estricto.
"""
import httpx
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Any
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands


COINGECKO_BASE = "https://api.coingecko.com/api/v3"

COINGECKO_IDS = {
    "BTC": "bitcoin", "ETH": "ethereum", "BNB": "binancecoin",
    "SOL": "solana",  "XRP": "ripple",   "ADA": "cardano",
    "DOGE": "dogecoin", "AVAX": "avalanche-2",
}


async def fetch_historical_ohlcv(symbol: str, days: int = 365) -> pd.DataFrame:
    """Datos OHLCV históricos de CoinGecko (gratis, sin auth)."""
    coin_id = COINGECKO_IDS.get(symbol.upper(), symbol.lower())
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(
            f"{COINGECKO_BASE}/coins/{coin_id}/ohlc",
            params={"vs_currency": "usd", "days": str(days)},
        )
        r.raise_for_status()
        raw = r.json()

    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    df = df.astype(float)
    df.sort_index(inplace=True)  # cronológico estricto
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega RSI, MACD y BB al DataFrame. CRÍTICO: calculados solo con datos pasados."""
    close = df["close"]
    df = df.copy()
    df["rsi"]         = RSIIndicator(close=close, window=14).rsi()
    df["macd"]        = MACD(close=close).macd()
    df["macd_signal"] = MACD(close=close).macd_signal()
    df["macd_hist"]   = MACD(close=close).macd_diff()
    bb = BollingerBands(close=close, window=20, window_dev=2)
    df["bb_upper"]    = bb.bollinger_hband()
    df["bb_lower"]    = bb.bollinger_lband()
    df["bb_mid"]      = bb.bollinger_mavg()
    return df.dropna()


def evaluate_condition(row: pd.Series, condition: dict) -> bool:
    """
    Evalúa una condición de la estrategia contra una vela.
    condition = {"indicator": "rsi", "operator": "<", "value": 30}
    """
    indicator = condition.get("indicator", "").lower()
    operator  = condition.get("operator", "")
    value     = float(condition.get("value", 0))

    actual = None
    if indicator == "rsi":          actual = row.get("rsi")
    elif indicator == "macd_hist":  actual = row.get("macd_hist")
    elif indicator == "price":      actual = row.get("close")
    elif indicator == "bb_lower":   actual = row.get("bb_lower")
    elif indicator == "bb_upper":   actual = row.get("bb_upper")

    if actual is None or pd.isna(actual):
        return False

    ops = {"<": actual < value, ">": actual > value,
           "<=": actual <= value, ">=": actual >= value, "==": actual == value}
    return ops.get(operator, False)


async def run_backtest(
    symbol: str,
    strategy: dict,
    date_from: str,
    date_to: str,
    initial_capital: float = 1000.0,
    timeframe: str = "1d",
) -> dict:
    """
    Corre el backtest y retorna métricas completas.

    strategy = {
        "entry_conditions": [{"indicator": "rsi", "operator": "<", "value": 30}],
        "exit_conditions":  [{"indicator": "rsi", "operator": ">", "value": 70}],
        "stop_loss_pct": 5,    # optional: % desde precio de entrada
        "take_profit_pct": 10, # optional
    }
    """
    # 1. Datos históricos
    days = (datetime.fromisoformat(date_to) - datetime.fromisoformat(date_from)).days + 30
    df = await fetch_historical_ohlcv(symbol, days=min(days, 365))

    # 2. Indicadores — calculados sobre todo el historial ANTES del filtro de fecha
    #    Esto evita look-ahead bias: los indicadores del día X solo usan datos hasta X
    df = add_indicators(df)

    # 3. Filtrar rango de fechas DESPUÉS de calcular indicadores
    df = df[date_from:date_to]

    if df.empty:
        return {"error": "No data for the selected range"}

    # 4. Simulación vela a vela
    capital     = initial_capital
    position    = None   # None | {"entry_price", "size", "stop_loss", "take_profit"}
    trades      = []
    equity_curve = []

    entry_conds  = strategy.get("entry_conditions", [])
    exit_conds   = strategy.get("exit_conditions", [])
    sl_pct       = strategy.get("stop_loss_pct")
    tp_pct       = strategy.get("take_profit_pct")

    for ts, row in df.iterrows():
        price = row["close"]

        # Chequear stop-loss / take-profit primero (si hay posición)
        if position:
            hit_sl = sl_pct and price <= position["stop_loss"]
            hit_tp = tp_pct and price >= position["take_profit"]
            exit_signal = all(evaluate_condition(row, c) for c in exit_conds) if exit_conds else False

            if hit_sl or hit_tp or exit_signal:
                pnl_pct  = (price - position["entry_price"]) / position["entry_price"] * 100
                pnl_abs  = capital * position["size"] * (pnl_pct / 100)
                capital += pnl_abs
                trades.append({
                    "entry_time":  position["entry_time"].isoformat(),
                    "exit_time":   ts.isoformat(),
                    "entry_price": round(position["entry_price"], 4),
                    "exit_price":  round(price, 4),
                    "pnl_pct":     round(pnl_pct, 4),
                    "pnl_abs":     round(pnl_abs, 4),
                    "reason":      "STOP_LOSS" if hit_sl else "TAKE_PROFIT" if hit_tp else "EXIT_SIGNAL",
                })
                position = None

        # Señal de entrada (solo si no estamos en posición — no pyramiding)
        elif entry_conds and all(evaluate_condition(row, c) for c in entry_conds):
            sl_price = price * (1 - sl_pct / 100) if sl_pct else None
            tp_price = price * (1 + tp_pct / 100) if tp_pct else None
            position = {
                "entry_price":  price,
                "entry_time":   ts,
                "size":         1.0,   # 100% del capital disponible
                "stop_loss":    sl_price,
                "take_profit":  tp_price,
            }

        equity_curve.append({"date": ts.isoformat(), "capital": round(capital, 2)})

    # Cierre forzado al final si queda posición abierta
    if position:
        last_price = df.iloc[-1]["close"]
        pnl_pct = (last_price - position["entry_price"]) / position["entry_price"] * 100
        trades.append({
            "entry_time":  position["entry_time"].isoformat(),
            "exit_time":   df.index[-1].isoformat(),
            "entry_price": round(position["entry_price"], 4),
            "exit_price":  round(last_price, 4),
            "pnl_pct":     round(pnl_pct, 4),
            "reason":      "END_OF_PERIOD",
        })

    # 5. Métricas
    total_trades  = len(trades)
    wins          = [t for t in trades if t.get("pnl_pct", 0) > 0]
    losses        = [t for t in trades if t.get("pnl_pct", 0) <= 0]
    total_return  = (capital - initial_capital) / initial_capital * 100

    # Buy-and-hold benchmark
    bh_return = (df.iloc[-1]["close"] - df.iloc[0]["close"]) / df.iloc[0]["close"] * 100

    # Max drawdown
    caps = [e["capital"] for e in equity_curve]
    peak = initial_capital
    max_dd = 0
    for c in caps:
        if c > peak:
            peak = c
        dd = (peak - c) / peak * 100
        if dd > max_dd:
            max_dd = dd

    # Sharpe ratio (simplificado, diario)
    returns_pct = [t.get("pnl_pct", 0) for t in trades]
    sharpe = (np.mean(returns_pct) / np.std(returns_pct) * np.sqrt(252)) if len(returns_pct) > 1 and np.std(returns_pct) > 0 else 0

    return {
        "symbol":           symbol.upper(),
        "date_from":        date_from,
        "date_to":          date_to,
        "initial_capital":  initial_capital,
        "final_capital":    round(capital, 2),
        "total_return_pct": round(total_return, 4),
        "bh_return_pct":    round(bh_return, 4),
        "total_trades":     total_trades,
        "winning_trades":   len(wins),
        "losing_trades":    len(losses),
        "win_rate":         round(len(wins) / total_trades * 100, 2) if total_trades else 0,
        "max_drawdown":     round(max_dd, 4),
        "sharpe_ratio":     round(float(sharpe), 4),
        "trades":           trades,
        "equity_curve":     equity_curve,
    }