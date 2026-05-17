# backend/tools/market_tool.py
"""
Tool: fetch_market_data
Obtiene precios actuales e históricos de Binance y CoinGecko (ambos gratuitos).
"""
import httpx
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
from typing import Optional
import asyncio


BINANCE_BASE = "https://api.binance.com/api/v3"
COINGECKO_BASE = "https://api.coingecko.com/api/v3"

SYMBOL_MAP = {
    "BTC": "BTCUSDT", "ETH": "ETHUSDT", "BNB": "BNBUSDT",
    "SOL": "SOLUSDT", "XRP": "XRPUSDT", "ADA": "ADAUSDT",
    "DOGE": "DOGEUSDT", "AVAX": "AVAXUSDT", "LINK": "LINKUSDT",
    "DOT": "DOTUSDT",
}

COINGECKO_IDS = {
    "BTC": "bitcoin", "ETH": "ethereum", "BNB": "binancecoin",
    "SOL": "solana", "XRP": "ripple", "ADA": "cardano",
    "DOGE": "dogecoin", "AVAX": "avalanche-2", "LINK": "chainlink",
    "DOT": "polkadot",
}


async def get_current_price(symbol: str) -> dict:
    """Precio actual de Binance (gratis, sin auth)."""
    ticker = SYMBOL_MAP.get(symbol.upper(), symbol.upper() + "USDT")
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            r = await client.get(f"{BINANCE_BASE}/ticker/24hr", params={"symbol": ticker})
            r.raise_for_status()
            d = r.json()
            return {
                "symbol": symbol.upper(),
                "price": float(d["lastPrice"]),
                "change_24h": float(d["priceChangePercent"]),
                "volume_24h": float(d["quoteVolume"]),
                "high_24h": float(d["highPrice"]),
                "low_24h": float(d["lowPrice"]),
                "source": "binance",
            }
        except Exception as e:
            return {"error": str(e), "symbol": symbol}


async def get_klines(symbol: str, interval: str = "1h", limit: int = 200) -> pd.DataFrame:
    """
    Velas OHLCV de Binance.
    interval: 1m, 5m, 15m, 1h, 4h, 1d
    """
    ticker = SYMBOL_MAP.get(symbol.upper(), symbol.upper() + "USDT")
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(
            f"{BINANCE_BASE}/klines",
            params={"symbol": ticker, "interval": interval, "limit": limit},
        )
        r.raise_for_status()
        raw = r.json()

    df = pd.DataFrame(raw, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore",
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df.set_index("open_time", inplace=True)
    return df[["open", "high", "low", "close", "volume"]]


def compute_indicators(df: pd.DataFrame) -> dict:
    """Calcula RSI, MACD y Bollinger Bands sobre el DataFrame de velas."""
    close = df["close"]

    rsi = RSIIndicator(close=close, window=14)
    macd_ind = MACD(close=close)
    bb = BollingerBands(close=close, window=20, window_dev=2)

    last = df.iloc[-1]
    return {
        "price":        round(float(last["close"]), 6),
        "rsi":          round(float(rsi.rsi().iloc[-1]), 2),
        "macd":         round(float(macd_ind.macd().iloc[-1]), 6),
        "macd_signal":  round(float(macd_ind.macd_signal().iloc[-1]), 6),
        "macd_hist":    round(float(macd_ind.macd_diff().iloc[-1]), 6),
        "bb_upper":     round(float(bb.bollinger_hband().iloc[-1]), 6),
        "bb_middle":    round(float(bb.bollinger_mavg().iloc[-1]), 6),
        "bb_lower":     round(float(bb.bollinger_lband().iloc[-1]), 6),
        "volume":       round(float(last["volume"]), 2),
    }


async def get_market_analysis(symbol: str, interval: str = "1h") -> dict:
    """
    Tool principal: precio actual + indicadores técnicos completos.
    Usado por el agente LangGraph.
    """
    try:
        price_task = get_current_price(symbol)
        klines_task = get_klines(symbol, interval=interval, limit=200)
        price_data, df = await asyncio.gather(price_task, klines_task)

        indicators = compute_indicators(df)

        # Últimas 50 velas para el frontend
        candles = df.tail(50).reset_index()
        candles_list = [
            {
                "time":   int(row["open_time"].timestamp()),
                "open":   round(row["open"], 6),
                "high":   round(row["high"], 6),
                "low":    round(row["low"], 6),
                "close":  round(row["close"], 6),
                "volume": round(row["volume"], 2),
            }
            for _, row in candles.iterrows()
        ]

        return {
            "symbol":     symbol.upper(),
            "interval":   interval,
            "price_data": price_data,
            "indicators": indicators,
            "candles":    candles_list,
            "analysis": {
                "rsi_signal":  "OVERSOLD" if indicators["rsi"] < 30 else "OVERBOUGHT" if indicators["rsi"] > 70 else "NEUTRAL",
                "macd_signal": "BULLISH"  if indicators["macd_hist"] > 0 else "BEARISH",
                "bb_position": (
                    "BELOW_LOWER" if indicators["price"] < indicators["bb_lower"]
                    else "ABOVE_UPPER" if indicators["price"] > indicators["bb_upper"]
                    else "INSIDE_BANDS"
                ),
            },
        }
    except Exception as e:
        return {"error": str(e), "symbol": symbol}


async def get_top_cryptos(limit: int = 10) -> list:
    """Top N cryptos por market cap via CoinGecko (gratis)."""
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(
            f"{COINGECKO_BASE}/coins/markets",
            params={
                "vs_currency": "usd",
                "order": "market_cap_desc",
                "per_page": limit,
                "page": 1,
                "sparkline": False,
            },
        )
        r.raise_for_status()
        data = r.json()

    return [
        {
            "symbol":       c["symbol"].upper(),
            "name":         c["name"],
            "price":        c["current_price"],
            "change_24h":   c["price_change_percentage_24h"],
            "market_cap":   c["market_cap"],
            "volume_24h":   c["total_volume"],
            "image":        c["image"],
        }
        for c in data
    ]