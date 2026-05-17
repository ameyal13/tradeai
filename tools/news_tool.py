# backend/tools/news_tool.py
"""
Tool: fetch_news
Obtiene noticias de CryptoPanic + RSS feeds gratuitos.
Analiza sentimiento con el LLM (Groq, gratis).
"""
import httpx
import feedparser
import asyncio
import os
from datetime import datetime, timezone
from typing import Optional


CRYPTOPANIC_BASE = "https://cryptopanic.com/api/v1/posts/"

RSS_FEEDS = [
    {"name": "CoinDesk",    "url": "https://www.coindesk.com/arc/outboundfeeds/rss/"},
    {"name": "Decrypt",     "url": "https://decrypt.co/feed"},
    {"name": "The Block",   "url": "https://www.theblock.co/rss.xml"},
    {"name": "CoinTelegraph","url": "https://cointelegraph.com/rss"},
]

SYMBOL_KEYWORDS = {
    "BTC":  ["bitcoin", "btc", "satoshi"],
    "ETH":  ["ethereum", "eth", "ether", "vitalik"],
    "SOL":  ["solana", "sol"],
    "BNB":  ["binance", "bnb"],
    "XRP":  ["ripple", "xrp"],
    "ADA":  ["cardano", "ada"],
    "DOGE": ["dogecoin", "doge"],
}


def detect_symbols(text: str) -> list[str]:
    """Detecta qué activos menciona un texto."""
    text_lower = text.lower()
    found = []
    for sym, keywords in SYMBOL_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            found.append(sym)
    return found


async def fetch_cryptopanic(symbols: Optional[str] = None, limit: int = 20) -> list:
    """Noticias de CryptoPanic (requiere API key gratuita)."""
    api_key = os.getenv("CRYPTOPANIC_API_KEY")
    if not api_key:
        return []

    params = {
        "auth_token": api_key,
        "kind": "news",
        "public": "true",
        "regions": "en",
    }
    if symbols:
        params["currencies"] = symbols

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(CRYPTOPANIC_BASE, params=params)
            r.raise_for_status()
            data = r.json()

        items = []
        for post in data.get("results", [])[:limit]:
            items.append({
                "source":       "CryptoPanic",
                "title":        post.get("title", ""),
                "url":          post.get("url", ""),
                "published_at": post.get("published_at", ""),
                "symbols":      [c["code"] for c in post.get("currencies", [])],
            })
        return items
    except Exception:
        return []


async def fetch_rss_feeds() -> list:
    """Parsea feeds RSS de fuentes cripto reconocidas."""
    items = []
    for feed_info in RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_info["url"])
            for entry in feed.entries[:5]:
                title = getattr(entry, "title", "")
                url   = getattr(entry, "link", "")
                pub   = getattr(entry, "published", str(datetime.now(timezone.utc)))
                if title and url:
                    items.append({
                        "source":       feed_info["name"],
                        "title":        title,
                        "url":          url,
                        "published_at": pub,
                        "symbols":      detect_symbols(title),
                    })
        except Exception:
            continue
    return items


async def get_news(symbol: Optional[str] = None, limit: int = 15) -> list:
    """
    Tool principal: agrega noticias de todas las fuentes.
    Usado por el agente LangGraph.
    """
    cp_task  = fetch_cryptopanic(symbols=symbol, limit=limit)
    rss_task = asyncio.to_thread(lambda: asyncio.run(fetch_rss_feeds()) if False else fetch_rss_feeds())

    cp_news, rss_news = await asyncio.gather(cp_task, fetch_rss_feeds())

    all_news = cp_news + rss_news

    # Dedup por URL
    seen_urls = set()
    unique = []
    for item in all_news:
        if item["url"] not in seen_urls:
            seen_urls.add(item["url"])
            unique.append(item)

    # Filtrar por símbolo si se especificó
    if symbol:
        sym_upper = symbol.upper()
        filtered = [n for n in unique if sym_upper in n.get("symbols", [])]
        unique = filtered if filtered else unique  # fallback to all if none found

    return unique[:limit]