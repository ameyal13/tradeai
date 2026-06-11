"""Structured news context for shadow-signal review.

This module is research-only. It does not generate trades and does not rely on
LLM output. It turns free news/RSS items into bounded risk context that the
agent review layer can use to APPROVE/CAUTION/BLOCK an already generated
shadow signal.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, Field

from tools.news_tool import detect_symbols, get_news


NEGATIVE_TERMS = {
    "bankruptcy",
    "ban",
    "breach",
    "crackdown",
    "crash",
    "delist",
    "exploit",
    "hack",
    "halt",
    "investigation",
    "lawsuit",
    "liquidation",
    "outage",
    "probe",
    "regulation",
    "sec",
    "selloff",
    "sued",
    "vulnerability",
}

POSITIVE_TERMS = {
    "adoption",
    "approval",
    "bullish",
    "etf",
    "funding",
    "partnership",
    "rally",
    "recovery",
    "upgrade",
}

HIGH_RISK_TERMS = {
    "breach",
    "crash",
    "delist",
    "exploit",
    "hack",
    "halt",
    "lawsuit",
    "outage",
    "sec",
}


class NewsContextItem(BaseModel):
    source: str = ""
    title: str = ""
    url: str = ""
    published_at: str | None = None
    symbols: list[str] = Field(default_factory=list)
    relevance_score: float = 0.0
    sentiment_score: float = 0.0
    risk_flags: list[str] = Field(default_factory=list)


class NewsContext(BaseModel):
    symbol: str
    fetched_at: str
    provider_status: str = "ok"
    item_count: int = 0
    relevant_item_count: int = 0
    sentiment_score: float = 0.0
    risk_score: float = 0.0
    risk_flags: list[str] = Field(default_factory=list)
    items: list[NewsContextItem] = Field(default_factory=list)


def _term_hits(text: str, terms: set[str]) -> list[str]:
    lower = text.lower()
    return sorted(term for term in terms if term in lower)


def score_news_item(item: dict[str, Any], symbol: str) -> NewsContextItem:
    """Score one raw news item with deterministic, testable heuristics."""
    symbol = symbol.upper()
    title = str(item.get("title") or "")
    raw_symbols = [str(value).upper() for value in item.get("symbols", [])]
    symbols = raw_symbols or detect_symbols(title)
    title_lower = title.lower()
    symbol_relevant = symbol in symbols or symbol.lower() in title_lower
    market_wide = any(sym in symbols for sym in {"BTC", "ETH"}) or "crypto" in title_lower
    relevance_score = 1.0 if symbol_relevant else 0.45 if market_wide else 0.15

    negative_hits = _term_hits(title, NEGATIVE_TERMS)
    positive_hits = _term_hits(title, POSITIVE_TERMS)
    high_risk_hits = _term_hits(title, HIGH_RISK_TERMS)
    sentiment_score = max(-1.0, min(1.0, 0.25 * len(positive_hits) - 0.35 * len(negative_hits)))
    risk_flags = [f"news_{term}" for term in high_risk_hits]
    if negative_hits and symbol_relevant:
        risk_flags.append("negative_symbol_news")
    elif negative_hits and market_wide:
        risk_flags.append("negative_market_news")

    return NewsContextItem(
        source=str(item.get("source") or ""),
        title=title,
        url=str(item.get("url") or ""),
        published_at=item.get("published_at"),
        symbols=symbols,
        relevance_score=relevance_score,
        sentiment_score=round(sentiment_score, 6),
        risk_flags=sorted(set(risk_flags)),
    )


def summarize_news_context(symbol: str, items: list[NewsContextItem]) -> NewsContext:
    """Aggregate scored news items into one bounded review context."""
    relevant = [item for item in items if item.relevance_score >= 0.4]
    weighted = sum(item.sentiment_score * item.relevance_score for item in relevant)
    weight = sum(item.relevance_score for item in relevant) or 1.0
    sentiment_score = round(weighted / weight, 6) if relevant else 0.0
    risk_flags = sorted({flag for item in relevant for flag in item.risk_flags})
    direct_high_risk = any("negative_symbol_news" in item.risk_flags for item in relevant)
    risk_score = min(
        100.0,
        len(risk_flags) * 18.0
        + max(0.0, -sentiment_score) * 45.0
        + (20.0 if direct_high_risk else 0.0),
    )
    return NewsContext(
        symbol=symbol.upper(),
        fetched_at=datetime.now(timezone.utc).isoformat(),
        provider_status="ok",
        item_count=len(items),
        relevant_item_count=len(relevant),
        sentiment_score=sentiment_score,
        risk_score=round(risk_score, 6),
        risk_flags=risk_flags,
        items=items[:10],
    )


async def build_news_context(
    symbol: str,
    limit: int = 10,
    fetcher: Callable[[str, int], Awaitable[list[dict[str, Any]]]] | None = None,
) -> NewsContext:
    """Fetch and summarize news context for a symbol without crashing callers."""
    fetch = fetcher or get_news
    try:
        raw_items = await fetch(symbol.upper(), limit)
    except Exception as exc:  # noqa: BLE001 - context failures must not block research.
        return NewsContext(
            symbol=symbol.upper(),
            fetched_at=datetime.now(timezone.utc).isoformat(),
            provider_status=f"error:{type(exc).__name__}",
            risk_flags=["news_context_unavailable"],
        )
    scored = [score_news_item(item, symbol) for item in raw_items[:limit]]
    return summarize_news_context(symbol, scored)
