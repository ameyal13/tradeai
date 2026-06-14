"""Bounded optional agent review models for shadow signals.

The agent layer may add context/risk notes only. It cannot change side, entry,
SL, TP, or create trades. If no provider is configured, this module returns a
neutral APPROVE review without contacting any LLM.
"""
from __future__ import annotations

import os
from typing import Any, Literal

from pydantic import BaseModel, Field


ReviewStatus = Literal["APPROVE", "CAUTION", "BLOCK"]


class SignalReviewRequest(BaseModel):
    symbol: str
    timeframe: str
    side: Literal["LONG", "SHORT"]
    entry_price: float
    stop_loss: float | None = None
    take_profit: float | None = None
    confidence: float
    reasoning: str = ""
    research_only: bool = True
    news_context: dict[str, Any] | None = None
    market_context: dict[str, Any] | None = None


class SignalReviewResponse(BaseModel):
    review_status: ReviewStatus = "APPROVE"
    confidence_adjustment: int = Field(default=0, ge=-10, le=5)
    risk_flags: list[str] = Field(default_factory=list)
    context_summary: str = "No provider configured; no external context review was performed."
    reasoning: str = "Shadow signal allowed by deterministic local gating only."
    provider: str | None = None
    model: str | None = None
    can_modify_trade_levels: bool = False


def review_shadow_signal(request: SignalReviewRequest) -> SignalReviewResponse:
    """Return a bounded review.

    The review layer is allowed to caution/block based on structured context,
    but it never modifies side, entry, stop loss, take profit, or RR.
    """
    context_risk_flags: list[str] = []
    context_summary = "No external context review was performed."
    news_context = request.news_context or {}
    market_context = request.market_context or {}
    if news_context:
        risk_score = float(news_context.get("risk_score") or 0)
        context_risk_flags = [str(flag) for flag in news_context.get("risk_flags", [])]
        sentiment_score = news_context.get("sentiment_score")
        context_summary = (
            f"News context checked: risk_score={risk_score}, "
            f"sentiment_score={sentiment_score}, items={news_context.get('item_count', 0)}."
        )
        if risk_score >= 80:
            return SignalReviewResponse(
                review_status="BLOCK",
                confidence_adjustment=-10,
                risk_flags=context_risk_flags or ["high_news_risk"],
                context_summary=context_summary,
                reasoning="Structured news context indicates high event risk; shadow signal is blocked for research safety.",
            )
        if risk_score >= 45 or context_risk_flags:
            return SignalReviewResponse(
                review_status="CAUTION",
                confidence_adjustment=-5,
                risk_flags=context_risk_flags or ["elevated_news_risk"],
                context_summary=context_summary,
                reasoning="Structured news context indicates elevated risk; trade levels are unchanged.",
            )

    if market_context:
        market_status = str(market_context.get("context_status") or "APPROVE").upper()
        market_flags = [str(flag) for flag in market_context.get("risk_flags", [])]
        context_summary = str(market_context.get("context_summary") or "Market context checked.")
        if market_status == "BLOCK":
            return SignalReviewResponse(
                review_status="BLOCK",
                confidence_adjustment=-10,
                risk_flags=market_flags or ["market_context_block"],
                context_summary=context_summary,
                reasoning="Structured market context indicates stacked technical risk; shadow signal is blocked for research safety.",
            )
        if market_status == "CAUTION" or market_flags:
            return SignalReviewResponse(
                review_status="CAUTION",
                confidence_adjustment=-5,
                risk_flags=market_flags or ["market_context_caution"],
                context_summary=context_summary,
                reasoning="Structured market context indicates elevated technical risk; trade levels are unchanged.",
            )

    provider = os.getenv("AGENT_PROVIDER")
    model = os.getenv("AGENT_MODEL")
    if not provider or provider.lower() in {"none", "disabled"}:
        if news_context or market_context:
            return SignalReviewResponse(
                context_summary=context_summary,
                reasoning="Structured context did not trigger caution/block; no LLM provider was used.",
            )
        return SignalReviewResponse()
    return SignalReviewResponse(
        review_status="CAUTION",
        confidence_adjustment=0,
        risk_flags=["agent_provider_configured_but_live_review_not_implemented"],
        context_summary="Provider configured, but shadow v1 keeps review as a safe stub.",
        reasoning="The agent cannot modify side, entry, stop loss, or take profit.",
        provider=provider,
        model=model,
    )
