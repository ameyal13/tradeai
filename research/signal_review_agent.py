"""Bounded optional agent review models for shadow signals.

The agent layer may add context/risk notes only. It cannot change side, entry,
SL, TP, or create trades. If no provider is configured, this module returns a
neutral APPROVE review without contacting any LLM.
"""
from __future__ import annotations

import os
from typing import Literal

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
    """Return a bounded review stub unless an agent provider is configured."""
    provider = os.getenv("AGENT_PROVIDER")
    model = os.getenv("AGENT_MODEL")
    if not provider or provider.lower() in {"none", "disabled"}:
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
