"""Sentiment features for Trading Copilot."""
from __future__ import annotations

from typing import Any

import httpx


FEAR_GREED_URL = "https://api.alternative.me/fng/?limit=1"
NEUTRAL_FEAR_GREED = {"value": 50, "classification": "Neutral", "timestamp": None}


def get_fear_greed_index() -> dict[str, Any]:
    try:
        response = httpx.get(FEAR_GREED_URL, timeout=10)
        response.raise_for_status()
        payload = response.json()
        row = (payload.get("data") or [])[0]
        return {
            "value": int(row["value"]),
            "classification": str(row["value_classification"]),
            "timestamp": row.get("timestamp"),
        }
    except Exception:
        return dict(NEUTRAL_FEAR_GREED)


def fear_greed_to_feature(value: int) -> float:
    if value <= 25:
        return 0.5
    if value < 45:
        return 0.2
    if value <= 55:
        return 0.0
    if value < 75:
        return -0.2
    return -0.5
