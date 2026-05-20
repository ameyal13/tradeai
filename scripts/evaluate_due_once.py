"""Evaluate pending prediction journal entries once.

This script is intended for local/manual runs. It uses Supabase when backend
environment variables are configured, otherwise the PredictionStore file
fallback is used.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import evaluate_due_predictions_once


async def main() -> None:
    result = await evaluate_due_predictions_once()
    print("Evaluate due predictions")
    print(f"found: {result['found']}")
    print(f"not_due: {result['not_due']}")
    print(f"evaluated: {result['evaluated']}")
    print(f"invalid: {result['invalid']}")
    print(f"errors: {result['errors_count']}")
    if result["errors_count"]:
        for error in result["details"]["errors"]:
            print(f"- {error['prediction_id']}: {error['error']}")


if __name__ == "__main__":
    asyncio.run(main())
