import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.historical_data import fetch_binance_klines
from tools.historical_replay import run_historical_replay


async def main():
    try:
        candles = await fetch_binance_klines("BTC", "1h", limit=120)
        result = run_historical_replay(
            candles,
            symbol="BTC",
            timeframe="1h",
            strategy_mode="deterministic",
            horizon_candles=3,
            min_history=50,
            max_predictions=10,
        )
        print("Downloaded candles:", len(candles))
        print("Replay metrics:")
        for row in result.get("metrics", []):
            print(row)
    except Exception as exc:
        print("Real-market smoke failed. Check internet access, Binance availability, and rate limits.")
        print(f"Error: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
