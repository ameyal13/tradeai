"""
Descarga datos históricos de Binance (gratis, sin API key para datos públicos).
Guarda como CSV en data/historical/{SYMBOL}_{INTERVAL}.csv

Uso:
  python scripts/download_historical_data.py --symbols BTC ETH SOL --interval 1h --days 730
"""
import argparse
import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.historical_data import fetch_binance_klines


def interval_to_minutes(interval: str) -> int:
    unit = interval[-1].lower()
    value = int(interval[:-1])
    if unit == "m":
        return value
    if unit == "h":
        return value * 60
    if unit == "d":
        return value * 1440
    raise ValueError(f"Intervalo no soportado: {interval}")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=["BTC", "ETH", "SOL"])
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--days", type=int, default=730)
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / "data" / "historical"
    output_dir.mkdir(parents=True, exist_ok=True)

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=args.days)
    limit = max(1, int(args.days * 24 * 60 / interval_to_minutes(args.interval)) + 5)

    for symbol in args.symbols:
        symbol = symbol.upper()
        print(f"Descargando {symbol} {args.interval} ({args.days} días)...")
        try:
            df = await fetch_binance_klines(
                symbol,
                args.interval,
                start_time=start_time,
                end_time=end_time,
                limit=limit,
            )
            output_path = output_dir / f"{symbol}_{args.interval}.csv"
            df.to_csv(output_path, index=False)
            print(f"{symbol}: {len(df)} velas guardadas en {output_path}")
        except Exception as exc:
            print(f"{symbol}: error descargando datos históricos: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
