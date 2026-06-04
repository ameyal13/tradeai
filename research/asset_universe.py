"""Local crypto asset universe helpers for research/shadow workflows."""
from __future__ import annotations


CRYPTO_UNIVERSE = [
    "BTC",
    "ETH",
    "SOL",
    "BNB",
    "XRP",
    "ADA",
    "AVAX",
    "LINK",
    "DOGE",
    "MATIC",
]


def normalize_crypto_symbols(symbols: list[str] | str | None = None) -> list[str]:
    """Return uppercase crypto symbols from a CLI list or comma string."""
    if symbols is None:
        return list(CRYPTO_UNIVERSE)
    if isinstance(symbols, str):
        raw = symbols.replace(",", " ").split()
    else:
        raw = []
        for item in symbols:
            raw.extend(str(item).replace(",", " ").split())
    cleaned = []
    for symbol in raw:
        value = symbol.strip().upper()
        if value:
            cleaned.append(value)
    return cleaned


def crypto_universe(symbols: list[str] | str | None = None) -> list[str]:
    """Return crypto symbols only; equities are intentionally unsupported here."""
    return normalize_crypto_symbols(symbols)
