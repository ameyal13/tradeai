"""Generate one batch of local shadow/paper signals from research configs.

Research only. No exchange orders, no prediction_journal writes, no operational
trade signals. Watchlist configs require --allow-watchlist-shadow.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.asset_universe import normalize_crypto_symbols  # noqa: E402
from research.signal_review_agent import SignalReviewRequest, review_shadow_signal  # noqa: E402
from research.telegram_notifier import format_shadow_signal_opened, send_telegram_message  # noqa: E402
from scripts.run_historical_experiments import load_experiment_candles  # noqa: E402
from scripts.summarize_research_registry import load_latest_registry_records  # noqa: E402
from tools.shadow_signal_journal import (  # noqa: E402
    DEFAULT_SHADOW_JOURNAL_PATH,
    BLOCKED,
    ShadowSignalJournal,
    build_shadow_signal_from_strategy,
    cost_profile_for_config,
    horizon_minutes_from_candles,
)
from tools.strategy_signals import generate_strategy_signal_from_df  # noqa: E402


DEFAULT_REFINED_REGISTRY = PROJECT_ROOT / "reports" / "research_daemon" / "refined_registry.jsonl"
DEFAULT_GENERAL_REGISTRY = PROJECT_ROOT / "reports" / "research_daemon" / "registry.jsonl"


def registry_path_from_choice(choice: str) -> Path:
    if choice == "general":
        return DEFAULT_GENERAL_REGISTRY
    if choice == "refined":
        return DEFAULT_REFINED_REGISTRY
    return Path(choice)


def strategy_params_from_config(config: dict[str, Any]) -> dict[str, Any]:
    horizon_minutes = horizon_minutes_from_candles(int(config["horizon_candles"]), str(config["timeframe"]))
    return {
        "min_risk_reward": float(config.get("risk_reward", 1.5)),
        "atr_stop_multiplier": float(config.get("atr_stop_multiplier", 1.5)),
        "min_train_rows": int(config.get("min_train_rows", 120)),
        "probability_buy_threshold": float(config.get("buy_threshold", 0.58)),
        "probability_sell_threshold": 1 - float(config.get("sell_threshold", 0.58)),
        "buy_win_threshold": float(config.get("buy_threshold", 0.58)),
        "sell_win_threshold": float(config.get("sell_threshold", 0.58)),
        "use_trade_labels": True,
        "trade_label_scheme": config.get("trade_label_scheme", "expected_value_classification"),
        "horizon_minutes": horizon_minutes,
    }


def load_candidate_configs(
    registry_path: str | Path,
    symbols: list[str] | None = None,
    allow_watchlist_shadow: bool = False,
    min_classification: str = "stable_research_candidate",
) -> list[dict[str, Any]]:
    """Load selectable configs from a registry without using test metrics."""
    allowed = {"stable_research_candidate"}
    if allow_watchlist_shadow or min_classification == "unstable_watchlist":
        allowed.add("unstable_watchlist")
    if min_classification == "stable_research_candidate":
        allowed = {"stable_research_candidate"} if not allow_watchlist_shadow else allowed
    symbol_filter = {symbol.upper() for symbol in symbols} if symbols else None
    configs: list[dict[str, Any]] = []
    for row in load_latest_registry_records(registry_path):
        if row.get("status") != "completed":
            continue
        classification = row.get("classification")
        if classification not in allowed:
            continue
        config = dict(row.get("config") or {})
        if symbol_filter and str(config.get("symbol", "")).upper() not in symbol_filter:
            continue
        config["config_id"] = row.get("config_id") or config.get("config_id")
        config["_source_classification"] = classification
        configs.append(config)
    return configs


async def generate_shadow_signals_once(
    registry: str = "refined",
    symbols: list[str] | None = None,
    max_signals: int = 5,
    allow_watchlist_shadow: bool = False,
    notify_telegram: bool = False,
    dry_run: bool = False,
    min_classification: str = "stable_research_candidate",
    journal_path: str | Path = DEFAULT_SHADOW_JOURNAL_PATH,
    refresh_cache: bool = True,
) -> list[dict[str, Any]]:
    registry_path = registry_path_from_choice(registry)
    journal = ShadowSignalJournal(journal_path)
    source_registry = "registry" if registry == "general" else "refined_registry" if registry == "refined" else str(registry_path)
    selected_symbols = normalize_crypto_symbols(symbols) if symbols else None
    configs = load_candidate_configs(
        registry_path,
        symbols=selected_symbols,
        allow_watchlist_shadow=allow_watchlist_shadow,
        min_classification=min_classification,
    )
    rows: list[dict[str, Any]] = []
    opened = 0
    for config in configs:
        if opened >= max_signals:
            break
        classification = str(config.get("_source_classification"))
        config_id = str(config.get("config_id") or config.get("experiment_id"))
        symbol = str(config.get("symbol", "")).upper()
        timeframe = str(config.get("timeframe", "1h"))
        base_row = {
            "config_id": config_id,
            "symbol": symbol,
            "timeframe": timeframe,
            "classification": classification,
            "research_only": True,
        }
        if classification == "unstable_watchlist" and not allow_watchlist_shadow:
            rows.append({**base_row, "status": "skipped_watchlist_requires_flag"})
            continue
        if journal.has_open_signal(config_id, symbol, timeframe):
            rows.append({**base_row, "status": "skipped_duplicate_open"})
            continue
        try:
            loaded = await load_experiment_candles(
                symbol,
                timeframe,
                max_candles=int(config.get("max_candles", 1500)),
                use_cache=True,
                refresh_cache=bool(refresh_cache),
            )
            horizon_minutes = horizon_minutes_from_candles(int(config["horizon_candles"]), timeframe)
            params = strategy_params_from_config(config)
            params["use_sentiment"] = True
            signal = generate_strategy_signal_from_df(
                loaded["candles"],
                strategy_mode=str(config.get("strategy_mode", "xgboost")),
                provider="none",
                horizon_minutes=horizon_minutes,
                strategy_params=params,
            ).to_dict()
            costs = cost_profile_for_config(config)
            shadow = build_shadow_signal_from_strategy(
                config=config,
                source_registry=source_registry,
                classification=classification,
                signal=signal,
                costs=costs,
                watchlist_shadow=classification == "unstable_watchlist",
            )
            if shadow is None:
                rows.append({**base_row, "status": "skipped_hold", "signal": signal.get("signal")})
                continue
            review = review_shadow_signal(SignalReviewRequest(
                symbol=symbol,
                timeframe=timeframe,
                side=shadow["side"],
                entry_price=float(shadow["entry_price"]),
                stop_loss=shadow.get("stop_loss"),
                take_profit=shadow.get("take_profit"),
                confidence=float(shadow.get("confidence") or 0),
                reasoning=str(shadow.get("notes") or ""),
            ))
            shadow["agent_review"] = review.model_dump()
            if review.review_status == "BLOCK":
                shadow["status"] = BLOCKED
                shadow["notes"] = f"{shadow.get('notes', '')} | BLOCKED by agent review"
            if not dry_run:
                shadow = journal.create_signal(shadow)
                if notify_telegram:
                    send_telegram_message(format_shadow_signal_opened(shadow))
            rows.append({**shadow, "dry_run": bool(dry_run)})
            if shadow["status"] != BLOCKED:
                opened += 1
        except Exception as exc:  # noqa: BLE001 - one config should not stop the batch.
            rows.append({**base_row, "status": "error", "error": f"{type(exc).__name__}: {exc}"})
    if not configs:
        rows.append({
            "status": "no_selectable_configs",
            "registry": str(registry_path),
            "allow_watchlist_shadow": bool(allow_watchlist_shadow),
            "message": "No stable configs found; pass --allow-watchlist-shadow to test watchlist shadow signals.",
            "research_only": True,
        })
    return rows


def print_rows(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        print(
            " | ".join([
                f"status={row.get('status')}",
                f"symbol={row.get('symbol', '')}",
                f"timeframe={row.get('timeframe', '')}",
                f"side={row.get('side', '')}",
                f"entry={row.get('entry_price', '')}",
                f"sl={row.get('stop_loss', '')}",
                f"tp={row.get('take_profit', '')}",
                f"horizon_candles={row.get('horizon_candles', '')}",
                f"horizon_minutes={row.get('horizon_minutes', '')}",
                f"config_id={row.get('config_id', '')}",
                f"classification={row.get('classification', '')}",
                f"error={row.get('error', '')}",
            ])
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate local shadow signals from research configs.")
    parser.add_argument("--registry", default="refined", help="refined, general, or a registry JSONL path.")
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--max-signals", type=int, default=5)
    parser.add_argument("--allow-watchlist-shadow", action="store_true")
    parser.add_argument("--notify-telegram", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--min-classification", choices=["stable_research_candidate", "unstable_watchlist"], default="stable_research_candidate")
    parser.add_argument("--journal-path", default=str(DEFAULT_SHADOW_JOURNAL_PATH))
    parser.add_argument("--refresh-cache", action="store_true", default=True)
    parser.add_argument("--no-refresh-cache", action="store_false", dest="refresh_cache")
    return parser


async def main() -> None:
    args = build_parser().parse_args()
    rows = await generate_shadow_signals_once(
        registry=args.registry,
        symbols=args.symbols,
        max_signals=args.max_signals,
        allow_watchlist_shadow=args.allow_watchlist_shadow,
        notify_telegram=args.notify_telegram,
        dry_run=args.dry_run,
        min_classification=args.min_classification,
        journal_path=args.journal_path,
        refresh_cache=args.refresh_cache,
    )
    print_rows(rows)


if __name__ == "__main__":
    asyncio.run(main())
