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
    shadow_signals_are_similar,
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
    include_not_allowed: bool = False,
) -> list[dict[str, Any]]:
    """Load selectable configs from a registry without using test metrics."""
    allowed = {"stable_research_candidate"}
    if allow_watchlist_shadow or min_classification == "unstable_watchlist" or include_not_allowed:
        allowed.add("unstable_watchlist")
    if min_classification == "stable_research_candidate" and not include_not_allowed:
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


def classify_signal_skip(signal: dict[str, Any]) -> dict[str, Any] | None:
    """Return a skip row for HOLD/invalid generated signals, otherwise None."""
    raw_signal = str(signal.get("signal", "HOLD")).upper()
    input_features = signal.get("input_features") or {}
    if raw_signal == "HOLD":
        return {
            "status": "skipped_hold",
            "skip_reason": "la señal actual fue HOLD",
            "signal": raw_signal,
            "hold_reason": input_features.get("hold_reason"),
            "probability_buy_win": input_features.get("probability_buy_win"),
            "probability_sell_win": input_features.get("probability_sell_win"),
            "confidence": signal.get("confidence"),
            "reasoning": signal.get("reasoning"),
        }
    try:
        entry = float(signal.get("entry_price") or 0)
    except (TypeError, ValueError):
        entry = 0.0
    stop_loss = signal.get("stop_loss")
    take_profit = signal.get("take_profit")
    try:
        stop = float(stop_loss) if stop_loss is not None else None
        target = float(take_profit) if take_profit is not None else None
    except (TypeError, ValueError):
        stop = None
        target = None
    invalid_levels = (
        raw_signal not in {"BUY", "SELL"}
        or entry <= 0
        or stop is None
        or target is None
        or (raw_signal == "BUY" and not (stop < entry < target))
        or (raw_signal == "SELL" and not (target < entry < stop))
    )
    if invalid_levels:
        return {
            "status": "skipped_invalid_levels",
            "skip_reason": "entry/SL/TP inválidos",
            "signal": raw_signal,
            "entry_price": signal.get("entry_price"),
            "stop_loss": signal.get("stop_loss"),
            "take_profit": signal.get("take_profit"),
            "confidence": signal.get("confidence"),
        }
    return None


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
    max_configs: int | None = None,
    max_configs_scanned: int | None = None,
) -> list[dict[str, Any]]:
    registry_path = registry_path_from_choice(registry)
    journal = None if dry_run else ShadowSignalJournal(journal_path)
    source_registry = "registry" if registry == "general" else "refined_registry" if registry == "refined" else str(registry_path)
    selected_symbols = normalize_crypto_symbols(symbols) if symbols else None
    configs = load_candidate_configs(
        registry_path,
        symbols=selected_symbols,
        allow_watchlist_shadow=allow_watchlist_shadow,
        min_classification=min_classification,
        include_not_allowed=True,
    )
    rows: list[dict[str, Any]] = []
    opened_this_batch: list[dict[str, Any]] = []
    opened = 0
    attempted = 0
    scan_limit = max_configs_scanned if max_configs_scanned is not None else max_configs
    for config in configs:
        if opened >= max_signals:
            break
        if scan_limit is not None and attempted >= int(scan_limit):
            break
        attempted += 1
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
            rows.append({
                **base_row,
                "status": "skipped_not_allowed",
                "skip_reason": "watchlist sin --allow-watchlist-shadow",
            })
            continue
        if journal is not None and journal.has_open_signal(config_id, symbol, timeframe):
            rows.append({
                **base_row,
                "status": "skipped_duplicate_open",
                "skip_reason": "ya existe señal OPEN para esta config/symbol/timeframe",
            })
            continue
        try:
            try:
                loaded = await load_experiment_candles(
                    symbol,
                    timeframe,
                    max_candles=int(config.get("max_candles", 1500)),
                    use_cache=True,
                    refresh_cache=bool(refresh_cache),
                )
                candles = loaded.get("candles")
                if candles is None or len(candles) == 0:
                    rows.append({
                        **base_row,
                        "status": "skipped_no_price",
                        "skip_reason": "no se pudo obtener precio/candles",
                        "error_type": "empty_candles",
                        "error_message": "No candles loaded for shadow signal generation.",
                    })
                    continue
            except Exception as exc:  # noqa: BLE001 - classify data/price failures.
                rows.append({
                    **base_row,
                    "status": "skipped_no_price",
                    "skip_reason": "no se pudo obtener precio/candles",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)[:180],
                })
                continue
            horizon_minutes = horizon_minutes_from_candles(int(config["horizon_candles"]), timeframe)
            params = strategy_params_from_config(config)
            params["use_sentiment"] = True
            signal = generate_strategy_signal_from_df(
                candles,
                strategy_mode=str(config.get("strategy_mode", "xgboost")),
                provider="none",
                horizon_minutes=horizon_minutes,
                strategy_params=params,
            ).to_dict()
            skip = classify_signal_skip(signal)
            if skip:
                rows.append({**base_row, **skip})
                continue
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
                rows.append({
                    **base_row,
                    "status": "skipped_invalid_levels",
                    "skip_reason": "entry/SL/TP inválidos",
                    "signal": signal.get("signal"),
                })
                continue
            similar = next((row for row in opened_this_batch if shadow_signals_are_similar(row, shadow)), None)
            if similar is None and journal is not None:
                similar = journal.find_open_similar_signal(shadow)
            if similar is not None:
                rows.append({
                    **base_row,
                    "status": "skipped_duplicate_open_similar",
                    "skip_reason": "ya existe senal OPEN similar para symbol/timeframe/side/niveles",
                    "duplicate_reason": "same_symbol_timeframe_side_entry_stop_take_profit",
                    "duplicate_shadow_signal_id": similar.get("shadow_signal_id"),
                    "duplicate_config_id": similar.get("config_id"),
                    "duplicate_horizon_candles": similar.get("horizon_candles"),
                    "side": shadow.get("side"),
                    "entry_price": shadow.get("entry_price"),
                    "stop_loss": shadow.get("stop_loss"),
                    "take_profit": shadow.get("take_profit"),
                })
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
                shadow["skip_reason"] = "agent review bloqueó"
            if not dry_run:
                shadow = journal.create_signal(shadow)
                if notify_telegram:
                    send_telegram_message(format_shadow_signal_opened(shadow))
            output_status = "skipped_agent_block" if shadow["status"] == BLOCKED else shadow["status"]
            rows.append({**shadow, "status": output_status, "journal_status": shadow["status"], "dry_run": bool(dry_run)})
            if shadow["status"] != BLOCKED:
                opened_this_batch.append(shadow)
                opened += 1
        except Exception as exc:  # noqa: BLE001 - one config should not stop the batch.
            rows.append({
                **base_row,
                "status": "skipped_error",
                "skip_reason": "excepción controlada",
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:180],
            })
    if not configs:
        rows.append({
            "status": "no_selectable_configs",
            "registry": str(registry_path),
            "allow_watchlist_shadow": bool(allow_watchlist_shadow),
            "message": "No stable configs found; pass --allow-watchlist-shadow to test watchlist shadow signals.",
            "research_only": True,
        })
    return rows


def summarize_generation_rows(
    rows: list[dict[str, Any]],
    journal_path: str | Path = DEFAULT_SHADOW_JOURNAL_PATH,
    max_signals: int | None = None,
    max_configs_scanned: int | None = None,
) -> dict[str, Any]:
    statuses: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status"))
        statuses[status] = statuses.get(status, 0) + 1
    configs_scanned = sum(1 for row in rows if row.get("config_id"))
    return {
        "selected_configs": configs_scanned,
        "configs_scanned": configs_scanned,
        "opened_signals": statuses.get("OPEN", 0),
        "skipped_hold": statuses.get("skipped_hold", 0),
        "skipped_duplicate_open": statuses.get("skipped_duplicate_open", 0),
        "skipped_duplicate_similar": statuses.get("skipped_duplicate_open_similar", 0),
        "skipped_errors": statuses.get("skipped_error", 0) + statuses.get("skipped_no_price", 0),
        "max_signals": max_signals,
        "max_configs_scanned": max_configs_scanned,
        "status_counts": statuses,
        "journal_path": str(journal_path),
    }


def print_rows(rows: list[dict[str, Any]], journal_path: str | Path = DEFAULT_SHADOW_JOURNAL_PATH) -> None:
    for row in rows:
        status = row.get("status")
        parts = [
                f"status={row.get('status')}",
                f"symbol={row.get('symbol', '')}",
                f"timeframe={row.get('timeframe', '')}",
                f"config_id={row.get('config_id', '')}",
                f"classification={row.get('classification', '')}",
        ]
        if status not in {
            "skipped_hold",
            "skipped_not_allowed",
            "skipped_duplicate_open",
            "skipped_duplicate_open_similar",
            "skipped_no_price",
            "skipped_error",
        }:
            parts.extend([
                f"side={row.get('side', '')}",
                f"entry={row.get('entry_price', '')}",
                f"sl={row.get('stop_loss', '')}",
                f"tp={row.get('take_profit', '')}",
                f"horizon_candles={row.get('horizon_candles', '')}",
                f"horizon_minutes={row.get('horizon_minutes', '')}",
            ])
        if row.get("skip_reason"):
            parts.append(f"reason={row.get('skip_reason')}")
        if row.get("duplicate_reason"):
            parts.append(f"duplicate_reason={row.get('duplicate_reason')}")
            parts.append(f"duplicate_config_id={row.get('duplicate_config_id', '')}")
        if status == "skipped_hold":
            parts.extend([
                f"hold_reason={row.get('hold_reason', '')}",
                f"prob_buy={row.get('probability_buy_win', '')}",
                f"prob_sell={row.get('probability_sell_win', '')}",
                f"confidence={row.get('confidence', '')}",
            ])
        if row.get("error_type") or row.get("error_message"):
            parts.extend([
                f"error_type={row.get('error_type', '')}",
                f"error_message={row.get('error_message', '')}",
            ])
        print(" | ".join(parts))
    summary = summarize_generation_rows(rows, journal_path=journal_path)
    print("Summary")
    print(f"selected configs: {summary['selected_configs']}")
    print(f"configs_scanned: {summary['configs_scanned']}")
    print(f"opened signals: {summary['opened_signals']}")
    print(f"skipped_hold: {summary['skipped_hold']}")
    print(f"skipped_duplicate_open: {summary['skipped_duplicate_open']}")
    print(f"skipped_duplicate_similar: {summary['skipped_duplicate_similar']}")
    print(f"skipped_errors: {summary['skipped_errors']}")
    print(f"max_signals: {summary['max_signals']}")
    print(f"max_configs_scanned: {summary['max_configs_scanned']}")
    print(f"status_counts: {summary['status_counts']}")
    print(f"journal_path: {summary['journal_path']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate local shadow signals from research configs.")
    parser.add_argument("--registry", default="refined", help="refined, general, or a registry JSONL path.")
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--max-signals", type=int, default=5)
    parser.add_argument("--max-configs-scanned", type=int, default=None)
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
        max_configs_scanned=args.max_configs_scanned,
    )
    print_rows(rows, journal_path=args.journal_path)


if __name__ == "__main__":
    asyncio.run(main())
