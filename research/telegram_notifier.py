"""Optional Telegram notifications for local research runs.

This module is research-only. It never sends trading signals or operational
trade instructions.
"""
from __future__ import annotations

import os
from typing import Any

import httpx


TELEGRAM_API_BASE = "https://api.telegram.org"


def telegram_enabled() -> bool:
    return bool(os.getenv("TELEGRAM_BOT_TOKEN") and os.getenv("TELEGRAM_CHAT_ID"))


def send_telegram_message(text: str) -> bool:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("Telegram disabled: missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID")
        return False
    try:
        response = httpx.post(
            f"{TELEGRAM_API_BASE}/bot{token}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": text[:3900],
                "disable_web_page_preview": True,
            },
            timeout=15,
        )
        response.raise_for_status()
        return True
    except Exception as exc:  # noqa: BLE001 - notifications must not break research runs.
        print(f"Telegram notification failed: {exc.__class__.__name__}: {exc}")
        return False


def _classification_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        classification = str(row.get("classification", "unknown"))
        counts[classification] = counts.get(classification, 0) + 1
    return counts


def _top_rows(rows: list[dict[str, Any]], metric_group: str, metric_name: str, limit: int = 3) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda item: (item.get(metric_group, {}) or {}).get(metric_name, float("-inf")) or float("-inf"),
        reverse=True,
    )[:limit]


def _row_label(row: dict[str, Any], metric_group: str) -> str:
    config = row.get("config", {})
    metrics = row.get(metric_group, {}) or {}
    return (
        f"{config.get('symbol')} {config.get('timeframe')} h{config.get('horizon_candles')} "
        f"RR{config.get('risk_reward')} ATR{config.get('atr_stop_multiplier')} {config.get('cost_mode')} "
        f"| PF {metrics.get('profit_factor')} | avg {metrics.get('avg_return_pct')} | DD {metrics.get('max_drawdown_pct')}%"
    )


def format_autopilot_summary_for_telegram(
    summary: dict[str, Any] | list[dict[str, Any]],
    markdown_path: str,
    jsonl_path: str,
) -> str:
    rows = summary.get("rows", []) if isinstance(summary, dict) else summary
    rows = list(rows or [])
    counts = _classification_counts(rows)
    candidates = counts.get("candidate_for_further_validation", 0) + counts.get("weak_candidate", 0)
    watchlist = counts.get("research_watchlist", 0) + counts.get("validation_candidate_test_failed", 0)
    count_text = ", ".join(f"{key}={value}" for key, value in sorted(counts.items())) or "none"

    lines = [
        "Research Autopilot finished",
        "Mode: research only, no trading signal",
        "",
        f"Experiments: {len(rows)}",
        f"Candidates: {candidates}",
        f"Watchlist: {watchlist}",
        f"Counts: {count_text}",
        "",
        "Best validation PF:",
    ]
    for index, row in enumerate(_top_rows(rows, "validation_metrics", "profit_factor"), start=1):
        lines.append(f"{index}. {_row_label(row, 'validation_metrics')}")

    lines.extend(["", "Best validation avg return:"])
    for index, row in enumerate(_top_rows(rows, "validation_metrics", "avg_return_pct"), start=1):
        lines.append(f"{index}. {_row_label(row, 'validation_metrics')}")

    lines.extend(["", "Best test PF (diagnostic only, not selectable):"])
    for index, row in enumerate(_top_rows(rows, "test_metrics", "profit_factor"), start=1):
        lines.append(f"{index}. {_row_label(row, 'test_metrics')}")

    lines.extend([
        "",
        "Test-only diagnostics are not selectable.",
        "",
        "Markdown:",
        markdown_path,
        "",
        "JSONL:",
        jsonl_path,
        "",
        "Research only. No trading signal.",
    ])
    return "\n".join(lines)[:3900]


def format_shadow_signal_opened(signal: dict[str, Any]) -> str:
    """Plain-text Telegram message for a shadow signal opening."""
    mode = "WATCHLIST SHADOW ONLY" if signal.get("watchlist_shadow") else "STABLE SHADOW"
    review = signal.get("agent_review") or {}
    risk_flags = review.get("risk_flags") or []
    risk_flag_text = ", ".join(str(flag) for flag in risk_flags) if risk_flags else "none"
    lines = [
        f"{mode}",
        "Research only. No trading signal.",
        "",
        f"Symbol: {signal.get('symbol')} {signal.get('timeframe')}",
        f"Side: {signal.get('side')}",
        f"Entry: {signal.get('entry_price')}",
        f"SL: {signal.get('stop_loss')}",
        f"TP: {signal.get('take_profit')}",
        f"RR: {signal.get('risk_reward')}",
        f"Horizon: {signal.get('horizon_candles')} candles / {signal.get('horizon_minutes')} minutes",
        f"Confidence: {signal.get('confidence')}",
        f"Config: {signal.get('config_id')}",
        f"Classification: {signal.get('classification')}",
        f"Agent review: {review.get('review_status', 'not_run')}",
        f"Risk flags: {risk_flag_text}",
        "",
        "No exchange order was placed.",
    ]
    return "\n".join(str(line) for line in lines)[:3900]


def format_shadow_signal_evaluated(signal: dict[str, Any]) -> str:
    """Plain-text Telegram message for a shadow signal outcome."""
    lines = [
        "Shadow signal evaluated",
        "Research only. No trading signal.",
        "",
        f"Symbol: {signal.get('symbol')} {signal.get('timeframe')}",
        f"Side: {signal.get('side')}",
        f"Outcome: {signal.get('outcome')}",
        f"Exit price: {signal.get('exit_price')}",
        f"PnL pct: {signal.get('pnl_pct')}",
        f"Exit reason: {signal.get('exit_reason')}",
        f"Fees: {signal.get('fees')}",
        f"Slippage/spread cost: {signal.get('slippage')}",
        f"Cost pct: commission={signal.get('commission_pct')} slippage={signal.get('slippage_pct')} spread={signal.get('spread_pct')}",
        f"Config: {signal.get('config_id')}",
    ]
    return "\n".join(str(line) for line in lines)[:3900]


def format_shadow_daily_summary(signals: list[dict[str, Any]]) -> str:
    """Plain-text summary for closed shadow signals."""
    total = len(signals)
    wins = sum(1 for row in signals if row.get("outcome") == "WIN")
    losses = sum(1 for row in signals if row.get("outcome") == "LOSS")
    expired = sum(1 for row in signals if row.get("outcome") == "EXPIRED")
    returns = [float(row.get("pnl_pct") or 0) for row in signals]
    profits = sum(value for value in returns if value > 0)
    loss_sum = abs(sum(value for value in returns if value < 0))
    profit_factor = round(profits / loss_sum, 6) if loss_sum else None
    avg_return = round(sum(returns) / total, 6) if total else 0
    win_rate = round(wins / total * 100, 6) if total else 0
    lines = [
        "Shadow summary",
        "Research only. No trading signal.",
        "",
        f"Total signals: {total}",
        f"Wins/losses/expired: {wins}/{losses}/{expired}",
        f"Win rate: {win_rate}",
        f"Profit factor: {profit_factor}",
        f"Average return pct: {avg_return}",
    ]
    return "\n".join(lines)[:3900]
