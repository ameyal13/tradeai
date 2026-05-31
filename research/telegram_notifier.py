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
