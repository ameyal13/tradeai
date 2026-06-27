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


def _safe_number(value: Any, digits: int = 4) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "-"
    return f"{numeric:.{digits}f}".rstrip("0").rstrip(".")


def _summary_line(summary: dict[str, Any]) -> str:
    closed = int(summary.get("closed") or 0)
    wins = int(summary.get("wins") or 0)
    losses = int(summary.get("losses") or 0)
    expired = int(summary.get("expired") or 0)
    pf = summary.get("profit_factor")
    avg = summary.get("avg_return")
    drawdown = summary.get("max_drawdown")
    return (
        f"{closed} cerradas | WIN/LOSS/EXP {wins}/{losses}/{expired} | "
        f"PF {_safe_number(pf)} | avg {_safe_number(avg)}% | DD {_safe_number(drawdown)}%"
    )


def _performance_interpretation(summary: dict[str, Any]) -> str:
    closed = int(summary.get("closed") or 0)
    pf = summary.get("profit_factor")
    avg = summary.get("avg_return")
    try:
        pf_value = float(pf)
    except (TypeError, ValueError):
        pf_value = None
    try:
        avg_value = float(avg)
    except (TypeError, ValueError):
        avg_value = None
    if closed < 30 and pf_value is not None and pf_value < 1.0:
        return "Muestra todavia pequena y PF menor a 1.0: sirve para monitorear, no para confiar capital."
    if closed < 30:
        return "Muestra todavia pequena: sirve para monitorear, no para confiar capital."
    if pf_value is not None and pf_value >= 1.1 and avg_value is not None and avg_value > 0:
        return "La muestra va mejorando, pero sigue siendo shadow/research hasta validar estabilidad."
    if pf_value is not None and pf_value >= 1.0:
        return "El sistema esta cerca de break-even; necesita mas cierres y menor drawdown."
    return "PF menor a 1.0: por ahora el sistema pierde en shadow. No usar dinero real."


def format_shadow_ops_cycle_brief(result: dict[str, Any]) -> str:
    """Human-readable Telegram brief for one shadow ops cycle.

    This is intentionally deterministic. It explains status and evidence, but
    never recommends placing a trade or changing strategy parameters.
    """
    health = result.get("health_before") or {}
    evaluation = result.get("evaluation") or {}
    generation = ((result.get("generation_cycle") or {}).get("generation_summary") or {})
    status_counts = generation.get("status_counts") or {}
    final = result.get("final_summary") or {}
    error_summary = result.get("evaluation_error_summary") or {}
    sync = result.get("supabase_sync") or {}
    cycles_sync = result.get("cycles_sync") or {}

    opened = int(generation.get("opened_signals") or 0)
    scanned = int(generation.get("configs_scanned") or 0)
    skipped_hold = int(generation.get("skipped_hold") or status_counts.get("skipped_hold") or 0)
    skipped_no_price = int(status_counts.get("skipped_no_price") or 0)
    skipped_errors = int(generation.get("skipped_errors") or 0)
    eval_errors = int(error_summary.get("count") or len(evaluation.get("errors") or []))
    final_open = int(final.get("open") or 0)
    health_status = health.get("health_status") or "unknown"
    generation_skipped = result.get("generation_skipped_reason") or "no"

    if opened > 0:
        state = "Se abrio 1 shadow signal. No hubo orden real."
    elif final_open > 0:
        state = "Hay una shadow signal abierta; el sistema espera su cierre."
    elif skipped_no_price > 0 or skipped_errors > 0 or eval_errors > 0:
        state = "Ciclo ejecutado con advertencias; revisar errores antes de confiar."
    elif scanned > 0 and skipped_hold > 0:
        state = "Ciclo sano: escaneo configs, pero todas quedaron en HOLD."
    elif generation_skipped != "no":
        state = f"Generacion omitida: {generation_skipped}."
    else:
        state = "Ciclo ejecutado; no se abrieron nuevas shadow signals."

    lines = [
        "TRADEAI Shadow Ops",
        "Research only. No trading signal.",
        "",
        f"Estado: {state}",
        "",
        "Ultimo ciclo:",
        f"- Health: {health_status}",
        f"- Eval cerradas: {evaluation.get('closed', 0)}",
        f"- Eval errors: {eval_errors}",
        f"- Configs escaneadas: {scanned}",
        f"- Senales abiertas: {opened}",
        f"- HOLD: {skipped_hold}",
        f"- Sin precio: {skipped_no_price}",
        f"- Errores generacion: {skipped_errors}",
        f"- Open final: {final_open}",
        f"- Supabase signals: {sync.get('ok')} ({sync.get('reason')})",
        f"- Supabase cycles: {cycles_sync.get('ok')} ({cycles_sync.get('reason')})",
        "",
        "Resultados acumulados:",
        _summary_line(final),
        "",
        "Interpretacion:",
        _performance_interpretation(final),
        "",
        "Accion recomendada:",
    ]
    if skipped_no_price > 0:
        lines.append("- Hay problema de market data/precio. Revisar proveedor antes de evaluar edge.")
    elif eval_errors > 0:
        lines.append("- Hay errores de evaluacion. Revisar si se repiten en el dashboard.")
    elif opened > 0:
        lines.append("- Monitorear el cierre de la shadow signal. No operar dinero real.")
    elif skipped_hold > 0:
        lines.append("- Seguir acumulando evidencia. No forzar trades ni bajar thresholds.")
    else:
        lines.append("- Continuar en shadow mode y revisar el proximo ciclo.")
    lines.extend([
        "",
        "No operar dinero real.",
        "Esto no es asesoramiento financiero. No exchange orders.",
    ])
    return "\n".join(lines)[:3900]


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
    summary = {
        "closed": total,
        "wins": wins,
        "losses": losses,
        "expired": expired,
        "profit_factor": profit_factor,
        "avg_return": avg_return,
        "max_drawdown": None,
    }
    lines = [
        "TRADEAI Shadow Summary",
        "Research only. No trading signal.",
        "",
        f"Senales cerradas: {total}",
        f"WIN/LOSS/EXPIRED: {wins}/{losses}/{expired}",
        f"Win rate: {win_rate}%",
        f"Profit factor: {_safe_number(profit_factor)}",
        f"Avg return: {_safe_number(avg_return)}%",
        "",
        "Interpretacion:",
        _performance_interpretation(summary),
        "",
        "No operar dinero real con estas metricas. Shadow/research only.",
    ]
    return "\n".join(lines)[:3900]
