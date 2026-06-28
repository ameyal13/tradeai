"""Audit live shadow performance from Supabase or the local JSONL journal.

This module is intentionally read-only. It does not generate signals, change
research classifications, tune models, or write to Supabase.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.summarize_shadow_signals import default_journal_path, load_latest_shadow_rows  # noqa: E402
from scripts.sync_shadow_journal_to_supabase import build_supabase_client_from_env  # noqa: E402
from tools.runtime_env import load_project_env  # noqa: E402


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "shadow"
MIN_CONFIG_SAMPLE = 5
STRONGER_CONFIG_SAMPLE = 10


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_stamp() -> str:
    return utc_now().strftime("%Y%m%d_%H%M%S")


def parse_datetime(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def finite_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def close_time(row: dict[str, Any]) -> datetime | None:
    for key in ("closed_at", "evaluated_at", "updated_at", "expires_at", "generated_at"):
        parsed = parse_datetime(row.get(key))
        if parsed is not None:
            return parsed
    return None


def is_closed(row: dict[str, Any]) -> bool:
    return str(row.get("status") or "").upper() in {"CLOSED", "EXPIRED"}


def closed_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        [dict(row) for row in rows if is_closed(row)],
        key=lambda row: close_time(row) or datetime.min.replace(tzinfo=timezone.utc),
    )


def is_strategy_eligible(row: dict[str, Any]) -> bool:
    if not is_closed(row):
        return False
    outcome = str(row.get("outcome") or "").upper()
    exit_reason = str(row.get("exit_reason") or "").lower()
    return outcome not in {"EXPIRED", "INVALID", ""} and exit_reason != "evaluation_http_error"


def strategy_eligible_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows if is_strategy_eligible(row)]


def max_drawdown_from_returns(returns: Iterable[float]) -> float:
    equity = 100.0
    peak = equity
    max_drawdown = 0.0
    for value in returns:
        equity *= 1 + float(value) / 100
        peak = max(peak, equity)
        if peak > 0:
            max_drawdown = max(max_drawdown, (peak - equity) / peak * 100)
    return round(max_drawdown, 6)


def performance_metrics(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    materialized = list(rows)
    closed = closed_rows(materialized)
    recorded_returns = [value for row in closed if (value := finite_float(row.get("pnl_pct"))) is not None]
    # Match the operational dashboard: expired/closed rows without a recorded
    # PnL contribute zero rather than disappearing from average return.
    returns = [finite_float(row.get("pnl_pct")) or 0.0 for row in closed]
    outcomes = Counter(str(row.get("outcome") or "UNKNOWN").upper() for row in closed)
    gross_profit = sum(value for value in returns if value > 0)
    gross_loss = abs(sum(value for value in returns if value < 0))
    profit_factor = round(gross_profit / gross_loss, 6) if gross_loss > 0 else None
    durations = []
    for row in closed:
        opened = parse_datetime(row.get("generated_at"))
        finished = close_time(row)
        if opened is not None and finished is not None and finished >= opened:
            durations.append((finished - opened).total_seconds() / 3600)
    return {
        "total": len(materialized),
        "open": sum(str(row.get("status") or "").upper() == "OPEN" for row in materialized),
        "closed": len(closed),
        "wins": int(outcomes.get("WIN", 0)),
        "losses": int(outcomes.get("LOSS", 0)),
        "breakeven": int(outcomes.get("BREAKEVEN", 0)),
        "expired": int(outcomes.get("EXPIRED", 0)),
        "invalid": int(outcomes.get("INVALID", 0)),
        "with_pnl": len(recorded_returns),
        "closed_without_pnl": len(closed) - len(recorded_returns),
        "win_rate": round(outcomes.get("WIN", 0) / len(closed) * 100, 6) if closed else None,
        "profit_factor": profit_factor,
        "gross_profit_pct": round(gross_profit, 6),
        "gross_loss_pct": round(gross_loss, 6),
        "avg_return_pct": round(sum(returns) / len(returns), 6) if returns else None,
        "total_return_pct": round(sum(returns), 6) if returns else None,
        "max_drawdown_pct": max_drawdown_from_returns(returns),
        "median_duration_hours": round(sorted(durations)[len(durations) // 2], 6) if durations else None,
    }


def grouped_performance(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        value = row.get(key)
        if isinstance(value, dict):
            value = json.dumps(value, sort_keys=True)
        grouped[str(value or "UNKNOWN")].append(row)
    return {name: performance_metrics(group) for name, group in sorted(grouped.items())}


def agent_review_status(row: dict[str, Any]) -> str:
    review = row.get("agent_review")
    if not isinstance(review, dict):
        return "UNKNOWN"
    return str(review.get("review_status") or "UNKNOWN").upper()


def confidence_bucket(value: Any) -> str:
    confidence = finite_float(value)
    if confidence is None:
        return "unknown"
    if confidence < 50:
        return "<50"
    if confidence < 55:
        return "50-55"
    if confidence < 60:
        return "55-60"
    if confidence < 65:
        return "60-65"
    if confidence < 70:
        return "65-70"
    return "70+"


def grouped_by_function(
    rows: list[dict[str, Any]],
    grouper,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(grouper(row))].append(row)
    return {name: performance_metrics(group) for name, group in sorted(grouped.items())}


def config_live_status(metrics: dict[str, Any]) -> str:
    sample = int(metrics.get("closed") or 0)
    pf = finite_float(metrics.get("profit_factor"))
    avg = finite_float(metrics.get("avg_return_pct"))
    if sample < MIN_CONFIG_SAMPLE:
        return "insufficient_live_sample"
    if sample >= STRONGER_CONFIG_SAMPLE and pf is not None and pf < 0.7 and (avg or 0) < 0:
        return "manual_pause_review"
    if pf is not None and pf < 1.0 and (avg or 0) <= 0:
        return "negative_live_watch"
    if pf is not None and pf >= 1.1 and (avg or 0) > 0:
        return "promising_but_unconfirmed"
    return "mixed_live_evidence"


def build_config_diagnostics(
    rows: list[dict[str, Any]],
    research_configs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    research_by_id = {str(row.get("config_id")): row for row in research_configs if row.get("config_id")}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in strategy_eligible_rows(rows):
        grouped[str(row.get("config_id") or "UNKNOWN")].append(row)
    diagnostics = []
    for config_id, config_rows in grouped.items():
        metrics = performance_metrics(config_rows)
        research = research_by_id.get(config_id, {})
        validation_pf = finite_float(research.get("median_validation_pf"))
        live_pf = finite_float(metrics.get("profit_factor"))
        diagnostics.append({
            "config_id": config_id,
            "symbol": next((row.get("symbol") for row in config_rows if row.get("symbol")), None),
            "timeframe": next((row.get("timeframe") for row in config_rows if row.get("timeframe")), None),
            "research_classification": research.get("classification") or config_rows[0].get("classification"),
            "research_validation_pf": validation_pf,
            "research_validation_avg_return": finite_float(research.get("median_validation_avg_return")),
            "live": metrics,
            "diagnostic_status": config_live_status(metrics),
            "historical_live_mismatch": bool(
                validation_pf is not None and validation_pf >= 1.0 and live_pf is not None and live_pf < 1.0
            ),
            "selection_uses_test": False,
        })
    return sorted(
        diagnostics,
        key=lambda row: (
            finite_float((row.get("live") or {}).get("profit_factor")) is not None,
            finite_float((row.get("live") or {}).get("profit_factor")) or -1,
            int((row.get("live") or {}).get("closed") or 0),
        ),
        reverse=True,
    )


def _cycle_metrics(cycles: list[dict[str, Any]]) -> dict[str, Any]:
    health = Counter(str(row.get("health_status") or "UNKNOWN") for row in cycles)
    evaluation_errors = sum(int(row.get("evaluation_errors") or 0) for row in cycles)
    no_price = 0
    for row in cycles:
        counts = row.get("status_counts") if isinstance(row.get("status_counts"), dict) else {}
        no_price += int(counts.get("skipped_no_price") or 0)
    latest = max((parse_datetime(row.get("finished_at")) for row in cycles), default=None)
    return {
        "total_cycles": len(cycles),
        "health_counts": dict(sorted(health.items())),
        "cycles_with_opened_signal": sum(int(row.get("opened_signals") or 0) > 0 for row in cycles),
        "signals_opened": sum(int(row.get("opened_signals") or 0) for row in cycles),
        "configs_scanned": sum(int(row.get("configs_scanned") or 0) for row in cycles),
        "evaluation_errors": evaluation_errors,
        "skipped_no_price": no_price,
        "latest_finished_at": latest.isoformat() if latest else None,
    }


def summarize_cycles(cycles: list[dict[str, Any]]) -> dict[str, Any]:
    result = _cycle_metrics(cycles)
    now = utc_now()
    for label, delta in (("last_24h", timedelta(hours=24)), ("last_7d", timedelta(days=7))):
        cutoff = now - delta
        recent = [
            row for row in cycles
            if (finished := parse_datetime(row.get("finished_at"))) is not None and finished >= cutoff
        ]
        result[label] = _cycle_metrics(recent)
    return result


def load_supabase_table(supabase: Any, table: str, *, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    page_size = min(1000, max(1, int(limit)))
    start = 0
    while start < limit:
        end = min(start + page_size, limit) - 1
        batch = supabase.table(table).select("*").range(start, end).execute().data or []
        rows.extend(batch)
        if len(batch) < page_size:
            break
        start += page_size
    return rows[:limit]


def build_report(
    *,
    signals: list[dict[str, Any]],
    cycles: list[dict[str, Any]] | None = None,
    research_configs: list[dict[str, Any]] | None = None,
    source: str,
) -> dict[str, Any]:
    cycles = list(cycles or [])
    research_configs = list(research_configs or [])
    closed = closed_rows(signals)
    strategy_rows = strategy_eligible_rows(signals)
    technical_exclusions = [row for row in closed if not is_strategy_eligible(row)]
    overall = performance_metrics(signals)
    report = {
        "generated_at": utc_now().isoformat(),
        "source": source,
        "guardrails": {
            "research_only": True,
            "read_only": True,
            "no_trading_signal": True,
            "no_supabase_writes": True,
            "test_metrics_not_used_for_selection": True,
        },
        "overall": overall,
        "strategy_performance": performance_metrics(strategy_rows),
        "technical_exclusions": {
            "count": len(technical_exclusions),
            "by_exit_reason": dict(sorted(Counter(str(row.get("exit_reason") or "UNKNOWN") for row in technical_exclusions).items())),
            "note": "Excluded from strategy diagnostics because they do not represent an economic market outcome.",
        },
        "by_symbol": grouped_performance(strategy_rows, "symbol"),
        "by_side": grouped_performance(strategy_rows, "side"),
        "by_timeframe": grouped_performance(strategy_rows, "timeframe"),
        "by_exit_reason": grouped_performance(closed, "exit_reason"),
        "by_agent_review": grouped_by_function(strategy_rows, agent_review_status),
        "confidence_buckets": grouped_by_function(strategy_rows, lambda row: confidence_bucket(row.get("confidence"))),
        "config_diagnostics": build_config_diagnostics(signals, research_configs),
        "cycle_health": summarize_cycles(cycles),
    }
    report["automatic_conclusion"] = automatic_conclusion(report)
    return report


def automatic_conclusion(report: dict[str, Any]) -> list[str]:
    operational = report.get("overall") or {}
    overall = report.get("strategy_performance") or operational
    closed = int(overall.get("closed") or 0)
    pf = finite_float(overall.get("profit_factor"))
    avg = finite_float(overall.get("avg_return_pct"))
    conclusions = ["Research only. No trading signal."]
    if closed < 30:
        conclusions.append("The live sample is small; config-level conclusions are preliminary.")
    if pf is not None and pf < 1.0:
        conclusions.append("Live shadow profit factor is below 1.0; the current system has negative expectancy.")
    if avg is not None and avg < 0:
        conclusions.append("Average live shadow return is negative after recorded costs.")
    missing_pnl = int(operational.get("closed_without_pnl") or 0)
    if missing_pnl:
        conclusions.append(
            f"{missing_pnl} closed/expired signals have no recorded PnL and are treated as zero by operational metrics."
        )
    mismatches = sum(bool(row.get("historical_live_mismatch")) for row in report.get("config_diagnostics") or [])
    if mismatches:
        conclusions.append(f"{mismatches} configs show positive historical PF but live PF below 1.0.")
    pause_reviews = sum(row.get("diagnostic_status") == "manual_pause_review" for row in report.get("config_diagnostics") or [])
    if pause_reviews:
        conclusions.append(f"{pause_reviews} configs meet the minimum sample for manual pause review; no automatic disable was applied.")
    conclusions.append("Do not promote to real money; use this report to design the next controlled research hypothesis.")
    return conclusions


def _metric(value: Any) -> str:
    numeric = finite_float(value)
    return "-" if numeric is None else f"{numeric:.4f}"


def render_markdown(report: dict[str, Any]) -> str:
    overall = report["overall"]
    strategy = report["strategy_performance"]
    lines = [
        "# Shadow Live Performance Audit",
        "",
        f"Generated at: `{report.get('generated_at')}`",
        f"Source: `{report.get('source')}`",
        "",
        "> Research only. No trading signal. This report never writes Supabase or changes a strategy.",
        "",
        "## Operational Totals (Dashboard Compatible)",
        "",
        f"- total/open/closed: `{overall.get('total')}` / `{overall.get('open')}` / `{overall.get('closed')}`",
        f"- wins/losses/expired: `{overall.get('wins')}` / `{overall.get('losses')}` / `{overall.get('expired')}`",
        f"- win rate: `{_metric(overall.get('win_rate'))}%`",
        f"- profit factor: `{_metric(overall.get('profit_factor'))}`",
        f"- average return: `{_metric(overall.get('avg_return_pct'))}%`",
        f"- total return: `{_metric(overall.get('total_return_pct'))}%`",
        f"- max drawdown: `{_metric(overall.get('max_drawdown_pct'))}%`",
        f"- closed without recorded PnL: `{overall.get('closed_without_pnl')}`",
        "",
        "## Strategy-Evaluable Performance",
        "",
        f"- eligible closed outcomes: `{strategy.get('closed')}`",
        f"- wins/losses/breakeven: `{strategy.get('wins')}` / `{strategy.get('losses')}` / `{strategy.get('breakeven')}`",
        f"- win rate: `{_metric(strategy.get('win_rate'))}%`",
        f"- profit factor: `{_metric(strategy.get('profit_factor'))}`",
        f"- average return: `{_metric(strategy.get('avg_return_pct'))}%`",
        f"- total return: `{_metric(strategy.get('total_return_pct'))}%`",
        f"- max drawdown: `{_metric(strategy.get('max_drawdown_pct'))}%`",
        f"- technical exclusions: `{report.get('technical_exclusions')}`",
        "",
        "## Automatic Conclusion",
        "",
    ]
    lines.extend(f"- {item}" for item in report.get("automatic_conclusion") or [])
    for title, key in (
        ("By Symbol", "by_symbol"),
        ("By Side", "by_side"),
        ("By Agent Review", "by_agent_review"),
        ("Confidence Buckets", "confidence_buckets"),
        ("By Exit Reason", "by_exit_reason"),
    ):
        lines.extend(["", f"## {title}", "", "| Group | Closed | PF | Avg Return % | Win Rate % | DD % |", "|---|---:|---:|---:|---:|---:|"])
        for name, metrics in (report.get(key) or {}).items():
            lines.append(
                f"| {name} | {metrics.get('closed')} | {_metric(metrics.get('profit_factor'))} | "
                f"{_metric(metrics.get('avg_return_pct'))} | {_metric(metrics.get('win_rate'))} | "
                f"{_metric(metrics.get('max_drawdown_pct'))} |"
            )
    lines.extend([
        "",
        "## Config Diagnostics",
        "",
        "| Config | Symbol | Closed | Live PF | Live Avg % | Research Val PF | Status | Historical/Live Mismatch |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ])
    for row in report.get("config_diagnostics") or []:
        live = row.get("live") or {}
        lines.append(
            f"| {row.get('config_id')} | {row.get('symbol')} | {live.get('closed')} | "
            f"{_metric(live.get('profit_factor'))} | {_metric(live.get('avg_return_pct'))} | "
            f"{_metric(row.get('research_validation_pf'))} | {row.get('diagnostic_status')} | "
            f"{row.get('historical_live_mismatch')} |"
        )
    cycle = report.get("cycle_health") or {}
    lines.extend([
        "",
        "## Railway Cycle Health",
        "",
        f"- cycles: `{cycle.get('total_cycles')}`",
        f"- health counts: `{cycle.get('health_counts')}`",
        f"- cycles with signal opened: `{cycle.get('cycles_with_opened_signal')}`",
        f"- configs scanned: `{cycle.get('configs_scanned')}`",
        f"- evaluation errors: `{cycle.get('evaluation_errors')}`",
        f"- skipped no price: `{cycle.get('skipped_no_price')}`",
        f"- latest finished: `{cycle.get('latest_finished_at')}`",
        f"- last 24h: `{cycle.get('last_24h')}`",
        f"- last 7d: `{cycle.get('last_7d')}`",
        "",
        "## Methodological Limits",
        "",
        "- Config diagnostics are not automatic trading decisions.",
        f"- Fewer than {MIN_CONFIG_SAMPLE} closed signals per config is insufficient live evidence.",
        "- Research validation metrics are compared only with live evidence; test PF is not used for selection.",
        "- The report does not retrain, tune, disable, or promote any model.",
    ])
    return "\n".join(lines).rstrip() + "\n"


def save_report(report: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    stamp = utc_stamp()
    json_path = target / f"shadow_live_performance_audit_{stamp}.json"
    markdown_path = target / f"shadow_live_performance_audit_{stamp}.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return {"json_path": str(json_path), "markdown_path": str(markdown_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only audit of live shadow performance.")
    parser.add_argument("--source", choices=("auto", "supabase", "local"), default="auto")
    parser.add_argument("--journal-path", default=str(default_journal_path()))
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--no-write-report", action="store_false", dest="write_report", default=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    load_project_env()
    supabase = build_supabase_client_from_env() if args.source != "local" else None
    if args.source == "supabase" and supabase is None:
        raise SystemExit("Supabase is not configured. No report was written.")
    use_supabase = supabase is not None and args.source in {"auto", "supabase"}
    if use_supabase:
        signals = load_supabase_table(supabase, "shadow_signals", limit=args.limit)
        cycles = load_supabase_table(supabase, "shadow_ops_cycles", limit=args.limit)
        research_configs = load_supabase_table(supabase, "research_configs", limit=args.limit)
        source = "supabase"
    else:
        signals = load_latest_shadow_rows(args.journal_path)[: args.limit]
        cycles = []
        research_configs = []
        source = "local_jsonl"
    report = build_report(
        signals=signals,
        cycles=cycles,
        research_configs=research_configs,
        source=source,
    )
    if args.write_report:
        report.update(save_report(report, args.output_dir))
    overall = report["overall"]
    strategy = report["strategy_performance"]
    print("Shadow Live Performance Audit")
    print("Research only. No trading signal.")
    print(f"source: {report['source']}")
    print(f"total/open/closed: {overall['total']}/{overall['open']}/{overall['closed']}")
    print(f"wins/losses/expired: {overall['wins']}/{overall['losses']}/{overall['expired']}")
    print(f"profit_factor: {overall['profit_factor']}")
    print(f"avg_return_pct: {overall['avg_return_pct']}")
    print(f"max_drawdown_pct: {overall['max_drawdown_pct']}")
    print(f"strategy_eligible_closed: {strategy['closed']}")
    print(f"strategy_profit_factor: {strategy['profit_factor']}")
    print(f"strategy_avg_return_pct: {strategy['avg_return_pct']}")
    print(f"technical_exclusions: {report['technical_exclusions']['count']}")
    print(f"config_diagnostics: {len(report['config_diagnostics'])}")
    if report.get("json_path"):
        print(f"json: {report['json_path']}")
        print(f"markdown: {report['markdown_path']}")


if __name__ == "__main__":
    main()
