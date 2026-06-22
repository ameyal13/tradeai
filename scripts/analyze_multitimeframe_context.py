"""Analyze 4h multi-timeframe context for evaluated shadow signals.

Read-only research diagnostic:
- reads evaluated research-only shadow signals from Supabase,
- fetches 4h asset/BTC context at each signal timestamp,
- writes a local Markdown report,
- does not write to Supabase and does not generate new signals.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.multitimeframe_context import compute_4h_context, normalize_signal_side  # noqa: E402
from tools.runtime_env import load_project_env  # noqa: E402


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "research_daemon"
EXCLUDED_OUTCOMES = {"EXPIRED", "INVALID", "INVALID_DATA"}


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def build_supabase_client_from_env():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")
    if not url or not key or "your_" in key or "xxxx" in str(url):
        return None
    try:
        from supabase import create_client

        return create_client(url, key)
    except Exception:
        return None


def parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        try:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


def is_research_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def load_evaluated_shadow_signals(supabase, limit: int) -> list[dict[str, Any]]:
    response = supabase.table("shadow_signals").select("*").eq("research_only", True).limit(int(limit)).execute()
    rows = response.data or []
    filtered: list[dict[str, Any]] = []
    for row in rows:
        outcome = str(row.get("outcome") or "").upper()
        if not outcome or outcome in EXCLUDED_OUTCOMES:
            continue
        if not is_research_true(row.get("research_only", True)):
            continue
        if parse_datetime(row.get("generated_at")) is None:
            continue
        if normalize_signal_side(str(row.get("side") or "")) not in {"LONG", "SHORT"}:
            continue
        filtered.append(row)
    return filtered[: int(limit)]


def profit_factor(pnls: list[float]) -> float | None:
    gross_profit = sum(value for value in pnls if value > 0)
    gross_loss = abs(sum(value for value in pnls if value < 0))
    if gross_loss == 0:
        return round(gross_profit, 6) if gross_profit > 0 else None
    return round(gross_profit / gross_loss, 6)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pnls = [finite_float(row.get("pnl_pct")) for row in rows]
    pnl_values = [value for value in pnls if value is not None]
    wins = [row for row in rows if str(row.get("outcome") or "").upper() == "WIN"]
    return {
        "count": len(rows),
        "win_rate": round(len(wins) / len(rows) * 100, 6) if rows else 0.0,
        "profit_factor": profit_factor(pnl_values),
        "avg_pnl_pct": round(sum(pnl_values) / len(pnl_values), 6) if pnl_values else None,
    }


def grouped_metrics(enriched: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in enriched:
        context = row["context"]
        groups[f"asset_aligned={context['asset_trend_aligned']}"].append(row)
        groups[f"btc_aligned={context['btc_trend_aligned']}"].append(row)
        groups[f"full_alignment={context['full_alignment']}"].append(row)
        groups[
            "cross|"
            f"asset={context['asset_trend_aligned']}|"
            f"btc={context['btc_trend_aligned']}|"
            f"full={context['full_alignment']}"
        ].append(row)
        groups[f"symbol={row.get('symbol')}"].append(row)
        groups[f"side={normalize_signal_side(str(row.get('side') or ''))}"].append(row)
        groups[f"symbol_side={row.get('symbol')}|{normalize_signal_side(str(row.get('side') or ''))}"].append(row)
    return {key: summarize_rows(value) for key, value in sorted(groups.items())}


def alignment_win_rate_delta(groups: dict[str, dict[str, Any]]) -> float | None:
    aligned = groups.get("full_alignment=True")
    non_aligned = groups.get("full_alignment=False")
    if not aligned or not non_aligned or aligned.get("count", 0) == 0 or non_aligned.get("count", 0) == 0:
        return None
    return round(float(aligned["win_rate"]) - float(non_aligned["win_rate"]), 6)


def automatic_conclusion(groups: dict[str, dict[str, Any]]) -> list[str]:
    delta = alignment_win_rate_delta(groups)
    conclusions = ["Research only. No trading signal."]
    if delta is None:
        conclusions.append("Insufficient full-alignment and non-alignment samples to judge context signal.")
        return conclusions
    conclusions.append(f"Full-alignment win-rate delta: {delta} percentage points.")
    if abs(delta) < 5:
        conclusions.append("context signal is weak, not recommended as filter")
    elif delta >= 5:
        conclusions.append("context signal warrants filter testing in next grid")
    else:
        conclusions.append("context signal may be negatively predictive; investigate as a possible avoid/filter flag")
    return conclusions


def render_markdown(enriched: list[dict[str, Any]], groups: dict[str, dict[str, Any]], warnings: list[str]) -> str:
    lines = [
        "# Multi-Timeframe Context Diagnostic",
        "",
        f"Generated at: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "## Guardrails",
        "",
        "- Research only. No trading signal.",
        "- Read-only Supabase access.",
        "- No model, feature, threshold, or cost changes.",
        "- No Supabase writes.",
        "- Test is diagnostic; this analyzes already evaluated shadow signals.",
        "",
        "## Summary",
        "",
        f"- analyzed signals: `{len(enriched)}`",
        f"- warnings: `{len(warnings)}`",
        "",
        "## Automatic Conclusion",
        "",
    ]
    lines.extend(f"- {item}" for item in automatic_conclusion(groups))
    lines.extend([
        "",
        "## Group Metrics",
        "",
        "| Group | Count | Win Rate | Profit Factor | Avg PnL % |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for key, metrics in groups.items():
        lines.append(
            f"| {key} | {metrics.get('count')} | {metrics.get('win_rate')} | "
            f"{metrics.get('profit_factor')} | {metrics.get('avg_pnl_pct')} |"
        )
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings[:50])
    return "\n".join(lines).rstrip() + "\n"


async def analyze(limit: int, output_dir: str | Path, dry_run: bool = False) -> dict[str, Any]:
    load_project_env()
    supabase = build_supabase_client_from_env()
    if supabase is None:
        raise RuntimeError("Supabase is not configured; set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY or SUPABASE_KEY.")
    signals = load_evaluated_shadow_signals(supabase, limit=limit)
    if dry_run:
        print("Multi-timeframe context dry-run")
        print("Research only. No trading signal.")
        print(f"eligible_signals: {len(signals)}")
        return {"eligible_signals": len(signals), "dry_run": True}

    enriched: list[dict[str, Any]] = []
    warnings: list[str] = []
    for signal in signals:
        signal_time = parse_datetime(signal.get("generated_at"))
        if signal_time is None:
            continue
        try:
            context = await compute_4h_context(
                str(signal.get("symbol") or ""),
                signal_time,
                str(signal.get("side") or ""),
            )
        except Exception as exc:  # noqa: BLE001 - per-signal network failures should not abort report.
            warning = (
                f"{signal.get('shadow_signal_id') or signal.get('id') or 'unknown'} "
                f"{signal.get('symbol')} {type(exc).__name__}: {exc}"
            )
            print(f"warning: {warning}")
            warnings.append(warning)
            continue
        enriched.append({**signal, "context": context})

    groups = grouped_metrics(enriched)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    markdown_path = target / f"multitimeframe_context_analysis_{utc_stamp()}.md"
    markdown_path.write_text(render_markdown(enriched, groups, warnings), encoding="utf-8")
    print("Multi-timeframe context analysis generated")
    print("Research only. No trading signal.")
    print(f"eligible_signals: {len(signals)}")
    print(f"analyzed_signals: {len(enriched)}")
    print(f"warnings: {len(warnings)}")
    print(f"markdown: {markdown_path}")
    return {
        "eligible_signals": len(signals),
        "analyzed_signals": len(enriched),
        "warnings": warnings,
        "groups": groups,
        "markdown_path": str(markdown_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze 4h context for evaluated shadow signals.")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--dry-run", action="store_true", default=False)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    asyncio.run(analyze(limit=args.limit, output_dir=args.output_dir, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
