"""Shadow Ops rate and sync audit.

Read-only diagnostic:
- reads local shadow signal journal and local shadow ops cycles,
- reads Supabase shadow_signals when service credentials are configured,
- reports generation cadence, scheduler gaps, and local-to-Supabase sync drift,
- never generates signals, evaluates outcomes, or writes Supabase.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.summarize_shadow_signals import default_journal_path, default_output_dir, load_latest_shadow_rows  # noqa: E402
from scripts.sync_shadow_journal_to_supabase import build_supabase_client_from_env  # noqa: E402
from tools.runtime_env import load_project_env  # noqa: E402
from tools.shadow_ops_cycle_repository import DEFAULT_SHADOW_OPS_CYCLES_PATH  # noqa: E402


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_stamp() -> str:
    return utc_now().strftime("%Y%m%d_%H%M%S")


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


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_supabase_shadow_signals(supabase, page_size: int = 1000) -> list[dict[str, Any]]:
    if supabase is None:
        return []
    rows: list[dict[str, Any]] = []
    start = 0
    while True:
        end = start + page_size - 1
        response = supabase.table("shadow_signals").select("*").range(start, end).execute()
        batch = response.data or []
        rows.extend(batch)
        if len(batch) < page_size:
            break
        start += page_size
    return rows


def rows_in_last_days(rows: list[dict[str, Any]], *, days: int, time_key: str) -> list[dict[str, Any]]:
    cutoff = utc_now() - timedelta(days=max(1, int(days)))
    filtered: list[dict[str, Any]] = []
    for row in rows:
        dt = parse_datetime(row.get(time_key))
        if dt is not None and dt >= cutoff:
            filtered.append(row)
    return filtered


def generation_by_day_symbol(rows: list[dict[str, Any]], days: int) -> dict[str, dict[str, int]]:
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows_in_last_days(rows, days=days, time_key="generated_at"):
        dt = parse_datetime(row.get("generated_at"))
        if dt is None:
            continue
        symbol = str(row.get("symbol") or "UNKNOWN").upper()
        grouped[dt.date().isoformat()][symbol] += 1
    return {day: dict(sorted(counter.items())) for day, counter in sorted(grouped.items())}


def generated_hour_distribution(rows: list[dict[str, Any]], days: int) -> dict[str, int]:
    hours = Counter()
    for row in rows_in_last_days(rows, days=days, time_key="generated_at"):
        dt = parse_datetime(row.get("generated_at"))
        if dt is not None:
            hours[f"{dt.hour:02d}:00"] += 1
    return {f"{hour:02d}:00": int(hours.get(f"{hour:02d}:00", 0)) for hour in range(24)}


def cycle_generation_rate(cycles: list[dict[str, Any]], days: int) -> dict[str, Any]:
    recent = rows_in_last_days(cycles, days=days, time_key="finished_at")
    runnable = [
        row
        for row in recent
        if not bool(row.get("dry_run"))
        and int(row.get("configs_scanned") or 0) > 0
        and str(row.get("health_status") or "") != "HEALTH_BLOCKED"
    ]
    generated = [row for row in runnable if int(row.get("opened_signals") or 0) > 0]
    no_generate = [row for row in runnable if int(row.get("opened_signals") or 0) == 0]
    hours_ran = Counter()
    hours_generated = Counter()
    for row in runnable:
        dt = parse_datetime(row.get("finished_at") or row.get("started_at"))
        if dt is not None:
            hours_ran[f"{dt.hour:02d}:00"] += 1
    for row in generated:
        dt = parse_datetime(row.get("finished_at") or row.get("started_at"))
        if dt is not None:
            hours_generated[f"{dt.hour:02d}:00"] += 1
    rate = round(len(generated) / len(runnable) * 100, 6) if runnable else None
    return {
        "cycle_rows_last_days": len(recent),
        "eligible_cycles_ran": len(runnable),
        "cycles_with_signal_opened": len(generated),
        "cycles_without_signal_opened": len(no_generate),
        "threshold_pass_rate_pct": rate,
        "hours_ran": {f"{hour:02d}:00": int(hours_ran.get(f"{hour:02d}:00", 0)) for hour in range(24)},
        "hours_with_signal_opened": {f"{hour:02d}:00": int(hours_generated.get(f"{hour:02d}:00", 0)) for hour in range(24)},
        "note": (
            "Uses shadow_ops_cycles.opened_signals as proxy for model exceeded threshold and opened a shadow signal."
        ),
    }


def sync_gap(local_rows: list[dict[str, Any]], supabase_rows: list[dict[str, Any]]) -> dict[str, Any]:
    remote_ids = {str(row.get("shadow_signal_id")) for row in supabase_rows if row.get("shadow_signal_id")}
    local_closed = [
        row
        for row in local_rows
        if str(row.get("status") or "").upper() in {"CLOSED", "EXPIRED"}
        and row.get("shadow_signal_id")
    ]
    missing = [row for row in local_closed if str(row.get("shadow_signal_id")) not in remote_ids]
    return {
        "local_closed_signals": len(local_closed),
        "supabase_signals": len(supabase_rows),
        "closed_local_missing_in_supabase": len(missing),
        "missing_shadow_signal_ids": [str(row.get("shadow_signal_id")) for row in missing[:50]],
        "sync_rate_pct": round((len(local_closed) - len(missing)) / len(local_closed) * 100, 6) if local_closed else None,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Shadow Ops Rate Report",
        "",
        f"Generated at: `{report.get('generated_at')}`",
        "",
        "## Guardrails",
        "",
        "- Research only. No trading signal.",
        "- Read-only diagnostic.",
        "- No Supabase writes.",
        "- No model, grid, threshold, feature, or cost changes.",
        "",
        "## Summary",
        "",
        f"- local journal: `{report.get('journal_path')}`",
        f"- local cycle journal: `{report.get('cycles_path')}`",
        f"- days analyzed: `{report.get('days')}`",
        f"- local signals: `{report.get('local_signal_count')}`",
        f"- Supabase configured: `{report.get('supabase_configured')}`",
        f"- Supabase signals: `{report.get('supabase_signal_count')}`",
        "",
        "## Generation By Day And Symbol",
        "",
    ]
    if report["generated_by_day_symbol"]:
        for day, counts in report["generated_by_day_symbol"].items():
            lines.append(f"- `{day}`: `{counts}`")
    else:
        lines.append("- No generated signals in window.")
    lines.extend([
        "",
        "## Generated Signal Hours UTC",
        "",
        f"`{report['generated_hour_distribution']}`",
        "",
        "## Cycle Threshold Pass Rate",
        "",
        f"- eligible cycles ran: `{report['cycle_generation_rate']['eligible_cycles_ran']}`",
        f"- cycles with signal opened: `{report['cycle_generation_rate']['cycles_with_signal_opened']}`",
        f"- cycles without signal opened: `{report['cycle_generation_rate']['cycles_without_signal_opened']}`",
        f"- threshold pass rate pct: `{report['cycle_generation_rate']['threshold_pass_rate_pct']}`",
        "",
        "## Local To Supabase Sync Gap",
        "",
        f"- local closed signals: `{report['sync_gap']['local_closed_signals']}`",
        f"- closed local missing in Supabase: `{report['sync_gap']['closed_local_missing_in_supabase']}`",
        f"- sync rate pct: `{report['sync_gap']['sync_rate_pct']}`",
    ])
    if report["sync_gap"]["missing_shadow_signal_ids"]:
        lines.append(f"- missing ids sample: `{report['sync_gap']['missing_shadow_signal_ids']}`")
    return "\n".join(lines).rstrip() + "\n"


def build_rate_report(
    *,
    journal_path: str | Path,
    cycles_path: str | Path,
    days: int,
) -> dict[str, Any]:
    load_project_env()
    local_rows = load_latest_shadow_rows(journal_path)
    cycles = load_jsonl(cycles_path)
    supabase = build_supabase_client_from_env()
    supabase_rows = load_supabase_shadow_signals(supabase)
    return {
        "generated_at": utc_now().isoformat(),
        "days": int(days),
        "journal_path": str(journal_path),
        "cycles_path": str(cycles_path),
        "local_signal_count": len(local_rows),
        "local_cycle_count": len(cycles),
        "supabase_configured": supabase is not None,
        "supabase_signal_count": len(supabase_rows),
        "generated_by_day_symbol": generation_by_day_symbol(local_rows, days),
        "generated_hour_distribution": generated_hour_distribution(local_rows, days),
        "cycle_generation_rate": cycle_generation_rate(cycles, days),
        "sync_gap": sync_gap(local_rows, supabase_rows),
    }


def save_report(report: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    stamp = utc_stamp()
    json_path = target / f"shadow_ops_rate_report_{stamp}.json"
    markdown_path = target / f"shadow_ops_rate_report_{stamp}.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return {"json_path": str(json_path), "markdown_path": str(markdown_path)}


def print_report(report: dict[str, Any]) -> None:
    print("Shadow Ops Rate Report")
    print("Research only. No trading signal.")
    print(f"days: {report['days']}")
    print(f"local_signals: {report['local_signal_count']}")
    print(f"local_cycles: {report['local_cycle_count']}")
    print(f"supabase_configured: {report['supabase_configured']}")
    print(f"supabase_signals: {report['supabase_signal_count']}")
    print(f"generated_by_day_symbol: {report['generated_by_day_symbol']}")
    print(f"generated_hour_distribution: {report['generated_hour_distribution']}")
    rate = report["cycle_generation_rate"]
    print(f"eligible_cycles_ran: {rate['eligible_cycles_ran']}")
    print(f"cycles_with_signal_opened: {rate['cycles_with_signal_opened']}")
    print(f"cycles_without_signal_opened: {rate['cycles_without_signal_opened']}")
    print(f"threshold_pass_rate_pct: {rate['threshold_pass_rate_pct']}")
    gap = report["sync_gap"]
    print(f"local_closed_signals: {gap['local_closed_signals']}")
    print(f"closed_local_missing_in_supabase: {gap['closed_local_missing_in_supabase']}")
    print(f"sync_rate_pct: {gap['sync_rate_pct']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Report shadow ops signal generation and sync cadence.")
    parser.add_argument("--journal-path", default=str(default_journal_path()))
    parser.add_argument("--cycles-path", default=str(DEFAULT_SHADOW_OPS_CYCLES_PATH))
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--output-dir", default=str(default_output_dir()))
    parser.add_argument("--no-write-report", action="store_false", dest="write_report", default=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = build_rate_report(journal_path=args.journal_path, cycles_path=args.cycles_path, days=args.days)
    if args.write_report:
        paths = save_report(report, args.output_dir)
        report.update(paths)
    print_report(report)
    if report.get("json_path"):
        print(f"json: {report['json_path']}")
        print(f"markdown: {report['markdown_path']}")


if __name__ == "__main__":
    main()
