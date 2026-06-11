"""Summarize local shadow signal journal.

Research/shadow only. This reads the local JSONL journal and produces local
Markdown/JSON reports. It never places orders and never writes Supabase.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.telegram_notifier import format_shadow_daily_summary, send_telegram_message  # noqa: E402
from tools.runtime_env import load_project_env  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def data_dir() -> Path:
    return Path(os.getenv("TRADEAI_DATA_DIR", "data"))


def reports_dir() -> Path:
    return Path(os.getenv("TRADEAI_REPORTS_DIR", "reports"))


def default_journal_path() -> Path:
    return data_dir() / "shadow_signal_journal.jsonl"


def default_output_dir() -> Path:
    return reports_dir() / "shadow"


def _float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def max_drawdown_pct(returns: list[float]) -> float:
    equity = 100.0
    peak = equity
    drawdown = 0.0
    for ret in returns:
        equity *= 1 + ret / 100
        peak = max(peak, equity)
        if peak > 0:
            drawdown = max(drawdown, (peak - equity) / peak * 100)
    return round(drawdown, 6)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    open_rows = [row for row in rows if row.get("status") == "OPEN"]
    closed_rows = [row for row in rows if row.get("status") in {"CLOSED", "EXPIRED"}]
    wins = [row for row in closed_rows if row.get("outcome") == "WIN"]
    losses = [row for row in closed_rows if row.get("outcome") == "LOSS"]
    expired = [row for row in rows if row.get("outcome") == "EXPIRED" or row.get("status") == "EXPIRED"]
    returns = [_float(row.get("pnl_pct")) or 0.0 for row in closed_rows]
    profits = sum(value for value in returns if value > 0)
    loss_sum = abs(sum(value for value in returns if value < 0))
    profit_factor = round(profits / loss_sum, 6) if loss_sum else None
    avg_return = round(sum(returns) / len(returns), 6) if returns else 0.0
    win_rate = round(len(wins) / len(closed_rows) * 100, 6) if closed_rows else 0.0
    return {
        "total": len(rows),
        "open": len(open_rows),
        "closed": len(closed_rows),
        "wins": len(wins),
        "losses": len(losses),
        "expired": len(expired),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_return": avg_return,
        "total_return_pct": round(sum(returns), 6),
        "max_drawdown": max_drawdown_pct(returns),
    }


def group_summaries(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key, "unknown"))].append(row)
    return {name: summarize_rows(group_rows) for name, group_rows in sorted(groups.items())}


def build_shadow_summary(journal_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(journal_path) if journal_path else default_journal_path()
    rows = load_latest_shadow_rows(path)
    return {
        "generated_at": utc_now(),
        "journal_path": str(path),
        "guardrails": {
            "research_only": True,
            "no_trading": True,
            "no_exchange_orders": True,
            "no_llm_trader": True,
        },
        "summary": summarize_rows(rows),
        "by_symbol": group_summaries(rows, "symbol"),
        "by_config": group_summaries(rows, "config_id"),
        "by_timeframe": group_summaries(rows, "timeframe"),
        "signals": rows,
    }


def load_latest_shadow_rows(path: str | Path) -> list[dict[str, Any]]:
    """Read latest journal state without creating files/directories."""
    target = Path(path)
    if not target.exists():
        return []
    latest: dict[str, dict[str, Any]] = {}
    for line in target.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        latest[str(row.get("shadow_signal_id"))] = row
    return sorted(latest.values(), key=lambda row: str(row.get("generated_at", "")))


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Shadow Signal Summary",
        "",
        f"Generated at: `{report.get('generated_at')}`",
        "",
        "## Guardrails",
        "",
        "- Research only. No trading signal.",
        "- No trading real. No exchange orders.",
        "- Shadow journal only.",
        "",
        "## Overall",
        "",
        f"- total: `{summary.get('total')}`",
        f"- open: `{summary.get('open')}`",
        f"- closed: `{summary.get('closed')}`",
        f"- wins/losses/expired: `{summary.get('wins')}` / `{summary.get('losses')}` / `{summary.get('expired')}`",
        f"- win rate: `{summary.get('win_rate')}`",
        f"- profit factor: `{summary.get('profit_factor')}`",
        f"- avg return: `{summary.get('avg_return')}`",
        f"- max drawdown: `{summary.get('max_drawdown')}`",
        "",
        "## By Symbol",
        "",
    ]
    for key, value in report.get("by_symbol", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## By Timeframe", ""])
    for key, value in report.get("by_timeframe", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## By Config", ""])
    for key, value in report.get("by_config", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines).rstrip() + "\n"


def save_shadow_summary(report: dict[str, Any], output_dir: str | Path | None = None) -> dict[str, Path]:
    target = Path(output_dir) if output_dir else default_output_dir()
    target.mkdir(parents=True, exist_ok=True)
    stamp = utc_stamp()
    json_path = target / f"shadow_summary_{stamp}.json"
    markdown_path = target / f"shadow_summary_{stamp}.md"
    report["json_path"] = str(json_path)
    report["markdown_path"] = str(markdown_path)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return {"json_path": json_path, "markdown_path": markdown_path}


def summarize_shadow_signals(
    journal_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    notify_telegram: bool = False,
    write_report: bool = True,
) -> dict[str, Any]:
    report = build_shadow_summary(journal_path)
    if write_report:
        paths = save_shadow_summary(report, output_dir=output_dir)
        report["json_path"] = str(paths["json_path"])
        report["markdown_path"] = str(paths["markdown_path"])
    if notify_telegram:
        closed = [row for row in report["signals"] if row.get("status") in {"CLOSED", "EXPIRED"}]
        send_telegram_message(format_shadow_daily_summary(closed))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize local shadow signal journal.")
    parser.add_argument("--journal-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--notify-telegram", action="store_true")
    parser.add_argument("--no-write-report", action="store_false", dest="write_report", default=True)
    return parser


def main() -> None:
    load_project_env()
    args = build_parser().parse_args()
    report = summarize_shadow_signals(
        journal_path=args.journal_path,
        output_dir=args.output_dir,
        notify_telegram=args.notify_telegram,
        write_report=args.write_report,
    )
    summary = report["summary"]
    print("Shadow signal summary")
    print(f"open: {summary['open']}")
    print(f"closed: {summary['closed']}")
    print(f"wins/losses/expired: {summary['wins']}/{summary['losses']}/{summary['expired']}")
    print(f"win_rate: {summary['win_rate']}")
    print(f"profit_factor: {summary['profit_factor']}")
    print(f"avg_return: {summary['avg_return']}")
    if report.get("json_path"):
        print(f"json: {report['json_path']}")
        print(f"markdown: {report['markdown_path']}")


if __name__ == "__main__":
    main()
