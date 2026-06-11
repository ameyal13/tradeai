"""Print a clean table of currently open shadow signals."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.summarize_shadow_signals import default_journal_path, load_latest_shadow_rows  # noqa: E402


def open_signal_rows(journal_path: str | Path | None = None) -> list[dict]:
    path = Path(journal_path) if journal_path else default_journal_path()
    return [row for row in load_latest_shadow_rows(path) if row.get("status") == "OPEN"]


def print_open_signals(journal_path: str | Path | None = None) -> None:
    rows = open_signal_rows(journal_path)
    headers = [
        "signal_id",
        "symbol",
        "tf",
        "side",
        "entry",
        "SL",
        "TP",
        "generated_at",
        "expires_at",
        "status",
        "config_id",
        "review",
        "risk_flags",
    ]
    print(" | ".join(headers))
    print("-" * 160)
    for row in rows:
        review = row.get("agent_review") or {}
        risk_flags = review.get("risk_flags") or []
        values = [
            str(row.get("shadow_signal_id", ""))[:8],
            str(row.get("symbol", "")),
            str(row.get("timeframe", "")),
            str(row.get("side", "")),
            str(row.get("entry_price", "")),
            str(row.get("stop_loss", "")),
            str(row.get("take_profit", "")),
            str(row.get("generated_at", "")),
            str(row.get("expires_at", "")),
            str(row.get("status", "")),
            str(row.get("config_id", "")),
            str(review.get("review_status", "")),
            ",".join(str(flag) for flag in risk_flags),
        ]
        print(" | ".join(values))
    print(f"open_count: {len(rows)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Print current open shadow signals.")
    parser.add_argument("--journal-path", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print_open_signals(args.journal_path)


if __name__ == "__main__":
    main()
