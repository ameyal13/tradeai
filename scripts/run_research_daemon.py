"""Run Research Daemon v1 locally.

This is research-only: no trading, no paper trading, no Supabase, no endpoints,
and no operational signals.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.research_daemon import run_research_cycle  # noqa: E402


def _format_config(config: dict) -> str:
    return (
        f"{config.get('symbol')} {config.get('timeframe')} "
        f"h{config.get('horizon_candles')} RR{config.get('risk_reward')} "
        f"ATR{config.get('atr_stop_multiplier')} {config.get('cost_mode')}"
    )


def print_progress(status: dict) -> None:
    completed = int(status.get("completed_in_cycle") or 0)
    if completed <= 0:
        return
    selected = int(status.get("selected_configs") or 0)
    current = status.get("current_config") or {}
    last = status.get("last_result") or {}
    aggregate = ((last.get("result") or {}).get("aggregate") or {})
    classification = last.get("classification") or last.get("status")
    eta = status.get("eta_seconds")
    eta_text = f"{eta}s" if eta is not None else "n/a"
    print(
        f"[{completed}/{selected}] {last.get('config_id')} {_format_config(current)} | "
        f"classification={classification} | "
        f"median_val_pf={aggregate.get('median_validation_pf')} | "
        f"median_val_avg={aggregate.get('median_validation_avg_return')} | "
        f"valid_windows={aggregate.get('valid_windows')} | "
        f"stable={status.get('stable_candidates_so_far')} | "
        f"watchlist={status.get('unstable_watchlist_so_far')} | "
        f"rejects={status.get('rejects_so_far')} | "
        f"errors={status.get('errors_so_far')} | "
        f"elapsed={status.get('elapsed_seconds')}s | ETA={eta_text}",
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local Research Daemon v1.")
    parser.add_argument("--once", action="store_true", default=True, help="Run one cycle. This is the only v1 mode.")
    parser.add_argument("--max-configs-per-cycle", type=int, default=5)
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", action="store_true", dest="resume", default=True)
    resume_group.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--retry-failed", action="store_true", default=False)
    parser.add_argument("--notify-telegram", action="store_true", default=False)
    parser.add_argument("--progress", action="store_true", default=True)
    parser.add_argument("--quiet", action="store_true", default=False)
    return parser


async def main() -> None:
    args = build_parser().parse_args()
    result = await run_research_cycle(
        max_configs_per_cycle=args.max_configs_per_cycle,
        resume=args.resume,
        retry_failed=args.retry_failed,
        notify_telegram=args.notify_telegram,
        progress_callback=print_progress if args.progress and not args.quiet else None,
    )
    summary = result["summary"]
    print("Research Daemon cycle finished")
    print("Research only. No trading signal.")
    print(f"interrupted: {result['interrupted']}")
    print(f"grid_size: {result['grid_size']}")
    print(f"runnable_before_cycle: {result['runnable_before_cycle']}")
    print(f"selected_configs: {result['selected_configs']}")
    print(f"evaluated: {summary['evaluated']}")
    print(f"classification_counts: {summary['classification_counts']}")
    print(f"stable_candidates: {len(summary['stable_candidates'])}")
    print(f"unstable_watchlist: {len(summary['unstable_watchlist'])}")
    if args.notify_telegram:
        print(f"telegram_sent: {result.get('telegram_sent', False)}")
    print(f"json: {result['json_path']}")
    print(f"markdown: {result['markdown_path']}")


if __name__ == "__main__":
    asyncio.run(main())
