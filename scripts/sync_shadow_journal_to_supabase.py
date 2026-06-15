"""Sync local shadow journal into Supabase shadow tables.

Research only. This does not generate signals, evaluate trades, or place
orders. It simply copies the append-only local JSONL state into Supabase when
backend service-role credentials are configured.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.runtime_env import load_project_env  # noqa: E402
from tools.shadow_ops_cycle_repository import DEFAULT_SHADOW_OPS_CYCLES_PATH, ShadowOpsCycleRepository  # noqa: E402
from tools.shadow_signal_journal import DEFAULT_SHADOW_JOURNAL_PATH  # noqa: E402
from tools.shadow_signal_repository import ShadowSignalRepository  # noqa: E402


def build_supabase_client_from_env():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key or "your_" in key or "xxxx" in str(url):
        return None
    try:
        from supabase import create_client

        return create_client(url, key)
    except Exception:
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sync local shadow journal to Supabase.")
    parser.add_argument("--journal-path", default=str(DEFAULT_SHADOW_JOURNAL_PATH))
    parser.add_argument("--cycles-path", default=str(DEFAULT_SHADOW_OPS_CYCLES_PATH))
    parser.add_argument("--dry-run", action="store_true", default=False)
    return parser


def main() -> None:
    load_project_env()
    args = build_parser().parse_args()
    supabase = build_supabase_client_from_env()
    repo = ShadowSignalRepository(supabase_client=supabase, journal_path=args.journal_path)
    cycles_repo = ShadowOpsCycleRepository(supabase_client=supabase, path=args.cycles_path)
    local_rows = repo.list_local_signals(limit=10_000)
    local_cycles = cycles_repo.list_local_cycles(limit=10_000)
    if args.dry_run:
        print("Shadow Supabase sync dry-run")
        print("Research only. No trading signal.")
        print(f"journal_path: {args.journal_path}")
        print(f"cycles_path: {args.cycles_path}")
        print(f"latest_local_signals: {len(local_rows)}")
        print(f"local_cycles: {len(local_cycles)}")
        print(f"supabase_configured: {supabase is not None}")
        return
    result = repo.sync_local_to_supabase()
    cycles_result = cycles_repo.sync_local_to_supabase()
    print("Shadow Supabase sync")
    print("Research only. No trading signal.")
    print(f"ok: {result.get('ok')}")
    print(f"reason: {result.get('reason')}")
    print(f"signals_upserted: {result.get('signals_upserted')}")
    print(f"events_upserted: {result.get('events_upserted')}")
    print(f"cycles_ok: {cycles_result.get('ok')}")
    print(f"cycles_reason: {cycles_result.get('reason')}")
    print(f"cycles_upserted: {cycles_result.get('cycles_upserted')}")
    print(f"journal_path: {result.get('journal_path')}")
    print(f"cycles_path: {cycles_result.get('path')}")


if __name__ == "__main__":
    main()
