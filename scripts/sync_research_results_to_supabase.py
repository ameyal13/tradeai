"""Sync local research registry summaries into Supabase.

Research only. This does not run experiments, generate signals, or place
orders. It copies aggregate research metrics for dashboard display.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.summarize_crypto_multi_research import DEFAULT_CRYPTO_MULTI_REGISTRY  # noqa: E402
from tools.research_result_repository import ResearchResultRepository  # noqa: E402
from tools.runtime_env import load_project_env  # noqa: E402


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
    parser = argparse.ArgumentParser(description="Sync research results to Supabase.")
    parser.add_argument("--registry", default=str(DEFAULT_CRYPTO_MULTI_REGISTRY))
    parser.add_argument("--source", default="crypto_multi")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    load_project_env()
    args = build_parser().parse_args()
    supabase = build_supabase_client_from_env()
    repo = ResearchResultRepository(supabase_client=supabase, registry_path=args.registry, source=args.source)
    local = repo.list_local_configs()
    if args.dry_run:
        print("Research Supabase sync dry-run")
        print("Research only. No trading signal.")
        print(f"registry: {args.registry}")
        print(f"source: {args.source}")
        print(f"local_configs: {len(local)}")
        print(f"supabase_configured: {supabase is not None}")
        return
    result = repo.sync_local_to_supabase()
    print("Research Supabase sync")
    print("Research only. No trading signal.")
    print(f"ok: {result.get('ok')}")
    print(f"reason: {result.get('reason')}")
    print(f"configs_upserted: {result.get('configs_upserted')}")
    print(f"registry_path: {result.get('registry_path')}")


if __name__ == "__main__":
    main()
