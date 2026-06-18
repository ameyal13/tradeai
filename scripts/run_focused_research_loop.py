"""Run focused research batches repeatedly with optional Supabase sync.

Research only. No trading signal. This script orchestrates the existing
focused research cycle; it does not change models, thresholds, costs, features,
or selection logic.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.focused_research_grid import build_focused_research_grid  # noqa: E402
from research.research_daemon import DEFAULT_DAEMON_DIR, run_research_cycle  # noqa: E402
from research.research_registry import ResearchRegistry, utc_now  # noqa: E402
from scripts.run_crypto_multi_asset_research import parse_symbols  # noqa: E402
from scripts.run_focused_research import (  # noqa: E402
    FOCUSED_CYCLES_DIR,
    FOCUSED_REGISTRY_PATH,
    FOCUSED_RESULTS_DIR,
    FOCUSED_STATUS_PATH,
)
from scripts.run_research_daemon import print_progress  # noqa: E402
from scripts.sync_research_results_to_supabase import build_supabase_client_from_env  # noqa: E402
from tools.research_result_repository import ResearchResultRepository  # noqa: E402
from tools.runtime_env import load_project_env  # noqa: E402


FOCUSED_LOOP_STATUS_PATH = DEFAULT_DAEMON_DIR / "focused_v2a_loop_status.json"
FOCUSED_LOOP_LOCK_PATH = DEFAULT_DAEMON_DIR / "focused_v2a_loop.lock"
FOCUSED_SOURCE = "focused_v2a"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run focused research batches in a controlled loop.")
    parser.add_argument("--max-cycles", type=int, default=1)
    parser.add_argument("--max-configs-per-cycle", type=int, default=10)
    parser.add_argument("--symbols", nargs="*", default=None, help="Subset of ADA, ETH, SOL.")
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--sync-supabase", action="store_true", default=False)
    parser.add_argument("--notify-telegram", action="store_true", default=False)
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", action="store_true", dest="resume", default=True)
    resume_group.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--retry-failed", action="store_true", default=False)
    parser.add_argument("--retry-running", action="store_true", default=False)
    parser.add_argument("--progress", action="store_true", default=True)
    parser.add_argument("--quiet", action="store_true", default=False)
    parser.add_argument("--status-path", default=str(FOCUSED_LOOP_STATUS_PATH))
    parser.add_argument("--lock-path", default=str(FOCUSED_LOOP_LOCK_PATH))
    return parser


def _json_sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_sanitize(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def acquire_lock(lock_path: str | Path) -> Path:
    path = Path(lock_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"focused_research_loop_already_running: {path}") from exc
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(json.dumps({"pid": os.getpid(), "created_at": utc_now()}, sort_keys=True))
    return path


def release_lock(lock_path: str | Path) -> None:
    try:
        Path(lock_path).unlink()
    except FileNotFoundError:
        return


def count_runnable(
    grid: list[dict[str, Any]],
    registry_path: str | Path = FOCUSED_REGISTRY_PATH,
    retry_failed: bool = False,
    retry_running: bool = False,
) -> int:
    registry = ResearchRegistry(registry_path)
    return len(registry.filter_runnable(grid, retry_failed=retry_failed, retry_running=retry_running))


def write_loop_status(status_path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(status_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_sanitize(payload), indent=2, sort_keys=True), encoding="utf-8")
    return path


def sync_focused_results_to_supabase(
    registry_path: str | Path = FOCUSED_REGISTRY_PATH,
    source: str = FOCUSED_SOURCE,
) -> dict[str, Any]:
    load_project_env()
    supabase = build_supabase_client_from_env()
    repo = ResearchResultRepository(supabase_client=supabase, registry_path=registry_path, source=source)
    return repo.sync_local_to_supabase()


def _status_payload(
    *,
    started_at: str,
    cycles_completed: int,
    max_cycles: int,
    max_configs_per_cycle: int,
    grid_size: int,
    remaining_configs: int,
    last_cycle: dict[str, Any] | None,
    sync_results: list[dict[str, Any]],
    interrupted: bool,
    error: str | None,
    started_monotonic: float,
) -> dict[str, Any]:
    return {
        "research_only": True,
        "no_trading_signal": True,
        "started_at": started_at,
        "updated_at": utc_now(),
        "cycles_completed": int(cycles_completed),
        "max_cycles": int(max_cycles),
        "max_configs_per_cycle": int(max_configs_per_cycle),
        "grid_size": int(grid_size),
        "remaining_configs": int(remaining_configs),
        "last_cycle": last_cycle,
        "sync_results": sync_results,
        "interrupted": bool(interrupted),
        "error": error,
        "elapsed_seconds": round(max(0.0, time.monotonic() - started_monotonic), 2),
    }


def _print_cycle_summary(cycle_index: int, result: dict[str, Any], remaining: int, sync_result: dict[str, Any] | None) -> None:
    summary = result.get("summary") or {}
    print(f"Focused loop cycle {cycle_index} finished")
    print("Research only. No trading signal.")
    print(f"selected_configs: {result.get('selected_configs')}")
    print(f"evaluated: {summary.get('evaluated')}")
    print(f"classification_counts: {summary.get('classification_counts')}")
    print(f"remaining_configs: {remaining}")
    if sync_result is not None:
        print(f"supabase_sync_ok: {sync_result.get('ok')}")
        print(f"configs_upserted: {sync_result.get('configs_upserted')}")
        print(f"sync_reason: {sync_result.get('reason')}")


async def run_focused_research_loop(
    *,
    max_cycles: int = 1,
    max_configs_per_cycle: int = 10,
    symbols: list[str] | None = None,
    sleep_seconds: float = 0.0,
    sync_supabase: bool = False,
    notify_telegram: bool = False,
    resume: bool = True,
    retry_failed: bool = False,
    retry_running: bool = False,
    quiet: bool = False,
    progress: bool = True,
    registry_path: str | Path = FOCUSED_REGISTRY_PATH,
    cycles_dir: str | Path = FOCUSED_CYCLES_DIR,
    results_dir: str | Path = FOCUSED_RESULTS_DIR,
    daemon_status_path: str | Path = FOCUSED_STATUS_PATH,
    loop_status_path: str | Path = FOCUSED_LOOP_STATUS_PATH,
    lock_path: str | Path = FOCUSED_LOOP_LOCK_PATH,
) -> dict[str, Any]:
    """Run focused research cycles until max_cycles or no runnable configs."""
    active_symbols = parse_symbols(symbols)
    grid = build_focused_research_grid(symbols=active_symbols)
    started_at = utc_now()
    started_monotonic = time.monotonic()
    cycles: list[dict[str, Any]] = []
    sync_results: list[dict[str, Any]] = []
    interrupted = False
    error = None
    acquired_lock = acquire_lock(lock_path)

    try:
        for cycle_index in range(1, max(0, int(max_cycles)) + 1):
            remaining_before = count_runnable(
                grid,
                registry_path=registry_path,
                retry_failed=retry_failed,
                retry_running=retry_running,
            ) if resume else len(grid)
            if remaining_before <= 0:
                break

            result = await run_research_cycle(
                grid=grid,
                max_configs_per_cycle=max_configs_per_cycle,
                resume=resume,
                retry_failed=retry_failed,
                retry_running=retry_running,
                notify_telegram=notify_telegram,
                registry_path=registry_path,
                cycles_dir=cycles_dir,
                results_dir=results_dir,
                status_path=daemon_status_path,
                progress_callback=print_progress if progress and not quiet else None,
            )
            cycles.append(result)
            if result.get("interrupted"):
                interrupted = True

            sync_result = None
            if sync_supabase:
                sync_result = sync_focused_results_to_supabase(registry_path=registry_path, source=FOCUSED_SOURCE)
                sync_results.append(sync_result)

            remaining_after = count_runnable(
                grid,
                registry_path=registry_path,
                retry_failed=retry_failed,
                retry_running=retry_running,
            ) if resume else max(0, len(grid) - len(cycles) * int(max_configs_per_cycle))

            status = _status_payload(
                started_at=started_at,
                cycles_completed=len(cycles),
                max_cycles=max_cycles,
                max_configs_per_cycle=max_configs_per_cycle,
                grid_size=len(grid),
                remaining_configs=remaining_after,
                last_cycle=result,
                sync_results=sync_results,
                interrupted=interrupted,
                error=error,
                started_monotonic=started_monotonic,
            )
            write_loop_status(loop_status_path, status)
            if not quiet:
                _print_cycle_summary(cycle_index, result, remaining_after, sync_result)

            if interrupted or remaining_after <= 0:
                break
            if sleep_seconds > 0 and cycle_index < max_cycles:
                time.sleep(float(sleep_seconds))
    except KeyboardInterrupt:
        interrupted = True
        error = "KeyboardInterrupt"
    except Exception as exc:  # noqa: BLE001 - write status before surfacing failure.
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        remaining = count_runnable(
            grid,
            registry_path=registry_path,
            retry_failed=retry_failed,
            retry_running=retry_running,
        ) if resume else 0
        final_status = _status_payload(
            started_at=started_at,
            cycles_completed=len(cycles),
            max_cycles=max_cycles,
            max_configs_per_cycle=max_configs_per_cycle,
            grid_size=len(grid),
            remaining_configs=remaining,
            last_cycle=cycles[-1] if cycles else None,
            sync_results=sync_results,
            interrupted=interrupted,
            error=error,
            started_monotonic=started_monotonic,
        )
        write_loop_status(loop_status_path, final_status)
        release_lock(acquired_lock)

    return {
        "started_at": started_at,
        "finished_at": utc_now(),
        "cycles_completed": len(cycles),
        "grid_size": len(grid),
        "remaining_configs": count_runnable(
            grid,
            registry_path=registry_path,
            retry_failed=retry_failed,
            retry_running=retry_running,
        ) if resume else 0,
        "sync_results": sync_results,
        "interrupted": interrupted,
        "error": error,
        "status_path": str(loop_status_path),
        "registry_path": str(registry_path),
    }


async def main() -> None:
    args = build_parser().parse_args()
    result = await run_focused_research_loop(
        max_cycles=args.max_cycles,
        max_configs_per_cycle=args.max_configs_per_cycle,
        symbols=parse_symbols(args.symbols),
        sleep_seconds=args.sleep_seconds,
        sync_supabase=args.sync_supabase,
        notify_telegram=args.notify_telegram,
        resume=args.resume,
        retry_failed=args.retry_failed,
        retry_running=args.retry_running,
        quiet=args.quiet,
        progress=args.progress,
        loop_status_path=args.status_path,
        lock_path=args.lock_path,
    )
    print("Focused research loop finished")
    print("Research only. No trading signal.")
    print(f"cycles_completed: {result['cycles_completed']}")
    print(f"remaining_configs: {result['remaining_configs']}")
    print(f"interrupted: {result['interrupted']}")
    print(f"error: {result['error']}")
    print(f"registry: {result['registry_path']}")
    print(f"status: {result['status_path']}")


if __name__ == "__main__":
    asyncio.run(main())
