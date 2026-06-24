"""Run Feature Policy Confirmation v1 locally.

Research only. This control grid compares baseline/time_only/candidate feature
sets by symbol before any feature policy can be promoted.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.feature_policy_confirmation_grid import build_feature_policy_confirmation_grid  # noqa: E402
from research.research_daemon import DEFAULT_DAEMON_DIR  # noqa: E402
from research.research_registry import ResearchRegistry, utc_now  # noqa: E402
from scripts.run_4h_focused_research import build_4h_focused_feature_frame  # noqa: E402
from scripts.run_crypto_multi_asset_research import parse_symbols  # noqa: E402
from scripts.run_feature_expansion_research import run_feature_expansion_config  # noqa: E402
from scripts.run_historical_experiments import load_experiment_candles  # noqa: E402
from scripts.run_research_daemon import print_progress  # noqa: E402


FEATURE_POLICY_REGISTRY_PATH = DEFAULT_DAEMON_DIR / "feature_policy_confirmation_v1_registry.jsonl"
FEATURE_POLICY_CYCLES_DIR = DEFAULT_DAEMON_DIR / "feature_policy_confirmation_v1_cycles"
FEATURE_POLICY_RESULTS_DIR = DEFAULT_DAEMON_DIR / "feature_policy_confirmation_v1_results"
FEATURE_POLICY_STATUS_PATH = DEFAULT_DAEMON_DIR / "feature_policy_confirmation_v1_current_status.json"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _json_sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_sanitize(item) for item in value]
    if isinstance(value, tuple):
        return [_json_sanitize(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def render_result_markdown(summary: dict[str, Any]) -> str:
    setup = summary["setups"][0]["setup"]
    result = summary["setups"][0]
    aggregate = result["aggregate"]
    lines = [
        "# Feature Policy Confirmation v1 Result",
        "",
        f"Generated at: `{summary['created_at']}`",
        "",
        "## Guardrails",
        "",
        "- Research only. No trading signal.",
        "- No live/shadow feature changes.",
        "- No Supabase writes.",
        "- Baseline/time_only controls are included before promotion.",
        "",
        "## Setup",
        "",
        f"- symbol: `{setup.get('symbol')}`",
        f"- timeframe: `{setup.get('timeframe')}`",
        f"- feature_set: `{setup.get('feature_set')}`",
        f"- horizon_candles: `{setup.get('horizon_candles')}`",
        f"- risk_reward: `{setup.get('risk_reward')}`",
        f"- atr_stop_multiplier: `{setup.get('atr_stop_multiplier')}`",
        f"- cost_mode: `{setup.get('cost_mode')}`",
        "",
        "## Aggregate",
        "",
        f"- classification: `{result.get('classification')}`",
        f"- median validation PF: `{aggregate.get('median_validation_pf')}`",
        f"- median validation avg return: `{aggregate.get('median_validation_avg_return')}`",
        f"- beats time_only rate: `{aggregate.get('beats_time_only_rate')}`",
        f"- beats dummy_random rate: `{aggregate.get('beats_dummy_random_rate')}`",
    ]
    return "\n".join(lines).rstrip() + "\n"


def save_config_result(result: dict[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = result["setup"]
    stamp = utc_stamp()
    name = f"feature_policy_{config['symbol']}_{config['feature_set']}_{config['config_id']}_{stamp}"
    json_path = output_dir / f"{name}.json"
    markdown_path = output_dir / f"{name}.md"
    summary = {
        "created_at": utc_now(),
        "methodology": {
            "phase": "feature_policy_confirmation_v1",
            "rolling_windows": True,
            "purge_candles": int(config["horizon_candles"]),
            "research_only": True,
            "no_trading_signal": True,
            "objective": "Confirm symbol-specific feature policies against baseline and time_only controls.",
        },
        "setups": [result],
    }
    summary["json_path"] = str(json_path)
    summary["markdown_path"] = str(markdown_path)
    json_path.write_text(json.dumps(_json_sanitize(summary), indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(render_result_markdown(summary), encoding="utf-8")
    return {"json_path": str(json_path), "markdown_path": str(markdown_path)}


def write_status(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_sanitize(payload), indent=2, sort_keys=True), encoding="utf-8")


async def run_feature_policy_confirmation_cycle(
    *,
    grid: list[dict[str, Any]],
    max_configs_per_cycle: int,
    resume: bool,
    retry_failed: bool,
    retry_running: bool,
    registry_path: Path = FEATURE_POLICY_REGISTRY_PATH,
    cycles_dir: Path = FEATURE_POLICY_CYCLES_DIR,
    results_dir: Path = FEATURE_POLICY_RESULTS_DIR,
    status_path: Path = FEATURE_POLICY_STATUS_PATH,
    progress_callback: Any = None,
) -> dict[str, Any]:
    registry = ResearchRegistry(registry_path)
    runnable = registry.filter_runnable(grid, retry_failed=retry_failed, retry_running=retry_running) if resume else grid
    selected = runnable[: max(0, int(max_configs_per_cycle))]
    feature_cache: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    results: list[dict[str, Any]] = []
    start = time.monotonic()
    cycle_started_at = utc_now()

    write_status(status_path, {
        "cycle_started_at": cycle_started_at,
        "selected_configs": len(selected),
        "completed_in_cycle": 0,
        "classification_counts": {},
        "updated_at": utc_now(),
    })

    for index, config in enumerate(selected, start=1):
        running = registry.mark_running(config)
        started_at = running.get("started_at")
        try:
            symbol = str(config["symbol"]).upper()
            if symbol not in feature_cache:
                loaded = await load_experiment_candles(
                    symbol,
                    str(config["timeframe"]),
                    max_candles=int(config["max_candles"]),
                    use_cache=True,
                    refresh_cache=False,
                )
                candles = loaded["candles"]
                feature_cache[symbol] = (candles, await build_4h_focused_feature_frame(symbol, candles))
            candles, features = feature_cache[symbol]
            setup_result = await run_feature_expansion_config(config, candles, features)
            paths = save_config_result(setup_result, results_dir)
            classification = str(setup_result["classification"])
            status = "insufficient_data" if classification == "needs_more_data" else "completed"
            registry.mark_finished(
                config,
                status=status,
                classification=classification,
                json_path=paths["json_path"],
                markdown_path=paths["markdown_path"],
                started_at=started_at,
            )
            row = {
                "config_id": config["config_id"],
                "status": status,
                "classification": classification,
                "config": config,
                "result": setup_result,
                **paths,
            }
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {str(exc)[:300]}"
            registry.mark_finished(config, status="failed", classification=None, error=error, started_at=started_at)
            row = {"config_id": config["config_id"], "status": "failed", "classification": "failed", "config": config, "error": error}
        results.append(row)
        counts: dict[str, int] = {}
        for item in results:
            label = str(item.get("classification") or item.get("status"))
            counts[label] = counts.get(label, 0) + 1
        status_payload = {
            "cycle_started_at": cycle_started_at,
            "current_index": index,
            "current_config": config,
            "completed_in_cycle": len(results),
            "selected_configs": len(selected),
            "classification_counts": counts,
            "last_result": row,
            "elapsed_seconds": round(time.monotonic() - start, 2),
            "updated_at": utc_now(),
        }
        write_status(status_path, status_payload)
        if progress_callback:
            progress_callback(status_payload)

    cycle = {
        "created_at": cycle_started_at,
        "finished_at": utc_now(),
        "grid_size": len(grid),
        "runnable_before_cycle": len(runnable),
        "selected_configs": len(selected),
        "results": results,
        "summary": {
            "evaluated": len(results),
            "classification_counts": {
                label: sum(1 for row in results if str(row.get("classification")) == label)
                for label in sorted({str(row.get("classification")) for row in results})
            },
        },
        "guardrails": {
            "research_only": True,
            "no_trading_signal": True,
            "no_supabase": True,
            "accuracy_not_used": True,
        },
    }
    cycles_dir.mkdir(parents=True, exist_ok=True)
    stamp = utc_stamp()
    json_path = cycles_dir / f"feature_policy_confirmation_cycle_{stamp}.json"
    markdown_path = cycles_dir / f"feature_policy_confirmation_cycle_{stamp}.md"
    cycle["json_path"] = str(json_path)
    cycle["markdown_path"] = str(markdown_path)
    json_path.write_text(json.dumps(_json_sanitize(cycle), indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(render_cycle_markdown(cycle), encoding="utf-8")
    return cycle


def render_cycle_markdown(cycle: dict[str, Any]) -> str:
    lines = [
        "# Feature Policy Confirmation v1 Cycle",
        "",
        "Research only. No trading signal.",
        "",
        f"- grid size: `{cycle.get('grid_size')}`",
        f"- selected configs: `{cycle.get('selected_configs')}`",
        f"- classification counts: `{(cycle.get('summary') or {}).get('classification_counts')}`",
        "",
        "## Results",
        "",
    ]
    for row in cycle.get("results", []):
        config = row.get("config") or {}
        aggregate = ((row.get("result") or {}).get("aggregate") or {})
        lines.append(
            f"- `{row.get('classification')}: {config.get('symbol')} {config.get('feature_set')}` "
            f"PF `{aggregate.get('median_validation_pf')}`, avg `{aggregate.get('median_validation_avg_return')}`, "
            f"beats_time `{aggregate.get('beats_time_only_rate')}`"
        )
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Feature Policy Confirmation v1.")
    parser.add_argument("--once", action="store_true", default=True)
    parser.add_argument("--max-configs-per-cycle", type=int, default=10)
    parser.add_argument("--symbols", nargs="*", default=None, help="Subset of ADA, ETH, SOL.")
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", action="store_true", dest="resume", default=True)
    resume_group.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--retry-failed", action="store_true", default=False)
    parser.add_argument("--retry-running", action="store_true", default=False)
    parser.add_argument("--progress", action="store_true", default=True)
    parser.add_argument("--quiet", action="store_true", default=False)
    return parser


async def main() -> None:
    args = build_parser().parse_args()
    symbols = parse_symbols(args.symbols)
    grid = build_feature_policy_confirmation_grid(symbols=symbols)
    result = await run_feature_policy_confirmation_cycle(
        grid=grid,
        max_configs_per_cycle=args.max_configs_per_cycle,
        resume=args.resume,
        retry_failed=args.retry_failed,
        retry_running=args.retry_running,
        progress_callback=print_progress if args.progress and not args.quiet else None,
    )
    print("Feature Policy Confirmation v1 cycle finished")
    print("Research only. No trading signal.")
    print(f"grid_size: {result['grid_size']}")
    print(f"runnable_before_cycle: {result['runnable_before_cycle']}")
    print(f"selected_configs: {result['selected_configs']}")
    print(f"evaluated: {result['summary']['evaluated']}")
    print(f"classification_counts: {result['summary']['classification_counts']}")
    print(f"registry: {FEATURE_POLICY_REGISTRY_PATH}")
    print(f"status: {FEATURE_POLICY_STATUS_PATH}")
    print(f"json: {result['json_path']}")
    print(f"markdown: {result['markdown_path']}")


if __name__ == "__main__":
    asyncio.run(main())
