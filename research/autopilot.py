"""Local Research Autopilot orchestration."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from research.experiment_grid import build_experiment_grid, load_grid_checkpoint, save_grid_checkpoint
from research.experiment_runner import run_experiment
from research.experiment_store import ExperimentStore
from research.telegram_notifier import format_autopilot_summary_for_telegram, send_telegram_message


DEFAULT_REPORTS_DIR = Path("reports/research_autopilot")
DEFAULT_CHECKPOINT_PATH = DEFAULT_REPORTS_DIR / "checkpoint.json"


async def run_autopilot(
    resume: bool = True,
    max_experiments: int | None = None,
    reports_dir: str | Path = DEFAULT_REPORTS_DIR,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
    notify_telegram: bool = False,
) -> dict[str, Any]:
    reports = Path(reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    checkpoint_file = Path(checkpoint_path)
    checkpoint = load_grid_checkpoint(checkpoint_file) if resume else {}
    grid = checkpoint.get("grid") or build_experiment_grid()
    result_path = checkpoint.get("result_path") if resume else None
    store = ExperimentStore(jsonl_path=result_path, reports_dir=reports)
    completed = set(checkpoint.get("completed_ids") or []) | store.completed_ids()
    markdown_path = checkpoint.get("markdown_path") if resume else None
    started_at = datetime.now(timezone.utc).isoformat()
    ran = 0

    try:
        for config in grid:
            experiment_id = str(config["experiment_id"])
            if experiment_id in completed:
                continue
            if max_experiments is not None and ran >= max_experiments:
                break
            print(f"Running {experiment_id}: {config['symbol']} {config['timeframe']} h{config['horizon_candles']} RR{config['risk_reward']} ATR{config['atr_stop_multiplier']} {config['cost_mode']}")
            result = await run_experiment(config)
            store.append_result(result)
            completed.add(experiment_id)
            ran += 1
            save_grid_checkpoint(
                checkpoint_file,
                grid,
                completed_ids=completed,
                result_path=str(store.jsonl_path),
                markdown_path=markdown_path,
            )
            print(f"  -> {result['classification']} validation_avg={result['validation_metrics'].get('avg_return_pct')} validation_pf={result['validation_metrics'].get('profit_factor')}")
    except KeyboardInterrupt:
        save_grid_checkpoint(
            checkpoint_file,
            grid,
            completed_ids=completed,
            result_path=str(store.jsonl_path),
            markdown_path=markdown_path,
        )
        markdown = store.generate_markdown_report(markdown_path)
        return {
            "interrupted": True,
            "started_at": started_at,
            "ran": ran,
            "completed": len(completed),
            "total": len(grid),
            "jsonl_path": str(store.jsonl_path),
            "markdown_path": str(markdown),
            "checkpoint_path": str(checkpoint_file),
        }

    markdown = store.generate_markdown_report(markdown_path)
    save_grid_checkpoint(
        checkpoint_file,
        grid,
        completed_ids=completed,
        result_path=str(store.jsonl_path),
        markdown_path=str(markdown),
    )
    telegram_sent = False
    if notify_telegram:
        try:
            message = format_autopilot_summary_for_telegram(
                store.load_all(),
                markdown_path=str(markdown),
                jsonl_path=str(store.jsonl_path),
            )
            telegram_sent = send_telegram_message(message)
        except Exception as exc:  # noqa: BLE001 - notification must not break the run.
            print(f"Telegram notification failed: {exc.__class__.__name__}: {exc}")
            telegram_sent = False
    return {
        "interrupted": False,
        "started_at": started_at,
        "ran": ran,
        "completed": len(completed),
        "total": len(grid),
        "jsonl_path": str(store.jsonl_path),
        "markdown_path": str(markdown),
        "checkpoint_path": str(checkpoint_file),
        "candidates": len(store.get_candidates()),
        "watchlist": len(store.get_watchlist()),
        "telegram_sent": telegram_sent,
    }
