"""Append-only local storage for Research Autopilot results."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CANDIDATE_CLASSES = {"weak_candidate", "candidate_for_further_validation", "validation_candidate_test_failed"}


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


class ExperimentStore:
    def __init__(self, jsonl_path: str | Path | None = None, reports_dir: str | Path = "reports/research_autopilot"):
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = Path(jsonl_path) if jsonl_path is not None else self.reports_dir / f"results_{utc_stamp()}.jsonl"

    def append_result(self, result: dict[str, Any]) -> None:
        """Append one result. Existing lines are never overwritten."""
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, sort_keys=True) + "\n")

    def load_all(self) -> list[dict[str, Any]]:
        if not self.jsonl_path.exists():
            return []
        rows = []
        for line in self.jsonl_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows

    def completed_ids(self) -> set[str]:
        return {str(row.get("experiment_id")) for row in self.load_all() if row.get("experiment_id")}

    def get_candidates(self) -> list[dict[str, Any]]:
        return [row for row in self.load_all() if row.get("classification") in CANDIDATE_CLASSES]

    def generate_markdown_report(self, markdown_path: str | Path | None = None) -> Path:
        rows = self.load_all()
        path = Path(markdown_path) if markdown_path is not None else self.reports_dir / f"autopilot_summary_{utc_stamp()}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_markdown(rows, str(self.jsonl_path)), encoding="utf-8")
        return path


def render_markdown(rows: list[dict[str, Any]], jsonl_path: str) -> str:
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row.get("classification", "unknown"))] = counts.get(str(row.get("classification", "unknown")), 0) + 1

    lines = [
        "# Research Autopilot Summary",
        "",
        f"Generated at: `{datetime.now(timezone.utc).isoformat()}`",
        f"JSONL: `{jsonl_path}`",
        "",
        "## Methodology Guardrails",
        "",
        "- Local research only; no trading, paper trading, Supabase, scheduler, or frontend.",
        "- Validation selects candidates; test is holdout confirmation only.",
        "- Accuracy is not used as a criterion.",
        "- All results are append-only, including rejects.",
        "",
        "## Classification Counts",
        "",
        f"`{counts}`",
        "",
        "## Results",
        "",
    ]
    for row in rows:
        config = row.get("config", {})
        validation = row.get("validation_metrics", {})
        test = row.get("test_metrics", {})
        lines.extend([
            f"### {row.get('classification', 'unknown')}: {config.get('symbol')} {config.get('timeframe')} h{config.get('horizon_candles')} RR{config.get('risk_reward')} ATR{config.get('atr_stop_multiplier')} {config.get('cost_mode')}",
            "",
            f"- experiment_id: `{row.get('experiment_id')}`",
            f"- validation: trades `{validation.get('n_trades')}`, avg `{validation.get('avg_return_pct')}`, PF `{validation.get('profit_factor')}`, DD `{validation.get('max_drawdown_pct')}`",
            f"- test: trades `{test.get('n_trades')}`, avg `{test.get('avg_return_pct')}`, PF `{test.get('profit_factor')}`, DD `{test.get('max_drawdown_pct')}`",
            f"- random validation avg/PF: `{(row.get('baselines', {}).get('validation', {}).get('random_same_count') or {}).get('average_return')}` / `{(row.get('baselines', {}).get('validation', {}).get('random_same_count') or {}).get('profit_factor')}`",
            f"- deterministic validation avg/PF: `{(row.get('baselines', {}).get('validation', {}).get('deterministic') or {}).get('avg_return_pct')}` / `{(row.get('baselines', {}).get('validation', {}).get('deterministic') or {}).get('profit_factor')}`",
            f"- reasons: `{row.get('reasons', [])}`",
            "",
        ])
    lines.extend([
        "## Interpretation",
        "",
        "- `reject`: validation failed core EV/PF/random criteria.",
        "- `weak_candidate`: validation positive and beats random, but not deterministic.",
        "- `candidate_for_further_validation`: validation strong and test did not contradict.",
        "- `validation_candidate_test_failed`: validation passed, but holdout test failed confirmation.",
        "",
        "Do not tune parameters using test results. Failed candidates are evidence, not errors.",
        "",
    ])
    return "\n".join(lines)
