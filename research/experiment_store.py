"""Append-only local storage for Research Autopilot results."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CANDIDATE_CLASSES = {"weak_candidate", "candidate_for_further_validation"}
WATCHLIST_CLASSES = {"research_watchlist", "validation_candidate_test_failed"}


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

    def get_watchlist(self) -> list[dict[str, Any]]:
        return [row for row in self.load_all() if row.get("classification") in WATCHLIST_CLASSES]

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

    candidates = [row for row in rows if row.get("classification") in CANDIDATE_CLASSES]
    watchlist = [row for row in rows if row.get("classification") in WATCHLIST_CLASSES]
    hard_rejects = [row for row in rows if row.get("classification") == "hard_reject"]
    rejects = [row for row in rows if row.get("classification") == "reject"]
    test_only_positive = [
        row for row in rows
        if "test_positive_but_validation_failed_do_not_select" in (row.get("reasons") or [])
    ]

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
        "## Top Diagnostics",
        "",
        "### Best By Validation PF",
        "",
    ]
    for row in sorted(rows, key=lambda item: (item.get("validation_metrics") or {}).get("profit_factor", 0), reverse=True)[:5]:
        lines.append(render_result_line(row))
    lines.extend([
        "",
        "### Best By Validation Avg Return",
        "",
    ])
    for row in sorted(rows, key=lambda item: (item.get("validation_metrics") or {}).get("avg_return_pct", 0), reverse=True)[:5]:
        lines.append(render_result_line(row))
    lines.extend([
        "",
        "### Best By Test PF (Diagnostic Only, Not Selectable)",
        "",
    ])
    for row in sorted(rows, key=lambda item: (item.get("test_metrics") or {}).get("profit_factor", 0), reverse=True)[:5]:
        lines.append(render_result_line(row))
    lines.extend([
        "",
        "## Candidates",
        "",
    ])
    lines.extend(render_group(candidates, "No candidates selected from validation."))
    lines.extend([
        "",
        "## Research Watchlist",
        "",
    ])
    lines.extend(render_group(watchlist, "No research watchlist items."))
    lines.extend([
        "",
        "## Hard Rejects",
        "",
    ]
    )
    lines.extend(render_group(hard_rejects, "No hard rejects."))
    lines.extend([
        "",
        "## Test-Only Positives, Not Selectable",
        "",
    ])
    lines.extend(render_group(test_only_positive, "No test-only positives."))
    lines.extend([
        "",
        "## Other Rejects",
        "",
    ])
    lines.extend(render_group(rejects, "No other rejects."))
    lines.extend(["", "## Full Results", ""])
    for row in rows:
        lines.extend(render_result_block(row))
    lines.extend([
        "## Interpretation",
        "",
        "- `hard_reject`: validation average return is nonpositive and PF is below 1.",
        "- `reject`: validation failed core EV/PF/random criteria.",
        "- `research_watchlist`: validation is positive/PF above 1, but it fails candidate rules such as drawdown or baseline comparison.",
        "- `weak_candidate`: validation positive and beats random, but not deterministic.",
        "- `candidate_for_further_validation`: validation strong and test did not contradict.",
        "- `validation_candidate_test_failed`: validation passed, but holdout test failed confirmation.",
        "",
        "Do not tune parameters using test results. Failed candidates are evidence, not errors.",
        "",
    ])
    return "\n".join(lines)


def _random_validation(row: dict[str, Any]) -> dict[str, Any]:
    return (row.get("baselines", {}).get("validation", {}).get("random_same_count") or {})


def _deterministic_validation(row: dict[str, Any]) -> dict[str, Any]:
    return (row.get("baselines", {}).get("validation", {}).get("deterministic") or {})


def _result_title(row: dict[str, Any]) -> str:
    config = row.get("config", {})
    return (
        f"{row.get('classification', 'unknown')}: {config.get('symbol')} {config.get('timeframe')} "
        f"h{config.get('horizon_candles')} RR{config.get('risk_reward')} "
        f"ATR{config.get('atr_stop_multiplier')} {config.get('cost_mode')}"
    )


def render_result_line(row: dict[str, Any]) -> str:
    validation = row.get("validation_metrics", {})
    test = row.get("test_metrics", {})
    diagnostics = row.get("diagnostics", {})
    return (
        f"- `{_result_title(row)}` validation avg/PF/DD "
        f"`{validation.get('avg_return_pct')}`/`{validation.get('profit_factor')}`/`{validation.get('max_drawdown_pct')}`, "
        f"test avg/PF `{test.get('avg_return_pct')}`/`{test.get('profit_factor')}`, "
        f"beats random `{diagnostics.get('beats_random_validation')}`, "
        f"beats deterministic `{diagnostics.get('beats_deterministic_validation')}`"
    )


def render_group(rows: list[dict[str, Any]], empty_message: str) -> list[str]:
    if not rows:
        return [f"- {empty_message}"]
    lines: list[str] = []
    for row in rows:
        lines.extend(render_result_block(row))
    return lines


def render_result_block(row: dict[str, Any]) -> list[str]:
    validation = row.get("validation_metrics", {})
    test = row.get("test_metrics", {})
    random = _random_validation(row)
    deterministic = _deterministic_validation(row)
    diagnostics = row.get("diagnostics", {})
    validation_exposure = diagnostics.get("validation_directional_exposure", {})
    test_exposure = diagnostics.get("test_directional_exposure", {})
    return [
        f"### {_result_title(row)}",
        "",
        f"- experiment_id: `{row.get('experiment_id')}`",
        f"- validation: trades `{validation.get('n_trades')}`, avg `{validation.get('avg_return_pct')}`, PF `{validation.get('profit_factor')}`, DD `{validation.get('max_drawdown_pct')}`",
        f"- test: trades `{test.get('n_trades')}`, avg `{test.get('avg_return_pct')}`, PF `{test.get('profit_factor')}`, DD `{test.get('max_drawdown_pct')}`",
        f"- random validation avg/PF: `{random.get('avg_return_pct')}` / `{random.get('profit_factor')}`",
        f"- deterministic validation avg/PF: `{deterministic.get('avg_return_pct')}` / `{deterministic.get('profit_factor')}`",
        f"- flags: beats_random `{diagnostics.get('beats_random_validation')}`, beats_deterministic `{diagnostics.get('beats_deterministic_validation')}`, validation_positive `{diagnostics.get('validation_positive')}`, test_confirms `{diagnostics.get('test_confirms')}`, high_drawdown `{diagnostics.get('high_drawdown_flag')}`",
        f"- validation direction: BUY `{validation_exposure.get('buy_trades')}`, SELL `{validation_exposure.get('sell_trades')}`, bias `{validation_exposure.get('directional_bias')}`",
        f"- test direction: BUY `{test_exposure.get('buy_trades')}`, SELL `{test_exposure.get('sell_trades')}`, bias `{test_exposure.get('directional_bias')}`",
        f"- reasons: `{row.get('reasons', [])}`",
        "",
    ]
