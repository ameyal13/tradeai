"""Audit stability of Focused Research Grid v2A watchlist configs.

Offline research only. Reads focused_v2A registry/results and writes a
diagnostic report. It does not run experiments, change models, generate
signals, write Supabase, or use test metrics for selection.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_refined_stability import (  # noqa: E402
    build_stability_audit,
    render_markdown,
    utc_stamp,
)
from scripts.summarize_research_registry import enrich_registry_records, load_latest_registry_records  # noqa: E402


DEFAULT_FOCUSED_REGISTRY = PROJECT_ROOT / "reports" / "research_daemon" / "focused_v2a_registry.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "research_daemon"


def save_focused_stability_audit(summary: dict, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Path]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    stamp = utc_stamp()
    json_path = target / f"focused_v2a_stability_audit_{stamp}.json"
    markdown_path = target / f"focused_v2a_stability_audit_{stamp}.md"
    summary["report_type"] = "focused_v2a_stability_audit"
    summary["json_path"] = str(json_path)
    summary["markdown_path"] = str(markdown_path)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(render_markdown(summary).replace("Refined Stability Audit", "Focused v2A Stability Audit"), encoding="utf-8")
    return {"json_path": json_path, "markdown_path": markdown_path}


def run_focused_audit(
    registry_path: str | Path = DEFAULT_FOCUSED_REGISTRY,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    top_n: int = 10,
) -> dict:
    records = enrich_registry_records(load_latest_registry_records(registry_path))
    summary = build_stability_audit(records, top_n=top_n)
    paths = save_focused_stability_audit(summary, output_dir=output_dir)
    summary["json_path"] = str(paths["json_path"])
    summary["markdown_path"] = str(paths["markdown_path"])
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit Focused Research Grid v2A stability.")
    parser.add_argument("--registry", default=str(DEFAULT_FOCUSED_REGISTRY))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--top-n", type=int, default=10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_focused_audit(
        registry_path=args.registry,
        output_dir=args.output_dir,
        top_n=args.top_n,
    )
    print("Focused v2A stability audit generated")
    print("Research only. No trading signal.")
    print(f"audited_configs: {summary['summary']['audited_configs']}")
    print(f"diagnostic_counts: {summary['summary']['diagnostic_counts']}")
    print(f"json: {summary['json_path']}")
    print(f"markdown: {summary['markdown_path']}")
    for row in summary.get("audits", []):
        print(
            f"- {row['label']} | {row['diagnostic_classification']} | "
            f"val_pf={row['median_validation_pf']} | val_avg={row['median_validation_avg_return']} | "
            f"test_confirm={row['test_confirm_rate']} | drawdown={row['worst_validation_drawdown']}"
        )


if __name__ == "__main__":
    main()
