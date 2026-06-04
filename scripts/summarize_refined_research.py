"""Summarize Refined Research Grid 2A results.

Offline research only. Does not run experiments, change models, generate
signals, or use test metrics for selection.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.summarize_research_registry import summarize_registry  # noqa: E402


DEFAULT_REFINED_REGISTRY = PROJECT_ROOT / "reports" / "research_daemon" / "refined_registry.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "research_daemon"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize refined Research Daemon registry.")
    parser.add_argument("--registry", default=str(DEFAULT_REFINED_REGISTRY))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--top-limit", type=int, default=15)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = summarize_registry(
        registry_path=args.registry,
        output_dir=args.output_dir,
        filename_prefix="refined_global_summary",
        top_limit=args.top_limit,
        refined=True,
    )
    overview = summary["summary"]
    print("Refined research global summary generated")
    print(f"total_configs: {overview['unique_configs']}")
    print(f"completed: {overview['unique_completed_configs']}")
    print(f"classification_counts: {overview['classification_counts']}")
    print(f"json: {summary['json_path']}")
    print(f"markdown: {summary['markdown_path']}")


if __name__ == "__main__":
    main()
