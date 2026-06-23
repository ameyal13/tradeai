"""Run focused Feature Intelligence Audit v1.

Research only. This script audits whether current XGBoost feature families
contain measurable signal. It does not generate shadow signals, write Supabase,
touch journals, or change live strategy behavior.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.feature_research import run_feature_audit  # noqa: E402
from tools.historical_data import fetch_binance_klines  # noqa: E402


DEFAULT_SYMBOLS = ["ADAUSDT", "ETHUSDT"]
DEFAULT_TIMEFRAME = "1h"
DEFAULT_HORIZONS = [4, 10]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "feature_audit"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def normalize_symbol_input(symbol: str) -> str:
    value = str(symbol).upper().replace("/", "").replace("-", "").strip()
    if value.endswith("USDT"):
        value = value[:-4]
    return value


def report_symbol(base_symbol: str) -> str:
    return f"{base_symbol.upper()}USDT"


def finite_number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def sort_metric_map(values: dict[str, Any], by_abs: bool = False) -> list[tuple[str, Any]]:
    def key(item: tuple[str, Any]) -> tuple[int, float]:
        value = item[1]
        if value is None:
            return (1, 0.0)
        numeric = finite_number(value)
        return (0, abs(numeric) if by_abs else numeric)

    return sorted(values.items(), key=key, reverse=True)


def automatic_conclusion(audit: dict[str, Any]) -> dict[str, Any]:
    all_current = audit.get("ablation_results", {}).get("all_current", {})
    dummy = audit.get("ablation_results", {}).get("dummy_random", {})
    trades = int(all_current.get("trades") or 0)
    pf = finite_number(all_current.get("profit_factor"))
    avg = finite_number(all_current.get("average_return"))
    dummy_pf = finite_number(dummy.get("profit_factor"))
    dummy_avg = finite_number(dummy.get("average_return"))
    beats_dummy = avg > dummy_avg and pf > dummy_pf
    family_pfs = [
        finite_number(row.get("profit_factor"))
        for row in audit.get("ablation_results", {}).values()
        if isinstance(row, dict) and row.get("status") == "ok"
    ]
    best_family_pf = max(family_pfs) if family_pfs else 0.0

    if all_current.get("status") != "ok" or trades < 30:
        label = "features débiles"
        reason = "insufficient validated trades to classify feature signal strongly"
    elif avg > 0 and pf > 1.1 and beats_dummy:
        label = "features útiles"
        reason = "all_current has positive EV, PF > 1.1, and beats dummy_random"
    elif pf >= 0.9 or avg > -0.05 or best_family_pf > 1.0:
        label = "features débiles"
        reason = "some feature families are near-flat or above PF 1, but evidence is not strong"
    else:
        label = "probable ruido"
        reason = "current feature families do not show enough out-of-sample edge"

    return {
        "label": label,
        "reason": reason,
        "all_current_profit_factor": pf,
        "all_current_average_return": avg,
        "dummy_profit_factor": dummy_pf,
        "dummy_average_return": dummy_avg,
        "best_family_profit_factor": best_family_pf,
        "trades": trades,
    }


def render_markdown(report: dict[str, Any]) -> str:
    audit = report["audit"]
    all_current = audit.get("ablation_results", {}).get("all_current", {})
    conclusion = report["conclusion"]
    lines = [
        "# Feature Intelligence Audit v1",
        "",
        f"Generated at: `{report['generated_at']}`",
        "",
        "## Guardrails",
        "",
        "- Research only. No trading signal.",
        "- No Supabase writes.",
        "- No shadow journal writes.",
        "- No model, threshold, grid, or live-feature changes.",
        "",
        "## Config",
        "",
        f"- symbol: `{report['symbol']}`",
        f"- timeframe: `{report['timeframe']}`",
        f"- horizon_candles: `{report['horizon_candles']}`",
        f"- rows: `{audit.get('rows')}`",
        f"- label_scheme: `{audit.get('label_scheme')}`",
        f"- label_level_mode: `{audit.get('label_level_mode')}`",
        "",
        "## Automatic Conclusion",
        "",
        f"- classification: `{conclusion['label']}`",
        f"- reason: `{conclusion['reason']}`",
        f"- all_current PF: `{conclusion['all_current_profit_factor']}`",
        f"- all_current avg return: `{conclusion['all_current_average_return']}`",
        f"- dummy_random PF: `{conclusion['dummy_profit_factor']}`",
        f"- dummy_random avg return: `{conclusion['dummy_average_return']}`",
        "",
        "## Feature Correlations To Future Return",
        "",
        "| Feature | Correlation |",
        "| --- | ---: |",
    ]
    for feature, value in sort_metric_map(audit.get("feature_correlations_to_future_return", {}), by_abs=True):
        lines.append(f"| `{feature}` | `{value}` |")

    lines.extend([
        "",
        "## Permutation Importance",
        "",
        "| Feature | Avg Return Delta When Shuffled |",
        "| --- | ---: |",
    ])
    for feature, value in sort_metric_map(all_current.get("permutation_importance", {})):
        lines.append(f"| `{feature}` | `{value}` |")

    lines.extend([
        "",
        "## Model Importance",
        "",
        "| Feature | XGBoost Importance |",
        "| --- | ---: |",
    ])
    for feature, value in sort_metric_map(all_current.get("model_importance", {})):
        lines.append(f"| `{feature}` | `{value}` |")

    lines.extend([
        "",
        "## Ablation Results",
        "",
        "| Family | Status | Trades | Avg Return | Profit Factor |",
        "| --- | --- | ---: | ---: | ---: |",
    ])
    for family, result in audit.get("ablation_results", {}).items():
        lines.append(
            f"| `{family}` | `{result.get('status')}` | `{result.get('trades', 0)}` | "
            f"`{result.get('average_return', 0)}` | `{result.get('profit_factor', 0)}` |"
        )

    lines.extend([
        "",
        "## Removal Candidates",
        "",
        f"`{', '.join(audit.get('removal_candidates') or []) or 'none'}`",
        "",
    ])
    return "\n".join(lines).rstrip() + "\n"


def save_report(report: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    symbol = report_symbol(report["symbol"])
    horizon = int(report["horizon_candles"])
    stamp = utc_stamp()
    json_path = target / f"feature_audit_{symbol}_h{horizon}_{stamp}.json"
    markdown_path = target / f"feature_audit_{symbol}_h{horizon}_{stamp}.md"
    report["report_paths"] = {"json": str(json_path), "markdown": str(markdown_path)}
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return report["report_paths"]


async def audit_symbol(
    symbol: str,
    *,
    timeframe: str,
    max_candles: int,
    horizons: list[int],
    output_dir: str | Path,
) -> list[dict[str, Any]]:
    base_symbol = normalize_symbol_input(symbol)
    candles = await fetch_binance_klines(base_symbol, timeframe, limit=int(max_candles))
    reports: list[dict[str, Any]] = []
    for horizon in horizons:
        audit = run_feature_audit(candles, horizon_candles=int(horizon))
        report = {
            "generated_at": utc_now(),
            "symbol_input": symbol,
            "symbol": base_symbol,
            "timeframe": timeframe,
            "max_candles": int(max_candles),
            "horizon_candles": int(horizon),
            "guardrails": {
                "research_only": True,
                "no_trading_signal": True,
                "no_supabase_writes": True,
                "no_shadow_journal_writes": True,
            },
            "conclusion": automatic_conclusion(audit),
            "audit": audit,
        }
        save_report(report, output_dir)
        reports.append(report)
    return reports


async def run_feature_audit_cli(args: argparse.Namespace) -> dict[str, Any]:
    horizons = args.horizon_candles if args.horizon_candles else DEFAULT_HORIZONS
    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for symbol in args.symbol:
        try:
            results.extend(await audit_symbol(
                symbol,
                timeframe=args.timeframe,
                max_candles=args.max_candles,
                horizons=horizons,
                output_dir=args.output_dir,
            ))
        except Exception as exc:  # noqa: BLE001 - one symbol should not stop the batch.
            errors.append({
                "symbol": symbol,
                "error_type": type(exc).__name__,
                "error": str(exc)[:300],
            })
    return {
        "generated_at": utc_now(),
        "symbols": args.symbol,
        "timeframe": args.timeframe,
        "max_candles": args.max_candles,
        "horizon_candles": horizons,
        "reports": results,
        "errors": errors,
    }


def print_summary(batch: dict[str, Any]) -> None:
    print("Feature Intelligence Audit v1")
    print("Research only. No trading signal.")
    print(f"reports_generated: {len(batch['reports'])}")
    print(f"errors: {len(batch['errors'])}")
    for report in batch["reports"]:
        all_current = report["audit"]["ablation_results"].get("all_current", {})
        print(
            " | ".join([
                f"symbol={report_symbol(report['symbol'])}",
                f"timeframe={report['timeframe']}",
                f"horizon={report['horizon_candles']}",
                f"conclusion={report['conclusion']['label']}",
                f"pf={all_current.get('profit_factor', 0)}",
                f"avg_return={all_current.get('average_return', 0)}",
                f"markdown={report['report_paths']['markdown']}",
            ])
        )
    for error in batch["errors"]:
        print(f"error {error['symbol']}: {error['error_type']} | {error['error']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run focused XGBoost feature intelligence audit.")
    parser.add_argument("--symbol", action="append", default=None, help="Repeatable. Accepts ADA, ADAUSDT, ETH, ETHUSDT.")
    parser.add_argument("--timeframe", default=DEFAULT_TIMEFRAME)
    parser.add_argument("--max-candles", type=int, default=5000)
    parser.add_argument(
        "--horizon-candles",
        type=int,
        action="append",
        default=None,
        help="Repeatable. Defaults to 4 and 10.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser


async def main() -> None:
    args = build_parser().parse_args()
    if args.symbol is None:
        args.symbol = list(DEFAULT_SYMBOLS)
    batch = await run_feature_audit_cli(args)
    print_summary(batch)


if __name__ == "__main__":
    asyncio.run(main())
