"""Analyze TRADEAI historical experiment reports.

This script reads the aggregate CSV/JSON produced by
scripts/run_historical_experiments.py and creates a research summary. It does
not require Supabase, API keys, frontend, or exchange credentials.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


MIN_EVALUATED = 30
MIN_CANDIDATE_EVALUATED = 50
MIN_DIRECTION_SAMPLE = 30
MIN_HOUR_SAMPLE = 10
HIGH_EXPIRED_RATIO = 0.50
HIGH_DRAWDOWN = 25.0


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_int(value: Any, default: int = 0) -> int:
    return int(parse_float(value, default))


def parse_structured(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if value is None or value == "":
        return {}
    text = str(value).strip()
    if not text:
        return {}
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(text)
        except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
            continue
    return {}


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    numeric_fields = [
        "total_predictions", "evaluated_predictions", "hold_count", "win_rate",
        "average_return", "total_return_pct", "profit_factor", "max_drawdown",
        "sharpe", "tp_hit_count", "sl_hit_count", "expired_count", "buy_count",
        "sell_count", "buy_win_rate", "sell_win_rate", "buy_average_return",
        "sell_average_return", "avg_probability_buy_win",
        "avg_probability_sell_win", "max_probability_buy_win",
        "max_probability_sell_win", "avg_confidence",
        "label_horizon_candles",
    ]
    normalized = dict(row)
    for field in numeric_fields:
        normalized[field] = parse_float(row.get(field))
    normalized["total_predictions"] = parse_int(row.get("total_predictions"))
    normalized["evaluated_predictions"] = parse_int(row.get("evaluated_predictions"))
    normalized["hold_count"] = parse_int(row.get("hold_count"))
    normalized["tp_hit_count"] = parse_int(row.get("tp_hit_count"))
    normalized["sl_hit_count"] = parse_int(row.get("sl_hit_count"))
    normalized["expired_count"] = parse_int(row.get("expired_count"))
    normalized["buy_count"] = parse_int(row.get("buy_count"))
    normalized["sell_count"] = parse_int(row.get("sell_count"))
    normalized["label_horizon_candles"] = parse_int(row.get("label_horizon_candles"))
    normalized["use_trade_labels"] = parse_bool(row.get("use_trade_labels"))
    for field in [
        "hold_reasons_summary", "hourly_performance_utc", "confidence_buckets",
        "feature_nan_summary", "label_params",
    ]:
        normalized[field] = parse_structured(row.get(field))
    evaluated = normalized["evaluated_predictions"]
    normalized["expired_ratio"] = (
        round(normalized["expired_count"] / evaluated, 6)
        if evaluated > 0 else 0.0
    )
    return normalized


def load_report(csv_path: str | Path | None = None, json_path: str | Path | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if csv_path:
        with Path(csv_path).open(newline="", encoding="utf-8") as handle:
            return [normalize_row(row) for row in csv.DictReader(handle)], {"source": str(csv_path)}
    if json_path:
        payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
        return [normalize_row(row) for row in payload.get("summary", [])], {
            "source": str(json_path),
            "config": payload.get("config", {}),
            "generated_at": payload.get("generated_at"),
        }
    raise ValueError("Either csv_path or json_path is required")


def classify_row(row: dict[str, Any]) -> str:
    warnings = str(row.get("warnings") or "").lower()
    evaluated = row["evaluated_predictions"]
    if "historical_data_error" in warnings or (row["total_predictions"] == 0 and warnings):
        return "data_error"
    if evaluated < MIN_EVALUATED:
        return "insufficient_data"
    if (
        evaluated >= MIN_CANDIDATE_EVALUATED
        and row["profit_factor"] > 1.1
        and row["average_return"] > 0
        and row["total_return_pct"] > 0
        and row["max_drawdown"] <= HIGH_DRAWDOWN
    ):
        return "candidate"
    if (
        evaluated >= MIN_EVALUATED
        and 0.9 <= row["profit_factor"] <= 1.1
        and abs(row["average_return"]) <= 0.05
        and row["total_return_pct"] >= -2
    ):
        return "weak_candidate"
    return "reject"


def combo_name(row: dict[str, Any]) -> str:
    mode = row.get("strategy_mode") or "unknown"
    label = row.get("label_type") or "no_label"
    level = row.get("label_level_mode") or "no_level"
    horizon = row.get("label_horizon_candles") or row.get("evaluation_horizon_candles") or "?"
    trade = "trade_labels" if row.get("use_trade_labels") else "price_return_or_rules"
    return f"{row.get('symbol')} {row.get('timeframe')} {mode} {trade} {label}/{level}/h{horizon}"


def analyze_direction(row: dict[str, Any]) -> tuple[list[str], list[str]]:
    findings: list[str] = []
    recommendations: list[str] = []
    buy_count = row["buy_count"]
    sell_count = row["sell_count"]
    if buy_count < MIN_DIRECTION_SAMPLE:
        findings.append(f"BUY sample is small ({buy_count}); do not infer long-only edge.")
    elif row["buy_average_return"] < 0:
        findings.append(f"BUY loses: count={buy_count}, win_rate={row['buy_win_rate']:.2f}, avg_return={row['buy_average_return']:.6f}.")
    elif row["buy_average_return"] > 0:
        findings.append(f"BUY is directionally positive in this sample, but needs walk-forward validation.")
        recommendations.append("Consider a long-only validation run, not production/paper trading yet.")

    if sell_count < MIN_DIRECTION_SAMPLE:
        findings.append(f"SELL sample is small ({sell_count}); do not infer short-only edge.")
    elif row["sell_average_return"] < 0:
        findings.append(f"SELL loses: count={sell_count}, win_rate={row['sell_win_rate']:.2f}, avg_return={row['sell_average_return']:.6f}.")
    elif row["sell_average_return"] > 0:
        findings.append(f"SELL is directionally positive in this sample, but needs walk-forward validation.")
        recommendations.append("Consider a short-only validation run, not production/paper trading yet.")

    if (
        buy_count >= MIN_DIRECTION_SAMPLE
        and sell_count >= MIN_DIRECTION_SAMPLE
        and row["buy_average_return"] < 0
        and row["sell_average_return"] < 0
    ):
        recommendations.append("Both directions lose with enough samples; reject this configuration for now.")
    return findings, recommendations


def analyze_probabilities(row: dict[str, Any]) -> tuple[list[str], list[str]]:
    findings: list[str] = []
    recommendations: list[str] = []
    buy_prob = row["avg_probability_buy_win"]
    sell_prob = row["avg_probability_sell_win"]
    if row["buy_count"] >= MIN_DIRECTION_SAMPLE and buy_prob >= 0.58 and row["buy_average_return"] < 0:
        findings.append(
            f"BUY probability is high on average ({buy_prob:.3f}) but BUY return is negative; calibration/target alignment is suspect."
        )
        recommendations.append("Review probability calibration with --save-trades and compare predicted probability buckets vs realized outcomes.")
    if row["sell_count"] >= MIN_DIRECTION_SAMPLE and sell_prob >= 0.58 and row["sell_average_return"] < 0:
        findings.append(
            f"SELL probability is high on average ({sell_prob:.3f}) but SELL return is negative; calibration/target alignment is suspect."
        )
        recommendations.append("Review SELL calibration before any short-only experiment.")
    confidence_buckets = row.get("confidence_buckets") or {}
    if not confidence_buckets:
        recommendations.append("Confidence bucket detail is missing; run reports with current generator or use --save-trades for calibration.")
    else:
        bad_buckets = [
            name for name, metrics in confidence_buckets.items()
            if isinstance(metrics, dict)
            and parse_int(metrics.get("evaluated_predictions")) >= 10
            and parse_float(metrics.get("average_return")) < 0
        ]
        if bad_buckets:
            findings.append(f"Confidence buckets with enough samples still lose: {', '.join(bad_buckets)}.")
    return findings, recommendations


def analyze_hourly(row: dict[str, Any]) -> tuple[list[str], list[str]]:
    findings: list[str] = []
    recommendations: list[str] = []
    hourly = row.get("hourly_performance_utc") or {}
    best = row.get("best_hour_utc")
    worst = row.get("worst_hour_utc")
    if not hourly:
        recommendations.append("Hourly aggregate is missing; re-run experiments with the current report generator.")
        return findings, recommendations
    if best and isinstance(hourly.get(best), dict):
        samples = parse_int(hourly[best].get("evaluated_predictions"))
        findings.append(f"Best UTC hour by average_return: {best} with {samples} evaluated predictions.")
        if samples < MIN_HOUR_SAMPLE:
            recommendations.append("Do not filter by best_hour_utc yet; sample is too small.")
    if worst and isinstance(hourly.get(worst), dict):
        samples = parse_int(hourly[worst].get("evaluated_predictions"))
        findings.append(f"Worst UTC hour by average_return: {worst} with {samples} evaluated predictions.")
        if samples < MIN_HOUR_SAMPLE:
            recommendations.append("Do not block worst_hour_utc yet; sample is too small.")
    return findings, recommendations


def analyze_hold_reasons(row: dict[str, Any]) -> tuple[list[str], list[str]]:
    findings: list[str] = []
    recommendations: list[str] = []
    reasons = row.get("hold_reasons_summary") or {}
    if not reasons:
        return findings, recommendations
    if "probabilities_below_threshold" in reasons:
        findings.append(f"HOLD is mostly model caution: probabilities_below_threshold={reasons['probabilities_below_threshold']}.")
        recommendations.append("Do not lower thresholds blindly; first test calibration and expected value.")
    for key in ["insufficient_buy_labels", "insufficient_sell_labels", "insufficient_raw_buy_labels", "insufficient_raw_sell_labels"]:
        if key in reasons:
            findings.append(f"Training data bottleneck detected: {key}={reasons[key]}.")
            recommendations.append("Increase candles/max history or inspect label scheme before changing model thresholds.")
    if "no_directional_edge" in reasons:
        findings.append(f"Model saw no clear directional edge in {reasons['no_directional_edge']} predictions.")
    return findings, recommendations


def analyze_row(row: dict[str, Any]) -> dict[str, Any]:
    classification = classify_row(row)
    reasons: list[str] = []
    recommendations: list[str] = []

    warnings = str(row.get("warnings") or "")
    if classification == "data_error":
        reasons.append(f"Data error: {warnings or 'no usable data returned'}.")
        recommendations.extend([
            "Re-run the symbol/timeframe and inspect network/Binance availability.",
            "Keep this combination out of strategy comparison until data loads cleanly.",
        ])
    elif classification == "insufficient_data":
        reasons.append(f"Only {row['evaluated_predictions']} evaluated predictions; below {MIN_EVALUATED}.")
        recommendations.append("Increase max_candles/max_predictions before drawing conclusions.")
    else:
        if row["profit_factor"] < 0.9:
            reasons.append(f"profit_factor {row['profit_factor']:.6f} < 0.9.")
        if row["average_return"] < 0:
            reasons.append(f"average_return is negative ({row['average_return']:.6f}).")
        if row["total_return_pct"] < 0:
            reasons.append(f"total_return_pct is negative ({row['total_return_pct']:.6f}).")
        if row["max_drawdown"] > HIGH_DRAWDOWN:
            reasons.append(f"max_drawdown is high ({row['max_drawdown']:.6f}).")
        if row["expired_ratio"] > HIGH_EXPIRED_RATIO:
            reasons.append(f"expired_ratio is high ({row['expired_ratio']:.2%}).")
            recommendations.append("Compare label schemes: expiry_return, hybrid_touch_or_expiry, expected_value_classification, or longer horizon.")
        if classification == "candidate":
            recommendations.append("Candidate only for more validation; do not paper trade until walk-forward and out-of-sample checks pass.")
        elif classification == "weak_candidate":
            recommendations.append("Weak candidate: preserve for more data and walk-forward comparison.")
        elif classification == "reject":
            recommendations.append("Reject for paper trading; keep only for diagnostics and hypothesis generation.")

    for analyzer in (analyze_direction, analyze_probabilities, analyze_hourly, analyze_hold_reasons):
        extra_findings, extra_recommendations = analyzer(row)
        reasons.extend(extra_findings)
        recommendations.extend(extra_recommendations)

    if row.get("label_type") == "trade_outcome_directional" and row.get("label_level_mode") == "atr":
        recommendations.append("Current ATR trade labels may be too touch/expiry sensitive; test an expiry-aware target next.")
    recommendations.append("Do not declare edge from this aggregate report alone; use --save-trades for probability calibration.")

    deduped_recommendations = list(dict.fromkeys(recommendations))
    return {
        "combo": combo_name(row),
        "classification": classification,
        "metrics": {
            "total_predictions": row["total_predictions"],
            "evaluated_predictions": row["evaluated_predictions"],
            "hold_count": row["hold_count"],
            "win_rate": row["win_rate"],
            "average_return": row["average_return"],
            "total_return_pct": row["total_return_pct"],
            "profit_factor": row["profit_factor"],
            "max_drawdown": row["max_drawdown"],
            "sharpe": row["sharpe"],
            "tp_hit_count": row["tp_hit_count"],
            "sl_hit_count": row["sl_hit_count"],
            "expired_count": row["expired_count"],
            "expired_ratio": row["expired_ratio"],
        },
        "reasons": reasons or ["No strong issue detected by aggregate rules."],
        "recommendations": deduped_recommendations,
        "raw": row,
    }


def analyze_report(rows: list[dict[str, Any]], metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    analyses = [analyze_row(row) for row in rows]
    counts: dict[str, int] = {}
    for item in analyses:
        counts[item["classification"]] = counts.get(item["classification"], 0) + 1
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": (metadata or {}).get("source"),
        "classification_counts": counts,
        "analyses": analyses,
        "global_recommendations": [
            "Do not move any rejected configuration to paper trading.",
            "Next useful experiment: compare current trade_outcome_directional labels against expiry-aware labels.",
            "Use --save-trades to validate probability calibration by predicted-probability buckets.",
            "Do not add neural networks until labels/features show a measurable edge with simpler models.",
        ],
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# TRADEAI Research Summary",
        "",
        f"Generated at: `{summary['generated_at']}`",
        f"Source: `{summary.get('source')}`",
        "",
        "## Classification Counts",
        "",
    ]
    for key in ["candidate", "weak_candidate", "reject", "insufficient_data", "data_error"]:
        lines.append(f"- `{key}`: {summary['classification_counts'].get(key, 0)}")
    lines.extend(["", "## Combinations", ""])

    for item in summary["analyses"]:
        metrics = item["metrics"]
        lines.extend([
            f"### {item['combo']}: {item['classification'].upper()}",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
        ])
        for key, value in metrics.items():
            lines.append(f"| {key} | {value} |")
        lines.extend(["", "Reasons:"])
        lines.extend(f"- {reason}" for reason in item["reasons"])
        lines.extend(["", "Recommendations:"])
        lines.extend(f"- {recommendation}" for recommendation in item["recommendations"])
        lines.append("")

    lines.extend(["## Global Recommendations", ""])
    lines.extend(f"- {item}" for item in summary["global_recommendations"])
    lines.append("")
    return "\n".join(lines)


def print_terminal_summary(summary: dict[str, Any]) -> None:
    print("TRADEAI research summary")
    print(f"Source: {summary.get('source')}")
    print(f"Classifications: {summary['classification_counts']}")
    print()
    for item in summary["analyses"]:
        metrics = item["metrics"]
        print(f"{item['combo']}: {item['classification'].upper()}")
        print(
            f"  evaluated={metrics['evaluated_predictions']} "
            f"pf={metrics['profit_factor']:.6f} "
            f"avg_return={metrics['average_return']:.6f} "
            f"expired_ratio={metrics['expired_ratio']:.2%}"
        )
        print("  Reasons:")
        for reason in item["reasons"][:6]:
            print(f"  - {reason}")
        print("  Recommendations:")
        for recommendation in item["recommendations"][:6]:
            print(f"  - {recommendation}")
        print()


def write_outputs(summary: dict[str, Any], output_dir: str | Path = "reports") -> dict[str, str]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    md_path = target / f"research_summary_{stamp}.md"
    json_path = target / f"research_summary_{stamp}.json"
    md_path.write_text(render_markdown(summary), encoding="utf-8")
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"markdown": str(md_path), "json": str(json_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze TRADEAI historical experiment reports.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", dest="csv_path")
    source.add_argument("--json", dest="json_path")
    parser.add_argument("--output-dir", default="reports")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows, metadata = load_report(csv_path=args.csv_path, json_path=args.json_path)
    summary = analyze_report(rows, metadata)
    paths = write_outputs(summary, args.output_dir)
    print_terminal_summary(summary)
    print(f"Markdown report: {paths['markdown']}")
    print(f"JSON report: {paths['json']}")


if __name__ == "__main__":
    main()
