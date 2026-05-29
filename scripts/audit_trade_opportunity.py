"""Audit whether TP/SL/horizon/cost rules contain exploitable opportunity.

This is an offline research script. It does not train models, does not place
trades, and does not persist to Supabase.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_historical_experiments import (  # noqa: E402
    DEFAULT_CACHE_DIR,
    effective_horizon_minutes,
    horizon_candles_for_interval,
    load_experiment_candles,
)
from tools.historical_replay import run_historical_replay  # noqa: E402
from tools.trade_opportunity_research import run_trade_opportunity_audit  # noqa: E402


def first_metric(result: dict[str, Any]) -> dict[str, Any]:
    metrics = result.get("metrics") or []
    return metrics[0] if metrics else {}


def deterministic_current_setup(
    candles,
    symbol: str,
    timeframe: str,
    horizon_candles: int,
    horizon_minutes: int,
    max_predictions: int,
) -> dict[str, Any]:
    result = run_historical_replay(
        candles,
        symbol=symbol,
        timeframe=timeframe,
        strategy_mode="deterministic",
        horizon_candles=horizon_candles,
        horizon_minutes=horizon_minutes,
        max_predictions=max_predictions,
        min_history=50,
        strategy_params={
            "min_risk_reward": 1.5,
            "atr_stop_multiplier": 1.5,
            "use_sentiment": False,
        },
    )
    metric = first_metric(result)
    return {
        "total_predictions": len(result.get("predictions") or []),
        "evaluated_predictions": metric.get("evaluated_predictions", len(result.get("outcomes") or [])),
        "win_rate": metric.get("win_rate", 0),
        "average_return": metric.get("average_return", 0),
        "total_return_pct": metric.get("total_return_pct", 0),
        "profit_factor": metric.get("profit_factor", 0),
        "max_drawdown": metric.get("max_drawdown", 0),
        "sharpe": metric.get("sharpe", 0),
    }


def parse_float_list(values: str) -> list[float]:
    return [float(item.strip()) for item in values.split(",") if item.strip()]


def parse_int_list(values: str) -> list[int]:
    return [int(item.strip()) for item in values.split(",") if item.strip()]


def render_markdown(report: dict[str, Any]) -> str:
    cost_diag = report.get("cost_formula_diagnostics", {})
    lines = [
        "# Trade Opportunity Audit",
        "",
        f"Generated at: `{report['generated_at']}`",
        "",
        "## Cost Diagnostics",
        "",
        f"- Formula: `{cost_diag.get('formula')}`",
        f"- Interpretation: {cost_diag.get('interpretation')}",
        f"- Double-counting assessment: {cost_diag.get('risk_of_double_counting')}",
        f"- Current medium profile: `{cost_diag.get('current_profile')}`",
        f"- Example entry/exit: `{cost_diag.get('example_entry')}` -> `{cost_diag.get('example_exit_price')}`",
        f"- Total cost pct of entry: `{cost_diag.get('total_cost_pct_of_entry')}`",
        f"- BUY example: `{cost_diag.get('buy_example')}`",
        f"- SELL example: `{cost_diag.get('sell_example')}`",
        "",
        "## Method",
        "",
        "- Measures opportunity under TP/SL/horizon/cost rules before model changes.",
        "- Uses delayed entry: signal at N, entry at open of N+1.",
        "- Oracle baselines are upper bounds, not deployable strategies.",
        "- Random baseline uses the same configured trade count as the model comparison target.",
        "",
        "## Summary",
        "",
    ]
    for item in report["results"]:
        if item.get("error"):
            lines.extend([
                f"### {item['symbol']} {item['timeframe']}: DATA ERROR",
                "",
                f"- Error: `{item['error']}`",
                "",
            ])
            continue
        current = item["audit"]["current_setup"] or {}
        deterministic = item["deterministic_current_setup"]
        counts = item["audit"]["classification_counts"]
        realistic_candidates = item["audit"].get("realistic_candidate_setups") or []
        lines.extend([
            f"### {item['symbol']} {item['timeframe']}",
            "",
            f"- Data source: `{item['data_source']}`",
            f"- Effective current horizon minutes: `{item['effective_horizon_minutes']}`",
            f"- Classification counts: `{counts}`",
            f"- Current setup classification: `{current.get('classification')}`",
            f"- BUY positive rate vs BE: `{current.get('buy_positive_rate')}` vs `{current.get('buy_break_even_win_rate')}`",
            f"- SELL positive rate vs BE: `{current.get('sell_positive_rate')}` vs `{current.get('sell_break_even_win_rate')}`",
            f"- Always BUY avg/PF: `{(current.get('always_buy') or {}).get('average_return')}` / `{(current.get('always_buy') or {}).get('profit_factor')}`",
            f"- Always SELL avg/PF: `{(current.get('always_sell') or {}).get('average_return')}` / `{(current.get('always_sell') or {}).get('profit_factor')}`",
            f"- Oracle positive trades: `{(current.get('oracle_positive') or {}).get('trades')}`",
            f"- Oracle top-k avg/PF: `{(current.get('oracle_top_k') or {}).get('average_return')}` / `{(current.get('oracle_top_k') or {}).get('profit_factor')}`",
            f"- Random same-count avg/PF: `{(current.get('random_same_count') or {}).get('average_return')}` / `{(current.get('random_same_count') or {}).get('profit_factor')}`",
            f"- Deterministic avg/PF: `{deterministic.get('average_return')}` / `{deterministic.get('profit_factor')}`",
            f"- Realistic candidate setups: `{len(realistic_candidates)}`",
            "",
            "Best sensitivity rows by oracle top-k:",
            "",
        ])
        for row in item["audit"]["best_rows"][:5]:
            lines.append(
                f"- horizon `{row['horizon_candles']}`, RR `{row['risk_reward']}`, ATR stop `{row['atr_stop_multiplier']}`, "
                f"cost `{row['cost_profile']}`: class `{row['classification']}`, "
                f"BUY rate/BE `{row['buy_positive_rate']}`/`{row['buy_break_even_win_rate']}`, "
                f"SELL rate/BE `{row['sell_positive_rate']}`/`{row['sell_break_even_win_rate']}`, "
                f"oracle top-k avg/PF `{row['oracle_top_k']['average_return']}`/`{row['oracle_top_k']['profit_factor']}`"
            )
        lines.extend([
            "",
            "Cost sensitivity:",
            "",
        ])
        for name, summary in item["audit"]["cost_sensitivity"].items():
            lines.append(f"- `{name}`: `{summary}`")
        lines.extend(["", "Realistic candidate setups:", ""])
        if realistic_candidates:
            for row in realistic_candidates[:8]:
                lines.append(
                    f"- horizon `{row['horizon_candles']}`, RR `{row['risk_reward']}`, ATR stop `{row['atr_stop_multiplier']}`, "
                    f"cost `{row['cost_profile']}`: class `{row['classification']}`, "
                    f"BUY edge `{row['buy_positive_minus_break_even']}`, SELL edge `{row['sell_positive_minus_break_even']}`, "
                    f"oracle avg/PF `{row['oracle_top_k']['average_return']}`/`{row['oracle_top_k']['profit_factor']}`, "
                    f"random avg `{row['random_same_count']['average_return']}`"
                )
        else:
            lines.append("- None under non-zero cost tiers.")
        lines.append("")
    lines.extend([
        "## Interpretation",
        "",
        "- `no_opportunity_detected`: even favorable selection has too little usable positive return.",
        "- `oracle_only_opportunity`: winners exist, but base rates/random behavior imply only a very strong selector could exploit them.",
        "- `weak_opportunity`: there may be a fragile zone, but it is not enough for paper trading.",
        "- `modelable_opportunity_candidate`: only a research candidate; still requires feature/model validation out of sample.",
        "",
        "If only `no_costs` looks viable, the bottleneck is friction. If longer horizons dominate, the current 4-candle setup is likely too short. If oracle is strong but random/always lose, the next problem is feature/model selection.",
        "",
        "Cost tiers:",
        "",
        "- `zero_costs`: diagnostic only.",
        "- `low_costs`: optimistic liquid-market research tier.",
        "- `medium_costs_current`: current conservative setup.",
        "- `high_costs_double`: stress test.",
        "",
    ])
    return "\n".join(lines)


def write_report(report: dict[str, Any], reports_dir: str | Path) -> dict[str, str]:
    target = Path(reports_dir)
    target.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = target / f"trade_opportunity_audit_{stamp}.json"
    md_path = target / f"trade_opportunity_audit_{stamp}.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


async def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    results = []
    horizons = parse_int_list(args.horizons)
    risk_rewards = parse_float_list(args.risk_rewards)
    atr_stop_multipliers = parse_float_list(args.atr_stop_multipliers)
    for symbol in [value.upper() for value in args.symbols]:
        for timeframe in args.timeframes:
            current_horizon_candles = horizon_candles_for_interval(args.horizon_minutes, timeframe)
            current_horizon_minutes = effective_horizon_minutes(args.horizon_minutes, timeframe)
            try:
                loaded = await load_experiment_candles(
                    symbol,
                    timeframe,
                    max_candles=args.max_candles,
                    use_cache=args.use_cache,
                    refresh_cache=args.refresh_cache,
                    cache_dir=args.cache_dir,
                )
                candles = loaded["candles"]
                audit = run_trade_opportunity_audit(
                    candles,
                    horizon_values=horizons,
                    risk_rewards=risk_rewards,
                    atr_stop_multipliers=atr_stop_multipliers,
                    top_k=args.top_k,
                    random_trade_count=args.random_trade_count,
                )
                deterministic = deterministic_current_setup(
                    candles,
                    symbol=symbol,
                    timeframe=timeframe,
                    horizon_candles=current_horizon_candles,
                    horizon_minutes=current_horizon_minutes,
                    max_predictions=args.max_predictions,
                )
                results.append({
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "data_source": loaded.get("data_source", ""),
                    "data_cache_path": loaded.get("data_cache_path", ""),
                    "data_warning": loaded.get("data_warning", ""),
                    "current_horizon_candles": current_horizon_candles,
                    "effective_horizon_minutes": current_horizon_minutes,
                    "audit": audit,
                    "deterministic_current_setup": deterministic,
                })
            except Exception as exc:  # noqa: BLE001
                results.append({
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "error": f"{exc.__class__.__name__}: {exc}",
                })
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "symbols": [value.upper() for value in args.symbols],
            "timeframes": args.timeframes,
            "horizon_minutes": args.horizon_minutes,
            "horizons": horizons,
            "risk_rewards": risk_rewards,
            "atr_stop_multipliers": atr_stop_multipliers,
            "max_candles": args.max_candles,
            "max_predictions": args.max_predictions,
            "top_k": args.top_k,
            "random_trade_count": args.random_trade_count,
            "use_cache": args.use_cache,
            "refresh_cache": args.refresh_cache,
            "cache_dir": str(args.cache_dir),
        },
        "results": results,
    }
    first_audit = next((item.get("audit") for item in results if item.get("audit")), None)
    report["cost_formula_diagnostics"] = (first_audit or {}).get("cost_formula_diagnostics", {})
    report["report_paths"] = write_report(report, args.reports_dir)
    return report


def print_summary(report: dict[str, Any]) -> None:
    print("symbol | timeframe | current_class | candidates | buy_rate/be | sell_rate/be | oracle_top_avg | random_avg | deterministic_avg")
    print("-------+-----------+---------------+------------+-------------+--------------+----------------+------------+------------------")
    for item in report["results"]:
        if item.get("error"):
            print(f"{item['symbol']} | {item['timeframe']} | data_error | 0/0 | 0/0 | 0 | 0 | 0")
            continue
        current = item["audit"]["current_setup"] or {}
        deterministic = item["deterministic_current_setup"]
        candidates = item["audit"].get("realistic_candidate_setups") or []
        print(
            f"{item['symbol']} | {item['timeframe']} | {current.get('classification')} | {len(candidates)} | "
            f"{current.get('buy_positive_rate')}/{current.get('buy_break_even_win_rate')} | "
            f"{current.get('sell_positive_rate')}/{current.get('sell_break_even_win_rate')} | "
            f"{(current.get('oracle_top_k') or {}).get('average_return')} | "
            f"{(current.get('random_same_count') or {}).get('average_return')} | "
            f"{deterministic.get('average_return')}"
        )
    print(f"Markdown report: {report['report_paths']['markdown']}")
    print(f"JSON report: {report['report_paths']['json']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit trade opportunity before model changes.")
    parser.add_argument("--symbols", nargs="+", default=["BTC", "ETH", "SOL"])
    parser.add_argument("--timeframes", nargs="+", default=["15m", "1h"])
    parser.add_argument("--max-candles", type=int, default=1500)
    parser.add_argument("--max-predictions", type=int, default=200)
    parser.add_argument("--horizon-minutes", type=int, default=60)
    parser.add_argument("--horizons", default="8,12,16,24")
    parser.add_argument("--risk-rewards", default="1.5,2.0")
    parser.add_argument("--atr-stop-multipliers", default="1.25,1.5")
    parser.add_argument("--top-k", type=int, default=80)
    parser.add_argument("--random-trade-count", type=int, default=80)
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--no-cache", action="store_false", dest="use_cache", default=True)
    parser.add_argument("--refresh-cache", action="store_true")
    return parser


async def main() -> None:
    report = await run_audit(build_parser().parse_args())
    print_summary(report)


if __name__ == "__main__":
    asyncio.run(main())
