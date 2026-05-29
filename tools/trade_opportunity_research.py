"""Trade opportunity research for TP/SL/cost setup viability.

This module answers a different question than model validation: before trying
more models, does the current trading setup create enough positive outcomes to
be worth modeling?
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from tools.feature_research import add_research_features, build_directional_net_returns
from tools.ml_engine import _atr_label_pcts
from tools.prediction_journal import calculate_profit_factor


DEFAULT_COST_PROFILES = {
    "zero_costs": {"commission_pct": 0.0, "slippage_pct": 0.0, "spread_pct": 0.0},
    "low_costs": {"commission_pct": 0.0004, "slippage_pct": 0.0001, "spread_pct": 0.0001},
    "medium_costs_current": {"commission_pct": 0.001, "slippage_pct": 0.0005, "spread_pct": 0.0003},
    "high_costs_double": {"commission_pct": 0.002, "slippage_pct": 0.001, "spread_pct": 0.0006},
}


def cost_formula_diagnostics(entry: float = 100.0, exit_price: float = 101.5) -> dict[str, Any]:
    """Document the exact evaluator cost formula with BUY/SELL examples."""
    profile = DEFAULT_COST_PROFILES["medium_costs_current"]
    commission_pct = profile["commission_pct"]
    slippage_pct = profile["slippage_pct"]
    spread_pct = profile["spread_pct"]
    fees_paid = abs(entry + exit_price) * commission_pct
    slippage_cost = abs(entry + exit_price) * slippage_pct
    spread_cost = abs(entry + exit_price) * (spread_pct / 2)
    total_cost_pct = (fees_paid + slippage_cost + spread_cost) / entry * 100
    buy_gross = (exit_price - entry) / entry * 100
    sell_gross = (entry - exit_price) / entry * 100
    return {
        "formula": "net_return_pct = gross_return_pct - ((entry + exit_price) * (commission_pct + slippage_pct + spread_pct / 2) / entry * 100)",
        "interpretation": "commission_pct and slippage_pct are modeled per side by multiplying entry+exit; spread_pct is full spread, so half-spread per side also sums across entry+exit.",
        "risk_of_double_counting": "No double subtraction was found in evaluator/replay metrics; costs are embedded once in return_pct. Backtest V2 adjusts execution prices for slippage/spread and separately records slippage_cost for reporting, but does not subtract slippage twice from pnl.",
        "current_profile": profile,
        "example_entry": entry,
        "example_exit_price": exit_price,
        "fees_paid_price_units": round(fees_paid, 8),
        "slippage_cost_price_units": round(slippage_cost, 8),
        "spread_cost_price_units": round(spread_cost, 8),
        "total_cost_pct_of_entry": round(total_cost_pct, 8),
        "buy_example": {
            "gross_return_pct": round(buy_gross, 8),
            "net_return_pct": round(buy_gross - total_cost_pct, 8),
        },
        "sell_example": {
            "gross_return_pct": round(sell_gross, 8),
            "net_return_pct": round(sell_gross - total_cost_pct, 8),
        },
        "recommended_research_profiles": DEFAULT_COST_PROFILES,
    }


def safe_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def profit_metrics(returns: list[float] | np.ndarray) -> dict[str, Any]:
    values = [float(value) for value in returns if safe_float(value) is not None]
    wins = [value for value in values if value > 0]
    losses = [value for value in values if value < 0]
    return {
        "trades": len(values),
        "win_rate": round(len(wins) / len(values) * 100, 6) if values else 0,
        "average_return": round(float(np.mean(values)), 6) if values else 0,
        "total_return_pct": round(float(np.sum(values)), 6) if values else 0,
        "profit_factor": calculate_profit_factor([{"return_pct": value} for value in values]) if values else 0,
        "average_win": round(float(np.mean(wins)), 6) if wins else 0,
        "average_loss": round(float(np.mean(losses)), 6) if losses else 0,
    }


def break_even_win_rate_pct(returns: pd.Series | np.ndarray) -> float | None:
    """Empirical break-even win rate from average win/loss magnitude."""
    values = np.asarray([float(value) for value in returns if safe_float(value) is not None], dtype=float)
    wins = values[values > 0]
    losses = values[values < 0]
    if len(wins) == 0 or len(losses) == 0:
        return None
    avg_win = float(np.mean(wins))
    avg_loss_abs = abs(float(np.mean(losses)))
    if avg_win + avg_loss_abs == 0:
        return None
    return round(avg_loss_abs / (avg_win + avg_loss_abs) * 100, 6)


def positive_rate_pct(returns: pd.Series | np.ndarray) -> float:
    values = np.asarray([float(value) for value in returns if safe_float(value) is not None], dtype=float)
    if len(values) == 0:
        return 0
    return round(float((values > 0).mean() * 100), 6)


def _best_side_returns(buy_returns: np.ndarray, sell_returns: np.ndarray) -> np.ndarray:
    buy = np.asarray(buy_returns, dtype=float)
    sell = np.asarray(sell_returns, dtype=float)
    both_nan = ~np.isfinite(buy) & ~np.isfinite(sell)
    best = np.maximum(
        np.where(np.isfinite(buy), buy, -np.inf),
        np.where(np.isfinite(sell), sell, -np.inf),
    )
    best[both_nan] = np.nan
    return best


def oracle_positive_returns(buy_returns: np.ndarray, sell_returns: np.ndarray) -> list[float]:
    best = _best_side_returns(buy_returns, sell_returns)
    return [float(value) for value in best if np.isfinite(value) and value > 0]


def oracle_top_k_returns(buy_returns: np.ndarray, sell_returns: np.ndarray, top_k: int) -> list[float]:
    best = _best_side_returns(buy_returns, sell_returns)
    values = sorted([float(value) for value in best if np.isfinite(value)], reverse=True)
    return values[: max(0, min(top_k, len(values)))]


def random_same_count_returns(
    buy_returns: np.ndarray,
    sell_returns: np.ndarray,
    trade_count: int,
    seed: int = 42,
    simulations: int = 25,
) -> dict[str, Any]:
    """Random baseline with the same trade count as a model run."""
    valid_positions = np.flatnonzero(np.isfinite(buy_returns) & np.isfinite(sell_returns))
    if len(valid_positions) == 0 or trade_count <= 0:
        return {"simulations": simulations, **profit_metrics([])}
    rng = np.random.default_rng(seed)
    averages = []
    totals = []
    profit_factors = []
    win_rates = []
    counts = []
    for _ in range(simulations):
        size = min(trade_count, len(valid_positions))
        selected = rng.choice(valid_positions, size=size, replace=False)
        sides = rng.integers(0, 2, size=size)
        returns = [
            float(buy_returns[pos]) if side == 0 else float(sell_returns[pos])
            for pos, side in zip(selected, sides)
        ]
        metrics = profit_metrics(returns)
        averages.append(metrics["average_return"])
        totals.append(metrics["total_return_pct"])
        profit_factors.append(metrics["profit_factor"])
        win_rates.append(metrics["win_rate"])
        counts.append(metrics["trades"])
    return {
        "simulations": simulations,
        "trades": int(round(float(np.mean(counts)))) if counts else 0,
        "win_rate": round(float(np.mean(win_rates)), 6) if win_rates else 0,
        "average_return": round(float(np.mean(averages)), 6) if averages else 0,
        "total_return_pct": round(float(np.mean(totals)), 6) if totals else 0,
        "profit_factor": round(float(np.mean(profit_factors)), 6) if profit_factors else 0,
    }


def classify_opportunity(row: dict[str, Any]) -> str:
    oracle = row.get("oracle_top_k") or {}
    random_baseline = row.get("random_same_count") or {}
    best_rate = max(row.get("buy_positive_rate", 0), row.get("sell_positive_rate", 0))
    best_be = min(
        value for value in [
            row.get("buy_break_even_win_rate"),
            row.get("sell_break_even_win_rate"),
        ]
        if value is not None
    ) if row.get("buy_break_even_win_rate") is not None or row.get("sell_break_even_win_rate") is not None else None

    if oracle.get("trades", 0) < 20 or oracle.get("average_return", 0) <= 0:
        return "no_opportunity_detected"
    if best_be is not None and best_rate < best_be and random_baseline.get("average_return", 0) < 0:
        return "oracle_only_opportunity"
    if (
        oracle.get("profit_factor", 0) > 1.5
        and best_be is not None
        and best_rate >= best_be
        and random_baseline.get("average_return", 0) >= -0.02
    ):
        return "modelable_opportunity_candidate"
    if oracle.get("profit_factor", 0) > 1.1 or oracle.get("average_return", 0) > 0:
        return "weak_opportunity"
    return "no_opportunity_detected"


def evaluate_opportunity_config(
    features: pd.DataFrame,
    horizon_candles: int,
    risk_reward: float,
    atr_stop_multiplier: float,
    cost_name: str,
    costs: dict[str, float],
    top_k: int = 80,
    random_trade_count: int = 80,
    min_valid_trades: int = 20,
) -> dict[str, Any]:
    stop_pcts, take_pcts = _atr_label_pcts(
        features,
        atr_stop_multiplier=atr_stop_multiplier,
        min_rr=risk_reward,
        atr_take_profit_multiplier=None,
        fallback_stop_loss_pct=0.03,
        fallback_take_profit_pct=0.03 * risk_reward,
    )
    returns = build_directional_net_returns(
        features,
        horizon_candles=horizon_candles,
        stop_loss_pct=0.03,
        take_profit_pct=0.03 * risk_reward,
        commission_pct=float(costs["commission_pct"]),
        slippage_pct=float(costs["slippage_pct"]),
        spread_pct=float(costs["spread_pct"]),
        stop_loss_pcts=stop_pcts,
        take_profit_pcts=take_pcts,
    )
    buy = returns["buy_return_pct"].to_numpy(dtype=float)
    sell = returns["sell_return_pct"].to_numpy(dtype=float)
    buy_valid = buy[np.isfinite(buy)]
    sell_valid = sell[np.isfinite(sell)]
    oracle_positive = oracle_positive_returns(buy, sell)
    oracle_top = oracle_top_k_returns(buy, sell, top_k=top_k)
    row = {
        "horizon_candles": horizon_candles,
        "risk_reward": risk_reward,
        "atr_stop_multiplier": atr_stop_multiplier,
        "cost_profile": cost_name,
        "commission_pct": float(costs["commission_pct"]),
        "slippage_pct": float(costs["slippage_pct"]),
        "spread_pct": float(costs["spread_pct"]),
        "valid_buy_trades": int(len(buy_valid)),
        "valid_sell_trades": int(len(sell_valid)),
        "buy_positive_rate": positive_rate_pct(buy_valid),
        "sell_positive_rate": positive_rate_pct(sell_valid),
        "buy_break_even_win_rate": break_even_win_rate_pct(buy_valid),
        "sell_break_even_win_rate": break_even_win_rate_pct(sell_valid),
        "buy_positive_minus_break_even": None,
        "sell_positive_minus_break_even": None,
        "always_buy": profit_metrics(buy_valid),
        "always_sell": profit_metrics(sell_valid),
        "oracle_positive": profit_metrics(oracle_positive),
        "oracle_top_k": profit_metrics(oracle_top),
        "random_same_count": random_same_count_returns(
            buy,
            sell,
            trade_count=random_trade_count,
            seed=42 + horizon_candles + int(risk_reward * 100) + int(atr_stop_multiplier * 100),
        ),
        "top_k": top_k,
        "random_trade_count": random_trade_count,
        "min_valid_trades": min_valid_trades,
    }
    if row["buy_break_even_win_rate"] is not None:
        row["buy_positive_minus_break_even"] = round(row["buy_positive_rate"] - row["buy_break_even_win_rate"], 6)
    if row["sell_break_even_win_rate"] is not None:
        row["sell_positive_minus_break_even"] = round(row["sell_positive_rate"] - row["sell_break_even_win_rate"], 6)
    row["classification"] = classify_opportunity(row)
    return row


def summarize_cost_sensitivity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_cost: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_cost.setdefault(row["cost_profile"], []).append(row)
    summary = {}
    for name, group in by_cost.items():
        summary[name] = {
            "best_oracle_top_k_average_return": round(max((row["oracle_top_k"]["average_return"] for row in group), default=0), 6),
            "best_always_buy_average_return": round(max((row["always_buy"]["average_return"] for row in group), default=0), 6),
            "best_always_sell_average_return": round(max((row["always_sell"]["average_return"] for row in group), default=0), 6),
            "modelable_candidates": sum(1 for row in group if row["classification"] == "modelable_opportunity_candidate"),
            "oracle_only": sum(1 for row in group if row["classification"] == "oracle_only_opportunity"),
        }
    return summary


def realistic_candidate_setups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return non-zero-cost candidate rows worth deeper model research."""
    candidates = []
    for row in rows:
        if row["cost_profile"] == "zero_costs":
            continue
        random_avg = row["random_same_count"]["average_return"]
        always_best = max(row["always_buy"]["average_return"], row["always_sell"]["average_return"])
        positive_edge = max(
            row["buy_positive_minus_break_even"] if row["buy_positive_minus_break_even"] is not None else -999,
            row["sell_positive_minus_break_even"] if row["sell_positive_minus_break_even"] is not None else -999,
        )
        if (
            row["classification"] in {"modelable_opportunity_candidate", "weak_opportunity", "oracle_only_opportunity"}
            and row["oracle_top_k"]["average_return"] > 0
            and row["oracle_top_k"]["profit_factor"] > 1.2
            and random_avg <= 0
            and always_best <= row["oracle_top_k"]["average_return"]
            and positive_edge > -15
        ):
            candidates.append(row)
    return sorted(
        candidates,
        key=lambda item: (
            item["classification"] == "modelable_opportunity_candidate",
            item["classification"] == "weak_opportunity",
            item["oracle_top_k"]["average_return"],
            item["oracle_top_k"]["profit_factor"],
        ),
        reverse=True,
    )


def run_trade_opportunity_audit(
    df: pd.DataFrame,
    horizon_values: list[int] | None = None,
    risk_rewards: list[float] | None = None,
    atr_stop_multipliers: list[float] | None = None,
    cost_profiles: dict[str, dict[str, float]] | None = None,
    top_k: int = 80,
    random_trade_count: int = 80,
) -> dict[str, Any]:
    features = add_research_features(df)
    horizons = horizon_values or [4, 8, 12, 16, 24]
    rewards = risk_rewards or [1.0, 1.2, 1.5, 2.0]
    atr_multipliers = atr_stop_multipliers or [0.75, 1.0, 1.25, 1.5]
    costs = cost_profiles or DEFAULT_COST_PROFILES
    rows = []
    for horizon in horizons:
        for reward in rewards:
            for atr_multiplier in atr_multipliers:
                for cost_name, profile in costs.items():
                    rows.append(evaluate_opportunity_config(
                        features,
                        horizon_candles=int(horizon),
                        risk_reward=float(reward),
                        atr_stop_multiplier=float(atr_multiplier),
                        cost_name=cost_name,
                        costs=profile,
                        top_k=top_k,
                        random_trade_count=random_trade_count,
                    ))
    classification_counts: dict[str, int] = {}
    for row in rows:
        classification_counts[row["classification"]] = classification_counts.get(row["classification"], 0) + 1
    best_rows = sorted(
        rows,
        key=lambda item: (
            item["oracle_top_k"]["average_return"],
            item["oracle_top_k"]["profit_factor"],
            item["always_buy"]["average_return"],
            item["always_sell"]["average_return"],
        ),
        reverse=True,
    )[:10]
    current_setup = [
        row for row in rows
        if row["horizon_candles"] == 4
        and abs(row["risk_reward"] - 1.5) < 1e-9
        and abs(row["atr_stop_multiplier"] - 1.5) < 1e-9
        and row["cost_profile"] == "medium_costs_current"
    ]
    if not current_setup:
        current_setup = [
            row for row in rows
            if row["horizon_candles"] == min(horizons)
            and abs(row["risk_reward"] - 1.5) < 1e-9
            and abs(row["atr_stop_multiplier"] - 1.5) < 1e-9
            and row["cost_profile"] == "medium_costs_current"
        ]
    candidates = realistic_candidate_setups(rows)
    return {
        "rows": rows,
        "classification_counts": classification_counts,
        "best_rows": best_rows,
        "realistic_candidate_setups": candidates[:20],
        "current_setup": current_setup[0] if current_setup else None,
        "cost_sensitivity": summarize_cost_sensitivity(rows),
        "cost_formula_diagnostics": cost_formula_diagnostics(),
        "methodology": {
            "entry_delay_candles": 1,
            "entry_price": "open[i+1]",
            "exit_rule": "TP/SL intrabar with conservative same-candle loss, otherwise horizon close",
            "purpose": "Opportunity audit, not model optimization.",
        },
    }
