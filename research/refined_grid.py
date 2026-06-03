"""Refined research grids derived from global daemon summaries."""
from __future__ import annotations

from typing import Any

from research.experiment_grid import ExperimentConfig
from research.research_registry import config_id


REFINED_SYMBOLS = ["SOL"]
REFINED_TIMEFRAMES = ["1h"]
REFINED_HORIZON_CANDLES = [10, 12, 14, 20]
REFINED_RISK_REWARDS = [2.0, 2.5, 2.8]
REFINED_ATR_STOP_MULTIPLIERS = [1.0, 1.25, 1.5, 1.75]
REFINED_COST_MODES = ["low_costs"]
REFINED_STRATEGY_MODES = ["xgboost"]
REFINED_MAX_CANDLES = 5000
REFINED_WINDOW_SIZE_CANDLES = 600
REFINED_STEP_SIZE_CANDLES = 250


def build_refined_sol_1h_grid(
    max_candles: int = REFINED_MAX_CANDLES,
    window_size_candles: int = REFINED_WINDOW_SIZE_CANDLES,
    step_size_candles: int = REFINED_STEP_SIZE_CANDLES,
    min_train_rows: int = 120,
) -> list[dict[str, Any]]:
    """Build Refined Grid 2A around SOL 1h low_costs watchlist zones."""
    configs: list[dict[str, Any]] = []
    for horizon in REFINED_HORIZON_CANDLES:
        for rr in REFINED_RISK_REWARDS:
            for atr_stop in REFINED_ATR_STOP_MULTIPLIERS:
                for cost_mode in REFINED_COST_MODES:
                    setup = ExperimentConfig(
                        symbol="SOL",
                        timeframe="1h",
                        horizon_candles=int(horizon),
                        risk_reward=float(rr),
                        atr_stop_multiplier=float(atr_stop),
                        cost_mode=cost_mode,
                        strategy_mode="xgboost",
                        max_candles=int(max_candles),
                        min_train_rows=int(min_train_rows),
                    ).to_dict()
                    row = {
                        **setup,
                        "window_size_candles": int(window_size_candles),
                        "step_size_candles": int(step_size_candles),
                        "research_phase": "refined_grid_2a",
                    }
                    row["config_id"] = config_id(row)
                    configs.append(row)
    return configs
