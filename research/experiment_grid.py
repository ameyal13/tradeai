"""Experiment grid and checkpoint helpers for the local Research Autopilot."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_SYMBOLS = ["ETH", "SOL"]
DEFAULT_TIMEFRAMES = ["1h"]
DEFAULT_HORIZON_CANDLES = [16, 24]
DEFAULT_RISK_REWARDS = [2.0]
DEFAULT_ATR_STOP_MULTIPLIERS = [1.25, 1.5]
DEFAULT_COST_MODES = ["low_costs", "medium_costs_current"]
DEFAULT_STRATEGY_MODES = ["xgboost"]


@dataclass(frozen=True)
class ExperimentConfig:
    symbol: str
    timeframe: str
    horizon_candles: int
    risk_reward: float
    atr_stop_multiplier: float
    cost_mode: str
    strategy_mode: str = "xgboost"
    max_candles: int = 1500
    min_train_rows: int = 120
    buy_threshold: float = 0.58
    sell_threshold: float = 0.58
    trade_label_scheme: str = "expected_value_classification"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["experiment_id"] = experiment_id(payload)
        return payload


def experiment_id(config: ExperimentConfig | dict[str, Any]) -> str:
    payload = asdict(config) if isinstance(config, ExperimentConfig) else dict(config)
    payload.pop("experiment_id", None)
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def build_experiment_grid(
    symbols: list[str] | None = None,
    timeframes: list[str] | None = None,
    horizon_candles: list[int] | None = None,
    risk_rewards: list[float] | None = None,
    atr_stop_multipliers: list[float] | None = None,
    cost_modes: list[str] | None = None,
    strategy_modes: list[str] | None = None,
    max_candles: int = 1500,
    min_train_rows: int = 120,
) -> list[dict[str, Any]]:
    """Build the approved MVP grid. Returned configs include stable ids."""
    configs: list[dict[str, Any]] = []
    for symbol in symbols or DEFAULT_SYMBOLS:
        for timeframe in timeframes or DEFAULT_TIMEFRAMES:
            for horizon in horizon_candles or DEFAULT_HORIZON_CANDLES:
                for rr in risk_rewards or DEFAULT_RISK_REWARDS:
                    for atr_stop in atr_stop_multipliers or DEFAULT_ATR_STOP_MULTIPLIERS:
                        for cost_mode in cost_modes or DEFAULT_COST_MODES:
                            for strategy_mode in strategy_modes or DEFAULT_STRATEGY_MODES:
                                config = ExperimentConfig(
                                    symbol=symbol.upper(),
                                    timeframe=timeframe,
                                    horizon_candles=int(horizon),
                                    risk_reward=float(rr),
                                    atr_stop_multiplier=float(atr_stop),
                                    cost_mode=cost_mode,
                                    strategy_mode=strategy_mode,
                                    max_candles=max_candles,
                                    min_train_rows=min_train_rows,
                                )
                                configs.append(config.to_dict())
    return configs


def save_grid_checkpoint(
    path: str | Path,
    grid: list[dict[str, Any]],
    completed_ids: set[str] | list[str] | None = None,
    result_path: str | None = None,
    markdown_path: str | None = None,
) -> dict[str, Any]:
    checkpoint = {
        "grid": grid,
        "completed_ids": sorted(set(completed_ids or [])),
        "result_path": result_path,
        "markdown_path": markdown_path,
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")
    return checkpoint


def load_grid_checkpoint(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        return {"grid": [], "completed_ids": [], "result_path": None, "markdown_path": None}
    return json.loads(target.read_text(encoding="utf-8"))
