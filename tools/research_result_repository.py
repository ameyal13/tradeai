"""Research result repository with Supabase sync and local fallback.

This is read-only research telemetry for dashboards. It does not run
experiments, generate signals, or use test metrics for selection.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scripts.summarize_crypto_multi_research import (
    DEFAULT_CRYPTO_MULTI_REGISTRY,
    build_crypto_multi_summary,
)
from scripts.summarize_research_registry import enrich_registry_records, load_latest_registry_records


def _float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def config_label(config: dict[str, Any]) -> str:
    return (
        f"{config.get('symbol')} {config.get('timeframe')} "
        f"h{config.get('horizon_candles')} RR{config.get('risk_reward')} "
        f"ATR{config.get('atr_stop_multiplier')} {config.get('cost_mode')} "
        f"{config.get('strategy_mode')}"
    )


def normalize_research_config_for_store(row: dict[str, Any], source: str = "crypto_multi") -> dict[str, Any]:
    config = row.get("config") or {}
    aggregate = row.get("aggregate") or {}
    return {
        "config_id": str(row.get("config_id")),
        "source": source,
        "status": row.get("status"),
        "classification": row.get("classification") or row.get("result_classification"),
        "symbol": config.get("symbol"),
        "timeframe": config.get("timeframe"),
        "strategy_mode": config.get("strategy_mode"),
        "horizon_candles": _int(config.get("horizon_candles")),
        "risk_reward": _float(config.get("risk_reward")),
        "atr_stop_multiplier": _float(config.get("atr_stop_multiplier")),
        "cost_mode": config.get("cost_mode"),
        "median_validation_pf": _float(aggregate.get("median_validation_pf")),
        "median_validation_avg_return": _float(aggregate.get("median_validation_avg_return")),
        "median_test_pf": _float(aggregate.get("median_test_pf")),
        "test_confirm_rate": _float(aggregate.get("test_confirm_rate")),
        "validation_positive_rate": _float(aggregate.get("validation_positive_rate")),
        "beats_random_rate": _float(aggregate.get("beats_random_rate")),
        "beats_deterministic_rate": _float(aggregate.get("beats_deterministic_rate")),
        "worst_validation_drawdown": _float(aggregate.get("worst_validation_drawdown")),
        "valid_windows": _int(aggregate.get("valid_windows")),
        "label": config_label(config),
        "config": config,
        "metrics": aggregate,
        "raw": row,
    }


class ResearchResultRepository:
    def __init__(
        self,
        supabase_client: Any = None,
        registry_path: str | Path = DEFAULT_CRYPTO_MULTI_REGISTRY,
        source: str = "crypto_multi",
    ):
        self.supabase = supabase_client
        self.registry_path = Path(registry_path)
        self.source = source

    def load_local_records(self) -> list[dict[str, Any]]:
        records = load_latest_registry_records(self.registry_path)
        return enrich_registry_records(records)

    def list_local_configs(self) -> list[dict[str, Any]]:
        return [normalize_research_config_for_store(row, source=self.source) for row in self.load_local_records()]

    def list_configs(self, source: str | None = None, limit: int = 500) -> list[dict[str, Any]]:
        target_source = source or self.source
        if self.supabase is not None:
            try:
                return (
                    self.supabase.table("research_configs")
                    .select("*")
                    .eq("source", target_source)
                    .order("median_validation_pf", desc=True, nullsfirst=False)
                    .limit(max(1, int(limit)))
                    .execute()
                    .data
                    or []
                )
            except Exception:
                pass
        return self.list_local_configs()[: max(0, int(limit))]

    def summary(self, source: str | None = None) -> dict[str, Any]:
        target_source = source or self.source
        configs = self.list_configs(source=target_source, limit=10_000)
        local_like = [
            {
                "config_id": row.get("config_id"),
                "status": row.get("status"),
                "classification": row.get("classification"),
                "config": row.get("config") or {},
                "aggregate": row.get("metrics") or {},
                "json_loaded": True,
            }
            for row in configs
        ]
        summary = build_crypto_multi_summary(local_like, top_limit=15)
        summary["source"] = "supabase" if self.supabase is not None else "local_registry"
        summary["records"] = configs
        return summary

    def sync_local_to_supabase(self) -> dict[str, Any]:
        if self.supabase is None:
            return {"ok": False, "reason": "supabase_not_configured", "configs_upserted": 0}
        if not self.registry_path.exists():
            return {
                "ok": False,
                "reason": "registry_missing",
                "configs_upserted": 0,
                "registry_path": str(self.registry_path),
            }
        configs = self.list_local_configs()
        if configs:
            self.supabase.table("research_configs").upsert(configs, on_conflict="config_id").execute()
        return {
            "ok": True,
            "reason": None,
            "configs_upserted": len(configs),
            "registry_path": str(self.registry_path),
        }


def load_summary_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
