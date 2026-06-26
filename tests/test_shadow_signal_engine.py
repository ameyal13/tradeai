import json
import os
import tempfile
import unittest
from io import StringIO
from datetime import timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pandas as pd

from research.asset_universe import crypto_universe
from research.signal_review_agent import SignalReviewRequest, SignalReviewResponse, review_shadow_signal
from research.telegram_notifier import (
    format_shadow_daily_summary,
    format_shadow_signal_evaluated,
    format_shadow_signal_opened,
)
from scripts.generate_shadow_signals_once import (
    build_parser as build_generate_parser,
    classify_signal_skip,
    generate_shadow_signals_once,
    load_candidate_configs,
    print_rows,
    registry_path_from_choice,
    summarize_generation_rows,
)
from scripts.evaluate_shadow_signals_once import build_parser as build_evaluate_parser
from tools.prediction_journal import utc_now
from tools.shadow_signal_journal import (
    OPEN,
    ShadowSignalJournal,
    build_shadow_signal_from_strategy,
    cost_profile_for_config,
    evaluate_shadow_signal_with_candles,
    horizon_minutes_from_candles,
)


class FakeStrategySignal:
    def __init__(self, signal="BUY"):
        self.signal = signal

    def to_dict(self):
        return {
            "strategy_mode": "xgboost",
            "strategy_name": "xgboost_temporal_split",
            "strategy_version": "v1",
            "signal": self.signal,
            "confidence": 62.0,
            "entry_price": 100.0,
            "stop_loss": 95.0,
            "take_profit": 110.0,
            "risk_reward_ratio": 2.0,
            "horizon_minutes": 720,
            "input_features": {"model_available": True},
            "reasoning": "test signal",
            "model_provider": "local_xgboost",
            "model_name": "xgboost_classifier_v1",
        }


class DictStrategySignal:
    def __init__(self, payload):
        self.payload = payload

    def to_dict(self):
        return dict(self.payload)


class FakeNewsContext:
    def __init__(self, payload):
        self.payload = payload

    def model_dump(self):
        return dict(self.payload)


class FakeMarketContext:
    def __init__(self, payload):
        self.payload = payload

    def model_dump(self):
        return dict(self.payload)


def registry_row(config_id="cfg1", classification="unstable_watchlist"):
    return {
        "config_id": config_id,
        "status": "completed",
        "classification": classification,
        "config": {
            "config_id": config_id,
            "experiment_id": "exp1",
            "symbol": "SOL",
            "timeframe": "1h",
            "strategy_mode": "xgboost",
            "horizon_candles": 12,
            "risk_reward": 2.0,
            "atr_stop_multiplier": 1.25,
            "cost_mode": "low_costs",
            "max_candles": 500,
            "min_train_rows": 120,
            "buy_threshold": 0.58,
            "sell_threshold": 0.58,
            "trade_label_scheme": "expected_value_classification",
        },
        "json_path": None,
        "markdown_path": None,
    }


def write_registry(path: Path, rows: list[dict]):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def sample_candles(start, rows):
    idx = pd.date_range(start, periods=rows, freq="1h", tz="UTC")
    return pd.DataFrame({
        "timestamp": idx,
        "open": [100.0] * rows,
        "high": [101.0] * rows,
        "low": [99.0] * rows,
        "close": [100.0] * rows,
        "volume": [1000.0] * rows,
    })


class ShadowSignalEngineTests(unittest.IsolatedAsyncioTestCase):
    async def test_no_signals_without_stable_or_allow_watchlist(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "registry.jsonl"
            journal = Path(tmp) / "shadow.jsonl"
            write_registry(registry, [registry_row(classification="unstable_watchlist")])

            rows = await generate_shadow_signals_once(
                registry=str(registry),
                journal_path=journal,
                allow_watchlist_shadow=False,
                max_signals=1,
            )

        self.assertEqual(rows[0]["status"], "skipped_not_allowed")
        self.assertIn("--allow-watchlist-shadow", rows[0]["skip_reason"])

    async def test_allow_watchlist_generates_marked_shadow_signal(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "registry.jsonl"
            journal = Path(tmp) / "shadow.jsonl"
            write_registry(registry, [registry_row(classification="unstable_watchlist")])
            with patch("scripts.generate_shadow_signals_once.load_experiment_candles", new=AsyncMock(return_value={
                "candles": sample_candles("2026-01-01", 300),
            })):
                with patch("scripts.generate_shadow_signals_once.generate_strategy_signal_from_df", return_value=FakeStrategySignal()):
                    rows = await generate_shadow_signals_once(
                        registry=str(registry),
                        journal_path=journal,
                        allow_watchlist_shadow=True,
                        max_signals=1,
                        notify_telegram=False,
                        refresh_cache=False,
                    )

        self.assertEqual(rows[0]["status"], OPEN)
        self.assertTrue(rows[0]["watchlist_shadow"])
        self.assertIn("WATCHLIST SHADOW ONLY", rows[0]["notes"])
        self.assertEqual(rows[0]["horizon_candles"], 12)
        self.assertEqual(rows[0]["horizon_minutes"], 720)

    async def test_duplicate_open_signal_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "registry.jsonl"
            journal_path = Path(tmp) / "shadow.jsonl"
            write_registry(registry, [registry_row(config_id="cfg1", classification="unstable_watchlist")])
            journal = ShadowSignalJournal(journal_path)
            journal.create_signal({
                "shadow_signal_id": "sig1",
                "config_id": "cfg1",
                "symbol": "SOL",
                "timeframe": "1h",
                "status": OPEN,
                "generated_at": utc_now().isoformat(),
                "expires_at": (utc_now() + timedelta(hours=12)).isoformat(),
                "research_only": True,
            })

            rows = await generate_shadow_signals_once(
                registry=str(registry),
                journal_path=journal_path,
                allow_watchlist_shadow=True,
                max_signals=1,
            )

        self.assertEqual(rows[0]["status"], "skipped_duplicate_open")

    async def test_similar_open_signal_from_different_config_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "registry.jsonl"
            journal = Path(tmp) / "shadow.jsonl"
            write_registry(registry, [
                registry_row(config_id="cfg1", classification="unstable_watchlist"),
                registry_row(config_id="cfg2", classification="unstable_watchlist"),
            ])
            with patch("scripts.generate_shadow_signals_once.load_experiment_candles", new=AsyncMock(return_value={
                "candles": sample_candles("2026-01-01", 300),
            })):
                with patch("scripts.generate_shadow_signals_once.generate_strategy_signal_from_df", return_value=FakeStrategySignal()):
                    rows = await generate_shadow_signals_once(
                        registry=str(registry),
                        journal_path=journal,
                        allow_watchlist_shadow=True,
                        max_signals=2,
                        max_configs_scanned=2,
                        dry_run=True,
                        refresh_cache=False,
                    )

        summary = summarize_generation_rows(rows, journal_path=journal)
        self.assertEqual(rows[0]["status"], OPEN)
        self.assertEqual(rows[1]["status"], "skipped_duplicate_open_similar")
        self.assertEqual(rows[1]["duplicate_reason"], "same_symbol_timeframe_side_entry_stop_take_profit")
        self.assertEqual(summary["opened_signals"], 1)
        self.assertEqual(summary["skipped_duplicate_similar"], 1)
        self.assertFalse(journal.exists())

    async def test_distinct_side_or_levels_are_allowed(self):
        buy = FakeStrategySignal(signal="BUY").to_dict()
        sell = FakeStrategySignal(signal="SELL").to_dict()
        sell["entry_price"] = 100.0
        sell["stop_loss"] = 105.0
        sell["take_profit"] = 90.0
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "registry.jsonl"
            journal = Path(tmp) / "shadow.jsonl"
            write_registry(registry, [
                registry_row(config_id="cfg1", classification="unstable_watchlist"),
                registry_row(config_id="cfg2", classification="unstable_watchlist"),
            ])
            with patch("scripts.generate_shadow_signals_once.load_experiment_candles", new=AsyncMock(return_value={
                "candles": sample_candles("2026-01-01", 300),
            })):
                with patch(
                    "scripts.generate_shadow_signals_once.generate_strategy_signal_from_df",
                    side_effect=[DictStrategySignal(buy), DictStrategySignal(sell)],
                ):
                    rows = await generate_shadow_signals_once(
                        registry=str(registry),
                        journal_path=journal,
                        allow_watchlist_shadow=True,
                        max_signals=2,
                        max_configs_scanned=2,
                        dry_run=True,
                        refresh_cache=False,
                    )

        self.assertEqual([row["status"] for row in rows], [OPEN, OPEN])
        self.assertEqual(rows[0]["side"], "LONG")
        self.assertEqual(rows[1]["side"], "SHORT")
        self.assertFalse(journal.exists())

    async def test_similar_duplicate_does_not_send_second_telegram(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "registry.jsonl"
            journal = Path(tmp) / "shadow.jsonl"
            write_registry(registry, [
                registry_row(config_id="cfg1", classification="unstable_watchlist"),
                registry_row(config_id="cfg2", classification="unstable_watchlist"),
            ])
            with patch("scripts.generate_shadow_signals_once.load_experiment_candles", new=AsyncMock(return_value={
                "candles": sample_candles("2026-01-01", 300),
            })):
                with patch("scripts.generate_shadow_signals_once.generate_strategy_signal_from_df", return_value=FakeStrategySignal()):
                    with patch("scripts.generate_shadow_signals_once.send_telegram_message") as send:
                        rows = await generate_shadow_signals_once(
                            registry=str(registry),
                            journal_path=journal,
                            allow_watchlist_shadow=True,
                            max_signals=2,
                            max_configs_scanned=2,
                            dry_run=False,
                            notify_telegram=True,
                            refresh_cache=False,
                        )
            latest = ShadowSignalJournal(journal).list_signals(status=OPEN)

        self.assertEqual([row["status"] for row in rows], [OPEN, "skipped_duplicate_open_similar"])
        self.assertEqual(send.call_count, 1)
        self.assertEqual(len(latest), 1)

    async def test_block_review_marks_signal_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "registry.jsonl"
            journal = Path(tmp) / "shadow.jsonl"
            write_registry(registry, [registry_row(classification="unstable_watchlist")])
            with patch("scripts.generate_shadow_signals_once.load_experiment_candles", new=AsyncMock(return_value={
                "candles": sample_candles("2026-01-01", 300),
            })):
                with patch("scripts.generate_shadow_signals_once.generate_strategy_signal_from_df", return_value=FakeStrategySignal()):
                    with patch("scripts.generate_shadow_signals_once.review_shadow_signal", return_value=SignalReviewResponse(review_status="BLOCK", risk_flags=["risk"])):
                        rows = await generate_shadow_signals_once(
                            registry=str(registry),
                            journal_path=journal,
                            allow_watchlist_shadow=True,
                            max_signals=1,
                            refresh_cache=False,
                        )

        self.assertEqual(rows[0]["status"], "skipped_agent_block")
        self.assertEqual(rows[0]["journal_status"], "BLOCKED")
        self.assertIn("BLOCKED", rows[0]["notes"])

    async def test_news_context_is_passed_to_agent_review_when_enabled(self):
        news_payload = {
            "symbol": "SOL",
            "risk_score": 50,
            "sentiment_score": -0.4,
            "item_count": 2,
            "risk_flags": ["negative_symbol_news"],
        }
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "registry.jsonl"
            journal = Path(tmp) / "shadow.jsonl"
            write_registry(registry, [registry_row(classification="unstable_watchlist")])
            with patch("scripts.generate_shadow_signals_once.load_experiment_candles", new=AsyncMock(return_value={
                "candles": sample_candles("2026-01-01", 300),
            })):
                with patch("scripts.generate_shadow_signals_once.generate_strategy_signal_from_df", return_value=FakeStrategySignal()):
                    with patch(
                        "scripts.generate_shadow_signals_once.build_news_context",
                        new=AsyncMock(return_value=FakeNewsContext(news_payload)),
                    ) as news:
                        rows = await generate_shadow_signals_once(
                            registry=str(registry),
                            journal_path=journal,
                            allow_watchlist_shadow=True,
                            max_signals=1,
                            dry_run=True,
                            refresh_cache=False,
                            use_news_context=True,
                        )

        news.assert_awaited_once_with("SOL")
        self.assertEqual(rows[0]["status"], OPEN)
        self.assertEqual(rows[0]["agent_review"]["review_status"], "CAUTION")
        self.assertEqual(rows[0]["news_context"]["risk_flags"], ["negative_symbol_news"])

    async def test_market_context_is_passed_to_agent_review_when_enabled(self):
        market_payload = {
            "context_status": "BLOCK",
            "confidence_adjustment": -10,
            "risk_flags": ["long_against_local_trend", "long_near_resistance"],
            "context_summary": "market context test",
            "metrics": {"trend": "bearish"},
            "can_modify_trade_levels": False,
            "research_only": True,
        }
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "registry.jsonl"
            journal = Path(tmp) / "shadow.jsonl"
            write_registry(registry, [registry_row(classification="unstable_watchlist")])
            with patch("scripts.generate_shadow_signals_once.load_experiment_candles", new=AsyncMock(return_value={
                "candles": sample_candles("2026-01-01", 300),
            })):
                with patch("scripts.generate_shadow_signals_once.generate_strategy_signal_from_df", return_value=FakeStrategySignal()):
                    with patch(
                        "scripts.generate_shadow_signals_once.build_market_context",
                        return_value=FakeMarketContext(market_payload),
                    ) as market:
                        rows = await generate_shadow_signals_once(
                            registry=str(registry),
                            journal_path=journal,
                            allow_watchlist_shadow=True,
                            max_signals=1,
                            dry_run=True,
                            refresh_cache=False,
                            use_market_context=True,
                        )

        market.assert_called_once()
        self.assertEqual(rows[0]["status"], "skipped_agent_block")
        self.assertEqual(rows[0]["market_context"]["context_status"], "BLOCK")
        self.assertFalse(rows[0]["agent_review"]["can_modify_trade_levels"])

    async def test_hold_signal_reports_explicit_hold_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "registry.jsonl"
            journal = Path(tmp) / "shadow.jsonl"
            write_registry(registry, [registry_row(classification="unstable_watchlist")])
            hold = FakeStrategySignal(signal="HOLD").to_dict()
            hold["input_features"] = {
                "hold_reason": "probabilities_below_threshold",
                "probability_buy_win": 0.41,
                "probability_sell_win": 0.39,
            }
            with patch("scripts.generate_shadow_signals_once.load_experiment_candles", new=AsyncMock(return_value={
                "candles": sample_candles("2026-01-01", 300),
            })):
                with patch("scripts.generate_shadow_signals_once.generate_strategy_signal_from_df", return_value=FakeStrategySignal(signal="HOLD")) as gen:
                    gen.return_value.to_dict = lambda: hold
                    rows = await generate_shadow_signals_once(
                        registry=str(registry),
                        journal_path=journal,
                        allow_watchlist_shadow=True,
                        max_signals=1,
                        dry_run=True,
                        refresh_cache=False,
                    )

        self.assertEqual(rows[0]["status"], "skipped_hold")
        self.assertEqual(rows[0]["hold_reason"], "probabilities_below_threshold")
        self.assertEqual(rows[0]["probability_buy_win"], 0.41)
        self.assertFalse(journal.exists())

    async def test_max_configs_limits_hold_iterations(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "registry.jsonl"
            journal = Path(tmp) / "shadow.jsonl"
            write_registry(registry, [
                registry_row(config_id="cfg1", classification="unstable_watchlist"),
                registry_row(config_id="cfg2", classification="unstable_watchlist"),
            ])
            with patch("scripts.generate_shadow_signals_once.load_experiment_candles", new=AsyncMock(return_value={
                "candles": sample_candles("2026-01-01", 300),
            })):
                with patch("scripts.generate_shadow_signals_once.generate_strategy_signal_from_df", return_value=FakeStrategySignal(signal="HOLD")):
                    rows = await generate_shadow_signals_once(
                        registry=str(registry),
                        journal_path=journal,
                        allow_watchlist_shadow=True,
                        max_signals=5,
                        max_configs=1,
                        dry_run=True,
                        refresh_cache=False,
                    )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "skipped_hold")

    async def test_max_configs_scanned_can_scan_holds_and_open_one_signal(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "registry.jsonl"
            journal = Path(tmp) / "shadow.jsonl"
            write_registry(registry, [
                registry_row(config_id=f"cfg{idx}", classification="unstable_watchlist")
                for idx in range(8)
            ])
            signals = [FakeStrategySignal(signal="HOLD") for _ in range(7)] + [FakeStrategySignal(signal="BUY")]
            with patch("scripts.generate_shadow_signals_once.load_experiment_candles", new=AsyncMock(return_value={
                "candles": sample_candles("2026-01-01", 300),
            })):
                with patch("scripts.generate_shadow_signals_once.generate_strategy_signal_from_df", side_effect=signals):
                    rows = await generate_shadow_signals_once(
                        registry=str(registry),
                        journal_path=journal,
                        allow_watchlist_shadow=True,
                        max_signals=1,
                        max_configs_scanned=8,
                        dry_run=True,
                        refresh_cache=False,
                    )

        summary = summarize_generation_rows(rows, journal_path=journal, max_signals=1, max_configs_scanned=8)
        self.assertEqual(len(rows), 8)
        self.assertEqual(summary["configs_scanned"], 8)
        self.assertEqual(summary["opened_signals"], 1)
        self.assertEqual(summary["skipped_hold"], 7)
        self.assertFalse(journal.exists())

    async def test_does_not_open_more_than_max_signals_when_scanning_more_configs(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "registry.jsonl"
            journal = Path(tmp) / "shadow.jsonl"
            write_registry(registry, [
                registry_row(config_id=f"cfg{idx}", classification="unstable_watchlist")
                for idx in range(3)
            ])
            with patch("scripts.generate_shadow_signals_once.load_experiment_candles", new=AsyncMock(return_value={
                "candles": sample_candles("2026-01-01", 300),
            })):
                with patch("scripts.generate_shadow_signals_once.generate_strategy_signal_from_df", return_value=FakeStrategySignal(signal="BUY")):
                    rows = await generate_shadow_signals_once(
                        registry=str(registry),
                        journal_path=journal,
                        allow_watchlist_shadow=True,
                        max_signals=1,
                        max_configs_scanned=3,
                        dry_run=True,
                        refresh_cache=False,
                    )

        summary = summarize_generation_rows(rows, journal_path=journal, max_signals=1, max_configs_scanned=3)
        self.assertEqual(summary["opened_signals"], 1)
        self.assertEqual(summary["configs_scanned"], 1)
        self.assertFalse(journal.exists())

    async def test_no_price_is_reported_separately(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "registry.jsonl"
            journal = Path(tmp) / "shadow.jsonl"
            write_registry(registry, [registry_row(classification="unstable_watchlist")])
            with patch("scripts.generate_shadow_signals_once.load_experiment_candles", new=AsyncMock(side_effect=RuntimeError("network down"))):
                with patch("scripts.generate_shadow_signals_once.asyncio.sleep", new=AsyncMock()):
                    rows = await generate_shadow_signals_once(
                        registry=str(registry),
                        journal_path=journal,
                        allow_watchlist_shadow=True,
                        max_signals=1,
                        dry_run=True,
                        refresh_cache=False,
                    )

        self.assertEqual(rows[0]["status"], "skipped_no_price")
        self.assertEqual(rows[0]["error_type"], "RuntimeError")
        self.assertEqual(rows[0]["error_category"], "network")
        self.assertEqual(rows[0]["fetch_attempts"], 3)
        self.assertIn("network down", rows[0]["error_message"])

    async def test_invalid_levels_are_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "registry.jsonl"
            journal = Path(tmp) / "shadow.jsonl"
            write_registry(registry, [registry_row(classification="unstable_watchlist")])
            invalid = FakeStrategySignal(signal="BUY").to_dict()
            invalid["stop_loss"] = 105.0
            with patch("scripts.generate_shadow_signals_once.load_experiment_candles", new=AsyncMock(return_value={
                "candles": sample_candles("2026-01-01", 300),
            })):
                with patch("scripts.generate_shadow_signals_once.generate_strategy_signal_from_df", return_value=FakeStrategySignal(signal="BUY")) as gen:
                    gen.return_value.to_dict = lambda: invalid
                    rows = await generate_shadow_signals_once(
                        registry=str(registry),
                        journal_path=journal,
                        allow_watchlist_shadow=True,
                        max_signals=1,
                        dry_run=True,
                        refresh_cache=False,
                    )

        self.assertEqual(rows[0]["status"], "skipped_invalid_levels")
        self.assertIn("entry/SL/TP", rows[0]["skip_reason"])


class ShadowEvaluationTests(unittest.TestCase):
    def make_signal(self, side="LONG", generated=None, expires=None):
        generated = generated or utc_now()
        expires = expires or generated + timedelta(hours=3)
        return {
            "shadow_signal_id": "sig1",
            "config_id": "cfg1",
            "symbol": "SOL",
            "timeframe": "1h",
            "strategy_mode": "xgboost",
            "strategy_name": "xgboost_temporal_split",
            "strategy_version": "v1",
            "side": side,
            "entry_price": 100.0,
            "stop_loss": 95.0 if side == "LONG" else 105.0,
            "take_profit": 110.0 if side == "LONG" else 90.0,
            "risk_reward": 2.0,
            "horizon_candles": 3,
            "horizon_minutes": 180,
            "confidence": 60,
            "generated_at": generated.isoformat(),
            "expires_at": expires.isoformat(),
            "status": OPEN,
            "commission_pct": 0.001,
            "slippage_pct": 0.0005,
            "spread_pct": 0.0003,
            "research_only": True,
        }

    def test_evaluation_long_tp_uses_high_low_and_costs(self):
        generated = utc_now() - timedelta(hours=4)
        signal = self.make_signal("LONG", generated=generated, expires=generated + timedelta(hours=3))
        candles = sample_candles(generated + timedelta(hours=1), 3)
        candles.loc[0, "high"] = 111.0

        updates = evaluate_shadow_signal_with_candles(signal, candles, now=generated + timedelta(hours=4))

        self.assertEqual(updates["outcome"], "WIN")
        self.assertEqual(updates["exit_reason"], "take_profit")
        self.assertGreater(updates["pnl_pct"], 0)
        self.assertLess(updates["pnl_pct"], 10.0)
        self.assertGreater(updates["fees"], 0)

    def test_evaluation_same_candle_ambiguous_is_loss(self):
        generated = utc_now() - timedelta(hours=4)
        signal = self.make_signal("LONG", generated=generated, expires=generated + timedelta(hours=3))
        candles = sample_candles(generated + timedelta(hours=1), 3)
        candles.loc[0, "high"] = 111.0
        candles.loc[0, "low"] = 94.0

        updates = evaluate_shadow_signal_with_candles(signal, candles, now=generated + timedelta(hours=4))

        self.assertEqual(updates["outcome"], "LOSS")
        self.assertEqual(updates["exit_reason"], "ambiguous_intrabar_conservative_loss")
        self.assertLess(updates["pnl_pct"], 0)

    def test_not_due_without_tp_or_sl_stays_open(self):
        generated = utc_now() - timedelta(hours=1)
        signal = self.make_signal("LONG", generated=generated, expires=generated + timedelta(hours=3))
        candles = sample_candles(generated + timedelta(minutes=30), 1)

        updates = evaluate_shadow_signal_with_candles(signal, candles, now=generated + timedelta(hours=1))

        self.assertIsNone(updates)


class ShadowSupportTests(unittest.TestCase):
    def test_horizon_candles_convert_to_minutes(self):
        self.assertEqual(horizon_minutes_from_candles(16, "1h"), 960)
        self.assertEqual(horizon_minutes_from_candles(4, "15m"), 60)

    def test_candidate_loader_does_not_use_test_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "registry.jsonl"
            row = registry_row(classification="unstable_watchlist")
            row["test_metrics"] = {"profit_factor": 999}
            write_registry(registry, [row])

            configs = load_candidate_configs(registry, allow_watchlist_shadow=False)

        self.assertEqual(configs, [])

    def test_candidate_loader_can_include_not_allowed_for_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "registry.jsonl"
            write_registry(registry, [registry_row(classification="unstable_watchlist")])

            configs = load_candidate_configs(registry, allow_watchlist_shadow=False, include_not_allowed=True)

        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0]["_source_classification"], "unstable_watchlist")

    def test_agent_review_cannot_modify_trade_levels(self):
        request = SignalReviewRequest(
            symbol="SOL",
            timeframe="1h",
            side="LONG",
            entry_price=100,
            stop_loss=95,
            take_profit=110,
            confidence=60,
        )
        with patch.dict(os.environ, {}, clear=True):
            response = review_shadow_signal(request)

        self.assertFalse(response.can_modify_trade_levels)
        self.assertFalse(hasattr(response, "entry_price"))
        self.assertEqual(response.review_status, "APPROVE")

    def test_agent_review_can_block_high_risk_news_without_modifying_trade(self):
        request = SignalReviewRequest(
            symbol="SOL",
            timeframe="1h",
            side="LONG",
            entry_price=100,
            stop_loss=95,
            take_profit=110,
            confidence=60,
            news_context={
                "risk_score": 95,
                "sentiment_score": -0.8,
                "item_count": 3,
                "risk_flags": ["negative_symbol_news", "news_hack"],
            },
        )

        response = review_shadow_signal(request)

        self.assertEqual(response.review_status, "BLOCK")
        self.assertEqual(response.confidence_adjustment, -10)
        self.assertFalse(response.can_modify_trade_levels)
        self.assertFalse(hasattr(response, "entry_price"))
        self.assertIn("negative_symbol_news", response.risk_flags)

    def test_crypto_universe_has_no_equities(self):
        universe = crypto_universe()

        self.assertIn("BTC", universe)
        self.assertIn("SOL", universe)
        self.assertNotIn("AAPL", universe)

    def test_telegram_plain_text_does_not_include_env_secrets(self):
        signal = {
            "watchlist_shadow": True,
            "symbol": "SOL",
            "timeframe": "1h",
            "side": "LONG",
            "entry_price": 100,
            "stop_loss": 95,
            "take_profit": 110,
            "risk_reward": 2,
            "horizon_candles": 12,
            "horizon_minutes": 720,
            "confidence": 60,
            "config_id": "cfg1",
            "classification": "unstable_watchlist",
            "agent_review": {
                "review_status": "CAUTION",
                "risk_flags": ["high_volatility", "news_event"],
            },
        }
        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "secret-token", "TELEGRAM_CHAT_ID": "secret-chat"}, clear=True):
            opened = format_shadow_signal_opened(signal)
            evaluated = format_shadow_signal_evaluated({**signal, "outcome": "WIN", "pnl_pct": 1.2})
            summary = format_shadow_daily_summary([{**signal, "outcome": "WIN", "pnl_pct": 1.2}])

        self.assertIn("Research only. No trading signal.", opened)
        self.assertIn("Agent review: CAUTION", opened)
        self.assertIn("Risk flags: high_volatility, news_event", opened)
        self.assertNotIn("secret-token", opened + evaluated + summary)
        self.assertNotIn("secret-chat", opened + evaluated + summary)
        self.assertNotIn("```", opened + evaluated + summary)

    def test_build_shadow_signal_stores_costs_and_horizons(self):
        config = registry_row()["config"]
        costs = cost_profile_for_config(config)
        shadow = build_shadow_signal_from_strategy(
            config=config,
            source_registry="refined_registry",
            classification="unstable_watchlist",
            signal=FakeStrategySignal().to_dict(),
            costs=costs,
            watchlist_shadow=True,
        )

        self.assertEqual(shadow["horizon_minutes"], 720)
        self.assertEqual(shadow["commission_pct"], costs["commission_pct"])
        self.assertTrue(shadow["research_only"])

    def test_cli_parsers_accept_expected_flags(self):
        args = build_generate_parser().parse_args([
            "--allow-watchlist-shadow",
            "--symbols",
            "SOL,ETH",
            "--max-signals",
            "2",
            "--max-configs-scanned",
            "8",
            "--use-news-context",
            "--use-market-context",
        ])
        self.assertTrue(args.allow_watchlist_shadow)
        self.assertEqual(args.max_signals, 2)
        self.assertEqual(args.max_configs_scanned, 8)
        self.assertTrue(args.use_news_context)
        self.assertTrue(args.use_market_context)

        eval_args = build_evaluate_parser().parse_args(["--notify-telegram"])
        self.assertTrue(eval_args.notify_telegram)

    def test_registry_choice_accepts_crypto_multi(self):
        path = registry_path_from_choice("crypto_multi")

        self.assertEqual(path.name, "crypto_multi_registry.jsonl")

    def test_classify_signal_skip_hold_and_invalid_levels(self):
        hold = FakeStrategySignal(signal="HOLD").to_dict()
        hold["input_features"] = {"hold_reason": "probabilities_below_threshold"}
        self.assertEqual(classify_signal_skip(hold)["status"], "skipped_hold")

        invalid = FakeStrategySignal(signal="BUY").to_dict()
        invalid["take_profit"] = 90.0
        self.assertEqual(classify_signal_skip(invalid)["status"], "skipped_invalid_levels")

    def test_generation_summary_counts_statuses(self):
        rows = [
            {"status": "OPEN", "config_id": "a"},
            {"status": "skipped_hold", "config_id": "b"},
            {"status": "skipped_duplicate_open", "config_id": "c"},
            {"status": "skipped_error", "config_id": "d"},
            {"status": "skipped_no_price", "config_id": "e"},
        ]

        summary = summarize_generation_rows(rows, journal_path="data/shadow.jsonl")

        self.assertEqual(summary["selected_configs"], 5)
        self.assertEqual(summary["opened_signals"], 1)
        self.assertEqual(summary["skipped_hold"], 1)
        self.assertEqual(summary["skipped_duplicate_open"], 1)
        self.assertEqual(summary["skipped_errors"], 2)
        self.assertEqual(summary["journal_path"], "data/shadow.jsonl")

    def test_print_rows_omits_empty_levels_for_skipped_hold_and_prints_summary(self):
        rows = [{
            "status": "skipped_hold",
            "symbol": "SOL",
            "timeframe": "1h",
            "config_id": "cfg1",
            "classification": "unstable_watchlist",
            "skip_reason": "la señal actual fue HOLD",
            "hold_reason": "probabilities_below_threshold",
            "probability_buy_win": 0.4,
            "probability_sell_win": 0.3,
            "confidence": 40,
        }]
        stream = StringIO()
        with patch("sys.stdout", stream):
            print_rows(rows, journal_path="data/shadow.jsonl")

        output = stream.getvalue()
        self.assertIn("status=skipped_hold", output)
        self.assertIn("hold_reason=probabilities_below_threshold", output)
        self.assertNotIn("side=", output)
        self.assertNotIn("entry=", output)
        self.assertIn("Summary", output)
        self.assertIn("journal_path: data/shadow.jsonl", output)


if __name__ == "__main__":
    unittest.main()
