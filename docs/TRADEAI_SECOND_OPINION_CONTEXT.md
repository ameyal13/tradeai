# TRADEAI Project Context For Second Opinion

Last updated: 2026-06-20

This document summarizes the current TRADEAI system, methodology, results, safety constraints, and open questions. It is intended to give another AI or technical reviewer enough context to propose a rigorous project plan without assuming that the system is profitable.

## 1. Project Vision

TRADEAI is a research-first crypto trading assistant. The goal is not to create a magical AI trader. The goal is to build a measurable system that can:

- Download and cache historical crypto OHLCV data.
- Generate deterministic and ML-based candidate signals.
- Build labels aligned with trade outcomes, TP/SL, costs, and horizons.
- Train local models such as XGBoost.
- Run historical replay and multi-window validation without lookahead.
- Compare strategies against baselines.
- Record shadow/paper signals.
- Evaluate outcomes against real future candles.
- Sync results to Supabase.
- Display metrics in a Vercel dashboard.
- Send Telegram notifications.
- Eventually support safer paper trading workflows and only much later consider real-money execution.

Current operating principle:

> Research only. No trading signal. No exchange orders. No withdrawals.

## 2. Safety Boundaries

The project intentionally does not do real trading.

Hard constraints:

- No real exchange orders.
- No withdrawal functionality.
- No private exchange keys required.
- No LLM-generated autonomous trades.
- No thresholds lowered just to force trades.
- No test metrics used to select candidates.
- No accuracy as the central metric.
- No secrets printed or committed.
- Supabase service role key must stay backend-only.
- Frontend uses only Supabase anon key.

The LLM/agent layer may review or explain a signal, but cannot:

- Invent a trade.
- Change side.
- Change entry.
- Change stop loss.
- Change take profit.
- Change risk/reward.
- Place orders.

## 3. Current Stack

Backend:

- Python
- FastAPI
- Local scripts for research, shadow ops, sync, summaries
- XGBoost model support
- Supabase integration
- Binance public market data

Frontend:

- React + Vite
- Vercel deployment
- Supabase client
- Dashboard pages for shadow signals and research telemetry

Database:

- Supabase PostgreSQL
- Tables include shadow/research sync structures added via `schema.sql`

Automation:

- Windows Task Scheduler local shadow ops
- PowerShell wrappers
- Telegram notifications

## 4. Major System Layers

### 4.1 Market Data Layer

Relevant files:

- `tools/historical_data.py`
- `tools/market_tool.py`

Responsibilities:

- Fetch public candles.
- Cache OHLCV locally.
- Retry/backoff for network issues.
- Support more than 1000 Binance candles by backward pagination.
- Avoid making trading decisions.

Important prior issue:

- Binance latest mode originally returned only 1000 candles when more were requested.
- Pagination was corrected so `max_candles` can load 1500, 3000, 5000, etc.

### 4.2 Feature Layer

Relevant files:

- `tools/strategy_signals.py`
- `tools/ml_engine.py`
- `tools/feature_research.py`
- `research/market_context_engine.py`

Original XGBoost feature columns:

- `rsi`
- `macd_hist`
- `ema_fast`
- `ema_slow`
- `relative_volume`
- `return_1`
- `return_3`
- `volatility_10`
- `atr`

Research-only time/regime features existed:

- hour sine/cosine
- day-of-week sine/cosine
- ATR percent
- EMA distance

Market Context Features v1 added opt-in research features:

- EMA trend strength
- EMA fast above slow
- distance to EMA slow
- distance to recent 20-candle high/low
- recent 20-candle range
- ATR regime over 50 candles
- volume regime over 50 candles
- 6/12 candle returns

Important conclusion:

- Market Context Features v1 directly inside XGBoost worsened validation evidence on most matched configs.
- Context may still be useful as a review/filter layer, but not proven as direct model input.

### 4.3 Label / Outcome Layer

Relevant files:

- `tools/trade_labels.py`
- `tools/ml_engine.py`
- `tools/prediction_journal.py`

The project moved away from simple "price goes up" labels because the evaluator measures TP/SL/cost outcomes. XGBoost trade labels now try to align with directional trade outcomes:

- BUY win probability
- SELL win probability
- costs
- ATR-based SL/TP
- horizon candles

An important evaluator correction was made:

- If TP or SL is hit, `return_pct` uses the actual TP/SL exit price, not final close.
- Conservative same-candle TP+SL ambiguity counts as loss.

This removed invalid cases like:

- WIN with negative return
- LOSS with positive return

### 4.4 Research Runner / Multi-Window Validation

Relevant files:

- `research/experiment_runner.py`
- `research/multi_window_validator.py`
- `research/research_daemon.py`
- `research/research_registry.py`

Methodology:

- Every config is evaluated over rolling windows.
- Each window has train/validation/test split.
- Purge/embargo equals horizon candles.
- Train only uses past data.
- Validation selects.
- Test only confirms or contradicts.
- Test must not promote a config.

Metrics:

- validation avg return
- validation profit factor
- validation drawdown
- test avg return
- test profit factor
- test confirm rate
- beats random same-count
- beats deterministic
- directional bias
- valid windows

Classifications:

- `stable_research_candidate`
- `unstable_watchlist`
- `multi_window_reject`
- `needs_more_data`

Current rule of thumb:

- Stable requires positive validation in at least 60% of windows, beats random in at least 60%, beats deterministic in at least 50%, median validation PF > 1.05, median validation average return > 0, and test not strongly contradicting.

### 4.5 Research Grids Already Run

#### Crypto Multi-Asset Grid v1

Relevant files:

- `research/crypto_multi_asset_grid.py`
- `scripts/run_crypto_multi_asset_research.py`
- `scripts/summarize_crypto_multi_research.py`

Grid:

- Symbols: BTC, ETH, SOL, BNB, XRP, ADA, AVAX, LINK
- Timeframe: 1h
- Strategy: XGBoost
- Horizons: 12, 14
- RR: 2.0, 2.5
- ATR stop: 1.0, 1.25
- Cost mode: low_costs

Result:

- 64 completed
- 21 unstable_watchlist
- 43 multi_window_reject
- 0 stable_research_candidate

Promising assets:

- ADA
- ETH
- SOL

Discarded temporarily:

- AVAX
- BNB
- BTC
- LINK
- XRP

#### Focused Research v2A

Relevant files:

- `research/focused_research_grid.py`
- `scripts/run_focused_research.py`
- `scripts/run_focused_research_loop.py`
- `scripts/summarize_focused_research.py`
- `scripts/audit_focused_stability.py`

Grid:

- Symbols: ADA, ETH, SOL
- Timeframe: 1h
- Horizons: 10, 12, 14
- RR: 2.0, 2.5, 2.8
- ATR stop: 1.0, 1.25, 1.5
- Cost mode: low_costs

Result:

- 81 completed
- 44 unstable_watchlist
- 37 multi_window_reject
- 0 stable_research_candidate

Dominant groups from summary:

- horizon 10
- RR 2.5
- ATR 1.5

Focused stability audit:

- Top watchlist configs were not stable mainly because of:
  - low test confirm
  - very high drawdown

Example top result:

- ADA 1h h12 RR2.0 ATR1.25 low_costs
- validation PF about 1.41
- validation avg about 0.18
- but test confirm weak and drawdown very high

#### Market Context Features v1

Relevant files:

- `research/market_context_research_grid.py`
- `scripts/run_market_context_research.py`
- `scripts/summarize_market_context_research.py`
- `scripts/compare_research_phases.py`

Grid:

- Symbols: ADA, ETH
- Timeframe: 1h
- Horizons: 10, 12
- RR: 2.0, 2.5
- ATR stop: 1.25, 1.5
- Cost mode: low_costs
- Feature family: `current_plus_market_context_v1`

Result:

- 16 completed
- 13 multi_window_reject
- 3 unstable_watchlist
- 0 stable_research_candidate

Top market_context_v1 watchlist:

- ETH 1h h12 RR2.0 ATR1.5 low_costs
- validation PF 1.127791
- validation avg 0.07519
- test PF 1.177249
- test confirm 0.368421
- worst drawdown about 63.46%

Comparison against focused_v2A on matched configs:

- 16 matched configs
- 15 worse_validation
- 1 improved_validation
- mean delta validation PF: -0.246003
- mean delta validation avg return: -0.153442
- mean delta worst drawdown: -14.78042

Interpretation:

- Market context features reduced drawdown on average but hurt validation returns and PF.
- Directly adding these context features to XGBoost is not currently justified.
- Context may be more useful as a filter/reviewer than as raw model input.

## 5. Shadow / Paper Signal System

Relevant files:

- `tools/shadow_signal_journal.py`
- `tools/shadow_signal_repository.py`
- `scripts/generate_shadow_signals_once.py`
- `scripts/evaluate_shadow_signals_once.py`
- `scripts/run_shadow_cycle_once.py`
- `scripts/run_shadow_ops_once.py`
- `scripts/summarize_shadow_signals.py`
- `scripts/sync_shadow_journal_to_supabase.py`
- `scripts/shadow_ops_healthcheck.py`

What it does:

- Reads research configs from registries.
- Generates live shadow signals only from allowed classifications.
- Does not place orders.
- Saves signals locally.
- Evaluates open signals when TP/SL/horizon resolves.
- Syncs to Supabase.
- Sends Telegram messages.
- Avoids duplicate open signals.
- Uses lock files to avoid concurrent runs.

Operational rule:

- It opens at most one shadow signal per cycle.
- If there is already an OPEN signal, it does not open another.

Known results at time of this document:

- A small number of shadow signals have been opened/evaluated.
- Early outcomes included SOL losses and an ADA win.
- Sample size is far too small for trust.

## 6. News / Agent / Market Context Layer

Relevant files:

- `research/news_context_engine.py`
- `research/market_context_engine.py`
- `research/signal_review_agent.py`

Current role:

- The agent/review layer can inspect a signal.
- It can return APPROVE / CAUTION / BLOCK.
- It can add risk flags.
- It can adjust confidence within bounded limits.
- It cannot modify trade levels or create trades.

News context:

- Optional.
- Can be used during shadow ops.
- Should be measured to see whether it improves outcomes.

Important design recommendation:

- Do not let DeepSeek, GPT, or any LLM directly decide BUY/SELL.
- Use LLMs for:
  - news summarization
  - market context explanation
  - risk review
  - post-trade reflection
  - hypothesis generation
  - report interpretation

## 7. Frontend / Dashboard

Relevant files:

- `frontend/src/...`

Current pages:

- Shadow signals dashboard
- Research dashboard
- News page
- Backtest/navigation pages depending on frontend state

Deployment:

- Frontend on Vercel.
- Backend on Railway.
- Supabase as DB.

Current dashboard shows:

- Open/closed shadow signals
- Win rate
- Profit factor
- Average return
- Max drawdown
- Signals by symbol
- Recent shadow signals
- Latest signal context
- Research summaries

Important limitation:

- The dashboard is mostly read-only.
- Buttons for launching research or shadow cycles from the web should be added carefully because they can create long-running jobs, duplicates, and operational risk.

## 8. Automation

Windows Task Scheduler is configured for local shadow ops:

- Hourly shadow ops wrapper
- Daily shadow summary wrapper

Railway is used for backend serving, but long-running research loops should be handled carefully. Risks of cloud automation:

- timeout limits
- concurrent jobs
- cost creep
- duplicate execution
- secrets management
- accidental operational endpoints
- long-running ML jobs on web dynos

Recommended approach:

- Keep research loops local or behind explicit controlled scripts until a durable job runner is designed.
- If moving research automation to Railway, use strict locks, job tables, and admin-only endpoints.

## 9. Current Evidence And Honest Status

What works:

- Historical data loading/cache.
- Multi-window validation.
- Research daemon.
- Research summaries.
- Shadow signal generation/evaluation.
- Supabase sync.
- Vercel dashboard.
- Telegram messages.
- Safety guardrails.

What does not work yet:

- No stable research candidates.
- XGBoost has not shown robust profitability.
- Market Context Features v1 worsened validation on most matched configs.
- Drawdowns remain too high.
- Test confirmation is weak.
- Shadow signal sample size is too small.

Current honest conclusion:

> TRADEAI is functioning as a research and shadow-monitoring platform, but it has not found a robustly tradeable strategy yet.

## 10. Key Open Technical Questions

1. Should market context be used as a filter/reviewer instead of as model input?
2. Would true multi-timeframe context improve signal quality?
3. Are current ATR/TP/SL rules structurally causing high drawdown?
4. Are features too weak, too noisy, or too correlated with unstable regimes?
5. Should the model predict expected value rather than win/loss?
6. Should the strategy separate long-only and short-only models by regime?
7. How much improvement comes from avoiding bad trades instead of selecting good trades?
8. How should news/sentiment be measured against outcomes?
9. What is the minimum shadow sample size before promoting a config?
10. Should the frontend get admin controls, or remain read-only until job safety is stronger?

## 11. Recommended Next Phase

Recommended next technical phase:

### Multi-Timeframe Context Review v1

Do not immediately add more raw features to XGBoost. Instead:

1. Build a read-only multi-timeframe context engine.
2. For each candidate/shadow signal, compute:
   - 15m momentum
   - 1h setup context
   - 4h trend/regime
   - 1d macro direction
   - BTC 1h/4h benchmark regime
3. Produce structured output:
   - local trend
   - higher-timeframe trend
   - benchmark alignment
   - volatility regime
   - support/resistance proximity
   - liquidity/volume confirmation
   - risk flags
4. First use it only as a diagnostic/reviewer.
5. Then run shadow/replay analysis:
   - Did signals with aligned 4h/BTC context perform better?
   - Did CAUTION/BLOCK flags reduce losses?
   - Did it reduce drawdown?

Only after that should it be considered as model features.

## 12. Questions For The Second Opinion Reviewer

Please evaluate:

1. Is the current validation methodology sufficient to avoid lookahead and overfitting?
2. Should TRADEAI keep using XGBoost, or shift to simpler models / expected value models?
3. Should market context be a model input, a filter, or a separate risk manager?
4. What minimum sample size and metrics should be required before paper trading?
5. How should news and sentiment be integrated without creating narrative bias?
6. What should be the next research grid, if any?
7. Should the project prioritize:
   - multi-timeframe context,
   - expected value regression,
   - better deterministic baselines,
   - regime classification,
   - or richer shadow outcome analysis?
8. What could be causing high drawdown despite some positive validation PF?
9. Are costs and ATR levels realistic?
10. What should be rejected permanently versus kept as a watchlist hypothesis?

## 13. Most Recent Quantitative Takeaway

Market Context Features v1 compared to Focused v2A:

- Matched configs: 16
- Worse validation: 15
- Improved validation: 1
- Mean delta validation PF: -0.246003
- Mean delta validation avg return: -0.153442
- Mean delta worst drawdown: -14.78042

This means:

- Drawdown improved, but performance deteriorated.
- The system may be over-filtering or adding noisy features.
- It is not enough to add "trader-like context" directly into the model.
- A separate context/risk review layer is more promising than raw feature injection.

## 14. Current Recommendation

Do not go to real trading.

Do not declare profitability.

Do not lower thresholds to force signals.

Do not let an LLM trade autonomously.

Do:

- Continue shadow monitoring.
- Continue syncing to Supabase and Vercel.
- Build stronger context diagnostics.
- Compare every new phase against prior baselines.
- Use validation as the selector and test as diagnostic only.
- Keep all bad results visible.

