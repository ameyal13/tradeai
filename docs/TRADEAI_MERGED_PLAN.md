# TRADEAI Merged Plan

Last updated: 2026-06-21

This document reconciles the internal TRADEAI context in `docs/TRADEAI_SECOND_OPINION_CONTEXT.md` with the external second-opinion plan from Claude/Anthropic. It is intended to become the near-term execution plan for TRADEAI.

## Executive Summary

Both plans agree on the most important point:

> TRADEAI is now a functioning research and shadow-monitoring platform, but it has not found a robustly tradeable strategy yet.

The next phase should not be more blind grid search. It should isolate why the promising watchlist configs fail to become stable:

1. Is the problem direction prediction?
2. Is the problem ATR/TP/SL sizing?
3. Is the problem high drawdown despite positive validation?
4. Is context useful as a filter/reviewer rather than as raw model features?
5. Is shadow/live behavior matching historical validation?

The owner's priority is:

1. Validate whether the system can be profitable with evidence.
2. Only after that, consider productization or selling access.

Therefore, research rigor and operational reliability come before monetization.

## Agreements Between Plans

### 1. Current Negative Result Is Valid Evidence

Across the major research phases, no stable strategy has been found:

- `crypto_multi`: 64 completed, 0 stable.
- `focused_v2a`: 81 completed, 0 stable.
- `market_context_v1`: 16 completed, 0 stable.

This is not a project failure. It is a useful negative result from a disciplined process.

### 2. Market Context Should Not Be Fed Blindly Into XGBoost

`market_context_v1` added trader-like context features directly to the model. Against matched `focused_v2a` configs:

- 16 matched configs.
- 15 worse validation.
- 1 improved validation.
- mean delta validation PF: -0.246003.
- mean delta validation avg return: -0.153442.

Conclusion:

- Do not keep using `market_context_v1` as direct XGBoost input.
- Context may still be useful as a diagnostic, risk filter, or signal review layer.

### 3. Minimum Shadow Sample Size Needs A Hard Gate

The project currently has shadow monitoring, but the threshold for promoting a config from research/watchlist toward paper trading is not yet hardcoded as a policy gate.

This should be explicit before any serious paper-trading or real-money discussion.

### 4. LLMs Should Review, Not Trade

Both plans agree:

- LLM/news/sentiment agents can summarize, critique, caution, block, and explain.
- They must not generate autonomous trades or alter side/entry/SL/TP/RR.

## Disagreements Or Nuance

### 1. Railway Cron Is Useful, But Not "Just Turn It On"

The second opinion recommends moving hourly shadow ops and daily summaries from Windows Task Scheduler to Railway cron/worker.

I agree directionally, but with a caution:

- Local Task Scheduler uses a local lock file.
- Railway cron/worker needs a durable cross-process lock, preferably in Supabase.
- Without that, duplicate cloud runs could open/evaluate/sync at the same time.

So the correct sequence is:

1. Design Supabase-backed job/cycle lock.
2. Make shadow ops idempotent under cloud concurrency.
3. Add missed-cycle alerting.
4. Only then move hourly jobs from local Windows scheduler to Railway.

### 2. Dependency Cleanup Is Real But Not Strategy-Critical

Verified locally:

- `celery`, `redis`, `python-jose`, `beautifulsoup4`, and `langchain-community` are listed in `requirements.txt`.
- Grep found no active imports for those packages in current code paths.
- `feedparser` is used in `tools/news_tool.py`; keep it.
- README already documents some packages as future-facing / not actively used.

Recommendation:

- Do a later dependency hygiene pass.
- Do not mix dependency cleanup with strategy research unless deployment size/cost becomes a real problem.

### 3. Expected-Value Regression Is Promising, But Not First

The second opinion suggests expected-value regression as a lower-priority modeling prototype.

I agree, but it should come after:

1. ATR/RR sizing isolation.
2. Multi-timeframe diagnostic review.

Reason:

- If the main issue is sizing/risk design, changing the model target will not fix it cleanly.
- If context flags can identify bad trades, we may improve expectancy by filtering before changing model architecture.

## Merged Execution Plan

## Phase 0 - Freeze Current Evidence

Status: mostly done.

Actions:

- Keep `docs/TRADEAI_SECOND_OPINION_CONTEXT.md`.
- Keep `docs/TRADEAI_MERGED_PLAN.md`.
- Treat `focused_v2a` as the current baseline research phase.
- Treat `market_context_v1` as a rejected direct-feature experiment.

Do not:

- Delete bad reports.
- Promote watchlist configs because test looked good.
- Re-run grids without a written hypothesis.

## Phase 1 - ATR/RR Sizing Isolation

Goal:

Determine whether drawdown and instability come mainly from trade sizing/exit design rather than direction prediction.

Hypothesis:

> The model may have partial direction signal, but ATR stop / RR / horizon design causes excessive drawdown or poor expectancy.

Implementation idea:

- Create a research phase that freezes a small set of best-known feature/model configs.
- Vary only:
  - horizon candles
  - risk reward
  - ATR stop multiplier
- Keep:
  - symbols fixed
  - timeframe fixed
  - model/features fixed
  - cost mode fixed
  - thresholds fixed
  - validation process fixed

Candidate starting subset:

- ADA 1h and ETH 1h.
- Start from focused_v2A watchlist zones.
- Use low_costs first, then medium_costs_current only on survivors.

Success signal:

- Validation PF improves.
- Validation avg return improves.
- Worst drawdown decreases materially.
- Win rate may stay similar.

If this happens:

- The main problem is likely exit/risk design.

If not:

- The problem is more likely direction/feature/model signal.

## Phase 2 - Multi-Timeframe Context Diagnostic

Goal:

Test trader-like context without feeding it directly into XGBoost.

Do not train with it initially.

Compute context flags for historical windows and/or existing shadow signals:

- 15m momentum.
- 1h setup context.
- 4h trend/regime.
- 1d macro direction.
- BTC 1h/4h benchmark alignment.
- volatility regime.
- support/resistance proximity.
- volume/liquidity condition.

Measure:

- Do aligned signals have higher validation return?
- Do blocked/caution signals reduce losses?
- Does context reduce drawdown?
- Does it work better for ETH than ADA?

Only after this diagnostic should any context signal be considered for model input.

## Phase 3 - Shadow Promotion Gate

Goal:

Define hard numeric rules before any config can move beyond research/watchlist.

Suggested gate fields:

- minimum evaluated shadow signals per config/symbol/timeframe.
- rolling shadow profit factor threshold.
- rolling average return threshold.
- maximum drawdown threshold.
- live shadow vs historical expectation drift check.
- no open unresolved signal backlog.
- no missed-cycle gaps above tolerance.

Initial conservative policy:

- Less than 30 evaluated shadow signals: monitoring only.
- 30-99 evaluated signals: research_watchlist only.
- 100+ evaluated signals with positive PF/avg/drawdown control: eligible for paper-trading review.

These exact numbers can be debated, but the gate must exist before promotion.

## Phase 4 - Operational Resilience

Goal:

Make shadow ops reliable enough that shadow statistics are trustworthy.

Current status:

- Windows Task Scheduler runs local shadow ops.
- Railway currently serves FastAPI backend.
- Railway does not yet run cron/worker shadow cycles.

Next steps:

1. Add Supabase-backed job lock / cycle idempotency.
2. Add missed-cycle detection.
3. Alert by Telegram when cycles are missed or stale.
4. Then consider Railway cron/worker for hourly shadow ops.

Do not move scheduler to Railway until duplicate prevention is durable.

## Phase 5 - Expected-Value Modeling

Goal:

If sizing/context diagnostics do not solve the issue, test whether the model should predict expected value instead of WIN/LOSS.

Possible targets:

- expected net return after costs.
- probability-weighted EV classification.
- return bucket classification.
- meta-labeling on deterministic candidate signals.

Guardrails:

- Use walk-forward validation.
- Pre-register grid and success criteria.
- Do not optimize on test.

## Phase 6 - Productization / Monetization

Not now.

Only consider after:

- stable research candidates exist.
- shadow sample is large enough.
- operational reliability is strong.
- legal/regulatory framing is reviewed.

Safer eventual product path:

- Research dashboard / analytics / educational tooling.

Riskier path:

- Selling explicit buy/sell signals.

Any monetization around trading signals may require legal review.

## Near-Term Priority Order

1. Build ATR/RR sizing isolation research.
2. Build multi-timeframe diagnostic review.
3. Add shadow promotion gate.
4. Add missed-cycle alerting.
5. Later: Supabase/Railway job lock and cloud cron.
6. Later: expected-value regression.
7. Much later: monetization.

## What Not To Do Next

- Do not run another broad grid without a hypothesis.
- Do not add more raw features to XGBoost blindly.
- Do not let an LLM choose trades.
- Do not move to real trading.
- Do not optimize on test.
- Do not treat watchlist as tradable.
- Do not build billing/auth before strategy evidence.

## Proposed Next Concrete Task

Build:

```text
research/sizing_isolation_grid.py
scripts/run_sizing_isolation_research.py
scripts/summarize_sizing_isolation_research.py
tests/test_sizing_isolation_grid.py
```

Scope:

- ADA and ETH.
- 1h.
- fixed feature family: current baseline XGBoost features, not market_context_v1.
- low_costs.
- horizons around 10/12/14.
- RR and ATR variants.
- multi-window mandatory.
- validation selects.
- test diagnostic only.

Primary question:

> Does changing only ATR/RR/horizon improve validation PF and reduce drawdown enough to explain the current failure?
