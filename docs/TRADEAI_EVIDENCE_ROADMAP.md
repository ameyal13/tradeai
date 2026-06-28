# TRADEAI Evidence Roadmap

## Current Position

TRADEAI is a research and shadow-trading system. It is not ready for real-money
execution. Railway currently evaluates and generates shadow signals, Supabase
stores outcomes, and Vercel exposes read-only monitoring.

The current live shadow metrics are negative. The correct response is to audit
the evidence and narrow hypotheses, not increase trade frequency or relax
thresholds.

## What Railway Learns Today

Each shadow cycle fetches recent market candles and trains temporary XGBoost
models for the scanned configurations. The trained objects are not persisted.
Shadow outcomes are stored, but they do not automatically alter features,
thresholds, hyperparameters, or source code.

This separation is deliberate. Automatic tuning against the same live sample
would create selection bias and uncontrolled overfitting.

## Phase Gates

### Gate 1: Operational Reliability

- Hourly cycles complete with `HEALTH_OK`.
- No persistent market-data or evaluation errors.
- Locks prevent concurrent cycles.
- Every OPEN signal eventually closes or expires.
- Supabase is the authoritative Railway store.

### Gate 2: Live Evidence Audit

- Audit performance by symbol, side, config, confidence, review status, and exit reason.
- Compare research validation PF with later shadow PF.
- Do not use test PF for selection.
- Do not disable a config automatically from a tiny sample.

### Gate 3: Manual Config Quarantine

- A config can be reviewed for pause only after at least 10 closed live signals.
- Pause candidates require negative average return and materially sub-1.0 PF.
- Every pause decision must be recorded with sample size and reason.
- Test-only positives cannot rescue failed validation or live evidence.

### Gate 4: Offline Model Improvement

- Create one hypothesis at a time: direction, horizon, labels, feature family, or calibration.
- Train and validate with purged temporal splits.
- Compare against random, deterministic, and the frozen incumbent.
- Store model version, feature policy, data interval, costs, and validation metrics.
- Never tune on the final test window.

### Gate 5: Shadow Challenger Evaluation

- Run incumbent and challenger in parallel without placing orders.
- Require positive net expectancy, PF above 1.1, controlled drawdown, and calibration improvement.
- Require evidence across multiple windows and more than one market regime.
- Do not promote from a single symbol, hour, or small sample.

### Gate 6: Paper Candidate

- At least 100 closed shadow outcomes overall.
- At least 30 closed outcomes for each promoted config/direction.
- Positive after-cost return and PF above 1.1 in both validation and later shadow evidence.
- No unresolved evaluator/data-integrity warnings.
- Real-money execution remains a separate explicit project and approval.

## Repository Hygiene Policy

Files belong to one of four groups:

1. Runtime: API, Railway shadow ops, Supabase repositories, and frontend monitoring.
2. Reproducible research: grids, validators, summaries, and tests that explain prior decisions.
3. Operational fallback: local JSONL, Windows wrappers, and sync scripts.
4. Deletion candidates: temporary files, generated logs, caches, exact duplicates, or unreachable code proven by references and tests.

Research scripts are not dead code merely because Railway does not import them.
They preserve methodology and must not be deleted without a replacement and a
documented migration. Generated `data/`, `reports/`, caches, and logs remain
outside Git.

## Immediate Next Decision

Run the Supabase-first shadow live performance audit. Use it to form one
controlled improvement hypothesis. Do not add more models, LLM trade decisions,
or broader grids until that audit identifies where expectancy is being lost.

Research only. No trading signal.
