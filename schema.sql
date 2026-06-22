-- Prediction Journal and Outcome Evaluator schema.
-- Apply in Supabase SQL editor or through your migration workflow.

create extension if not exists "pgcrypto";

create table if not exists prediction_journal (
  id uuid primary key default gen_random_uuid(),
  user_id uuid null,
  symbol text not null,
  timeframe text not null,
  strategy_mode text not null check (strategy_mode in ('deterministic', 'model_based', 'hybrid', 'xgboost')),
  strategy_name text not null,
  strategy_version text not null,
  signal text not null check (signal in ('BUY', 'SELL', 'HOLD')),
  confidence numeric not null,
  entry_price numeric not null,
  stop_loss numeric null,
  take_profit numeric null,
  risk_reward_ratio numeric null,
  horizon_minutes integer not null,
  input_features jsonb not null default '{}'::jsonb,
  reasoning text not null default '',
  model_provider text null,
  model_name text null,
  status text not null default 'pending' check (status in ('pending', 'evaluated', 'invalid')),
  created_at timestamptz not null default now()
);

create table if not exists prediction_outcomes (
  id uuid primary key default gen_random_uuid(),
  prediction_id uuid not null references prediction_journal(id) on delete cascade,
  evaluated_at timestamptz not null default now(),
  exit_price numeric null,
  return_pct numeric not null,
  max_favorable_excursion_pct numeric not null,
  max_adverse_excursion_pct numeric not null,
  hit_stop_loss boolean not null default false,
  hit_take_profit boolean not null default false,
  outcome text not null check (outcome in ('WIN', 'LOSS', 'BREAKEVEN', 'EXPIRED', 'INVALID_DATA')),
  fees_paid numeric not null default 0,
  slippage_cost numeric not null default 0,
  raw_path jsonb not null default '[]'::jsonb
);

alter table prediction_journal enable row level security;
alter table prediction_outcomes enable row level security;

drop policy if exists "prediction_journal_select_own_or_global" on prediction_journal;
create policy "prediction_journal_select_own_or_global"
on prediction_journal for select
using (user_id is null or auth.uid() = user_id);

drop policy if exists "prediction_journal_insert_own" on prediction_journal;
create policy "prediction_journal_insert_own"
on prediction_journal for insert
with check (user_id is null or auth.uid() = user_id);

drop policy if exists "prediction_journal_update_own" on prediction_journal;
create policy "prediction_journal_update_own"
on prediction_journal for update
using (user_id is null or auth.uid() = user_id)
with check (user_id is null or auth.uid() = user_id);

drop policy if exists "prediction_outcomes_select_visible" on prediction_outcomes;
create policy "prediction_outcomes_select_visible"
on prediction_outcomes for select
using (
  exists (
    select 1 from prediction_journal
    where prediction_journal.id = prediction_outcomes.prediction_id
    and (prediction_journal.user_id is null or auth.uid() = prediction_journal.user_id)
  )
);

drop policy if exists "prediction_outcomes_insert_visible" on prediction_outcomes;
create policy "prediction_outcomes_insert_visible"
on prediction_outcomes for insert
with check (
  exists (
    select 1 from prediction_journal
    where prediction_journal.id = prediction_outcomes.prediction_id
    and (prediction_journal.user_id is null or auth.uid() = prediction_journal.user_id)
  )
);

-- Shadow/Paper signal storage for research dashboards.
-- Backend service-role API writes/reads these tables; do not expose service role
-- keys to the frontend. RLS is enabled and no public anon policy is created here.

create table if not exists shadow_signals (
  shadow_signal_id text primary key,
  config_id text null,
  source_registry text null,
  classification text null check (classification is null or classification in ('stable_research_candidate', 'unstable_watchlist', 'manual')),
  symbol text not null,
  timeframe text not null,
  strategy_mode text null,
  side text null check (side is null or side in ('LONG', 'SHORT')),
  entry_price numeric null,
  stop_loss numeric null,
  take_profit numeric null,
  risk_reward numeric null,
  horizon_candles integer null,
  horizon_minutes integer null,
  confidence numeric null,
  generated_at timestamptz null,
  expires_at timestamptz null,
  status text not null check (status in ('OPEN', 'CLOSED', 'EXPIRED', 'BLOCKED')),
  outcome text null check (outcome is null or outcome in ('WIN', 'LOSS', 'BREAKEVEN', 'EXPIRED', 'INVALID')),
  exit_price numeric null,
  exit_reason text null,
  pnl_pct numeric null,
  pnl_amount numeric null,
  commission_pct numeric null,
  slippage_pct numeric null,
  spread_pct numeric null,
  mfe_pct numeric null,
  mae_pct numeric null,
  notes text null,
  input_features jsonb not null default '{}'::jsonb,
  agent_review jsonb not null default '{}'::jsonb,
  news_context jsonb not null default '{}'::jsonb,
  market_context jsonb not null default '{}'::jsonb,
  model_provider text null,
  model_name text null,
  research_only boolean not null default true,
  watchlist_shadow boolean not null default false,
  updated_at timestamptz not null default now(),
  raw jsonb not null default '{}'::jsonb
);

create index if not exists shadow_signals_symbol_timeframe_idx
on shadow_signals(symbol, timeframe, generated_at desc);

create index if not exists shadow_signals_status_idx
on shadow_signals(status, generated_at desc);

create table if not exists shadow_signal_events (
  id uuid primary key default gen_random_uuid(),
  shadow_signal_id text not null references shadow_signals(shadow_signal_id) on delete cascade,
  event_sequence integer not null,
  event_type text not null,
  status text null,
  outcome text null,
  recorded_at timestamptz not null default now(),
  payload jsonb not null default '{}'::jsonb,
  unique (shadow_signal_id, event_sequence)
);

create index if not exists shadow_signal_events_signal_idx
on shadow_signal_events(shadow_signal_id, event_sequence);

alter table shadow_signals enable row level security;
alter table shadow_signal_events enable row level security;

create table if not exists shadow_ops_cycles (
  cycle_id text primary key,
  started_at timestamptz null,
  finished_at timestamptz null,
  dry_run boolean not null default false,
  health_status text null,
  evaluated_closed integer not null default 0,
  evaluation_errors integer not null default 0,
  open_after_evaluation integer not null default 0,
  generation_skipped_reason text null,
  opened_signals integer not null default 0,
  configs_scanned integer not null default 0,
  skipped_hold integer not null default 0,
  skipped_duplicate_open integer not null default 0,
  skipped_duplicate_similar integer not null default 0,
  skipped_errors integer not null default 0,
  status_counts jsonb not null default '{}'::jsonb,
  final_open integer not null default 0,
  final_closed integer not null default 0,
  sync_supabase boolean not null default false,
  supabase_sync_ok boolean not null default false,
  supabase_sync_reason text null,
  research_only boolean not null default true,
  raw jsonb not null default '{}'::jsonb
);

create index if not exists shadow_ops_cycles_finished_idx
on shadow_ops_cycles(finished_at desc);

alter table shadow_ops_cycles enable row level security;

create table if not exists research_configs (
  config_id text primary key,
  source text not null default 'crypto_multi',
  status text null,
  classification text null,
  symbol text null,
  timeframe text null,
  strategy_mode text null,
  horizon_candles integer null,
  risk_reward numeric null,
  atr_stop_multiplier numeric null,
  cost_mode text null,
  median_validation_pf numeric null,
  median_validation_avg_return numeric null,
  median_test_pf numeric null,
  test_confirm_rate numeric null,
  validation_positive_rate numeric null,
  beats_random_rate numeric null,
  beats_deterministic_rate numeric null,
  worst_validation_drawdown numeric null,
  valid_windows integer null,
  label text null,
  config jsonb not null default '{}'::jsonb,
  metrics jsonb not null default '{}'::jsonb,
  raw jsonb not null default '{}'::jsonb,
  synced_at timestamptz not null default now()
);

create index if not exists research_configs_source_idx
on research_configs(source, classification);

create index if not exists research_configs_symbol_idx
on research_configs(symbol, timeframe);

create index if not exists research_configs_validation_pf_idx
on research_configs(median_validation_pf desc nulls last);

alter table research_configs enable row level security;

create table if not exists shadow_ops_locks (
  lock_name text primary key,
  owner_id text not null,
  acquired_at timestamptz not null default now(),
  expires_at timestamptz not null,
  heartbeat_at timestamptz not null default now(),
  cycle_id text null,
  metadata jsonb not null default '{}'::jsonb
);

alter table shadow_ops_locks enable row level security;
