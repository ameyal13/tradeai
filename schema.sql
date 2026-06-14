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
