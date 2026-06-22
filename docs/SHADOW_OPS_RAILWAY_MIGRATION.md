# Shadow Ops Railway Migration Design

Research only. No trading signal. No exchange orders.

This document describes the reliability design for moving Shadow Ops from a
single local Windows scheduler to a Railway-hosted worker later. It is a design
note only; Railway execution is not implemented in this phase.

## Goals

- Run one shadow ops cycle at a time.
- Prevent duplicate signal generation if two processes start concurrently.
- Keep the local/Supabase journal append-only and auditable.
- Detect stale or missing cycles.
- Preserve current guardrails: no real trading, no exchange orders, no test
  metrics used for selection.

## Proposed Distributed Lock

Use a dedicated Supabase table, not an application-local lock file, because
Railway may run more than one process or restart a process on another host.

Suggested table:

```sql
create table if not exists shadow_ops_locks (
  lock_name text primary key,
  owner_id text not null,
  acquired_at timestamptz not null default now(),
  expires_at timestamptz not null,
  heartbeat_at timestamptz not null default now(),
  cycle_id text null,
  metadata jsonb not null default '{}'::jsonb
);
```

Primary lock:

```text
lock_name = 'shadow_ops_once'
```

Acquisition rule:

1. Generate a unique `owner_id` per process, for example hostname + pid +
   UUID.
2. Attempt to insert `lock_name='shadow_ops_once'`.
3. If insert succeeds, the process owns the lock.
4. If insert fails because the row exists:
   - read `expires_at`;
   - if `expires_at > now()`, another process owns it, so skip this cycle;
   - if `expires_at <= now()`, atomically replace the row only if the old
     `expires_at` is still expired.

The safest implementation is a Supabase RPC/Postgres function that performs the
insert-or-steal operation transactionally. A plain read-then-write in Python can
still race.

## Orphaned Locks

An orphaned lock can happen if the process dies after acquiring the lock but
before releasing it.

Mitigation:

- Locks must have a short TTL, for example 20-40 minutes for an hourly cycle.
- The worker updates `heartbeat_at` during long operations.
- Another process may steal the lock only after `expires_at`.
- The cycle record should include:
  - `cycle_id`
  - `started_at`
  - `finished_at`
  - `health_status`
  - `opened_signals`
  - `configs_scanned`
  - `supabase_sync_ok`
  - `error` if any

If a lock expires but the previous cycle has no `finished_at`, the next process
may run, but the healthcheck should flag:

```text
stale_lock_or_incomplete_cycle
```

The lock release should delete the lock row only if `owner_id` matches the
current process. A process must never delete a lock owned by another owner.

## Healthcheck Detection For Missing Cycles

The existing local healthcheck can be extended for Railway by checking
Supabase `shadow_ops_cycles`.

Suggested checks:

1. Read the latest row from `shadow_ops_cycles` ordered by `finished_at desc`.
2. Compare `finished_at` to `now()`.
3. If the latest finished cycle is older than the expected cadence plus grace
   period, mark warning:

```text
shadow_ops_cycle_stale
```

Example:

- expected cadence: 1 hour
- grace period: 20 minutes
- warning threshold: latest `finished_at` older than 80 minutes

4. If a lock exists and `expires_at < now()`, mark warning:

```text
stale_shadow_ops_lock
```

5. If a lock exists, `expires_at > now()`, but `heartbeat_at` is older than a
   configured threshold, mark warning:

```text
lock_heartbeat_stale
```

6. If there are OPEN signals past `expires_at`, mark warning:

```text
open_signals_due_for_evaluation
```

## Why Not Reuse The Local Lock File

The local lock file is sufficient for Windows Task Scheduler on one machine.
It is not sufficient for Railway because:

- instances may restart;
- filesystem state may be ephemeral;
- more than one worker could start;
- a local lock cannot coordinate across processes on different hosts.

## Migration Sequence

Recommended order:

1. Keep Windows Task Scheduler as the current operating mode.
2. Add the Supabase lock table and an RPC for atomic acquire/release.
3. Add a dry-run Railway worker that only runs healthcheck and lock acquire
   tests.
4. Enable Railway shadow ops without signal generation first.
5. Enable evaluation and Supabase sync.
6. Enable generation only after duplicate prevention is verified.

Do not enable real trading. This is still shadow/paper research only.
