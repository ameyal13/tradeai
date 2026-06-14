# Shadow Supabase Readiness

Research only. No trading signal. No exchange orders.

## What This Adds

- `shadow_signals`: latest state for each shadow/paper signal.
- `shadow_signal_events`: append-only event history copied from the local JSONL journal.
- Read-only API endpoints:
  - `GET /shadow/health`
  - `GET /shadow/signals`
  - `GET /shadow/summary`
- Local sync script:
  - `scripts/sync_shadow_journal_to_supabase.py`

The frontend should call the backend API. Do not expose `SUPABASE_SERVICE_ROLE_KEY`
to Vercel/frontend code.

## Manual Supabase Step

1. Open Supabase SQL Editor.
2. Apply the shadow table section from `schema.sql`.
3. Keep RLS enabled.
4. Do not create public anon policies unless you intentionally want public reads.

The backend service role can read/write these tables. The frontend should use
the backend API instead of direct table access.

## Local Verification

Dry-run sync:

```powershell
.\.venv\Scripts\python.exe scripts\sync_shadow_journal_to_supabase.py --dry-run
```

Actual sync, after applying schema:

```powershell
.\.venv\Scripts\python.exe scripts\sync_shadow_journal_to_supabase.py
```

API smoke:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_api_smoke.py tests\test_shadow_signal_repository.py -q
```

## Deployment Path

1. Apply `schema.sql` in Supabase.
2. Run local sync once.
3. Deploy backend to Railway with:
   - `SUPABASE_URL`
   - `SUPABASE_SERVICE_ROLE_KEY`
   - Telegram vars if notifications are needed.
4. Deploy frontend to Vercel with only anon/public frontend vars.
5. Build the dashboard against backend read-only endpoints first.

Do not move real exchange execution into this flow.
