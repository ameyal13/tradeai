# Free Shadow Ops Runtime

Research only. No trading signal. No exchange orders.

Railway is optional. When Railway is offline or the trial expires, TRADEAI can
keep running the shadow cycle with:

- Vercel: frontend
- Supabase: database
- GitHub Actions: hourly shadow cron and daily summary cron

## Required GitHub Secrets

Add these in GitHub:

`Settings -> Secrets and variables -> Actions -> New repository secret`

- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

Do not add these to the frontend. Do not commit real values.

The workflows set `TRADEAI_SUPABASE_FIRST=true`, so the Python scripts read and
write Supabase directly instead of using local JSONL journals.

## Workflows

Hourly shadow ops:

- File: `.github/workflows/shadow_ops.yml`
- Schedule: `0 * * * *`
- Command:
  `python scripts/run_shadow_ops_once.py --max-signals 1 --max-configs-scanned 21 --use-news-context --notify-telegram`

Daily summary:

- File: `.github/workflows/shadow_summary.yml`
- Schedule: `0 6 * * *`
- Command:
  `python scripts/run_shadow_summary_cron.py --notify-telegram`

Both workflows can also be run manually from the GitHub Actions tab with
`workflow_dispatch`.

## Supabase Read-Only Frontend Fallback

The Vercel frontend first tries the backend API. If the backend is offline, the
Shadow Signals page reads directly from Supabase using the frontend anon key.

For that to work, apply the RLS policies in `schema.sql` for:

- `shadow_signals`
- `shadow_signal_events`
- `shadow_ops_cycles`
- `research_configs`

Only `SELECT` is granted to `anon`. Writes still require backend/service-role
credentials.

## Vercel Environment Variables

The frontend needs only anon/read-only values:

- `VITE_SUPABASE_URL`
- `VITE_SUPABASE_ANON_KEY`
- `VITE_API_URL` can remain set to a backend URL, but the dashboard now falls
  back to Supabase when the backend is unavailable.

Never expose `SUPABASE_SERVICE_ROLE_KEY` in Vercel frontend variables.

## What This Does Not Do

- It does not place trades.
- It does not connect to an exchange account.
- It does not use real money.
- It does not make the strategy profitable.
- It only keeps collecting measured shadow outcomes while Railway is not used.

## Verification

1. Apply the updated `schema.sql` policies in Supabase.
2. Add GitHub secrets.
3. Push the repository.
4. In GitHub Actions, manually run `Shadow Ops`.
5. Confirm the run logs show `supabase_first: True`.
6. Confirm Telegram receives the research-only message.
7. Open Vercel `/signals`.
8. If the backend is offline, confirm the warning says Supabase fallback is active.
