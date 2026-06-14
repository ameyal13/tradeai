# Railway Backend Deploy

Research only. No trading real. No exchange orders.

## What Railway Should Run

Railway should deploy the FastAPI backend from the repo root using `Dockerfile`.

The container starts:

```text
uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1
```

Railway provides `PORT` automatically.

## Required Railway Variables

Set these in Railway service variables:

```text
SUPABASE_URL
SUPABASE_SERVICE_ROLE_KEY
CORS_ORIGINS
```

Recommended `CORS_ORIGINS` before Vercel exists:

```text
http://localhost:5173
```

After Vercel deploy, set it to:

```text
https://your-vercel-domain.vercel.app,http://localhost:5173
```

Optional, for Telegram-capable backend scripts:

```text
TELEGRAM_BOT_TOKEN
TELEGRAM_CHAT_ID
CRYPTOPANIC_API_KEY
AGENT_PROVIDER
AGENT_MODEL
```

Do not set frontend-only `VITE_*` variables in backend unless a future need is
explicitly documented.

## Do Not Put In Vercel

Never put this in Vercel/frontend:

```text
SUPABASE_SERVICE_ROLE_KEY
```

The frontend should call backend endpoints first:

```text
GET /health
GET /shadow/health
GET /shadow/signals
GET /shadow/summary
```

## Manual Deploy Steps

1. Create a Railway project.
2. Add a service from GitHub repo.
3. Use the repo root as service root.
4. Let Railway detect/use the `Dockerfile`.
5. Add required variables above.
6. Deploy.
7. Open the generated Railway domain:

```text
https://your-railway-domain/health
https://your-railway-domain/shadow/summary
```

## After Deploy

If `/shadow/summary` is empty:

1. Confirm `schema.sql` was applied in Supabase.
2. Confirm `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` are set in Railway.
3. Run local sync once:

```powershell
.\.venv\Scripts\python.exe scripts\sync_shadow_journal_to_supabase.py
```

4. Refresh `/shadow/summary`.

## What Not To Deploy Yet

- No real trading.
- No exchange keys.
- No exchange orders.
- No Railway cron for heavy research replay yet.
- No scheduler loop inside the web process.

Keep Windows Task Scheduler as the local shadow cycle until the Railway API and
dashboard are stable.
