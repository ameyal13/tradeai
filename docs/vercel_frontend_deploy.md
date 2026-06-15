# Vercel Frontend Deploy

Research only. No trading real. No exchange orders.

## What Vercel Should Deploy

Deploy the `frontend/` directory as a Vite React app.

Build command:

```text
npm run build
```

Output directory:

```text
dist
```

## Required Vercel Variables

Set this after Railway backend deploys:

```text
VITE_API_URL=https://your-railway-domain
```

Optional for future direct Supabase anon-only features:

```text
VITE_SUPABASE_URL
VITE_SUPABASE_ANON_KEY
```

Do not set:

```text
SUPABASE_SERVICE_ROLE_KEY
```

The dashboard should consume backend endpoints first:

```text
/shadow/health
/shadow/signals
/shadow/summary
```

## Local Frontend Smoke

With backend running locally:

```powershell
cd frontend
npm.cmd run build
npm.cmd run dev
```

Open:

```text
http://localhost:5173/signals
```

## Current Dashboard Scope

- Read-only shadow signal monitoring.
- Summary metrics.
- Recent signals.
- Guardrails visible.
- No trade execution controls.
- No exchange keys.
