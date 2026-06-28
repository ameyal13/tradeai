# Trading Copilot

Trading Copilot is a crypto trading research system. It is not financial advice and it does not guarantee profitability.

## Runtime

Recommended Python version:

```powershell
Python 3.11.9
```

Do not use Python 3.14 for this project right now. The pinned FastAPI/Pydantic stack depends on `pydantic-core`, and installs can fail on Python 3.14 because compatible wheels may not be available.

Render runtime hint:

```text
python-3.11.9
```

## Backend Setup

```powershell
cd C:\Users\david\OneDrive\Desktop\TRADEAI
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run the backend:

```powershell
.\.venv\Scripts\python.exe -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Run tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Frontend Setup

```powershell
cd C:\Users\david\OneDrive\Desktop\TRADEAI\frontend
npm.cmd install
npm.cmd run dev
```

Frontend dev server: `http://localhost:5173`

## Environment Variables

Use `.env.example` and `frontend/.env.example` as templates. Do not commit real secrets.

Supabase is optional for local journal/replay development because the backend has a file-based fallback for prediction journal data under `data/`, which is gitignored.

## Dependency Notes

The stack is tested/recommended for Python 3.11. Direct dependencies without
any source import are intentionally omitted. Add a package only when a tested
runtime path requires it; Railway does not use Celery, Redis, or an external
scheduler in the current architecture.

## Shadow Live Performance Audit

Railway writes shadow signals and outcomes directly to Supabase. Run the
read-only live audit with:

```powershell
.\.venv\Scripts\python.exe scripts\analyze_shadow_live_performance.py --source supabase
```

The report is saved under `reports/shadow/`. It compares live shadow outcomes
by symbol, side, config, confidence, and review status. It never writes to
Supabase, changes a model, or generates a signal.

## Manual Real-Market Smoke

This script uses the internet and should not be run as part of unit tests:

```powershell
.\.venv\Scripts\python.exe -m scripts.smoke_real_market
.\.venv\Scripts\python.exe scripts\smoke_real_market.py
```

## Paper Trading Manual Loop

This MVP flow only records and evaluates paper-trading predictions. It does not place real exchange orders.

Generate one batch of signals:

```powershell
.\.venv\Scripts\python.exe scripts\generate_signals_once.py
```

Wait for the configured horizon, currently 60 minutes by default, then evaluate due predictions:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_due_once.py
```

The generator skips a symbol/timeframe/strategy mode when a recent pending prediction already exists inside the configured horizon.
