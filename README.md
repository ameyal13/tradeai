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

The stack is tested/recommended for Python 3.11. Some packages in `requirements.txt` may be future-facing or not actively used in the current code paths:

- `celery`
- `redis`
- `pydantic-settings`
- `langchain-community`
- `python-jose[cryptography]`
- `beautifulsoup4`
- `aiohttp`

These were not removed in this stabilization pass to avoid accidental breakage.

## Manual Real-Market Smoke

This script uses the internet and should not be run as part of unit tests:

```powershell
.\.venv\Scripts\python.exe -m scripts.smoke_real_market
.\.venv\Scripts\python.exe scripts\smoke_real_market.py
```
