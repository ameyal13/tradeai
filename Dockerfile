# Dockerfile for the FastAPI backend at repo root.
FROM python:3.11-slim

WORKDIR /app

# Instala dependencias del sistema mínimas
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Dependencias de Python primero (mejor cache de layers)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Código de la app
COPY . .

# Puerto que expone Railway/Render. Railway injects PORT at runtime.
ENV PORT=8000
EXPOSE 8000

# Health check para Railway/Render
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import os, httpx; httpx.get(f'http://localhost:{os.getenv(\"PORT\", \"8000\")}/health').raise_for_status()"

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
