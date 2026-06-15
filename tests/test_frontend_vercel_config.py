import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_vercel_spa_rewrite_exists():
    config_path = ROOT / "frontend" / "vercel.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    rewrites = config.get("rewrites", [])
    assert {
        "source": "/(.*)",
        "destination": "/index.html",
    } in rewrites


def test_frontend_defines_signals_route():
    app_source = (ROOT / "frontend" / "src" / "App.jsx").read_text(encoding="utf-8")

    assert 'path="signals"' in app_source
    assert "<ShadowSignalsPage />" in app_source


def test_frontend_reads_vite_api_url():
    api_source = (ROOT / "frontend" / "src" / "lib" / "api.js").read_text(encoding="utf-8")

    assert "import.meta.env.VITE_API_URL" in api_source
    assert ".replace(/\\/+$/, '')" in api_source
