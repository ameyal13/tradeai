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


def test_shadow_dashboard_exposes_monitoring_panels():
    page_source = (ROOT / "frontend" / "src" / "components" / "ShadowSignalsPage.jsx").read_text(encoding="utf-8")

    assert "Active shadow signal" in page_source
    assert "Data freshness" in page_source
    assert "Latest signal context" in page_source
    assert "Top configs" in page_source
    assert "Config performance" in page_source
    assert "fewer than 30 closed shadow signals" in page_source
    assert "Confidence vs outcome" in page_source
    assert "Do not raise or lower thresholds from this view alone" in page_source
