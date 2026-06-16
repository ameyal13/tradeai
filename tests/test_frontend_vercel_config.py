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
    layout_source = (ROOT / "frontend" / "src" / "components" / "Layout.jsx").read_text(encoding="utf-8")

    assert 'path="signals"' in app_source
    assert "<ShadowSignalsPage />" in app_source
    assert 'path="research"' in app_source
    assert "<ResearchPage />" in app_source
    assert "/research" in layout_source


def test_frontend_reads_vite_api_url():
    api_source = (ROOT / "frontend" / "src" / "lib" / "api.js").read_text(encoding="utf-8")

    assert "import.meta.env.VITE_API_URL" in api_source
    assert ".replace(/\\/+$/, '')" in api_source
    assert "/research/summary" in api_source


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
    assert "Last cycle diagnostics" in page_source
    assert "Skipped HOLD" in page_source
    assert "api.shadow.cycles" in page_source


def test_research_dashboard_supports_focused_source():
    page_source = (ROOT / "frontend" / "src" / "components" / "ResearchPage.jsx").read_text(encoding="utf-8")

    assert "crypto_multi" in page_source
    assert "focused_v2a" in page_source
    assert "api.research.summary" in page_source
