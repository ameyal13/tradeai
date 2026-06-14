from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_dockerfile_uses_railway_port():
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "ENV PORT=8000" in text
    assert "${PORT:-8000}" in text
    assert "localhost:{os.getenv" in text
    assert "--host 0.0.0.0" in text


def test_dockerignore_excludes_secrets_and_generated_state():
    text = (ROOT / ".dockerignore").read_text(encoding="utf-8")

    for entry in [".env", "frontend/.env", "data", "reports", "logs", ".venv"]:
        assert entry in text


def test_railway_docs_are_research_only_and_no_secret_values():
    text = (ROOT / "docs" / "railway_backend_deploy.md").read_text(encoding="utf-8")

    assert "Research only" in text
    assert "No exchange orders" in text
    assert "SUPABASE_SERVICE_ROLE_KEY" in text
    assert "your_service_role_key" not in text
    assert "TELEGRAM_BOT_TOKEN=" not in text
