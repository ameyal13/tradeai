"""Runtime environment helpers for local CLI scripts."""
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_project_env() -> bool:
    """Load the project .env for CLI entrypoints without overriding process env."""
    env_path = project_root() / ".env"
    if not env_path.exists():
        return False
    return bool(load_dotenv(dotenv_path=env_path, override=False))
