from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WINDOWS_DIR = ROOT / "scripts" / "windows"


def read_script(name: str) -> str:
    return (WINDOWS_DIR / name).read_text(encoding="utf-8")


def test_shadow_ops_wrapper_runs_safe_cycle_and_logs():
    text = read_script("run_shadow_ops_task.ps1")

    assert "scripts\\run_shadow_ops_once.py" in text
    assert "--max-signals 1" in text
    assert "--max-configs-scanned 21" in text
    assert "--use-news-context" in text
    assert "--notify-telegram" in text
    assert "--sync-supabase" in text
    assert "logs\\shadow_ops" in text
    assert "exit $ExitCode" in text
    assert "TELEGRAM_BOT_TOKEN" not in text
    assert "TELEGRAM_CHAT_ID" not in text


def test_summary_wrapper_runs_shadow_summary_and_logs():
    text = read_script("run_shadow_summary_task.ps1")

    assert "scripts\\summarize_shadow_signals.py" in text
    assert "--notify-telegram" in text
    assert "logs\\shadow_ops" in text
    assert "exit $ExitCode" in text


def test_install_and_uninstall_scripts_reference_expected_tasks():
    install = read_script("install_shadow_ops_tasks.ps1")
    uninstall = read_script("uninstall_shadow_ops_tasks.ps1")

    assert "TRADEAI Shadow Ops Hourly" in install
    assert "TRADEAI Shadow Summary Daily" in install
    assert "RunOnlyIfNetworkAvailable" in install
    assert "run_shadow_ops_task.ps1" in install
    assert "run_shadow_summary_task.ps1" in install
    assert "TRADEAI Shadow Ops Hourly" in uninstall
    assert "TRADEAI Shadow Summary Daily" in uninstall


def test_manual_test_script_does_not_register_scheduler():
    text = read_script("test_shadow_ops_task.ps1")

    assert "shadow_ops_healthcheck.py" in text
    assert "--test-telegram" in text
    assert "run_shadow_ops_task.ps1" in text
    assert "Register-ScheduledTask" not in text


def test_logs_directory_is_gitignored():
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")

    assert "logs/" in gitignore
