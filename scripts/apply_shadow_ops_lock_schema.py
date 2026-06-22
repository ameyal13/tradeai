"""Apply the Shadow Ops distributed lock schema to Supabase.

This script is intentionally narrow: it extracts the ``shadow_ops_locks`` block
from schema.sql and applies only that block. It does not modify models,
research configs, signals, or journals.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.sync_shadow_journal_to_supabase import build_supabase_client_from_env  # noqa: E402
from tools.runtime_env import load_project_env  # noqa: E402


START = "create table if not exists shadow_ops_locks"
END = "alter table shadow_ops_locks enable row level security;"


def extract_shadow_ops_lock_sql(schema_path: str | Path = PROJECT_ROOT / "schema.sql") -> str:
    text = Path(schema_path).read_text(encoding="utf-8")
    start = text.find(START)
    if start < 0:
        raise RuntimeError("shadow_ops_locks block not found in schema.sql")
    end = text.find(END, start)
    if end < 0:
        raise RuntimeError("shadow_ops_locks RLS statement not found in schema.sql")
    end += len(END)
    return text[start:end].strip() + "\n"


def db_url_from_env() -> str | None:
    for key in ("SUPABASE_DB_URL", "DATABASE_URL", "POSTGRES_URL"):
        value = os.getenv(key)
        if value and "your_" not in value and "xxxx" not in value:
            return value
    return None


def apply_with_psql(sql: str, db_url: str) -> dict[str, object]:
    psql = shutil.which("psql")
    if not psql:
        return {"ok": False, "reason": "psql_not_found"}
    with tempfile.NamedTemporaryFile("w", suffix=".sql", delete=False, encoding="utf-8") as handle:
        handle.write(sql)
        sql_path = handle.name
    try:
        proc = subprocess.run(
            [psql, db_url, "-v", "ON_ERROR_STOP=1", "-f", sql_path],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        return {
            "ok": proc.returncode == 0,
            "reason": None if proc.returncode == 0 else "psql_failed",
            "stdout": proc.stdout.strip()[-800:],
            "stderr": proc.stderr.strip()[-800:],
        }
    finally:
        Path(sql_path).unlink(missing_ok=True)


def apply_with_rpc(sql: str) -> dict[str, object]:
    supabase = build_supabase_client_from_env()
    if supabase is None:
        return {"ok": False, "reason": "supabase_not_configured"}
    try:
        supabase.rpc("exec_sql", {"sql": sql}).execute()
        return {"ok": True, "reason": None}
    except Exception as exc:  # noqa: BLE001 - report safely, do not print secrets.
        return {"ok": False, "reason": "exec_sql_rpc_unavailable", "error": f"{type(exc).__name__}: {str(exc)[:240]}"}


def apply_schema(sql: str) -> dict[str, object]:
    db_url = db_url_from_env()
    if db_url:
        result = apply_with_psql(sql, db_url)
        if result.get("ok"):
            result["method"] = "psql"
            return result
        rpc_result = apply_with_rpc(sql)
        rpc_result["psql_result"] = {key: value for key, value in result.items() if key not in {"stdout", "stderr"}}
        rpc_result["method"] = "rpc_fallback"
        return rpc_result
    result = apply_with_rpc(sql)
    result["method"] = "rpc"
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply shadow_ops_locks schema to Supabase.")
    parser.add_argument("--dry-run", action="store_true", default=False)
    return parser


def main() -> None:
    load_project_env()
    args = build_parser().parse_args()
    sql = extract_shadow_ops_lock_sql()
    if args.dry_run:
        print(sql)
        return
    result = apply_schema(sql)
    print("Shadow Ops lock schema apply")
    print("Research only. No trading signal.")
    print(f"ok: {result.get('ok')}")
    print(f"method: {result.get('method')}")
    print(f"reason: {result.get('reason')}")
    if result.get("stderr"):
        print(f"stderr: {result.get('stderr')}")
    if result.get("error"):
        print(f"error: {result.get('error')}")
    if not result.get("ok"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
