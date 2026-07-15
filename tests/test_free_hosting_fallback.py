from pathlib import Path


def test_schema_includes_read_only_shadow_fallback_policies():
    schema = Path("schema.sql").read_text(encoding="utf-8")

    assert 'create policy "shadow_signals_anon_select_research_only"' in schema
    assert 'on shadow_signals for select' in schema
    assert 'to anon' in schema
    assert 'using (research_only = true)' in schema
    assert 'create policy "shadow_ops_cycles_anon_select_research_only"' in schema
    assert 'create policy "research_configs_anon_select"' in schema


def test_github_actions_shadow_ops_use_supabase_first():
    workflow = Path(".github/workflows/shadow_ops.yml").read_text(encoding="utf-8")

    assert 'TRADEAI_SUPABASE_FIRST: "true"' in workflow
    assert "python scripts/run_shadow_ops_once.py" in workflow
    assert "--max-signals 1" in workflow
    assert "--notify-telegram" in workflow


def test_github_actions_summary_uses_supabase_first():
    workflow = Path(".github/workflows/shadow_summary.yml").read_text(encoding="utf-8")

    assert 'TRADEAI_SUPABASE_FIRST: "true"' in workflow
    assert "python scripts/run_shadow_summary_cron.py --notify-telegram" in workflow
