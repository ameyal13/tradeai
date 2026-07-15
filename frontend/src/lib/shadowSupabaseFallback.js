import { supabase, supabaseConfigured } from './supabase.js'

function numeric(value, fallback = 0) {
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : fallback
}

function isClosed(row) {
  return row?.status === 'CLOSED' || row?.status === 'EXPIRED'
}

function isStrategyEligible(row) {
  const outcome = String(row?.outcome || '').toUpperCase()
  const exitReason = String(row?.exit_reason || '')
  return !['', 'EXPIRED', 'INVALID'].includes(outcome) && exitReason !== 'evaluation_http_error'
}

function maxDrawdownPct(returns) {
  let equity = 100
  let peak = equity
  let drawdown = 0
  for (const ret of returns) {
    equity *= 1 + ret / 100
    peak = Math.max(peak, equity)
    if (peak > 0) drawdown = Math.max(drawdown, ((peak - equity) / peak) * 100)
  }
  return Number(drawdown.toFixed(6))
}

function summarizeRows(rows) {
  const openRows = rows.filter((row) => row.status === 'OPEN')
  const closedRows = rows.filter(isClosed)
  const wins = closedRows.filter((row) => row.outcome === 'WIN')
  const losses = closedRows.filter((row) => row.outcome === 'LOSS')
  const expired = rows.filter((row) => row.outcome === 'EXPIRED' || row.status === 'EXPIRED')
  const returns = closedRows.map((row) => numeric(row.pnl_pct))
  const gains = returns.filter((value) => value > 0).reduce((acc, value) => acc + value, 0)
  const lossSum = Math.abs(returns.filter((value) => value < 0).reduce((acc, value) => acc + value, 0))
  return {
    total: rows.length,
    open: openRows.length,
    closed: closedRows.length,
    wins: wins.length,
    losses: losses.length,
    expired: expired.length,
    win_rate: closedRows.length ? Number(((wins.length / closedRows.length) * 100).toFixed(6)) : 0,
    profit_factor: lossSum ? Number((gains / lossSum).toFixed(6)) : gains > 0 ? null : 0,
    avg_return: returns.length ? Number((returns.reduce((acc, value) => acc + value, 0) / returns.length).toFixed(6)) : 0,
    total_return_pct: Number(returns.reduce((acc, value) => acc + value, 0).toFixed(6)),
    max_drawdown: maxDrawdownPct(returns),
  }
}

function groupSummaries(rows, key) {
  const groups = new Map()
  for (const row of rows) {
    const value = String(row[key] || 'unknown')
    groups.set(value, [...(groups.get(value) || []), row])
  }
  return Object.fromEntries([...groups.entries()].sort(([a], [b]) => a.localeCompare(b)).map(([name, group]) => [name, summarizeRows(group)]))
}

function strategySummary(rows) {
  const closed = rows.filter(isClosed)
  const eligible = closed.filter(isStrategyEligible)
  const excluded = closed.filter((row) => !isStrategyEligible(row))
  const byReason = {}
  for (const row of excluded) {
    const reason = String(row.exit_reason || row.outcome || 'UNKNOWN')
    byReason[reason] = (byReason[reason] || 0) + 1
  }
  return {
    summary: summarizeRows(eligible),
    technical_exclusions: excluded.length,
    technical_exclusions_by_exit_reason: Object.fromEntries(Object.entries(byReason).sort(([a], [b]) => a.localeCompare(b))),
  }
}

function classifyConfigHealth(summary, wins, losses, eligibleClosed) {
  const pf = summary.profit_factor === null || summary.profit_factor === undefined ? null : Number(summary.profit_factor)
  const avg = numeric(summary.avg_return)
  const drawdown = numeric(summary.max_drawdown)
  const reasons = []
  if (eligibleClosed < 5) return ['insufficient_sample', ['fewer_than_5_strategy_outcomes']]
  if (wins === 0 && losses >= 3) reasons.push('zero_wins_after_multiple_losses')
  if (pf !== null && pf < 0.75 && avg < 0) reasons.push('pf_below_0_75_and_avg_negative')
  if (eligibleClosed >= 8 && avg < 0 && drawdown >= 8) reasons.push('negative_avg_with_high_drawdown')
  if (reasons.length) return ['quarantine_candidate', reasons]
  if (eligibleClosed >= 10 && pf !== null && pf >= 1.15 && avg > 0) return ['keep_candidate', ['pf_above_1_15_and_avg_positive']]
  if (pf !== null && pf >= 1.0 && avg > 0) return ['watch', ['positive_but_sample_or_pf_not_enough']]
  return ['watch', ['mixed_or_unproven_live_evidence']]
}

function buildConfigHealth(rows) {
  const groups = new Map()
  for (const row of rows) {
    const configId = String(row.config_id || 'unknown')
    groups.set(configId, [...(groups.get(configId) || []), row])
  }
  const counts = {}
  const configs = [...groups.entries()].map(([configId, group]) => {
    const sorted = [...group].sort((a, b) => String(a.generated_at || a.updated_at || '').localeCompare(String(b.generated_at || b.updated_at || '')))
    const latest = sorted[sorted.length - 1] || {}
    const eligibleRows = group.filter((row) => isClosed(row) && isStrategyEligible(row))
    const technicalExclusions = group.filter((row) => isClosed(row) && !isStrategyEligible(row))
    const rowSummary = summarizeRows(eligibleRows)
    const [recommendation, reasons] = classifyConfigHealth(rowSummary, rowSummary.wins || 0, rowSummary.losses || 0, rowSummary.closed || 0)
    counts[recommendation] = (counts[recommendation] || 0) + 1
    const longCount = group.filter((row) => row.side === 'LONG').length
    const shortCount = group.filter((row) => row.side === 'SHORT').length
    return {
      config_id: configId,
      recommendation,
      reasons,
      symbol: latest.symbol,
      timeframe: latest.timeframe,
      classification: latest.classification,
      source_registry: latest.source_registry,
      strategy_mode: latest.strategy_mode,
      latest_status: latest.status,
      latest_outcome: latest.outcome,
      latest_generated_at: latest.generated_at,
      latest_updated_at: latest.updated_at,
      total_signals: group.length,
      open_signals: group.filter((row) => row.status === 'OPEN').length,
      operational_closed: group.filter(isClosed).length,
      strategy_closed: rowSummary.closed || 0,
      wins: rowSummary.wins || 0,
      losses: rowSummary.losses || 0,
      win_rate: rowSummary.win_rate,
      profit_factor: rowSummary.profit_factor,
      avg_return: rowSummary.avg_return,
      total_return_pct: rowSummary.total_return_pct,
      max_drawdown: rowSummary.max_drawdown,
      technical_exclusions: technicalExclusions.length,
      long_count: longCount,
      short_count: shortCount,
      direction_bias: longCount > shortCount * 2 ? 'LONG' : shortCount > longCount * 2 ? 'SHORT' : 'mixed',
    }
  })
  configs.sort((a, b) => (b.strategy_closed || 0) - (a.strategy_closed || 0))
  return {
    source: 'supabase_fallback',
    summary: {
      total_configs: configs.length,
      recommendation_counts: counts,
      research_only: true,
      auto_quarantine_enabled: false,
      min_strategy_outcomes_for_quarantine: 5,
      min_strategy_outcomes_for_keep: 10,
    },
    configs,
  }
}

export async function loadShadowSupabaseFallback({ signalLimit = 200, cycleLimit = 20 } = {}) {
  if (!supabaseConfigured || !supabase) {
    throw new Error('Supabase fallback unavailable: missing VITE_SUPABASE_URL or VITE_SUPABASE_ANON_KEY')
  }

  const [signalsResult, cyclesResult] = await Promise.all([
    supabase
      .from('shadow_signals')
      .select('*')
      .eq('research_only', true)
      .order('generated_at', { ascending: false })
      .limit(signalLimit),
    supabase
      .from('shadow_ops_cycles')
      .select('*')
      .eq('research_only', true)
      .order('finished_at', { ascending: false })
      .limit(cycleLimit),
  ])

  if (signalsResult.error) throw new Error(`Supabase fallback signals error: ${signalsResult.error.message}`)
  if (cyclesResult.error) throw new Error(`Supabase fallback cycles error: ${cyclesResult.error.message}`)

  const signals = signalsResult.data || []
  const cycles = cyclesResult.data || []
  const strategy = strategySummary(signals)
  const summary = {
    source: 'supabase_fallback',
    generated_at: new Date().toISOString(),
    summary: summarizeRows(signals),
    strategy_eligible: strategy.summary,
    technical_exclusions: strategy.technical_exclusions,
    technical_exclusions_by_exit_reason: strategy.technical_exclusions_by_exit_reason,
    by_symbol: groupSummaries(signals, 'symbol'),
    by_config: groupSummaries(signals, 'config_id'),
    by_timeframe: groupSummaries(signals, 'timeframe'),
    signals,
  }

  return {
    health: {
      source: 'supabase_fallback',
      supabase_available: true,
      summary: summary.summary,
      generated_at: summary.generated_at,
      fallback_reason: 'backend_unavailable',
    },
    summary,
    signals,
    cycles,
    configHealth: buildConfigHealth(signals),
  }
}
