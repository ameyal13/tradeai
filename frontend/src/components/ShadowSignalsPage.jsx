import { useEffect, useMemo, useState } from 'react'
import { Activity, AlertTriangle, Clock, Database, RefreshCw, ShieldCheck, TrendingUp } from 'lucide-react'
import { api } from '../lib/api.js'

function formatNumber(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '-'
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: digits })
}

function formatPct(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '-'
  return `${formatNumber(value, digits)}%`
}

function formatDate(value) {
  if (!value) return '-'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return '-'
  return date.toLocaleString()
}

function minutesUntil(value) {
  if (!value) return null
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return null
  return Math.round((date.getTime() - Date.now()) / 60000)
}

function shortId(value) {
  return value ? String(value).slice(0, 8) : '-'
}

function toNumber(value, fallback = 0) {
  const numeric = Number(value)
  return Number.isFinite(numeric) ? numeric : fallback
}

function statusClass(status) {
  if (status === 'OPEN') return 'badge-medium'
  if (status === 'CLOSED') return 'badge-low'
  if (status === 'EXPIRED') return 'badge-hold'
  if (status === 'BLOCKED') return 'badge-high'
  return 'badge-low'
}

function outcomeClass(outcome) {
  if (outcome === 'WIN') return 'text-green'
  if (outcome === 'LOSS') return 'text-red'
  if (outcome === 'BREAKEVEN') return 'text-amber'
  return 'text-muted'
}

function Kpi({ label, value, tone = 'neutral' }) {
  const color = tone === 'good' ? 'var(--green)' : tone === 'bad' ? 'var(--red)' : tone === 'warn' ? 'var(--amber)' : 'var(--text)'
  return (
    <section className="card">
      <div className="text-muted mono" style={{ fontSize: 11, marginBottom: 8 }}>{label}</div>
      <div className="mono" style={{ fontSize: 24, color }}>{value}</div>
    </section>
  )
}

function StatLine({ label, value, tone = 'neutral' }) {
  const className = tone === 'good' ? 'text-green' : tone === 'bad' ? 'text-red' : tone === 'warn' ? 'text-amber' : ''
  return (
    <div className="stat-line">
      <span className="text-muted">{label}</span>
      <span className={`mono ${className}`}>{value}</span>
    </div>
  )
}

function GuardrailStrip({ health }) {
  return (
    <section
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: 16,
        padding: '12px 16px',
        border: '1px solid rgba(0,212,160,0.22)',
        background: 'rgba(0,212,160,0.07)',
        borderRadius: 'var(--radius)',
        marginBottom: 16,
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        <ShieldCheck size={18} color="var(--green)" />
        <div>
          <div className="mono" style={{ color: 'var(--green)', fontSize: 13 }}>Research only. No trading signal.</div>
          <div className="text-muted" style={{ fontSize: 12 }}>
            No exchange orders. Source: {health?.source || '-'} | Supabase: {health?.supabase_available ? 'available' : 'not configured'}
          </div>
        </div>
      </div>
      <div className="mono text-muted" style={{ fontSize: 11 }}>{formatDate(health?.timestamp)}</div>
    </section>
  )
}

function FreshnessPanel({ health, summary, signals }) {
  const newest = signals[0] || {}
  const latestUpdated = newest.updated_at || newest.recorded_at || newest.generated_at || health?.timestamp
  const source = summary?.source || health?.source || '-'
  return (
    <section className="card">
      <h2 className="panel-title"><Database size={15} /> Data freshness</h2>
      <div className="stack">
        <StatLine label="Source" value={source} />
        <StatLine label="Last backend check" value={formatDate(health?.timestamp)} />
        <StatLine label="Last signal update" value={formatDate(latestUpdated)} />
        <StatLine label="Supabase" value={health?.supabase_available ? 'available' : 'not configured'} tone={health?.supabase_available ? 'good' : 'warn'} />
      </div>
    </section>
  )
}

function ActiveSignalPanel({ signal }) {
  if (!signal) {
    return (
      <section className="card">
        <h2 className="panel-title"><Clock size={15} /> Active shadow signal</h2>
        <div className="text-muted">No OPEN shadow signal right now.</div>
      </section>
    )
  }
  const remaining = minutesUntil(signal.expires_at)
  const remainingLabel = remaining === null ? '-' : remaining >= 0 ? `${remaining} min` : `${Math.abs(remaining)} min overdue`
  return (
    <section className="card active-signal">
      <div className="active-signal-head">
        <h2 className="panel-title"><Clock size={15} /> Active shadow signal</h2>
        <span className="badge badge-medium">OPEN</span>
      </div>
      <div className="active-main">
        <div>
          <div className="mono active-symbol">{signal.symbol} {signal.timeframe}</div>
          <div style={{ marginTop: 8 }}>
            <span className={`badge ${signal.side === 'LONG' ? 'badge-buy' : 'badge-sell'}`}>{signal.side || '-'}</span>
          </div>
        </div>
        <div className="active-levels">
          <StatLine label="Entry" value={formatNumber(signal.entry_price, 6)} />
          <StatLine label="Stop loss" value={formatNumber(signal.stop_loss, 6)} tone="bad" />
          <StatLine label="Take profit" value={formatNumber(signal.take_profit, 6)} tone="good" />
          <StatLine label="Expires in" value={remainingLabel} tone={remaining !== null && remaining < 0 ? 'warn' : 'neutral'} />
        </div>
      </div>
      <div className="active-meta">
        <span>RR {formatNumber(signal.risk_reward, 2)}</span>
        <span>Confidence {formatPct(signal.confidence)}</span>
        <span>Horizon {signal.horizon_candles || '-'} candles / {signal.horizon_minutes || '-'} min</span>
        <span>Config {shortId(signal.config_id)}</span>
      </div>
    </section>
  )
}

function SignalContextPanel({ signal }) {
  const features = signal?.input_features || signal?.raw?.input_features || {}
  const review = signal?.agent_review || signal?.raw?.agent_review || {}
  const news = signal?.news_context || signal?.raw?.news_context || {}
  const market = signal?.market_context || signal?.raw?.market_context || {}
  const riskFlags = Array.isArray(review.risk_flags) ? review.risk_flags : []
  return (
    <section className="card">
      <h2 className="panel-title"><TrendingUp size={15} /> Latest signal context</h2>
      {!signal ? (
        <div className="text-muted">No signal context available yet.</div>
      ) : (
        <div className="stack">
          <StatLine label="Latest signal" value={`${signal.symbol || '-'} ${signal.side || '-'} ${signal.status || '-'}`} />
          <StatLine label="Model" value={signal.model_name || features.model_name || '-'} />
          <StatLine label="Buy probability" value={formatPct(features.probability_buy_win, 2)} />
          <StatLine label="Sell probability" value={formatPct(features.probability_sell_win, 2)} />
          <StatLine label="Agent review" value={review.review_status || '-'} tone={review.review_status === 'BLOCK' ? 'bad' : review.review_status === 'CAUTION' ? 'warn' : 'neutral'} />
          <StatLine label="News context" value={news.context_status || news.provider_status || news.sentiment || '-'} />
          <StatLine label="Market context" value={market.context_status || market.review_status || '-'} />
          <div>
            <div className="text-muted" style={{ marginBottom: 6 }}>Risk flags</div>
            <div className="tag-row">
              {riskFlags.length ? riskFlags.slice(0, 4).map((flag) => <span className="tag" key={flag}>{flag}</span>) : <span className="text-muted">None recorded</span>}
            </div>
          </div>
        </div>
      )}
    </section>
  )
}

function buildConfigRows(signals, byConfig) {
  const grouped = new Map()
  for (const signal of signals) {
    const configId = signal.config_id || 'unknown'
    if (!grouped.has(configId)) {
      grouped.set(configId, [])
    }
    grouped.get(configId).push(signal)
  }

  return Array.from(grouped.entries()).map(([configId, rows]) => {
    const summary = byConfig?.[configId] || {}
    const latest = [...rows].sort((a, b) => String(b.updated_at || b.recorded_at || b.generated_at || '').localeCompare(String(a.updated_at || a.recorded_at || a.generated_at || '')))[0] || {}
    const longCount = rows.filter((row) => row.side === 'LONG').length
    const shortCount = rows.filter((row) => row.side === 'SHORT').length
    const closed = toNumber(summary.closed)
    const wins = toNumber(summary.wins)
    const losses = toNumber(summary.losses)
    const avgConfidence = rows.reduce((acc, row) => acc + toNumber(row.confidence), 0) / Math.max(rows.length, 1)
    return {
      configId,
      summary,
      latest,
      symbol: latest.symbol || '-',
      timeframe: latest.timeframe || '-',
      classification: latest.classification || '-',
      modelName: latest.model_name || latest.input_features?.model_name || '-',
      total: toNumber(summary.total, rows.length),
      open: toNumber(summary.open),
      closed,
      wins,
      losses,
      winRate: toNumber(summary.win_rate),
      profitFactor: summary.profit_factor,
      avgReturn: toNumber(summary.avg_return),
      drawdown: toNumber(summary.max_drawdown),
      longCount,
      shortCount,
      avgConfidence,
      horizon: latest.horizon_candles || '-',
      horizonMinutes: latest.horizon_minutes || '-',
      riskReward: latest.risk_reward,
      lastUpdated: latest.updated_at || latest.recorded_at || latest.generated_at,
    }
  }).sort((a, b) => {
    const pfA = a.profitFactor === null || a.profitFactor === undefined ? -Infinity : Number(a.profitFactor)
    const pfB = b.profitFactor === null || b.profitFactor === undefined ? -Infinity : Number(b.profitFactor)
    if (pfA !== pfB) return pfB - pfA
    return b.avgReturn - a.avgReturn
  })
}

function summarizeBucket(rows) {
  const closed = rows.filter((row) => row.status === 'CLOSED' || row.status === 'EXPIRED')
  const wins = closed.filter((row) => row.outcome === 'WIN')
  const losses = closed.filter((row) => row.outcome === 'LOSS')
  const returns = closed.map((row) => toNumber(row.pnl_pct))
  const profits = returns.filter((value) => value > 0).reduce((acc, value) => acc + value, 0)
  const lossSum = Math.abs(returns.filter((value) => value < 0).reduce((acc, value) => acc + value, 0))
  return {
    total: rows.length,
    closed: closed.length,
    wins: wins.length,
    losses: losses.length,
    winRate: closed.length ? wins.length / closed.length * 100 : 0,
    avgReturn: closed.length ? returns.reduce((acc, value) => acc + value, 0) / closed.length : 0,
    profitFactor: lossSum > 0 ? profits / lossSum : null,
  }
}

function buildConfidenceBuckets(signals) {
  const buckets = [
    { label: '<50%', min: -Infinity, max: 50, rows: [] },
    { label: '50-60%', min: 50, max: 60, rows: [] },
    { label: '60-70%', min: 60, max: 70, rows: [] },
    { label: '70-80%', min: 70, max: 80, rows: [] },
    { label: '80%+', min: 80, max: Infinity, rows: [] },
  ]
  for (const signal of signals) {
    const confidence = toNumber(signal.confidence, NaN)
    if (!Number.isFinite(confidence)) continue
    const bucket = buckets.find((item) => confidence >= item.min && confidence < item.max)
    if (bucket) bucket.rows.push(signal)
  }
  return buckets.map((bucket) => ({
    label: bucket.label,
    ...summarizeBucket(bucket.rows),
  }))
}

function SymbolSummary({ bySymbol }) {
  const rows = Object.entries(bySymbol || {})
  return (
    <section className="card">
      <h2 className="mono" style={{ fontSize: 15, marginBottom: 12 }}>By Symbol</h2>
      {rows.length === 0 ? (
        <div className="text-muted">No symbol data yet.</div>
      ) : (
        <div style={{ overflowX: 'auto' }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>Symbol</th>
                <th>Total</th>
                <th>Open</th>
                <th>Closed</th>
                <th>Win Rate</th>
                <th>PF</th>
                <th>Avg Return</th>
                <th>Drawdown</th>
              </tr>
            </thead>
            <tbody>
              {rows.map(([symbol, item]) => (
                <tr key={symbol}>
                  <td className="mono">{symbol}</td>
                  <td>{item.total}</td>
                  <td>{item.open}</td>
                  <td>{item.closed}</td>
                  <td>{formatPct(item.win_rate)}</td>
                  <td>{formatNumber(item.profit_factor, 4)}</td>
                  <td className={Number(item.avg_return) >= 0 ? 'text-green' : 'text-red'}>{formatPct(item.avg_return, 4)}</td>
                  <td>{formatPct(item.max_drawdown)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  )
}

function ConfigSummary({ byConfig }) {
  const rows = Object.entries(byConfig || {})
    .sort(([, a], [, b]) => Number(b.closed || 0) - Number(a.closed || 0))
    .slice(0, 8)
  return (
    <section className="card">
      <h2 className="panel-title">Top configs</h2>
      {rows.length === 0 ? (
        <div className="text-muted">No config data yet.</div>
      ) : (
        <div style={{ overflowX: 'auto' }}>
          <table className="data-table compact-table">
            <thead>
              <tr>
                <th>Config</th>
                <th>Total</th>
                <th>Win Rate</th>
                <th>PF</th>
                <th>Avg Return</th>
              </tr>
            </thead>
            <tbody>
              {rows.map(([config, item]) => (
                <tr key={config}>
                  <td className="mono">{shortId(config)}</td>
                  <td>{item.total}</td>
                  <td>{formatPct(item.win_rate)}</td>
                  <td>{formatNumber(item.profit_factor, 4)}</td>
                  <td className={Number(item.avg_return) >= 0 ? 'text-green' : 'text-red'}>{formatPct(item.avg_return, 4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  )
}

function ConfigDeepDive({ configRows, selectedConfig, onSelectConfig }) {
  const selected = configRows.find((row) => row.configId === selectedConfig) || configRows[0]
  const sampleWarning = selected && selected.closed < 30
  return (
    <section className="card">
      <div className="section-head">
        <h2 className="panel-title">Config performance</h2>
        <select
          value={selected?.configId || ''}
          onChange={(event) => onSelectConfig(event.target.value)}
          disabled={configRows.length === 0}
        >
          {configRows.length === 0 && <option value="">No configs</option>}
          {configRows.map((row) => (
            <option key={row.configId} value={row.configId}>
              {shortId(row.configId)} | {row.symbol} {row.timeframe}
            </option>
          ))}
        </select>
      </div>

      {configRows.length === 0 ? (
        <div className="text-muted">No config performance available yet.</div>
      ) : (
        <>
          <div className="config-detail-grid">
            <StatLine label="Selected config" value={shortId(selected.configId)} />
            <StatLine label="Symbol/timeframe" value={`${selected.symbol} ${selected.timeframe}`} />
            <StatLine label="Closed sample" value={selected.closed} tone={sampleWarning ? 'warn' : 'neutral'} />
            <StatLine label="Wins / losses" value={`${selected.wins} / ${selected.losses}`} />
            <StatLine label="Profit factor" value={formatNumber(selected.profitFactor, 4)} tone={Number(selected.profitFactor) > 1 ? 'good' : 'warn'} />
            <StatLine label="Avg return" value={formatPct(selected.avgReturn, 4)} tone={selected.avgReturn > 0 ? 'good' : selected.avgReturn < 0 ? 'bad' : 'neutral'} />
            <StatLine label="Drawdown" value={formatPct(selected.drawdown)} tone={selected.drawdown > 10 ? 'bad' : 'neutral'} />
            <StatLine label="Direction mix" value={`LONG ${selected.longCount} / SHORT ${selected.shortCount}`} />
            <StatLine label="Avg confidence" value={formatPct(selected.avgConfidence)} />
            <StatLine label="Horizon" value={`${selected.horizon} candles / ${selected.horizonMinutes} min`} />
            <StatLine label="Risk reward" value={formatNumber(selected.riskReward, 2)} />
            <StatLine label="Last update" value={formatDate(selected.lastUpdated)} />
          </div>
          {sampleWarning && (
            <div className="sample-warning">
              This config has fewer than 30 closed shadow signals. Treat the result as early evidence, not a deployable edge.
            </div>
          )}
          <div style={{ overflowX: 'auto', marginTop: 14 }}>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Config</th>
                  <th>Symbol</th>
                  <th>Closed</th>
                  <th>Open</th>
                  <th>W/L</th>
                  <th>PF</th>
                  <th>Avg</th>
                  <th>DD</th>
                  <th>Direction</th>
                  <th>Updated</th>
                </tr>
              </thead>
              <tbody>
                {configRows.map((row) => (
                  <tr
                    key={row.configId}
                    className={row.configId === selected?.configId ? 'selected-row' : ''}
                    onClick={() => onSelectConfig(row.configId)}
                  >
                    <td className="mono">{shortId(row.configId)}</td>
                    <td className="mono">{row.symbol} {row.timeframe}</td>
                    <td>{row.closed}</td>
                    <td>{row.open}</td>
                    <td>{row.wins}/{row.losses}</td>
                    <td>{formatNumber(row.profitFactor, 4)}</td>
                    <td className={row.avgReturn >= 0 ? 'text-green' : 'text-red'}>{formatPct(row.avgReturn, 4)}</td>
                    <td>{formatPct(row.drawdown)}</td>
                    <td>LONG {row.longCount} / SHORT {row.shortCount}</td>
                    <td className="mono">{formatDate(row.lastUpdated)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </section>
  )
}

function ConfidenceBuckets({ buckets }) {
  const usefulBuckets = buckets.filter((bucket) => bucket.total > 0)
  const best = usefulBuckets.reduce((winner, bucket) => {
    if (!winner) return bucket
    return bucket.avgReturn > winner.avgReturn ? bucket : winner
  }, null)
  return (
    <section className="card">
      <div className="section-head">
        <h2 className="panel-title">Confidence vs outcome</h2>
        <span className="text-muted mono" style={{ fontSize: 11 }}>diagnostic only</span>
      </div>
      {usefulBuckets.length === 0 ? (
        <div className="text-muted">No confidence data available yet.</div>
      ) : (
        <>
          <div className="bucket-grid">
            {buckets.map((bucket) => {
              const isBest = best && bucket.label === best.label && bucket.closed > 0
              return (
                <div className={`bucket-card ${isBest ? 'bucket-best' : ''}`} key={bucket.label}>
                  <div className="mono bucket-label">{bucket.label}</div>
                  <StatLine label="Closed" value={bucket.closed} />
                  <StatLine label="Win rate" value={formatPct(bucket.winRate)} />
                  <StatLine label="Avg return" value={formatPct(bucket.avgReturn, 4)} tone={bucket.avgReturn > 0 ? 'good' : bucket.avgReturn < 0 ? 'bad' : 'neutral'} />
                  <StatLine label="PF" value={formatNumber(bucket.profitFactor, 4)} tone={Number(bucket.profitFactor) > 1 ? 'good' : 'warn'} />
                </div>
              )
            })}
          </div>
          <div className="sample-warning">
            Confidence is useful only if higher buckets keep improving after many closed signals. Do not raise or lower thresholds from this view alone.
          </div>
        </>
      )}
    </section>
  )
}

function SignalsTable({ signals }) {
  return (
    <section className="card">
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12, marginBottom: 12 }}>
        <h2 className="mono" style={{ fontSize: 15 }}>Recent Shadow Signals</h2>
        <span className="text-muted mono" style={{ fontSize: 11 }}>{signals.length} rows</span>
      </div>
      {signals.length === 0 ? (
        <div className="text-muted">No shadow signals found.</div>
      ) : (
        <div style={{ overflowX: 'auto' }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>Generated</th>
                <th>Symbol</th>
                <th>Side</th>
                <th>Status</th>
                <th>Outcome</th>
                <th>Entry</th>
                <th>SL</th>
                <th>TP</th>
                <th>PnL</th>
                <th>Expires</th>
                <th>Confidence</th>
                <th>Horizon</th>
                <th>Review</th>
                <th>Config</th>
              </tr>
            </thead>
            <tbody>
              {signals.map((signal) => {
                const review = signal.agent_review || {}
                return (
                  <tr key={signal.shadow_signal_id}>
                    <td className="mono">{formatDate(signal.generated_at)}</td>
                    <td className="mono">{signal.symbol}</td>
                    <td><span className={`badge ${signal.side === 'LONG' ? 'badge-buy' : 'badge-sell'}`}>{signal.side || '-'}</span></td>
                    <td><span className={`badge ${statusClass(signal.status)}`}>{signal.status}</span></td>
                    <td className={outcomeClass(signal.outcome)}>{signal.outcome || '-'}</td>
                    <td>{formatNumber(signal.entry_price, 6)}</td>
                    <td>{formatNumber(signal.stop_loss, 6)}</td>
                    <td>{formatNumber(signal.take_profit, 6)}</td>
                    <td className={Number(signal.pnl_pct) >= 0 ? 'text-green' : 'text-red'}>{formatPct(signal.pnl_pct, 4)}</td>
                    <td className="mono">{formatDate(signal.expires_at)}</td>
                    <td>{formatPct(signal.confidence)}</td>
                    <td>{signal.horizon_candles || '-'}c / {signal.horizon_minutes || '-'}m</td>
                    <td>{review.review_status || '-'}</td>
                    <td className="mono">{shortId(signal.config_id)}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </section>
  )
}

export default function ShadowSignalsPage() {
  const [state, setState] = useState({ loading: true, error: '', health: null, summary: null, signals: [] })
  const [selectedConfig, setSelectedConfig] = useState('')

  async function load() {
    setState((current) => ({ ...current, loading: true, error: '' }))
    try {
      const [health, summary, signals] = await Promise.all([
        api.shadow.health(),
        api.shadow.summary(),
        api.shadow.signals({ limit: 50 }),
      ])
      setState({
        loading: false,
        error: '',
        health,
        summary: summary.data,
        signals: signals.data || [],
      })
    } catch (err) {
      setState({ loading: false, error: err.message, health: null, summary: null, signals: [] })
    }
  }

  useEffect(() => {
    load()
  }, [])

  const summary = state.summary?.summary || state.health?.summary || {}
  const pfTone = Number(summary.profit_factor) > 1 ? 'good' : Number(summary.profit_factor) > 0 ? 'warn' : 'neutral'
  const avgTone = Number(summary.avg_return) > 0 ? 'good' : Number(summary.avg_return) < 0 ? 'bad' : 'neutral'
  const source = state.summary ? 'backend shadow API' : 'loading'

  const sortedSignals = useMemo(() => {
    return [...state.signals].sort((a, b) => {
      if (a.status === 'OPEN' && b.status !== 'OPEN') return -1
      if (a.status !== 'OPEN' && b.status === 'OPEN') return 1
      return String(b.updated_at || b.recorded_at || b.generated_at || '').localeCompare(String(a.updated_at || a.recorded_at || a.generated_at || ''))
    })
  }, [state.signals])
  const latestOpen = useMemo(() => sortedSignals.filter((item) => item.status === 'OPEN'), [sortedSignals])
  const latestSignal = sortedSignals[0] || null
  const configRows = useMemo(() => buildConfigRows(sortedSignals, state.summary?.by_config || {}), [sortedSignals, state.summary])
  const confidenceBuckets = useMemo(() => buildConfidenceBuckets(sortedSignals), [sortedSignals])

  useEffect(() => {
    if (configRows.length === 0) {
      setSelectedConfig('')
      return
    }
    if (!configRows.some((row) => row.configId === selectedConfig)) {
      setSelectedConfig(configRows[0].configId)
    }
  }, [configRows, selectedConfig])

  return (
    <>
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 16, alignItems: 'flex-start', marginBottom: 16 }}>
        <div>
          <h1 className="mono" style={{ fontSize: 24, color: 'var(--text)', marginBottom: 6 }}>Shadow Signals</h1>
          <p className="text-muted" style={{ maxWidth: 780 }}>
            Paper/shadow monitoring backed by measured outcomes. This dashboard is read-only and does not place orders.
          </p>
        </div>
        <button className="btn btn-secondary" type="button" onClick={load} disabled={state.loading}>
          <RefreshCw size={14} />
          Refresh
        </button>
      </div>

      {state.error && (
        <section className="card" style={{ borderColor: 'rgba(255,77,106,0.35)', marginBottom: 16 }}>
          <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
            <AlertTriangle size={18} color="var(--red)" />
            <span className="text-red">Backend unavailable: {state.error}</span>
          </div>
        </section>
      )}

      {state.loading && (
        <section className="card" style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
          <span className="spinner" />
          <span className="text-muted">Loading shadow metrics from {source}...</span>
        </section>
      )}

      {!state.loading && !state.error && (
        <>
          <GuardrailStrip health={state.health} />
          <div className="kpi-grid">
            <Kpi label="Open" value={summary.open ?? 0} tone={latestOpen.length > 0 ? 'warn' : 'neutral'} />
            <Kpi label="Closed" value={summary.closed ?? 0} />
            <Kpi label="Win Rate" value={formatPct(summary.win_rate)} />
            <Kpi label="Profit Factor" value={formatNumber(summary.profit_factor, 4)} tone={pfTone} />
            <Kpi label="Avg Return" value={formatPct(summary.avg_return, 4)} tone={avgTone} />
            <Kpi label="Max Drawdown" value={formatPct(summary.max_drawdown)} tone={Number(summary.max_drawdown) > 10 ? 'bad' : 'neutral'} />
          </div>

          <div className="ops-grid">
            <ActiveSignalPanel signal={latestOpen[0]} />
            <FreshnessPanel health={state.health} summary={state.summary} signals={sortedSignals} />
            <SignalContextPanel signal={latestSignal} />
          </div>

          <div className="shadow-grid">
            <SymbolSummary bySymbol={state.summary?.by_symbol || {}} />
            <ConfigSummary byConfig={state.summary?.by_config || {}} />
          </div>

          <ConfigDeepDive
            configRows={configRows}
            selectedConfig={selectedConfig}
            onSelectConfig={setSelectedConfig}
          />

          <ConfidenceBuckets buckets={confidenceBuckets} />

          <div className="shadow-grid">
            <section className="card">
              <h2 className="mono" style={{ fontSize: 15, marginBottom: 12 }}>Integrity</h2>
              <div style={{ display: 'grid', gap: 10 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <Activity size={16} color="var(--green)" />
                  <span>No real trading controls exposed.</span>
                </div>
                <div className="text-muted">Current source: {state.summary?.source || state.health?.source || '-'}</div>
                <div className="text-muted">Signals are watchlist/research outputs until stable candidates exist.</div>
              </div>
            </section>
            <section className="card">
              <h2 className="mono" style={{ fontSize: 15, marginBottom: 12 }}>Selection state</h2>
              <div className="stack">
                <StatLine label="Stable candidates" value="0 confirmed" tone="warn" />
                <StatLine label="Current mode" value="watchlist shadow" />
                <StatLine label="Signal limit" value="max 1 open signal" />
                <StatLine label="Dashboard mode" value="read-only" />
              </div>
            </section>
          </div>

          <SignalsTable signals={sortedSignals} />
        </>
      )}
    </>
  )
}
