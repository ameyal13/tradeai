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
