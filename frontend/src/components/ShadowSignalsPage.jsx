import { useEffect, useMemo, useState } from 'react'
import { Activity, AlertTriangle, RefreshCw, ShieldCheck } from 'lucide-react'
import { api } from '../lib/api.js'

function formatNumber(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '-'
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: digits })
}

function formatPct(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '-'
  return `${formatNumber(value, digits)}%`
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
      <div className="mono text-muted" style={{ fontSize: 11 }}>{health?.timestamp ? new Date(health.timestamp).toLocaleString() : '-'}</div>
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
                <th>Review</th>
                <th>Config</th>
              </tr>
            </thead>
            <tbody>
              {signals.map((signal) => {
                const review = signal.agent_review || {}
                return (
                  <tr key={signal.shadow_signal_id}>
                    <td className="mono">{signal.generated_at ? new Date(signal.generated_at).toLocaleString() : '-'}</td>
                    <td className="mono">{signal.symbol}</td>
                    <td><span className={`badge ${signal.side === 'LONG' ? 'badge-buy' : 'badge-sell'}`}>{signal.side || '-'}</span></td>
                    <td><span className={`badge ${statusClass(signal.status)}`}>{signal.status}</span></td>
                    <td className={outcomeClass(signal.outcome)}>{signal.outcome || '-'}</td>
                    <td>{formatNumber(signal.entry_price, 6)}</td>
                    <td>{formatNumber(signal.stop_loss, 6)}</td>
                    <td>{formatNumber(signal.take_profit, 6)}</td>
                    <td className={Number(signal.pnl_pct) >= 0 ? 'text-green' : 'text-red'}>{formatPct(signal.pnl_pct, 4)}</td>
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

  const latestOpen = useMemo(() => state.signals.filter((item) => item.status === 'OPEN'), [state.signals])

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

          <div className="shadow-grid">
            <SymbolSummary bySymbol={state.summary?.by_symbol || {}} />
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
          </div>

          <SignalsTable signals={state.signals} />
        </>
      )}
    </>
  )
}
