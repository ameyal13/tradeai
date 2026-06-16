import { useEffect, useMemo, useState } from 'react'
import { AlertTriangle, BarChart3, RefreshCw, ShieldCheck } from 'lucide-react'
import { api } from '../lib/api.js'

function formatNumber(value, digits = 4) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '-'
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: digits })
}

function formatPct(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '-'
  return `${formatNumber(value, digits)}%`
}

function classificationClass(value) {
  if (value === 'stable_research_candidate') return 'badge-buy'
  if (value === 'unstable_watchlist') return 'badge-medium'
  if (value === 'multi_window_reject') return 'badge-high'
  return 'badge-low'
}

function Kpi({ label, value, tone = 'neutral' }) {
  const color = tone === 'good' ? 'var(--green)' : tone === 'bad' ? 'var(--red)' : tone === 'warn' ? 'var(--amber)' : 'var(--text)'
  return (
    <section className="card">
      <div className="text-muted mono" style={{ fontSize: 11, marginBottom: 8 }}>{label}</div>
      <div className="mono" style={{ fontSize: 23, color }}>{value}</div>
    </section>
  )
}

function TopTable({ title, rows, diagnostic = false }) {
  return (
    <section className="card">
      <div className="section-head">
        <h2 className="panel-title">{title}</h2>
        {diagnostic && <span className="text-amber mono" style={{ fontSize: 11 }}>diagnostic only</span>}
      </div>
      {!rows?.length ? (
        <div className="text-muted">No rows available.</div>
      ) : (
        <div style={{ overflowX: 'auto' }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>Config</th>
                <th>Class</th>
                <th>Val PF</th>
                <th>Val Avg</th>
                <th>Val+</th>
                <th>Beats Random</th>
                <th>Beats Det</th>
                <th>Test PF</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.config_id}>
                  <td className="mono">{row.label}</td>
                  <td><span className={`badge ${classificationClass(row.classification)}`}>{row.classification}</span></td>
                  <td>{formatNumber(row.median_validation_pf)}</td>
                  <td className={Number(row.median_validation_avg_return) >= 0 ? 'text-green' : 'text-red'}>{formatPct(row.median_validation_avg_return, 4)}</td>
                  <td>{formatPct(row.validation_positive_rate)}</td>
                  <td>{formatPct(row.beats_random_rate)}</td>
                  <td>{formatPct(row.beats_deterministic_rate)}</td>
                  <td>{formatNumber(row.median_test_pf)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  )
}

function AssetDiagnostics({ rows }) {
  return (
    <section className="card">
      <h2 className="panel-title">Asset diagnostics</h2>
      {!rows?.length ? (
        <div className="text-muted">No asset diagnostics available.</div>
      ) : (
        <div style={{ overflowX: 'auto' }}>
          <table className="data-table compact-table">
            <thead>
              <tr>
                <th>Asset</th>
                <th>Configs</th>
                <th>Watchlist</th>
                <th>Reject</th>
                <th>Stable</th>
                <th>Median PF</th>
                <th>Median Avg</th>
                <th>Promising</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.symbol}>
                  <td className="mono">{row.symbol}</td>
                  <td>{row.count}</td>
                  <td>{row.unstable_watchlist}</td>
                  <td>{row.multi_window_reject}</td>
                  <td>{row.stable_research_candidate}</td>
                  <td>{formatNumber(row.median_validation_pf)}</td>
                  <td className={Number(row.median_validation_avg_return) >= 0 ? 'text-green' : 'text-red'}>{formatPct(row.median_validation_avg_return, 4)}</td>
                  <td>{row.promising ? 'yes' : 'no'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  )
}

export default function ResearchPage() {
  const [state, setState] = useState({ loading: true, error: '', data: null })
  const [source, setSource] = useState('crypto_multi')

  async function load() {
    setState((current) => ({ ...current, loading: true, error: '' }))
    try {
      const res = await api.research.summary({ source })
      setState({ loading: false, error: '', data: res.data })
    } catch (err) {
      setState({ loading: false, error: err.message, data: null })
    }
  }

  useEffect(() => {
    load()
  }, [source])

  const summary = state.data?.summary || {}
  const counts = summary.classification_counts || {}
  const assets = state.data?.asset_diagnostics?.assets_by_watchlist_count || []
  const conclusions = state.data?.conclusion || []
  const dataSource = state.data ? 'backend research API' : 'loading'
  const top = state.data?.top || {}
  const stable = Number(summary.stable_research_candidate || 0)
  const watchlist = Number(summary.unstable_watchlist || 0)

  const conclusionTone = useMemo(() => {
    if (stable > 0) return 'good'
    if (watchlist > 0) return 'warn'
    return 'bad'
  }, [stable, watchlist])

  return (
    <>
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 16, alignItems: 'flex-start', marginBottom: 16 }}>
        <div>
          <h1 className="mono" style={{ fontSize: 24, color: 'var(--text)', marginBottom: 6 }}>Research</h1>
          <p className="text-muted" style={{ maxWidth: 820 }}>
            Multi-window research telemetry. Validation selects; test metrics are diagnostic only. Research only, no trading signal.
          </p>
        </div>
        <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
          <select value={source} onChange={(event) => setSource(event.target.value)}>
            <option value="crypto_multi">crypto_multi</option>
            <option value="focused_v2a">focused_v2a</option>
          </select>
          <button className="btn btn-secondary" type="button" onClick={load} disabled={state.loading}>
            <RefreshCw size={14} />
            Refresh
          </button>
        </div>
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
          <span className="text-muted">Loading research metrics from {dataSource}...</span>
        </section>
      )}

      {!state.loading && !state.error && (
        <>
          <section className="card" style={{ marginBottom: 16, borderColor: 'rgba(0,212,160,0.22)', background: 'rgba(0,212,160,0.05)' }}>
            <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
              <ShieldCheck size={18} color="var(--green)" />
              <div>
                <div className="mono" style={{ color: 'var(--green)', fontSize: 13 }}>Research only. No trading signal.</div>
                <div className="text-muted" style={{ fontSize: 12 }}>No test metric is used to select candidates. Accuracy is not used.</div>
              </div>
            </div>
          </section>

          <div className="kpi-grid">
            <Kpi label="Total Configs" value={summary.total_configs ?? 0} />
            <Kpi label="Completed" value={summary.completed ?? 0} />
            <Kpi label="Stable" value={stable} tone={stable > 0 ? 'good' : 'neutral'} />
            <Kpi label="Watchlist" value={watchlist} tone={watchlist > 0 ? 'warn' : 'neutral'} />
            <Kpi label="Reject" value={counts.multi_window_reject ?? 0} tone="bad" />
            <Kpi label="JSON Loaded" value={summary.json_loaded ?? 0} />
          </div>

          <section className="card" style={{ marginBottom: 16 }}>
            <h2 className="panel-title"><BarChart3 size={15} /> Automatic conclusion</h2>
            <div className="stack">
              {conclusions.map((item) => (
                <div className={conclusionTone === 'good' ? 'text-green' : conclusionTone === 'warn' ? 'text-amber' : 'text-muted'} key={item}>
                  {item}
                </div>
              ))}
            </div>
          </section>

          <AssetDiagnostics rows={assets} />

          <div className="research-grid">
            <TopTable title="Top validation PF" rows={top.median_validation_pf || []} />
            <TopTable title="Top validation avg return" rows={top.median_validation_avg_return || []} />
          </div>

          <TopTable title="Top test PF" rows={top.median_test_pf_diagnostic_only || []} diagnostic />
        </>
      )}
    </>
  )
}
