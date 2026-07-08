import { Fragment, useEffect, useMemo, useRef, useState } from 'react'
import { createChart, LineStyle } from 'lightweight-charts'
import { Activity, AlertTriangle, Clock, Database, RefreshCw, ShieldCheck, Target, TrendingUp, X } from 'lucide-react'
import { api } from '../lib/api.js'

function formatNumber(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '-'
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: digits })
}

function formatPct(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '-'
  return `${formatNumber(value, digits)}%`
}

function formatProbability(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '-'
  const numeric = Number(value)
  const asPercent = Math.abs(numeric) <= 1 ? numeric * 100 : numeric
  return `${formatNumber(asPercent, digits)}%`
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

function toFiniteNumber(value) {
  const numeric = Number(value)
  return Number.isFinite(numeric) ? numeric : null
}

function getSignalCloseTime(signal) {
  return signal?.closed_at
    || signal?.evaluated_at
    || signal?.updated_at
    || signal?.recorded_at
    || signal?.expires_at
    || signal?.generated_at
}

function isFinishedSignal(signal) {
  return Boolean(signal && signal.status !== 'OPEN' && (signal.outcome || signal.status === 'CLOSED' || signal.status === 'EXPIRED'))
}

function hoursBetween(start, end) {
  if (!start || !end) return null
  const startDate = new Date(start)
  const endDate = new Date(end)
  if (Number.isNaN(startDate.getTime()) || Number.isNaN(endDate.getTime())) return null
  return (endDate.getTime() - startDate.getTime()) / 3600000
}

function normalizeChartPoints(points) {
  let lastTime = 0
  return points
    .filter((point) => point && Number.isFinite(point.time) && Number.isFinite(point.value))
    .sort((a, b) => a.time - b.time)
    .map((point) => {
      const nextTime = point.time <= lastTime ? lastTime + 1 : point.time
      lastTime = nextTime
      return { ...point, time: nextTime }
    })
}

function buildEquityCurve(signals) {
  let cumulative = 0
  const points = []
  const sorted = [...signals]
    .filter((signal) => isFinishedSignal(signal) && toFiniteNumber(signal.pnl_pct) !== null && getSignalCloseTime(signal))
    .sort((a, b) => new Date(getSignalCloseTime(a)).getTime() - new Date(getSignalCloseTime(b)).getTime())

  for (const signal of sorted) {
    cumulative += Number(signal.pnl_pct)
    points.push({
      time: Math.floor(new Date(getSignalCloseTime(signal)).getTime() / 1000),
      value: Number(cumulative.toFixed(6)),
      signal,
    })
  }
  return normalizeChartPoints(points)
}

function calculateProfitFactor(returns) {
  const gains = returns.filter((value) => value > 0).reduce((acc, value) => acc + value, 0)
  const losses = Math.abs(returns.filter((value) => value < 0).reduce((acc, value) => acc + value, 0))
  if (losses > 0) return gains / losses
  if (gains > 0) return 5
  return 0
}

function buildRollingProfitFactor(signals, windowSize = 10) {
  const sorted = [...signals]
    .filter((signal) => isFinishedSignal(signal) && toFiniteNumber(signal.pnl_pct) !== null && getSignalCloseTime(signal))
    .sort((a, b) => new Date(getSignalCloseTime(a)).getTime() - new Date(getSignalCloseTime(b)).getTime())

  const points = sorted.map((signal, index) => {
    const window = sorted.slice(Math.max(0, index - windowSize + 1), index + 1)
    const returns = window.map((item) => Number(item.pnl_pct))
    return {
      time: Math.floor(new Date(getSignalCloseTime(signal)).getTime() / 1000),
      value: Number(calculateProfitFactor(returns).toFixed(6)),
      signal,
    }
  })
  return normalizeChartPoints(points)
}

function compactEntries(value, max = 6) {
  if (!value || typeof value !== 'object') return []
  return Object.entries(value)
    .filter(([, item]) => item !== null && item !== undefined && ['string', 'number', 'boolean'].includes(typeof item))
    .slice(0, max)
}

function findDeepNumber(source, names) {
  if (!source || typeof source !== 'object') return 0
  const wanted = new Set(names.map((name) => String(name).toLowerCase()))
  const stack = [source]
  const seen = new Set()
  while (stack.length) {
    const item = stack.pop()
    if (!item || typeof item !== 'object' || seen.has(item)) continue
    seen.add(item)
    for (const [key, value] of Object.entries(item)) {
      if (wanted.has(String(key).toLowerCase())) {
        const numeric = toFiniteNumber(value)
        if (numeric !== null) return numeric
      }
      if (value && typeof value === 'object') stack.push(value)
    }
  }
  return 0
}

function getCycleIssues(cycle) {
  if (!cycle) return null
  const skippedNoPrice = findDeepNumber(cycle, ['skipped_no_price'])
  const evaluationErrors = findDeepNumber(cycle, ['evaluation_errors', 'evaluation_error_count'])
  const skippedErrors = findDeepNumber(cycle, ['skipped_errors'])
  if (skippedNoPrice <= 0 && evaluationErrors <= 0 && skippedErrors <= 0) return null
  return {
    skippedNoPrice,
    evaluationErrors,
    skippedErrors,
    finishedAt: cycle.finished_at || cycle.updated_at || cycle.created_at,
  }
}

function signalKey(signal) {
  return signal?.shadow_signal_id || `${signal?.config_id || 'unknown'}-${signal?.generated_at || ''}`
}

function getCandles(marketData) {
  return Array.isArray(marketData?.candles) ? marketData.candles : []
}

function estimateMarketLevels(candles) {
  const recent = candles.slice(-24)
  if (!recent.length) return { support: null, resistance: null, rangePct: null, volumeRatio: null }
  const lows = recent.map((candle) => toFiniteNumber(candle.low)).filter((value) => value !== null)
  const highs = recent.map((candle) => toFiniteNumber(candle.high)).filter((value) => value !== null)
  const closes = recent.map((candle) => toFiniteNumber(candle.close)).filter((value) => value !== null)
  const volumes = recent.map((candle) => toFiniteNumber(candle.volume)).filter((value) => value !== null)
  const lastClose = closes[closes.length - 1] || null
  const support = lows.length ? Math.min(...lows) : null
  const resistance = highs.length ? Math.max(...highs) : null
  const rangePct = support && resistance && lastClose ? ((resistance - support) / lastClose) * 100 : null
  const avgVolume = volumes.length ? volumes.reduce((acc, value) => acc + value, 0) / volumes.length : null
  const volumeRatio = avgVolume && volumes.length ? volumes[volumes.length - 1] / avgVolume : null
  return { support, resistance, rangePct, volumeRatio }
}

function intervalTone(marketData) {
  const analysis = marketData?.analysis || {}
  const indicators = marketData?.indicators || {}
  const macd = analysis.macd_signal || (toNumber(indicators.macd_hist) >= 0 ? 'BULLISH' : 'BEARISH')
  const rsi = toFiniteNumber(indicators.rsi)
  if (macd === 'BULLISH' && (rsi === null || rsi < 72)) return 'bullish'
  if (macd === 'BEARISH' && (rsi === null || rsi > 28)) return 'bearish'
  return 'neutral'
}

function alignmentForSide(side, tone) {
  if (side === 'LONG') return tone === 'bullish'
  if (side === 'SHORT') return tone === 'bearish'
  return false
}

function buildDecisionRead(signal, marketByInterval) {
  if (!signal) return null
  const intervals = ['15m', '1h', '4h']
  const tones = intervals.map((interval) => ({
    interval,
    tone: intervalTone(marketByInterval[interval]),
    aligned: alignmentForSide(signal.side, intervalTone(marketByInterval[interval])),
  }))
  const loaded = tones.filter((item) => marketByInterval[item.interval])
  const alignedCount = loaded.filter((item) => item.aligned).length
  const bearishCount = loaded.filter((item) => item.tone === 'bearish').length
  const bullishCount = loaded.filter((item) => item.tone === 'bullish').length
  const direction = signal.side === 'LONG' ? 'compradora' : signal.side === 'SHORT' ? 'vendedora' : 'neutral'
  let label = 'Sin suficiente contexto'
  let tone = 'warn'
  if (loaded.length >= 2 && alignedCount >= 2) {
    label = `Contexto multi-timeframe alineado con la idea ${direction}`
    tone = 'good'
  } else if (loaded.length >= 2 && alignedCount === 0) {
    label = `Contexto multi-timeframe contradice la idea ${direction}`
    tone = 'bad'
  } else if (loaded.length) {
    label = `Contexto mixto: ${bullishCount} bullish / ${bearishCount} bearish`
    tone = 'warn'
  }
  return { label, tone, tones, alignedCount, loadedCount: loaded.length }
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

function MetricLineChart({ points, baseline = null, height = 220, segmented = false, positiveAboveZero = false }) {
  const containerRef = useRef(null)

  useEffect(() => {
    if (!containerRef.current) return undefined
    const chart = createChart(containerRef.current, {
      height,
      layout: {
        background: { color: 'transparent' },
        textColor: '#5a6a7a',
        fontFamily: 'IBM Plex Mono',
      },
      grid: {
        vertLines: { color: 'rgba(42,51,64,0.35)' },
        horzLines: { color: 'rgba(42,51,64,0.35)' },
      },
      rightPriceScale: {
        borderColor: 'rgba(42,51,64,0.65)',
      },
      timeScale: {
        borderColor: 'rgba(42,51,64,0.65)',
        timeVisible: true,
        secondsVisible: false,
      },
      crosshair: {
        vertLine: { color: 'rgba(90,106,122,0.35)' },
        horzLine: { color: 'rgba(90,106,122,0.35)' },
      },
    })

    if (points.length) {
      if (baseline !== null) {
        const baselineSeries = chart.addLineSeries({
          color: 'rgba(232,237,243,0.42)',
          lineWidth: 1,
          lineStyle: LineStyle.Dashed,
          priceLineVisible: false,
          lastValueVisible: false,
        })
        baselineSeries.setData([
          { time: points[0].time, value: baseline },
          { time: points[points.length - 1].time, value: baseline },
        ])
      }

      if (segmented && points.length > 1) {
        for (let index = 1; index < points.length; index += 1) {
          const previous = points[index - 1]
          const current = points[index]
          const color = current.value >= previous.value ? '#00d4a0' : '#ff4d6a'
          const series = chart.addLineSeries({
            color,
            lineWidth: 2,
            priceLineVisible: false,
            lastValueVisible: false,
          })
          series.setData([previous, current])
        }
      } else {
        const last = points[points.length - 1]
        const color = positiveAboveZero ? (last.value >= baseline ? '#00d4a0' : '#ff4d6a') : '#4090ff'
        const series = chart.addLineSeries({
          color,
          lineWidth: 2,
          priceLineVisible: false,
        })
        series.setData(points)
      }
      chart.timeScale().fitContent()
    }

    const resize = () => {
      if (containerRef.current) {
        chart.applyOptions({ width: containerRef.current.clientWidth })
      }
    }
    resize()
    window.addEventListener('resize', resize)
    return () => {
      window.removeEventListener('resize', resize)
      chart.remove()
    }
  }, [points, baseline, height, segmented, positiveAboveZero])

  if (!points.length) {
    return (
      <div className="chart-empty" style={{ minHeight: height }}>
        <span className="text-muted">No closed signal data yet.</span>
      </div>
    )
  }

  return <div className="metric-chart" ref={containerRef} style={{ height }} />
}

function PerformanceCharts({ signals }) {
  const equityPoints = useMemo(() => buildEquityCurve(signals), [signals])
  const rollingPfPoints = useMemo(() => buildRollingProfitFactor(signals, 10), [signals])
  return (
    <section className="card chart-section">
      <div className="section-head">
        <h2 className="panel-title">Equity curve</h2>
        <span className="text-muted mono" style={{ fontSize: 11 }}>{equityPoints.length} closed points</span>
      </div>
      <MetricLineChart points={equityPoints} baseline={0} segmented height={240} />

      <div className="section-head chart-subhead">
        <h2 className="panel-title">Rolling profit factor</h2>
        <span className="text-muted mono" style={{ fontSize: 11 }}>last 10 closed signals</span>
      </div>
      <MetricLineChart points={rollingPfPoints} baseline={1} height={180} positiveAboveZero />
    </section>
  )
}

function StrategyEvidenceStrip({ strategySummary, technicalExclusions, exclusionsByReason }) {
  const exclusions = Number(technicalExclusions || 0)
  const reasons = Object.entries(exclusionsByReason || {})
  return (
    <section className="strategy-evidence-strip">
      <div>
        <div className="mono strategy-evidence-title">Strategy-evaluable outcomes</div>
        <div className="text-muted" style={{ fontSize: 12 }}>
          Excludes technical expirations such as old evaluation HTTP errors. This is the cleaner signal-quality metric.
        </div>
      </div>
      <div className="strategy-evidence-metrics">
        <StatLine label="Valid closed" value={strategySummary?.closed ?? 0} />
        <StatLine label="W/L" value={`${strategySummary?.wins ?? 0}/${strategySummary?.losses ?? 0}`} />
        <StatLine label="Strategy PF" value={formatNumber(strategySummary?.profit_factor, 4)} tone={Number(strategySummary?.profit_factor) > 1 ? 'good' : 'warn'} />
        <StatLine label="Avg return" value={formatPct(strategySummary?.avg_return, 4)} tone={Number(strategySummary?.avg_return) > 0 ? 'good' : 'bad'} />
        <StatLine label="Technical exclusions" value={exclusions} tone={exclusions > 0 ? 'warn' : 'neutral'} />
      </div>
      {reasons.length > 0 && (
        <div className="tag-row">
          {reasons.map(([reason, count]) => <span className="tag" key={reason}>{reason}: {String(count)}</span>)}
        </div>
      )}
    </section>
  )
}

function recommendationTone(recommendation) {
  if (recommendation === 'keep_candidate') return 'good'
  if (recommendation === 'quarantine_candidate') return 'bad'
  if (recommendation === 'watch') return 'warn'
  return 'neutral'
}

function recommendationClass(recommendation) {
  if (recommendation === 'keep_candidate') return 'badge-buy'
  if (recommendation === 'quarantine_candidate') return 'badge-high'
  if (recommendation === 'watch') return 'badge-medium'
  return 'badge-low'
}

function ConfigHealthPanel({ report, onOpenConfig }) {
  const summary = report?.summary || {}
  const configs = report?.configs || []
  const counts = summary.recommendation_counts || {}
  const focusedConfigs = configs.slice(0, 12)
  return (
    <section className="card config-health-panel">
      <div className="section-head">
        <div>
          <h2 className="panel-title">Config health</h2>
          <p className="text-muted" style={{ fontSize: 13 }}>
            Read-only recommendations from live shadow outcomes. This does not pause configs automatically.
          </p>
        </div>
        <div className="tag-row">
          {Object.entries(counts).length ? (
            Object.entries(counts).map(([key, value]) => <span className="tag" key={key}>{key}: {String(value)}</span>)
          ) : (
            <span className="tag">no config evidence</span>
          )}
        </div>
      </div>
      {focusedConfigs.length === 0 ? (
        <div className="text-muted">No config health data yet.</div>
      ) : (
        <>
          <div className="config-health-grid">
            <StatLine label="Configs tracked" value={summary.total_configs ?? 0} />
            <StatLine label="Quarantine candidates" value={counts.quarantine_candidate ?? 0} tone={(counts.quarantine_candidate || 0) > 0 ? 'bad' : 'neutral'} />
            <StatLine label="Keep candidates" value={counts.keep_candidate ?? 0} tone={(counts.keep_candidate || 0) > 0 ? 'good' : 'neutral'} />
            <StatLine label="Insufficient sample" value={counts.insufficient_sample ?? 0} />
          </div>
          <div style={{ overflowX: 'auto', marginTop: 14 }}>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Recommendation</th>
                  <th>Config</th>
                  <th>Symbol</th>
                  <th>Valid</th>
                  <th>W/L</th>
                  <th>PF</th>
                  <th>Avg</th>
                  <th>DD</th>
                  <th>Direction</th>
                  <th>Reasons</th>
                </tr>
              </thead>
              <tbody>
                {focusedConfigs.map((row) => (
                  <tr key={row.config_id}>
                    <td><span className={`badge ${recommendationClass(row.recommendation)}`}>{row.recommendation}</span></td>
                    <td>
                      <button className="table-link mono" type="button" onClick={() => onOpenConfig(row.config_id)}>
                        {shortId(row.config_id)}
                      </button>
                    </td>
                    <td className="mono">{row.symbol} {row.timeframe}</td>
                    <td>{row.strategy_closed}</td>
                    <td>{row.wins}/{row.losses}</td>
                    <td>{formatNumber(row.profit_factor, 4)}</td>
                    <td className={Number(row.avg_return) >= 0 ? 'text-green' : 'text-red'}>{formatPct(row.avg_return, 4)}</td>
                    <td>{formatPct(row.max_drawdown)}</td>
                    <td>{row.direction_bias}</td>
                    <td>
                      <div className="tag-row">
                        {(row.reasons || []).slice(0, 3).map((reason) => <span className="tag" key={`${row.config_id}-${reason}`}>{reason}</span>)}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="sample-warning">
            Quarantine candidates are evidence flags only. The shadow generator still uses the current approved registry until a separate, tested gating change is made.
          </div>
        </>
      )}
    </section>
  )
}

function SignalPriceChart({ marketData, signal, interval, height = 360 }) {
  const containerRef = useRef(null)
  const candles = useMemo(() => getCandles(marketData), [marketData])

  useEffect(() => {
    if (!containerRef.current) return undefined
    const chart = createChart(containerRef.current, {
      height,
      layout: {
        background: { color: 'transparent' },
        textColor: '#5a6a7a',
        fontFamily: 'IBM Plex Mono',
      },
      grid: {
        vertLines: { color: 'rgba(42,51,64,0.35)' },
        horzLines: { color: 'rgba(42,51,64,0.35)' },
      },
      rightPriceScale: {
        borderColor: 'rgba(42,51,64,0.65)',
      },
      timeScale: {
        borderColor: 'rgba(42,51,64,0.65)',
        timeVisible: true,
        secondsVisible: false,
      },
      crosshair: {
        vertLine: { color: 'rgba(90,106,122,0.35)' },
        horzLine: { color: 'rgba(90,106,122,0.35)' },
      },
    })

    if (candles.length) {
      const series = chart.addCandlestickSeries({
        upColor: '#00d4a0',
        downColor: '#ff4d6a',
        borderUpColor: '#00d4a0',
        borderDownColor: '#ff4d6a',
        wickUpColor: '#00d4a0',
        wickDownColor: '#ff4d6a',
      })
      series.setData(candles.map((candle) => ({
        time: candle.time,
        open: Number(candle.open),
        high: Number(candle.high),
        low: Number(candle.low),
        close: Number(candle.close),
      })))

      const levels = [
        { label: 'ENTRY', value: toFiniteNumber(signal?.entry_price), color: '#4090ff', style: LineStyle.Solid },
        { label: 'SL', value: toFiniteNumber(signal?.stop_loss), color: '#ff4d6a', style: LineStyle.Dashed },
        { label: 'TP', value: toFiniteNumber(signal?.take_profit), color: '#00d4a0', style: LineStyle.Dashed },
      ]
      for (const level of levels) {
        if (level.value !== null) {
          series.createPriceLine({
            price: level.value,
            color: level.color,
            lineWidth: 1,
            lineStyle: level.style,
            axisLabelVisible: true,
            title: level.label,
          })
        }
      }

      const generatedAt = signal?.generated_at ? Math.floor(new Date(signal.generated_at).getTime() / 1000) : null
      if (generatedAt) {
        const nearest = candles.reduce((winner, candle) => {
          if (!winner) return candle
          return Math.abs(candle.time - generatedAt) < Math.abs(winner.time - generatedAt) ? candle : winner
        }, null)
        if (nearest) {
          series.setMarkers([{
            time: nearest.time,
            position: signal?.side === 'SHORT' ? 'aboveBar' : 'belowBar',
            color: signal?.side === 'SHORT' ? '#ff4d6a' : '#00d4a0',
            shape: signal?.side === 'SHORT' ? 'arrowDown' : 'arrowUp',
            text: `${signal?.side || 'SIGNAL'} ${shortId(signal?.config_id)}`,
          }])
        }
      }

      chart.timeScale().fitContent()
    }

    const resize = () => {
      if (containerRef.current) {
        chart.applyOptions({ width: containerRef.current.clientWidth })
      }
    }
    resize()
    window.addEventListener('resize', resize)
    return () => {
      window.removeEventListener('resize', resize)
      chart.remove()
    }
  }, [candles, signal, interval, height])

  if (!candles.length) {
    return (
      <div className="chart-empty" style={{ minHeight: height }}>
        <span className="text-muted">No candle data for {interval} yet.</span>
      </div>
    )
  }

  return <div className="price-chart" ref={containerRef} style={{ height }} />
}

function TimeframeContextCard({ interval, marketData, signal }) {
  const indicators = marketData?.indicators || {}
  const analysis = marketData?.analysis || {}
  const candles = getCandles(marketData)
  const levels = estimateMarketLevels(candles)
  const tone = intervalTone(marketData)
  const aligned = signal ? alignmentForSide(signal.side, tone) : false
  return (
    <section className={`timeframe-card ${aligned ? 'timeframe-aligned' : ''}`}>
      <div className="timeframe-head">
        <span className="mono">{interval}</span>
        <span className={`badge ${tone === 'bullish' ? 'badge-buy' : tone === 'bearish' ? 'badge-sell' : 'badge-hold'}`}>
          {tone}
        </span>
      </div>
      {!marketData ? (
        <div className="text-muted">Loading...</div>
      ) : (
        <div className="stack">
          <StatLine label="Price" value={formatNumber(indicators.price, 6)} />
          <StatLine label="RSI" value={formatNumber(indicators.rsi, 2)} tone={analysis.rsi_signal === 'OVERBOUGHT' ? 'warn' : analysis.rsi_signal === 'OVERSOLD' ? 'good' : 'neutral'} />
          <StatLine label="MACD" value={analysis.macd_signal || '-'} tone={analysis.macd_signal === 'BULLISH' ? 'good' : 'bad'} />
          <StatLine label="BB" value={analysis.bb_position || '-'} />
          <StatLine label="Support" value={formatNumber(levels.support, 6)} />
          <StatLine label="Resistance" value={formatNumber(levels.resistance, 6)} />
          <StatLine label="Volume ratio" value={formatNumber(levels.volumeRatio, 2)} tone={levels.volumeRatio > 1.2 ? 'good' : 'neutral'} />
        </div>
      )}
    </section>
  )
}

function SignalDecisionView({ signals, selectedSignalId, onSelectSignal }) {
  const [interval, setInterval] = useState('1h')
  const [marketState, setMarketState] = useState({ loading: false, error: '', data: {} })
  const signal = useMemo(() => {
    if (!signals.length) return null
    return signals.find((item) => signalKey(item) === selectedSignalId) || signals.find((item) => item.status === 'OPEN') || signals[0]
  }, [signals, selectedSignalId])

  useEffect(() => {
    if (signal?.timeframe) {
      setInterval(signal.timeframe)
    }
  }, [signal?.timeframe, signalKey(signal)])

  useEffect(() => {
    if (!signal?.symbol) return undefined
    let cancelled = false
    const intervals = Array.from(new Set(['15m', signal.timeframe || '1h', '1h', '4h', interval]))
    setMarketState((current) => ({ ...current, loading: true, error: '' }))
    Promise.allSettled(intervals.map((item) => api.market.detail(signal.symbol, item).then((res) => [item, res.data])))
      .then((results) => {
        if (cancelled) return
        const nextData = {}
        const errors = []
        for (const result of results) {
          if (result.status === 'fulfilled') {
            nextData[result.value[0]] = result.value[1]
          } else {
            errors.push(result.reason?.message || String(result.reason))
          }
        }
        setMarketState({ loading: false, error: errors[0] || '', data: nextData })
      })
    return () => { cancelled = true }
  }, [signal?.symbol, signal?.timeframe, signalKey(signal), interval])

  const decision = useMemo(() => buildDecisionRead(signal, marketState.data), [signal, marketState.data])
  const selectedMarket = marketState.data[interval] || marketState.data[signal?.timeframe || '1h'] || null
  const selectedLevels = estimateMarketLevels(getCandles(selectedMarket))
  const features = signal?.input_features || signal?.raw?.input_features || {}
  const review = signal?.agent_review || signal?.raw?.agent_review || {}
  const signalOptions = signals.slice(0, 80)

  if (!signal) {
    return (
      <section className="card decision-view">
        <h2 className="panel-title"><Target size={15} /> Signal decision context</h2>
        <div className="text-muted">No shadow signal available yet.</div>
      </section>
    )
  }

  return (
    <section className="card decision-view">
      <div className="decision-head">
        <div>
          <h2 className="panel-title"><Target size={15} /> Signal decision context</h2>
          <p className="text-muted decision-copy">
            Read-only view of the current shadow plan. It explains levels and context, but it does not place orders or override strategy rules.
          </p>
        </div>
        <div className="decision-controls">
          <select value={signalKey(signal)} onChange={(event) => onSelectSignal(event.target.value)}>
            {signalOptions.map((item) => (
              <option key={signalKey(item)} value={signalKey(item)}>
                {item.status === 'OPEN' ? 'OPEN | ' : ''}{item.symbol} {item.side} {formatDate(item.generated_at)} | {shortId(item.config_id)}
              </option>
            ))}
          </select>
          <div className="segmented">
            {['15m', '1h', '4h'].map((item) => (
              <button
                className={interval === item ? 'segmented-active' : ''}
                key={item}
                type="button"
                onClick={() => setInterval(item)}
              >
                {item}
              </button>
            ))}
          </div>
        </div>
      </div>

      {marketState.error && (
        <div className="sample-warning" style={{ marginBottom: 12 }}>
          Market context partially unavailable: {marketState.error}
        </div>
      )}

      <div className="decision-grid">
        <div className="decision-chart-card">
          <div className="section-head">
            <div>
              <div className="mono decision-title">{signal.symbol} {interval} {signal.side}</div>
              <div className="text-muted" style={{ fontSize: 12 }}>
                Generated {formatDate(signal.generated_at)} | Config {shortId(signal.config_id)}
              </div>
            </div>
            {marketState.loading && <span className="spinner" />}
          </div>
          <SignalPriceChart marketData={selectedMarket} signal={signal} interval={interval} />
        </div>

        <div className="decision-side-panel">
          <section className="level-card">
            <div className="mono" style={{ marginBottom: 10 }}>Shadow trade plan</div>
            <StatLine label="Side" value={signal.side || '-'} tone={signal.side === 'LONG' ? 'good' : 'bad'} />
            <StatLine label="Entry" value={formatNumber(signal.entry_price, 6)} />
            <StatLine label="Stop loss" value={formatNumber(signal.stop_loss, 6)} tone="bad" />
            <StatLine label="Take profit" value={formatNumber(signal.take_profit, 6)} tone="good" />
            <StatLine label="Risk reward" value={formatNumber(signal.risk_reward, 2)} />
            <StatLine label="Horizon" value={`${signal.horizon_candles || '-'} candles / ${signal.horizon_minutes || '-'} min`} />
            <StatLine label="Confidence" value={formatPct(signal.confidence)} />
          </section>

          <section className={`level-card context-${decision?.tone || 'warn'}`}>
            <div className="mono" style={{ marginBottom: 10 }}>Tactical read</div>
            <div className={decision?.tone === 'good' ? 'text-green' : decision?.tone === 'bad' ? 'text-red' : 'text-amber'}>
              {decision?.label || 'No context loaded yet.'}
            </div>
            <div className="stack" style={{ marginTop: 12 }}>
              <StatLine label="Buy probability" value={formatProbability(features.probability_buy_win, 2)} />
              <StatLine label="Sell probability" value={formatProbability(features.probability_sell_win, 2)} />
              <StatLine label="Agent review" value={review.review_status || '-'} tone={review.review_status === 'BLOCK' ? 'bad' : review.review_status === 'CAUTION' ? 'warn' : 'neutral'} />
              <StatLine label="Near support" value={formatNumber(selectedLevels.support, 6)} />
              <StatLine label="Near resistance" value={formatNumber(selectedLevels.resistance, 6)} />
              <StatLine label="24-candle range" value={formatPct(selectedLevels.rangePct)} />
            </div>
          </section>
        </div>
      </div>

      <div className="timeframe-grid">
        {['15m', '1h', '4h'].map((item) => (
          <TimeframeContextCard key={item} interval={item} marketData={marketState.data[item]} signal={signal} />
        ))}
      </div>

      <div className="sample-warning">
        This is paper/shadow research. A visually aligned chart is not enough for real-money execution; measured PF, drawdown, and larger live samples remain the gate.
      </div>
    </section>
  )
}

function CycleWarningBanner({ issue }) {
  if (!issue) return null
  return (
    <section className="cycle-warning">
      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        <AlertTriangle size={18} color="var(--amber)" />
        <div>
          <div className="mono" style={{ color: 'var(--amber)' }}>Last shadow cycle warning</div>
          <div className="text-muted" style={{ fontSize: 12 }}>
            skipped_no_price={issue.skippedNoPrice} | evaluation_errors={issue.evaluationErrors} | skipped_errors={issue.skippedErrors} | {formatDate(issue.finishedAt)}
          </div>
        </div>
      </div>
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
          <StatLine label="Buy probability" value={formatProbability(features.probability_buy_win, 2)} />
          <StatLine label="Sell probability" value={formatProbability(features.probability_sell_win, 2)} />
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

function ConfigSummary({ byConfig, onOpenConfig }) {
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
                  <td>
                    <button className="table-link mono" type="button" onClick={() => onOpenConfig(config)}>
                      {shortId(config)}
                    </button>
                  </td>
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

function ConfigSignalsDrawer({ configId, signals, onClose }) {
  const configSignals = useMemo(() => {
    return signals
      .filter((signal) => signal.config_id === configId)
      .sort((a, b) => new Date(a.generated_at || 0).getTime() - new Date(b.generated_at || 0).getTime())
  }, [configId, signals])
  const equityPoints = useMemo(() => buildEquityCurve(configSignals), [configSignals])
  const closed = configSignals.filter((signal) => isFinishedSignal(signal))

  if (!configId) return null

  return (
    <div className="drawer-backdrop" onClick={onClose}>
      <aside className="drawer-panel" onClick={(event) => event.stopPropagation()}>
        <div className="drawer-head">
          <div>
            <div className="text-muted mono" style={{ fontSize: 11 }}>Config performance</div>
            <h2 className="mono" style={{ fontSize: 18 }}>{shortId(configId)}</h2>
          </div>
          <button className="icon-btn" type="button" onClick={onClose} aria-label="Close config details">
            <X size={18} />
          </button>
        </div>

        <div className="config-detail-grid" style={{ marginBottom: 16 }}>
          <StatLine label="Signals" value={configSignals.length} />
          <StatLine label="Closed" value={closed.length} />
          <StatLine label="Wins" value={closed.filter((signal) => signal.outcome === 'WIN').length} tone="good" />
          <StatLine label="Losses" value={closed.filter((signal) => signal.outcome === 'LOSS').length} tone="bad" />
        </div>

        <MetricLineChart points={equityPoints} baseline={0} segmented height={170} />

        <div style={{ overflowX: 'auto', marginTop: 16 }}>
          <table className="data-table compact-table">
            <thead>
              <tr>
                <th>Generated</th>
                <th>Side</th>
                <th>Status</th>
                <th>Outcome</th>
                <th>PnL</th>
                <th>Close</th>
              </tr>
            </thead>
            <tbody>
              {configSignals.map((signal) => (
                <tr key={signal.shadow_signal_id}>
                  <td className="mono">{formatDate(signal.generated_at)}</td>
                  <td><span className={`badge ${signal.side === 'LONG' ? 'badge-buy' : 'badge-sell'}`}>{signal.side || '-'}</span></td>
                  <td><span className={`badge ${statusClass(signal.status)}`}>{signal.status}</span></td>
                  <td className={outcomeClass(signal.outcome)}>{signal.outcome || '-'}</td>
                  <td className={Number(signal.pnl_pct) >= 0 ? 'text-green' : 'text-red'}>{formatPct(signal.pnl_pct, 4)}</td>
                  <td className="mono">{formatDate(getSignalCloseTime(signal))}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </aside>
    </div>
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

function CycleDiagnostics({ cycles }) {
  const latest = cycles[0]
  const statusCounts = latest?.status_counts || latest?.raw?.status_counts || {}
  return (
    <section className="card">
      <div className="section-head">
        <h2 className="panel-title">Last cycle diagnostics</h2>
        <span className="text-muted mono" style={{ fontSize: 11 }}>HOLD / skip reasons</span>
      </div>
      {!latest ? (
        <div className="text-muted">No shadow ops cycle has been synced yet.</div>
      ) : (
        <>
          <div className="config-detail-grid">
            <StatLine label="Finished" value={formatDate(latest.finished_at)} />
            <StatLine label="Health" value={latest.health_status || '-'} tone={latest.health_status === 'HEALTH_OK' ? 'good' : 'warn'} />
            <StatLine label="Configs scanned" value={latest.configs_scanned ?? 0} />
            <StatLine label="Opened signals" value={latest.opened_signals ?? 0} tone={Number(latest.opened_signals) > 0 ? 'good' : 'neutral'} />
            <StatLine label="Skipped HOLD" value={latest.skipped_hold ?? 0} tone={Number(latest.skipped_hold) > 0 ? 'warn' : 'neutral'} />
            <StatLine label="Duplicate open" value={latest.skipped_duplicate_open ?? 0} />
            <StatLine label="Duplicate similar" value={latest.skipped_duplicate_similar ?? 0} />
            <StatLine label="Skipped errors" value={latest.skipped_errors ?? 0} tone={Number(latest.skipped_errors) > 0 ? 'bad' : 'neutral'} />
            <StatLine label="Generation skipped" value={latest.generation_skipped_reason || 'no'} tone={latest.generation_skipped_reason ? 'warn' : 'neutral'} />
            <StatLine label="Evaluated closed" value={latest.evaluated_closed ?? 0} />
            <StatLine label="Final open" value={latest.final_open ?? 0} tone={Number(latest.final_open) > 0 ? 'warn' : 'neutral'} />
            <StatLine label="Cycle sync" value={latest.supabase_sync_ok ? 'ok' : (latest.supabase_sync_reason || 'not ok')} tone={latest.supabase_sync_ok ? 'good' : 'warn'} />
          </div>
          <div className="tag-row" style={{ marginTop: 12 }}>
            {Object.entries(statusCounts).length ? (
              Object.entries(statusCounts).map(([key, value]) => <span className="tag" key={key}>{key}: {String(value)}</span>)
            ) : (
              <span className="text-muted">No status counts recorded.</span>
            )}
          </div>
        </>
      )}
    </section>
  )
}

function SignalsTable({ signals, onOpenConfig }) {
  const [filters, setFilters] = useState({ symbol: '', outcome: '', configId: '' })
  const [expandedId, setExpandedId] = useState('')
  const symbols = useMemo(() => Array.from(new Set(signals.map((signal) => signal.symbol).filter(Boolean))).sort(), [signals])
  const outcomes = useMemo(() => Array.from(new Set(signals.map((signal) => signal.outcome).filter(Boolean))).sort(), [signals])
  const configs = useMemo(() => Array.from(new Set(signals.map((signal) => signal.config_id).filter(Boolean))).sort(), [signals])
  const filteredSignals = useMemo(() => {
    return signals.filter((signal) => {
      if (filters.symbol && signal.symbol !== filters.symbol) return false
      if (filters.outcome && signal.outcome !== filters.outcome) return false
      if (filters.configId && signal.config_id !== filters.configId) return false
      return true
    })
  }, [filters, signals])

  return (
    <section className="card">
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12, marginBottom: 12 }}>
        <h2 className="mono" style={{ fontSize: 15 }}>Recent Shadow Signals</h2>
        <span className="text-muted mono" style={{ fontSize: 11 }}>{filteredSignals.length} / {signals.length} rows</span>
      </div>
      <div className="table-filters">
        <select value={filters.symbol} onChange={(event) => setFilters((current) => ({ ...current, symbol: event.target.value }))}>
          <option value="">All symbols</option>
          {symbols.map((symbol) => <option key={symbol} value={symbol}>{symbol}</option>)}
        </select>
        <select value={filters.outcome} onChange={(event) => setFilters((current) => ({ ...current, outcome: event.target.value }))}>
          <option value="">All outcomes</option>
          {outcomes.map((outcome) => <option key={outcome} value={outcome}>{outcome}</option>)}
        </select>
        <select value={filters.configId} onChange={(event) => setFilters((current) => ({ ...current, configId: event.target.value }))}>
          <option value="">All configs</option>
          {configs.map((config) => <option key={config} value={config}>{shortId(config)}</option>)}
        </select>
      </div>

      {filteredSignals.length === 0 ? (
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
              {filteredSignals.map((signal) => {
                const review = signal.agent_review || {}
                const features = signal.input_features || signal.raw?.input_features || {}
                const market = signal.market_context || signal.raw?.market_context || {}
                const news = signal.news_context || signal.raw?.news_context || {}
                const closeHours = hoursBetween(signal.generated_at, getSignalCloseTime(signal))
                const expanded = expandedId === signal.shadow_signal_id
                return (
                  <Fragment key={signal.shadow_signal_id}>
                    <tr
                      className={expanded ? 'selected-row expandable-row' : 'expandable-row'}
                      onClick={() => setExpandedId(expanded ? '' : signal.shadow_signal_id)}
                    >
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
                      <td>
                        <button
                          className="table-link mono"
                          type="button"
                          onClick={(event) => {
                            event.stopPropagation()
                            onOpenConfig(signal.config_id)
                          }}
                        >
                          {shortId(signal.config_id)}
                        </button>
                      </td>
                    </tr>
                    {expanded && (
                      <tr className="signal-detail-row">
                        <td colSpan={14}>
                          <div className="signal-detail-grid">
                            <StatLine label="Buy probability" value={formatProbability(features.probability_buy_win, 2)} />
                            <StatLine label="Sell probability" value={formatProbability(features.probability_sell_win, 2)} />
                            <StatLine label="Hours to close" value={closeHours === null ? '-' : formatNumber(closeHours, 2)} />
                            <StatLine label="Close time" value={formatDate(getSignalCloseTime(signal))} />
                            <StatLine label="Market context" value={market.context_status || market.review_status || market.trend || '-'} />
                            <StatLine label="News context" value={news.context_status || news.provider_status || news.sentiment || '-'} />
                          </div>
                          <div className="detail-tags">
                            {compactEntries(market).map(([key, value]) => <span className="tag" key={`market-${key}`}>market.{key}: {String(value)}</span>)}
                            {compactEntries(features).map(([key, value]) => <span className="tag" key={`feature-${key}`}>feature.{key}: {String(value)}</span>)}
                          </div>
                        </td>
                      </tr>
                    )}
                  </Fragment>
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
  const [state, setState] = useState({ loading: true, error: '', health: null, summary: null, signals: [], cycles: [], configHealth: null })
  const [selectedConfig, setSelectedConfig] = useState('')
  const [configDrawerId, setConfigDrawerId] = useState('')
  const [selectedDecisionSignalId, setSelectedDecisionSignalId] = useState('')

  async function load() {
    setState((current) => ({ ...current, loading: true, error: '' }))
    try {
      const [health, summary, signals, cycles, configHealth] = await Promise.all([
        api.shadow.health(),
        api.shadow.summary(),
        api.shadow.signals({ limit: 200 }),
        api.shadow.cycles({ limit: 20 }),
        api.shadow.configHealth(),
      ])
      setState({
        loading: false,
        error: '',
        health,
        summary: summary.data,
        signals: signals.data || [],
        cycles: cycles.data || [],
        configHealth: configHealth.data,
      })
    } catch (err) {
      setState({ loading: false, error: err.message, health: null, summary: null, signals: [], cycles: [], configHealth: null })
    }
  }

  useEffect(() => {
    load()
  }, [])

  const summary = state.summary?.summary || state.health?.summary || {}
  const strategySummary = state.summary?.strategy_eligible || {}
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
  const cycleIssue = useMemo(() => getCycleIssues(state.cycles?.[0]), [state.cycles])

  useEffect(() => {
    if (configRows.length === 0) {
      setSelectedConfig('')
      return
    }
    if (!configRows.some((row) => row.configId === selectedConfig)) {
      setSelectedConfig(configRows[0].configId)
    }
  }, [configRows, selectedConfig])

  useEffect(() => {
    if (sortedSignals.length === 0) {
      setSelectedDecisionSignalId('')
      return
    }
    if (!sortedSignals.some((item) => signalKey(item) === selectedDecisionSignalId)) {
      setSelectedDecisionSignalId(signalKey(latestOpen[0] || sortedSignals[0]))
    }
  }, [sortedSignals, selectedDecisionSignalId, latestOpen])

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
          <CycleWarningBanner issue={cycleIssue} />
          <div className="kpi-grid">
            <Kpi label="Open" value={summary.open ?? 0} tone={latestOpen.length > 0 ? 'warn' : 'neutral'} />
            <Kpi label="Closed" value={summary.closed ?? 0} />
            <Kpi label="Win Rate" value={formatPct(summary.win_rate)} />
            <Kpi label="Profit Factor" value={formatNumber(summary.profit_factor, 4)} tone={pfTone} />
            <Kpi label="Avg Return" value={formatPct(summary.avg_return, 4)} tone={avgTone} />
            <Kpi label="Max Drawdown" value={formatPct(summary.max_drawdown)} tone={Number(summary.max_drawdown) > 10 ? 'bad' : 'neutral'} />
          </div>

          <StrategyEvidenceStrip
            strategySummary={strategySummary}
            technicalExclusions={state.summary?.technical_exclusions}
            exclusionsByReason={state.summary?.technical_exclusions_by_exit_reason}
          />

          <div className="ops-grid">
            <ActiveSignalPanel signal={latestOpen[0]} />
            <FreshnessPanel health={state.health} summary={state.summary} signals={sortedSignals} />
            <SignalContextPanel signal={latestSignal} />
          </div>

          <SignalDecisionView
            signals={sortedSignals}
            selectedSignalId={selectedDecisionSignalId}
            onSelectSignal={setSelectedDecisionSignalId}
          />

          <ConfigHealthPanel report={state.configHealth} onOpenConfig={setConfigDrawerId} />

          <div className="shadow-grid">
            <SymbolSummary bySymbol={state.summary?.by_symbol || {}} />
            <ConfigSummary byConfig={state.summary?.by_config || {}} onOpenConfig={setConfigDrawerId} />
          </div>

          <PerformanceCharts signals={sortedSignals} />

          <ConfigDeepDive
            configRows={configRows}
            selectedConfig={selectedConfig}
            onSelectConfig={setSelectedConfig}
          />

          <ConfidenceBuckets buckets={confidenceBuckets} />

          <CycleDiagnostics cycles={state.cycles} />

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

          <SignalsTable signals={sortedSignals} onOpenConfig={setConfigDrawerId} />
          <ConfigSignalsDrawer configId={configDrawerId} signals={sortedSignals} onClose={() => setConfigDrawerId('')} />
        </>
      )}
    </>
  )
}
