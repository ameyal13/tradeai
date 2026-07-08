// frontend/src/lib/api.js
const BASE = (import.meta.env.VITE_API_URL || 'http://localhost:8000').replace(/\/+$/, '')

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...options.headers },
    ...options,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    const detail = err.detail || err.message || err.error || res.statusText || `HTTP ${res.status}`
    throw new Error(detail)
  }
  return res.json()
}

export const api = {
  market: {
    overview:   ()                        => request('/market/overview'),
    detail:     (symbol, interval = '1h') => request(`/market/${symbol}?interval=${interval}`),
  },
  news: {
    feed:       (symbol, limit = 15)      => request(`/news${symbol ? `?symbol=${symbol}&limit=${limit}` : `?limit=${limit}`}`),
  },
  signals: {
    generate:   (symbol, interval, provider = 'groq') =>
      request('/signals/generate', { method: 'POST', body: JSON.stringify({ symbol, interval, provider }) }),
    list:       (symbol, limit = 20)      => request(`/signals${symbol ? `?symbol=${symbol}&limit=${limit}` : `?limit=${limit}`}`),
  },
  shadow: {
    health:     ()                        => request('/shadow/health'),
    summary:    ()                        => request('/shadow/summary'),
    signals:    ({ status = '', symbol = '', limit = 50 } = {}) => {
      const params = new URLSearchParams()
      if (status) params.set('status', status)
      if (symbol) params.set('symbol', symbol)
      params.set('limit', String(limit))
      return request(`/shadow/signals?${params.toString()}`)
    },
    cycles:     ({ limit = 20 } = {}) => request(`/shadow/cycles?limit=${limit}`),
    configHealth: ()                   => request('/shadow/config-health'),
  },
  research: {
    summary:    ({ source = 'crypto_multi' } = {}) => request(`/research/summary?source=${encodeURIComponent(source)}`),
  },
  backtest: {
    run:        (payload)                 => request('/backtest/run', { method: 'POST', body: JSON.stringify(payload) }),
    get:        (id)                      => request(`/backtest/${id}`),
    list:       ()                        => request('/backtest'),
  },
  chat: {
    send:       (message, history = [], session_id = null, provider = 'groq') =>
      request('/chat', { method: 'POST', body: JSON.stringify({ message, history, session_id, provider }) }),
  },
}

// WebSocket helper
export function createPriceSocket(symbol, onMessage) {
  const WS_BASE = import.meta.env.VITE_WS_URL || 'ws://localhost:8000'
  const ws = new WebSocket(`${WS_BASE}/ws/prices/${symbol}`)
  ws.onmessage = (e) => onMessage(JSON.parse(e.data))
  ws.onerror   = (e) => console.error('WS error', e)
  return ws
}
