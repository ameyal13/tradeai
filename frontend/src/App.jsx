import { useEffect, useState } from 'react'
import { Navigate, Route, Routes } from 'react-router-dom'
import Layout from './components/Layout.jsx'
import ShadowSignalsPage from './components/ShadowSignalsPage.jsx'
import { api } from './lib/api.js'

function PageHeader({ title, description }) {
  return (
    <div style={{ marginBottom: '1rem' }}>
      <h1 style={{ fontFamily: 'var(--mono)', fontSize: 24, color: 'var(--text)', marginBottom: 6 }}>
        {title}
      </h1>
      <p style={{ color: 'var(--muted)', maxWidth: 760 }}>{description}</p>
    </div>
  )
}

function DashboardPage() {
  const [state, setState] = useState({ loading: true, error: '', data: [] })

  useEffect(() => {
    let mounted = true
    api.market.overview()
      .then((res) => mounted && setState({ loading: false, error: '', data: res.data || [] }))
      .catch((err) => mounted && setState({ loading: false, error: err.message, data: [] }))
    return () => { mounted = false }
  }, [])

  return (
    <>
      <PageHeader
        title="Dashboard"
        description="Vista mínima para validar que el frontend habla con el backend antes de construir el motor de evaluación."
      />
      <section className="card">
        {state.loading && <div className="text-muted">Cargando mercado...</div>}
        {state.error && <div className="text-red">Backend no disponible: {state.error}</div>}
        {!state.loading && !state.error && (
          <div style={{ display: 'grid', gap: 8 }}>
            {state.data.map((coin) => (
              <div key={coin.symbol} style={{ display: 'flex', justifyContent: 'space-between', gap: 16 }}>
                <span className="mono">{coin.symbol}</span>
                <span>${Number(coin.price).toLocaleString()}</span>
              </div>
            ))}
          </div>
        )}
      </section>
    </>
  )
}

function PlaceholderPage({ title, description }) {
  return (
    <>
      <PageHeader title={title} description={description} />
      <section className="card text-muted">
        Base conectada. La siguiente prioridad es Backtest Engine V2 y Prediction Journal.
      </section>
    </>
  )
}

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<ShadowSignalsPage />} />
        <Route path="chart" element={<PlaceholderPage title="Graficas" description="Ruta reservada para mercado y velas." />} />
        <Route path="signals" element={<ShadowSignalsPage />} />
        <Route path="news" element={<PlaceholderPage title="Noticias" description="Ruta reservada para contexto y fuentes cacheadas." />} />
        <Route path="backtest" element={<PlaceholderPage title="Backtest" description="Ruta reservada para probar Backtest Engine V2." />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  )
}
