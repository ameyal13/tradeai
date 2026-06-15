// frontend/src/components/Layout.jsx
import { NavLink, Outlet } from 'react-router-dom'

const NAV = [
  { to: '/', icon: 'D', label: 'Dashboard' },
  { to: '/chart', icon: 'C', label: 'Graficas' },
  { to: '/signals', icon: 'S', label: 'Senales' },
  { to: '/research', icon: 'R', label: 'Research' },
  { to: '/news', icon: 'N', label: 'Noticias' },
  { to: '/backtest', icon: 'B', label: 'Backtest' },
]

export default function Layout() {
  return (
    <div style={{ display: 'flex', minHeight: '100vh' }}>
      <aside style={{
        width: 200,
        background: 'var(--bg2)',
        borderRight: '1px solid var(--border)',
        display: 'flex',
        flexDirection: 'column',
        padding: '1.25rem 0',
        flexShrink: 0,
      }}>
        <div style={{ padding: '0 1.25rem 1.5rem', borderBottom: '1px solid var(--border)' }}>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 13, color: 'var(--muted)', marginBottom: 2 }}>
            TRADING
          </div>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 18, fontWeight: 600, color: 'var(--green)' }}>
            COPILOT
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 8 }}>
            <span className="loading-dot" style={{ width: 6, height: 6 }} />
            <span style={{ fontSize: 11, color: 'var(--muted)', fontFamily: 'var(--mono)' }}>LIVE</span>
          </div>
        </div>

        <nav style={{ flex: 1, padding: '1rem 0.75rem', display: 'flex', flexDirection: 'column', gap: 2 }}>
          {NAV.map(({ to, icon, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              style={({ isActive }) => ({
                display: 'flex',
                alignItems: 'center',
                gap: 10,
                padding: '8px 12px',
                borderRadius: 'var(--radius)',
                textDecoration: 'none',
                fontSize: 13,
                fontWeight: 500,
                color: isActive ? 'var(--green)' : 'var(--muted)',
                background: isActive ? 'rgba(0,212,160,0.08)' : 'transparent',
                border: isActive ? '1px solid rgba(0,212,160,0.15)' : '1px solid transparent',
              })}
            >
              <span className="mono" style={{ fontSize: 12, lineHeight: 1, width: 14 }}>{icon}</span>
              {label}
            </NavLink>
          ))}
        </nav>

        <div style={{ padding: '1rem 1.25rem', borderTop: '1px solid var(--border)' }}>
          <div style={{ fontSize: 11, color: 'var(--muted)', fontFamily: 'var(--mono)', lineHeight: 1.6 }}>
            <div>MVP v1.0</div>
            <div style={{ color: 'rgba(90,106,122,0.6)', marginTop: 4, fontSize: 10 }}>
              No es asesoramiento financiero
            </div>
          </div>
        </div>
      </aside>

      <main style={{ flex: 1, overflow: 'auto', padding: '1.5rem' }}>
        <Outlet />
      </main>
    </div>
  )
}
