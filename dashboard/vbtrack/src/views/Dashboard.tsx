import { useState } from 'react'
import { useDashboardData } from '../controllers/useDashboardData'
import { ActionFeed } from './components/ActionFeed'
import { ActionSummary } from './components/ActionSummary'
import { CardTabs } from './components/CardTabs'
import { HeroVideo } from './components/HeroVideo'
import { RallyLog } from './components/RallyLog'
import { SessionList } from './components/SessionList'
import { StatusBadge } from './components/StatusBadge'
import './dashboard.css'

export function Dashboard() {
  const { sessions, selectedSessionId, detail, loadingSessions, loadingDetail, error, selectSession } =
    useDashboardData()
  const [showSessions, setShowSessions] = useState(false)

  const ballDetectedPct = detail
    ? Math.round((detail.ball.filter((p) => !p.interpolated).length / Math.max(1, detail.ball.length)) * 100)
    : null

  return (
    <div className="viz-root">
      <div className="hero-section">
        <header className="dash-header">
          <div>
            <p className="eyebrow">Mongo Volleyball Hackathon 2026</p>
            <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
              <h1>{detail ? detail.session.source : 'vbtrack'}</h1>
              {detail && <StatusBadge status={detail.session.status} />}
            </div>
            <p>Volleyball player + ball tracking — sample data for demo purposes</p>
          </div>
          <button type="button" className="dash-badge" onClick={() => setShowSessions((v) => !v)}>
            {loadingSessions
              ? 'loading sessions…'
              : `${showSessions ? 'Hide' : 'Show'} sessions (${sessions.length})`}
          </button>
        </header>

        <div className="hero-row">
          <HeroVideo src="/videos/annotated_10.mp4" width={1280} height={720} label="tracked footage — annotated_10.mp4" />
          {showSessions && (
            <aside className="card session-switcher">
              <h2>Sessions</h2>
              <SessionList
                sessions={sessions}
                selectedId={selectedSessionId}
                onSelect={(id) => {
                  selectSession(id)
                  setShowSessions(false)
                }}
              />
            </aside>
          )}
        </div>
      </div>

      {detail && <RallyLog rallyLog={detail.rallyLog} ballTrackedPct={ballDetectedPct} />}

      {error && <p className="empty-state">Error: {error}</p>}

      <div className="dash-main">
        {loadingDetail && !detail && <p className="empty-state">Loading session…</p>}

        {detail && (
          <CardTabs
            tabs={[
              {
                id: 'summary',
                title: 'Action summary',
                badge: detail.actionsSource === 'mongo' ? 'LLM labels from MongoDB' : 'mock data (API offline)',
                content: <ActionSummary actions={detail.actions} stats={detail.playerStats} />,
              },
              { id: 'feed', title: 'Play-by-play', content: <ActionFeed actions={detail.actions} /> },
            ]}
          />
        )}
      </div>
    </div>
  )
}
