import { useDashboardData } from '../controllers/useDashboardData'
import { ActionFeed } from './components/ActionFeed'
import { ActionSummary } from './components/ActionSummary'
import { BallTrajectoryChart } from './components/BallTrajectoryChart'
import { HeroVideo } from './components/HeroVideo'
import { PlayerCourtMap } from './components/PlayerCourtMap'
import { PlayerTracksChart } from './components/PlayerTracksChart'
import { RallyLog } from './components/RallyLog'
import { SessionList } from './components/SessionList'
import { StatTile } from './components/StatTile'
import { StatusBadge } from './components/StatusBadge'
import './dashboard.css'

function formatDuration(frameCount: number, fps: number): string {
  const totalSeconds = Math.round(frameCount / fps)
  const m = Math.floor(totalSeconds / 60)
  const s = totalSeconds % 60
  return `${m}:${s.toString().padStart(2, '0')}`
}

export function Dashboard() {
  const { sessions, selectedSessionId, detail, loadingSessions, loadingDetail, error, selectSession } =
    useDashboardData()

  const ballDetectedPct = detail
    ? Math.round((detail.ball.filter((p) => !p.interpolated).length / Math.max(1, detail.ball.length)) * 100)
    : null
  const avgPlayerConf = detail && detail.tracks.length
    ? detail.tracks.reduce((sum, t) => sum + t.avgConf, 0) / detail.tracks.length
    : null

  return (
    <div className="viz-root">
      <header className="dash-header">
        <div>
          <p className="eyebrow">Mongo Volleyball Hackathon 2026</p>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <h1>{detail ? detail.session.source : 'vbtrack'}</h1>
            {detail && <StatusBadge status={detail.session.status} />}
          </div>
          <p>Volleyball player + ball tracking — sample data for demo purposes</p>
        </div>
        <span className="dash-badge">
          {loadingSessions ? 'loading sessions…' : `${sessions.length} sessions recorded`}
        </span>
      </header>

      <HeroVideo src="/videos/annotated_10.mp4" width={1280} height={720} label="tracked footage — annotated_10.mp4" />

      {detail && <RallyLog rallyLog={detail.rallyLog} ballTrackedPct={ballDetectedPct} />}

      {error && <p className="empty-state">Error: {error}</p>}

      <div className="dash-grid">
        <aside className="card">
          <h2>Sessions</h2>
          <SessionList sessions={sessions} selectedId={selectedSessionId} onSelect={selectSession} />
        </aside>

        <main>
          {loadingDetail && !detail && <p className="empty-state">Loading session…</p>}

          {detail && (
            <>
              <div className="stat-grid">
                <StatTile value={detail.session.frameCount.toLocaleString()} label="Frames processed" />
                <StatTile value={formatDuration(detail.session.frameCount, detail.session.fps)} label="Duration (mm:ss)" />
                <StatTile value={`${detail.session.width}×${detail.session.height}`} label="Resolution" />
                <StatTile value={ballDetectedPct !== null ? `${ballDetectedPct}%` : '—'} label="Ball detected (vs. interpolated)" />
                <StatTile value={String(detail.tracks.length)} label="Players tracked" />
                <StatTile value={avgPlayerConf !== null ? avgPlayerConf.toFixed(2) : '—'} label="Avg player confidence" />
              </div>

              <div className="card">
                <h2>Ball trajectory</h2>
                <BallTrajectoryChart points={detail.ball} width={detail.session.width} height={detail.session.height} />
              </div>

              <div className="section-row">
                <div className="card">
                  <h2>Player court positions (avg)</h2>
                  <PlayerCourtMap tracks={detail.tracks} width={detail.session.width} height={detail.session.height} />
                </div>
                <div className="card">
                  <h2>Playtime by player</h2>
                  <PlayerTracksChart tracks={detail.tracks} />
                </div>
              </div>

              <div className="section-row">
                <div className="card">
                  <h2>Action summary</h2>
                  <ActionSummary actions={detail.actions} />
                </div>
                <div className="card">
                  <h2>Play-by-play</h2>
                  <ActionFeed actions={detail.actions} />
                </div>
              </div>
            </>
          )}
        </main>
      </div>
    </div>
  )
}
