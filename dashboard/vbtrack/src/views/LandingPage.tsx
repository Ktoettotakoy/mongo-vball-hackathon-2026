import { Link } from 'react-router-dom'
import { useDashboardData } from '../controllers/useDashboardData'
import { ContainerScroll } from './components/ContainerScroll'
import { StatTile } from './components/StatTile'
import './dashboard.css'
import './landing.css'

export function LandingPage() {
  const { detail } = useDashboardData()

  const ballDetectedPct = detail
    ? Math.round((detail.ball.filter((p) => !p.interpolated).length / Math.max(1, detail.ball.length)) * 100)
    : null

  return (
    <div className="viz-root landing-root">
      <ContainerScroll
        titleComponent={
          <>
            <p className="eyebrow">Mongo Volleyball Hackathon 2026</p>
            <h1 className="cs-title">
              Make Volleyball Match Trackable <br />
              <span className="cs-title-big">every touch, every rally</span>
            </h1>
          </>
        }
      >
        <video
          src="/videos/annotated_10.mp4"
          autoPlay
          loop
          muted
          playsInline
          draggable={false}
        />
      </ContainerScroll>

      <div className="sponsor-section">
        <a className="sponsor-half" href="https://www.mongodb.com" target="_blank" rel="noreferrer">
          <span className="sponsor-eyebrow">Sponsor</span>
          <span className="sponsor-name">MongoDB</span>
        </a>
        <a className="sponsor-half" href="https://hotplit.sh" target="_blank" rel="noreferrer">
          <span className="sponsor-eyebrow">Sponsor</span>
          <span className="sponsor-name">hotplit.sh</span>
        </a>
      </div>

      <div className="landing-hero">
        <p className="landing-sub">
          Volleyball player + ball tracking, written frame-by-frame into MongoDB. Placeholder landing page —
          swap this copy and links for the real pitch.
        </p>
        <Link to="/dashboard" className="landing-cta">
          View dashboard →
        </Link>
      </div>

      <div className="landing-grid">
        <StatTile value={detail ? detail.session.frameCount.toLocaleString() : '—'} label="Frames tracked" />
        <StatTile value={detail ? String(detail.tracks.length) : '—'} label="Players tracked" />
        <StatTile value={ballDetectedPct !== null ? `${ballDetectedPct}%` : '—'} label="Ball detection rate" />
      </div>
    </div>
  )
}
