import type { PlayerTrack } from '../../models/types'
import { trackColor } from '../trackColor'

export function PlayerTracksChart({ tracks }: { tracks: PlayerTrack[] }) {
  if (tracks.length === 0) {
    return <p className="empty-state">No player tracks in this session.</p>
  }
  const max = Math.max(...tracks.map((t) => t.framesSeen))
  return (
    <div>
      {tracks.map((t) => (
        <div className="bar-row" key={t.trackId}>
          <span className="track-label">#{t.trackId}</span>
          <div className="bar-track">
            <div
              className="bar-fill"
              style={{ width: `${(t.framesSeen / max) * 100}%`, background: trackColor(t.trackId) }}
            />
          </div>
          <span className="value">{t.avgConf.toFixed(2)}</span>
        </div>
      ))}
      <div className="chart-legend">
        <span>bar length = frames tracked · right column = avg detection confidence</span>
      </div>
    </div>
  )
}
