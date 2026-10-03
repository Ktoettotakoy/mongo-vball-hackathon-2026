import type { PlayerTrack } from '../../models/types'

// Scatter forms carry a 3-series color cap (the full 8-slot order only clears
// adjacent-pair CVD gates, not all-pairs) - the top 3 tracks by playtime get a
// validated color, the rest fold into a single muted "other" dot. Every dot is
// still direct-labeled with its track id, so identity never rests on color alone.
const TOP_COLORS = ['var(--series-1)', 'var(--series-2)', 'var(--series-3)']

export function PlayerCourtMap({ tracks, width, height }: { tracks: PlayerTrack[]; width: number; height: number }) {
  if (tracks.length === 0) {
    return <p className="empty-state">No player tracks in this session.</p>
  }
  const ranked = [...tracks].sort((a, b) => b.framesSeen - a.framesSeen)
  const otherCount = Math.max(0, ranked.length - 3)

  return (
    <div className="chart-wrap">
      <svg className="chart-svg" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Average player positions on court">
        <rect x={1} y={1} width={width - 2} height={height - 2} fill="none" stroke="var(--gridline)" rx={6} />
        <line x1={width / 2} y1={0} x2={width / 2} y2={height} stroke="var(--axis)" strokeDasharray="6 8" strokeWidth={2} />
        {ranked.map((t, i) => {
          const color = i < 3 ? TOP_COLORS[i] : 'var(--series-other)'
          return (
            <g key={t.trackId}>
              <circle cx={t.avgFoot.x} cy={t.avgFoot.y} r={14} fill={color} opacity={0.85} />
              <text
                x={t.avgFoot.x}
                y={t.avgFoot.y + 5}
                textAnchor="middle"
                fontSize={13}
                fontWeight={600}
                fill="var(--surface-1)"
              >
                {t.trackId}
              </text>
            </g>
          )
        })}
      </svg>
      <div className="chart-legend">
        {ranked.slice(0, 3).map((t, i) => (
          <span className="swatch" key={t.trackId}>
            <span className="chip" style={{ background: TOP_COLORS[i] }} />
            player #{t.trackId} ({t.framesSeen.toLocaleString()} frames)
          </span>
        ))}
        {otherCount > 0 && (
          <span className="swatch">
            <span className="chip" style={{ background: 'var(--series-other)' }} />
            {otherCount} other player{otherCount > 1 ? 's' : ''}
          </span>
        )}
      </div>
    </div>
  )
}
