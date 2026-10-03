import { useMemo, useState } from 'react'
import type { BallPoint } from '../../models/types'

interface Props {
  points: BallPoint[]
  width: number
  height: number
}

// Sequential blue ramp (palette.md "Sequential hue"), light -> dark by time progress.
const SEQ_STEPS = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']

function seqColor(p: number): string {
  const idx = Math.min(SEQ_STEPS.length - 1, Math.floor(p * SEQ_STEPS.length))
  return SEQ_STEPS[idx]
}

export function BallTrajectoryChart({ points, width, height }: Props) {
  const [hover, setHover] = useState<number | null>(null)

  const path = useMemo(
    () => points.map((p, i) => `${i === 0 ? 'M' : 'L'}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' '),
    [points],
  )

  if (points.length === 0) {
    return <p className="empty-state">No ball detections in this session.</p>
  }

  const hovered = hover !== null ? points[hover] : null

  return (
    <div className="chart-wrap">
      <svg className="chart-svg" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Ball trajectory on court">
        <rect x={1} y={1} width={width - 2} height={height - 2} fill="none" stroke="var(--gridline)" rx={6} />
        <line x1={width / 2} y1={0} x2={width / 2} y2={height} stroke="var(--axis)" strokeDasharray="6 8" strokeWidth={2} />
        <path d={path} fill="none" stroke="var(--series-1)" strokeWidth={3} strokeLinecap="round" strokeLinejoin="round" opacity={0.55} />
        {points.map((p, i) => {
          const progress = i / (points.length - 1 || 1)
          const r = i === hover ? 7 : 4
          return (
            <circle
              key={p.frame}
              cx={p.x}
              cy={p.y}
              r={r}
              fill={p.interpolated ? 'var(--surface-1)' : seqColor(progress)}
              stroke={p.interpolated ? 'var(--axis)' : 'none'}
              strokeDasharray={p.interpolated ? '2 2' : undefined}
              strokeWidth={p.interpolated ? 1.5 : 0}
              onMouseEnter={() => setHover(i)}
              onMouseLeave={() => setHover((h) => (h === i ? null : h))}
            />
          )
        })}
      </svg>
      {hovered && (
        <div
          className="tooltip-box"
          style={{ left: `${(hovered.x / width) * 100}%`, top: `${(hovered.y / height) * 100}%` }}
        >
          frame {hovered.frame} · t={hovered.t.toFixed(2)}s
          <br />
          speed {Math.hypot(hovered.vx, hovered.vy).toFixed(0)} px/s
          <br />
          {hovered.interpolated ? 'interpolated' : `conf ${hovered.conf?.toFixed(2)}`}
        </div>
      )}
      <div className="chart-legend">
        <span className="swatch">
          <span className="chip" style={{ background: 'var(--series-1)' }} />
          detected (light → dark = early → late)
        </span>
        <span className="swatch">
          <span className="chip" style={{ background: 'var(--surface-1)', border: '1.5px dashed var(--axis)' }} />
          interpolated (gap-filled)
        </span>
      </div>
    </div>
  )
}
