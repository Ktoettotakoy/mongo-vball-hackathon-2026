import { useMemo, useState } from 'react'
import type { RallyLog as RallyLogData, RallyOutcome } from '../../models/types'

interface Props {
  rallyLog: RallyLogData
  ballTrackedPct: number | null
}

const OUTCOME_LABEL: Record<RallyOutcome, string> = {
  kill: 'Kill',
  attack_out: 'Attack out',
  net_fault: 'Net fault',
  serve_error: 'Serve error',
  ball_handling: 'Ball handling',
}

const BUCKETS = [
  { label: '0-3', test: (d: number) => d < 3 },
  { label: '3-6', test: (d: number) => d >= 3 && d < 6 },
  { label: '6-9', test: (d: number) => d >= 6 && d < 9 },
  { label: '9-12', test: (d: number) => d >= 9 && d < 12 },
  { label: '12+', test: (d: number) => d >= 12 },
]

function formatClock(seconds: number): string {
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${m}:${s.toString().padStart(2, '0')}`
}

export function RallyLog({ rallyLog, ballTrackedPct }: Props) {
  const { teamA, teamB, rallies } = rallyLog
  const [filter, setFilter] = useState<'all' | 'a' | 'b' | 'long'>('all')

  const stats = useMemo(() => {
    const total = rallies.length
    const avg = rallies.reduce((s, r) => s + r.durationS, 0) / Math.max(1, total)
    const longest = rallies.reduce((m, r) => Math.max(m, r.durationS), 0)
    const errors = rallies.filter((r) => r.outcome !== 'kill').length
    const last = rallies[rallies.length - 1]?.scoreAfter ?? { a: 0, b: 0 }
    const buckets = BUCKETS.map((b) => ({ label: b.label, count: rallies.filter((r) => b.test(r.durationS)).length }))
    const maxBucket = Math.max(1, ...buckets.map((b) => b.count))
    return { total, avg, longest, errors, last, buckets, maxBucket }
  }, [rallies])

  const filtered = useMemo(() => {
    if (filter === 'a') return rallies.filter((r) => r.team === teamA.code)
    if (filter === 'b') return rallies.filter((r) => r.team === teamB.code)
    if (filter === 'long') return rallies.filter((r) => r.durationS >= 8)
    return rallies
  }, [rallies, filter, teamA.code, teamB.code])

  return (
    <section className="rally-log">
      <div className="rally-section rally-score-head">
        <p className="eyebrow">Score at the end of the clip</p>
        <div className="rally-score-row">
          <div className="team-block">
            <span className="team-name">
              <span className="dot" style={{ background: teamA.color }} />
              {teamA.code}
            </span>
            <div className="score">{stats.last.a}</div>
          </div>
          <span className="vs">v</span>
          <div className="team-block right">
            <span className="team-name">
              {teamB.code}
              <span className="dot" style={{ background: teamB.color }} />
            </span>
            <div className="score">{stats.last.b}</div>
          </div>
        </div>
        <p className="rally-meta">
          {stats.total} rallies · first {stats.total} rallies of a simulated set
        </p>
      </div>

      <div className="rally-section rally-stat-row">
        <div>
          <p className="label">Average rally</p>
          <p className="value">
            {stats.avg.toFixed(1)}
            <span className="unit">s</span>
          </p>
        </div>
        <div>
          <p className="label">Longest rally</p>
          <p className="value">
            {stats.longest.toFixed(1)}
            <span className="unit">s</span>
          </p>
        </div>
        <div>
          <p className="label">Ball tracked</p>
          <p className="value">
            {ballTrackedPct ?? '—'}
            <span className="unit">% of frames</span>
          </p>
        </div>
        <div>
          <p className="label">Points from errors</p>
          <p className="value">
            {stats.errors}
            <span className="unit"> of {stats.total}</span>
          </p>
        </div>
      </div>

      <div className="rally-section">
        <p className="eyebrow">Rally length, seconds</p>
        <div className="hist-bars">
          {stats.buckets.map((b) => (
            <div className="hist-col" key={b.label}>
              <span className="hist-count">{b.count}</span>
              <div className="hist-bar" style={{ height: `${Math.max(6, (b.count / stats.maxBucket) * 90)}px` }} />
              <span className="hist-label">{b.label}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="rally-section rally-filters">
        <button type="button" data-active={filter === 'all'} onClick={() => setFilter('all')}>
          All
        </button>
        <button type="button" data-active={filter === 'a'} onClick={() => setFilter('a')}>
          <span className="dot" style={{ background: teamA.color }} />
          {teamA.code}
        </button>
        <button type="button" data-active={filter === 'b'} onClick={() => setFilter('b')}>
          <span className="dot" style={{ background: teamB.color }} />
          {teamB.code}
        </button>
        <button type="button" data-active={filter === 'long'} onClick={() => setFilter('long')}>
          Long, 8s+
        </button>
      </div>

      <ul className="rally-list">
        {filtered.map((r) => {
          const team = r.team === teamA.code ? teamA : teamB
          return (
            <li className="rally-row" key={r.index}>
              <span className="rally-idx">R{r.index.toString().padStart(2, '0')}</span>
              <span className="rally-t">{formatClock(r.tStart)}</span>
              <span className="team-tag" style={{ background: team.color }}>
                {team.code}
              </span>
              <span className="rally-outcome">
                {OUTCOME_LABEL[r.outcome]} · {r.durationS.toFixed(1)} s
              </span>
              <span className="rally-score-cell">
                {r.scoreAfter.a}–{r.scoreAfter.b}
                <br />
                <small>{r.touches} touches</small>
              </span>
            </li>
          )
        })}
        {filtered.length === 0 && <li className="empty-state">No rallies match this filter.</li>}
      </ul>
    </section>
  )
}
