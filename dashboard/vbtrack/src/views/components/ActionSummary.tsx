import type { ActionEvent } from '../../models/types'

interface Row {
  trackId: string
  serves: number
  spikes: number
  kills: number
  blocks: number
  blocksWon: number
  total: number
  successful: number
}

function aggregate(actions: ActionEvent[]): Row[] {
  const byTrack = new Map<string, Row>()
  for (const a of actions) {
    const row = byTrack.get(a.trackId) ?? {
      trackId: a.trackId,
      serves: 0,
      spikes: 0,
      kills: 0,
      blocks: 0,
      blocksWon: 0,
      total: 0,
      successful: 0,
    }
    row.total += 1
    if (a.successful) row.successful += 1
    if (a.action === 'serve') row.serves += 1
    if (a.action === 'spike') {
      row.spikes += 1
      if (a.successful) row.kills += 1
    }
    if (a.action === 'block') {
      row.blocks += 1
      if (a.successful) row.blocksWon += 1
    }
    byTrack.set(a.trackId, row)
  }
  return [...byTrack.values()].sort((a, b) => b.kills - a.kills || b.total - a.total)
}

export function ActionSummary({ actions }: { actions: ActionEvent[] }) {
  if (actions.length === 0) {
    return <p className="empty-state">No actions recognized in this session.</p>
  }
  const rows = aggregate(actions)

  return (
    <table className="ak-table">
      <thead>
        <tr>
          <th>Player</th>
          <th>Serves</th>
          <th>Spikes (kills)</th>
          <th>Blocks</th>
          <th>Success %</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.trackId}>
            <td>
              <span className="ak-player">{r.trackId}</span>
            </td>
            <td>{r.serves}</td>
            <td>
              {r.spikes} ({r.kills})
            </td>
            <td>
              {r.blocks} ({r.blocksWon})
            </td>
            <td>{Math.round((r.successful / r.total) * 100)}%</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}
