import type { ActionEvent, PlayerStats } from '../../models/types'

interface Row {
  trackId: string
  playerNo: number | null
  serves: number
  servesWon: number
  spikes: number
  kills: number
  blocks: number
  blocksWon: number
  total: number
  successful: number
}

// Client-side fallback, used only when the API (MongoDB aggregate) is down.
function aggregate(actions: ActionEvent[]): Row[] {
  const byTrack = new Map<string, Row>()
  for (const a of actions) {
    const row = byTrack.get(a.trackId) ?? {
      trackId: a.trackId,
      playerNo: null,
      serves: 0,
      servesWon: 0,
      spikes: 0,
      kills: 0,
      blocks: 0,
      blocksWon: 0,
      total: 0,
      successful: 0,
    }
    row.total += 1
    if (a.playerNo !== null) row.playerNo = a.playerNo
    if (a.successful) row.successful += 1
    if (a.action === 'serve') {
      row.serves += 1
      if (a.successful) row.servesWon += 1
    }
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

interface Props {
  actions: ActionEvent[]
  // Rows aggregated per track_id in MongoDB; null = aggregate `actions` here.
  stats?: PlayerStats[] | null
}

export function ActionSummary({ actions, stats }: Props) {
  const rows: Row[] = stats ?? aggregate(actions)
  if (rows.length === 0) {
    return <p className="empty-state">No actions recognized in this session.</p>
  }

  return (
    <table className="ak-table">
      <thead>
        <tr>
          <th>Player</th>
          <th>Serves (in)</th>
          <th>Spikes (kills)</th>
          <th>Blocks (won)</th>
          <th>Success %</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.trackId}>
            <td>
              <span className="ak-player">{r.trackId}</span>
              {r.playerNo !== null && <small> · no. {r.playerNo}</small>}
            </td>
            <td>
              {r.serves} ({r.servesWon})
            </td>
            <td>
              {r.spikes} ({r.kills})
            </td>
            <td>
              {r.blocks} ({r.blocksWon})
            </td>
            <td>{r.total ? Math.round((r.successful / r.total) * 100) : 0}%</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}
