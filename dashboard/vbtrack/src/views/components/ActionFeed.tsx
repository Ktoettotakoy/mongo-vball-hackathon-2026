import type { ActionEvent } from '../../models/types'

const ACTION_LABEL: Record<ActionEvent['action'], string> = {
  serve: 'Serve',
  spike: 'Spike',
  block: 'Block',
}

export function ActionFeed({ actions }: { actions: ActionEvent[] }) {
  if (actions.length === 0) {
    return <p className="empty-state">No actions recognized in this session.</p>
  }

  return (
    <ul className="action-feed">
      {actions.map((a, i) => (
        <li className="action-feed-row" key={i}>
          <span className="action-feed-player">{a.playerNo !== null ? `#${a.playerNo}` : a.trackId}</span>
          <span className="action-feed-type">
            {ACTION_LABEL[a.action]}
            {a.blockType && ` (${a.blockType})`}
          </span>
          <span className={`action-feed-result ${a.successful ? 'ok' : 'fail'}`}>
            {a.successful ? 'Successful' : 'Unsuccessful'}
          </span>
        </li>
      ))}
    </ul>
  )
}
