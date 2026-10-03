import type { Session } from '../../models/types'
import { StatusBadge } from './StatusBadge'

interface Props {
  sessions: Session[]
  selectedId: string | null
  onSelect: (id: string) => void
}

export function SessionList({ sessions, selectedId, onSelect }: Props) {
  if (sessions.length === 0) {
    return <p className="empty-state">No sessions recorded yet.</p>
  }
  return (
    <ul className="session-list">
      {sessions.map((s) => (
        <li key={s.id}>
          <button
            type="button"
            className="session-item"
            data-active={s.id === selectedId}
            onClick={() => onSelect(s.id)}
          >
            <span className="src">{s.source}</span>
            <span className="meta">
              {s.frameCount.toLocaleString()} frames · {new Date(s.startedAt).toLocaleDateString()}
            </span>
            <StatusBadge status={s.status} />
          </button>
        </li>
      ))}
    </ul>
  )
}
