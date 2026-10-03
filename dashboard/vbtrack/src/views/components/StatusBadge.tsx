import type { SessionStatus } from '../../models/types'

const STATUS_META: Record<SessionStatus, { color: string; label: string }> = {
  done: { color: 'var(--status-good)', label: 'Done' },
  running: { color: 'var(--status-warning)', label: 'Running' },
  stopped: { color: 'var(--status-serious)', label: 'Stopped' },
  stream_lost: { color: 'var(--status-critical)', label: 'Stream lost' },
}

export function StatusBadge({ status }: { status: SessionStatus }) {
  const meta = STATUS_META[status]
  return (
    <span className="status-badge" style={{ color: meta.color }}>
      <span className="dot" style={{ background: meta.color }} />
      {meta.label}
    </span>
  )
}
