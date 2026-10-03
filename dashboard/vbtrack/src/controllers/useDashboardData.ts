// Controller layer: owns state + orchestration, wires the Model (dataService)
// to the Views without the views touching data-fetching directly.

import { useEffect, useState } from 'react'
import { fetchSessionDetail, fetchSessions } from '../models/dataService'
import type { Session, SessionDetail } from '../models/types'

interface DashboardState {
  sessions: Session[]
  selectedSessionId: string | null
  detail: SessionDetail | null
  loadingSessions: boolean
  loadingDetail: boolean
  error: string | null
  selectSession: (id: string) => void
}

export function useDashboardData(): DashboardState {
  const [sessions, setSessions] = useState<Session[]>([])
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null)
  const [detail, setDetail] = useState<SessionDetail | null>(null)
  const [loadingSessions, setLoadingSessions] = useState(true)
  const [loadingDetail, setLoadingDetail] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    fetchSessions()
      .then((list) => {
        if (cancelled) return
        setSessions(list)
        setSelectedSessionId((current) => current ?? list[0]?.id ?? null)
      })
      .catch((err: Error) => !cancelled && setError(err.message))
      .finally(() => !cancelled && setLoadingSessions(false))
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    if (!selectedSessionId) return
    let cancelled = false
    setLoadingDetail(true)
    fetchSessionDetail(selectedSessionId)
      .then((d) => !cancelled && setDetail(d))
      .catch((err: Error) => !cancelled && setError(err.message))
      .finally(() => !cancelled && setLoadingDetail(false))
    return () => {
      cancelled = true
    }
  }, [selectedSessionId])

  return {
    sessions,
    selectedSessionId,
    detail,
    loadingSessions,
    loadingDetail,
    error,
    selectSession: setSelectedSessionId,
  }
}
