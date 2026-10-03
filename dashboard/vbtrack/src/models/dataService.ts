// Model layer: the only module the controllers talk to. Today it resolves
// mock data; pointing it at a real vbtrack API later (e.g. wrapping
// vbtrack/queries.py behind FastAPI) means rewriting this file only.

import { generateActionEvents, generateBallTrajectory, generatePlayerTracks, generateRallyLog, getSession, listSessions } from './mockData'
import type { Session, SessionDetail } from './types'

const SIMULATED_LATENCY_MS = 220

function delay<T>(value: T): Promise<T> {
  return new Promise((resolve) => setTimeout(() => resolve(value), SIMULATED_LATENCY_MS))
}

export function fetchSessions(): Promise<Session[]> {
  return delay(listSessions())
}

export function fetchSessionDetail(sessionId: string): Promise<SessionDetail> {
  const session = getSession(sessionId)
  if (!session) return Promise.reject(new Error(`Unknown session: ${sessionId}`))
  const ball = generateBallTrajectory(sessionId, session.width, session.height)
  const tracks = generatePlayerTracks(sessionId, session.width, session.height, session.frameCount)
  const rallyLog = generateRallyLog(sessionId)
  const actions = generateActionEvents(sessionId, tracks)
  return delay({ session, ball, tracks, rallyLog, actions })
}
