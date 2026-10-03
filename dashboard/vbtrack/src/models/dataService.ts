// Model layer: the only module the controllers talk to. Sessions, ball and
// tracks are still mock data. Action events and per-player stats come from
// the vbtrack API (MongoDB) and fall back to mock data when the API is down.

import { generateActionEvents, generateBallTrajectory, generatePlayerTracks, generateRallyLog, getSession, listSessions } from './mockData'
import type { ActionEvent, ActionType, PlayerStats, Session, SessionDetail } from './types'

const SIMULATED_LATENCY_MS = 220

function delay<T>(value: T): Promise<T> {
  return new Promise((resolve) => setTimeout(() => resolve(value), SIMULATED_LATENCY_MS))
}

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(path)
  if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`)
  return (await res.json()) as T
}

// --- API wire shapes (snake_case, as stored in MongoDB) ---
interface ApiEvent {
  player_no: number | null
  track_id: string
  action: ActionType
  successful: boolean
  block_type?: 'full' | 'solo' | null
}

interface ApiPlayer {
  track_id: string
  player_no: number | null
  videos: string[]
  total: number
  successful: number
  serves: number
  serves_won: number
  spikes: number
  kills: number
  blocks: number
  blocks_won: number
  full_blocks: number
  solo_blocks: number
  success_pct: number
}

function toAction(e: ApiEvent): ActionEvent {
  const a: ActionEvent = { playerNo: e.player_no, trackId: e.track_id, action: e.action, successful: e.successful }
  if (e.block_type) a.blockType = e.block_type
  return a
}

function toStats(p: ApiPlayer): PlayerStats {
  return {
    trackId: p.track_id,
    playerNo: p.player_no,
    videos: p.videos,
    total: p.total,
    successful: p.successful,
    serves: p.serves,
    servesWon: p.serves_won,
    spikes: p.spikes,
    kills: p.kills,
    blocks: p.blocks,
    blocksWon: p.blocks_won,
    fullBlocks: p.full_blocks,
    soloBlocks: p.solo_blocks,
    successPct: p.success_pct,
  }
}

export async function fetchPlayerStats(video?: string): Promise<PlayerStats[]> {
  const q = video ? `?video=${encodeURIComponent(video)}` : ''
  return (await getJson<ApiPlayer[]>(`/api/players${q}`)).map(toStats)
}

export async function fetchActions(video?: string): Promise<ActionEvent[]> {
  const q = video ? `?video=${encodeURIComponent(video)}` : ''
  return (await getJson<ApiEvent[]>(`/api/events${q}`)).map(toAction)
}

export function fetchSessions(): Promise<Session[]> {
  return delay(listSessions())
}

export async function fetchSessionDetail(sessionId: string): Promise<SessionDetail> {
  const session = getSession(sessionId)
  if (!session) throw new Error(`Unknown session: ${sessionId}`)
  const ball = generateBallTrajectory(sessionId, session.width, session.height)
  const tracks = generatePlayerTracks(sessionId, session.width, session.height, session.frameCount)
  const rallyLog = generateRallyLog(sessionId)

  try {
    const [actions, playerStats] = await Promise.all([fetchActions(), fetchPlayerStats()])
    return { session, ball, tracks, rallyLog, actions, playerStats, actionsSource: 'mongo' }
  } catch (err) {
    console.warn('vbtrack API unavailable, using mock actions:', err)
    const actions = generateActionEvents(sessionId, tracks)
    return delay({ session, ball, tracks, rallyLog, actions, playerStats: null, actionsSource: 'mock' })
  }
}
