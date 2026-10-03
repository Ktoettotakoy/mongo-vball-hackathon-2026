// Model layer: deterministic sample data standing in for a real vbtrack Mongo
// deployment, shaped exactly like vbtrack/storage.py's sessions/frames/tracks
// collections. Swap this module for a real fetch-based one to go live later.

import type { ActionEvent, ActionType, BallPoint, PlayerTrack, RallyEvent, RallyLog, RallyOutcome, Session, SessionStatus, Team } from './types'

// Mulberry32 PRNG so the demo renders identically on every reload.
function mulberry32(seed: number) {
  let a = seed
  return () => {
    a |= 0
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

const SOURCES = ['match_set1.mp4', 'match_set2.mp4', 'rtsp://court-cam-1/stream1', 'scrimmage_0921.mp4', 'match_set3.mp4']
const STATUSES: SessionStatus[] = ['done', 'done', 'running', 'stream_lost', 'stopped']

const SESSION_META = SOURCES.map((source, i) => {
  const rand = mulberry32(1000 + i)
  const fps = 30
  const durationMin = 8 + rand() * 20
  const frameCount = Math.round(durationMin * 60 * fps)
  const startedAt = new Date(Date.now() - (SOURCES.length - i) * 86_400_000 - rand() * 3_600_000)
  const live = source.startsWith('rtsp://')
  return {
    id: `sess_${(i + 1).toString().padStart(3, '0')}`,
    source,
    live,
    fps,
    width: 1920,
    height: 1080,
    startedAt: startedAt.toISOString(),
    endedAt: STATUSES[i] === 'running' ? null : new Date(startedAt.getTime() + durationMin * 60_000).toISOString(),
    status: STATUSES[i],
    frameCount,
    playerWeights: 'yolo11n.pt',
    ballWeights: i % 2 === 0 ? 'volleyball_best.pt' : null,
    tracker: i % 3 === 0 ? 'botsort.yaml' : 'bytetrack.yaml',
    stride: 1,
  } satisfies Session
})

export function listSessions(): Session[] {
  return SESSION_META
}

export function getSession(id: string): Session | undefined {
  return SESSION_META.find((s) => s.id === id)
}

// --- Ball trajectory: a simulated rally, bouncing between two sides of the
// court with a parabolic arc per touch, occasional short interpolated gaps. ---
export function generateBallTrajectory(sessionId: string, width: number, height: number): BallPoint[] {
  const rand = mulberry32(hashCode(sessionId) ^ 0x42)
  const fps = 30
  const net = width / 2
  const points: BallPoint[] = []
  let side: 1 | -1 = 1
  let frame = 0
  let x = net
  let y = height * 0.5

  for (let touch = 0; touch < 10; touch++) {
    const targetX = side > 0 ? net + 150 + rand() * (width / 2 - 250) : net - 150 - rand() * (width / 2 - 250)
    const peakHeight = height * (0.15 + rand() * 0.25)
    const touchFrames = Math.round(18 + rand() * 24)
    const startX = x
    const startY = y
    for (let i = 1; i <= touchFrames; i++) {
      const p = i / touchFrames
      const px = startX + (targetX - startX) * p
      // simple parabolic arc for y (screen coords: smaller y = higher)
      const arc = 4 * peakHeight * p * (1 - p)
      const py = startY + (height * 0.6 - startY) * p - arc
      const interpolated = rand() < 0.08
      const vx = ((targetX - startX) / touchFrames) * fps
      const vy = i === 1 ? 0 : (py - points[points.length - 1].y) * fps
      points.push({
        frame,
        t: Math.round((frame / fps) * 1000) / 1000,
        x: Math.round(px * 10) / 10,
        y: Math.round(py * 10) / 10,
        vx: Math.round(vx * 100) / 100,
        vy: Math.round(vy * 100) / 100,
        conf: interpolated ? null : Math.round((0.4 + rand() * 0.55) * 1000) / 1000,
        interpolated,
      })
      frame++
    }
    x = targetX
    y = height * 0.6
    side = side > 0 ? -1 : 1
  }
  return points
}

// --- Player tracks: six players, three per side, random-walking inside
// their own half for the session's duration. ---
export function generatePlayerTracks(sessionId: string, width: number, height: number, frameCount: number): PlayerTrack[] {
  const rand = mulberry32(hashCode(sessionId) ^ 0x1337)
  const tracks: PlayerTrack[] = []
  for (let i = 0; i < 6; i++) {
    const side = i < 3 ? -1 : 1
    const laneX = width / 2 + side * (200 + (i % 3) * 260)
    const laneY = height * (0.3 + ((i % 3) * 0.22))
    const framesSeen = Math.round(frameCount * (0.7 + rand() * 0.28))
    const firstFrame = Math.round(rand() * (frameCount - framesSeen))
    // Jersey numbers are often unreadable (back turned, far away, blur) -
    // per vbtrack/jersey.py only ~half the tracks get a confident vote.
    const number = rand() < 0.5 ? String(1 + Math.floor(rand() * 18)) : null
    tracks.push({
      trackId: i + 1,
      firstFrame,
      lastFrame: firstFrame + framesSeen,
      firstT: Math.round((firstFrame / 30) * 100) / 100,
      lastT: Math.round(((firstFrame + framesSeen) / 30) * 100) / 100,
      framesSeen,
      avgConf: Math.round((0.72 + rand() * 0.24) * 1000) / 1000,
      avgFoot: {
        x: Math.round((laneX + (rand() - 0.5) * 120) * 10) / 10,
        y: Math.round((laneY + (rand() - 0.5) * 160) * 10) / 10,
      },
      number,
    })
  }
  return tracks.sort((a, b) => b.framesSeen - a.framesSeen)
}

// --- Action events: serve/spike/block recognitions, matching the shape the
// action-recognition model returns. player_no stays null here since jersey
// voting isn't wired into that model's output yet - same as the real sample. ---
const ACTION_WEIGHTS: Array<{ action: ActionType; weight: number }> = [
  { action: 'serve', weight: 0.2 },
  { action: 'spike', weight: 0.55 },
  { action: 'block', weight: 0.25 },
]

function pickAction(rand: () => number): ActionType {
  const r = rand()
  let acc = 0
  for (const { action, weight } of ACTION_WEIGHTS) {
    acc += weight
    if (r < acc) return action
  }
  return 'spike'
}

export function generateActionEvents(sessionId: string, tracks: PlayerTrack[]): ActionEvent[] {
  const rand = mulberry32(hashCode(sessionId) ^ 0x0ac7)
  if (tracks.length === 0) return []
  const events: ActionEvent[] = []

  for (let i = 0; i < 28; i++) {
    const track = tracks[Math.floor(rand() * tracks.length)]
    const action = pickAction(rand)
    const successful =
      action === 'serve' ? rand() < 0.85 : action === 'block' ? rand() < 0.6 : rand() < 0.48
    const event: ActionEvent = {
      playerNo: null,
      trackId: `#${track.trackId}`,
      action,
      successful,
    }
    if (action === 'block') event.blockType = rand() < 0.75 ? 'full' : 'solo'
    events.push(event)
  }
  return events
}

// --- Rally log: a broadcast-style scoreboard for the clip - which team won
// each rally, how (kill vs. unforced error), how long it ran, how many
// touches. Fixed at 24 rallies per session, like "first 24 rallies of a
// simulated set" in the reference design. ---
const TEAM_A: Team = { code: 'TCD', color: '#8b93f5' }
const TEAM_B: Team = { code: 'UCC', color: '#f16c63' }
const ERROR_OUTCOMES: RallyOutcome[] = ['attack_out', 'net_fault', 'serve_error', 'ball_handling']

export function generateRallyLog(sessionId: string): RallyLog {
  const rand = mulberry32(hashCode(sessionId) ^ 0x5a11)
  let a = 0
  let b = 0
  let t = 4 + rand() * 6
  const rallies: RallyEvent[] = []

  for (let i = 0; i < 24; i++) {
    const team = rand() < 0.5 ? TEAM_A.code : TEAM_B.code
    const isError = rand() < 0.33
    const outcome: RallyOutcome = isError ? ERROR_OUTCOMES[Math.floor(rand() * ERROR_OUTCOMES.length)] : 'kill'
    const touches = isError ? 2 + Math.round(rand() * 4) : 3 + Math.round(rand() * 5)
    const durationS = Math.round((2 + touches * 0.6 + rand() * 1.5) * 10) / 10
    if (team === TEAM_A.code) a++
    else b++
    rallies.push({ index: i + 1, tStart: Math.round(t * 10) / 10, team, outcome, durationS, touches, scoreAfter: { a, b } })
    t += durationS + 3 + rand() * 12
  }

  return { teamA: TEAM_A, teamB: TEAM_B, rallies }
}

function hashCode(s: string): number {
  let h = 0
  for (let i = 0; i < s.length; i++) h = (Math.imul(31, h) + s.charCodeAt(i)) | 0
  return h
}
