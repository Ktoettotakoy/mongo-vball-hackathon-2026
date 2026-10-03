// Model layer: shapes mirroring the MongoDB schema written by vbtrack/storage.py

export type SessionStatus = 'running' | 'done' | 'stopped' | 'stream_lost'

export interface Session {
  id: string
  source: string
  live: boolean
  fps: number
  width: number
  height: number
  startedAt: string
  endedAt: string | null
  status: SessionStatus
  frameCount: number
  playerWeights: string
  ballWeights: string | null
  tracker: string
  stride: number
}

export interface BallPoint {
  frame: number
  t: number
  x: number
  y: number
  vx: number
  vy: number
  conf: number | null
  interpolated: boolean
}

export interface PlayerTrack {
  trackId: number
  firstFrame: number
  lastFrame: number
  firstT: number
  lastT: number
  framesSeen: number
  avgConf: number
  avgFoot: { x: number; y: number }
  // Real field as of the jersey-reader pipeline update: voted jersey number,
  // or null when it never got a confident read (see vbtrack/jersey.py).
  number: string | null
}

// Action-recognition events (serve/spike/block). Matches the shape returned
// by the action-detection model: player_no is null until jersey voting is
// wired into that model's output, so UI falls back to trackId for display.
export type ActionType = 'serve' | 'spike' | 'block'

export interface ActionEvent {
  playerNo: number | null
  trackId: string // e.g. "#19"
  action: ActionType
  successful: boolean
  blockType?: 'full' | 'solo'
}

// Rally-level event log. Not produced by the current pipeline either - it
// needs rally/touch-sequence + team-side detection on top of the ball
// tracker. Mocked to demo the "broadcast scoreboard" view on the frontend.
export type RallyOutcome = 'kill' | 'attack_out' | 'net_fault' | 'serve_error' | 'ball_handling'

export interface Team {
  code: string
  color: string
}

export interface RallyEvent {
  index: number
  tStart: number // seconds into the clip
  team: string // team code that WON the point
  outcome: RallyOutcome
  durationS: number
  touches: number
  scoreAfter: { a: number; b: number }
}

export interface RallyLog {
  teamA: Team
  teamB: Team
  rallies: RallyEvent[]
}

// One row per track_id, aggregated in MongoDB (vbtrack/stats.py, GET /api/players).
export interface PlayerStats {
  trackId: string
  playerNo: number | null
  videos: string[]
  total: number
  successful: number
  serves: number
  servesWon: number
  spikes: number
  kills: number
  blocks: number
  blocksWon: number
  fullBlocks: number
  soloBlocks: number
  successPct: number
}

export interface SessionDetail {
  session: Session
  ball: BallPoint[]
  tracks: PlayerTrack[]
  rallyLog: RallyLog
  actions: ActionEvent[]
  // null when the API is down: the UI then aggregates the mock actions itself.
  playerStats: PlayerStats[] | null
  actionsSource: 'mongo' | 'mock'
}
