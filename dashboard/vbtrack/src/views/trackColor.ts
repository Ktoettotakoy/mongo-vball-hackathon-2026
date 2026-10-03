const SLOT_VARS = [
  'var(--series-1)', 'var(--series-2)', 'var(--series-3)', 'var(--series-4)',
  'var(--series-5)', 'var(--series-6)', 'var(--series-7)', 'var(--series-8)',
]

// Color follows the player's track_id (identity), not their rank in a list,
// so a re-sort never repaints which player owns which color.
export function trackColor(trackId: number): string {
  return SLOT_VARS[(trackId - 1) % SLOT_VARS.length]
}
