/**
 * Pure helpers behind components/SpectrumPlot and the spec-driven figures:
 * nearest channel on a wavenumber grid (stored ascending or descending),
 * quantile bands in the several layouts the exports use, and per-channel
 * quantiles across a set of traces. No React, so `npm test` runs them.
 */

/** Index of the grid value closest to `v`; the grid may be ascending or descending. */
export function nearestIndex(grid: ArrayLike<number>, v: number): number {
  const n = grid.length
  if (n === 0) return -1
  const desc = n > 1 && grid[0] > grid[n - 1]
  let lo = 0, hi = n - 1
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1
    if (desc ? grid[mid] > v : grid[mid] < v) lo = mid
    else hi = mid
  }
  return Math.abs(grid[lo] - v) <= Math.abs(grid[hi] - v) ? lo : hi
}

/** Quantiles q (0–1) of each channel across traces: out[k][j] is quantile qs[k] of channel j. */
export function channelQuantiles(traces: ArrayLike<number>[], qs: number[]): number[][] {
  const out = qs.map(() => [] as number[])
  if (!traces.length) return out
  const n = traces[0].length
  const col: number[] = []
  for (let j = 0; j < n; j++) {
    col.length = 0
    for (const t of traces) { const v = t[j]; if (v != null && Number.isFinite(v)) col.push(v) }
    col.sort((a, b) => a - b)
    qs.forEach((q, k) => out[k].push(col.length ? quantileSorted(col, q) : NaN))
  }
  return out
}

function quantileSorted(s: number[], q: number): number {
  const i = (s.length - 1) * q, lo = Math.floor(i), hi = Math.ceil(i)
  return s[lo] + (s[hi] - s[lo]) * (i - lo)
}

/**
 * The exports store quantile bands three ways: five rows (10/25/50/75/90 %), three rows
 * (25/50/75 %), or named fields. Returns { lo, mid, hi } for the interquartile band.
 */
export function iqrRows(q: ArrayLike<number>[] | { q25: ArrayLike<number>; median: ArrayLike<number>; q75: ArrayLike<number> }) {
  if (!Array.isArray(q)) return { lo: q.q25, mid: q.median, hi: q.q75 }
  if (q.length === 5) return { lo: q[1], mid: q[2], hi: q[3] }
  if (q.length === 3) return { lo: q[0], mid: q[1], hi: q[2] }
  throw new Error(`iqrRows: expected 3 or 5 quantile rows, got ${q.length}`)
}
