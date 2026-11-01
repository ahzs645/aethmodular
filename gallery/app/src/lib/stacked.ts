/**
 * The arithmetic behind components/StackedPanels, pure so node tests it (lib/stacked.test.mjs): how tall each
 * panel of a stack is (its own height, or its share of a Dashboard tile's fitted height), where each starts, and
 * the runs a line is broken into so a sampling gap is never bridged.
 */

export interface StackSize {
  /** this panel's share of a fitted height, and of `base` at its natural size (default 1) */
  share?: number
  /** a fixed natural height, px (outside a tile); `share` still divides a fitted height */
  height?: number
}
export interface StackGeometry {
  /** each panel's plot height, px */
  heights: number[]
  /** each panel's top, px from the top of the svg */
  tops: number[]
  /** the whole svg's height */
  total: number
}

/**
 * Heights for a stack of panels. Outside a tile (`fit` null) each panel is its `height`, or `base` × its share;
 * in a tile the room left after the margins and gaps is divided by share, never below `min` (a tile too small
 * for every panel scrolls rather than squashing them unreadable).
 */
export function stackGeometry(panels: readonly StackSize[], { fit, base, gap, top, bottom, min = 40 }: {
  fit: number | null; base: number; gap: number; top: number; bottom: number; min?: number
}): StackGeometry {
  const n = panels.length
  const shares = panels.map((p) => (p.share != null && p.share > 0 ? p.share : 1))
  const sum = shares.reduce((a, b) => a + b, 0) || 1
  const room = fit === null ? 0 : fit - top - bottom - Math.max(0, n - 1) * gap
  const heights = panels.map((p, i) => fit === null ? (p.height ?? base * shares[i]) : Math.max(min, (room * shares[i]) / sum))
  const tops: number[] = []
  let y = top
  for (const h of heights) { tops.push(y); y += h + gap }
  const total = top + bottom + heights.reduce((a, b) => a + b, 0) + Math.max(0, n - 1) * gap
  return { heights, tops, total }
}

/**
 * The runs of a dated series a line joins: a missing or non-finite value ends a run, and so does a step of more
 * than `maxGapDays` between neighbours, so a gap in the record shows as a break, never a straight bridge.
 * A run of one point is a point the chart draws as a dot.
 */
export function lineRuns<T>(pts: readonly T[], dateOf: (p: T) => Date, valueOf: (p: T) => number | null | undefined, maxGapDays = Infinity): { t: Date; v: number; p: T }[][] {
  const out: { t: Date; v: number; p: T }[][] = []
  let cur: { t: Date; v: number; p: T }[] = []
  for (const p of pts) {
    const v = valueOf(p)
    if (v == null || !Number.isFinite(v)) { if (cur.length) out.push(cur); cur = []; continue }
    const t = dateOf(p)
    if (cur.length && +t - +cur[cur.length - 1].t > maxGapDays * 864e5) { out.push(cur); cur = [] }
    cur.push({ t, v, p })
  }
  if (cur.length) out.push(cur)
  return out
}

/** the index of the item nearest `at` in items sorted by `valueOf` (d3.bisector().center without d3), or -1 */
export function nearestIndex<T>(items: readonly T[], valueOf: (t: T) => number, at: number): number {
  if (!items.length || !Number.isFinite(at)) return -1
  let lo = 0, hi = items.length - 1
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if (valueOf(items[mid]) < at) lo = mid + 1
    else hi = mid
  }
  return lo > 0 && Math.abs(valueOf(items[lo - 1]) - at) <= Math.abs(valueOf(items[lo]) - at) ? lo - 1 : lo
}
