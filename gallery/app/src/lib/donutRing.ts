/**
 * Placing call-outs (donuts, labels) on a ring of slots around a map, each joined to its site by a
 * leader line: the arrangement of SPARTAN's network composition figure. Sites keep their angular
 * order around the map's centre on the ring, so the leaders do not cross, and among those orders the
 * one with the shortest total leader length wins.
 */
export interface Pt { x: number; y: number }

const angle = (p: Pt, c: Pt) => Math.atan2(p.y - c.y, p.x - c.x)

/**
 * The slot index for each point (slots.length ≥ points.length; spare slots stay empty). Points and
 * slots are both taken in angular order around `centre`; the assignment is the order-preserving one
 * (over every rotation of the ring) with the least total distance.
 */
export function assignRing(points: Pt[], slots: Pt[], centre: Pt): number[] {
  const n = points.length, s = slots.length
  if (!n) return []
  if (s < n) throw new Error(`assignRing: ${n} points need at least ${n} slots, got ${s}`)
  const pOrder = points.map((_, i) => i).sort((a, b) => angle(points[a], centre) - angle(points[b], centre))
  const sOrder = slots.map((_, i) => i).sort((a, b) => angle(slots[a], centre) - angle(slots[b], centre))
  const d = (pi: number, si: number) => Math.hypot(points[pi].x - slots[si].x, points[pi].y - slots[si].y)
  let best = Infinity, bestPick: number[] = []
  for (let rot = 0; rot < s; rot++) {
    const ring = sOrder.map((_, k) => sOrder[(k + rot) % s])
    // cost[i][j]: points 0..i placed, point i on ring slot j (strictly increasing j)
    const cost = Array.from({ length: n }, () => new Float64Array(s).fill(Infinity))
    const from = Array.from({ length: n }, () => new Int32Array(s).fill(-1))
    for (let j = 0; j <= s - n; j++) cost[0][j] = d(pOrder[0], ring[j])
    for (let i = 1; i < n; i++) {
      let runMin = Infinity, runArg = -1
      for (let j = i; j <= s - n + i; j++) {
        if (cost[i - 1][j - 1] < runMin) { runMin = cost[i - 1][j - 1]; runArg = j - 1 }
        cost[i][j] = runMin + d(pOrder[i], ring[j])
        from[i][j] = runArg
      }
    }
    let j = -1, total = Infinity
    for (let k = n - 1; k < s; k++) if (cost[n - 1][k] < total) { total = cost[n - 1][k]; j = k }
    if (total < best) {
      best = total
      const pick = new Array<number>(n)
      for (let i = n - 1; i >= 0; i--) { pick[pOrder[i]] = ring[j]; j = from[i][j] }
      bestPick = pick
    }
  }
  return bestPick
}

/** Slots round a width × height box: a row along the top and bottom, a column down each side. */
export function ringSlots(width: number, height: number, cell: { w: number; h: number }) {
  const across = Math.max(1, Math.floor(width / cell.w))
  const down = Math.max(0, Math.floor((height - 2 * cell.h) / cell.h))
  const slots: (Pt & { side: 'top' | 'bottom' | 'left' | 'right' })[] = []
  for (let k = 0; k < across; k++) {
    const x = (k + 0.5) * width / across
    slots.push({ x, y: cell.h / 2, side: 'top' }, { x, y: height - cell.h / 2, side: 'bottom' })
  }
  const step = down ? (height - 2 * cell.h) / down : 0
  for (let k = 0; k < down; k++) {
    const y = cell.h + (k + 0.5) * step
    slots.push({ x: cell.w / 2, y, side: 'left' }, { x: width - cell.w / 2, y, side: 'right' })
  }
  return slots
}
