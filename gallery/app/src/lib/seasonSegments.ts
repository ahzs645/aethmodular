import { seasonOccurrence, seasonOnly, shortSeason } from './seasons.ts'
import type { SeasonMeta } from './types'

export const SEASON_ORDERS = ['Year, then season', 'Season, then year'] as const
export type SeasonOrder = (typeof SEASON_ORDERS)[number]
interface DatedPoint { d: Date; row: { date: string; season: string } }
export interface SeasonSegment {
  name: string; label: string; year: string; color: string; a: number; b: number; d0: number; d1: number
}

/** Keep whole season occurrences, then arrange their years/seasons without changing the filters. */
export function seasonSegments<T extends DatedPoint>(points: readonly T[], seasons: readonly SeasonMeta[], order: SeasonOrder) {
  const groups = new Map<string, { name: string; year: number; start: string; color: string; pts: T[] }>()
  for (const p of points) {
    const occurrence = seasonOccurrence(p.row.date, seasons)
    const name = seasonOnly(occurrence?.season.name ?? p.row.season)
    const year = occurrence?.year ?? p.d.getFullYear()
    const start = occurrence?.start ?? `${year}-01-01`
    const key = `${name}|${start}`
    if (!groups.has(key)) groups.set(key, { name, year, start, color: occurrence?.season.color ?? '', pts: [] })
    groups.get(key)!.pts.push(p)
  }
  const seasonIndex = (name: string) => {
    const i = seasons.findIndex(s => seasonOnly(s.name) === name)
    return i < 0 ? seasons.length : i
  }
  const sorted = [...groups.values()].sort((a, b) => (order === 'Year, then season'
    ? a.year - b.year || seasonIndex(a.name) - seasonIndex(b.name)
    : seasonIndex(a.name) - seasonIndex(b.name) || a.year - b.year) || a.start.localeCompare(b.start))
  const out: (T & { seg: number })[] = [], segs: SeasonSegment[] = []
  sorted.forEach((g, k) => {
    g.pts.sort((a, b) => +a.d - +b.d)
    segs.push({ name: g.name, label: shortSeason(g.name) || 'No season', year: String(g.year), color: g.color,
      a: out.length, b: out.length + g.pts.length, d0: +g.pts[0].d, d1: +g.pts[g.pts.length - 1].d })
    out.push(...g.pts.map(p => ({ ...p, seg: k })))
  })
  return { points: out, segs }
}

export const seasonParent = (s: SeasonSegment, order: SeasonOrder) => order === 'Year, then season' ? s.year : s.name

/** Date-proportional blocks with a readable minimum; larger gaps separate parent groups. */
export function seasonSegmentLayout(segs: readonly SeasonSegment[], order: SeasonOrder, width: number, minWidth = 100) {
  const gaps: number[] = segs.map((s, i) => !i ? 0 : seasonParent(s, order) === seasonParent(segs[i - 1], order) ? 10 : 24)
  const gapTotal = gaps.reduce((a, b) => a + b, 0)
  const total = Math.max(width, segs.length * minWidth + gapTotal)
  const days = segs.map(s => (s.d1 - s.d0) / 86_400_000 + 1)
  const widths = segs.map(() => 0), pending = new Set(segs.map((_, i) => i))
  let room = total - gapTotal
  while (pending.size) {
    const duration = [...pending].reduce((sum, i) => sum + days[i], 0)
    const small = [...pending].filter(i => room * days[i] / duration < minWidth)
    if (!small.length) {
      for (const i of pending) widths[i] = room * days[i] / duration
      break
    }
    for (const i of small) { widths[i] = minWidth; room -= minWidth; pending.delete(i) }
  }
  let right = 0
  const lefts = widths.map((w, i) => { const left = right + gaps[i]; right = left + w; return left })
  return { width: total, lefts, widths, gaps }
}
