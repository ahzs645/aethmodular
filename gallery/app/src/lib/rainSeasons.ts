/**
 * Rain-rule seasons: each site's seasons, year by year, from its own daily rain. This is a gallery rule, not
 * an algorithm the Addis PMF paper publishes: the paper says only that its 2023 seasons were set "using
 * precipitation data and following the seasonal representation used in Ethiopia", and the thresholds below
 * were fitted to reproduce its dates. It follows the dynamic seasons of addis_03_meteorology (Task 3, a
 * rolling rain sum against thresholds).
 *
 * For each calendar year, on the rolling mean of daily rain (`window` days, centred):
 *   rains begin     first day the mean exceeds `onset` mm/day          -> Less Rainy
 *   heavy rains     first day after that it exceeds `heavy` mm/day     -> More Rainy
 *   rains end       the day after the last day above `heavy`           -> Second Dry
 * Before the rains begin is First Dry. A year whose mean never passes `heavy` has no More Rainy:
 * Less Rainy runs to the last day above `onset`; a year without rains in the record is First Dry
 * (Second Dry if its record only starts in the second half, e.g. a December start).
 *
 * Seasons cover only days the rolling mean was computed from rain (at least half the window present):
 * missing rain is never read as a dry day, and the seasons stop where the record does. A filter outside
 * them has no rain-rule season (App shows it as "No weather").
 *
 * Defaults (30 days, 1.5, 9 mm/day) were fitted to the paper's 2023 dates on ERA5 at Addis:
 * rains begin 26 Feb (paper 26 Feb), heavy 25 Apr (29 Apr), end 26 Sep (1 Oct).
 */
import type { SeasonMeta } from './types'

export interface RainRule { window: number; onset: number; heavy: number }
export const DEFAULT_RAIN_RULE: RainRule = { window: 30, onset: 1.5, heavy: 9 }

/** the four seasons' names, colours and (for month axes only) their usual months at Addis */
export const RAIN_SEASONS = [
  { name: 'First Dry', color: '#f8cfa5', months: [1, 2] },
  { name: 'Less Rainy', color: '#a8d8e8', months: [3, 4] },
  { name: 'More Rainy', color: '#5aaee8', months: [5, 6, 7, 8, 9] },
  { name: 'Second Dry', color: '#f09a5a', months: [10, 11, 12] },
] as const

export const rainRuleParam = (r: RainRule) => `${r.window},${r.onset},${r.heavy}`
export function parseRainRule(v: string | null): RainRule | null {
  const [w, o, h] = (v ?? '').split(',').map(Number)
  return Number.isFinite(w) && w >= 3 && Number.isFinite(o) && o >= 0 && Number.isFinite(h) && h > o ? { window: Math.round(w), onset: o, heavy: h } : null
}

const iso = (t: number) => new Date(t).toISOString().slice(0, 10)
const DAY = 86_400_000

/**
 * The rule's seasons for one site: the four seasons, each with `spans` ([start, end) ISO dates),
 * one set per calendar year in the daily record.
 */
export function rainRuleCalendar(daily: { date: string[]; p: (number | null)[] }, rule: RainRule = DEFAULT_RAIN_RULE): SeasonMeta[] {
  const n = daily.date.length
  const t = daily.date.map((d) => Date.parse(`${d}T00:00:00Z`))
  // centred rolling mean, needing half the window present
  const half = Math.floor(rule.window / 2)
  const mean: (number | null)[] = new Array(n).fill(null)
  for (let i = 0; i < n; i++) {
    let s = 0, k = 0
    for (let j = Math.max(0, i - half); j <= Math.min(n - 1, i + half - (rule.window % 2 ? 0 : 1)); j++) {
      const v = daily.p[j]
      if (v != null && Number.isFinite(v)) { s += v; k++ }
    }
    mean[i] = k >= rule.window / 2 ? s / k : null
  }
  // the days the rule can see: runs of consecutive dates with a rolling mean, as [start, end) times
  const seen: [number, number][] = []
  for (let i = 0; i < n; i++) {
    if (mean[i] == null) continue
    const last = seen[seen.length - 1]
    if (last && last[1] === t[i]) last[1] = t[i] + DAY
    else seen.push([t[i], t[i] + DAY])
  }
  const spans: [string, string][][] = RAIN_SEASONS.map(() => [])
  const add = (k: number, a: number, b: number) => {
    for (const [s0, s1] of seen) {
      const lo = Math.max(a, s0), hi = Math.min(b, s1)
      if (lo < hi) spans[k].push([iso(lo), iso(hi)])
    }
  }
  const years = [...new Set(daily.date.map((d) => d.slice(0, 4)))]
  for (const y of years) {
    const idx = daily.date.map((d, i) => (d.startsWith(y) && mean[i] != null ? i : -1)).filter((i) => i >= 0)
    if (!idx.length) continue          // no rain for this year: no season, not a dry one
    const y0 = Date.parse(`${y}-01-01T00:00:00Z`), y1 = Date.parse(`${Number(y) + 1}-01-01T00:00:00Z`)
    const above = (thr: number, from: number) => idx.filter((i) => t[i] >= from && mean[i]! > thr)
    const wet = above(rule.onset, y0)
    // no rains in this year's record: the dry season it is in (a record starting in
    // December is the end of the previous rains, Second Dry; one ending in February, First Dry)
    if (!wet.length) { add(daily.date[idx[0]].slice(5) >= '07-01' ? 3 : 0, y0, y1); continue }
    const onset = t[wet[0]]
    const heavy = above(rule.heavy, onset)
    const end = heavy.length ? t[heavy[heavy.length - 1]] + DAY : t[wet[wet.length - 1]] + DAY
    add(0, y0, onset)
    if (heavy.length) {
      add(1, onset, t[heavy[0]])
      add(2, t[heavy[0]], end)
    } else add(1, onset, end)
    add(3, end, y1)
  }
  return RAIN_SEASONS.map((s, i) => ({ name: s.name, color: s.color, months: [...s.months], spans: spans[i] }))
}
