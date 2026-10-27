/**
 * The arithmetic for the Overview's hourly charts (gallery/data/export_aeth_overview.py): IR BCc per
 * local-standard hour, and what addis_02_temporal_patterns did with the Jacros minutes beyond a daily
 * view. Free of React and d3, so node can test it (hourly.test.mjs).
 */
import { seasonOfDate } from '../../../lib/seasons.ts'
import type { SeasonMeta } from '../../../lib/types'
import { quantileSorted, summarize, type Summary } from './stats.ts'

/** overview/hourly/<CODE>.json: IR BCc (ng/m³) on every hour from `first` */
export interface HourlyFile { site: string; first: string; offset: number; unit: string; IR: (number | null)[] }

export interface Hour {
  /** local standard time, read as UTC (getUTC* give the local fields) */
  t: Date
  date: string
  hour: number
  /** 0 = Monday */
  dow: number
  month: number
  season: SeasonMeta | null
  /** IR BCc, µg/m³ */
  v: number
}

const H = 3600e3
const parse = (s: string) => new Date(`${s.replace(' ', 'T')}:00Z`)

/** The file's valid hours between dates lo and hi (inclusive, YYYY-MM-DD; null = no limit), seasoned by date. */
export function toHours(f: HourlyFile, seasons: readonly SeasonMeta[], lo: string | null = null, hi: string | null = null): Hour[] {
  const t0 = parse(f.first.length === 13 ? `${f.first}:00` : f.first).getTime()
  const out: Hour[] = []
  const seasonOf = new Map<string, SeasonMeta | null>()
  f.IR.forEach((v, i) => {
    if (v === null) return
    const t = new Date(t0 + i * H)
    const date = t.toISOString().slice(0, 10)
    if ((lo && date < lo) || (hi && date > hi)) return
    const month = t.getUTCMonth() + 1
    if (!seasonOf.has(date)) seasonOf.set(date, seasonOfDate(date, month, seasons))
    out.push({ t, date, hour: t.getUTCHours(), dow: (t.getUTCDay() + 6) % 7, month, season: seasonOf.get(date)!, v: v / 1000 })
  })
  return out
}

export interface DayShape {
  date: string; dow: number; season: SeasonMeta | null
  n: number; mean: number; max: number
  /** the local hour of the day's highest hour */
  peakHour: number
  /** the highest hour over the day's mean: how much a 24 h average flattens the day */
  ratio: number
}

/**
 * Each local calendar day with at least `minHours` valid hours: its mean, its highest hour and when
 * that came (addis_02 Task 3's "peak hour", per day rather than of the mean cycle).
 */
export function dayShapes(hours: readonly Hour[], minHours = 18): DayShape[] {
  const out: DayShape[] = []
  let i = 0
  while (i < hours.length) {
    let j = i
    while (j < hours.length && hours[j].date === hours[i].date) j++
    if (j - i >= minHours) {
      let sum = 0, k = i
      for (let m = i; m < j; m++) { sum += hours[m].v; if (hours[m].v > hours[k].v) k = m }
      const mean = sum / (j - i)
      out.push({ date: hours[i].date, dow: hours[i].dow, season: hours[i].season, n: j - i, mean, max: hours[k].v, peakHour: hours[k].hour, ratio: mean > 0 ? hours[k].v / mean : NaN })
    }
    i = j
  }
  return out
}

/** Share of the items falling in each hour 0–23 (null when there are none). */
export function hourShares(hoursOf: readonly number[]): (number | null)[] {
  const n = new Array(24).fill(0)
  for (const h of hoursOf) n[h]++
  const total = hoursOf.length
  return n.map((x) => (total ? x / total : null))
}

/** The IR value a share q of the hours stays below (the episode threshold uses 0.95). */
export function hourQuantile(hours: readonly Hour[], q: number): number | null {
  const v = hours.map((h) => h.v).sort((a, b) => a - b)
  return v.length ? quantileSorted(v, q) : null
}

/** addis_02 Task 2's weekly pattern: mean IR for each weekday (rows, Mon first) and hour (columns), with counts. */
export function weekHourGrid(hours: readonly Hour[]): { mean: (number | null)[][]; n: number[][]; overall: number | null } {
  const sum = Array.from({ length: 7 }, () => new Array(24).fill(0)), n = Array.from({ length: 7 }, () => new Array(24).fill(0))
  let s = 0
  for (const h of hours) { sum[h.dow][h.hour] += h.v; n[h.dow][h.hour]++; s += h.v }
  return { mean: sum.map((row, d) => row.map((x, k) => (n[d][k] ? x / n[d][k] : null))), n, overall: hours.length ? s / hours.length : null }
}

/** overview/index.json episodes, column-wise (start in local standard time) */
export interface EpisodeColumns { start: string[]; hours: number[]; peak: number[]; peak_at: number[]; mean: number[]; excess: number[] }
export interface Episode {
  i: number; start: Date; date: string; startHour: number; dow: number; season: SeasonMeta | null
  hours: number; peak: number; peakAt: number; mean: number; excess: number
}

/** The episodes starting between lo and hi, in µg/m³, seasoned by their start date. */
export function toEpisodes(e: EpisodeColumns, seasons: readonly SeasonMeta[], lo: string | null = null, hi: string | null = null): Episode[] {
  return e.start.flatMap((s, i) => {
    const start = parse(s.length === 13 ? `${s}:00` : s), date = s.slice(0, 10)
    if ((lo && date < lo) || (hi && date > hi)) return []
    return [{ i, start, date, startHour: start.getUTCHours(), dow: (start.getUTCDay() + 6) % 7, season: seasonOfDate(date, start.getUTCMonth() + 1, seasons),
      hours: e.hours[i], peak: e.peak[i] / 1000, peakAt: e.peak_at[i], mean: e.mean[i] / 1000, excess: e.excess[i] / 1000 }]
  })
}

/** Episodes per 30 days of valid data (720 valid hours), their median length and peak, for a group of hours. */
export function episodeRate(eps: readonly Episode[], validHours: number): { n: number; per30: number | null; hours: Summary | null; peak: Summary | null } {
  return { n: eps.length, per30: validHours ? (eps.length * 720) / validHours : null, hours: summarize(eps.map((e) => e.hours)), peak: summarize(eps.map((e) => e.peak)) }
}
