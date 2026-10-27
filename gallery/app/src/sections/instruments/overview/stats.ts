/**
 * The arithmetic behind the aethalometer Overview, kept free of React and d3 so node can test it
 * (stats.test.mjs). A "day" is one 24 h bin of the daily file (gallery/data/export_aethalometer.py),
 * labelled at its start in local time; everything here works on the five MA350 channels' BCc.
 */
import { pearson, twoSample } from '../../../lib/stats.ts'
import { seasonOfDate } from '../../../lib/seasons.ts'
import type { SeasonMeta } from '../../../lib/types'

export const CHANNELS = ['UV', 'Blue', 'Green', 'Red', 'IR'] as const
export type Channel = (typeof CHANNELS)[number]
/**
 * MA350 attenuation cross-sections (m²/g), the ones export_aethalometer.py multiplies BCc by to get
 * absorption (src/external/calibration.py): BCc × σ = b_abs in Mm⁻¹ when BCc is in µg/m³.
 */
export const SIGMA: Record<Channel, number> = { UV: 24.069, Blue: 19.07, Green: 17.028, Red: 14.091, IR: 10.12 }

/** The daily file's columns this module reads (lib.ts AethDaily has them all). */
export interface DailyColumns {
  start: string[]
  UV: (number | null)[]; Blue: (number | null)[]; Green: (number | null)[]; Red: (number | null)[]; IR: (number | null)[]
  coverage: (number | null)[]
  flow: (string | null)[]
}

export interface Day {
  /** the bin's start, the local wall-clock time read as UTC (so getUTC* give the local fields) */
  t: Date
  start: string
  /** the calendar date of the bin's middle (YYYY-MM-DD): the day most of its 24 hours fall on */
  date: string
  month: number
  /** 0 = Monday … 6 = Sunday, of `date` */
  dow: number
  season: SeasonMeta | null
  /** BCc (µg/m³) after the screen; null where missing or screened out */
  bc: Record<Channel, number | null>
  /** minutes with data, % of the day (absent at Addis) */
  coverage: number | null
  flow: string | null
}

/**
 * Which daily values to drop. addis_02_temporal_patterns_daily dropped negative BCc and anything
 * above mean + 3 SD of the channel (the mean and SD of the whole record, after the negatives).
 */
export type Screen = 'none' | 'neg' | '3sd'
export interface ScreenCount { neg: number; high: number }

const HOUR = 3600e3
export const parseStart = (s: string) => new Date(`${s.replace(' ', 'T')}:00Z`)
const isoDate = (t: Date) => t.toISOString().slice(0, 10)

/**
 * The file's bins as days. A bin is dated by its middle (start + 12 h): Addis bins start at 09:00, so
 * that is the start's own date, as in the notebooks; Beijing, Delhi and JPL bins start at 15:00, so
 * 15 of their 24 hours are on the next day, and that is the day they count for.
 */
export function toDays(d: DailyColumns, seasons: readonly SeasonMeta[], screen: Screen): { days: Day[]; dropped: Record<Channel, ScreenCount> } {
  const dropped = Object.fromEntries(CHANNELS.map((c) => [c, { neg: 0, high: 0 }])) as Record<Channel, ScreenCount>
  const limit = {} as Record<Channel, number>
  for (const c of CHANNELS) {
    const vs = d[c].filter((v): v is number => v !== null && Number.isFinite(v) && v >= 0)
    const s = summarize(vs)
    limit[c] = screen === '3sd' && s && s.sd !== null ? s.mean + 3 * s.sd : Infinity
  }
  const days = d.start.map((start, i) => {
    const t = parseStart(start)
    const mid = new Date(t.getTime() + 12 * HOUR)
    const date = isoDate(mid), month = mid.getUTCMonth() + 1
    const bc = {} as Record<Channel, number | null>
    for (const c of CHANNELS) {
      const v = d[c][i]
      if (v === null || !Number.isFinite(v)) bc[c] = null
      else if (screen !== 'none' && v < 0) { bc[c] = null; dropped[c].neg++ }
      else if (v > limit[c]) { bc[c] = null; dropped[c].high++ }
      else bc[c] = v
    }
    return { t, start, date, month, dow: (mid.getUTCDay() + 6) % 7, season: seasonOfDate(date, month, seasons), bc, coverage: d.coverage[i] ?? null, flow: d.flow[i] ?? null }
  })
  return { days, dropped }
}

export interface Summary { n: number; mean: number; sd: number | null; median: number; q10: number; q25: number; q75: number; q90: number; min: number; max: number }

/** linear-interpolated quantile of ascending values (numpy's default, as pandas' quantile) */
export function quantileSorted(v: readonly number[], p: number): number {
  const idx = (v.length - 1) * p, lo = Math.floor(idx), hi = Math.ceil(idx)
  return lo === hi ? v[lo] : v[lo] + (idx - lo) * (v[hi] - v[lo])
}

/** n, mean, sample SD and quantiles of the finite values; null when there are none */
export function summarize(values: Iterable<number | null | undefined>): Summary | null {
  const v = [...values].filter((x): x is number => x !== null && x !== undefined && Number.isFinite(x)).sort((a, b) => a - b)
  if (!v.length) return null
  const mean = v.reduce((s, x) => s + x, 0) / v.length
  const sd = v.length > 1 ? Math.sqrt(v.reduce((s, x) => s + (x - mean) ** 2, 0) / (v.length - 1)) : null
  const q = (p: number) => quantileSorted(v, p)
  return { n: v.length, mean, sd, median: q(0.5), q10: q(0.1), q25: q(0.25), q75: q(0.75), q90: q(0.9), min: v[0], max: v[v.length - 1] }
}

/**
 * A centred moving mean over calendar time: each day's value is the mean of the days within
 * ±`half` days of it that have one, kept where at least `minN` do. addis_02 used a 30-row
 * window (`rolling(30, center=True, min_periods=7)`); counted in days, a gap in the record
 * cannot stretch the window across months.
 */
export function movingMean(days: readonly Day[], get: (d: Day) => number | null, half = 15, minN = 7): (number | null)[] {
  const ts = days.map((d) => d.t.getTime()), vs = days.map(get)
  const out: (number | null)[] = []
  let lo = 0, hi = 0, sum = 0, n = 0
  for (let i = 0; i < days.length; i++) {
    while (hi < days.length && ts[hi] <= ts[i] + half * 24 * HOUR) { const v = vs[hi]; if (v !== null) { sum += v; n++ } hi++ }
    while (ts[lo] < ts[i] - half * 24 * HOUR) { const v = vs[lo]; if (v !== null) { sum -= v; n-- } lo++ }
    out.push(n >= minN ? sum / n : null)
  }
  return out
}

/** The change from one day to the next, only between consecutive bins (24 h apart, ±1 h for a DST change; both with a value). */
export function dayToDay(days: readonly Day[], get: (d: Day) => number | null): { day: Day; dv: number }[] {
  const out: { day: Day; dv: number }[] = []
  for (let i = 1; i < days.length; i++) {
    const a = get(days[i - 1]), b = get(days[i])
    if (a !== null && b !== null && Math.abs(days[i].t.getTime() - days[i - 1].t.getTime() - 24 * HOUR) <= HOUR) out.push({ day: days[i], dv: b - a })
  }
  return out
}

export interface WeekendEffect {
  weekday: Summary; weekend: Summary
  /** (weekend mean − weekday mean) / weekday mean, % (negative: lower at weekends) */
  change: number
  /** its 95 % interval from the Welch standard error (normal approximation) */
  lo: number; hi: number
  /** Welch's two-sided p (lib/stats twoSample) */
  p: number
}

/**
 * Saturday and Sunday against Monday to Friday, as addis_02 Task 2 (which reported the reduction
 * (weekday − weekend) / weekday and a Student t-test). Daily values are autocorrelated, so p is
 * optimistic; read the interval, not the star.
 */
export function weekendEffect(days: readonly Day[], get: (d: Day) => number | null): WeekendEffect | null {
  const pick = (we: boolean) => days.filter((d) => (d.dow >= 5) === we).map(get).filter((v): v is number => v !== null && Number.isFinite(v))
  return compareWeekend(pick(false), pick(true))
}

/** The weekend effect from the two sets of values (weekday, weekend) directly. */
export function compareWeekend(weekday: readonly number[], weekend: readonly number[]): WeekendEffect | null {
  const wd = summarize(weekday), we = summarize(weekend), ts = twoSample([...weekend], [...weekday])
  if (!wd || !we || !ts || wd.mean === 0 || wd.sd === null || we.sd === null) return null
  const se = Math.sqrt(wd.sd ** 2 / wd.n + we.sd ** 2 / we.n)
  const k = 100 / wd.mean
  const change = (we.mean - wd.mean) * k
  return { weekday: wd, weekend: we, change, lo: change - 1.96 * se * Math.abs(k), hi: change + 1.96 * se * Math.abs(k), p: ts.p }
}

/**
 * Each channel's absorption relative to 880 nm, day by day, over the days with all five channels
 * positive. `absorption`: b_abs(λ)/b_abs(880) with b_abs = BCc × σ (the spectral shape of the
 * aerosol). `bcc`: BCc(λ)/BCc(IR), the notebooks' UV/IR ratio, which divides out most of the λ⁻¹
 * of black carbon through the cross-sections and so reads about one for pure BC.
 */
export function relativeSpectra(days: readonly Day[], kind: 'absorption' | 'bcc'): Record<Channel, number[]> {
  const out = Object.fromEntries(CHANNELS.map((c) => [c, [] as number[]])) as Record<Channel, number[]>
  for (const d of days) {
    if (!CHANNELS.every((c) => (d.bc[c] ?? 0) > 0)) continue
    for (const c of CHANNELS) out[c].push(kind === 'absorption' ? (d.bc[c]! * SIGMA[c]) / (d.bc.IR! * SIGMA.IR) : d.bc[c]! / d.bc.IR!)
  }
  return out
}

/**
 * The exponent of a power law through points (λ, y): −(slope of ln y on ln λ), ordinary least squares.
 * For a relative absorption spectrum it is the five-wavelength AAE of that spectrum.
 */
export function powerLawExponent(lambda: readonly number[], y: readonly number[]): number | null {
  const pts = lambda.map((l, i) => [Math.log(l), Math.log(y[i])]).filter(([a, b]) => Number.isFinite(a) && Number.isFinite(b))
  if (pts.length < 2) return null
  const mx = pts.reduce((s, p) => s + p[0], 0) / pts.length, my = pts.reduce((s, p) => s + p[1], 0) / pts.length
  const sxx = pts.reduce((s, p) => s + (p[0] - mx) ** 2, 0), sxy = pts.reduce((s, p) => s + (p[0] - mx) * (p[1] - my), 0)
  return sxx === 0 ? null : -sxy / sxx
}

/** Pearson r between every pair of channels over the days with all five (Multi_Site_Analysis_FollowUp 7a). */
export function channelCorrelation(days: readonly Day[]): { r: (number | null)[][]; n: number } {
  const full = days.filter((d) => CHANNELS.every((c) => d.bc[c] !== null))
  const cols = CHANNELS.map((c) => full.map((d) => d.bc[c]!))
  return { r: cols.map((a) => cols.map((b) => pearson(a, b))), n: full.length }
}

/**
 * The days in runs, as index lists, so lines break between runs: by default runs of consecutive
 * bins (each 24 h after the last, ±1 h); `maxGapDays` lets a smoothed line bridge shorter gaps.
 */
export function splitRuns(days: readonly Day[], maxGapDays = 1): number[][] {
  const out: number[][] = []
  days.forEach((d, i) => {
    if (i > 0 && d.t.getTime() - days[i - 1].t.getTime() <= maxGapDays * 24 * HOUR + HOUR) out[out.length - 1].push(i)
    else out.push([i])
  })
  return out
}
