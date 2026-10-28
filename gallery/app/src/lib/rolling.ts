/**
 * A rolling statistic over dated points, defined once for every chart that draws one (the calibration
 * Dated series first). The default is ftir_29's: pandas `rolling('45D', min_periods=5).median()`, a trailing
 * window (t − 45 days, t]. `centred` puts the same width around each point instead. Pure, so node tests it
 * (lib/rolling.test.mjs); components/RollingControls is its control.
 */
export type RollStat = 'median' | 'mean'
export type RollAlign = 'trailing' | 'centred'
export interface RollSpec { days: number; stat: RollStat; align: RollAlign; minN: number }

export const ROLL_DEFAULT: RollSpec = { days: 45, stat: 'median', align: 'trailing', minN: 5 }
export const ROLL_STATS: readonly RollStat[] = ['median', 'mean']
export const ROLL_ALIGNS: readonly RollAlign[] = ['trailing', 'centred']
/** the longest window a control accepts, in days (ten years) */
export const ROLL_MAX_DAYS = 3650

const DAY = 864e5

const median = (vs: number[]) => {
  const s = [...vs].sort((a, b) => a - b), m = s.length >> 1
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2
}
const mean = (vs: number[]) => vs.reduce((a, b) => a + b, 0) / vs.length

/**
 * The statistic at each point of `pts` (sorted by date): trailing is pandas' (t − days, t], centred is
 * [t − days/2, t + days/2]. Null where the window holds fewer than `minN` finite values.
 */
export function rollingByTime<T>(pts: readonly T[], dateOf: (p: T) => Date, valueOf: (p: T) => number, spec: RollSpec): { d: Date; v: number | null }[] {
  const t = pts.map((p) => +dateOf(p)), v = pts.map(valueOf)
  const span = spec.days * DAY, f = spec.stat === 'mean' ? mean : median
  let lo = 0, hi = 0
  return pts.map((p, i) => {
    const [from, to] = spec.align === 'trailing' ? [t[i] - span, t[i]] : [t[i] - span / 2, t[i] + span / 2]
    // both ends only move forward as the points do
    while (lo < t.length && (spec.align === 'trailing' ? t[lo] <= from : t[lo] < from)) lo++
    if (hi < i + 1) hi = i + 1
    while (hi < t.length && t[hi] <= to) hi++
    const win = v.slice(lo, hi).filter(Number.isFinite)
    return { d: dateOf(p), v: win.length >= spec.minN ? f(win) : null }
  })
}

/** "45-day rolling median", "30-day centred rolling mean" */
export const rollLabel = (s: RollSpec) => `${s.days}-day ${s.align === 'centred' ? 'centred ' : ''}rolling ${s.stat}`

/** a typed day count, whole and within 1–ROLL_MAX_DAYS, or the fallback */
export const clampDays = (text: string, fallback: number) => {
  const n = Math.round(Number(text))
  return Number.isFinite(n) && n >= 1 ? Math.min(ROLL_MAX_DAYS, n) : fallback
}
