/**
 * The arithmetic behind the Diurnal page's spectral diagnostics (addis_05_diurnal_wavelength_analysis
 * and multisite_diurnal_wavelength_analysis): Delta-C, the coefficient of variation across the five
 * channels, ratios and BCc Ångström exponents as the notebooks define them, each channel's hour-of-day
 * profile over its daily mean, the representative days and the time-block summary. The notebooks ran
 * on minutes; here every quantity is taken on the export's day-hour means. Free of React and d3, so
 * node can test it (spectral.test.mjs).
 */

export const SPECTRAL_CHANNELS = ['UV', 'Blue', 'Green', 'Red', 'IR'] as const
export type SpectralChannel = (typeof SPECTRAL_CHANNELS)[number]
/** the MA350's wavelengths (addis_05; the multisite notebook labelled them with the AE33's 370/520/660) */
export const WAVELENGTH_NM: Record<SpectralChannel, number> = { UV: 375, Blue: 470, Green: 528, Red: 625, IR: 880 }
/** the notebooks' floor: ratios need IR above it, AAE both channels, the CV a five-channel mean (µg/m³) */
export const FLOOR = 0.1

type Val = number | null | undefined
const ok = (v: Val): v is number => v != null && Number.isFinite(v)

/** Delta-C, UV BCc − IR BCc (the notebooks' biomass-burning tracer) */
export const deltaC = (uv: Val, ir: Val): number | null => (ok(uv) && ok(ir) ? uv - ir : null)

/** a channel over IR, where IR is above the floor (the notebooks' `<name>/IR`) */
export const overIr = (v: Val, ir: Val): number | null => (ok(v) && ok(ir) && ir > FLOOR ? v / ir : null)

/** AAE from a BCc ratio, −ln(a / b) / ln(λa / λb), both above the floor (the notebooks' AAE_UV_IR, AAE_Blue_IR) */
export function bccAae(a: Val, b: Val, la: number, lb: number): number | null {
  return ok(a) && ok(b) && a > FLOOR && b > FLOOR ? -Math.log(a / b) / Math.log(la / lb) : null
}

/**
 * The spread of the five channels: SD / mean over the channels present (pandas skips a missing one;
 * SD with n − 1), where the mean is above the floor (the notebooks' wl_CV).
 */
export function wavelengthCv(values: readonly Val[]): number | null {
  const v = values.filter(ok)
  if (v.length < 2) return null
  const mean = v.reduce((a, b) => a + b, 0) / v.length
  if (!(mean > FLOOR)) return null
  const sd = Math.sqrt(v.reduce((a, b) => a + (b - mean) ** 2, 0) / (v.length - 1))
  return sd / mean
}

/** a quantile of sorted values, linear between ranks (pandas' and d3's default) */
export function quantileSorted(sorted: readonly number[], p: number): number | null {
  const n = sorted.length
  if (!n) return null
  const i = (n - 1) * p, lo = Math.floor(i), hi = Math.min(n - 1, lo + 1)
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (i - lo)
}
export function median(values: readonly Val[]): number | null {
  return quantileSorted(values.filter(ok).sort((a, b) => a - b), 0.5)
}

/** per day-hour, every derived quantity of the notebooks' compute_derived_metrics */
export interface Derived {
  deltaC: (number | null)[]; cv: (number | null)[]
  uvIr: (number | null)[]; blueIr: (number | null)[]; greenIr: (number | null)[]; redIr: (number | null)[]
  aaeUvIr: (number | null)[]; aaeBlueIr: (number | null)[]
}
export function derive(ch: Record<SpectralChannel, readonly Val[]>): Derived {
  const n = ch.IR.length
  const at = (c: SpectralChannel, i: number) => ch[c][i]
  const out: Derived = { deltaC: [], cv: [], uvIr: [], blueIr: [], greenIr: [], redIr: [], aaeUvIr: [], aaeBlueIr: [] }
  for (let i = 0; i < n; i++) {
    const ir = at('IR', i)
    out.deltaC.push(deltaC(at('UV', i), ir))
    out.cv.push(wavelengthCv(SPECTRAL_CHANNELS.map((c) => at(c, i))))
    out.uvIr.push(overIr(at('UV', i), ir)); out.blueIr.push(overIr(at('Blue', i), ir))
    out.greenIr.push(overIr(at('Green', i), ir)); out.redIr.push(overIr(at('Red', i), ir))
    out.aaeUvIr.push(bccAae(at('UV', i), ir, WAVELENGTH_NM.UV, WAVELENGTH_NM.IR))
    out.aaeBlueIr.push(bccAae(at('Blue', i), ir, WAVELENGTH_NM.Blue, WAVELENGTH_NM.IR))
  }
  return out
}

/** the median of a column in each of the 24 hours (null with fewer than `minN` values) */
export function hourMedians(hours: readonly number[], values: readonly Val[], keep: (i: number) => boolean = () => true, minN = 1): { median: (number | null)[]; n: number[] } {
  const bins: number[][] = Array.from({ length: 24 }, () => [])
  hours.forEach((h, i) => { const v = values[i]; if (ok(v) && keep(i)) bins[h].push(v) })
  return { median: bins.map((b) => (b.length >= minN ? median(b) : null)), n: bins.map((b) => b.length) }
}

/**
 * A channel's hour-of-day profile against its own day (the notebooks' heatmap): each hour's median over
 * the mean of the 24 hourly medians, so 1 is the daily average. Null where the hour has no median.
 */
export function normalisedProfile(medians: readonly (number | null)[]): { ratio: (number | null)[]; dailyMean: number | null } {
  const v = medians.filter(ok)
  const dailyMean = v.length ? v.reduce((a, b) => a + b, 0) / v.length : null
  return { ratio: medians.map((m) => (ok(m) && dailyMean != null && dailyMean > 0 ? m / dailyMean : null)), dailyMean }
}

/** each day's median CV over its day-hours, for days with at least `minMinutes` minutes of valid IR */
export function dailyCv(days: readonly string[], cv: readonly Val[], ir: readonly Val[], minutes: (i: number) => number, minMinutes = 1200): { day: string; cv: number; minutes: number }[] {
  const acc = new Map<string, { cv: number[]; minutes: number }>()
  days.forEach((d, i) => {
    const a = acc.get(d) ?? { cv: [], minutes: 0 }
    if (ok(ir[i])) a.minutes += minutes(i)
    const c = cv[i]
    if (ok(c)) a.cv.push(c)
    acc.set(d, a)
  })
  return [...acc].flatMap(([day, a]) => {
    const m = median(a.cv)
    return m != null && a.minutes >= minMinutes ? [{ day, cv: m, minutes: a.minutes }] : []
  })
}

export type RepresentativeKind = 'High CV' | 'Medium CV' | 'Low CV'
/**
 * The multisite notebook's find_representative_days: the sixth-highest and sixth-lowest daily CV (skipping
 * the extremes) and the middle day by rank; none with fewer than 10 days.
 */
export function representativeDays<T extends { day: string; cv: number }>(daily: readonly T[]): { kind: RepresentativeKind; day: T }[] | null {
  if (daily.length < 10) return null
  const asc = [...daily].sort((a, b) => a.cv - b.cv || a.day.localeCompare(b.day))
  const desc = [...asc].reverse()
  const k = Math.min(5, Math.min(20, daily.length) - 1)
  return [{ kind: 'High CV', day: desc[k] }, { kind: 'Medium CV', day: asc[Math.floor(asc.length / 2)] }, { kind: 'Low CV', day: asc[k] }]
}

/** the multisite notebook's time blocks, [start, end) local hours */
export const TIME_BLOCKS: { label: string; from: number; to: number }[] = [
  { label: 'Night (00–05)', from: 0, to: 5 }, { label: 'Morning (05–09)', from: 5, to: 9 }, { label: 'Midday (09–16)', from: 9, to: 16 },
  { label: 'Evening (16–21)', from: 16, to: 21 }, { label: 'Late (21–24)', from: 21, to: 24 },
]
export const inBlock = (h: number, b: { from: number; to: number }) => h >= b.from && h < b.to

export interface BlockRow {
  block: string
  /** day-hours with valid IR, and the minutes behind them */
  n: number; minutes: number
  ir: number | null; uvIr: number | null; aaeUvIr: number | null; aaeBlueIr: number | null; deltaC: number | null; cv: number | null
}
/** the summary table's rows for one set of day-hours (`keep`): medians per time block */
export function blockSummary(hours: readonly number[], ir: readonly Val[], d: Derived, keep: (i: number) => boolean, minutes: (i: number) => number): BlockRow[] {
  return TIME_BLOCKS.map((b) => {
    const idx: number[] = []
    hours.forEach((h, i) => { if (inBlock(h, b) && keep(i)) idx.push(i) })
    const col = (c: readonly Val[]) => median(idx.map((i) => c[i]))
    const valid = idx.filter((i) => ok(ir[i]))
    return { block: b.label, n: valid.length, minutes: valid.reduce((a, i) => a + minutes(i), 0),
      ir: col(ir), uvIr: col(d.uvIr), aaeUvIr: col(d.aaeUvIr), aaeBlueIr: col(d.aaeBlueIr), deltaC: col(d.deltaC), cv: col(d.cv) }
  })
}

/** Pearson r over the pairs where both are finite */
export function pearson(x: readonly Val[], y: readonly Val[]): number | null {
  const p: [number, number][] = []
  x.forEach((a, i) => { const b = y[i]; if (ok(a) && ok(b)) p.push([a, b]) })
  if (p.length < 3) return null
  const mx = p.reduce((s, [a]) => s + a, 0) / p.length, my = p.reduce((s, [, b]) => s + b, 0) / p.length
  let sxy = 0, sxx = 0, syy = 0
  for (const [a, b] of p) { sxy += (a - mx) * (b - my); sxx += (a - mx) ** 2; syy += (b - my) ** 2 }
  return sxx > 0 && syy > 0 ? sxy / Math.sqrt(sxx * syy) : null
}
