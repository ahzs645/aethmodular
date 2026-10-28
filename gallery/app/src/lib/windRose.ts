/**
 * Wind and a measurement together, for the Weather tab's pollution rose, polar plot and wind-speed
 * chart (charts/weather/PollutionRose, PolarPlot, WindSpeedConcentration). A filter has one
 * concentration but many hours of wind: each hour of its sampling window counts once, in the
 * direction sector and speed bin it blew from, carrying the filter's concentration. So a sector's
 * statistic is over hours, weighted by how long each filter's air came from there, the way openair's
 * pollutionRose and polarPlot treat hourly data. Pure, so node tests it (lib/windRose.test.mjs).
 *
 * Two inputs: wind.json (16 sectors × 0.5 m/s speed bins, sparse, export_weather.py) and, where that
 * file is not exported yet, roses.json (8 sectors × 3 speed classes).
 */

/** hours of one filter's window in one cell of the wind grid */
export interface WindCell { sector: number; speed: number; hours: number }
/** one filter's wind over its window and its concentration */
export interface FilterWind { id: string; value: number; cells: WindCell[]; calm: number }

/** roses.json's 25 counts (8 sectors × 3 speed classes, row-major, then calm) as cells */
export function cellsFromRose8(counts: readonly number[]): { cells: WindCell[]; calm: number } {
  const cells: WindCell[] = []
  for (let s = 0; s < 8; s++) for (let k = 0; k < 3; k++) {
    const h = counts[s * 3 + k]
    if (h > 0) cells.push({ sector: s, speed: k, hours: h })
  }
  return { cells, calm: counts[24] ?? 0 }
}

/** wind.json's sparse list (calm hours, then cell, hours pairs; cell = sector × nSpeed + bin) as cells */
export function cellsFromSparse(enc: readonly number[], nSpeed: number): { cells: WindCell[]; calm: number } {
  const cells: WindCell[] = []
  for (let i = 1; i + 1 < enc.length; i += 2) {
    if (enc[i + 1] > 0) cells.push({ sector: Math.floor(enc[i] / nSpeed), speed: enc[i] % nSpeed, hours: enc[i + 1] })
  }
  return { cells, calm: enc[0] ?? 0 }
}

/**
 * The weighted quantile at p (0–1): each value sits at the middle of its share of the total weight,
 * linear in between, clamped at the ends. With equal weights it is the ordinary median (the mean of
 * the two middle values of an even count). Null with no positive weight.
 */
export function weightedQuantile(values: readonly number[], weights: readonly number[], p: number): number | null {
  const pairs = values.map((v, i) => [v, weights[i]] as const).filter(([v, w]) => Number.isFinite(v) && w > 0).sort((a, b) => a[0] - b[0])
  if (!pairs.length) return null
  const total = pairs.reduce((a, [, w]) => a + w, 0)
  let cum = 0
  const pos = pairs.map(([, w]) => { const at = (cum + w / 2) / total; cum += w; return at })
  if (p <= pos[0]) return pairs[0][0]
  for (let i = 1; i < pairs.length; i++) {
    if (p <= pos[i]) return pairs[i - 1][0] + ((p - pos[i - 1]) / (pos[i] - pos[i - 1])) * (pairs[i][0] - pairs[i - 1][0])
  }
  return pairs[pairs.length - 1][0]
}

export type RoseStat = 'mean' | 'median' | 'percentile'

/**
 * Class edges for the stacked pollution rose: the filters' concentrations at `qs`, rounded to two
 * significant figures and kept only where they rise, so the classes hold similar numbers of filters
 * at any site rather than putting a skewed PM2.5 almost all in the lowest of evenly spaced classes.
 */
export function concentrationEdges(values: readonly number[], qs: readonly number[] = [0.25, 0.5, 0.75, 0.9]): number[] {
  const v = values.filter(Number.isFinite).slice().sort((a, b) => a - b)
  if (v.length < 2) return []
  const q = (p: number) => { const i = (v.length - 1) * p, lo = Math.floor(i), hi = Math.ceil(i); return v[lo] + (i - lo) * (v[hi] - v[lo]) }
  const round = (x: number) => (x === 0 ? 0 : Number(x.toPrecision(2)))
  const out: number[] = []
  for (const p of qs) { const e = round(q(p)); if (!out.length || e > out[out.length - 1]) out.push(e) }
  return out
}
/** the class a value falls in: 0 below the first edge, edges.length at or above the last */
export const classOf = (v: number, edges: readonly number[]) => { let k = 0; while (k < edges.length && v >= edges[k]) k++; return k }

export interface RoseSector {
  /** hours from this sector, and the filters that had any */
  hours: number; filters: number
  /** the statistic of the concentration over those hours (null below minFilters) */
  value: number | null
  /** hours from this sector by concentration class (the stacked rose) */
  classHours: number[]
}
export interface PollutionRose { sectors: RoseSector[]; hours: number; calm: number; calmValue: number | null; filters: number }

/**
 * The pollution rose of a group of filters over `nSectors`: per sector the hours, the hour-weighted
 * mean, median or percentile `p` (0–100) of the concentration, and the hours split by concentration
 * class (`edges`, concentrationEdges). Calm hours have no direction: they are counted apart, with the
 * same statistic. A sector with fewer than `minFilters` filters gets no statistic.
 */
export function pollutionRose(filters: readonly FilterWind[], nSectors: number, opts: { stat: RoseStat; p?: number; edges: readonly number[]; minFilters?: number }): PollutionRose {
  const { stat, p = 75, edges, minFilters = 3 } = opts
  const vals: number[][] = Array.from({ length: nSectors }, () => [])
  const wts: number[][] = Array.from({ length: nSectors }, () => [])
  const classHours = Array.from({ length: nSectors }, () => new Array<number>(edges.length + 1).fill(0))
  const calmV: number[] = [], calmW: number[] = []
  let calm = 0
  for (const f of filters) {
    if (!Number.isFinite(f.value)) continue
    const k = classOf(f.value, edges)
    const bySector = new Array<number>(nSectors).fill(0)
    for (const c of f.cells) bySector[c.sector] += c.hours
    bySector.forEach((h, s) => { if (h > 0) { vals[s].push(f.value); wts[s].push(h); classHours[s][k] += h } })
    if (f.calm > 0) { calm += f.calm; calmV.push(f.value); calmW.push(f.calm) }
  }
  const statOf = (v: number[], w: number[]) => {
    if (v.length < minFilters) return null
    if (stat === 'mean') { const tw = w.reduce((a, b) => a + b, 0); return tw ? v.reduce((a, x, i) => a + x * w[i], 0) / tw : null }
    return weightedQuantile(v, w, stat === 'median' ? 0.5 : p / 100)
  }
  const sectors = vals.map((v, s) => ({ hours: wts[s].reduce((a, b) => a + b, 0), filters: v.length, value: statOf(v, wts[s]), classHours: classHours[s] }))
  return { sectors, hours: sectors.reduce((a, s) => a + s.hours, 0), calm, calmValue: statOf(calmV, calmW), filters: filters.filter((f) => Number.isFinite(f.value)).length }
}

export interface PolarCell { sector: number; speed: number; hours: number; filters: number; value: number | null }

/**
 * openair's polarPlot on the wind grid: the hour-weighted mean concentration in each sector × speed
 * cell. With `smooth`, each cell is a Gaussian-kernel (Nadaraya–Watson) mean over its neighbours, one
 * sector and one speed bin wide, the direction wrapping round; the weights are the neighbours' hours,
 * so a cell many hours fell in counts for more. A cell is drawn when its hours (smoothed: its
 * kernel-weighted hours) reach `minHours`, and unsmoothed also `minFilters` filters.
 */
export function polarGrid(filters: readonly FilterWind[], nSectors: number, nSpeed: number, opts: { smooth?: boolean; minHours?: number; minFilters?: number } = {}): PolarCell[] {
  const { smooth = true, minHours = 10, minFilters = 3 } = opts
  const n = nSectors * nSpeed
  const hours = new Array<number>(n).fill(0), sum = new Array<number>(n).fill(0)
  const seen: Set<string>[] = Array.from({ length: n }, () => new Set())
  for (const f of filters) {
    if (!Number.isFinite(f.value)) continue
    for (const c of f.cells) {
      const i = c.sector * nSpeed + c.speed
      hours[i] += c.hours; sum[i] += c.hours * f.value; seen[i].add(f.id)
    }
  }
  const out: PolarCell[] = []
  for (let s = 0; s < nSectors; s++) for (let k = 0; k < nSpeed; k++) {
    const i = s * nSpeed + k
    if (!smooth) {
      out.push({ sector: s, speed: k, hours: hours[i], filters: seen[i].size, value: hours[i] >= minHours && seen[i].size >= minFilters ? sum[i] / hours[i] : null })
      continue
    }
    let wh = 0, ws = 0
    const near = new Set<string>()
    for (let ds = -2; ds <= 2; ds++) for (let dk = -2; dk <= 2; dk++) {
      const kk = k + dk
      if (kk < 0 || kk >= nSpeed) continue
      const j = (((s + ds) % nSectors) + nSectors) % nSectors * nSpeed + kk
      const w = Math.exp(-(ds * ds + dk * dk) / 2)
      wh += w * hours[j]; ws += w * sum[j]
      if (hours[j] > 0 && Math.abs(ds) <= 1 && Math.abs(dk) <= 1) seen[j].forEach((id) => near.add(id))
    }
    out.push({ sector: s, speed: k, hours: wh, filters: near.size, value: wh >= minHours && near.size >= minFilters ? ws / wh : null })
  }
  return out
}

export interface SpeedBin { lo: number; hi: number; n: number; median: number | null; q1: number | null; q3: number | null }

/**
 * The concentration by the filters' window-mean wind speed, in bins `width` m/s wide from 0: the
 * median and (from 4 filters, as MonthlyBand) the interquartile range of each bin's filters.
 */
export function speedBins(points: readonly { speed: number; value: number }[], width: number): SpeedBin[] {
  const ok = points.filter((p) => Number.isFinite(p.speed) && Number.isFinite(p.value) && p.speed >= 0)
  if (!ok.length || !(width > 0)) return []
  const top = Math.floor(Math.max(...ok.map((p) => p.speed)) / width)
  const bins: number[][] = Array.from({ length: top + 1 }, () => [])
  for (const p of ok) bins[Math.floor(p.speed / width)].push(p.value)
  const q = (v: number[], p: number) => { const i = (v.length - 1) * p, lo = Math.floor(i), hi = Math.ceil(i); return v[lo] + (i - lo) * (v[hi] - v[lo]) }
  return bins.map((v, i) => {
    v.sort((a, b) => a - b)
    return { lo: i * width, hi: (i + 1) * width, n: v.length, median: v.length ? q(v, 0.5) : null, q1: v.length >= 4 ? q(v, 0.25) : null, q3: v.length >= 4 ? q(v, 0.75) : null }
  })
}

/**
 * Where a rose's ring labels go: along the gap between two sectors nearest `near` degrees (22.5° at
 * 8 sectors, 33.75° at 16), clear of the N axis and its compass letter, so a label sits between
 * wedges rather than on one; steep enough that labels a few px apart in radius stack one above the
 * other instead of running into each other sideways.
 */
export function ringLabelBearing(nSectors: number, near = 25): number {
  const w = 360 / nSectors
  return (Math.round((near - w / 2) / w) * w + w / 2 + 360) % 360
}
