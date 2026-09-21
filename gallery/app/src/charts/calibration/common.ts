import * as d3 from 'd3'
import type { CalibFile, CalibRow, MetaFile, MetricRow, SweepRow } from '@/lib/types'

/** A configuration is what the explorer fits: cohort × cutoff × selection space × calibration spectra. */
export interface Config {
  co: string
  cut: number | null
  sel: string | null
  sp: string
}

export const configKey = (c: Config) => `${c.co}|${c.cut ?? ''}|${c.sel ?? ''}|${c.sp}`
export const sameConfig = (a: Config, b: Config) => configKey(a) === configKey(b)

export function configLabel(c: Config, calib: CalibFile): string {
  const cohort = calib.cohorts[c.co] ?? c.co
  const cut = c.cut !== null ? `-${c.cut}` : ''
  const sel = c.sel && c.sel !== 'raw' ? ` (selected on ${c.sel})` : ''
  return `${cohort}${cut}${sel} × ${c.sp}`
}

export const shortConfig = (c: Config) => `${c.co}${c.cut !== null ? '-' + c.cut : ''}${c.sel && c.sel !== 'raw' ? '·' + c.sel : ''} × ${c.sp}`

/** Fixed cohort colours — not in config.py, so declared once here. */
export const COHORT_COLOR: Record<string, string> = {
  pool: '#7b8794',
  smoke: '#8d5524',
  eth_shaped: '#E67E22',
  analogs: '#8e44ad',
  ocec: '#2b6cb0',
}
export const cohortColor = (co: string) => COHORT_COLOR[co] ?? '#5b6470'

/** Line dash per calibration spectra space, so raw / AIRSpec / deriv2 read apart at a glance. */
export const SPECTRA_DASH: Record<string, string | undefined> = { raw: undefined, airspec: '7 4', deriv2: '2 3', neutral: '10 3 2 3' }

export const METRICS = [
  'Deming intercept',
  'Deming slope',
  'OLS intercept',
  'OLS slope',
  'R²',
  'held-out TOR R²',
  'Deming intercept @ MAC 6',
] as const
export type Metric = (typeof METRICS)[number]

/** Read a metric off a grid row, on the fixed evaluation set or all pairs. */
export function metricOf(r: CalibRow, m: Metric, allPairs: boolean): number | null {
  switch (m) {
    case 'Deming intercept': return allPairs ? r.adb : r.db
    case 'Deming slope': return allPairs ? r.adm : r.dm
    case 'OLS intercept': return allPairs ? r.aob : r.ob
    case 'OLS slope': return allPairs ? r.aom : r.om
    case 'R²': return allPairs ? r.ar2 : r.r2
    case 'held-out TOR R²': return r.ho
    case 'Deming intercept @ MAC 6': return r.db6
  }
}

export const isSlopeMetric = (m: Metric) => m.endsWith('slope')
export const isInterceptMetric = (m: Metric) => m.includes('intercept')

export function sweepMetric(r: SweepRow, m: Metric): number | null {
  switch (m) {
    case 'Deming intercept': return r.db
    case 'Deming slope': return r.dm
    case 'OLS intercept': return r.ob
    case 'OLS slope': return r.om
    case 'R²': return r.r2
    case 'held-out TOR R²': return r.ho
    default: return null
  }
}

/** The explorer's optimisation score: |intercept| + w·|slope − 1|, Deming, fixed set. */
export const score = (r: CalibRow, w = 5) =>
  r.db !== null && r.dm !== null ? Math.abs(r.db) + w * Math.abs(r.dm - 1) : null

export const passes = (r: CalibRow, calib: CalibFile) =>
  r.dm !== null && r.dm >= calib.slope_box[0] && r.dm <= calib.slope_box[1] && (r.ho ?? 0) >= calib.heldout_floor

export const fmtFit = (m: number | null, b: number | null) =>
  m === null || b === null ? '—' : `${m.toFixed(2)}x ${b < 0 ? '−' : '+'} ${Math.abs(b).toFixed(2)}`

/** Colour for an evaluation group: the Ethiopian season colours from config when the label matches, else a fixed ordinal. */
export function groupColorFn(meta: MetaFile, groups: string[]): (g: string) => string {
  const bySeason = new Map<string, string>()
  for (const conv of Object.values(meta.season_conventions)) for (const s of conv) bySeason.set(s.name, s.color)
  for (const s of meta.seasons) bySeason.set(s.name, s.color)
  const others = [...new Set(groups.filter((g) => !bySeason.has(g)))].sort()
  const ord = d3.scaleOrdinal<string>().domain(others).range(d3.schemeTableau10)
  return (g) => bySeason.get(g) ?? ord(g)
}

/** The metrics row for one evaluation set and MAC (the explorer's crossplot convention). */
export function metricRow(rows: MetricRow[] | undefined, set: 'fixed' | 'all', mac: number | null): MetricRow | null {
  if (!rows?.length) return null
  return (
    rows.find((r) => r.evaluation_set === set && (mac === null ? r.MAC === null : r.MAC === mac)) ??
    rows.find((r) => r.evaluation_set === set) ??
    rows[0]
  )
}

export const PRESET_COLOR: Record<string, string> = {}
