import * as d3 from 'd3'
import type { CalibFile, CalibRow, MetaFile, MetricRow, SweepRow } from '@/lib/types'
import { METRIC, PREPROCESSING, SELECTION_SPACE, label } from '@/lib/labels'

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
  const sel = c.sel && c.sel !== 'raw' ? ` (${label(SELECTION_SPACE, c.sel)})` : ''
  return `${cohort}${cut}${sel} × ${label(PREPROCESSING, c.sp)}`
}

export const shortConfig = (c: Config) => `${c.co}${c.cut !== null ? '-' + c.cut : ''}${c.sel && c.sel !== 'raw' ? '·' + c.sel : ''} × ${label(PREPROCESSING, c.sp)}`

/** A preset's exported label with the preprocessing in reader-facing words ("× AIRSpec" → "× Spline baseline"). */
export const presetName = (p: { label: string; spectra: string }) =>
  p.label.replace(/× (raw|AIRSpec|SG 2nd derivative|2nd derivative)(?=$|\s)/, `× ${label(PREPROCESSING, p.spectra)}`)

/** Fixed cohort colours — not in config.py, so declared once here. */
export const COHORT_COLOR: Record<string, string> = {
  pool: '#7b8794',
  smoke: '#8d5524',
  eth_shaped: '#E67E22',
  analogs: '#8e44ad',
  ocec: '#2b6cb0',
}
export const cohortColor = (co: string) => COHORT_COLOR[co] ?? '#5b6470'

/** Colour per calibration spectra space, shared by every chart that puts the preprocessings side by side. */
export const SPECTRA_COLOR: Record<string, string> = { raw: '#1f2933', airspec: '#2171b5', deriv2: '#c026d3', neutral: '#0f766e' }
export const spectraColor = (sp: string) => SPECTRA_COLOR[sp] ?? '#5b6470'


export const METRICS = [
  'Deming intercept',
  'Deming slope',
  'R²',
  'held-out TOR R²',
  'Deming intercept @ MAC 6',
] as const
export type Metric = (typeof METRICS)[number]

/** Reader-facing name for each metric key (the keys stay the explorer's internal names). */
export const METRIC_LABEL: Record<Metric, string> = {
  'Deming intercept': METRIC.testIntercept,
  'Deming slope': METRIC.testSlope,
  'R²': METRIC.testR2,
  'held-out TOR R²': METRIC.cvR2,
  'Deming intercept @ MAC 6': 'Test set Deming intercept at MAC 6 (Addis, µg/m³)',
}
export const metricLabel = (m: string) => METRIC_LABEL[m as Metric] ?? m

/** Read a metric off a grid row, on the fixed evaluation set or all pairs. */
export function metricOf(r: CalibRow, m: Metric, allPairs: boolean): number | null {
  switch (m) {
    case 'Deming intercept': return allPairs ? r.adb : r.db
    case 'Deming slope': return allPairs ? r.adm : r.dm
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
  for (const s of meta.seasons) {
    bySeason.set(s.name, s.color)
    // per-site calendars qualify names ("Beijing · Winter (Dec–Feb)"); calibration runs use the bare season
    if (s.site) bySeason.set(s.name.slice(s.site.length + 3), s.color)
  }
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
