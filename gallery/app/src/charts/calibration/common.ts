import * as d3 from 'd3'
import type { CalibFile, CalibRow, CalibRun, FilterRow, MetaFile, MetricRow, SweepRow } from '@/lib/types'
import { baseFilterId } from '@/lib/highlight'
import { useEffect, useState } from 'react'
import { loadSpartanFilters } from '@/lib/spartan'
import { COHORT_SHORT, METRIC, PREPROCESSING, PREPROCESSING_SHORT, SELECTION_SPACE, label } from '@/lib/labels'
import { BASELINE_COLOR, INK } from '@/lib/theme'

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

/** a configuration in a few words: 'Lowest OC/EC 800 × Spline baseline' (cohort, cutoff, selection spectra if not raw, model spectra) */
export const shortConfig = (c: Config) => `${label(COHORT_SHORT, c.co)}${c.cut !== null ? ' ' + c.cut : ''}${c.sel && c.sel !== 'raw' ? ` (selected on ${label(PREPROCESSING_SHORT, c.sel).toLowerCase()})` : ''} × ${label(PREPROCESSING, c.sp)}`

/** A preset's exported label with the preprocessing in reader-facing words ("× AIRSpec" → "× Spline baseline"). */
export const presetName = (p: { label: string; spectra: string }) =>
  p.label.replace(/× (raw|AIRSpec|SG 2nd derivative|2nd derivative)(?=$|\s)/, `× ${label(PREPROCESSING, p.spectra)}`)

/** Fixed cohort colours — not in config.py, so declared once here. */
export const COHORT_COLOR: Record<string, string> = {
  get pool() { return INK.muted },
  smoke: '#8d5524',
  eth_shaped: '#E67E22',
  analogs: '#8e44ad',
  ocec: '#2b6cb0',
}
export const cohortColor = (co: string) => COHORT_COLOR[co] ?? INK.axis

/** Colour per calibration spectra space, shared by every chart that puts the preprocessings side by side. */
// the calibration grid's four spectra spaces (their order is the legend order); the spline baseline takes
// its tab-wide colour (lib/theme BASELINE_COLOR) since #2171b5 is reserved for the accent and OLS
export const SPECTRA_COLOR: Record<string, string> = { get raw() { return INK.text }, get airspec() { return BASELINE_COLOR.airspec }, deriv2: '#c026d3', get neutral() { return INK.good } }
export const spectraColor = (sp: string) => SPECTRA_COLOR[sp] ?? BASELINE_COLOR[sp] ?? INK.axis


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
export const score = (r: CalibRow, w = 5) => {
  const m = r.guardDm !== undefined ? r.guardDm : r.dm, b = r.guardDb !== undefined ? r.guardDb : r.db
  return b !== null && m !== null ? Math.abs(b) + w * Math.abs(m - 1) : null
}

export const passes = (r: CalibRow, calib: CalibFile) => {
  const m = r.guardDm !== undefined ? r.guardDm : r.dm
  return m !== null && m >= calib.slope_box[0] && m <= calib.slope_box[1] && (r.ho ?? 0) >= calib.heldout_floor
}

/** A saved fit as "0.820x − 1.060", at the agreed coefficient precision (lib/format coef, 3 dp). */
export const fmtFit = (m: number | null, b: number | null) =>
  m === null || b === null ? '—' : `${m.toFixed(3)}x ${b < 0 ? '−' : '+'} ${Math.abs(b).toFixed(3)}`

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

/**
 * A filter's deployed SPARTAN EC (the network's FTIR EC product), and where it came from: the
 * explorer's own column where it exported one (the Addis fixed set), else the gallery's in-house
 * 'EC (FTIR)' for the same filter (the same values: on the 190 Addis filters that have both they agree
 * to 5e-5 µg/m³), else the public SPARTAN release (`network`, for sites and filters the gallery's
 * focus rows lack; it differs from the in-house value by more than 0.01 µg/m³ on 19 of 542 shared filters).
 */
export const DEPLOYED_FIELD = 'EC (FTIR)'
export type DeployedSource = 'run' | 'gallery' | 'public release'
export function deployedECFrom(run: CalibRun, i: number, byId: ReadonlyMap<string, FilterRow>, network?: ReadonlyMap<string, FilterRow> | null): { v: number; from: DeployedSource } | null {
  const own = run.eval.deployed?.[i]
  if (own !== null && own !== undefined && Number.isFinite(own)) return { v: own, from: 'run' }
  const id = run.eval.id[i]
  if (!id) return null
  const key = baseFilterId(id)
  const g = byId.get(key)?.[DEPLOYED_FIELD]
  if (typeof g === 'number' && Number.isFinite(g)) return { v: g, from: 'gallery' }
  const n = network?.get(key)?.[DEPLOYED_FIELD]
  return typeof n === 'number' && Number.isFinite(n) ? { v: n, from: 'public release' } : null
}
export const deployedEC = (run: CalibRun, i: number, byId: ReadonlyMap<string, FilterRow>, network?: ReadonlyMap<string, FilterRow> | null) =>
  deployedECFrom(run, i, byId, network)?.v ?? null

/** The public SPARTAN release's rows for these site codes, by filter id (null until loaded). */
export function useNetworkRows(codes: readonly string[]): ReadonlyMap<string, FilterRow> | null {
  const key = [...new Set(codes)].sort().join(',')
  const [rows, setRows] = useState<{ key: string; map: Map<string, FilterRow> } | null>(null)
  useEffect(() => {
    if (!key) return
    let live = true
    loadSpartanFilters(key.split(',')).then((r) => { if (live) setRows({ key, map: new Map(r.rows.map((x) => [x.id, x])) }) }).catch(() => {})
    return () => { live = false }
  }, [key])
  return rows && rows.key === key ? rows.map : null
}
