/**
 * The Data quality page's data: the daily bins (aethalometer/daily, gallery/data/export_aethalometer.py) joined to
 * the instrument's QC columns for the same bins (aethalometer/quality, gallery/data/export_aeth_quality.py), the
 * flow-fix periods and thresholds from research/ftir_hips_chem/scripts/config.py, and the small helpers the
 * charts share. Bins are 24 h, labelled at their (local) start; times are kept as UTC dates of the local clock.
 */
import { useEffect, useState } from 'react'
import * as d3 from 'd3'
import { loadJson, useExport } from '@/lib/data'
import type { MetaFile } from '@/lib/types'
import { loadAethDaily, type AethDaily } from '../lib'

export const CHANNELS = ['UV', 'Blue', 'Green', 'Red', 'IR'] as const
export type Channel = (typeof CHANNELS)[number]

export interface FlowPeriods { before_end: string | null; after_start: string | null; has_before_data: boolean; description: string; notes: string }
export interface QualityIndex {
  source: string
  /** config.FLOW_FIX_PERIODS by gallery site code */
  flow_periods: Record<string, FlowPeriods>
  /** config.SMOOTH_RAW_THRESHOLDS (%) */
  smooth_raw_thresholds: number[]
  /** the defaults of src/data/qc/pkl_cleaning.py's cleaning steps */
  cleaning_bounds: {
    flow_ratio: { lower: number; upper: number }
    flow_range: { flow_threshold: number; setpoint: number }
    leak_ratio: { lower_bound: number; upper_bound: number }
    bcc_ratio: { lower_bound: number; upper_bound: number }
    bcc_denominator: { threshold: number; threshold_IR: number }
  }
  /** series key (without its channel) -> what it is */
  notes: Record<string, string>
  sites: Record<string, { site: string; file: string; bins: number; series: string[]; channels: Channel[] }>
}
interface QualityFile { start: string[]; series: Record<string, (number | null)[]> }

export const useQualityIndex = () => useExport<QualityIndex>('aethalometer/quality/index.json')

/** the five instruments, in the order the page lists them: the four filter sites, then Jacros (a second MA350 at Addis) */
export const INSTRUMENTS = ['CHTS', 'INDH', 'USPA', 'ETAD', 'ETAD-JACROS'] as const
export const JACROS = 'ETAD-JACROS'
export function instrumentName(code: string, meta: MetaFile) {
  if (code === JACROS) return 'Jacros (Addis)'
  return meta.sites.find((s) => s.code === code)?.name ?? code
}
/** a site's colour from meta (config.SITES); Jacros, not a filter site, a darker step of Addis's */
export function instrumentColor(code: string, meta: MetaFile): string {
  if (code === JACROS) return d3.color(instrumentColor('ETAD', meta))?.darker(0.9).formatHex() ?? '#9a5b00'
  return meta.sites.find((s) => s.code === code)?.color ?? '#7b8794'
}

/** one daily bin with everything the page draws */
export interface Day {
  i: number
  start: string
  /** the bin's local start as a UTC date (so the axis reads the local clock) */
  t: Date
  flow: string | null
  bc: Record<Channel, number | null>
  irSmoothed: number | null
  coverage: number | null
  /** QC series by key ('flow1', 'IR.ATN1' …); missing keys and values are null */
  q: (key: string) => number | null
}

export const localDate = (s: string) => new Date(`${s.slice(0, 10)}T${s.length > 10 ? s.slice(11, 16) : '00:00'}:00Z`)

function join(daily: AethDaily, quality: QualityFile | null): Day[] {
  const at = new Map(quality?.start.map((s, i) => [s, i]) ?? [])
  return daily.start.map((s, i) => {
    const qi = at.get(s)
    return {
      i, start: s, t: localDate(s), flow: daily.flow[i],
      bc: { UV: daily.UV[i], Blue: daily.Blue[i], Green: daily.Green[i], Red: daily.Red[i], IR: daily.IR[i] },
      irSmoothed: daily.IR_smoothed[i], coverage: daily.coverage[i],
      q: (key: string) => (qi === undefined ? null : quality?.series[key]?.[qi] ?? null),
    }
  })
}

/** every instrument's joined days, loaded once (null while loading); a site whose files are missing is left out */
export function useAllDays(): Record<string, Day[]> | null {
  const [state, setState] = useState<Record<string, Day[]> | null>(null)
  useEffect(() => {
    let live = true
    Promise.all(INSTRUMENTS.map((code) => Promise.all([
      loadAethDaily(code),
      loadJson<QualityFile>(`aethalometer/quality/${code}.json`).catch(() => null),
    ]).then(([d, q]) => [code, join(d, q)] as const).catch(() => null)))
      .then((parts) => { if (live) setState(Object.fromEntries(parts.filter((p): p is readonly [typeof INSTRUMENTS[number], Day[]] => p !== null))) })
    return () => { live = false }
  }, [])
  return state
}

/** flow_periods.classify_flow_period on an ISO date: before ≤ before_end, after ≥ after_start, else the gap */
export type Period = 'before' | 'gap' | 'after' | 'no_fix'
export function periodOf(date: string, p: FlowPeriods | undefined): Period {
  if (!p || (!p.before_end && !p.after_start)) return 'no_fix'
  const d = date.slice(0, 10)
  if (p.before_end && d <= p.before_end) return 'before'
  if (p.after_start && d >= p.after_start) return 'after'
  return 'gap'
}
/** the period colours of plotting/comparisons.py flow_periods (before red, after green, gap grey) */
export const PERIOD_COLOR: Record<Period, string> = { before: '#E74C3C', gap: '#95A5A6', after: '#2ECC71', no_fix: '#3498DB' }
export const PERIOD_LABEL: Record<Period, string> = { before: 'Before the fix', gap: 'Between (gap)', after: 'After the fix', no_fix: 'No fix defined' }

/** the q-th quantile of finite values (linear, as numpy's default) */
export function quantile(values: number[], q: number): number | null {
  const v = values.filter(Number.isFinite).sort((a, b) => a - b)
  return v.length ? d3.quantileSorted(v, q) ?? null : null
}
export const finite = (v: number | null | undefined): v is number => v != null && Number.isFinite(v)
export const ratio = (a: number | null | undefined, b: number | null | undefined) => (finite(a) && finite(b) && b !== 0 ? a / b : null)
