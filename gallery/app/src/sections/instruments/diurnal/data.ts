/**
 * The Diurnal page's data (gallery/data/export_aeth_diurnal.py → public/data/aethalometer/diurnal/):
 * the column (AERONET retrievals per local day-hour, every SPARTAN site with a station), the surface
 * (MA350 hourly means per day-hour, only where the raw minute files were exported) and the ftir_28
 * brown-carbon table. Tables come as columns with day steps (`dd`, cumulated here into dates) so the
 * page can bin days by whichever season calendar is active, and filter them by date.
 */
import * as d3 from 'd3'
import { useEffect, useState } from 'react'
import { loadJson, useExport } from '@/lib/data'
import { seasonOfDate } from '@/lib/seasons'
import type { SeasonMeta } from '@/lib/types'
import { themed } from '@/lib/theme'

export const DIURNAL_DIR = 'aethalometer/diurnal'
export const EXPORT_COMMAND = 'uv run --with timezonefinder python gallery/data/export_aeth_diurnal.py'
export const HOURS = d3.range(24)
export const CHANNELS = ['UV', 'Blue', 'Green', 'Red', 'IR'] as const
export type Channel = (typeof CHANNELS)[number]
/** the MA350 channels drawn as in ftir_28 (violet to near-infrared ink); Red and IR get a lighter step on the dark page */
export const CHANNEL_COLOUR: Record<Channel, string> = { UV: '#7A4FA3', Blue: '#2C6E9E', Green: '#3C7A52', get Red() { return themed('#B23327', '#e0695c') }, get IR() { return themed('#55595f', '#b4bac2') } }
/** an hour is drawn when it has at least this many day-hours behind it */
export const MIN_DAYS = 5
/** a day-hour of MA350 minutes counts toward the cycles with at least this many valid IR minutes */
export const MIN_MINUTES = 30

type Col = (number | null)[]
interface DayTable { dd: number[] }

export interface ColumnIndex {
  generated: string; level: string; source: string; note: string
  sites: Record<string, {
    station: string; offset: number; days: number; first: string; last: string; retrievals: number; inversions: number; inv_hours: number[]
    /** direct-sun retrievals and inversions by local hour */
    hour_counts: { aod: number[]; inv: number[] }
    /** median number of hours with a retrieval, over the days with any */
    hours_per_day: number
  }>
}
export interface ColumnFile {
  site: string; station: string; offset: number; first: string; last: string; days: number
  scale: { aod500: number; ae: number; fmf: number }
  aod: DayTable & { h: number[]; n: number[]; aod500: Col; ae: Col; fmf: Col }
  inv: DayTable & { h: number[]; n: number[] }
  inv_hours: number[]
}
export interface SurfaceIndex {
  generated: string; source: string; note: string
  /** the minute files looked for under aethalometry_dir(), per site */
  searched: Record<string, string[]>
  sites: Record<string, { file: string; resolution: 'minute' | 'hourly'; records: number; days: number; first: string; last: string; matched: boolean; coverage_hours: number[] }>
}
export interface SurfaceFile {
  site: string; file: string; offset: number; first: string; last: string; days: number
  /** minute records (cycles, matching, within-day CV) or hourly means (cycles and coverage only) */
  resolution: 'minute' | 'hourly'
  /** channels are integers in this unit */
  bcc_unit: 'ng/m3'
  /** n: valid IR minutes behind each day-hour (minute records only) */
  hourly: DayTable & { h: number[]; n?: (number | null)[]; aae_uv_ir?: Col; aae_blue_ir?: Col } & Partial<Record<Channel, Col>>
  matched?: DayTable & { h: number[]; n: number[]; bc_ir: Col; aod500: Col; ae: Col; aae: Col; bc_aod: Col }
  intraday?: DayTable & { n: number[]; bc_cv: Col; aod_cv: Col }
  coverage?: DayTable & { hours: number[]; minutes: number[]; minutes_in: number[]; ratio_mean: Col; ratio_median: Col }
}
export interface BrcFile {
  generated: string; source: string
  sigma_atn: Record<Channel, number>; wavelengths_nm: Record<Channel, number>
  date: string[]
  b: Record<Channel, Col>
  aae: Record<'UV' | 'Blue' | 'Green' | 'Red', Col>
  brc: Col; uv_out: boolean[]
  summary: {
    days: number
    aae_red_ir: { mean: number; sd: number; median: number; above_anchor_pct: number }
    brc: { mean: number; median: number; negative_pct: number; at_target_pct: number }
    target: { mean: number; min: number; max: number; setups: { setup: string; intercept: number; slope: number; C: number }[] }
    closing_aae: { mean_target: number; per_setup: number[] }
    uv: { out_of_range_pct: number; nonpositive: number; below_ir: number; r_ir: number }
    green: { aae_mean: number; aae_median: number; negative_pct: number; deficit_pct: number; r_ir: number }
    red: { iqr_halfwidth_pct: number; bcc_ratio_median: number; r_ir: number }
    blue: { aae_mean: number; r_ir: number }
  }
}

export const useColumnIndex = () => useExport<ColumnIndex>(`${DIURNAL_DIR}/aeronet/index.json`)
/** absent until the exporter has run where the Drive is mounted: a missing file is a state, not an error */
export const useSurfaceIndex = () => useExport<SurfaceIndex>(`${DIURNAL_DIR}/aeth/index.json`)
export const useBrc = () => useExport<BrcFile>(`${DIURNAL_DIR}/brc_test.json`)

/** one site's file (null while loading or when the site has none) */
export function useSiteFile<T>(part: 'aeronet' | 'aeth', code: string | null): T | null {
  const [state, setState] = useState<{ code: string; data: T } | null>(null)
  useEffect(() => {
    if (!code) return
    let live = true
    loadJson<T>(`${DIURNAL_DIR}/${part}/${code}.json`).then((data) => { if (live) setState({ code, data }) }).catch(() => { /* none for this site */ })
    return () => { live = false }
  }, [part, code])
  return state && state.code === code ? state.data : null
}

const DAY = 86_400_000
/** a table's dates, from its first-date offset and day steps */
export function dates(first: string, t: DayTable): string[] {
  const t0 = Date.parse(`${first}T00:00:00Z`)
  let k = 0
  return t.dd.map((step) => { k += step; return new Date(t0 + k * DAY).toISOString().slice(0, 10) })
}

/** the season of each date under a site's calendar ('' when it has none) */
export function seasonsOf(days: string[], seasons: readonly SeasonMeta[]): string[] {
  const memo = new Map<string, string>()
  return days.map((d) => {
    if (!memo.has(d)) memo.set(d, seasonOfDate(d, +d.slice(5, 7), seasons)?.name ?? '')
    return memo.get(d)!
  })
}

export interface HourStat { hour: number; n: number; median: number | null; lo: number | null; hi: number | null; mean: number | null }
/** median, interquartile range and mean of the values in each hour (null below MIN_DAYS) */
export function byHour(hours: number[], values: Col, keep: (i: number) => boolean = () => true): HourStat[] {
  const bins: number[][] = HOURS.map(() => [])
  hours.forEach((h, i) => { const v = values[i]; if (v != null && Number.isFinite(v) && keep(i)) bins[h].push(v) })
  return bins.map((b, hour) => {
    b.sort(d3.ascending)
    const enough = b.length >= MIN_DAYS
    return { hour, n: b.length, median: enough ? d3.quantileSorted(b, 0.5) ?? null : null, lo: enough ? d3.quantileSorted(b, 0.25) ?? null : null,
      hi: enough ? d3.quantileSorted(b, 0.75) ?? null : null, mean: enough ? d3.mean(b) ?? null : null }
  })
}

/** a surface channel in µg/m³ (the file keeps ng/m³ integers) */
export const bcc = (file: SurfaceFile, c: Channel): Col => (file.hourly[c] ?? []).map((v) => (v == null ? null : v / 1000))

/** a scaled integer column back to values */
export const unscale = (c: Col, s: number): Col => c.map((v) => (v == null ? null : v / s))

export const hourLabel = (h: number) => `${String(h).padStart(2, '0')}:00`
/** consecutive hours as [first, last] runs: [7, 8, 9, 10, 14] -> [[7, 10], [14, 14]] */
export function hourSpans(hours: readonly number[]): [number, number][] {
  const runs: [number, number][] = []
  for (const h of [...hours].sort((a, b) => a - b)) {
    const last = runs[runs.length - 1]
    if (last && h === last[1] + 1) last[1] = h
    else runs.push([h, h])
  }
  return runs
}
/** "07–10, 14–17": consecutive hours run together */
export function hourRuns(hours: readonly number[]): string {
  return hourSpans(hours).map(([a, b]) => (a === b ? String(a).padStart(2, '0') : `${String(a).padStart(2, '0')}–${String(b).padStart(2, '0')}`)).join(', ')
}
export const utcLabel = (offset: number) => `UTC${offset >= 0 ? '+' : '−'}${Math.abs(offset)}`
