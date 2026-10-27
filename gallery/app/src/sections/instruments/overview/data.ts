/**
 * What the Overview's charts share: the instruments (sites.json), their daily files as days under
 * the page's screen and span, and the channel and method colours. The arithmetic is in ./stats.ts.
 */
import { useEffect, useMemo, useState } from 'react'
import * as d3 from 'd3'
import { themed } from '@/lib/theme'
import type { FilterRow, MetaFile } from '@/lib/types'
import { seasonsForSite } from '@/siteSeasons'
import { loadJson, useExport } from '@/lib/data'
import { loadAethDaily, useAethSites, type AethDaily, type AethSite } from '../lib'
import type { EpisodeColumns, HourlyFile } from './hourly'
import { CHANNELS, toDays, type Channel, type Day, type Screen, type ScreenCount } from './stats'

export { CHANNELS, type Channel, type Day, type Screen }

/**
 * Channel colours of research/ftir_hips_chem/scripts/plotting/timeseries.py (the notebooks'
 * wavelength_colors); IR's slate is lifted on the dark page. Read at render time (themed).
 */
const CHANNEL_LIGHT: Record<Channel, string> = { UV: '#9B59B6', Blue: '#3498DB', Green: '#2ECC71', Red: '#E74C3C', IR: '#34495E' }
export const channelColor = (c: Channel) => (c === 'IR' ? themed(CHANNEL_LIGHT.IR, '#a9bdd2') : CHANNEL_LIGHT[c])
/** "IR (880 nm)" */
export const channelName = (c: Channel, meta: MetaFile) => `${c} (${meta.wavelengths_nm[c] ?? '?'} nm)`
/** the per-filter field export_aethalometer.py merges onto the rows for a channel */
export const filterField = (c: Channel) => `${c} BCc (aethalometer)`

/**
 * addis_02.5_temporal_patterns_filter's three measurements of the same filters, its colours.
 * The aethalometer's is the overlap-weighted mean of the daily bins over the filter's own sampling
 * window (the notebook matched one 9 AM day within ±1 day); HIPS BC is Fabs / MAC.
 */
export const FILTER_METHODS = [
  { key: 'aeth', label: 'Aethalometer IR BCc', field: filterField('IR'), color: '#1f77b4' },
  { key: 'ftir', label: 'FTIR EC', field: 'EC (FTIR)', color: '#d62728' },
  { key: 'hips', label: 'HIPS Fabs / MAC', field: 'HIPS BC', color: '#2ca02c' },
] as const

export interface Instrument {
  /** the daily file's code: a site code, or ETAD-JACROS for the longer Addis file of the same MA350-0238 (the Jacros file) */
  code: string
  /** the SPARTAN site it stands at, and its gallery name */
  siteCode: string
  site: string
  label: string
  info: AethSite
}

/** Every instrument with a daily file, at a site the gallery knows. */
export function useInstruments(meta: MetaFile): Instrument[] | null {
  const sites = useAethSites()
  return useMemo(() => {
    if (!sites) return null
    return Object.entries(sites.sites).flatMap(([code, info]) => {
      const siteCode = code.split('-')[0]
      const site = meta.sites.find((s) => s.code === siteCode)?.name
      if (!site) return []
      const second = code !== siteCode
      return [{ code, siteCode, site, info, label: second ? `${site}, longer record (${info.site} file, same instrument)` : `${site} (${info.file})` }]
    })
  }, [sites, meta.sites])
}

/** The daily files of these instruments, loaded as they are asked for (null: failed). */
export function useDailyFiles(codes: readonly string[]): Record<string, AethDaily | null> {
  const [files, setFiles] = useState<Record<string, AethDaily | null>>({})
  const key = [...codes].sort().join(',')
  useEffect(() => {
    let live = true
    for (const code of key ? key.split(',') : []) {
      loadAethDaily(code).then((d) => { if (live) setFiles((f) => (f[code] === d ? f : { ...f, [code]: d })) })
        .catch(() => { if (live) setFiles((f) => ({ ...f, [code]: null })) })
    }
    return () => { live = false }
  }, [key])
  return files
}

/** 'filters': the days inside the site's filters in view (±1 day); 'record': the instrument's whole file */
export type Span = 'filters' | 'record'

export interface Series {
  inst: Instrument
  /** days after the screen and the span */
  days: Day[]
  /** the file's bins before the span */
  bins: number
  dropped: Record<Channel, ScreenCount>
  /** the site's filters in view (all rows at the site, with or without aethalometer values) */
  rows: FilterRow[]
  /** the dates the span keeps (null: no limit) */
  span: [string | null, string | null]
}

const shift = (date: string, days: number) => new Date(Date.parse(`${date}T00:00:00Z`) + days * 864e5).toISOString().slice(0, 10)

/** The dates [lo, hi] a span keeps at a site: its filters' first and last sample date ±1 day, or no limit. */
export function spanDates(siteRows: readonly FilterRow[], span: Span): [string | null, string | null] {
  if (span !== 'filters' || !siteRows.length) return [null, null]
  const dates = siteRows.map((r) => r.date).sort()
  return [shift(dates[0], -1), shift(dates[dates.length - 1], 1)]
}

/** One instrument's days, dated, seasoned on the site's own calendar, screened and clipped. */
export function buildSeries(inst: Instrument, file: AethDaily, meta: MetaFile, rows: readonly FilterRow[], screen: Screen, span: Span): Series {
  const siteRows = rows.filter((r) => r.site === inst.site)
  const { days, dropped } = toDays(file, seasonsForSite(meta, inst.site), screen)
  const [lo, hi] = spanDates(siteRows, span)
  const kept = days.filter((d) => (!lo || d.date >= lo) && (!hi || d.date <= hi))
  return { inst, days: kept, bins: days.length, dropped, rows: siteRows, span: [lo, hi] }
}

/** the series for these instruments, once their files are in (missing files are left out) */
export function useSeries(insts: readonly Instrument[], meta: MetaFile, rows: readonly FilterRow[], screen: Screen, span: Span) {
  const files = useDailyFiles(insts.map((i) => i.code))
  const series = useMemo(() => insts.flatMap((inst) => {
    const f = files[inst.code]
    return f ? [buildSeries(inst, f, meta, rows, screen, span)] : []
  }), [insts, files, meta, rows, screen, span])
  const loading = insts.some((i) => !(i.code in files))
  const failed = insts.filter((i) => files[i.code] === null).map((i) => i.code)
  return { series, loading, failed }
}

/** the screen, as the controls and subtitles name it */
export const SCREEN_LABEL: Record<Screen, string> = {
  none: 'Keep every day',
  neg: 'Drop negative days',
  '3sd': 'addis_02 screen',
}
export const SCREEN_TEXT: Record<Screen, string> = {
  none: 'every daily value as exported',
  neg: 'negative daily BCc dropped',
  '3sd': 'negative daily BCc and values above the channel\'s mean + 3 SD dropped (addis_02\'s screen)',
}

/** a site's colour from meta, as on every tab */
export const siteColor = (meta: MetaFile, site: string) => meta.sites.find((s) => s.name === site)?.color ?? '#7b8794'
/** an instrument's colour: its site's, a step darker for a site's second daily file (the longer Jacros record) */
export const instColor = (meta: MetaFile, inst: Instrument) =>
  inst.code === inst.siteCode ? siteColor(meta, inst.site) : d3.color(siteColor(meta, inst.site))?.darker(0.9).formatHex() ?? siteColor(meta, inst.site)

/** a log axis's tick label: only at 1, 2 and 5 × 10ⁿ, so the in-between grid lines stay unlabelled */
export const logTick = (v: number) => {
  const m = v / 10 ** Math.floor(Math.log10(v) + 1e-9)
  return [1, 2, 5].some((k) => Math.abs(m - k) < 1e-6) ? String(+v.toPrecision(3)) : ''
}

export const DOW = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
export const MONTH_LETTER = 'JFMAMJJASOND'.split('')
export const MONTH_NAME = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

/** overview/index.json (gallery/data/export_aeth_overview.py) */
export interface OverviewSite {
  site: string; resolution: 'minute' | 'hourly'; hourly_file: string; minute_file: string | null; offset: number
  first: string; last: string; hours: number; valid_hours: number
  /** the record's 95th-percentile hour, ng/m³: the episode threshold */
  threshold: number; traced: number
  episodes: EpisodeColumns
}
export interface Exclusion { date: string; filter_id: string | null; reason: string }
export interface OverviewIndex {
  generated: string; source: string; note: string
  rules: { min_minutes: number; episode_quantile: number; episode_min_hours: number; trace_pad_hours: number }
  sites: Record<string, OverviewSite>
  /** outliers.EXCLUDED_SAMPLES by site code */
  exclusions: Record<string, Exclusion[]>
}
/** overview/traces/<CODE>.json: minute IR (ng/m³) around the top episodes, from t0 (local standard) */
export interface TraceFile { site: string; unit: string; pad_hours: number; traces: { episode: number; t0: string; IR: (number | null)[] }[] }

export const useOverviewIndex = () => useExport<OverviewIndex>('aethalometer/overview/index.json')
export const loadHourly = (code: string) => loadJson<HourlyFile>(`aethalometer/overview/hourly/${code}.json`)
export const loadTraces = (code: string) => loadJson<TraceFile>(`aethalometer/overview/traces/${code}.json`)

/** one exported file per site code, loaded as asked for (null: failed) */
export function useSiteFiles<T>(codes: readonly string[], load: (code: string) => Promise<T>): Record<string, T | null> {
  const [files, setFiles] = useState<Record<string, T | null>>({})
  const key = [...new Set(codes)].sort().join(',')
  useEffect(() => {
    let live = true
    for (const code of key ? key.split(',') : []) {
      load(code).then((d) => { if (live) setFiles((f) => (f[code] === d ? f : { ...f, [code]: d })) })
        .catch(() => { if (live) setFiles((f) => ({ ...f, [code]: null })) })
    }
    return () => { live = false }
  }, [key]) // eslint-disable-line react-hooks/exhaustive-deps
  return files
}
