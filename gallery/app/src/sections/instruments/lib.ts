/**
 * The aethalometer data (gallery/data/export_aethalometer.py): per-filter values merged into the rows
 * as fields ("IR BCc (aethalometer)" …), the daily bins per site, and the site list. Loaded only by the
 * Instruments section, so a public build that leaves the section out never fetches it.
 */
import { useEffect, useState } from 'react'
import { loadJson, useExport } from '@/lib/data'
import type { SectionFields } from '../types'

export const AETH_GROUP = 'Aethalometer (sampling window)'

export interface AethIndex {
  source: string; note: string
  fields: string[]; field_units: Record<string, string>; count_fields: string[]
  /** site code -> filters with values */
  sites: Record<string, number>
}
export interface AethFilters {
  rows: Record<string, (number | null)[]>
  /** id -> [overlap hours, daily bins] */
  counts: Record<string, [number, number]>
}
/** daily/<CODE>.json: one entry per 24 h bin, labelled at its start (local) */
export interface AethDaily {
  start: string[]
  flow: (string | null)[]
  UV: (number | null)[]; Blue: (number | null)[]; Green: (number | null)[]; Red: (number | null)[]; IR: (number | null)[]
  IR_smoothed: (number | null)[]
  AAE_UV_IR: (number | null)[]; AAE_Blue_IR: (number | null)[]
  Red_babs: (number | null)[]
  coverage: (number | null)[]
}
export interface AethSite {
  site: string; file: string; bins: number; first: string; last: string
  bin_start_hours: Record<string, number>; matched_filters: number; coverage_field: boolean
}
export interface AethSites { source: string; note: string; sites: Record<string, AethSite> }

export const loadAethDaily = (code: string) => loadJson<AethDaily>(`aethalometer/daily/${code}.json`)
export const useAethSites = () => useExport<AethSites>('aethalometer/sites.json').data

/** the per-filter fields for the sites in view (the useWeather / useAeronet pattern) */
export function useAethFields(codes: readonly string[]): (SectionFields & { counts: AethFilters['counts'] }) | null {
  const index = useExport<AethIndex>('aethalometer/index.json').data
  const key = index ? [...new Set(codes)].filter((c) => index.sites[c]).sort().join(',') : ''
  const [state, setState] = useState<{ key: string; data: SectionFields & { counts: AethFilters['counts'] } } | null>(null)
  useEffect(() => {
    if (!index) return
    let live = true
    const want = key ? key.split(',') : []
    Promise.all(want.map((c) => loadJson<AethFilters>(`aethalometer/filters/${c}.json`)))
      .then((parts) => {
        if (!live) return
        setState({ key, data: { group: AETH_GROUP, fields: index.fields, field_units: index.field_units,
          rows: Object.assign({}, ...parts.map((p) => p.rows)), counts: Object.assign({}, ...parts.map((p) => p.counts)) } })
      })
      .catch(() => { /* optional, like the weather */ })
    return () => { live = false }
  }, [index, key])
  return state?.data ?? null
}
