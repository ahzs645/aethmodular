/**
 * AERONET over each filter's sampling window (gallery/data/export_aeronet.py): the mean of the
 * sun photometer's individual retrievals inside the window, at the station chosen for each SPARTAN
 * site. App merges the per-filter values into the rows as ordinary fields ("AOD 500 nm (AERONET)" …),
 * as it does the weather, so every chart can plot them; the AERONET tab also reads the daily series
 * and the station list from here.
 */
import { useEffect, useState } from 'react'
import { loadJson, useExport } from './data.ts'
import type { FilterRow } from './types.ts'

export interface AeronetFile {
  source: string
  note: string
  level: string
  fields: string[]
  field_units: Record<string, string>
  count_fields: string[]
  /** id -> values in `fields` order */
  rows: Record<string, (number | null)[]>
  /** id -> [AOD retrievals, days with AOD, inversions] inside the window */
  counts: Record<string, [number, number, number]>
}

/** aeronet/index.json: the fields, units and sources, and how many filters each site's file holds */
export interface AeronetIndex extends Omit<AeronetFile, 'rows' | 'counts'> { sites: Record<string, number> }

export interface AeronetStation {
  station: string
  km: number
  filters: number
  with_aod: number
  with_inversion: number
  candidates: { station: string; km: number; window_days_with_aod: number }[]
}
export interface AeronetStations { level: string; source: string; sites: Record<string, AeronetStation> }

export interface DailyAeronet {
  date: string[]
  aod440: (number | null)[]; aod500: (number | null)[]; aod675: (number | null)[]
  ae: (number | null)[]; pw: (number | null)[]; fmf: (number | null)[]
  aaod440: (number | null)[]; aaod675: (number | null)[]; aae: (number | null)[]
  ssa440: (number | null)[]; ssa675: (number | null)[]
}

export const AERONET_GROUP = 'AERONET (sampling window)'
export const loadDailyAeronet = (code: string) => loadJson<DailyAeronet>(`aeronet/daily/${code}.json`)

/**
 * The per-filter AERONET values for the sites in view: the index once, then one file per site
 * (aeronet/filters/<CODE>.json), as useWeather does. Optional: a site without a file has none.
 */
export function useAeronet(codes: readonly string[]): AeronetFile | null {
  const index = useExport<AeronetIndex>('aeronet/index.json').data
  const key = index ? [...new Set(codes)].filter((c) => index.sites[c]).sort().join(',') : ''
  const [state, setState] = useState<{ key: string; data: AeronetFile } | null>(null)
  useEffect(() => {
    if (!index) return
    let live = true
    const want = key ? key.split(',') : []
    Promise.all(want.map((c) => loadJson<Pick<AeronetFile, 'rows' | 'counts'>>(`aeronet/filters/${c}.json`)))
      .then((parts) => {
        if (!live) return
        const { sites: _sites, ...rest } = index
        setState({ key, data: { ...rest, rows: Object.assign({}, ...parts.map((p) => p.rows)), counts: Object.assign({}, ...parts.map((p) => p.counts)) } })
      })
      .catch(() => { /* AERONET is optional; the rows go without it */ })
    return () => { live = false }
  }, [index, key])
  return state?.data ?? null
}

export const AOD500 = 'AOD 500 nm (AERONET)'
export const AAOD675 = 'AAOD 675 nm (AERONET)'
export const AAE = 'AAE 440-870 (AERONET)'
export const SSA440 = 'SSA 440 nm (AERONET)'
export const FMF = 'Fine-mode fraction 500 nm (AERONET)'
export const INV_AOD440 = 'Inversion AOD 440 nm (AERONET)'
/** AERONET's Level 2 absorption threshold: below it the inversion's AAOD, SSA and AAE are uncertain */
export const L2_AOD440 = 0.4

const AOD_FIELDS = ['AOD 440 nm (AERONET)', AOD500, 'AOD 675 nm (AERONET)', 'AOD 870 nm (AERONET)']
/** A filter with a direct-sun AOD in its window, at any wavelength (some photometers lack 500 nm at times). */
export const hasAod = (r: FilterRow) => AOD_FIELDS.some((f) => typeof r[f] === 'number')

/** The quantities the tab's summary shows per site, in order. */
export const AERONET_SUMMARY = [
  { field: AOD500, label: 'AOD 500', digits: 2 },
  { field: FMF, label: 'Fine fraction', digits: 2 },
  { field: AAOD675, label: 'AAOD 675', digits: 3 },
  { field: AAE, label: 'AAE', digits: 2 },
  { field: SSA440, label: 'SSA 440', digits: 2 },
] as const

/**
 * The filter's surface absorption at the HIPS wavelength (Mm⁻¹): HIPS Fabs where the filter has it,
 * else SPARTAN's public BC × 10 (that product is Fabs / 10).
 */
export function surfaceFabs(r: FilterRow): number | null {
  const f = r['HIPS Fabs']
  if (typeof f === 'number') return f
  const bc = r['BC (public HIPS)']
  return typeof bc === 'number' ? bc * 10 : null
}

/**
 * Effective absorption scale height (m): column AAOD at 675 nm over surface Fabs at 633 nm.
 * Not a boundary-layer height; it compares sites measured the same way (research/ftir_ec_phase3/
 * AERONET_SPARTAN_2026-08-22.md). Null without both, or with a non-positive Fabs.
 */
export function scaleHeight(r: FilterRow): number | null {
  const aaod = r[AAOD675], fabs = surfaceFabs(r)
  return typeof aaod === 'number' && fabs !== null && fabs > 0 ? aaod / (fabs * 1e-6) : null
}
