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

/**
 * The AERONET level the tab reads: Level 1.5 (what every tab's AERONET fields are) or Level 2.0, exported
 * beside it in aeronet/lev20/ with the same files. The tab's own toggle; the URL key is `aerlev`.
 */
export type AeronetLevel = '15' | '20'
export const LEVELS: readonly AeronetLevel[] = ['15', '20']
export const LEVEL_PARAM = 'aerlev'
export const levelLabel = (l: AeronetLevel) => (l === '20' ? 'Level 2.0' : 'Level 1.5')
const levelDir = (l: AeronetLevel) => (l === '20' ? 'aeronet/lev20' : 'aeronet')
export const loadDailyAeronet = (code: string, level: AeronetLevel = '15') => loadJson<DailyAeronet>(`${levelDir(level)}/daily/${code}.json`)
export const stationsPath = (level: AeronetLevel) => `${levelDir(level)}/stations.json`

/**
 * The per-filter AERONET values for the sites in view: the index once, then one file per site
 * (aeronet/filters/<CODE>.json), as useWeather does. Optional: a site without a file has none.
 */
export function useAeronet(codes: readonly string[], level: AeronetLevel = '15'): AeronetFile | null {
  const index = useExport<AeronetIndex>(`${levelDir(level)}/index.json`).data
  const key = index ? [...new Set(codes)].filter((c) => index.sites[c]).sort().join(',') : ''
  const [state, setState] = useState<{ key: string; level: AeronetLevel; data: AeronetFile } | null>(null)
  useEffect(() => {
    if (!index) return
    let live = true
    const want = key ? key.split(',') : []
    Promise.all(want.map((c) => loadJson<Pick<AeronetFile, 'rows' | 'counts'>>(`${levelDir(level)}/filters/${c}.json`)))
      .then((parts) => {
        if (!live) return
        const { sites: _sites, ...rest } = index
        setState({ key, level, data: { ...rest, rows: Object.assign({}, ...parts.map((p) => p.rows)), counts: Object.assign({}, ...parts.map((p) => p.counts)) } })
      })
      .catch(() => { /* AERONET is optional; the rows go without it */ })
    return () => { live = false }
  }, [index, key, level])
  // while another level loads, nothing rather than the other level's values
  return state && state.level === level ? state.data : null
}

/**
 * The rows with their AERONET fields read from another level's file (the tab's Level toggle): every AERONET
 * field is replaced, missing where that level has no value, and the reader's ratios that use an AERONET
 * field ("HIPS BC / AOD 500 nm (AERONET)") are recomputed from the new values.
 */
export function withLevel(rows: FilterRow[], file: AeronetFile | null, ratioFields: readonly string[]): FilterRow[] {
  if (!file) return rows
  const own = new Set(file.fields)
  const ratios = ratioFields.flatMap((f) => {
    const [num, den] = f.split(' / ')
    return den && (own.has(num) || own.has(den)) ? [{ f, num, den }] : []
  })
  return rows.map((r) => {
    const v = file.rows[r.id]
    const out: FilterRow = { ...r }
    file.fields.forEach((f, i) => { out[f] = v ? v[i] : null })
    for (const q of ratios) {
      const a = out[q.num], b = out[q.den]
      out[q.f] = typeof a === 'number' && typeof b === 'number' && b !== 0 ? a / b : null
    }
    return out
  })
}

export const AOD500 = 'AOD 500 nm (AERONET)'
export const AAOD675 = 'AAOD 675 nm (AERONET)'
export const AAE = 'AAE 440-870 (AERONET)'
export const SSA440 = 'SSA 440 nm (AERONET)'
export const FMF = 'Fine-mode fraction 500 nm (AERONET)'
export const INV_AOD440 = 'Inversion AOD 440 nm (AERONET)'
export const AOD440 = 'AOD 440 nm (AERONET)'
export const AE = 'Angstrom exponent 440-870 (AERONET)'
export const PW = 'Precipitable water (AERONET)'
export const FINE_AOD = 'Fine-mode AOD 500 nm (AERONET)'
export const COARSE_AOD = 'Coarse-mode AOD 500 nm (AERONET)'
export const AAE_440_675 = 'AAE 440-675 (AERONET)'
export const SAE_440_675 = 'SAE 440-675 (AERONET)'
export const EAE_INV = 'Inversion extinction AE 440-870 (AERONET)'
export const DAOD = 'dAOD 440-870 (AERONET)'
export const DSSA = 'dSSA 440-870 (AERONET)'
export const AEROSOL_TYPE = 'Aerosol type class (AERONET)'
/** the inversion's single-scattering albedo by wavelength (nm) */
export const SSA_BY_WAVELENGTH: readonly [number, string][] = [
  [440, 'SSA 440 nm (AERONET)'], [675, 'SSA 675 nm (AERONET)'], [870, 'SSA 870 nm (AERONET)'], [1020, 'SSA 1020 nm (AERONET)'],
]

/**
 * HIPS filter lots by sampling date where the repo documents them (research/ftir_ec_phase3/
 * SPARTAN_LOT_INVENTORY_2026-08-23.md, from SPARTAN_HIPS_Batch1-51.v2.csv LotId): Addis only, to the month.
 * The per-filter LotId is not in the gallery's rows, so a month two lots share is a period of its own.
 */
export const HIPS_LOT_PERIODS: Record<string, { label: string; from: string; to: string }[]> = {
  ETAD: [
    { label: 'Lot 248', from: '2022-12-01', to: '2023-02-28' },
    { label: 'Lot 248 or 251 (changeover month)', from: '2023-03-01', to: '2023-03-31' },
    { label: 'Lot 251', from: '2023-04-01', to: '2025-08-31' },
    { label: 'Lot 253', from: '2025-11-01', to: '2026-01-31' },
  ],
}
export const lotPeriodOf = (code: string, date: string) =>
  HIPS_LOT_PERIODS[code]?.find((p) => date >= p.from && date <= p.to)?.label ?? null

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
 * else SPARTAN's public BC × MAC. That product is Fabs / 10, and the gallery's rows carry it rescaled to
 * the reader's MAC (lib/viewFields), so `mac` is the MAC the rows were viewed at (lib/mac useMac).
 */
export function surfaceFabs(r: FilterRow, mac = 10): number | null {
  const f = r['HIPS Fabs']
  if (typeof f === 'number') return f
  const bc = r['BC (public HIPS)']
  return typeof bc === 'number' ? bc * mac : null
}

/**
 * Effective absorption scale height (m): column AAOD at 675 nm over surface Fabs at 633 nm.
 * `mac`: as for surfaceFabs. Not a boundary-layer height; it compares sites measured the same way (research/ftir_ec_phase3/
 * AERONET_SPARTAN_2026-08-22.md). Null without both, or with a non-positive Fabs.
 */
export function scaleHeight(r: FilterRow, mac = 10): number | null {
  const aaod = r[AAOD675], fabs = surfaceFabs(r, mac)
  return typeof aaod === 'number' && fabs !== null && fabs > 0 ? aaod / (fabs * 1e-6) : null
}
