/**
 * Which filters the chart tabs draw: the four focus sites (filters.json, HIPS + FTIR,
 * the default), or a whole network chosen in the Filters sidebar.
 *
 * - SPARTAN: public ChemSpec PM2.5 at every site (spartan_reference.json + spartan_map.json).
 *   EC/OC are FTIR-derived; HIPS and FTIR functional groups are not public.
 * - IMPROVE: one file per site (improve_samples/, gallery/data/export_improve_samples.py),
 *   fetched only when picked, because ~388,000 samples cannot be drawn at once.
 *
 * Network rows use the FilterRow shape, so every chart works unchanged. Only the focus
 * sites have local season calendars (siteSeasons.ts); networks use shared month bins.
 */
import { useEffect, useMemo, useState } from 'react'
import { loadJson } from './data'
import type { FilterRow, SeasonMeta, SiteMeta } from './types'

export type Network = 'focus' | 'spartan' | 'improve'
export const NETWORKS: { key: Network; label: string; note: string }[] = [
  { key: 'focus', label: 'Focus sites', note: 'Beijing, Delhi, JPL and Addis Ababa: SPARTAN filters with HIPS and FTIR' },
  { key: 'spartan', label: 'SPARTAN', note: 'Every SPARTAN site, public PM2.5 chemistry; EC/OC are FTIR-derived, no HIPS' },
  { key: 'improve', label: 'IMPROVE', note: 'Every IMPROVE site: TOR EC/OC, HIPS absorption, mass, iron, silicon, soil' },
]
export const isNetwork = (v: string | null): v is Network => v === 'focus' || v === 'spartan' || v === 'improve'

/** IMPROVE sites load per site; more than this many at once makes the SVG charts crawl. */
export const MAX_IMPROVE_SITES = 10

/** Axis and measurement defaults per network, used when the current choice has no data there. */
export const NETWORK_DEFAULTS: Record<Network, { x: string; y: string; field: string }> = {
  focus: { x: 'HIPS BC', y: 'EC (FTIR)', field: 'EC (FTIR)' },
  spartan: { x: 'PM2.5 mass', y: 'EC (FTIR)', field: 'EC (FTIR)' },
  improve: { x: 'HIPS BC', y: 'EC (TOR)', field: 'EC (TOR)' },
}

/** Shared month bins for network views: no hemisphere-specific names, so they read the same in Chile and Alaska. */
export const MONTH_QUARTERS: SeasonMeta[] = [
  { name: 'Dec–Feb', months: [12, 1, 2], color: '#6F86BA' },
  { name: 'Mar–May', months: [3, 4, 5], color: '#67A96A' },
  { name: 'Jun–Aug', months: [6, 7, 8], color: '#D99145' },
  { name: 'Sep–Nov', months: [9, 10, 11], color: '#AA6B78' },
]

/** Categorical site colours for network views (the focus sites keep their own). Cycles past 12 sites. */
const SITE_PALETTE = ['#2171b5', '#d95f0e', '#238b45', '#cb181d', '#6a51a3', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#08519c', '#a63603']
export const networkColor = (i: number) => SITE_PALETTE[i % SITE_PALETTE.length]

interface NetworkData {
  /** every site the network offers, for the site picker */
  sites: SiteMeta[]
  /** rows of the sites that are loaded (all for SPARTAN; the picked ones for IMPROVE) */
  rows: FilterRow[]
  fields: string[]
  field_units: Record<string, string>
  field_groups: { label: string; fields: string[] }[]
}

interface SpartanMapSite { code: string; name: string; lat: number | null; lon: number | null; n_filters: number; date_min: string; date_max: string }
interface SpartanRef { fields: string[]; field_units: Record<string, string>; rows: Record<string, string | number | boolean | null>[] }
interface ImproveIndex {
  fields: string[]; field_units: Record<string, string>; field_groups: { label: string; fields: string[] }[]
  sites: { code: string; name: string; state: string | null; country: string | null; lat: number | null; lon: number | null; n_filters: number; date_min: string; date_max: string }[]
}
type ImproveSite = { code: string; date: string[]; poc: number[] } & Record<string, (number | null)[] | string[] | number[] | string | number>

const COUNTRY: Record<string, string> = { CL: 'Chile', KR: 'South Korea', CA: 'Canada', MX: 'Mexico', VI: 'US Virgin Islands', PR: 'Puerto Rico' }

const SPARTAN_GROUPS = [
  { label: 'Carbon', fields: ['EC (FTIR)', 'OC (FTIR)', 'OM (FTIR)', 'EC (ChemSpec FTIR)', 'OC (ChemSpec FTIR)'] },
  { label: 'Mass', fields: ['PM2.5 mass'] },
  { label: 'Ions', fields: ['Sulfate', 'Nitrate', 'Ammonium', 'Potassium', 'Sodium', 'Calcium', 'Magnesium'] },
  { label: 'Metals & crustal', fields: ['Silicon', 'Aluminum', 'Iron', 'Titanium', 'Zinc', 'Copper', 'Lead'] },
]

const site = (code: string, name: string, location: string, lat: number | null, lon: number | null, n: number, a: string, b: string): SiteMeta =>
  ({ code, name, location, color: '#7f7f7f', timezone: '', lat: lat ?? NaN, lon: lon ?? NaN, n_filters: n, date_min: a, date_max: b })

function row(id: string, s: SiteMeta, date: string, excluded: boolean, reason: string | null, values: Record<string, number | null>): FilterRow {
  return { id, site: s.name, code: s.code, color: s.color, date, year: Number(date.slice(0, 4)) || null,
    month: Number(date.slice(5, 7)) || null, season: '', lat: Number.isFinite(s.lat) ? s.lat : null,
    lon: Number.isFinite(s.lon) ? s.lon : null, excluded, exclusion_reason: reason, ...values }
}

async function loadSpartan(): Promise<NetworkData> {
  const [ref, map] = await Promise.all([loadJson<SpartanRef>('spartan_reference.json'), loadJson<{ sites: SpartanMapSite[] }>('spartan_map.json')])
  const byCode = new Map(map.sites.map((x) => [x.code, x]))
  const present = [...new Set(ref.rows.map((r) => String(r.code)))].sort((a, b) =>
    (byCode.get(a)?.name ?? a).localeCompare(byCode.get(b)?.name ?? b))
  const sites = present.map((code) => {
    const m = byCode.get(code)
    const n = ref.rows.filter((r) => r.code === code)
    const dates = n.map((r) => String(r.date)).sort()
    return site(code, m?.name ?? code, m?.name ?? code, m?.lat ?? null, m?.lon ?? null, n.length, dates[0] ?? '', dates[dates.length - 1] ?? '')
  })
  const sMap = new Map(sites.map((s) => [s.code, s]))
  const rows = ref.rows.map((r) => {
    const s = sMap.get(String(r.code))!
    const values = Object.fromEntries(ref.fields.map((f) => [f, typeof r[f] === 'number' ? (r[f] as number) : null]))
    return row(String(r.id), s, String(r.date), !!r.excluded, (r.exclusion_reason as string) || null, values)
  })
  const fields = ref.fields.filter((f) => rows.some((r) => typeof r[f] === 'number'))
  return { sites, rows, fields, field_units: ref.field_units, field_groups: SPARTAN_GROUPS.map((g) => ({ ...g, fields: g.fields.filter((f) => fields.includes(f)) })).filter((g) => g.fields.length) }
}

async function loadImprove(picked: string[]): Promise<NetworkData> {
  const index = await loadJson<ImproveIndex>('improve_samples/index.json')
  // US sites read by state; the overseas ones by country
  const place = (x: ImproveIndex['sites'][number]) => x.country === 'US' || !x.country ? x.state ?? '' : COUNTRY[x.country] ?? x.country
  const sites = index.sites.map((x) => site(x.code, x.name, place(x) || x.code,
    x.lat, x.lon, x.n_filters, x.date_min, x.date_max))
  const sMap = new Map(sites.map((s) => [s.code, s]))
  const files = await Promise.all(picked.filter((c) => sMap.has(c)).map((c) => loadJson<ImproveSite>(`improve_samples/${c}.json`)))
  const rows: FilterRow[] = []
  for (const f of files) {
    const s = sMap.get(f.code)!
    f.date.forEach((d, i) => {
      const values = Object.fromEntries(index.fields.map((k) => [k, (f[k] as (number | null)[])[i] ?? null]))
      rows.push(row(`${f.code}-${d}${f.poc[i] !== 1 ? `-${f.poc[i]}` : ''}`, s, d, false, null, values))
    })
  }
  return { sites, rows, fields: index.fields, field_units: index.field_units, field_groups: index.field_groups }
}

/** The chosen network's sites and rows; null while loading, and for the focus sites (App uses filters.json). */
export function useNetworkData(network: Network, improveSites: string[]) {
  const key = network === 'improve' ? `improve:${[...improveSites].sort().join(',')}` : network
  const [state, setState] = useState<{ key: string; data: NetworkData | null; error: string } | null>(null)
  useEffect(() => {
    if (network === 'focus') return
    let live = true
    ;(network === 'spartan' ? loadSpartan() : loadImprove(improveSites))
      .then((data) => { if (live) setState({ key, data, error: '' }) })
      .catch((e) => { if (live) setState({ key, data: null, error: String(e instanceof Error ? e.message : e) }) })
    return () => { live = false }
    // key captures network and the picked sites
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key])
  // keep showing the previous IMPROVE pick while another site loads, so charts do not blank between clicks
  const [last, setLast] = useState<{ network: Network; data: NetworkData } | null>(null)
  const current = state?.key === key ? state.data : null
  useEffect(() => { if (current) setLast({ network, data: current }) }, [current, network])
  return useMemo(() => ({
    data: current ?? (last?.network === network ? last.data : null),
    loading: network !== 'focus' && state?.key !== key,
    error: state?.key === key ? state.error : '',
  }), [current, last, network, state, key])
}
