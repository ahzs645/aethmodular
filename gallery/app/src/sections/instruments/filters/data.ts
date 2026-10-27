/**
 * The "Against the filters" page's data: one record per filter with the aethalometer at the chosen
 * channel beside HIPS, FTIR EC and iron. The aethalometer fields are already on the rows (../lib.ts);
 * smoothed channels and spot attenuation come from gallery/data/export_aeth_filters.py
 * (aethalometer/filters_extra/), loaded here for the sites in view only.
 */
import { useEffect, useMemo, useState } from 'react'
import * as d3 from 'd3'
import { loadJson, useExport } from '@/lib/data'
import type { FilterRow, MetaFile } from '@/lib/types'
import { num } from './mac'

export const CHANNELS = ['UV', 'Blue', 'Green', 'Red', 'IR'] as const
export type Channel = (typeof CHANNELS)[number]
/** HIPS reads at 632.8 nm (He–Ne; docs/filter-optics-reference.md §2b): the MA350's Red (625 nm) is the nearest channel */
export const HIPS_NM = 633
export const NEAREST_TO_HIPS: Channel = 'Red'

export const bccField = (ch: Channel, smoothed = false) => (smoothed ? `${ch} BCc smoothed (aethalometer)` : `${ch} BCc (aethalometer)`)
export const atnField = (ch: Channel) => `${ch} ATN1 (aethalometer)`
export const HIPS_FABS = 'HIPS Fabs', HIPS_BC = 'HIPS BC', FTIR_EC = 'EC (FTIR)', IRON = 'Iron'
/** fine soil, IMPROVE's oxide sum (iron_correction_test.ipynb Test 2): 2.20 Al + 2.49 Si + 1.63 Ca + 2.42 Fe + 1.94 Ti */
const SOIL: [string, number][] = [['Aluminum', 2.2], ['Silicon', 2.49], ['Calcium', 1.63], ['Iron', 2.42], ['Titanium', 1.94]]

export interface ExtraIndex {
  fields: string[]; field_units: Record<string, string>
  /** MA350 attenuation cross-sections σ_ATN (m²/g) by channel, from export_aethalometer.SIGMA */
  sigma: Record<Channel, number>
  wavelengths_nm: Record<Channel, number>
  sites: Record<string, number>
}

/** filters_extra for the sites in view: id -> field -> value, with the index (σ, wavelengths) */
export function useAethExtra(codes: readonly string[]): { index: ExtraIndex | null; values: Map<string, Record<string, number | null>> } {
  const index = useExport<ExtraIndex>('aethalometer/filters_extra/index.json').data
  const key = index ? [...new Set(codes)].filter((c) => index.sites[c]).sort().join(',') : ''
  const [state, setState] = useState<{ key: string; values: Map<string, Record<string, number | null>> } | null>(null)
  useEffect(() => {
    if (!index) return
    let live = true
    Promise.all((key ? key.split(',') : []).map((c) => loadJson<{ rows: Record<string, (number | null)[]> }>(`aethalometer/filters_extra/${c}.json`)))
      .then((parts) => {
        if (!live) return
        const values = new Map<string, Record<string, number | null>>()
        for (const p of parts) for (const [id, vs] of Object.entries(p.rows)) values.set(id, Object.fromEntries(index.fields.map((f, i) => [f, vs[i]])))
        setState({ key, values })
      })
      .catch(() => { /* optional: the page draws without ATN and the smoothed channels */ })
    return () => { live = false }
  }, [index, key])
  return { index: index ?? null, values: state?.key === key ? state.values : NO_EXTRA }
}
const NO_EXTRA = new Map<string, Record<string, number | null>>()

/** One filter with the aethalometer at the chosen channel and the filter measurements it is compared with. */
export interface Pair {
  row: FilterRow
  id: string; site: string; code: string; date: string; season: string; siteColor: string
  /** aethalometer BCc at the channel (µg/m³) */
  bc: number | null
  /** attenuation coefficient b_ATN = BCc × σ_ATN (Mm⁻¹): still carries the filter's multiple-scattering enhancement */
  batn: number | null
  /** spot attenuation over the window (tape loading, not a concentration) */
  atn: number | null
  fabs: number | null
  /** HIPS Fabs ÷ MAC 10 (µg/m³), the public BC */
  hipsBc: number | null
  ec: number | null
  iron: number | null
  fineSoil: number | null
}

export function buildPairs(rows: FilterRow[], meta: MetaFile, extra: Map<string, Record<string, number | null>>, ch: Channel, smoothed: boolean, sigma: Record<Channel, number> | null): Pair[] {
  const mac = meta.mac_value || 10
  return rows.flatMap((r) => {
    const x = extra.get(r.id)
    const bc = smoothed ? (ch === 'IR' ? num(r[bccField('IR', true)]) : num(x?.[bccField(ch, true)])) : num(r[bccField(ch)])
    if (bc === null && num(r[bccField('IR')]) === null) return []   // no aethalometer over this filter
    const fabs = num(r[HIPS_FABS])
    const soil = SOIL.map(([f, w]) => (num(r[f]) === null ? null : (num(r[f]) as number) * w))
    return [{
      row: r, id: r.id, site: r.site, code: r.code, date: r.date, season: r.season,
      siteColor: meta.sites.find((s) => s.name === r.site)?.color ?? (r.color as string),
      bc, batn: bc !== null && sigma ? bc * sigma[ch] : null, atn: num(x?.[atnField(ch)]),
      fabs, hipsBc: num(r[HIPS_BC]) ?? (fabs === null ? null : fabs / mac), ec: num(r[FTIR_EC]), iron: num(r[IRON]),
      fineSoil: soil.every((v) => v !== null) ? soil.reduce((a, b) => a! + b!, 0) : null,
    }]
  })
}

/** the sites in view that have pairs, in the meta's order */
export function pairSites(pairs: Pair[], meta: MetaFile) {
  const n = new Map<string, number>()
  for (const p of pairs) n.set(p.site, (n.get(p.site) ?? 0) + 1)
  return meta.sites.filter((s) => n.has(s.name)).map((s) => ({ name: s.name, code: s.code, color: s.color, n: n.get(s.name)! }))
}
export type PairSite = ReturnType<typeof pairSites>[number]

/** a date axis over some filters, 3 % wider each side so the end points and tick labels stay inside the chart */
export function dateDomain(dates: string[]): [Date, Date] | undefined {
  const t = dates.map((d) => Date.parse(d.slice(0, 10))).filter(Number.isFinite)
  if (!t.length) return undefined
  const lo = Math.min(...t), hi = Math.max(...t), pad = Math.max(3 * 864e5, (hi - lo) * 0.03)
  return [new Date(lo - pad), new Date(hi + pad)]
}

/** date ticks for a span of filters: day and month when it covers less than about seven months, else month and year */
export function dateTicks(dates: string[]): (v: unknown) => string {
  const t = dates.map((d) => Date.parse(d.slice(0, 10))).filter(Number.isFinite)
  const short = t.length > 0 && Math.max(...t) - Math.min(...t) < 210 * 864e5
  const f = d3.timeFormat(short ? '%-d %b' : '%b %Y')
  return (v) => f(v as Date)
}

/** the season's colour (the calendar in use), else muted */
export const seasonColor = (meta: MetaFile, season: string, fallback: string) => meta.seasons.find((s) => s.name === season)?.color ?? fallback

/** the seasons present in some pairs, in the calendar's order */
export function seasonsIn(pairs: Pair[], meta: MetaFile): string[] {
  const have = new Set(pairs.map((p) => p.season))
  const order = meta.seasons.map((s) => s.name)
  return [...have].filter(Boolean).sort((a, b) => order.indexOf(a) - order.indexOf(b))
}

export const usePairs = (rows: FilterRow[], meta: MetaFile, extra: ReturnType<typeof useAethExtra>, ch: Channel, smoothed: boolean) =>
  useMemo(() => buildPairs(rows, meta, extra.values, ch, smoothed, extra.index?.sigma ?? null), [rows, meta, extra.values, extra.index, ch, smoothed])

/**
 * The MAC definitions the notebooks used, each a ratio of a pair's fields, with the value it is
 * compared against. `aethOverEc`'s reference is σ_ATN at the channel: b_ATN / EC equals σ_ATN exactly
 * when the aethalometer's BC equals FTIR EC.
 */
export type MacKey = 'hipsOverEc' | 'hipsOverBc' | 'aethOverEc'
export interface MacDef { key: MacKey; label: string; short: string; num: (p: Pair) => number | null; den: (p: Pair) => number | null; reference: (sigma: number) => number; refLabel: string }
export const MAC_DEFS: MacDef[] = [
  { key: 'hipsOverEc', label: 'HIPS Fabs ÷ FTIR EC', short: 'HIPS ÷ EC', num: (p) => p.fabs, den: (p) => p.ec, reference: () => 10, refLabel: 'SPARTAN MAC 10 m²/g at 633 nm' },
  { key: 'hipsOverBc', label: 'HIPS Fabs ÷ aethalometer BCc', short: 'HIPS ÷ aeth BC', num: (p) => p.fabs, den: (p) => p.bc, reference: () => 10, refLabel: 'MAC 10 m²/g' },
  { key: 'aethOverEc', label: 'aethalometer b_ATN ÷ FTIR EC', short: 'b_ATN ÷ EC', num: (p) => p.batn, den: (p) => p.ec, reference: (s) => s, refLabel: 'σ_ATN of the channel (BCc = EC)' },
]
export const macOf = (d: MacDef, p: Pair): number | null => {
  const a = d.num(p), b = d.den(p)
  return a === null || b === null || b <= 0 ? null : a / b
}
