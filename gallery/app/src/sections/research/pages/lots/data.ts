/**
 * Research → PTFE lot comparison: the export (public/data/research/lot_spectra/, from
 * gallery/data/export_research_lot_spectra.py), its per-view spectra decoded once per session, and the filters the
 * charts use, joined to the gallery rows in view (common/lotSpectra.ts) with the season names and lot colours.
 */
import { useEffect, useMemo, useState } from 'react'
import { loadBinary, loadDerived, loadJson, useExport } from '@/lib/data'
import { seasonOfDate } from '@/lib/seasons'
import type { FilterRow, FiltersFile, MetaFile } from '@/lib/types'
import { useSearchParam } from '@/lib/url'
import { decodeView, MEMBER_SETS, membersOf, NOTEBOOK_SEASONS, type LotExport, type Member, type MemberSet, type ViewKey } from '../../common/lotSpectra'
import { SERIES } from '../../common/ui'

export const EXPORTER = 'gallery/data/export_research_lot_spectra.py'
export const FOLDER = 'research/lot_spectra'
export const NB = 'research/ftir_etad_char/char_12_weakley_transferability.ipynb (scripts/run_char_12.py)'

/** char_12's lot colours (LOT_COLORS: 248 SERIES[3], 251 SERIES[2]); other lots from the page ramp */
const LOT_COLOR: Record<number, string> = { 248: '#eda100', 251: '#1baf7a', 253: '#2a78d6' }
export const lotColor = (lot: number, i = 0) => LOT_COLOR[lot] ?? SERIES[(i + 3) % SERIES.length]

export const SET_LABEL: Record<MemberSet, string> = { view: 'Gallery rows in view', plus: 'In view + no gallery row', notebook: "char_12's 239 (Addis)" }
export const SET_TITLE = "Which filters: the exported spectra whose gallery row is in view (your sites, seasons, dates and exclusions); those plus, for the sites in view, the PM2.5 spectra the gallery has no row for (sampled after its series ends; seasoned by your calendar, no exclusions or dates applied); or char_12's own 239 Addis filters with its calendar, which reproduces its tables."
export const useMemberSet = () => useSearchParam<MemberSet>('ls.set', 'view', MEMBER_SETS)

export function useLotExport() {
  return useExport<LotExport>(`${FOLDER}/lot_spectra.json`, (d) => (d.schema_version === 1 ? null : 'unsupported data version'))
}

/** one view's spectra, decoded (null while loading) */
export function useView(exp: LotExport | null, key: ViewKey): { values: Float32Array | null; error: string } {
  const [state, setState] = useState<{ key: string; values: Float32Array | null; error: string }>({ key: '', values: null, error: '' })
  const file = exp ? `${FOLDER}/${exp.views[key].file}` : null
  useEffect(() => {
    if (!exp || !file) return
    let live = true
    const n = exp.filters.id.length, p = exp.views[key].points
    loadDerived(`lot-spectra:${file}`, async () => decodeView(await loadBinary(file), n, p))
      .then((values) => { if (live) setState({ key: file, values, error: '' }) })
      .catch((e) => { if (live) setState({ key: file, values: null, error: String(e instanceof Error ? e.message : e) }) })
    return () => { live = false }
  }, [exp, file, key])
  return state.key === file ? state : { values: null, error: '' }
}

/** every gallery row id (filters.json, already loaded by the app): a spectrum outside it has no gallery row at all */
function useKnownIds(): Set<string> | null {
  const [ids, setIds] = useState<Set<string> | null>(null)
  useEffect(() => {
    let live = true
    loadDerived('lot-spectra:known', async () => new Set((await loadJson<FiltersFile>('filters.json')).rows.map((r) => r.id)))
      .then((s) => { if (live) setIds(s) }).catch(() => { if (live) setIds(new Set()) })
    return () => { live = false }
  }, [])
  return ids
}

export interface Lots {
  members: Member[]
  /** exported spectra of the sites in view with no gallery row */
  unmatched: number
  set: MemberSet
  /** site code → display name */
  siteName: (code: string) => string
  siteColor: (code: string) => string
  /** the seasons among the members, in calendar order from October */
  seasons: { name: string; color: string }[]
}

export function useLots(exp: LotExport | null, rows: FilterRow[], meta: MetaFile, set: MemberSet): Lots | null {
  const known = useKnownIds()
  return useMemo(() => {
    if (!exp || !known) return null
    const nameOf = new Map([...Object.entries(exp.site_names ?? {}), ...meta.sites.map((s) => [s.code, s.name] as [string, string])])
    for (const r of rows) nameOf.set(r.code, r.site)
    const colorOf = new Map(meta.sites.map((s) => [s.code, s.color]))
    for (const r of rows) colorOf.set(r.code, r.color)
    const calendarOf = (code: string) => meta.seasons.filter((s) => !s.site || s.site === nameOf.get(code))
    const { members, unmatched } = membersOf(exp, set, rows, known, (code, date, month) => seasonOfDate(date, month, calendarOf(code))?.name ?? null)
    const first = new Map<string, number>()
    for (const m of members) first.set(m.season, Math.min(first.get(m.season) ?? 99, (m.month + 2) % 12))
    const names = [...first.keys()].sort((a, b) => first.get(a)! - first.get(b)!)
    const palette = [...meta.seasons, ...Object.values(meta.season_conventions ?? {}).flat()]
    const seasonColor = (name: string, i: number) => palette.find((s) => s.name === name)?.color
      ?? (NOTEBOOK_SEASONS.some((s) => s.name === name) ? palette.find((s) => s.name.split(' ')[0] === name.split(' ')[0])?.color : undefined) ?? SERIES[i % SERIES.length]
    return {
      members, unmatched, set,
      siteName: (c: string) => nameOf.get(c) ?? c,
      siteColor: (c: string) => colorOf.get(c) ?? SERIES[0],
      seasons: names.map((name, i) => ({ name, color: seasonColor(name, i) })),
    }
  }, [exp, known, rows, meta, set])
}
