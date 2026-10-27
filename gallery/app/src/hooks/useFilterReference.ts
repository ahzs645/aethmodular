import { useEffect, useMemo, useState } from 'react'
import { useSearchParam } from '@/lib/url'
import { seasonOnly } from '@/lib/seasons'
import { loadSpartanIndex, loadSpartanFilters, loadSpartanReference, type SpartanIndex } from '@/lib/spartan'
import { cleanReference, mergeReferenceRows, sampleRows, type QuantileReference, type ReferenceRow } from '@/lib/referencePopulation'
import { loadJson } from '@/lib/data'
import { MAX_IMPROVE_SITES, loadImprove, networkColor } from '@/lib/networks'
import type { FilterRow, MetaFile } from '@/lib/types'

/** the IMPROVE site list (gallery/data/export_improve_samples.py): for choosing reference sites only */
type ImproveSites = { sites: { code: string; name: string; state: string | null; country: string | null; lat: number | null; lon: number | null; n_filters: number }[] }

/**
 * The detail drawer's reference is independent of the main plots' site/date selection.
 * "Choose multiple sites…" offers SPARTAN and IMPROVE sites together (IMPROVE up to MAX_IMPROVE_SITES,
 * as in the gallery: each site is its own file). The networks share only some measurements (PM2.5
 * mass, iron, silicon, HIPS Fabs and BC at MAC 10, sampled volume; EC/OC are FTIR in one and TOR in the
 * other), so with several sites a measurement is only placed when every chosen site that has filters
 * measured it: `partial` lists the rest with the sites that lack them, instead of a reference that
 * quietly narrows to the sites that have it.
 */
/** the gallery's season of a filter at a site (App seasonOf: the active calendar, per site) */
export type SeasonOf = (site: string, month: number | null | undefined, fallback: string, date?: string | null) => string
export const UNDATED = 'Undated'

export function useFilterReference(row: FilterRow, meta: MetaFile, allRows: FilterRow[], studyRows: FilterRow[], seasonOf?: SeasonOf) {
  // the reference's season: every season, the filter's own (at its site, under a site calendar), or one season
  const [seasonPick, setSeasonPick] = useSearchParam<string>('sampleSeason', 'all')
  const [mode, setMode] = useSearchParam<string>('sampleRef', 'same')
  const [picked, setPicked] = useSearchParam<string>('sampleSites', row.code)
  const [index, setIndex] = useState<SpartanIndex | null>(null)
  const [improve, setImprove] = useState<ImproveSites | null>(null)
  const [indexError, setIndexError] = useState('')
  // `summary`: the whole IMPROVE network as quantiles (improve_reference.json); its rows are then a sample, for the crossplot
  const [data, setData] = useState<{ key: string; rows: ReferenceRow[]; summary?: QuantileReference } | null>(null)
  const [loadError, setLoadError] = useState('')
  useEffect(() => {
    let live = true
    loadSpartanIndex().then(v => { if (live) setIndex(v) }).catch(e => { if (live) setIndexError(String(e)) })
    return () => { live = false }
  }, [])
  // the IMPROVE list only when sites are being chosen (or a link names some): the drawer never needs it otherwise
  useEffect(() => {
    if (mode !== 'sites' || improve) return
    let live = true
    loadJson<ImproveSites>('improve_samples/index.json').then(v => { if (live) setImprove(v) }).catch(() => { /* SPARTAN sites still offered */ })
    return () => { live = false }
  }, [mode, improve])
  // Only the small site index is loaded up front. A named site or a multi-selection loads
  // those sites' filter numbers; the entire network is fetched only when explicitly chosen.
  const spartanSites = useMemo(() => {
    const local = new Map(meta.sites.map(s => [s.code, s]))
    return (index?.sites ?? meta.sites).map((s, i) => ({ ...s, name: local.get(s.code)?.name ?? s.name, color: local.get(s.code)?.color ?? networkColor(i), network: 'SPARTAN' as const }))
      .sort((a, b) => a.name.localeCompare(b.name))
  }, [index, meta.sites])
  const improveSites = useMemo(() => (improve?.sites ?? []).map((s, i) => ({ code: s.code, name: s.name, lat: s.lat, lon: s.lon, n_filters: s.n_filters,
    location: s.state ?? s.country ?? '', color: networkColor(spartanSites.length + i), network: 'IMPROVE' as const })), [improve, spartanSites.length])
  const improveCodes = useMemo(() => new Set(improveSites.map(s => s.code)), [improveSites])
  const sites = useMemo(() => [...spartanSites, ...improveSites], [spartanSites, improveSites])
  // one site by name stays a SPARTAN (or gallery) site: 217 IMPROVE names would bury the list; they are chosen under "multiple sites"
  const options = useMemo(() => [...new Set(['same', 'sites', 'study', 'network', 'improve', ...spartanSites.map(s => s.name), ...meta.sites.map(s => s.name)])], [spartanSites, meta.sites])
  const codes = useMemo(() => [...new Set(mode === 'sites' ? picked.split(',').filter(c => sites.some(s => s.code === c)) : spartanSites.filter(s => s.name === mode).map(s => s.code))].sort(), [mode, picked, sites, spartanSites])
  const key = mode === 'network' ? '*' : mode === 'improve' ? 'improve:*' : codes.join(',')
  const remote = mode === 'network' || mode === 'improve' || mode === 'sites' || (mode !== 'same' && mode !== 'study' && (!index || index.sites.some(s => codes.includes(s.code))))
  useEffect(() => {
    if (!remote || !index) return
    let live = true
    setLoadError('')
    const spartanCodes = codes.filter(c => !improveCodes.has(c)), improvePicked = codes.filter(c => improveCodes.has(c))
    if (mode === 'improve') {
      loadJson<QuantileReference>('improve_reference.json')
        .then(summary => { if (live) setData({ key, rows: sampleRows(summary), summary }) }).catch(e => { if (live) setLoadError(String(e)) })
      return () => { live = false }
    }
    const request = mode === 'network' ? loadSpartanReference().then(v => v.rows)
      : Promise.all([spartanCodes.length || !improvePicked.length ? loadSpartanFilters(spartanCodes).then(v => v.rows) : [],
        improvePicked.length ? loadImprove(improvePicked).then(v => v.rows as unknown as ReferenceRow[]) : []]).then(([a, b]) => [...a, ...b])
    request.then(rows => { if (live) setData({ key, rows }) }).catch(e => { if (live) setLoadError(String(e)) })
    return () => { live = false }
  }, [remote, index, key, improveCodes]) // codes and mode are represented by the request key
  const known = useMemo(() => [...studyRows, ...allRows], [studyRows, allRows])
  const pooled = useMemo(() => {
    const out = remote ? cleanReference(mergeReferenceRows(data?.key === key ? data.rows : [], known))
      : cleanReference(mode === 'study' ? studyRows : allRows, mode === 'same' ? row.site : mode === 'study' ? undefined : mode)
    // every reference filter takes the gallery's season, as the charts' filters do: rows that come from the site
    // files or the evaluation set carry none of their own (they were 'Unknown season' in the crossplot)
    if (!seasonOf) return out
    return out.map((r) => {
      const date = typeof r.date === 'string' ? r.date : ''
      const own = typeof r.season === 'string' && r.season ? r.season : UNDATED
      const season = seasonOf(String(r.site), date ? Number(date.slice(5, 7)) : null, own, date)
      return season === r.season ? r : { ...r, season }
    })
  }, [remote, data, key, known, mode, studyRows, allRows, row.site, seasonOf])
  // the IMPROVE network is quantiles over all seasons: it has no season choice
  const ownSeason = typeof row.season === 'string' && row.season ? row.season : null
  const seasonChoice = mode === 'improve' ? 'all' : seasonPick
  const season = seasonChoice === 'all' ? null : seasonChoice === 'same' ? ownSeason : seasonChoice
  const rows = useMemo(() => (season ? pooled.filter((r) => r.season === season) : pooled), [pooled, season])
  // the seasons the reference has, in the calendar's order
  const seasons = useMemo(() => {
    const order = new Map(meta.seasons.map((s, i) => [s.name, i]))
    return [...new Set(pooled.map((r) => String(r.season)))].sort((a, b) => (order.get(a) ?? 99) - (order.get(b) ?? 99) || a.localeCompare(b))
  }, [pooled, meta.seasons])
  const names = sites.filter(s => codes.includes(s.code)).map(s => s.name)
  const label = mode === 'same' ? row.site : mode === 'study' ? 'Four gallery sites' : mode === 'network' ? 'SPARTAN network' : mode === 'improve' ? 'IMPROVE network'
    : mode === 'sites' ? (names.length <= 2 ? names.join(' + ') || 'No reference sites selected' : `${names.length} selected sites`) : mode
  const seasonLabel = season ? ` · ${seasonOnly(season)}` : ''
  const error = remote ? indexError || loadError : ''
  const loading = remote && !error && (!index || data?.key !== key)
  const improveCount = codes.filter(c => improveCodes.has(c)).length
  // the sites dropdown's whole choice: an IMPROVE site is a file of its own, so past the cap only the first MAX_IMPROVE_SITES are kept
  const setSites = (next: string[]) => {
    let improveLeft = MAX_IMPROVE_SITES
    const kept = next.filter(c => !improveCodes.has(c) || improveLeft-- > 0)
    setPicked([...new Set(kept)].sort().join(',') || '-')
  }
  // a measurement some chosen sites never measured: by field, the sites without it (several chosen sites only;
  // the whole SPARTAN network keeps each measurement's own coverage, as its note says)
  const partial = useMemo(() => {
    const out = new Map<string, string[]>()
    // the whole IMPROVE network: what it never measures (FTIR EC/OC, ions, most elements) is listed, not placed
    if (mode === 'improve') {
      const summary = data?.key === key ? data.summary : undefined
      if (summary) for (const f of meta.fields) if (!summary.quantiles[f]) out.set(f, ['IMPROVE'])
      return out
    }
    if (mode === 'network' || mode === 'same' || loading) return out
    const withRows = [...new Set(rows.map(r => r.code))]
    if (withRows.length < 2) return out
    const nameOf = new Map(sites.map(s => [s.code, s.name]))
    // every measurement the rows carry, not only the drawer's list: ratios use fields outside it (ChemSpec FTIR)
    const fields = new Set(meta.fields)
    for (const r of rows) for (const [k, v] of Object.entries(r)) if (typeof v === 'number') fields.add(k)
    for (const f of fields) {
      const has = new Set(rows.filter(r => typeof r[f] === 'number' && Number.isFinite(r[f])).map(r => r.code))
      const lacking = withRows.filter(c => !has.has(c))
      if (has.size && lacking.length) out.set(f, lacking.map(c => nameOf.get(c) ?? c))
    }
    return out
  }, [mode, loading, rows, sites, meta.fields, data, key])
  return { mode, setMode, options, rows, label: label + seasonLabel, error, loading, sites, codes, setSites,
    networkSize: new Set(rows.map(r => r.code)).size, partial,
    season: { value: seasonChoice, set: setSeasonPick, options: seasons, own: ownSeason, label: seasonLabel, available: mode !== 'improve' },
    summary: mode === 'improve' && data?.key === key ? data.summary ?? null : null,
    improveNote: improveSites.length ? `${improveCount} of ${MAX_IMPROVE_SITES} IMPROVE sites` : undefined }
}
