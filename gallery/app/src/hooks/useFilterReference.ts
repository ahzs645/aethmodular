import { useEffect, useMemo, useState } from 'react'
import { useSearchParam } from '@/lib/url'
import { loadSpartanIndex, loadSpartanFilters, loadSpartanReference, type SpartanIndex } from '@/lib/spartan'
import { cleanReference, mergeReferenceRows, type ReferenceRow } from '@/lib/referencePopulation'
import { networkColor } from '@/lib/networks'
import type { FilterRow, MetaFile } from '@/lib/types'

/** The detail drawer's reference is independent of the main plots' site/date selection. */
export function useFilterReference(row: FilterRow, meta: MetaFile, allRows: FilterRow[], studyRows: FilterRow[]) {
  const [mode, setMode] = useSearchParam<string>('sampleRef', 'same')
  const [picked, setPicked] = useSearchParam<string>('sampleSites', row.code)
  const [index, setIndex] = useState<SpartanIndex | null>(null)
  const [indexError, setIndexError] = useState('')
  const [data, setData] = useState<{ key: string; rows: ReferenceRow[] } | null>(null)
  const [loadError, setLoadError] = useState('')
  useEffect(() => {
    let live = true
    loadSpartanIndex().then(v => { if (live) setIndex(v) }).catch(e => { if (live) setIndexError(String(e)) })
    return () => { live = false }
  }, [])
  // Only the small site index is loaded up front. A named site or a multi-selection loads
  // those sites' filter numbers; the entire network is fetched only when explicitly chosen.
  const sites = useMemo(() => {
    const local = new Map(meta.sites.map(s => [s.code, s]))
    return (index?.sites ?? meta.sites).map((s, i) => ({ ...s, name: local.get(s.code)?.name ?? s.name, color: local.get(s.code)?.color ?? networkColor(i) }))
      .sort((a, b) => a.name.localeCompare(b.name))
  }, [index, meta.sites])
  const options = useMemo(() => [...new Set(['same', 'sites', 'study', 'network', ...sites.map(s => s.name), ...meta.sites.map(s => s.name)])], [sites, meta.sites])
  const codes = useMemo(() => [...new Set(mode === 'sites' ? picked.split(',').filter(c => sites.some(s => s.code === c)) : sites.filter(s => s.name === mode).map(s => s.code))].sort(), [mode, picked, sites])
  const key = mode === 'network' ? '*' : codes.join(',')
  const remote = mode === 'network' || mode === 'sites' || (mode !== 'same' && mode !== 'study' && (!index || index.sites.some(s => codes.includes(s.code))))
  useEffect(() => {
    if (!remote || !index) return
    let live = true
    setLoadError('')
    const request = mode === 'network' ? loadSpartanReference().then(v => v.rows) : loadSpartanFilters(codes).then(v => v.rows)
    request.then(rows => { if (live) setData({ key, rows }) }).catch(e => { if (live) setLoadError(String(e)) })
    return () => { live = false }
  }, [remote, index, key]) // codes and mode are represented by the request key
  const known = useMemo(() => [...studyRows, ...allRows], [studyRows, allRows])
  const rows = useMemo(() => {
    if (remote) return cleanReference(mergeReferenceRows(data?.key === key ? data.rows : [], known))
    return cleanReference(mode === 'study' ? studyRows : allRows, mode === 'same' ? row.site : mode === 'study' ? undefined : mode)
  }, [remote, data, key, known, mode, studyRows, allRows, row.site])
  const names = sites.filter(s => codes.includes(s.code)).map(s => s.name)
  const label = mode === 'same' ? row.site : mode === 'study' ? 'Four gallery sites' : mode === 'network' ? 'SPARTAN network'
    : mode === 'sites' ? (names.length <= 2 ? names.join(' + ') || 'No reference sites selected' : `${names.length} selected sites`) : mode
  const error = remote ? indexError || loadError : ''
  const loading = remote && !error && (!index || data?.key !== key)
  const toggleSite = (code: string) => setPicked((codes.includes(code) ? codes.filter(c => c !== code).join(',') : [...codes, code].sort().join(',')) || '-')
  return { mode, setMode, options, rows, label, error, loading, sites, codes, toggleSite, clearSites: () => setPicked('-'),
    networkSize: new Set(rows.map(r => r.code)).size }
}
