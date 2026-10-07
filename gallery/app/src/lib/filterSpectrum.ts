import { useEffect, useState } from 'react'
import { BASE, Traces, fetchBinary } from '@/lib/similarity'
import { baseFilterId } from '@/lib/highlight'

/**
 * One filter's baseline-corrected spectrum from the Spectral similarity export
 * (gallery/data/export_similarity.py), for the filter drawers. Nothing is
 * fetched until a drawer asks, and the traces binary only once the filter is
 * known to be in the export. VIBES uses the separate full-range export and
 * its own grid; AIRSpec retains the original spline-corrected grid.
 *
 * Id resolution: "improve:<FilterId>" and "etad:<MediaId>" are similarity
 * sample ids already; an Addis SPARTAN id ("ETAD-0017", or "ETAD-0017-1") goes
 * through etad_ids.json (gallery/data/export_etad_ids.py) on its base id.
 */

export const SPECTRUM_METHODS = ['AIRSpec', 'VIBES'] as const
export type SpectrumMethod = (typeof SPECTRUM_METHODS)[number]

interface SimIndex { bin_wn: number[]; n: number; index: Map<string, number> }

let simPromise: Promise<SimIndex> | null = null
let etadPromise: Promise<Record<string, string[]>> | null = null

const getJson = <T,>(name: string): Promise<T> =>
  fetch(BASE + name).then((r) => {
    if (!r.ok) throw new Error(`${name}: HTTP ${r.status}`)
    return r.json() as Promise<T>
  })

function simIndex(): Promise<SimIndex> {
  simPromise ??= getJson<{ bin_wn: number[]; filters: { sample_id: string[] } }>('similarity.json').then((m) => ({
    bin_wn: m.bin_wn,
    n: m.filters.sample_id.length,
    index: new Map(m.filters.sample_id.map((id, i) => [id, i])),
  }))
  simPromise.catch(() => { simPromise = null })
  return simPromise
}

function etadIds(): Promise<Record<string, string[]>> {
  etadPromise ??= getJson<{ ids: Record<string, string[]> }>('etad_ids.json').then((j) => j.ids)
  etadPromise.catch(() => { etadPromise = null })
  return etadPromise
}

interface SpectrumTrace { wn: number[]; y: Float32Array; rankCap?: number; backgroundRank?: number; presentationStatus?: string }
interface TraceSource { wn: number[]; traces: Traces; rankCap?: number; backgroundRank?: number; presentationStatus?: string }
interface FullGrid { bin_wn: number[]; n: number; rank_cap?: number; background_rank?: number; presentation_status?: string }

export const traceVersionLabel = (t: { rankCap?: number; backgroundRank?: number; presentationStatus?: string }) =>
  t.presentationStatus === 'completed_uncapped' ? `completed, background rank ${t.backgroundRank ?? 70}` : `preliminary, rank cap ${t.rankCap ?? 30}`

const traceCache = new Map<string, Promise<TraceSource>>()
function traces(method: SpectrumMethod, sim: SimIndex, rank: '30' | '70'): Promise<TraceSource> {
  const source = method === 'VIBES' ? (rank === '70' ? 'VIBES-full-r70' : 'VIBES-full') : method
  let p = traceCache.get(source)
  if (!p) {
    // VIBES is fitted on its full range, not the archived truncated-grid run.
    // Each correction retains its own grid; the two bin counts are different.
    const grid: Promise<FullGrid> = method === 'VIBES'
      ? getJson<FullGrid>(`traces_${source}.json`)
      : Promise.resolve({ bin_wn: sim.bin_wn, n: sim.n })
    p = Promise.all([fetchBinary(`traces_${source}.bin`), grid]).then(([buf, meta]) => {
      if (meta.n !== sim.n || !meta.bin_wn.length || !meta.bin_wn.every(Number.isFinite)) throw new Error('Spectrum grid does not match the filter table')
      return { wn: meta.bin_wn, traces: new Traces(buf, sim.n, meta.bin_wn.length), rankCap: meta.rank_cap, backgroundRank: meta.background_rank, presentationStatus: meta.presentation_status }
    })
    p.catch(() => traceCache.delete(source))
    traceCache.set(source, p)
  }
  return p
}

/** Similarity sample ids for a drawer id; empty when the filter is not in the export. */
async function resolve(id: string, sim: SimIndex): Promise<string[]> {
  if (id.startsWith('improve:') || id.startsWith('etad:')) return sim.index.has(id) ? [id] : []
  const base = baseFilterId(id)
  if (!/^ETAD-\d{4}$/.test(base)) return []
  const map = await etadIds().catch(() => ({} as Record<string, string[]>))
  return (map[base] ?? []).filter((s) => sim.index.has(s))
}

export interface FilterSpectrumData {
  /** the drawer id this was loaded for */
  forId: string
  requestKey: string
  sampleId: string
  /** Each baseline has its own wavenumber grid and corrected absorbance. */
  series: Partial<Record<SpectrumMethod, SpectrumTrace>>
  unavailable: SpectrumMethod[]
}

/**
 * The spectrum for `id` under the requested baselines. `null` while loading,
 * or when the filter cannot be resolved. Individual trace failures are
 * reported in `unavailable`, so one failed baseline does not hide the other.
 */
export function useFilterSpectrum(id: string | null | undefined, methods: readonly SpectrumMethod[], vibesRank: '30' | '70' = '70'): FilterSpectrumData | null {
  const [data, setData] = useState<FilterSpectrumData | null>(null)
  const key = methods.join(',')
  const requestKey = `${id}|${key}|${vibesRank}`
  useEffect(() => {
    if (!id) return
    let live = true
    ;(async () => {
      const sim = await simIndex()
      const sids = await resolve(id, sim)
      if (!sids.length || !live) return
      const sampleId = sids[0]
      const k = sim.index.get(sampleId)!
      const loaded = await Promise.allSettled(methods.map(async (m) => {
        const source = await traces(m, sim, vibesRank)
        return [m, { wn: source.wn, y: source.traces.of(k), rankCap: source.rankCap, backgroundRank: source.backgroundRank, presentationStatus: source.presentationStatus }] as const
      }))
      const series: FilterSpectrumData['series'] = {}, unavailable: SpectrumMethod[] = []
      loaded.forEach((result, i) => {
        if (result.status === 'fulfilled') series[result.value[0]] = result.value[1]
        else unavailable.push(methods[i])
      })
      if (live) setData({ forId: id, requestKey, sampleId, series, unavailable })
    })().catch(() => { /* no spectrum is not an error in a drawer */ })
    return () => { live = false }
  }, [id, key, vibesRank]) // eslint-disable-line react-hooks/exhaustive-deps
  // a method switch keeps the old lines until the new ones land; a new filter clears them
  return data && data.requestKey === requestKey ? data : null
}
