import { loadEtadIds, loadSimilarityMeta, loadTraces, type Traces } from '@/lib/similarity'
import { loadDerived, useLoaded } from '@/lib/data'
import { baseFilterId } from '@/lib/highlight'
import { improveSampleFor, parseImproveRowId } from '@/lib/improveLink'
import { loadPilotFilters, loadPilotSpectra, pilotRecord } from '@/lib/pilotFilters'

/**
 * One filter's baseline-corrected spectrum from the Spectral similarity export
 * (gallery/data/export_similarity.py), for the filter drawers. Nothing is
 * fetched until a drawer asks, and the traces binary only once the filter is
 * known to be in the export. VIBES uses the separate full-range export and
 * its own grid; AIRSpec retains the original spline-corrected grid. Files are
 * loaded through lib/similarity, so the drawer shares them with every page.
 *
 * Id resolution: "improve:<FilterId>" and "etad:<MediaId>" are similarity
 * sample ids already; an Addis SPARTAN id ("ETAD-0017", or "ETAD-0017-1") goes
 * through etad_ids.json (gallery/data/export_etad_ids.py) on its base id; an IMPROVE
 * network row ("PINN1-2022-11-22") through its site and day (lib/improveLink).
 * The Ethiopia pilot filters (Adama J-numbers, Bishoftu ETBI ids) come from their own
 * export instead (pilot_spectra.json, gallery/data/export_pilot_filters.py), same bins.
 */

export const SPECTRUM_METHODS = ['AIRSpec', 'VIBES'] as const
export type SpectrumMethod = (typeof SPECTRUM_METHODS)[number]

interface SimIndex { bin_wn: number[]; n: number; index: Map<string, number> }

/** Why an Addis filter does or does not have a spectrum / test-set result (export_etad_ids.py). */
export interface AddisStatus { in_run: boolean; in_test_set: boolean; note: string }
interface EtadJson { ids: Record<string, string[]>; status?: Record<string, AddisStatus> }

const simIndex = (): Promise<SimIndex> => loadDerived('similarity-index', async () => {
  const m = await loadSimilarityMeta()
  return { bin_wn: m.bin_wn, n: m.filters.sample_id.length, index: new Map(m.filters.sample_id.map((id, i) => [id, i])) }
})
const etadJson = () => loadEtadIds<EtadJson>()
const etadIds = () => etadJson().then((j) => j.ids)

interface SpectrumTrace { wn: number[]; y: Float32Array; rankCap?: number; backgroundRank?: number; presentationStatus?: string }
interface TraceSource { wn: number[]; traces: Traces; rankCap?: number; backgroundRank?: number; presentationStatus?: string }

export const traceVersionLabel = (t: { rankCap?: number; backgroundRank?: number; presentationStatus?: string }) =>
  t.presentationStatus === 'completed_uncapped' ? `completed, background rank ${t.backgroundRank ?? 70}` : `preliminary, rank cap ${t.rankCap ?? 30}`

// VIBES is fitted on its full range, not the archived truncated-grid run; each run keeps its own grid
async function traces(method: SpectrumMethod, rank: '30' | '70'): Promise<TraceSource> {
  const t = await loadTraces(method === 'VIBES' ? (rank === '70' ? 'VIBES-full-r70' : 'VIBES-full') : method)
  return { wn: t.wn, traces: t.traces, rankCap: t.run?.rank_cap, backgroundRank: t.run?.background_rank, presentationStatus: t.run?.presentation_status }
}

/** Similarity sample ids for a drawer id; empty when the filter is not in the export. */
async function resolve(id: string, sim: SimIndex): Promise<string[]> {
  if (id.startsWith('improve:') || id.startsWith('etad:')) return sim.index.has(id) ? [id] : []
  if (parseImproveRowId(id)) {
    const sample = improveSampleFor(await loadSimilarityMeta(), id)
    return sample && sim.index.has(sample) ? [sample] : []
  }
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
  /** which export the traces came from: the Spectral similarity export, or the pilot filters' */
  source: 'similarity' | 'pilot'
  /** the pilot's Adama analysis-to-filter map is provisional (date-based) */
  provisional?: boolean
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
  const requestKey = `${id}|${methods.join(',')}|${vibesRank}`
  // the similarity export holds IMPROVE and Addis spectra; any other id may still be a pilot filter
  const similar = !!id && (id.startsWith('improve:') || id.startsWith('etad:') || /^ETAD-\d{4}/.test(id) || !!parseImproveRowId(id))
  // no spectrum is not an error in a drawer: a failure reads as none
  return useLoaded(id ? requestKey : null, async (): Promise<FilterSpectrumData | null> => {
    const pilot = pilotRecord(await loadPilotFilters().catch(() => null), id)
    if (pilot) return pilotSpectrum(id!, requestKey, pilot.spectrum, pilot.spectrum_map === 'provisional', methods)
    if (!similar) return null
    const sim = await simIndex()
    const sids = await resolve(id!, sim)
    if (!sids.length) return null
    const sampleId = sids[0]
    const k = sim.index.get(sampleId)!
    const loaded = await Promise.allSettled(methods.map(async (m) => {
      const source = await traces(m, vibesRank)
      return [m, { wn: source.wn, y: source.traces.of(k), rankCap: source.rankCap, backgroundRank: source.backgroundRank, presentationStatus: source.presentationStatus }] as const
    }))
    const series: FilterSpectrumData['series'] = {}, unavailable: SpectrumMethod[] = []
    loaded.forEach((result, i) => {
      if (result.status === 'fulfilled') series[result.value[0]] = result.value[1]
      else unavailable.push(methods[i])
    })
    return { forId: id!, requestKey, sampleId, source: 'similarity', series, unavailable }
  }).data
}

/** Base id (ETAD-0017) for an Addis drawer id or an "etad:<MediaId>" sample id; null otherwise. */
/** A pilot filter's corrected spectra (completed rank-70 VIBES and the spline baseline), in the similarity traces' bins. */
async function pilotSpectrum(forId: string, requestKey: string, sampleId: string, provisional: boolean, methods: readonly SpectrumMethod[]): Promise<FilterSpectrumData | null> {
  const spec = await loadPilotSpectra().catch(() => null)
  const k = spec?.ids.indexOf(sampleId) ?? -1
  if (!spec || k < 0) return null
  const series: FilterSpectrumData['series'] = {}
  if (methods.includes('AIRSpec')) series.AIRSpec = { wn: spec.AIRSpec.bin_wn, y: Float32Array.from(spec.AIRSpec.y[k]) }
  if (methods.includes('VIBES')) series.VIBES = { wn: spec.VIBES.bin_wn, y: Float32Array.from(spec.VIBES.y[k]),
    backgroundRank: spec.VIBES.background_rank, presentationStatus: spec.VIBES.presentation_status }
  return { forId, requestKey, sampleId, source: 'pilot', provisional, series, unavailable: [] }
}

async function addisBase(id: string): Promise<string | null> {
  if (id.startsWith('etad:')) {
    const ids = await etadIds()
    return Object.keys(ids).find((b) => ids[b].includes(id)) ?? null
  }
  const base = baseFilterId(id)
  return /^ETAD-\d{4}$/.test(base) ? base : null
}

/**
 * Status of an Addis filter: in the test set, in the spectra run only (no HIPS),
 * a PM10 or blank filter, and so on. Null for non-Addis ids and while loading.
 */
export function useAddisStatus(id: string | null | undefined): AddisStatus | null {
  const addis = !!id && (id.startsWith('etad:') || /^ETAD-\d{4}/.test(id))
  return useLoaded(addis ? id : null, async () => {
    const base = await addisBase(id!)
    const all = (await etadJson()).status ?? {}
    return base ? all[base] ?? { in_run: false, in_test_set: false, note: 'not in the frozen spectra run or the Addis test set' } : null
  }).data
}

/** All Addis statuses keyed by similarity sample id ("etad:<MediaId>"), for tooltips. */
export function useAddisStatusBySample(): Map<string, AddisStatus> {
  return useLoaded('addis-status-by-sample', () => etadJson().then((j) => {
    const m = new Map<string, AddisStatus>()
    for (const [b, sids] of Object.entries(j.ids)) for (const sid of sids) if (j.status?.[b]) m.set(sid, j.status[b])
    return m
  })).data ?? NO_STATUS
}
const NO_STATUS = new Map<string, AddisStatus>()
