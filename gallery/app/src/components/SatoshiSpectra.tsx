import { useEffect, useMemo, useState } from 'react'
import { CountPicker, clampCount } from '@/components/Pickers'
import { ChartFrame, Empty, Toggle } from '@/components/ChartFrame'
import { SpectraTraces } from '@/components/SimilaritySpectra'
import { useSearchParam } from '@/lib/url'
import { usePageChoice } from '@/lib/pageChoices'
import { useHighlightActions } from '@/lib/highlight'
import { baselineName } from '@/lib/labels'
import { Traces, fetchBinary, loadEtadIds, loadSimilarityMeta, siteName, type SimilarityMeta } from '@/lib/similarity'
import { addisTargetSpectra, rankedAnalogSpectra, type AddisSpectrumIds, type AnalogBaseline, type AnalogRegion } from '@/lib/satoshiSpectra'

const TOP_N = ['50', '100', '250', '500'] as const
const BASELINES: AnalogBaseline[] = ['AIRSpec', 'VIBES-full-r70']
const baselineLabel = (baseline: string) => baselineName(baseline)
type Columns = Record<string, (number | string | null)[]>

export function SatoshiSpectra({ membership, groups, region }: { membership: Columns; groups: string[]; region: AnalogRegion }) {
  // the page's baseline and season (pages/SatoshiFollowupPage PageChoices)
  const { value: baseline } = usePageChoice('baseline', BASELINES) as { value: AnalogBaseline }
  const { value: group } = usePageChoice('season', groups)
  // any count up to the 500 stored analogs; the presets are the usual ones
  const [topRaw, setTopN] = useSearchParam<string>('satSpecN', '500')
  const topN = clampCount(topRaw, 500, 500)
  const [test, setTest] = useSearchParam<'on' | 'off'>('satSpecTest', 'on', ['on', 'off'])
  const [meta, setMeta] = useState<SimilarityMeta | null>(null)
  const [ids, setIds] = useState<AddisSpectrumIds | null>(null)
  const [error, setError] = useState('')
  const [loaded, setLoaded] = useState<{ baseline: AnalogBaseline; traces: Traces } | null>(null)
  const hl = useHighlightActions()

  useEffect(() => {
    let live = true
    Promise.all([loadSimilarityMeta(), loadEtadIds<AddisSpectrumIds>()]).then(([meta, ids]) => {
      if (!ids.status) throw new Error('etad_ids.json has no Addis status')
      if (live) { setMeta(meta); setIds(ids) }
    }).catch(e => { if (live) setError(String(e)) })
    return () => { live = false }
  }, [])

  const activeMeta = useMemo(() => {
    if (!meta) return null
    const grid = meta.method_grids?.[baseline]
    return grid ? { ...meta, ...grid } : meta
  }, [meta, baseline])
  useEffect(() => {
    if (!activeMeta) return
    let live = true
    setError('')
    fetchBinary(`traces_${baseline}.bin`).then(buffer => {
      const traces = new Traces(buffer, activeMeta.filters.sample_id.length, activeMeta.bin_wn.length)
      if (live) setLoaded({ baseline, traces })
    }).catch(e => { if (live) setError(String(e)) })
    return () => { live = false }
  }, [activeMeta, baseline])

  const selection = useMemo(() => {
    if (!activeMeta || !ids) return null
    const analogs = rankedAnalogSpectra(membership, activeMeta, baseline, region, group)
    const target = addisTargetSpectra(activeMeta, ids, group)
    const shown = analogs.rows.filter(row => row.rank <= Number(topN) &&
      (test === 'on' || activeMeta.sites[activeMeta.filters.site[row.filter]].split !== 'test'))
    return { ...target, shown, missing: analogs.missing }
  }, [activeMeta, ids, membership, baseline, region, group, topN, test])

  const pinned = activeMeta && hl.pinnedId ? activeMeta.filters.sample_id.indexOf(hl.pinnedId) : -1
  const onPin = (filter: number | null) => {
    if (filter == null || !activeMeta) { hl.dismissSample(); return }
    const id = activeMeta.filters.sample_id[filter]
    const site = activeMeta.sites[activeMeta.filters.site[filter]]
    const row = selection?.shown.find(row => row.filter === filter)
    hl.openRecord({ id, source: `Meeting spectra · ${baselineLabel(baseline)} · analogs of ${group}`,
      site: `${site.site} · ${siteName(site)}`, date: activeMeta.filters.date[filter] ?? undefined,
      fields: [['Filter role', id.startsWith('etad:') ? 'Addis test set' : 'IMPROVE spectral analog'],
        ['Similarity rank', row?.rank ?? null], ['r to median Addis spectrum', row?.r ?? null],
        ['Similarity region', region === 'oct1' ? '600–3600 cm⁻¹, excluding 1800–2500 and 1050–1250 cm⁻¹' : '500–3600 cm⁻¹, excluding 1800–2500 cm⁻¹']],
      note: 'Saved analog ranks from the meeting analysis, compared with the median Addis spectrum. Displayed spectra are 8-channel means; neither ranks nor calibrations are refitted here.' })
  }

  if (error || !activeMeta || !selection || !loaded || loaded.baseline !== baseline) return <ChartFrame id="satoshi-spectra" title="Spectra" exportable={false}>
    <Empty>{error ? `Could not load spectra: ${error}` : 'Loading per-filter spectra…'}</Empty>
  </ChartFrame>
  if (selection.missing.length) return <ChartFrame id="satoshi-spectra" title="Spectra" exportable={false}>
    <Empty>{selection.missing.length} saved analogs are missing from the spectral export. Re-export the matching spectra to show the complete ranked cohort.</Empty>
  </ChartFrame>
  return <SpectraTraces id="satoshi-spectra" source="gallery/app/src/components/SatoshiSpectra.tsx"
    meta={activeMeta} traces={loaded.traces} target={selection.target} targetFilters={selection.filters}
    targetLabel={`${group} · frozen Addis test set`} topFilters={selection.shown} compare={null}
    mask={region === 'oct1' ? 'oct1' : 'no_co2_max3600'} hoverFilter={null} pinned={pinned >= 0 ? pinned : null}
    onPin={onPin} method={baselineLabel(baseline)} regionParam="satSpecIgnored" scoreLabel="r to median Addis spectrum"
    pinInstructions="Hover a trace to identify it; click to pin it and open its filter details. Ranks compare each analog with the median Addis spectrum."
    provenance="Saved Satoshi follow-up membership ranks · export_similarity.py binned per-filter traces · frozen Addis test-set membership"
    controls={<>
      <CountPicker label="Most similar" value={topN} presets={TOP_N} max={500} onChange={setTopN} noun="analogs" />
      <Toggle label="Outer test sites" checked={test === 'on'} onChange={value => setTest(value ? 'on' : 'off')} />
    </>} />
}
