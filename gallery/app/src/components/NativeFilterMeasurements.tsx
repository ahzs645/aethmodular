import { useEffect, useState } from 'react'
import { baseFilterId } from '@/lib/highlight'
import { loadSpartanIndex, loadNativeSite, loadNativeTraces, type NativeSite, type NativeTraces, type SpartanIndex } from '@/lib/spartan'
import { SpectrumModal, SpectrumPlot } from '@/components/FilterSpectrumPlot'
import { useAddisStatus, useFilterSpectrum } from '@/lib/filterSpectrum'
import { BASELINE_COLOR, INK } from '@/lib/theme'
import { Toggle } from '@/components/ChartFrame'
import { Combobox } from '@/components/Combobox'
import { DrawerScroll } from '@/components/DrawerShell'

const TRACE_KEYS = ['raw', 'airspec', 'vibes'] as const
type TraceKey = (typeof TRACE_KEYS)[number]
const LABEL: Record<TraceKey, string> = { raw: 'Raw', airspec: 'Spline baseline', vibes: 'VIBES' }
const COLORS = { get raw() { return INK.text }, get airspec() { return BASELINE_COLOR.AIRSpec }, get vibes() { return BASELINE_COLOR['VIBES-full-r70'] } }
const NAMES: Record<TraceKey, string> = { raw: 'Raw absorbance', airspec: 'Corrected, spline baseline', vibes: 'Corrected, VIBES' }
const fmtDate = (d: string) => (d ? new Date(`${d}T00:00:00`).toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' }) : '')

/**
 * The filter's FTIR spectrum from the native SPARTAN archive: raw absorbance, and the AIRSpec and
 * VIBES corrections precomputed for every analysis (no model refit), switched in one control.
 * Provenance and the HIPS readouts sit behind details so the drawer stays short.
 */
export function NativeFilterMeasurements({ id }: { id: string }) {
  const base = baseFilterId(id)
  const code = /^[A-Z]{4}-\d{4}$/.test(base) ? base.slice(0, 4) : null
  const [state, setState] = useState<{ code: string; index: SpartanIndex | null; data: NativeSite | null; error: string } | null>(null)
  const [choice, setChoice] = useState<{ base: string; analysis: string } | null>(null)
  const [trace, setTrace] = useState<{ key: string; traces: NativeTraces | null; error: string } | null>(null)
  // any combination of the traces, drawn together; at least one stays on. The spline-corrected
  // spectrum by default (raw where a filter has no AIRSpec trace)
  const [picked, setPicked] = useState<TraceKey[]>(['airspec'])
  const [expanded, setExpanded] = useState(false)
  useEffect(() => {
    if (!code) return
    let live = true
    loadSpartanIndex().then(async index => {
      const site = index.sites.find(s => s.code === code)
      return { index, data: site ? await loadNativeSite(site) : null }
    }).then(({ index, data }) => { if (live) setState({ code, index, data, error: '' }) })
      .catch(e => { if (live) setState({ code, index: null, data: null, error: String(e) }) })
    return () => { live = false }
  }, [code])
  const native = state?.code === code ? state.data : null
  const vibesInfo = state?.index?.baselines?.vibes ?? null
  // An archive entry is one physical filter (no entry mixes PM2.5, PM10 and blanks), so a PM2.5
  // filter shows its PM2.5 scans; a filter held only as PM10 (ETAD-0248) shows its own PM10 scan,
  // labelled. Blanks are never drawn here.
  const entry = native?.spectra[base]
  const pm25 = entry?.analyses.filter(a => a.filter_type === 'PM2.5') ?? []
  const analyses = pm25.length ? pm25 : entry?.analyses.filter(a => a.filter_type === 'PM10') ?? []
  const pm10 = !pm25.length && analyses.length > 0
  const analysis = analyses.find(a => choice?.base === base && a.analysis_id === choice.analysis) ?? analyses[0]
  const key = `${base}:${analysis?.analysis_id ?? ''}:${entry?.file ?? ''}`
  useEffect(() => {
    if (!analysis || !entry) return
    let live = true
    loadNativeTraces(entry.file, analysis, vibesInfo?.grid).then(traces => { if (live) setTrace({ key, traces, error: '' }) })
      .catch(e => { if (live) setTrace({ key, traces: null, error: String(e) }) })
    return () => { live = false }
  }, [key, vibesInfo]) // eslint-disable-line react-hooks/exhaustive-deps
  if (!code) return null
  const error = state?.code === code ? state.error : ''
  const current = trace?.key === key ? trace : null
  const t = current?.traces
  // Until the network VIBES run is exported, an Addis filter in the frozen runs shows that run's
  // VIBES trace (10 cm⁻¹ bins), fetched only when VIBES is ticked; elsewhere VIBES is offered only
  // where the export has it (never for a filter without one, e.g. IMPROVE)
  const status = useAddisStatus(id)
  const frozenVibes = !!t && !t.vibes && !!status?.in_run
  const frozen = useFilterSpectrum(frozenVibes && picked.includes('vibes') ? id : null, ['VIBES'])
  const vibes = t?.vibes ?? (frozenVibes ? frozen?.series.VIBES : undefined)
  const has: Record<TraceKey, boolean> = { raw: !!t, airspec: !!t?.airspec, vibes: !!t?.vibes || frozenVibes }
  const options = TRACE_KEYS.filter(k => has[k])
  const shownKeys = picked.filter(k => has[k])
  const shown = shownKeys.length ? shownKeys : ['raw' as const]
  const series = t ? { raw: t.raw, airspec: t.airspec, vibes } : {}
  const names = { ...NAMES, vibes: frozenVibes ? 'Corrected, VIBES (frozen Addis run, 10 cm⁻¹ bins)' : NAMES.vibes }
  const toggle = (k: TraceKey) => setPicked(cur => cur.includes(k) ? (cur.length > 1 ? cur.filter(x => x !== k) : cur) : [...cur.filter(x => has[x]), k])
  const plot = (large = false) => t && <SpectrumPlot spec={{ series }} shown={shown} names={names} colors={COLORS} expanded={large}
    plotLabel={`${shown.map(k => LABEL[k]).join(' and ')} FTIR spectrum`} roundWindow={false}
    actions={!large && <button type="button" className="btn quiet" aria-haspopup="dialog" title="Open a larger plot" onClick={() => setExpanded(true)}>Expand</button>} />
  // any combination, so one switch per spectrum (the gallery's Toggle); at least one stays on
  const controls = options.length > 1 && <span className="control" role="group" aria-label="Spectra shown"
    title="Tick any combination. Raw: −log₁₀(sample/background) as measured. Spline baseline (AIRSpec) and VIBES: the same spectrum baseline-corrected.">
    {options.map(k => <Toggle key={k} label={LABEL[k]} checked={shown.includes(k)} onChange={() => toggle(k)} />)}
  </span>
  const legend = <div className="ftir-legend">{shown.map(k => <span key={k}><i style={{ borderColor: COLORS[k] }} />{names[k]}{k === 'vibes' && frozenVibes && !vibes ? ' · loading…' : ''}</span>)}</div>
  const otherSite = code !== 'ETAD'
  // what used to sit behind "About this spectrum", now the analysis line's tooltip
  const about = analysis ? [
    `Raw absorbance = −log₁₀(sample/background) at full native resolution (${analysis.n.toLocaleString()} points)${analysis.invalid_points > 0 ? `, ${analysis.invalid_points} nonpositive single-beam pairs shown as gaps` : ''}; filter ${analysis.filter_id}, sample scan ${analysis.sample_scan_id}, background scan ${analysis.background_scan_id}. QC is the source's code; every analysis, repeats included, is kept.`,
    has.airspec ? 'Spline baseline (AIRSpec): the APRLssb segmented smoothing spline (df 6 and 4) fitted to this spectrum and subtracted; 1425–4000 cm⁻¹ only, as by design.' : '',
    has.vibes && vibesInfo ? `VIBES: the gallery's saved blank background (${vibesInfo.grid.model_blanks} blanks, rank ${vibesInfo.grid.model_rank}; not refitted), applied after averaging this spectrum onto its ${vibesInfo.grid.n.toLocaleString()}-point grid.${otherSite ? ' That background was learned from IMPROVE and Addis blanks, not this site’s own.' : ''}` : '',
  ].filter(Boolean).join('\n\n') : ''

  return <section className="drawer-section">
    <div className="ftir-head">
      <h3>FTIR spectrum</h3>
      {analysis && <span className="ftir-meta" title={about}>
        {analyses.length > 1
          ? <Combobox label="" title="FTIR analysis" value={String(analysis.analysis_id)} onChange={(v) => setChoice({ base, analysis: v })}
            options={analyses.map(a => ({ key: String(a.analysis_id), label: `Analysis ${a.analysis_id} · ${fmtDate(a.analysis_date)}` }))} />
          : <>Analysis {analysis.analysis_id} · {fmtDate(analysis.analysis_date)}</>}
        {' · '}QC {analysis.qc || '—'}{pm10 && <> · <strong>PM10 filter</strong></>}</span>}
    </div>
    {error ? <p role="alert">The spectrum could not be loaded: {error}</p>
      : !native ? <p className="ftir-meta">{state?.code === code ? 'No native spectrum for this site in this export.' : 'Loading…'}</p>
      : !analysis ? <p className="ftir-meta">{entry
        ? 'The native FTIR archive holds this filter only as a blank scan, so there is no sample spectrum to draw.'
        : 'This filter is not in the native FTIR archive, so there is no spectrum to draw; its measurements come from the public data.'}</p>
      : <>
        {pm10 && <p className="ftir-meta">The archive holds this filter as PM10, not PM2.5: its own spectrum is shown, but it sits outside every PM2.5 comparison.</p>}
        {controls}
        {current?.error ? <p role="alert">Spectrum could not be loaded: {current.error}</p> : t ? <>{plot()}{legend}</> : <p className="ftir-meta">Loading this filter’s spectrum…</p>}
        {otherSite && shown.includes('vibes') && <p className="ftir-meta">VIBES here uses a background learned from IMPROVE and Addis blanks, not this site’s own.</p>}
        {expanded && t && <SpectrumModal id={id} title="FTIR spectrum" onClose={() => setExpanded(false)}>
          <div className="frame-controls" style={{ marginBottom: 8 }}>{controls}</div>
          {plot(true)}{legend}
        </SpectrumModal>}
      </>}
  </section>
}

/**
 * The raw HIPS readouts matched to this PM2.5 filter (native archive), as a compact table for the
 * drawer's "Where this filter sits" section, beside the delivered HIPS values. Nothing renders when
 * the filter has none.
 */
export function HipsReadings({ id }: { id: string }) {
  const base = baseFilterId(id)
  const code = /^[A-Z]{4}-\d{4}$/.test(base) ? base.slice(0, 4) : null
  const [state, setState] = useState<{ code: string; data: NativeSite | null } | null>(null)
  useEffect(() => {
    if (!code) return
    let live = true
    loadSpartanIndex().then(async index => {
      const site = index.sites.find(s => s.code === code)
      return site ? loadNativeSite(site) : null
    }).then(data => { if (live) setState({ code, data }) }).catch(() => { if (live) setState({ code, data: null }) })
    return () => { live = false }
  }, [code])
  const hips = (state?.code === code ? state.data?.hips[base] : null)?.filter(r => r.filter_type === 'PM2.5') ?? []
  if (!hips.length) return null
  return <DrawerScroll label="HIPS instrument readings"><table className="placement hips-readings">
    <thead>
      <tr><th colSpan={8} title="Instrument transmittance and reflectance readouts from the native archive. They are not the delivered Fabs or BC and never replace them.">HIPS instrument readings</th></tr>
      <tr className="sub"><th>Analysis</th><th>Date</th><th>QC</th><th>nm</th><th>T</th><th>R</th><th>Raw T</th><th>Raw R</th></tr>
    </thead>
    <tbody>{hips.map((r, i) => <tr key={`${r.analysis_id}-${i}`}><td>{r.analysis_id}</td><td>{r.timestamp.slice(0, 10)}</td><td>{r.qc || '—'}</td>
      {[r.Wavelength, r.Transmittance, r.Reflectance, r.TransmittanceRaw, r.ReflectanceRaw].map((v, j) => <td key={j}>{v ?? '—'}</td>)}
    </tr>)}</tbody>
  </table></DrawerScroll>
}
