import { useEffect, useState } from 'react'
import { baseFilterId } from '@/lib/highlight'
import { loadSpartanIndex, loadNativeSite, loadNativeTrace, type NativeSite } from '@/lib/spartan'
import { SpectrumModal, SpectrumPlot } from '@/components/FilterSpectrumPlot'
import type { FilterSpectrumData } from '@/lib/filterSpectrum'

/** The native archive is independent of the frozen Addis calibration runs. */
export function NativeFilterMeasurements({ id }: { id: string }) {
  const base = baseFilterId(id)
  const code = /^[A-Z]{4}-\d{4}$/.test(base) ? base.slice(0, 4) : null
  const [state, setState] = useState<{ code: string; data: NativeSite | null; error: string } | null>(null)
  const [choice, setChoice] = useState<{ base: string; analysis: string } | null>(null)
  const [trace, setTrace] = useState<{ key: string; spec: FilterSpectrumData | null; error: string } | null>(null)
  const [expanded, setExpanded] = useState(false)
  useEffect(() => {
    if (!code) return
    let live = true
    loadSpartanIndex().then(async index => {
      const site = index.sites.find(s => s.code === code)
      return site ? loadNativeSite(site) : null
    }).then(data => { if (live) setState({ code, data, error: '' }) })
      .catch(e => { if (live) setState({ code, data: null, error: String(e) }) })
    return () => { live = false }
  }, [code])
  const native = state?.code === code ? state.data : null
  // PM10 and blanks are retained in the archive but never substituted for a PM2.5 drawer.
  const entry = native?.spectra[base]
  const analyses = entry?.analyses.filter(a => a.filter_type === 'PM2.5') ?? []
  const hips = native?.hips[base]?.filter(r => r.filter_type === 'PM2.5') ?? []
  const analysis = analyses.find(a => choice?.base === base && a.analysis_id === choice.analysis) ?? analyses[0]
  const key = `${base}:${analysis?.analysis_id ?? ''}:${entry?.file ?? ''}`
  useEffect(() => {
    if (!analysis || !entry) return
    let live = true
    loadNativeTrace(entry.file, analysis).then(series => {
      if (live) setTrace({ key, spec: { forId: id, requestKey: key, sampleId: analysis.analysis_id,
        series: { AIRSpec: series }, unavailable: [] }, error: '' })
    }).catch(e => { if (live) setTrace({ key, spec: null, error: String(e) }) })
    return () => { live = false }
  }, [key]) // eslint-disable-line react-hooks/exhaustive-deps
  if (!code) return null
  const current = trace?.key === key ? trace : null
  const error = state?.code === code ? state.error : ''
  const plot = (large = false) => current?.spec && <SpectrumPlot spec={current.spec} shown={['AIRSpec']} expanded={large}
    names={{ AIRSpec: 'Raw absorbance', VIBES: 'Raw absorbance' }} plotLabel="Native raw absorbance spectrum" roundWindow={false} />
  return <section className="drawer-section">
    <h3>Native FTIR and HIPS records</h3>
    {error ? <p role="alert">Native records could not be loaded: {error}</p> : !native ? <p>{state?.code === code ? 'No native site records in this export.' : 'Loading native record coverage…'}</p> : <>
      <p className="chart-note" style={{ display: 'block' }}>{analyses.length} FTIR analyses · {hips.length} HIPS sample readings matched to this PM2.5 filter.
        {!analyses.length && ' No matching native spectrum in this export; public measurements may still be available.'}</p>
      {analysis && <>
        <label>FTIR analysis <select value={analysis.analysis_id} onChange={e => setChoice({ base, analysis: e.target.value })}>
          {analyses.map(a => <option key={a.analysis_id} value={a.analysis_id}>{a.analysis_id} · {a.analysis_date} · QC {a.qc || 'unreported'}</option>)}
        </select></label>
        <p className="chart-note" style={{ display: 'block' }}>Raw absorbance = −log₁₀(sample/background), without baseline correction. Full native resolution ({analysis.n.toLocaleString()} points).
          {analysis.invalid_points > 0 && ` ${analysis.invalid_points} nonpositive single-beam pairs are shown as gaps.`}
          {' '}Filter {analysis.filter_id} · sample scan {analysis.sample_scan_id} · background scan {analysis.background_scan_id}.
          QC is the source code; all exported analyses, including repeats, are retained.</p>
        {current?.error ? <p role="alert">Spectrum could not be loaded: {current.error}</p> : current?.spec ? <>
          {plot()}<button type="button" className="btn" onClick={() => setExpanded(true)}>Expand raw spectrum</button>
        </> : <p>Loading this filter’s spectrum…</p>}
        {expanded && current?.spec && <SpectrumModal id={id} title="Raw FTIR spectrum" onClose={() => setExpanded(false)}>{plot(true)}</SpectrumModal>}
      </>}
      {!!hips.length && <details><summary>Raw HIPS sample readings ({hips.length})</summary>
        <p className="chart-note" style={{ display: 'block' }}>Instrument transmittance and reflectance readouts. These are not delivered Fabs or BC concentrations; they do not replace those fields.</p>
        <div style={{ overflowX: 'auto' }}><table><thead><tr><th>Analysis</th><th>Date</th><th>QC</th><th>nm</th><th>T</th><th>R</th><th>Raw T</th><th>Raw R</th></tr></thead>
          <tbody>{hips.map((r, i) => <tr key={`${r.analysis_id}-${i}`}><td>{r.analysis_id}</td><td>{r.timestamp.slice(0, 10)}</td><td>{r.qc || '—'}</td>
            {[r.Wavelength, r.Transmittance, r.Reflectance, r.TransmittanceRaw, r.ReflectanceRaw].map((v, j) => <td key={j}>{v ?? '—'}</td>)}
          </tr>)}</tbody></table></div>
      </details>}
    </>}
  </section>
}
