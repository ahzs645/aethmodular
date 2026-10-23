import { useState } from 'react'
import { Segmented } from '@/components/ChartFrame'
import { INK, FONT, BASELINE_COLOR } from '@/lib/theme'
import { useAddisStatus, useFilterSpectrum, traceVersionLabel, type AddisStatus, type FilterSpectrumData, type SpectrumMethod } from '@/lib/filterSpectrum'
import { SpectrumModal, SpectrumPlot } from '@/components/FilterSpectrumPlot'
import { NativeFilterMeasurements } from '@/components/NativeFilterMeasurements'

// the baseline colours used on the AIRSpec / VIBES pages
// the drawer's VIBES is always the completed full-range run
const COLOR: Record<SpectrumMethod, string> = { get AIRSpec() { return BASELINE_COLOR.AIRSpec }, get VIBES() { return BASELINE_COLOR['VIBES-full-r70'] } }
const NAME: Record<SpectrumMethod, string> = { AIRSpec: 'Spline baseline', VIBES: 'VIBES baseline' }
const VIEWS = ['Spline baseline', 'VIBES baseline', 'both'] as const
const METHODS_OF: Record<(typeof VIEWS)[number], SpectrumMethod[]> = {
  'Spline baseline': ['AIRSpec'],
  'VIBES baseline': ['VIBES'],
  both: ['AIRSpec', 'VIBES'],
}
/** One line saying where an Addis filter stands (1 Oct: label excluded samples clearly). */
function AddisStatusNote({ status }: { status: AddisStatus }) {
  return <p className="chart-note" style={{ display: 'block', margin: '4px 0 8px' }}>
    <span className={`badge ${status.in_test_set ? '' : 'danger'}`} style={{ marginRight: 6 }}>{status.in_test_set ? 'Addis test set' : 'not in the Addis test set'}</span>
    {status.note.charAt(0).toUpperCase() + status.note.slice(1)}.
  </p>
}

/**
 * This filter's baseline-corrected FTIR spectrum, when the Spectral similarity
 * export carries it. For Addis filters a status line always says whether the
 * filter is in the test set and, if not, why (no HIPS, PM10, blank, not in the run);
 * other filters without a spectrum render nothing, as before.
 */
export function FilterSpectrum({ id }: { id: string }) {
  // SPARTAN filters (Addis included) have the native FTIR block; the frozen export's corrected
  // trace stays only for the IMPROVE samples, which have no native spectrum
  const native = /^[A-Z]{4}-\d{4}/.test(id)
  return <><NativeFilterMeasurements id={id} />{!native && <CorrectedFilterSpectrum id={id} />}</>
}

function CorrectedFilterSpectrum({ id, embedded = false }: { id: string; embedded?: boolean }) {
  const [view, setView] = useState<(typeof VIEWS)[number]>('both')
  const [expanded, setExpanded] = useState(false)
  const methods = METHODS_OF[view]
  const spec = useFilterSpectrum(id, methods)
  const status = useAddisStatus(id)
  if (!spec) return embedded ? <p className="chart-note" style={{ display: 'block' }}>{status && !status.in_run ? 'This filter is not in the frozen runs.' : 'Loading…'}</p>
    : status && !status.in_run ? <section className="drawer-section">
    <h3>Baseline-corrected spectrum</h3>
    <AddisStatusNote status={status} />
  </section> : null
  const shown = methods.filter((m) => spec.series[m])
  if (!shown.length) return <section className="drawer-section">
    <h3>Baseline-corrected spectrum</h3>
    <div className="frame-controls"><Segmented value={view} options={VIEWS} onChange={setView} /></div>
    <p className="chart-note" style={{ display: 'block' }}>The requested corrected spectra could not be loaded.</p>
  </section>

  const Wrap = embedded ? 'div' : 'section'
  return (
    <Wrap className={embedded ? undefined : 'drawer-section'}>
      {!embedded && <h3>Baseline-corrected spectrum</h3>}
      {!embedded && status && <AddisStatusNote status={status} />}
      <div className="frame-controls" style={{ marginBottom: 4, flexWrap: 'wrap' }}>
        <Segmented value={view} options={VIEWS} onChange={setView} />
      </div>
      <SpectrumPlot key={id} spec={spec} shown={shown}
        actions={<button type="button" className="btn quiet" aria-haspopup="dialog" onClick={() => setExpanded(true)}>Expand</button>} />
      <SpectrumLegend shown={shown} />
      <SpectrumNote spec={spec} methods={methods} shown={shown} />
      {expanded && <SpectrumModal id={id} onClose={() => setExpanded(false)}>
        <div className="frame-controls" style={{ marginBottom: 8, flexWrap: 'wrap' }}>
          <Segmented value={view} options={VIEWS} onChange={setView} />
        </div>
        <SpectrumPlot key={id} spec={spec} shown={shown} expanded />
        <SpectrumLegend shown={shown} />
        <SpectrumNote spec={spec} methods={methods} shown={shown} />
      </SpectrumModal>}
    </Wrap>
  )
}

function SpectrumLegend({ shown }: { shown: SpectrumMethod[] }) {
  return <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap', fontSize: 11.5, color: INK.text, fontFamily: FONT.family, margin: '2px 0 4px' }}>
    {shown.map(k => <span key={k} style={{ display: 'inline-flex', alignItems: 'center', gap: 5 }}>
      <span style={{ width: 16, height: 2, background: COLOR[k], display: 'inline-block' }} />{NAME[k]}
    </span>)}
  </div>
}

function SpectrumNote({ spec, methods, shown }: { spec: FilterSpectrumData; methods: SpectrumMethod[]; shown: SpectrumMethod[] }) {
  return <p className="chart-note" style={{ display: 'block' }}>
    From the Spectral similarity export ({spec.sampleId}): 8-channel means (~10 cm⁻¹), for display only.
    {spec.series.VIBES && shown.includes('VIBES') && <> VIBES uses the full-range fit (4000–500 cm⁻¹) · {traceVersionLabel(spec.series.VIBES)}. {shown.includes('AIRSpec') && <>Spline baseline stops near 1425 cm⁻¹.</>}</>}
    {spec.unavailable.filter(k => methods.includes(k)).map(k => <span key={k}> {NAME[k]} could not be loaded{ k === 'VIBES' && '; the archived truncated fit is not substituted'}.</span>)}
  </p>
}
