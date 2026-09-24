import { useState } from 'react'
import * as d3 from 'd3'
import { Segmented } from '@/components/ChartFrame'
import { XAxis, YAxis } from '@/components/Axes'
import { INK, FONT } from '@/lib/theme'
import { useFilterSpectrum, type SpectrumMethod } from '@/lib/filterSpectrum'

// the baseline colours used on the AIRSpec / VIBES pages
const COLOR: Record<SpectrumMethod, string> = { AIRSpec: '#2C6E9E', VIBES: '#7A4FA3' }
const NAME: Record<SpectrumMethod, string> = { AIRSpec: 'Spline baseline (AIRSpec)', VIBES: 'VIBES baseline' }
const VIEWS = ['Spline baseline (AIRSpec)', 'VIBES baseline', 'both'] as const
const METHODS_OF: Record<(typeof VIEWS)[number], SpectrumMethod[]> = {
  'Spline baseline (AIRSpec)': ['AIRSpec'],
  'VIBES baseline': ['VIBES'],
  both: ['AIRSpec', 'VIBES'],
}

/**
 * This filter's baseline-corrected FTIR spectrum, when the Spectral similarity
 * export carries it. Renders nothing otherwise (and while loading), so drawers
 * for filters without a spectrum look exactly as before.
 */
export function FilterSpectrum({ id }: { id: string }) {
  const [view, setView] = useState<(typeof VIEWS)[number]>('both')
  const methods = METHODS_OF[view]
  const spec = useFilterSpectrum(id, methods)
  if (!spec) return null
  const shown = methods.filter((m) => spec.y[m])
  if (!shown.length) return null

  const W = 420, m = { top: 10, right: 12, bottom: 44, left: 62 }, ih = 170
  const iw = W - m.left - m.right, H = ih + m.top + m.bottom
  const wn = spec.wn
  const xs = d3.scaleLinear().domain([d3.max(wn) ?? 4000, d3.min(wn) ?? 1425]).range([0, iw])
  let lo = Infinity, hi = -Infinity
  for (const k of shown) for (const v of spec.y[k]!) { if (v < lo) lo = v; if (v > hi) hi = v }
  const pad = (hi - lo) * 0.05 || 0.01
  const ys = d3.scaleLinear().domain([lo - pad, hi + pad]).range([ih, 0]).nice()
  const line = (y: Float32Array) => d3.line<number>().x((_, i) => xs(wn[i])).y((v) => ys(v)).defined((v) => Number.isFinite(v))(Array.from(y)) ?? ''

  return (
    <section className="drawer-section">
      <h3>Baseline-corrected spectrum</h3>
      <div className="frame-controls" style={{ marginBottom: 4, flexWrap: 'wrap' }}>
        <Segmented value={view} options={VIEWS} onChange={setView} />
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: 'block', maxWidth: W }}>
        <g transform={`translate(${m.left},${m.top})`}>
          <YAxis scale={ys} x={0} label="absorbance" gridWidth={iw} tickCount={4} labelX={-50} format={(v: number) => d3.format('~g')(v)} />
          <XAxis scale={xs} y={ih} label="wavenumber (cm⁻¹)" tickCount={6} labelDy={36} />
          {ys.domain()[0] < 0 && ys.domain()[1] > 0 && <line x1={0} x2={iw} y1={ys(0)} y2={ys(0)} stroke={INK.identity} strokeDasharray="3 3" />}
          {shown.map((k) => (
            <path key={k} d={line(spec.y[k]!)} fill="none" stroke={COLOR[k]} strokeWidth={1.4} strokeOpacity={shown.length > 1 ? 0.85 : 1} />
          ))}
        </g>
      </svg>
      <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap', fontSize: 11.5, color: INK.text, fontFamily: FONT.family, margin: '2px 0 4px' }}>
        {shown.map((k) => (
          <span key={k} style={{ display: 'inline-flex', alignItems: 'center', gap: 5 }}>
            <span style={{ width: 16, height: 2, background: COLOR[k], display: 'inline-block' }} />
            {NAME[k]}
          </span>
        ))}
      </div>
      <p className="chart-note" style={{ display: 'block' }}>
        From the Spectral similarity export ({spec.sampleId}): 8-channel means (~10 cm⁻¹), for display only.
      </p>
    </section>
  )
}
