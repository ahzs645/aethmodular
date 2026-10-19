import { useEffect, useId, useRef, useState, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import * as d3 from 'd3'
import { RefLine } from '@/components/RefLines'
import { Toggle } from '@/components/ChartFrame'
import { XAxis, YAxis } from '@/components/Axes'
import { INK, BASELINE_COLOR } from '@/lib/theme'
import { useDimensions } from '@/hooks/useGalleryData'
import type { FilterSpectrumData, SpectrumMethod } from '@/lib/filterSpectrum'

const COLOR: Record<SpectrumMethod, string> = { AIRSpec: BASELINE_COLOR.AIRSpec, VIBES: BASELINE_COLOR['VIBES-full-r70'] }
const NAME: Record<SpectrumMethod, string> = { AIRSpec: 'Spline baseline (AIRSpec)', VIBES: 'VIBES baseline' }

export function SpectrumModal({ id, onClose, children, title = 'Baseline-corrected spectrum' }: { id: string; onClose: () => void; children: ReactNode; title?: string }) {
  const dialogRef = useRef<HTMLDialogElement>(null)
  const closeRef = useRef<HTMLButtonElement>(null)
  const titleId = useId()
  useEffect(() => {
    const dialog = dialogRef.current!
    const previousFocus = document.activeElement
    const previousOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    dialog.showModal()
    closeRef.current?.focus()
    return () => {
      dialog.close()
      document.body.style.overflow = previousOverflow
      if (previousFocus instanceof HTMLElement && previousFocus.isConnected) previousFocus.focus()
    }
  }, [])
  return createPortal(<dialog ref={dialogRef} className="spectrum-modal" aria-labelledby={titleId} aria-modal="true"
    onCancel={e => { e.preventDefault(); onClose() }}
    onKeyDown={e => {
      // Escape closes only the modal; drawer navigation must not run behind it.
      e.stopPropagation()
      if (e.key === 'Escape') { e.preventDefault(); onClose() }
      if (e.key === 'Tab') {
        const controls = Array.from(e.currentTarget.querySelectorAll<HTMLElement>('button:not([disabled]), input:not([disabled]), select:not([disabled]), a[href], [tabindex]:not([tabindex="-1"])'))
          .filter(el => el.tabIndex >= 0 && el.getClientRects().length > 0)
        const first = controls[0], last = controls[controls.length - 1]
        if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last?.focus() }
        else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first?.focus() }
      }
    }}
    onClick={e => {
      e.stopPropagation()
      if (e.target !== e.currentTarget) return
      const b = e.currentTarget.getBoundingClientRect()
      if (e.clientX < b.left || e.clientX > b.right || e.clientY < b.top || e.clientY > b.bottom) onClose()
    }}>
    <header className="spectrum-modal-head">
      <div><h2 id={titleId}>{title}</h2><p>{id}</p></div>
      <button ref={closeRef} type="button" className="btn quiet" aria-label="Close spectrum modal" onClick={onClose}>✕</button>
    </header>
    {children}
  </dialog>, document.body)
}

/** Zoom the wavenumber window; absorbance follows the visible measurements. */
export function SpectrumPlot({ spec, shown, expanded = false, names = NAME, plotLabel = 'Baseline-corrected spectrum plot', roundWindow = true }: { spec: FilterSpectrumData; shown: SpectrumMethod[]; expanded?: boolean; names?: Record<SpectrumMethod, string>; plotLabel?: string; roundWindow?: boolean }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const clipId = useId()
  const hitRef = useRef<SVGRectElement>(null)
  const behaviorRef = useRef<d3.ZoomBehavior<SVGRectElement, unknown> | null>(null)
  const [enabled, setEnabled] = useState(false)
  const enabledRef = useRef(enabled)
  enabledRef.current = enabled
  const [zoom, setZoom] = useState({ rangeKey: '', transform: d3.zoomIdentity })
  const W = expanded ? Math.max(280, Math.min(1056, width || 1000)) : 420
  const m = { top: 10, right: 12, bottom: 44, left: 62 }, ih = expanded ? 300 : 170
  const iw = W - m.left - m.right, H = ih + m.top + m.bottom
  const wn = shown.flatMap(k => spec.series[k]!.wn)
  // Round bin centres to readable endpoints, without extending either trace.
  const endpoint = (v: number) => roundWindow ? Math.round(v / 25) * 25 : v
  const maxWn = endpoint(d3.max(wn) ?? 4000)
  const minWn = endpoint(d3.min(wn) ?? 1425)
  const rangeKey = `${maxWn}:${minWn}`
  const transform = zoom.rangeKey === rangeKey ? zoom.transform : d3.zoomIdentity
  const xs = transform.rescaleX(d3.scaleLinear().domain([maxWn, minWn]).range([0, iw]))
  const [high, low] = xs.domain()

  useEffect(() => {
    if (!hitRef.current) return
    const hit = d3.select(hitRef.current)
    const behavior = d3.zoom<SVGRectElement, unknown>()
      .extent([[0, 0], [iw, ih]])
      .translateExtent([[0, 0], [iw, ih]])
      .scaleExtent([1, 32])
      .filter(e => enabledRef.current && (!e.ctrlKey || e.type === 'wheel') && !e.button)
      .on('zoom', e => setZoom({ rangeKey, transform: e.transform }))
    behaviorRef.current = behavior
    hit.call(behavior).call(behavior.transform, d3.zoomIdentity)
    hit.on('dblclick.zoom', null)
    return () => { hit.on('.zoom', null); behaviorRef.current = null }
  }, [rangeKey, iw, ih])

  const scaleBy = (factor: number) => {
    if (!hitRef.current || !behaviorRef.current) return
    setEnabled(true)
    d3.select(hitRef.current).call(behaviorRef.current.scaleBy, factor, [iw / 2, ih / 2])
  }
  const reset = () => {
    if (hitRef.current && behaviorRef.current) d3.select(hitRef.current).call(behaviorRef.current.transform, d3.zoomIdentity)
    setEnabled(false)
  }
  let lo = Infinity, hi = -Infinity
  const absent: SpectrumMethod[] = []
  for (const k of shown) {
    const trace = spec.series[k]!
    let n = 0
    for (let i = 0; i < trace.y.length; i++) {
      const v = trace.y[i]
      if (trace.wn[i] >= low && trace.wn[i] <= high && Number.isFinite(v)) {
        lo = Math.min(lo, v); hi = Math.max(hi, v); n++
      }
    }
    if (!n) absent.push(k)
  }
  if (!Number.isFinite(lo)) { lo = 0; hi = 0 }
  const pad = (hi - lo) * 0.05 || 0.01
  const ys = d3.scaleLinear().domain([lo - pad, hi + pad]).range([ih, 0]).nice()
  const line = (k: SpectrumMethod) => {
    const trace = spec.series[k]!
    return d3.line<number>().x((_, i) => xs(trace.wn[i])).y(v => ys(v)).defined(v => Number.isFinite(v))(Array.from(trace.y)) ?? ''
  }

  return <div ref={wrapRef}>
    <div className="frame-controls" style={{ flexWrap: 'wrap', marginBottom: 4 }}>
      <Toggle label="Zoom / pan" checked={enabled} onChange={setEnabled} />
      <button type="button" className="btn quiet" aria-label="Zoom spectrum out" disabled={transform.k <= 1} onClick={() => scaleBy(0.5)}>−</button>
      <button type="button" className="btn quiet" aria-label="Zoom spectrum in" disabled={transform.k >= 32} onClick={() => scaleBy(2)}>+</button>
      <button type="button" className="btn quiet" onClick={reset}>Reset zoom</button>
      <span style={{ fontSize: 11, color: INK.muted }}>{transform.k.toFixed(1)}× · {Math.round(high)}–{Math.round(low)} cm⁻¹</span>
    </div>
    <svg aria-label={plotLabel} viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: 'block', maxWidth: W }}>
      <defs><clipPath id={clipId}><rect width={iw} height={ih} /></clipPath></defs>
      <g transform={`translate(${m.left},${m.top})`}>
        <YAxis scale={ys} x={0} label="absorbance" gridWidth={iw} tickCount={4} labelX={-50} format={(v: number) => d3.format('~g')(v)} />
        <XAxis scale={xs} y={ih} label="wavenumber (cm⁻¹)" tickCount={6} labelDy={36} />
        <g clipPath={`url(#${clipId})`}>
          {ys.domain()[0] < 0 && ys.domain()[1] > 0 && <RefLine orient="h" at={ys(0)} from={0} to={iw} />}
          {shown.map(k => <path key={k} aria-label={`${names[k]} spectrum`} d={line(k)} fill="none" stroke={COLOR[k]} strokeWidth={1.4} strokeOpacity={shown.length > 1 ? 0.85 : 1} />)}
        </g>
        <rect ref={hitRef} width={iw} height={ih} fill="transparent" pointerEvents={enabled ? 'all' : 'none'} style={{ cursor: enabled ? 'grab' : undefined, touchAction: enabled ? 'none' : 'auto' }} />
      </g>
    </svg>
    <p style={{ fontSize: 11, color: INK.muted, margin: '2px 0 6px' }}>Enable Zoom / pan to scroll or pinch to zoom, and drag to pan. Absorbance adjusts to the visible range.</p>
    {absent.map(k => <p key={k} style={{ fontSize: 11, color: INK.muted }}>{names[k]} has no measurements in this window.</p>)}
  </div>
}
