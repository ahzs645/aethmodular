import { useMemo, useRef, useState, type ReactNode } from 'react'
import * as d3 from 'd3'
import { XAxis, YAxis } from '@/components/Axes'
import { Marker, RefLine } from '@/components/RefLines'
import { SpectralRegions, type Region } from '@/components/SpectralRegions'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { robustDomain } from '@/lib/axes'
import { nearestIndex } from '@/lib/spectra'
import { FONT, INK, TRACE, themed } from '@/lib/theme'

/**
 * The gallery's one spectrum chart: a reversed wavenumber axis, region
 * shading, faint per-filter traces, emphasised lines, median/IQR bands, and
 * hover and pin (docs/gallery-chart-standards.md §2). Every series carries its
 * own wavenumber grid, so traces and quantiles on different grids, or a
 * method that stops at 1425 cm⁻¹, draw together; the visible `window` clips
 * them all.
 *
 * Hover is one mechanism for every spectrum: `nearest` names the trace under
 * the pointer (within 8 px) and clicking pins it; `readout` draws a crosshair
 * and lists every line's value there. Pass `highlight={useHighlight()}` (and
 * series ids that are filter keys) to link hover and pin across the page;
 * without it the plot keeps its own.
 */
export interface SpectrumSeries {
  /** hover/pin key; a filter key when linked to the page highlight */
  id: string
  wn: ArrayLike<number>
  y: ArrayLike<number>
  color: string
  /** 'trace' = one of many filters, faint; 'line' = a summary or a few named lines */
  role?: 'trace' | 'line'
  label?: string
  /** tooltip lines; default [label ?? id] */
  tip?: string[]
  opacity?: number
  width?: number
  dash?: string
  /** legend key, for `dim` */
  group?: string
  /** traces only: false keeps it out of hover */
  hoverable?: boolean
}
export interface SpectrumBand {
  id: string
  wn: ArrayLike<number>
  lo: ArrayLike<number>
  hi: ArrayLike<number>
  color: string
  opacity?: number
  group?: string
}
/** A labelled span drawn over the regions, e.g. a reference band position (Si-O, SO₄ ν3). */
export interface SpectrumAnnotation { lo: number; hi: number; label: string; fill?: string }
export interface SpectrumHighlight {
  hoverId: string | null
  pinnedId: string | null
  setHover: (id: string | null) => void
  setPinned: (id: string | null) => void
}
type Scales = { x: d3.ScaleLinear<number, number>; y: d3.ScaleContinuousNumeric<number, number>; iw: number; ih: number }

export function SpectrumPlot({
  series = [], bands = [], window: win, height = 300, width: fixedWidth, title,
  yLabel = 'Absorbance', xLabel = 'Wavenumber (cm⁻¹)', yDomain, yClip, includeZero = false, logY = false, zeroLine = false,
  regions = [], annotations = [], markers = [], hover = 'nearest', highlight, lookup, dim, yFormat, readoutFormat, background, children, minWidth,
}: {
  series?: SpectrumSeries[]
  bands?: SpectrumBand[]
  /** visible wavenumbers, [high, low]; default the data's extent */
  window?: [number, number]
  height?: number
  /** a fixed width (an inset beside another chart); default the container's */
  width?: number
  /** HTML heading above the plot, so it wraps on a narrow screen */
  title?: ReactNode
  yLabel?: string
  xLabel?: string
  yDomain?: [number, number]
  /** quantiles for a clipped y range (lib/axes robustDomain); default the full extent */
  yClip?: [number, number]
  includeZero?: boolean
  logY?: boolean
  /** an agreement line at y = 0 (differences, loadings) */
  zeroLine?: boolean
  regions?: readonly Region[]
  annotations?: SpectrumAnnotation[]
  /** labelled lines under the data: 'v' at a wavenumber (a band position), 'h' at a y value (VIP = 1) */
  markers?: { at: number; orient?: 'v' | 'h'; label?: string; color?: string }[]
  hover?: 'nearest' | 'readout' | 'none'
  highlight?: SpectrumHighlight
  /** finds a pinned or hovered filter that is not drawn as a series (pinned elsewhere on the page) */
  lookup?: (id: string) => { wn: ArrayLike<number>; y: ArrayLike<number> } | null
  /** legend dimming: opacity multiplier for a series' group */
  dim?: (group: string) => number
  yFormat?: (v: number) => string
  /** values in the readout tooltip; default 4 significant figures */
  readoutFormat?: (v: number) => string
  /** extra marks under the data, in the plot's coordinates */
  background?: (s: Scales) => ReactNode
  /** extra marks over the data (the escape hatch) */
  children?: (s: Scales) => ReactNode
  /** default 300, or none when `width` is fixed (a facet or an inset) */
  minWidth?: number
}) {
  const ref = useRef<HTMLDivElement>(null)
  const { width: measured } = useDimensions(ref)
  const tip = useTooltip(ref)
  const [localHover, setLocalHover] = useState<string | null>(null)
  const [localPin, setLocalPin] = useState<string | null>(null)
  const [cross, setCross] = useState<number | null>(null)
  const hl: SpectrumHighlight = highlight ?? { hoverId: localHover, pinnedId: localPin, setHover: setLocalHover, setPinned: setLocalPin }

  const m = { top: 10, right: 14, bottom: 46, left: 62 }
  const w = Math.max(minWidth ?? (fixedWidth != null ? 0 : 300), fixedWidth ?? measured)
  const iw = w - m.left - m.right, ih = height - m.top - m.bottom

  const [hiWn, loWn] = useMemo<[number, number]>(() => {
    if (win) return [Math.max(...win), Math.min(...win)]
    let lo = Infinity, hi = -Infinity
    for (const s of [...series, ...bands]) for (let i = 0; i < s.wn.length; i++) { const v = s.wn[i]; if (v < lo) lo = v; if (v > hi) hi = v }
    return Number.isFinite(lo) ? [hi, lo] : [4000, 500]
  }, [win, series, bands])
  const inWin = (v: number) => v <= hiWn && v >= loWn

  const pinned = hl.pinnedId ? series.find((s) => s.id === hl.pinnedId) ?? lookupAs(hl.pinnedId) : null
  const hovered = hl.hoverId && hl.hoverId !== hl.pinnedId ? series.find((s) => s.id === hl.hoverId) ?? lookupAs(hl.hoverId) : null
  function lookupAs(id: string) { const t = lookup?.(id); return t ? { id, ...t } : null }

  const yDom = useMemo<[number, number]>(() => {
    if (yDomain) return yDomain
    const vals: number[] = []
    const take = (wn: ArrayLike<number>, v: ArrayLike<number>, step: number) => {
      for (let i = 0; i < v.length; i += step) if (inWin(wn[i])) { const y = v[i]; if (y != null && Number.isFinite(y) && (!logY || y > 0)) vals.push(y) }
    }
    // many traces: every 4th channel is plenty for a range
    const step = series.filter((s) => s.role === 'trace').length > 50 ? 4 : 1
    for (const s of series) take(s.wn, s.y, s.role === 'trace' ? step : 1)
    for (const b of bands) { take(b.wn, b.lo, 1); take(b.wn, b.hi, 1) }
    if (pinned) take(pinned.wn, pinned.y, 1)
    const [lo, hi] = yClip ?? [0, 1]
    return robustDomain(vals, lo, hi, { include: includeZero ? [0] : [], fallback: logY ? [1e-4, 1] : [0, 1] }).domain
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [yDomain, series, bands, hiWn, loWn, yClip, includeZero, logY, pinned])

  const x = d3.scaleLinear().domain([hiWn, loWn]).range([0, iw])
  const y = (logY ? d3.scaleLog() : d3.scaleLinear()).domain(yDom).range([ih, 0]).nice() as d3.ScaleContinuousNumeric<number, number>
  const line = (wn: ArrayLike<number>, v: ArrayLike<number>) => d3.line<number>()
    .defined((i) => inWin(wn[i]) && v[i] != null && Number.isFinite(v[i]) && (!logY || v[i] > 0))
    .x((i) => x(wn[i])).y((i) => y(v[i]))(d3.range(v.length)) ?? ''
  const area = (b: SpectrumBand) => d3.area<number>()
    .defined((i) => inWin(b.wn[i]) && Number.isFinite(b.lo[i]) && Number.isFinite(b.hi[i]))
    .x((i) => x(b.wn[i])).y0((i) => y(b.lo[i])).y1((i) => y(b.hi[i]))(d3.range(b.lo.length)) ?? ''
  const fmt = yFormat ?? ((v: number) => Math.abs(v) < 1e-3 && v !== 0 ? v.toExponential(0) : String(+v.toPrecision(3)))
  const op = (g?: string) => (g && dim ? dim(g) : 1)

  const traces = series.filter((s) => s.role === 'trace')
  const lines = series.filter((s) => s.role !== 'trace')
  const hoverable = series.filter((s) => s.hoverable !== false)

  const onMove = (e: React.MouseEvent<SVGRectElement>) => {
    const [mx, my] = d3.pointer(e)
    const wn = x.invert(mx)
    if (hover === 'readout') {
      setCross(wn)
      const rf = readoutFormat ?? ((v: number) => (Number.isFinite(v) ? String(+v.toPrecision(4)) : '—'))
      const rows = lines.map((s) => { const i = nearestIndex(s.wn, wn); return `${s.label ?? s.id}: ${i >= 0 ? rf(s.y[i]) : '—'}` })
      tip.show(e, [`${Math.round(wn)} cm⁻¹`, ...rows])
      return
    }
    // nearest trace within 8 px at the pointer's channel; one index per distinct grid
    const idxOf = new Map<ArrayLike<number>, number>()
    let best: SpectrumSeries | null = null, bestD = 8
    for (const s of hoverable) {
      let i = idxOf.get(s.wn)
      if (i == null) { i = nearestIndex(s.wn, wn); idxOf.set(s.wn, i) }
      const v = s.y[i]
      if (v == null || !Number.isFinite(v)) continue
      const d = Math.abs(y(v) - my)
      if (d < bestD) { bestD = d; best = s }
    }
    if (best) { hl.setHover(best.id); tip.show(e, [...(best.tip ?? [best.label ?? best.id]), 'click to pin it']) }
    else { hl.setHover(null); tip.hide() }
  }
  const onLeave = () => { setCross(null); if (hover === 'nearest') hl.setHover(null); tip.hide() }
  const onClick = () => { if (hover === 'nearest') hl.setPinned(hl.hoverId && hl.hoverId !== hl.pinnedId ? hl.hoverId : null) }

  return (
    <div ref={ref} style={{ position: 'relative', width: fixedWidth }}>
      {title && <div className="facet-title" style={{ marginLeft: m.left }}>{title}</div>}
      <svg width={w} height={height} fontFamily={FONT.family}>
        <g transform={`translate(${m.left},${m.top})`}>
          <SpectralRegions x={x} height={ih} regions={regions} />
          {annotations.map((a) => {
            const x0 = x(Math.min(a.hi, hiWn)), x1 = x(Math.max(a.lo, loWn))
            if (x1 <= x0) return null
            return <g key={a.label} pointerEvents="none">
              <rect x={x0} width={x1 - x0} height={ih} fill={a.fill ?? themed('#e8eef6', '#1f2b3a')} />
              <text x={(x0 + x1) / 2} y={11} textAnchor="middle" fontSize={10} fill={INK.muted}>{a.label}</text>
            </g>
          })}
          <YAxis scale={y as d3.ScaleLinear<number, number>} x={0} label={yLabel} tickCount={5} gridWidth={iw} labelX={-50} format={(v) => fmt(+v)} />
          <XAxis scale={x} y={ih} label={xLabel} tickCount={Math.max(3, Math.min(8, Math.floor(iw / 80)))} labelDy={38} />
          {zeroLine && !logY && yDom[0] < 0 && yDom[1] > 0 && <RefLine orient="h" at={y(0)} from={0} to={iw} />}
          {background?.({ x, y, iw, ih })}
          {markers.map((mk, i) => (mk.orient ?? 'v') === 'v'
            ? mk.at <= hiWn && mk.at >= loWn && <Marker key={i} orient="v" at={x(mk.at)} from={0} to={ih} label={mk.label} color={mk.color} labelSide={x(mk.at) > iw * 0.8 ? 'left' : 'right'} />
            : <Marker key={i} orient="h" at={y(mk.at)} from={0} to={iw} label={mk.label} color={mk.color} />)}
          <g pointerEvents="none">
            {bands.map((b) => <path key={b.id} d={area(b)} fill={b.color} fillOpacity={(b.opacity ?? TRACE.bandOpacity) * op(b.group)} />)}
            {traces.map((s) => <path key={s.id} d={line(s.wn, s.y)} fill="none" stroke={s.color}
              strokeOpacity={(s.opacity ?? 0.18) * op(s.group)} strokeWidth={s.width ?? 0.8} strokeDasharray={s.dash} />)}
            {lines.map((s) => <path key={s.id} d={line(s.wn, s.y)} fill="none" stroke={s.color}
              strokeOpacity={(s.opacity ?? 1) * op(s.group)} strokeWidth={s.width ?? 1.6} strokeDasharray={s.dash} />)}
            {children?.({ x, y, iw, ih })}
            {hovered && <path d={line(hovered.wn, hovered.y)} fill="none" {...TRACE.hovered} />}
            {pinned && <path d={line(pinned.wn, pinned.y)} fill="none" {...TRACE.pinned} />}
            {cross != null && <line x1={x(cross)} x2={x(cross)} y1={0} y2={ih} stroke={INK.text} strokeOpacity={0.35} />}
          </g>
          {hover !== 'none' && <rect width={iw} height={ih} fill="transparent" style={{ cursor: hover === 'nearest' && hl.hoverId ? 'pointer' : 'crosshair' }}
            onMouseMove={onMove} onMouseLeave={onLeave} onClick={onClick} />}
        </g>
      </svg>
      {tip.node}
    </div>
  )
}
