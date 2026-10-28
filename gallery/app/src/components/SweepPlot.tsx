import { useRef, useState, type ReactNode } from 'react'
import * as d3 from 'd3'
import { XAxis, YAxis } from '@/components/Axes'
import { Marker, RefLine } from '@/components/RefLines'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { robustDomain } from '@/lib/axes'
import { FONT, INK } from '@/lib/theme'

/**
 * The gallery's one "value against a setting" chart: a metric against PLS k,
 * a cutoff, a rank, a stage or a date, one line per series, with optional
 * intervals (a band, or whiskers on the points), the agreement value, labelled
 * markers and the chosen setting ringed (docs/gallery-chart-standards.md §2).
 *
 * x may be numbers (linear or log), dates (time) or category names (point,
 * in the order first seen). With `yClip`, points outside the clipped range are
 * drawn hollow at the edge and left out of the line, and the count is noted.
 * Hover names the nearest point (within 12 px); `onPoint` makes points clickable.
 */
/** log axes label only 1, 2 and 5 × 10ⁿ (and only 1 × 10ⁿ over many decades); the other ticks stay unlabelled */
function logLabel(x: d3.ScaleLogarithmic<number, number>) {
  const [a, b] = x.domain(), decades = Math.abs(Math.log10(b) - Math.log10(a))
  return (v: unknown) => {
    const n = +(v as number), m = n / 10 ** Math.floor(Math.log10(n) + 1e-9)
    const keep = decades > 3 ? Math.abs(m - 1) < 1e-6 : [1, 2, 5].some((t) => Math.abs(m - t) < 1e-6)
    return keep ? String(+n.toPrecision(2)) : ''
  }
}

export type SweepX = number | Date | string
export interface SweepPoint { x: SweepX; y: number | null | undefined; lo?: number | null; hi?: number | null; tip?: string[]; id?: string }
export interface SweepSeries {
  id: string
  label: string
  color: string
  points: SweepPoint[]
  dash?: string
  /** draw a dot at each point; default when a series has ≤ 60 points */
  marks?: boolean
  /** draw lo/hi as a shaded band (default) or as whiskers on the points */
  interval?: 'band' | 'whisker' | 'none'
  width?: number
  /** dot radius (default 3) */
  r?: number
  /** 'step' steps at each point; 'mid-step' steps halfway between points (bins given at their centres, see histogramSeries) */
  curve?: 'linear' | 'monotone' | 'step' | 'mid-step'
  /** a missing y breaks the line and band instead of joining its neighbours (a daily record with gaps) */
  gaps?: boolean
  /** opacity of the lo/hi band (default 0.15) */
  fill?: number
  /** legend key, for `dim` */
  group?: string
}

/**
 * A histogram as a SweepPlot series: filled bars as one step outline, counts (or the share of `total`) per bin.
 * `edges` are the bin edges (counts.length + 1); `tip` names a bin. Each bin is a point at its centre, so hover
 * finds bins, plus a point at either outer edge so the outline closes on the axis range.
 */
export function histogramSeries({ id, label, color, edges, counts, total, fill = 0.45, tip, ...rest }: {
  id: string; label: string; color: string; edges: number[]; counts: number[]
  /** divide each count by this (a share); else counts */
  total?: number
  /** band opacity: 0 for an outline only */
  fill?: number
  tip?: (i: number, value: number) => string[]
} & Partial<Omit<SweepSeries, 'id' | 'label' | 'color' | 'points'>>): SweepSeries {
  const v = (c: number) => (total ? c / Math.max(1, total) : c)
  const n = counts.length
  const at = (i: number) => ({ y: v(counts[i]), lo: fill ? 0 : null, hi: fill ? v(counts[i]) : null, tip: tip?.(i, v(counts[i])) })
  const points: SweepPoint[] = n ? [{ x: edges[0], ...at(0) }, ...counts.map((_, i) => ({ x: (edges[i] + edges[i + 1]) / 2, ...at(i) })), { x: edges[n], ...at(n - 1) }] : []
  return { id, label, color, points, curve: 'mid-step', marks: false, width: 1.4, fill, interval: fill ? 'band' : 'none', ...rest }
}
export interface SweepMarker { at: number | Date; label?: string; color?: string; labelSide?: 'left' | 'right' }
type XScale = d3.ScaleLinear<number, number> | d3.ScaleLogarithmic<number, number> | d3.ScaleTime<number, number> | d3.ScalePoint<string>

export function SweepPlot({
  series, xScale: kind, xDomain, xLabel, yLabel, yScale = 'linear', yDomain, yClip, include = [], reference,
  xMarkers = [], yMarkers = [], chosen = [], dividers = [], onPoint, height = 240, width: fixedWidth, title,
  dim, xFormat, yFormat, background, children, minWidth, pick,
}: {
  series: SweepSeries[]
  xScale?: 'linear' | 'log' | 'time' | 'point'
  xDomain?: [number, number] | [Date, Date]
  xLabel?: string
  yLabel: string
  /** axis titles that open the nearest axis picker (components/AxisChoices): 'x', 'y' or both */
  pick?: 'x' | 'y' | 'both'
  yScale?: 'linear' | 'log'
  yDomain?: [number, number]
  /** quantiles for a clipped y range; outside points go hollow at the edge */
  yClip?: [number, number]
  /** values the y range must include (a reference, a floor) */
  include?: number[]
  /** the agreement value (1 for a slope, 0 for an intercept), drawn as a reference line */
  reference?: number
  xMarkers?: SweepMarker[]
  yMarkers?: { at: number; label?: string; color?: string }[]
  /** the chosen setting of a series, ringed */
  chosen?: { series: string; x: SweepX }[]
  /** point scale: a dotted divider after these categories, e.g. between IMPROVE and Addis stages */
  dividers?: string[]
  onPoint?: (s: SweepSeries, p: SweepPoint) => void
  height?: number
  width?: number
  title?: ReactNode
  dim?: (group: string) => number
  xFormat?: (v: any) => string
  yFormat?: (v: number) => string
  /** extra marks under the data, e.g. a guardrail box */
  background?: (s: { x: XScale; y: d3.ScaleContinuousNumeric<number, number>; iw: number; ih: number }) => ReactNode
  /** extra marks over the data (the escape hatch) */
  children?: (s: { x: XScale; y: d3.ScaleContinuousNumeric<number, number>; iw: number; ih: number }) => ReactNode
  /** default 260, or none when `width` is fixed (a facet) */
  minWidth?: number
}) {
  const ref = useRef<HTMLDivElement>(null)
  const { width: measured } = useDimensions(ref)
  const tip = useTooltip(ref)
  const [hover, setHover] = useState<{ s: string; i: number } | null>(null)
  const m = { top: 12, right: 16, bottom: 46, left: 62 }
  const w = Math.max(minWidth ?? (fixedWidth != null ? 0 : 260), fixedWidth ?? measured)
  const iw = w - m.left - m.right, ih = height - m.top - m.bottom

  const all = series.flatMap((s) => s.points)
  const first = all.find((p) => p.x != null)?.x
  const k = kind ?? (first instanceof Date ? 'time' : typeof first === 'string' ? 'point' : 'linear')
  let x: XScale
  if (k === 'point') x = d3.scalePoint<string>().domain([...new Set(all.map((p) => String(p.x)))]).range([0, iw]).padding(0.4)
  else if (k === 'time') x = d3.scaleTime().domain((xDomain as [Date, Date]) ?? (d3.extent(all, (p) => p.x as Date) as [Date, Date])).range([0, iw])
  else {
    const dom = (xDomain as [number, number]) ?? (d3.extent(all.filter((p) => k !== 'log' || (p.x as number) > 0), (p) => p.x as number) as [number, number])
    x = (k === 'log' ? d3.scaleLog() : d3.scaleLinear()).domain(dom[0] == null ? [0, 1] : dom).range([0, iw])
    if (k === 'linear' && !xDomain) (x as d3.ScaleLinear<number, number>).nice()
  }
  const px = (v: SweepX) => k === 'point' ? (x as d3.ScalePoint<string>)(String(v)) ?? 0 : (x as d3.ScaleLinear<number, number>)(v as number)

  const ys: number[] = []
  for (const p of all) for (const v of [p.y, p.lo, p.hi]) if (v != null && Number.isFinite(v) && (yScale !== 'log' || v > 0)) ys.push(v)
  const inc = [...include, ...(reference != null ? [reference] : []), ...yMarkers.map((mk) => mk.at)]
  const rd = yDomain ? { domain: yDomain, below: 0, above: 0 } : robustDomain(ys, yClip?.[0] ?? 0, yClip?.[1] ?? 1, { include: inc, fallback: yScale === 'log' ? [0.1, 1] : [0, 1] })
  const y = (yScale === 'log' ? d3.scaleLog() : d3.scaleLinear()).domain(rd.domain).range([ih, 0]) as d3.ScaleContinuousNumeric<number, number>
  if (!yDomain) y.nice()
  const [y0, y1] = y.domain()
  const clamped = (v: number) => v < Math.min(y0, y1) || v > Math.max(y0, y1)
  const py = (v: number) => y(Math.max(Math.min(y0, y1), Math.min(Math.max(y0, y1), v)))
  const curve = (c?: SweepSeries['curve']) => (c === 'monotone' ? d3.curveMonotoneX : c === 'step' ? d3.curveStepAfter : c === 'mid-step' ? d3.curveStep : d3.curveLinear)
  const ok = (v: number | null | undefined): v is number => v != null && Number.isFinite(v) && (yScale !== 'log' || v > 0)
  const nClamped = all.filter((p) => ok(p.y) && clamped(p.y)).length
  const fmtY = yFormat ?? ((v: number) => String(+v.toPrecision(3)))
  const op = (g?: string) => (g && dim ? dim(g) : 1)
  const isChosen = (s: SweepSeries, p: SweepPoint) => chosen.some((c) => c.series === s.id && String(c.x) === String(p.x))

  const onMove = (e: React.MouseEvent<SVGRectElement>) => {
    const [mx, my] = d3.pointer(e)
    let best: { s: number; i: number } | null = null, bestD = 12
    series.forEach((s, si) => s.points.forEach((p, i) => {
      if (!ok(p.y)) return
      const d = Math.hypot(px(p.x) - mx, py(p.y) - my)
      if (d < bestD) { bestD = d; best = { s: si, i } }
    }))
    if (!best) { setHover(null); tip.hide(); return }
    const { s: si, i } = best as { s: number; i: number }
    const s = series[si], p = s.points[i]
    setHover({ s: s.id, i })
    const xs = p.x instanceof Date ? d3.timeFormat('%Y-%m-%d')(p.x) : xFormat ? xFormat(p.x) : String(p.x)
    tip.show(e, p.tip ?? [s.label, `${xLabel || 'x'}: ${xs}`, `${yLabel}: ${fmtY(p.y as number)}${p.lo != null && p.hi != null ? ` (${fmtY(p.lo)} to ${fmtY(p.hi)})` : ''}`, ...(onPoint ? ['click to select'] : [])])
  }
  const hp = hover ? series.find((s) => s.id === hover.s)?.points[hover.i] : null

  return (
    <div ref={ref} style={{ position: 'relative', width: fixedWidth }}>
      {title && <div className="facet-title" style={{ marginLeft: m.left }}>{title}</div>}
      <svg width={w} height={height} fontFamily={FONT.family}>
        <g transform={`translate(${m.left},${m.top})`}>
          <YAxis scale={y as d3.ScaleLinear<number, number>} x={0} label={yLabel} tickCount={5} gridWidth={iw} labelX={-50} format={(v) => fmtY(+v)} pick={pick === 'y' || pick === 'both' ? 'y' : undefined} />
          <XAxis scale={x as d3.ScaleLinear<number, number>} y={ih} label={xLabel} tickCount={Math.max(3, Math.min(8, Math.floor(iw / 70)))} labelDy={38} pick={pick === 'x' || pick === 'both' ? 'x' : undefined}
            format={xFormat ?? (k === 'time' ? (v) => d3.timeFormat('%b %Y')(v as Date) : k === 'log' ? logLabel(x as d3.ScaleLogarithmic<number, number>) : undefined)} />
          {reference != null && !clamped(reference) && <RefLine orient="h" at={y(reference)} from={0} to={iw} />}
          {dividers.map((after) => {
            const dom = (x as d3.ScalePoint<string>).domain(), i = dom.indexOf(after)
            if (k !== 'point' || i < 0 || i >= dom.length - 1) return null
            const at = (px(dom[i]) + px(dom[i + 1])) / 2
            return <line key={after} x1={at} x2={at} y1={0} y2={ih} stroke={INK.axis} strokeDasharray="1 3" />
          })}
          {xMarkers.map((mk, i) => { const at = px(mk.at as number)
            return <Marker key={`x${i}`} orient="v" at={at} from={0} to={ih} label={mk.label} color={mk.color} labelSide={mk.labelSide ?? (at > iw * 0.7 ? 'left' : 'right')} labelDy={i * 12} /> })}
          {background?.({ x, y, iw, ih })}
          {yMarkers.map((mk, i) => !clamped(mk.at) && <Marker key={`y${i}`} orient="h" at={y(mk.at)} from={0} to={iw} label={mk.label} color={mk.color} />)}
          <g pointerEvents="none">
            {series.map((s) => {
              // lines run in x order whatever order the points came in (categories keep theirs)
              // with gaps, a missing or clamped y breaks the line where it falls; else the line joins the drawable points
              const drawn = (p: SweepPoint) => ok(p.y) && !clamped(p.y as number)
              const pts = s.gaps ? [...s.points] : s.points.filter(drawn)
              if (k !== 'point') pts.sort((a, b) => +(a.x as number) - +(b.x as number))
              const band = (s.interval ?? 'band') === 'band' && pts.some((p) => ok(p.lo) && ok(p.hi))
              const marks = s.marks ?? s.points.length <= 60
              return <g key={s.id} opacity={op(s.group)}>
                {band && <path fill={s.color} fillOpacity={s.fill ?? 0.15} d={d3.area<SweepPoint>().defined((p) => ok(p.lo) && ok(p.hi)).curve(curve(s.curve))
                  .x((p) => px(p.x)).y0((p) => py(p.lo as number)).y1((p) => py(p.hi as number))(pts) ?? ''} />}
                <path fill="none" stroke={s.color} strokeWidth={s.width ?? 1.8} strokeDasharray={s.dash}
                  d={d3.line<SweepPoint>().defined(drawn).curve(curve(s.curve)).x((p) => px(p.x)).y((p) => y(p.y as number))(pts) ?? ''} />
                {s.interval === 'whisker' && s.points.filter((p) => ok(p.lo) && ok(p.hi)).map((p, i) =>
                  <line key={`w${i}`} x1={px(p.x)} x2={px(p.x)} y1={py(p.lo as number)} y2={py(p.hi as number)} stroke={s.color} strokeWidth={1.2} strokeOpacity={0.7} />)}
                {s.points.filter((p) => ok(p.y)).map((p, i) => {
                  const out = clamped(p.y as number), ch = isChosen(s, p)
                  if (!marks && !out && !ch) return null
                  return <circle key={i} cx={px(p.x)} cy={py(p.y as number)} r={ch ? (s.r ?? 3) + 2 : s.r ?? 3}
                    fill={out || ch ? INK.panel : s.color} stroke={s.color} strokeWidth={out || ch ? 1.6 : 0} />
                })}
              </g>
            })}
            {children?.({ x, y, iw, ih })}
            {hp && ok(hp.y) && <circle cx={px(hp.x)} cy={py(hp.y)} r={6} fill="none" stroke={INK.text} strokeWidth={1.4} />}
            {nClamped > 0 && <text x={iw} y={-2} textAnchor="end" fontSize={10} fill={INK.muted}>{nClamped} point{nClamped > 1 ? 's' : ''} beyond the range, drawn hollow at the edge</text>}
          </g>
          <rect width={iw} height={ih} fill="transparent" style={{ cursor: onPoint && hover ? 'pointer' : 'default' }}
            onMouseMove={onMove} onMouseLeave={() => { setHover(null); tip.hide() }}
            onClick={() => { if (onPoint && hover) { const s = series.find((q) => q.id === hover.s); if (s) onPoint(s, s.points[hover.i]) } }} />
        </g>
      </svg>
      {tip.node}
    </div>
  )
}
