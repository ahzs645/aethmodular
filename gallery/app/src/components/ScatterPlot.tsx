import { useId, useMemo, useRef, type ReactNode } from 'react'
import * as d3 from 'd3'
import { AxisSwap, OrientedPlotLayer, useAxisSwap, XAxis, YAxis } from '@/components/Axes'
import { IdentityLine, RefLine } from '@/components/RefLines'
import { RegressionGroupLines, RegressionLines, useRegressionSummary } from '@/components/RegressionOverlay'
import { ScatterLayer } from '@/components/ScatterLayer'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { robustDomain } from '@/lib/axes'
import { useFitHeight } from '@/lib/chartFit'
import { fidProps } from '@/lib/filterEdits'
import { baseFilterId, dotFill, dotRim, useHighlightActions, type FilterRecord } from '@/lib/highlight'
import { useExclMode, useExclusionSplit, ExcludedNote, useMarkReason, type MarkPoint } from '@/lib/markedRows'
import { monthBlocks } from '@/lib/regressionCI'
import { orientErrorRatio, orientPoints, sourceScales } from '@/lib/plotOrientation'
import { groupedRegressions, type FitGrouping } from '@/lib/siteRegression'
import { regression } from '@/lib/stats'
import { FONT, INK, type DotLevel } from '@/lib/theme'

/** How a point is drawn: a dot (default), or a marker shape for a summary point among the dots. */
export type ScatterShape = 'dot' | 'square' | 'triangle' | 'star' | 'cross'

/**
 * One point. With an `id` it is a filter: it follows Filters → excluded, links hover and pin across the gallery,
 * right-clicks to the filter menu, and a click pins it and opens `open` (a page's own drawer), else `rec` (the
 * record drawer). Without an id it only shows its tooltip (a configuration, a site median).
 */
export interface ScatterPoint {
  x: number; y: number; color: string
  /** radius (default 3.5) */
  r?: number
  hollow?: boolean
  shape?: ScatterShape
  /** a ring that encodes something (an outlier, a flag, a stored spectrum), instead of the panel-coloured rim */
  stroke?: string; strokeWidth?: number
  tip?: string[]
  /** React key, when two points share an id (one filter drawn per method) */
  key?: string
  id?: string
  rec?: FilterRecord
  open?: () => void
  /** opacity factor, e.g. a legend's `dim(label)` while another series is pointed at */
  dim?: number
  /** fill opacity instead of the chart's `density` (a summary marker drawn opaque among faint dots) */
  opacity?: number
  /** this point's dot opacity level instead of the chart's `density` */
  level?: DotLevel
  /** a nudge along x in pixels: jitter in a category band, or two series side by side */
  dx?: number
  /** for a fit per site or per season (else the record's site) */
  site?: string; season?: string; seasonColor?: string
}

/** A category axis: one band per entry, label in the gutter; a point's coordinate is the band index plus its place in it (0–1). */
export interface ScatterBand { key: string; label: string; color?: string }

type Scale = d3.ScaleContinuousNumeric<number, number> | d3.ScaleTime<number, number>
/** what the render props receive: the plot's scales and size, in plot pixels */
export interface ScatterScales { x: d3.ScaleContinuousNumeric<number, number>; y: d3.ScaleContinuousNumeric<number, number>; iw: number; ih: number }
type AxisKind = 'linear' | 'log' | 'time'

/** log axes label only the 1-2-5 ticks of each decade, so the minor ticks stay as marks without crowding text */
export const logTickFormat = (v: unknown) => {
  const n = +(v as number), m = Math.round(n / 10 ** Math.floor(Math.log10(n)))
  return [1, 2, 5].includes(m) ? (n >= 1 ? d3.format('~s')(n) : String(+n.toPrecision(1))) : ''
}
const linearFormat = (v: unknown) => String(+(+(v as number)).toPrecision(3))
const timeFormat = (v: unknown) => d3.timeFormat('%b %Y')(v as Date)

/** a marker shape at (cx, cy), radius r */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function Mark({ shape = 'dot', cx, cy, r, ...paint }: { shape?: ScatterShape; cx: number; cy: number; r: number; [attr: string]: any }) {
  if (shape === 'square') return <rect x={cx - r * 0.8} y={cy - r * 0.8} width={r * 1.6} height={r * 1.6} {...paint} />
  if (shape === 'triangle') return <path d={`M${cx},${cy - r} L${cx + r},${cy + r * 0.8} L${cx - r},${cy + r * 0.8} Z`} {...paint} />
  if (shape === 'star') return <path d={d3.symbol(d3.symbolStar, r * r * 3)() ?? ''} transform={`translate(${cx},${cy})`} {...paint} />
  if (shape === 'cross') return <path d={`M${cx - r},${cy - r}L${cx + r},${cy + r}M${cx - r},${cy + r}L${cx + r},${cy - r}`} fill="none" stroke={paint.fill} strokeOpacity={paint.opacity} strokeWidth={3} />
  return <circle cx={cx} cy={cy} r={r} {...paint} />
}

/**
 * The gallery's general x–y scatter (docs/gallery-chart-standards.md §1): two different quantities, each axis linear,
 * log or time, or a category axis of bands (a strip of dots per method or lot). Components/Crossplot stays the 1:1
 * comparison of two measurements of the same thing (one range, the identity line, saved fits); this is everything else.
 *
 *  - points through components/ScatterLayer: the shared hover, pin ring, click to open, right-click filter menu and one
 *    nearest-point pointer layer; filters follow Filters → excluded (hidden, or × under Mark) like the Crossplot's
 *  - ranges: fixed, the points' extent, or clipped to quantiles (lib/axes robustDomain) with `include`; points beyond a
 *    range are pinned to its edge (`outside`)
 *  - square plot area capped at `maxSide` (as every crossplot), or a full-width plot `height` tall; in a Dashboard tile
 *    the fitted height (lib/chartFit)
 *  - `refX` / `refY` agreement lines and `identity` (components/RefLines); `fit` refits the counted points through the
 *    gallery's regression pieces (pooled, per site or per season), summary above the axes
 *  - `pick`: the axis titles open the nearest axis picker (components/AxisChoices)
 *  - `swap`: a curved two-way button in the lower-left margin swaps the axes (components/Axes AxisSwap, lib/plotOrientation;
 *    left out of exports). With `pick` and two editable fields it swaps the fields themselves; otherwise the plot keeps
 *    its own orientation in the URL. Points, ranges, reference lines and the refit (λ inverted) follow the swap; layers
 *    drawn by `under` / `children` are handed the original axes' scales and transposed with the points
 *  - `under` / `children`: render props for the odd extra mark under or over the points (a band, labels, a line)
 */
export function ScatterPlot({
  points, title, xLabel, yLabel, xScale = 'linear', yScale = 'linear', xDomain, yDomain, xClip, yClip, xInclude = [], yInclude = [],
  nice = true, identity = false, outside = 'pin', maxSide = 420, minSide = 200, height, margin, labelX, xTicks, yTicks, xFormat, yFormat,
  xBands, yBands, bandStripes = false, bandLabelSize = 11, refX = [], refY = [], fit, errorRatio, density = 'sparse', pick,
  note = 'hand', onPick, focusR = 6, swap = false, under, children,
}: {
  points: ScatterPoint[]
  /** a title over the plot (12.5/600, in exports), as the Crossplot draws one */
  title?: string
  xLabel?: string
  yLabel?: string
  xScale?: AxisKind
  yScale?: Exclude<AxisKind, 'time'>
  /** a fixed range; else from the counted points (or their quantiles with `xClip`) */
  xDomain?: [number, number]
  yDomain?: [number, number]
  /** quantiles for a clipped range (robustDomain); points beyond it are pinned to the edge */
  xClip?: [number, number]
  yClip?: [number, number]
  /** values a computed range must include (a target, 0) */
  xInclude?: number[]
  yInclude?: number[]
  /** round the ranges to nice ticks (default on) */
  nice?: boolean
  /** a 1:1 line, with both axes on one range */
  identity?: boolean
  /** points beyond a range: pinned to its edge (default), clipped away, or drawn where they fall */
  outside?: 'pin' | 'clip' | 'draw'
  /** square plot area: largest side */
  maxSide?: number
  minSide?: number
  /** a full-width plot area this tall instead of a square */
  height?: number
  margin?: Partial<{ top: number; right: number; bottom: number; left: number }>
  /** the y title's x (default -48) */
  labelX?: number
  xTicks?: number
  yTicks?: number
  xFormat?: (v: any) => string
  yFormat?: (v: any) => string
  /** a category axis: band i spans [i, i + 1]; x bands label the ticks, y bands label the gutter (top band first) */
  xBands?: ScatterBand[]
  yBands?: ScatterBand[]
  /** shade every other y band, across the gutter */
  bandStripes?: boolean
  bandLabelSize?: number
  /** agreement lines (slope 1, intercept 0, a zero difference) at these values */
  refX?: number[]
  refY?: number[]
  /** refit the counted points under the chosen estimators: one pooled line, or one per site or season */
  fit?: FitGrouping | 'none'
  /** Deming error-variance ratio λ for the refit (default 1) */
  errorRatio?: number
  /** dot fill opacity level (lib/theme DOT_OPACITY) */
  density?: DotLevel
  /** axis titles open the nearest axis picker: true for both */
  pick?: boolean | 'x' | 'y'
  /** the excluded-filters note: this chart's own line ('hand'), the gallery's ExcludedNote ('mark'), or none */
  note?: 'hand' | 'mark' | false
  /** what a click does instead of pinning and opening the point's record */
  onPick?: (p: ScatterPoint) => void
  /** the focused mark's smallest radius */
  focusR?: number
  /** offer the axis swap: true keys the orientation by the axis titles; a string names it (panels that swap together share one) */
  swap?: boolean | string
  under?: (s: ScatterScales) => ReactNode
  children?: (s: ScatterScales) => ReactNode
}) {
  // the axis swap: everything below draws the swapped plot, and hands its render props the original axes' scales
  const canSwap = !!swap && !xBands && !yBands && xScale !== 'time'
  const orientation = useAxisSwap(typeof swap === 'string' ? swap : `scatter.${xLabel ?? ''}.${yLabel ?? ''}`, pick === true)
  const swapped = canSwap && orientation.swapped
  const oriented = useMemo(() => orientPoints(points, swapped), [points, swapped])
  // a click hands `onPick` the caller's own point, not its swapped copy
  const inputPoints = points
  const original = useMemo(() => (swapped ? new Map(oriented.map((p, i) => [p, inputPoints[i]])) : null), [oriented, swapped]) // eslint-disable-line react-hooks/exhaustive-deps
  if (swapped) {
    points = oriented
    ;[xLabel, yLabel] = [yLabel, xLabel]
    ;[xScale, yScale] = [yScale, xScale as Exclude<AxisKind, 'time'>]
    ;[xDomain, yDomain] = [yDomain, xDomain]
    ;[xClip, yClip] = [yClip, xClip]
    ;[xInclude, yInclude] = [yInclude, xInclude]
    ;[xTicks, yTicks] = [yTicks, xTicks]
    ;[xFormat, yFormat] = [yFormat, xFormat]
    ;[refX, refY] = [refY, refX]
    if (pick === 'x' || pick === 'y') pick = pick === 'x' ? 'y' : 'x'
  }
  const ref = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(ref)
  const tip = useTooltip(ref)
  const hl = useHighlightActions()
  const fitHeight = useFitHeight()
  const clipId = useId()
  const m = { top: 14, right: 18, bottom: 46, left: 62, ...margin }
  const titleH = title ? 14 : 0
  const xLog = xScale === 'log', yLog = yScale === 'log'
  // every point that can be drawn; Filters → excluded takes filters out of the points, the ranges and any fit
  const drawable = useMemo(() => points.filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y) && (!xLog || p.x > 0) && (!yLog || p.y > 0)), [points, xLog, yLog])
  const split = useExclusionSplit(drawable, (p) => p.id)
  const ok = split.counted, nOut = split.nOut
  const markMode = useExclMode() === 'mark'
  const reasonOf = useMarkReason()

  // ranges: fixed, clipped, or the counted points' extent; bands fix the range to their count
  const range = (vals: number[], fixed: [number, number] | undefined, clip: [number, number] | undefined, include: number[], log: boolean): [number, number] => {
    if (fixed) return fixed
    const r = robustDomain(vals, clip?.[0] ?? 0, clip?.[1] ?? 1, { include, fallback: log ? [0.1, 1] : [0, 1] }).domain
    return r[0] === r[1] ? (log ? [r[0] / 2, r[0] * 2] : [r[0] - 1, r[1] + 1]) : r
  }
  let xd = xBands ? [0, xBands.length] as [number, number] : range(ok.map((p) => p.x), xDomain, xClip, xInclude, xLog)
  let yd = yBands ? [yBands.length, 0] as [number, number] : range(ok.map((p) => p.y), yDomain, yClip, yInclude, yLog)
  if (identity) {
    const lo = Math.min(xd[0], yd[0]), hi = Math.max(xd[1], yd[1])
    xd = [xLog ? Math.max(lo, 1e-3) : Math.min(0, lo), hi]
    yd = [yLog ? Math.max(lo, 1e-3) : Math.min(0, lo), hi]
  }

  // the plot area: a square (capped at maxSide), or full width and `height` tall; a tile's fitted height wins
  const avail = (width || maxSide + m.left + m.right) - m.left - m.right
  const fitInner = fitHeight == null ? null : fitHeight - m.top - m.bottom - titleH
  const square = height == null
  const side = Math.max(minSide, Math.min(avail, maxSide, fitInner ?? Infinity))
  const iw = square ? side : Math.max(minSide, avail)
  const ih0 = square ? side : Math.max(80, fitInner ?? height!)

  // the same scales until the ranges change, so the dots and the pointer index survive a re-render (a tooltip)
  const x = useMemo(() => {
    const s: Scale = xScale === 'time' ? d3.scaleTime().domain(xd).range([0, iw]) : (xLog ? d3.scaleLog() : d3.scaleLinear()).domain(xd).range([0, iw])
    if (nice && !xBands) (s as d3.ScaleLinear<number, number>).nice()
    return s as d3.ScaleContinuousNumeric<number, number>
  }, [xd.join(), iw, xScale, nice, !!xBands]) // eslint-disable-line react-hooks/exhaustive-deps
  const fitting = !!fit && fit !== 'none' && !xLog && !yLog && xScale !== 'time'

  // the fit, on the counted points; one per site where they come from several sites, per season where they say theirs
  const lambda = orientErrorRatio(errorRatio, swapped)
  const siteOf = (p: ScatterPoint) => p.site ?? p.rec?.site ?? 'Unknown site'
  const fitGroups = useMemo(() => (fitting && fit !== 'pooled' ? groupedRegressions(ok, fit as FitGrouping, {
    x: (p) => p.x, y: (p) => p.y, site: siteOf, color: (p) => p.color,
    season: (p) => p.season ?? 'Unknown season', seasonColor: (p) => p.seasonColor ?? INK.muted,
    options: (members) => ({ errorsInVariables: true, lambda, bootstrapGroups: monthBlocks(members.map((p) => p.rec ?? {})) }),
  }) : []), [ok, fit, lambda, fitting]) // eslint-disable-line react-hooks/exhaustive-deps
  const pooled = useMemo(() => (fitting && fit === 'pooled' ? regression(ok.map((p) => p.x), ok.map((p) => p.y),
    { errorsInVariables: true, lambda, bootstrapGroups: monthBlocks(ok.map((p) => p.rec ?? {})) }) : null), [ok, fit, lambda, fitting])
  const w = iw + m.left + m.right
  const summary = useRegressionSummary({ stats: pooled, groups: fit === 'pooled' ? undefined : fitGroups, width: w - 24, maxWidth: square ? (width || w) - 24 : undefined, enabled: fitting })
  // a square plot's fit boxes may run wider than it (one row across the frame): the SVG takes their width, the plot stays centred
  const svgW = Math.max(w, summary.width + 24), dx = (svgW - w) / 2
  const top = m.top + titleH + (summary.height ? summary.height + 14 : 0)
  const ih = ih0
  const y = useMemo(() => {
    const s = (yLog ? d3.scaleLog() : d3.scaleLinear()).domain(yd).range([ih, 0])
    if (nice && !yBands) s.nice()
    return s as d3.ScaleContinuousNumeric<number, number>
  }, [yd.join(), ih, yLog, nice, !!yBands]) // eslint-disable-line react-hooks/exhaustive-deps

  const keyOf = (p: ScatterPoint) => (p.id ? baseFilterId(p.id) : null)
  // pinned at the edge: a point beyond a range sits on its axis line instead of leaving the plot
  const at = useMemo(() => {
    const [x0, x1] = x.range(), [y1, y0] = y.range()
    const pin = outside === 'pin'
    return (p: ScatterPoint) => {
      const cx = x(p.x) + (p.dx ?? 0), cy = y(p.y)
      return [pin ? Math.max(Math.min(x0, x1), Math.min(Math.max(x0, x1), cx)) : cx, pin ? Math.max(Math.min(y0, y1), Math.min(Math.max(y0, y1), cy)) : cy] as const
    }
  }, [x, y, outside])
  const xy = useMemo(() => ok.map(at), [ok, at])
  // under Mark, the excluded points as × inside the axes (components/ScatterLayer `marks`), never in the points
  const marks = useMemo<MarkPoint[]>(() => {
    if (!markMode || !split.marked.length) return []
    const [xa, xb] = x.domain().map(Number), [ya, yb] = y.domain()
    const inside = (v: number, a: number, b: number) => v >= Math.min(a, b) && v <= Math.max(a, b)
    return split.marked.flatMap((p) => !p.id || !inside(p.x, xa, xb) || !inside(p.y, ya, yb) ? []
      : [{ id: baseFilterId(p.id), at: [x(p.x), y(p.y)] as const, tip: [...(p.tip ?? [p.id]), `excluded: ${reasonOf(p.id) ?? 'by hand'}`, 'drawn only: not in any fit, R² or count'] }])
  }, [markMode, split.marked, x, y, reasonOf])
  const dots = useMemo(() => ok.map((p, i) => {
    const key = keyOf(p), hollow = p.hollow && (p.shape ?? 'dot') === 'dot'
    const fill = p.opacity != null ? { fillOpacity: p.opacity } : dotFill(p.level ?? density)
    return <Mark key={p.key ?? i} shape={p.shape} cx={xy[i][0]} cy={xy[i][1]} r={p.r ?? 3.5} fill={hollow ? 'none' : p.color} opacity={p.dim}
      {...fill} stroke={p.stroke ?? (hollow ? p.color : dotRim().stroke)} strokeWidth={p.strokeWidth ?? (hollow ? 1.5 : dotRim().strokeWidth)}
      {...(key ? fidProps(key) : {})} />
  }), [ok, xy, density])

  const scales: ScatterScales = { x, y, iw, ih }
  // swapped: the render props draw in the original axes, and OrientedPlotLayer transposes what they draw
  const source = useMemo(() => (swapped ? sourceScales(x, y, true) : null), [x, y, swapped])
  const layered = (render: ((s: ScatterScales) => ReactNode) | undefined) => !render ? null
    : source ? <OrientedPlotLayer swapped width={iw} height={ih}>{render({ ...source, iw, ih })}</OrientedPlotLayer> : render(scales)
  const fmt = (kind: AxisKind) => (kind === 'log' ? logTickFormat : kind === 'time' ? timeFormat : linearFormat)
  // a band axis shows its labels at the band centres (a band scale over the same pixels)
  const xBandScale = useMemo(() => (xBands ? d3.scaleBand().domain(xBands.map((b) => b.key)).range([0, iw]) : null), [xBands, iw])
  const bandLabel = useMemo(() => new Map((xBands ?? []).map((b) => [b.key, b.label])), [xBands])
  const pickX = pick === true || pick === 'x' ? 'x' as const : undefined
  const pickY = pick === true || pick === 'y' ? 'y' as const : undefined
  const outNote = note !== 'hand' || nOut <= 0 ? null
    : `${nOut} excluded filter${nOut === 1 ? '' : 's'} (Filters → excluded) ${nOut === 1 ? 'is' : 'are'} ${markMode ? 'drawn as × where inside the axes' : 'hidden here'}.`
  const layer = (
    <ScatterLayer items={ok} xy={xy} idOf={keyOf} dots={dots} width={iw} height={ih} tip={tip} reach={(p) => p.r ?? 3.5} marks={marks}
      focusR={(p) => Math.max(p.r ?? 3.5, focusR)}
      focus={(p, [cx, cy]) => <Mark shape={p.shape} cx={cx} cy={cy} r={Math.max(p.r ?? 3.5, focusR)} fill={p.hollow ? 'none' : p.color} opacity={p.dim}
        {...dotFill('sparse')} stroke={INK.text} strokeWidth={2} />}
      tipOf={(p) => (p.tip ? (p.rec || p.open ? [...p.tip, p.id ? 'click to pin it on every chart and open its record' : 'click to open its record'] : p.tip) : null)}
      pickable={(p) => !!(onPick || p.open || p.rec || keyOf(p))}
      onPick={(p) => {
        if (onPick) { onPick(original?.get(p) ?? p); return }
        const key = keyOf(p)
        if (p.open) p.open(); else if (p.rec) hl.openRecord(p.rec)
        if (key) hl.setPinned(key)
      }}>
      {fitting && (fit === 'pooled' ? <RegressionLines stats={pooled} x={x} y={y} domain={x.domain()} width={iw} height={ih} />
        : <RegressionGroupLines groups={fitGroups} x={x} y={y} width={iw} height={ih} />)}
      {layered(children)}
    </ScatterLayer>
  )
  return (
    <div ref={ref} style={{ position: 'relative' }}>
      <svg width={svgW} height={top + ih + m.bottom} fontFamily={FONT.family} style={{ display: 'block', margin: '0 auto' }}>
        {title && <text x={m.left + dx} y={14} fontSize={12.5} fontWeight={600} fill={INK.text}>{title}</text>}
        {fitting && <g transform={`translate(12,${m.top + titleH + 8})`}>{summary.node}</g>}
        <g transform={`translate(${m.left + dx},${top})`}>
          {yBands?.map((b, i) => <g key={b.key}>
            {bandStripes && i % 2 === 0 && <rect x={-m.left + 4} y={y(i)} width={iw + m.left - 4} height={y(i + 1) - y(i)} fill={INK.page} />}
            <text x={-10} y={(y(i) + y(i + 1)) / 2} dy="0.32em" textAnchor="end" fontSize={bandLabelSize} fontWeight={bandStripes ? 600 : 400} fill={b.color ?? INK.text}>{b.label}</text>
          </g>)}
          {!yBands && <YAxis scale={y as d3.ScaleLinear<number, number>} x={0} label={yLabel} tickCount={yTicks ?? 5} gridWidth={iw} labelX={labelX ?? -48} format={yFormat ?? fmt(yScale)} pick={pickY} />}
          {xBandScale ? <XAxis scale={xBandScale} y={ih} label={xLabel} labelDy={38} format={(k) => bandLabel.get(k) ?? k} pick={pickX} />
            : <XAxis scale={x as d3.ScaleLinear<number, number>} y={ih} label={xLabel} tickCount={xTicks ?? Math.max(3, Math.floor(iw / 80))} labelDy={38} format={xFormat ?? fmt(xScale)} pick={pickX} />}
          {refX.map((v) => <RefLine key={`rx${v}`} orient="v" at={x(v)} from={0} to={ih} />)}
          {refY.map((v) => <RefLine key={`ry${v}`} orient="h" at={y(v)} from={0} to={iw} />)}
          {identity && (() => { const [a, b] = x.domain(); return <IdentityLine x1={x(a)} y1={y(a)} x2={x(b)} y2={y(b)} /> })()}
          {canSwap && <AxisSwap x={-m.left + 24} y={ih + 26} onSwap={orientation.swap} />}
          {layered(under)}
          {outside === 'clip' ? <>
            <defs><clipPath id={clipId}><rect width={iw} height={ih} /></clipPath></defs>
            <g clipPath={`url(#${clipId})`}>{layer}</g>
          </> : layer}
        </g>
      </svg>
      {tip.node}
      {outNote && <p className="chart-note">{outNote}</p>}
      {note === 'mark' && <ExcludedNote count={nOut} drawn={markMode} />}
    </div>
  )
}
