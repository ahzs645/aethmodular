/**
 * A binned 2-D density from an exporter (cells of a bins × bins grid, each with a count and optionally a value),
 * with points drawn over it: the IMPROVE network as a cloud, a few hundred SPARTAN filters on top. For data too
 * large to ship row by row (research/improve_envelope).
 *
 * Built from the gallery's shared pieces, as the other crossplots are (docs/gallery-chart-standards.md): a square
 * plot centred in its `.chart-wrap`, sized by `useFitHeight` in a dashboard tile; the points through ScatterLayer
 * (one pointer layer, the gallery hover, click to open a filter's record, right-click for its menu, excluded filters
 * as ×); the cell under the pointer answered through the same layer's `onMiss`; one `Legend` row under the plot
 * holding the point keys, the window and the ramp (which is also the palette picker).
 */
import { useMemo, useRef, type ReactNode } from 'react'
import { useColourScheme } from '@/lib/colourSchemes'
import * as d3 from 'd3'
import { AxisSwap, OrientedPlotLayer, useAxisSwap, XAxis, YAxis } from '@/components/Axes'
import { sourceScales } from '@/lib/plotOrientation'
import { ColorLegend } from '@/components/ColorLegend'
import { Legend, type LegendItem } from '@/components/Legend'
import { ScatterLayer } from '@/components/ScatterLayer'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { fitInner, useFitHeight } from '@/lib/chartFit'
import { fidProps } from '@/lib/filterEdits'
import { baseFilterId } from '@/lib/filterId'
import { dotFill, dotRim } from '@/lib/highlight'
import type { MarkPoint } from '@/lib/markedRows'
import { fmt } from '@/lib/format'
import { FONT, INK } from '@/lib/theme'

/** a point over the cloud; with an `id` it is a gallery filter (hover, pin, record, filter menu) */
export interface DensityPoint { x: number; y: number; color: string; hollow?: boolean; tip: string[]; id?: string }
/** an excluded filter under Filters → excluded: Mark, in data coordinates; drawn as × (components/ScatterLayer) */
export interface DensityMark { id: string; x: number; y: number; tip: string[] }

const M = { top: 12, right: 14, bottom: 46, left: 60 }
/** the legend row under the plot, for the fitted height in a tile */
const LEGEND_H = 44

/** a count from the log ramp's tick: 1, 3.2, 10, 100, 1k, 10k */
const countTick = (v: number) => { const c = 10 ** v; return c >= 1000 ? d3.format('.2~s')(c) : d3.format('.2~r')(c) }

export function DensityPlot({ id, cells, bins, x: [x0, x1], y: [y0, y1], xLabel, yLabel, mode = 'count', valueLabel = 'filters per cell', cloudLabel,
  points = [], marks = [], legend = [], box, maxSide = 520, children }: {
  /** the chart's id: its palette is kept in the link under it (lib/colourSchemes) */
  id: string
  /** [i, j, count] or [i, j, count, value] */
  cells: number[][]
  bins: number
  x: [number, number]
  y: [number, number]
  xLabel: string
  yLabel: string
  /** colour by log count, or by the cell's value (e.g. median fAbs) */
  mode?: 'count' | 'value'
  /** what a cell's value is (mode 'value'), in its tooltip */
  valueLabel?: string
  /** the ramp's label: what the binned cloud is, e.g. "IMPROVE filters per cell" (default: "filters per cell", or valueLabel) */
  cloudLabel?: string
  points?: DensityPoint[]
  marks?: DensityMark[]
  /** the keys for the points and any child overlay, ahead of the window and the ramp; `parent` groups them */
  legend?: LegendItem[]
  /** a dashed window, e.g. the Addis p05–p95; `key` names it in the legend (default: its label) */
  box?: { x: [number, number]; y: [number, number]; label: string; key?: string }
  /** the square's largest side, px (the gallery's crossplots stop at 560) */
  maxSide?: number
  children?: (s: { x: d3.ScaleLinear<number, number>; y: d3.ScaleLinear<number, number> }) => ReactNode
}) {
  const ref = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(ref)
  const tip = useTooltip(ref)
  const fit = useFitHeight()
  const { swapped, swap } = useAxisSwap(`density.${id}`)
  const side = Math.max(160, Math.min(maxSide, width - M.left - M.right, fitInner(fit, M.top + M.bottom + LEGEND_H, 160) ?? Infinity))
  const xs = useMemo(() => d3.scaleLinear().domain([x0, x1]).range([0, side]), [x0, x1, side])
  const ys = useMemo(() => d3.scaleLinear().domain([y0, y1]).range([side, 0]), [y0, y1, side])
  const display = sourceScales(xs, ys, swapped)
  const cw = side / bins
  const colours = useColourScheme(id, 'sequential')
  // the Gallery ramp's palest step is the page itself: start one step in, as before
  const stops = colours.name === 'Gallery' && !colours.reverse ? colours.stops.slice(1) : colours.stops
  const valueOf = (c: number[]) => (mode === 'count' ? Math.log10(c[2]) : c[3])
  const vals = cells.map(valueOf).filter(Number.isFinite)
  const ramp = d3.scaleQuantize<string>().domain([d3.min(vals) ?? 0, d3.max(vals) ?? 1]).range(stops)
  // the cells under the points, drawn once per size and palette; a hover redraws none of them
  const cellRects = useMemo(() => cells.map((c) => {
    const v = valueOf(c)
    return <rect key={`${c[0]}-${c[1]}`} x={c[0] * cw} y={side - (c[1] + 1) * cw} width={cw + 0.5} height={cw + 0.5} fill={Number.isFinite(v) ? ramp(v) : INK.empty} />
  }), [cells, cw, side, stops.join(), mode]) // eslint-disable-line react-hooks/exhaustive-deps
  const cellAt = useMemo(() => new Map(cells.map((c) => [`${c[0]}-${c[1]}`, c])), [cells])
  const range = (a: number, b: number, i: number) => `${fmt(a + (i * (b - a)) / bins, 2)}–${fmt(a + ((i + 1) * (b - a)) / bins, 2)}`
  const cellTip = (c: number[]) => [`${xLabel}: ${range(x0, x1, c[0])}`, `${yLabel}: ${range(y0, y1, c[1])}`,
    `${c[2].toLocaleString()} ${c[2] === 1 ? 'filter' : 'filters'}${c.length > 3 ? ` · ${valueLabel} ${fmt(c[3], 2)}` : ''}`]

  const shown = useMemo(() => points.filter((p) => p.x >= x0 && p.x <= x1 && p.y >= y0 && p.y <= y1), [points, x0, x1, y0, y1])
  const xy = useMemo(() => shown.map((p) => [xs(p.x), ys(p.y)] as const), [shown, xs, ys])
  const keyOf = (p: DensityPoint) => (p.id ? baseFilterId(p.id) : null)
  const dots = useMemo(() => shown.map((p, i) => <circle key={i} cx={xy[i][0]} cy={xy[i][1]} r={3.2} fill={p.hollow ? 'none' : p.color}
    stroke={p.hollow ? p.color : dotRim().stroke} strokeWidth={p.hollow ? 1.3 : dotRim().strokeWidth} {...dotFill('sparse')}
    {...(p.id ? fidProps(baseFilterId(p.id)) : {})} />), [shown, xy])
  const markPoints = useMemo<MarkPoint[]>(() => marks.filter((p) => p.x >= x0 && p.x <= x1 && p.y >= y0 && p.y <= y1)
    .map((p) => ({ id: baseFilterId(p.id), at: [xs(p.x), ys(p.y)] as const, tip: p.tip })), [marks, xs, ys, x0, x1, y0, y1])

  // the legend reads counts on a log ramp: ticks at log10(count), labelled as counts
  const legendScale = Object.assign((v: number) => ramp(v), { domain: () => ramp.domain() })
  const items: LegendItem[] = [...legend, ...(box ? [{ label: box.key ?? box.label, color: INK.warn, shape: 'dash' as const }] : [])]
  return (
    <div ref={ref} className={`chart-wrap centered${fit === null ? '' : ' fit-square'}`} style={{ position: 'relative' }}>
      <svg width={side + M.left + M.right} height={side + M.top + M.bottom} fontFamily={FONT.family}>
        <g transform={`translate(${M.left},${M.top})`}>
          <YAxis scale={display.y} x={0} label={swapped ? xLabel : yLabel} gridWidth={side} tickCount={5} labelX={-46} />
          <XAxis scale={display.x} y={side} label={swapped ? yLabel : xLabel} tickCount={5} />
          <AxisSwap x={-38} y={side + 26} onSwap={swap} />
          <OrientedPlotLayer swapped={swapped} width={side} height={side}>
            <g pointerEvents="none">{cellRects}</g>
            {box && <g pointerEvents="none">
              <rect x={xs(box.x[0])} y={ys(box.y[1])} width={Math.max(0, xs(box.x[1]) - xs(box.x[0]))} height={Math.max(0, ys(box.y[0]) - ys(box.y[1]))} fill="none" stroke={INK.warn} strokeWidth={1.6} strokeDasharray="5 3" />
              <text x={xs(box.x[0]) + 4} y={ys(box.y[1]) - 4} fontSize={10.5} fill={INK.warn}>{box.label}</text>
            </g>}
            {children && <g pointerEvents="none">{children({ x: xs, y: ys })}</g>}
            {/* the points, the excluded ×, and the cell under the pointer when no point is near it */}
            <ScatterLayer items={shown} xy={xy} idOf={keyOf} dots={dots} width={side} height={side} tip={tip} marks={markPoints}
              focus={(p, [cx, cy]) => <circle cx={cx} cy={cy} r={6} fill={p.hollow ? 'none' : p.color} stroke={p.hollow ? p.color : INK.text} strokeWidth={2} {...dotFill('standard')} />}
              tipOf={(p) => (keyOf(p) ? [...p.tip, 'click to pin it on every chart and open its record'] : p.tip)}
              onMiss={(e) => {
                if (!e) { tip.hide(); return }
                const [mx, my] = d3.pointer(e)
                const c = cellAt.get(`${Math.floor(mx / cw)}-${Math.floor((side - my) / cw)}`)
                if (c) tip.show(e, cellTip(c)); else tip.hide()
              }} />
          </OrientedPlotLayer>
        </g>
      </svg>
      <Legend items={items} relabelFits={false}>
        <ColorLegend scale={legendScale} scheme={colours.scheme} label={cloudLabel ?? (mode === 'count' ? 'filters per cell' : valueLabel)}
          format={mode === 'count' ? countTick : undefined} />
      </Legend>
      {tip.node}
    </div>
  )
}
