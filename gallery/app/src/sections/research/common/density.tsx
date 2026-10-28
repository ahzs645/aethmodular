/**
 * A binned 2-D density from an exporter (cells of a bins × bins grid, each with a count and optionally a value),
 * with points drawn over it: the IMPROVE network as a cloud, a few hundred SPARTAN filters on top. For data too
 * large to ship row by row (research/improve_envelope).
 */
import { useRef, type ReactNode } from 'react'
import { useColourScheme } from '@/lib/colourSchemes'
import * as d3 from 'd3'
import { XAxis, YAxis } from '@/components/Axes'
import { ColorLegend } from '@/components/ColorLegend'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { fmt } from '@/lib/format'
import { FONT, INK } from '@/lib/theme'

export interface DensityPoint { x: number; y: number; color: string; hollow?: boolean; tip: string[] }

export function DensityPlot({ id, cells, bins, x: [x0, x1], y: [y0, y1], xLabel, yLabel, mode = 'count', valueLabel = 'filters per cell', points = [], box, maxSide = 460, children }: {
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
  valueLabel?: string
  points?: DensityPoint[]
  /** a shaded window, e.g. the Addis p05–p95 */
  box?: { x: [number, number]; y: [number, number]; label: string }
  maxSide?: number
  children?: (s: { x: d3.ScaleLinear<number, number>; y: d3.ScaleLinear<number, number> }) => ReactNode
}) {
  const ref = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(ref)
  const tip = useTooltip(ref)
  const m = { top: 12, right: 14, bottom: 46, left: 60 }
  const side = Math.max(200, Math.min(maxSide, width - m.left - m.right))
  const xs = d3.scaleLinear().domain([x0, x1]).range([0, side])
  const ys = d3.scaleLinear().domain([y0, y1]).range([side, 0])
  const cw = side / bins
  const colours = useColourScheme(id, 'sequential')
  // the Gallery ramp's palest step is the page itself: start one step in, as before
  const stops = colours.name === 'Gallery' && !colours.reverse ? colours.stops.slice(1) : colours.stops
  const vals = cells.map((c) => (mode === 'count' ? Math.log10(c[2]) : c[3])).filter(Number.isFinite)
  const ramp = d3.scaleQuantize<string>().domain([d3.min(vals) ?? 0, d3.max(vals) ?? 1]).range(stops)
  // the legend reads counts, though the ramp runs on log10(count)
  const legendScale = mode === 'count'
    ? Object.assign((v: number) => ramp(Math.log10(v)), { domain: () => ramp.domain().map((v) => 10 ** v) })
    : Object.assign((v: number) => ramp(v), { domain: () => ramp.domain() })
  return (
    <div ref={ref} style={{ position: 'relative' }}>
      <svg width={side + m.left + m.right} height={side + m.top + m.bottom} fontFamily={FONT.family}>
        <g transform={`translate(${m.left},${m.top})`}>
          <YAxis scale={ys} x={0} label={yLabel} gridWidth={side} tickCount={5} labelX={-46} />
          <XAxis scale={xs} y={side} label={xLabel} tickCount={5} />
          {cells.map((c) => {
            const v = mode === 'count' ? Math.log10(c[2]) : c[3]
            return <rect key={`${c[0]}-${c[1]}`} x={c[0] * cw} y={side - (c[1] + 1) * cw} width={cw + 0.5} height={cw + 0.5} fill={Number.isFinite(v) ? ramp(v) : INK.empty}
              onMouseMove={(e) => tip.show(e, [`${xLabel}: ${fmt(x0 + (c[0] * (x1 - x0)) / bins, 2)}–${fmt(x0 + ((c[0] + 1) * (x1 - x0)) / bins, 2)}`,
                `${yLabel}: ${fmt(y0 + (c[1] * (y1 - y0)) / bins, 2)}–${fmt(y0 + ((c[1] + 1) * (y1 - y0)) / bins, 2)}`, `${c[2].toLocaleString()} filters${c.length > 3 ? ` · ${valueLabel} ${fmt(c[3], 2)}` : ''}`])}
              onMouseLeave={tip.hide} />
          })}
          {box && <g pointerEvents="none">
            <rect x={xs(box.x[0])} y={ys(box.y[1])} width={Math.max(0, xs(box.x[1]) - xs(box.x[0]))} height={Math.max(0, ys(box.y[0]) - ys(box.y[1]))} fill="none" stroke={INK.warn} strokeWidth={1.6} strokeDasharray="5 3" />
            <text x={xs(box.x[0]) + 4} y={ys(box.y[1]) - 4} fontSize={10.5} fill={INK.warn}>{box.label}</text>
          </g>}
          {children?.({ x: xs, y: ys })}
          {points.filter((p) => p.x >= x0 && p.x <= x1 && p.y >= y0 && p.y <= y1).map((p, i) => <circle key={i} cx={xs(p.x)} cy={ys(p.y)} r={3.2} fill={p.hollow ? 'none' : p.color} fillOpacity={0.85}
            stroke={p.hollow ? p.color : INK.panel} strokeWidth={p.hollow ? 1.3 : 0.6} onMouseMove={(e) => tip.show(e, p.tip)} onMouseLeave={tip.hide} />)}
        </g>
      </svg>
      <ColorLegend scale={legendScale} scheme={colours.scheme}
        label={mode === 'count' ? 'filters per cell (log)' : valueLabel} format={(v) => (v >= 100 ? d3.format('.2s')(v) : fmt(v, v >= 10 ? 0 : 1))} />
      {tip.node}
    </div>
  )
}
