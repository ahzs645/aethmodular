import { useMemo, useRef } from 'react'
import * as d3 from 'd3'
import { XAxis } from '@/components/Axes'
import { Legend, useLegend, type LegendItem } from '@/components/Legend'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { useFitHeight } from '@/lib/chartFit'
import { f2 } from '@/lib/format'
import { FONT, INK } from '@/lib/theme'

/**
 * Rows of ranges, one measure per row on its own axis (linear or log): a reference distribution as an
 * outer and an inner band with a median tick, single reference values as dashed ticks, and points (the
 * samples under test) as markers, with a verdict on the right saying where the points sit.
 *
 * Every layer is a legend series: clicking one hides it and each row's axis rescales to what is left;
 * pointing at one dims the others. The width follows the panel (the verdict drops under the axis when it
 * is narrow), the label gutter follows the labels, and the row height follows the fitted height in the
 * single-chart view and Dashboard tiles (lib/chartFit). Hover any mark for its values.
 */
export interface RangeSeries {
  key: string
  label: string
  color: string
  /** band: the row's `band` (outer range, inner range, median); tick: one value per row; points: several */
  kind: 'band' | 'tick' | 'points'
  symbol?: 'diamond' | 'dot'
}
export interface RangeBand { lo?: number; q1?: number; mid?: number; q3?: number; hi?: number }
export interface RangeRow {
  key: string
  label: string
  /** the second line under the label, e.g. the unit */
  unit?: string
  log?: boolean
  /** the band series' values in this row (one band series per chart) */
  band?: RangeBand
  /** a tick series' value in this row, by series key */
  ticks?: Record<string, number | null | undefined>
  /** a points series' values in this row, by series key */
  points?: Record<string, number[]>
  verdict?: { text: string; color: string; detail?: string }
  /** values in tooltips (default two decimals) */
  format?: (v: number) => string
  /** the axis's tick labels (default compact: 0.1, 1, 10, 2.5k) */
  tickFormat?: (v: number) => string
}

const ROW_H = 46
const VERDICT_GAP = 14
const compact = d3.format('~s')
const tickText = (v: number) => (Math.abs(v) >= 1000 ? compact(v) : d3.format('~g')(+v.toPrecision(3)))
const fin = (v: number | null | undefined): v is number => v != null && Number.isFinite(v)

let measurer: CanvasRenderingContext2D | null = null
/** the drawn width of a label, so the gutters follow the text instead of a guessed character width */
function textWidth(text: string, size: number, weight = 400) {
  measurer ??= document.createElement('canvas').getContext('2d')
  if (!measurer) return text.length * size * 0.6
  measurer.font = `${weight} ${size}px ${FONT.family}`
  return measurer.measureText(text).width
}

export function RangeRows({ rows, series, bandNames = ['5–95 %', '25–75 %'], pointTip }: {
  rows: RangeRow[]
  series: RangeSeries[]
  /** what the outer and inner bands are, for the tooltip */
  bandNames?: [string, string]
  /** a point's tooltip lines; defaults to the series, the row and the value */
  pointTip?: (row: RangeRow, s: RangeSeries, value: number, index: number) => string[]
}) {
  const ref = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(ref)
  const tip = useTooltip(ref)
  const lg = useLegend()
  const fit = useFitHeight()
  const shown = series.filter((s) => lg.show(s.label))
  const op = (s: RangeSeries) => lg.dim(s.label)

  const labelW = useMemo(() => Math.min(240, Math.max(90, 16 + Math.max(0, ...rows.map((r) => Math.max(textWidth(r.label, 11.5, 600), textWidth(r.unit ?? '', 9.5))))) ), [rows])
  const verdictW = useMemo(() => Math.min(230, Math.max(0, ...rows.map((r) => r.verdict ? 14 + Math.max(textWidth(r.verdict.text, 10.5, 600), textWidth(r.verdict.detail ?? '', 9.5)) : 0))), [rows])
  // narrow: the verdict goes under its row's axis instead of beside it
  const stacked = verdictW > 0 && width - labelW - verdictW < 300
  const pw = Math.max(160, width - labelW - (stacked ? 0 : verdictW) - 16)
  const extra = stacked ? 2 * VERDICT_GAP : 0
  const rowH = fit === null ? ROW_H + extra : Math.max(ROW_H - 6 + extra, Math.min(96, (fit - 44) / Math.max(1, rows.length)))

  return (
    <div ref={ref} style={{ position: 'relative' }}>
      <svg width={labelW + pw + (stacked ? 0 : verdictW) + 16} height={rows.length * rowH + 8} fontFamily={FONT.family} style={{ display: 'block' }}>
        {rows.map((r, i) => {
          const fmt = r.format ?? f2
          const cy = i * rowH + Math.min(rowH / 2 - extra / 2, 26)
          const band = shown.find((s) => s.kind === 'band') ? r.band : undefined
          const vals = [
            ...(band ? [band.lo ?? band.q1, band.hi ?? band.q3] : []),
            ...shown.filter((s) => s.kind === 'tick').map((s) => r.ticks?.[s.key]),
            ...shown.filter((s) => s.kind === 'points').flatMap((s) => r.points?.[s.key] ?? []),
          ].filter((v): v is number => fin(v) && (!r.log || v > 0))
          const ext = d3.extent(vals) as [number, number]
          const dom: [number, number] = !vals.length ? (r.log ? [1, 10] : [0, 1]) : ext[0] === ext[1] ? (r.log ? [ext[0] / 2, ext[0] * 2] : [ext[0] - 1, ext[0] + 1]) : ext
          const x = (r.log ? d3.scaleLog() : d3.scaleLinear()).domain(dom).range([0, pw]).nice()
          const bandSeries = series.find((s) => s.kind === 'band')
          return <g key={r.key}>
            <text x={labelW - 10} y={cy} dy="0.32em" textAnchor="end" fontSize={11.5} fontWeight={600} fill={INK.text}>{r.label}</text>
            {r.unit && <text x={labelW - 10} y={cy + 13} dy="0.32em" textAnchor="end" fontSize={9.5} fill={INK.muted}>{r.unit}</text>}
            <g transform={`translate(${labelW},0)`}>
              {band && bandSeries && <g opacity={op(bandSeries)}
                onMouseEnter={(e) => tip.show(e, [r.label, bandSeries.label,
                  ...(fin(band.lo) && fin(band.hi) ? [`${bandNames[0]}: ${fmt(band.lo)}–${fmt(band.hi)} ${r.unit ?? ''}`] : []),
                  ...(fin(band.q1) && fin(band.q3) ? [`${bandNames[1]}: ${fmt(band.q1)}–${fmt(band.q3)}`] : []),
                  ...(fin(band.mid) ? [`median ${fmt(band.mid)}`] : [])])} onMouseLeave={tip.hide}>
                {fin(band.lo) && fin(band.hi) && <rect x={x(band.lo)} y={cy - 8} width={Math.max(1, x(band.hi) - x(band.lo))} height={16} fill={bandSeries.color} fillOpacity={0.25} />}
                {fin(band.q1) && fin(band.q3) && <rect x={x(band.q1)} y={cy - 8} width={Math.max(1, x(band.q3) - x(band.q1))} height={16} fill={bandSeries.color} fillOpacity={0.6} />}
                {fin(band.mid) && <line x1={x(band.mid)} x2={x(band.mid)} y1={cy - 10} y2={cy + 10} stroke={d3.color(bandSeries.color)?.darker(1.4).formatHex() ?? INK.text} strokeWidth={2} />}
              </g>}
              {shown.filter((s) => s.kind === 'tick').map((s) => {
                const v = r.ticks?.[s.key]
                if (!fin(v) || (r.log && v <= 0)) return null
                return <g key={s.key} opacity={op(s)} onMouseEnter={(e) => tip.show(e, [s.label, `${r.label}: ${fmt(v)} ${r.unit ?? ''}`])} onMouseLeave={tip.hide}>
                  {/* a wider invisible line, so a 2 px tick is easy to point at */}
                  <line x1={x(v)} x2={x(v)} y1={cy - 13} y2={cy + 13} stroke="transparent" strokeWidth={8} />
                  <line x1={x(v)} x2={x(v)} y1={cy - 13} y2={cy + 13} stroke={s.color} strokeWidth={2} strokeDasharray="2 2" />
                </g>
              })}
              {shown.filter((s) => s.kind === 'points').map((s) => <g key={s.key} opacity={op(s)}>
                {(r.points?.[s.key] ?? []).map((v, j) => !fin(v) || (r.log && v <= 0) ? null : (
                  <path key={j} d={d3.symbol(s.symbol === 'dot' ? d3.symbolCircle : d3.symbolDiamond, 80)() ?? ''} transform={`translate(${x(v)},${cy})`}
                    fill={s.color} stroke={INK.panel} strokeWidth={1}
                    onMouseEnter={(e) => tip.show(e, pointTip ? pointTip(r, s, v, j) : [s.label, `${r.label}: ${fmt(v)} ${r.unit ?? ''}`])} onMouseLeave={tip.hide} />
                ))}
              </g>)}
              <XAxis scale={x} y={cy + 12} tickCount={Math.max(2, Math.min(6, Math.floor(pw / 90)))} format={(tv) => (r.tickFormat ?? tickText)(+tv)} />
            </g>
            {r.verdict && <g transform={stacked ? `translate(${labelW},${cy + 34})` : `translate(${labelW + pw + 14},${cy})`}>
              <text dy="0.32em" fontSize={10.5} fontWeight={600} fill={r.verdict.color}>{r.verdict.text}</text>
              {r.verdict.detail && <text x={stacked ? textWidth(r.verdict.text, 10.5, 600) + 8 : 0} y={stacked ? 0 : 13} dy="0.32em" fontSize={9.5} fill={INK.muted}>{r.verdict.detail}</text>}
            </g>}
          </g>
        })}
      </svg>
      <Legend {...lg.props} items={series.map((s): LegendItem => ({ label: s.label, color: s.color,
        shape: s.kind === 'band' ? 'band' : s.kind === 'tick' ? 'dashed' : s.symbol === 'dot' ? 'dot' : 'diamond' }))} />
      {tip.node}
    </div>
  )
}
