import { useRef } from 'react'
import * as d3 from 'd3'
import { XAxis } from '@/components/Axes'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { FONT, INK } from '@/lib/theme'

/**
 * Grouped dot-interval rows (a forest plot): blocks of rows, e.g. one block per
 * season, each row a labelled estimate with its interval, one or more panels
 * side by side (slope and intercept), each with its agreement line. The one
 * implementation behind every "median and 10–90 %" strip on the research pages
 * (docs/gallery-chart-standards.md §2.3).
 */
export type IntervalShape = 'dot' | 'hollow' | 'square' | 'diamond'
export interface IntervalMark {
  mid: number | null | undefined
  lo?: number | null
  hi?: number | null
  color: string
  shape?: IntervalShape
  /** vertical offset within the row, to stack two marks (e.g. Addis above, IMPROVE below) */
  dy?: number
  /** marker radius; whisker drawn lighter for secondary marks */
  r?: number
  faint?: boolean
}
export interface IntervalRow {
  key: string; label: string; marks: IntervalMark[][]; tip?: string[]
  labelColor?: string
  /** right-hand annotation, e.g. "n=42" */
  note?: string
  selected?: boolean
  onClick?: () => void
}
export interface IntervalBlock { key: string; title?: string; color?: string; rows: IntervalRow[] }
export interface IntervalPanel {
  /** axis title under the panel */
  label: string
  /** agreement value, drawn as the reference line (1 for a slope, 0 for an intercept) */
  reference?: number
  /** heading above the panel when there are several */
  heading?: string
  domain?: [number, number]
  /** a labelled data value to compare against (dashed), e.g. the target group's own r */
  marker?: { value: number; label: string }
  /** draw each mark's interval from the panel's left edge (a lollipop stem) */
  stems?: boolean
  format?: (v: number) => string
}

export function IntervalStrip({ blocks, panels, labelWidth = 220, rowHeight = 20, blockGap = 14, minPanelWidth = 260, noteWidth = 0 }: {
  blocks: IntervalBlock[]
  panels: IntervalPanel[]
  labelWidth?: number
  rowHeight?: number
  blockGap?: number
  minPanelWidth?: number
  /** room for each row's right-hand note */
  noteWidth?: number
}) {
  const ref = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(ref)
  const tip = useTooltip(ref)
  const panelGap = 28
  const m = { top: panels.some((p) => p.heading || p.marker) ? 24 : 10, right: 20 + noteWidth, bottom: 44, left: labelWidth }
  const pw = Math.max(minPanelWidth, (width - m.left - m.right - panelGap * (panels.length - 1)) / panels.length)
  const scales = panels.map((p, k) => {
    if (p.domain) return d3.scaleLinear().domain(p.domain).range([0, pw])
    const vals = blocks.flatMap((b) => b.rows.flatMap((r) => (r.marks[k] ?? []).flatMap((mk) => [mk.lo, mk.mid, mk.hi])))
      .filter((v): v is number => v != null && Number.isFinite(v))
    const extra = [...(p.reference != null ? [p.reference] : []), ...(p.marker ? [p.marker.value] : [])]
    return d3.scaleLinear().domain(d3.extent([...vals, ...extra]) as [number, number]).nice().range([0, pw])
  })
  // block layout: a title line (when the block has a title), its rows, then a gap; titles get their own
  // line so they never collide with long right-aligned row labels
  const HEADER = 16
  let cursor = 0
  const layout = blocks.map((b) => {
    const top = cursor, rowsTop = top + (b.title ? HEADER : 0)
    cursor = rowsTop + b.rows.length * rowHeight + blockGap
    return { b, top, rowsTop }
  })
  const ih = Math.max(rowHeight, cursor - blockGap)
  const w = m.left + panels.length * pw + (panels.length - 1) * panelGap + m.right
  const finite = (v: number | null | undefined): v is number => v != null && Number.isFinite(v)
  const mark = (mk: IntervalMark, x: d3.ScaleLinear<number, number>, cy: number, i: number, stems = false, selected = false) => {
    const y0 = cy + (mk.dy ?? 0), r = (mk.r ?? 4.5) + (selected ? 1 : 0), cx = finite(mk.mid) ? x(mk.mid) : null
    return <g key={i}>
      {stems && cx != null && <line x1={0} x2={cx} y1={y0} y2={y0} stroke={mk.color} strokeOpacity={0.45} strokeWidth={1.5} />}
      {selected && cx != null && <circle cx={cx} cy={y0} r={r + 1.5} fill="none" stroke={INK.text} strokeWidth={1} />}
      {finite(mk.lo) && finite(mk.hi) && <line x1={x(mk.lo)} x2={x(mk.hi)} y1={y0} y2={y0} stroke={mk.color} strokeWidth={mk.faint ? 1.4 : 1.8} strokeOpacity={mk.faint ? 0.45 : 0.8} />}
      {cx != null && (mk.shape ?? 'dot') === 'dot' && <circle cx={cx} cy={y0} r={r} fill={mk.color} />}
      {cx != null && mk.shape === 'hollow' && <circle cx={cx} cy={y0} r={r} fill="white" stroke={mk.color} strokeWidth={1.5} />}
      {cx != null && mk.shape === 'square' && <rect x={cx - r} y={y0 - r} width={2 * r} height={2 * r} fill={mk.color} />}
      {cx != null && mk.shape === 'diamond' && <path d={`M${cx} ${y0 - r} L${cx + r} ${y0} L${cx} ${y0 + r} L${cx - r} ${y0} Z`} fill="white" stroke={mk.color} strokeWidth={1} />}
    </g>
  }
  return (
    <div ref={ref} style={{ position: 'relative' }}>
      <svg width={w} height={ih + m.top + m.bottom} fontFamily={FONT.family}>
        <g transform={`translate(${m.left},${m.top})`}>
          {layout.map(({ b, top, rowsTop }, bi) => <g key={b.key}>
            {bi % 2 === 0 && blocks.length > 1 && <rect x={-m.left + 4} y={top - 3} width={w - 8} height={rowsTop - top + b.rows.length * rowHeight + 4} fill={INK.page} />}
            {b.title && <text x={-m.left + 8} y={top + 11} fontSize={11.5} fontWeight={700} fill={b.color ?? INK.text}>{b.title}</text>}
          </g>)}
          {panels.map((p, k) => {
            const x = scales[k], left = k * (pw + panelGap)
            return <g key={k} transform={`translate(${left},0)`}>
              {p.heading && <text x={0} y={-10} fontSize={12} fontWeight={600} fill={INK.text}>{p.heading}</text>}
              {p.reference != null && <line x1={x(p.reference)} x2={x(p.reference)} y1={-4} y2={ih + 4} stroke={INK.axis} strokeOpacity={0.7} />}
              {p.marker && <g>
                <line x1={x(p.marker.value)} x2={x(p.marker.value)} y1={-6} y2={ih + 4} stroke={INK.text} strokeDasharray="3 3" />
                <text x={x(p.marker.value)} y={-10} textAnchor="middle" fontSize={10} fontFamily={FONT.mono} fill={INK.muted}>{p.marker.label}</text>
              </g>}
              <XAxis scale={x} y={ih + 4} tickCount={6} label={p.label} labelDy={36} format={(v) => p.format ? p.format(+v) : String(+(+v).toFixed(2))} />
            </g>
          })}
          {layout.map(({ b, rowsTop }) => b.rows.map((r, j) => {
            const cy = rowsTop + j * rowHeight + rowHeight / 2
            return <g key={`${b.key}|${r.key}`} onMouseEnter={(e) => r.tip && tip.show(e, r.tip)} onMouseLeave={tip.hide}
              onClick={r.onClick} style={r.onClick ? { cursor: 'pointer' } : undefined}>
              {r.selected && <rect x={-m.left + 4} y={cy - rowHeight / 2} width={w - 8} height={rowHeight} fill={INK.accent} fillOpacity={0.08} />}
              <text x={-10} y={cy} dy="0.32em" textAnchor="end" fontSize={11} fill={r.labelColor ?? INK.text} fontWeight={r.selected ? 600 : 400}>{r.label}</text>
              {panels.map((p, k) => <g key={k} transform={`translate(${k * (pw + panelGap)},0)`}>{(r.marks[k] ?? []).map((mk, i) => mark(mk, scales[k], cy, i, p.stems, r.selected))}</g>)}
              {r.note && <text x={w - m.left - m.right + 8} y={cy} dy="0.32em" fontSize={10.5} fontFamily={FONT.mono} fill={INK.muted}>{r.note}</text>}
              <rect x={-m.left + 4} y={cy - rowHeight / 2} width={w - 8} height={rowHeight} fill="transparent" />
            </g>
          }))}
        </g>
      </svg>
      {tip.node}
    </div>
  )
}
