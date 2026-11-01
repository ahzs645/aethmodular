import { useMemo, useRef, type MouseEvent, type ReactNode } from 'react'
import * as d3 from 'd3'
import { XAxis } from '@/components/Axes'
import { RefLine } from '@/components/RefLines'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip, type TipBody } from '@/hooks/useTooltip'
import { useTheme } from '@/lib/colorTheme'
import { textWidth } from '@/lib/fitLabel'
import { fmt } from '@/lib/format'
import { rowTops } from '@/lib/heatMatrix'
import { FONT, INK, inkOn } from '@/lib/theme'

/** A part of a stacked bar */
export interface BarSegment { key: string; value: number; color: string; opacity?: number; label?: string | null; tip?: TipBody }

/** One labelled row: a bar, a stack of segments, or bars side by side, with ticks over them */
export interface BarRow {
  key: string
  label: string
  /** the full name, in the tooltip when the label is cut */
  title?: string
  labelColor?: string
  labelWeight?: number
  italic?: boolean
  /** a single bar from zero */
  value?: number | null
  /** colour by the sign of this number (the gallery's r: blue positive, red negative), else `color` */
  signed?: number | null
  color?: string
  opacity?: number
  /** a stacked bar, left to right */
  segments?: BarSegment[]
  /** thin bars side by side in the row (two populations), each from zero */
  bars?: { key: string; value: number; color: string; opacity?: number }[]
  /** short ticks across the row at values (a share to compare with) */
  marks?: { value: number; color?: string }[]
  /** the text after the bar (default: the formatted value or the stack's total); null for none */
  valueLabel?: string | null
  /** muted text after the value label */
  note?: string
  tip?: TipBody
  /** a gap (and, with groupLabels, a heading) before each new group */
  group?: string
}

export interface BarRowsProps {
  rows: BarRow[]
  /** the value axis; default from zero to the largest value (and the smallest, when negative), rounded */
  domain?: [number, number]
  /** or only its top */
  max?: number
  format?: (v: number) => string
  /** the axis under the rows: its title, tick count and format (false: none) */
  axis?: { label?: string; ticks?: number; format?: (v: number) => string } | false
  /** a reference line (zero is drawn on its own when the axis spans it) */
  reference?: number
  sort?: 'none' | 'desc' | 'asc'
  rowHeight?: number
  barHeight?: number
  labelSize?: number
  /** labels wider than this are cut, the full name in their tooltip */
  maxLabelWidth?: number
  /** print each row's value label */
  valueLabels?: boolean
  zebra?: boolean
  groupLabels?: boolean
  groupGap?: number
  onRow?: (row: BarRow) => void
  selected?: string | null
  /** width to fill (default: measured) */
  width?: number
  children?: ReactNode
}

const total = (r: BarRow) => (r.segments ? d3.sum(r.segments, (s) => s.value) : r.bars ? d3.max(r.bars, (b) => b.value) ?? 0 : r.value ?? null)

/**
 * Labelled horizontal bar rows: a screen's scores, a ranking, shares of a whole, two populations side
 * by side. The label gutter is measured from the labels (long ones are cut, the full name in the
 * tooltip), values print after the bars through lib/format, signed rows take the gallery's r colours,
 * and the axis is the shared XAxis. Figure's `bars` kind draws one from a table.
 *
 *   <BarRows rows={scan.map((s) => ({ key: s.f, label: s.f, value: s.rho ** 2, signed: s.rho }))} axis={{ label: 'Spearman ρ²' }} sort="desc" />
 */
export function BarRows(p: BarRowsProps) {
  useTheme()
  const ref = useRef<HTMLDivElement>(null)
  const measured = useDimensions(ref).width
  const tip = useTooltip(ref)
  const width = p.width ?? measured
  const fmtV = p.format ?? ((v: number) => fmt(v, 2))
  const rows = useMemo(() => {
    const shown = p.rows.filter((r) => total(r) !== null)
    if (!p.sort || p.sort === 'none') return shown
    return [...shown].sort((a, b) => (p.sort === 'desc' ? 1 : -1) * ((total(b) ?? 0) - (total(a) ?? 0)))
  }, [p.rows, p.sort])
  const size = p.labelSize ?? 11.5
  const rowH = p.rowHeight ?? 20, barH = p.barHeight ?? rowH - 6

  // the label gutter, measured; at most 40 % of a narrow panel
  const cap = Math.min(p.maxLabelWidth ?? 240, Math.max(90, width * 0.4))
  const labels = useMemo(() => rows.map((r) => {
    const w = textWidth(r.label, size, r.labelWeight ?? 400)
    if (w <= cap) return { text: r.label, cut: false, w }
    let s = r.label
    while (s.length > 1 && textWidth(`${s}…`, size, r.labelWeight ?? 400) > cap) s = s.slice(0, -1)
    return { text: `${s.trimEnd()}…`, cut: true, w: cap }
  }), [rows, size, cap])
  const labelW = Math.max(0, ...labels.map((l) => l.w)) + 8
  const valueOf = (r: BarRow) => (r.valueLabel !== undefined ? r.valueLabel : p.valueLabels === false ? null : fmtV(total(r) ?? 0))
  const tail = (r: BarRow) => [valueOf(r), r.note].filter(Boolean).join('  ')
  const tailW = Math.max(0, ...rows.map((r) => { const t = tail(r); return t ? textWidth(t, 10.5, 400, FONT.mono) + 8 : 0 }))
  const pw = Math.max(120, width - labelW - tailW - 8)

  const domain = useMemo<[number, number]>(() => {
    if (p.domain) return p.domain
    const vals = rows.flatMap((r) => [total(r) ?? 0, ...(r.marks ?? []).map((m) => m.value)])
    return [Math.min(0, d3.min(vals) ?? 0), p.max ?? (Math.max(0, d3.max(vals) ?? 0) || 1)]
  }, [rows, p.domain, p.max])
  const x = useMemo(() => { const s = d3.scaleLinear().domain(domain).range([0, pw]); return p.domain ? s : s.nice() }, [domain, pw, p.domain])

  const headH = p.groupLabels ? 22 : 0
  const { tops, height: rowsH } = rowTops(rows.map((r) => r.group), rowH, p.groupLabels ? headH : p.groupGap ?? 8)
  const axisH = p.axis === false ? 4 : p.axis?.label ? 38 : 22
  const h = headH + rowsH + axisH + 4
  const zero = x(0)
  // a row's tooltip, and a segment's over it (hooks/useTooltip tip.bind); a segment keeps its move from reaching the row
  const rowTip = (body?: TipBody) => (body ? tip.bind(body) : { onMouseLeave: tip.hide })
  const segTip = (body?: TipBody) => {
    if (!body) return {}
    const h = tip.bind(body)
    return { ...h, onMouseMove: (e: MouseEvent) => { e.stopPropagation(); h.onMouseMove(e) } }
  }

  return (
    <div ref={ref} className="chart-wrap">
      <svg width={labelW + pw + tailW} height={h} fontFamily={FONT.family}>
        {p.groupLabels && rows.map((r, i) => (i === 0 || r.group !== rows[i - 1].group) && r.group
          ? <text key={`g${r.key}`} x={0} y={tops[i] + headH - 8} fontSize={11} fontWeight={700} fill={INK.text}>{r.group}</text> : null)}
        <g transform={`translate(0,${headH})`}>
          {rows.map((r, i) => {
            const y = tops[i], t = total(r) ?? 0
            const c = r.signed != null ? (r.signed >= 0 ? INK.positive : INK.negative) : r.color ?? INK.accent
            const on = p.selected === r.key
            const tailText = tail(r)
            const end = r.segments ? x(Math.max(0, t)) : x(t)
            return <g key={r.key} transform={`translate(0,${y})`} {...rowTip(r.tip ?? (labels[i].cut ? [r.title ?? r.label] : undefined))}
              onClick={p.onRow ? () => p.onRow!(r) : undefined} style={p.onRow ? { cursor: 'pointer' } : undefined}>
              {(p.zebra && i % 2 === 0) || on ? <rect x={0} y={0} width={labelW + pw} height={rowH} fill={on ? INK.tint : INK.zebra} /> : null}
              <text x={labelW - 8} y={rowH / 2} dy="0.35em" textAnchor="end" fontSize={size} fontWeight={on ? 650 : r.labelWeight} fontStyle={r.italic ? 'italic' : undefined} fill={r.labelColor ?? INK.text}>{labels[i].text}</text>
              <g transform={`translate(${labelW},0)`}>
                {r.segments ? (() => {
                  let acc = 0
                  return r.segments.map((s) => {
                    const x0 = x(acc), x1 = x(acc + s.value)
                    acc += s.value
                    const w = Math.max(0, x1 - x0)
                    return <g key={s.key} {...segTip(s.tip)}>
                      <rect x={x0} y={(rowH - barH) / 2} width={w} height={barH} fill={s.color} fillOpacity={s.opacity ?? 0.9} />
                      {s.label && w > textWidth(s.label, 10) + 8 && <text x={x0 + w / 2} y={rowH / 2} dy="0.35em" textAnchor="middle" fontSize={10} fill={inkOn(s.color)} pointerEvents="none">{s.label}</text>}
                    </g>
                  })
                })()
                : r.bars ? r.bars.map((b, k) => {
                  const bh = (barH - (r.bars!.length - 1)) / r.bars!.length
                  return <rect key={b.key} x={Math.min(zero, x(b.value))} y={(rowH - barH) / 2 + k * (bh + 1)} width={Math.abs(x(b.value) - zero)} height={bh} fill={b.color} fillOpacity={b.opacity ?? 1} />
                })
                : <rect x={Math.min(zero, x(t))} y={(rowH - barH) / 2} width={Math.max(1, Math.abs(x(t) - zero))} height={barH} fill={c} fillOpacity={r.opacity ?? 0.75} />}
                {(r.marks ?? []).map((m, k) => <line key={k} x1={x(m.value)} x2={x(m.value)} y1={(rowH - barH) / 2 - 2} y2={(rowH + barH) / 2 + 2} stroke={m.color ?? INK.text} strokeWidth={1.5} />)}
                {tailText && <text x={t < 0 && !r.segments ? end - 4 : end + 4} y={rowH / 2} dy="0.35em" textAnchor={t < 0 && !r.segments ? 'end' : 'start'} fontSize={10.5} fontFamily={FONT.mono} fill={INK.muted}>{tailText}</text>}
              </g>
            </g>
          })}
          <g transform={`translate(${labelW},0)`}>
            {domain[0] < 0 && domain[1] > 0 && <RefLine orient="v" at={zero} from={0} to={rowsH} />}
            {p.reference != null && <RefLine orient="v" at={x(p.reference)} from={0} to={rowsH} />}
            {p.axis !== false && <XAxis scale={x} y={rowsH + 2} tickCount={p.axis?.ticks ?? 4} format={p.axis?.format ?? fmtV} label={p.axis?.label} labelDy={30} />}
          </g>
        </g>
      </svg>
      {p.children}
      {tip.node}
    </div>
  )
}
