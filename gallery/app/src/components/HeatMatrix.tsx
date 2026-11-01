import { useMemo, useRef, useState, useSyncExternalStore, type MouseEvent, type ReactNode, type SVGProps } from 'react'
import { ColorLegend } from '@/components/ColorLegend'
import type { SchemeChoice } from '@/components/ColourSchemePicker'
import { HeatCellBody } from '@/components/HeatCell'
import { Key, type KeyItem } from '@/components/Key'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip, type TipBody } from '@/hooks/useTooltip'
import { useFitHeight } from '@/lib/chartFit'
import { useTheme } from '@/lib/colorTheme'
import { textWidth } from '@/lib/fitLabel'
import { colAt, groupStarts, rowAt, rowTops } from '@/lib/heatMatrix'
import { FONT, INK } from '@/lib/theme'

/** A row or a column of a matrix: its key, and how its label is drawn. A bare string is its own key and label. */
export interface MatrixItem {
  key: string
  /** the label drawn (default: the key); '' draws none */
  label?: string
  /** a shorter label for a column too narrow for `label` (Jan → J) */
  short?: string
  /** the full name, in the tooltip when the label is cut */
  title?: string
  /** label ink */
  color?: string
  weight?: number
  size?: number
  italic?: boolean
  /** set the label in the mono face (configuration codes) */
  mono?: boolean
  /** a coloured dot before a row label (a site's colour) */
  dot?: string
  /** muted text after a row label, e.g. "n=24" */
  suffix?: string
  /** consecutive rows (or columns) with the same group form a block: a gap and a label for rows, a label and a rule for columns */
  group?: string
  /** the tooltip on the label */
  tip?: TipBody
}
export type MatrixAxis = readonly (string | MatrixItem)[]

/** What one cell draws. A bare number is a value coloured by `color`; null an empty cell; undefined no cell at all. */
export interface MatrixCell {
  value?: number | null
  /** the fill, when it is not `color(value)`; null draws the empty-cell colour */
  fill?: string | null
  /** the printed text (default: `format(value)` where it fits); null prints nothing */
  label?: string | null
  /** a smaller second line */
  sublabel?: string | null
  tip?: TipBody
  opacity?: number
  outline?: boolean
  stroke?: string
  strokeWidth?: number
  /** label ink, where the fill's own pick is wrong (a faded cell) */
  ink?: string
  /** attributes on the cell's rect, e.g. lib/filterEdits fidsProps so a right-click finds its filters */
  props?: SVGProps<SVGRectElement> | Record<`data-${string}`, string>
}
type Cell = MatrixCell | number | null | undefined

export interface MatrixLegend {
  /** the ramp; default the matrix's `color` when it is a d3 scale */
  scale?: { domain(): number[]; (v: number): string }
  scheme?: SchemeChoice
  label?: string
  format?: (v: number) => string
  note?: string
  ticks?: number
  width?: number
}

/** A column of text beside the cells (a row's n, its median) */
export interface ExtraColumn { key: string; header?: string; side?: 'left' | 'right'; text: (i: number) => string; color?: string }

/** Where everything sits, for overlays drawn by `children` */
export interface MatrixLayout {
  left: number; top: number; cellW: number; cellH: number; innerW: number; innerH: number; width: number; height: number
  x: (j: number) => number; y: (i: number) => number
}

export interface HeatMatrixProps {
  rows: MatrixAxis
  columns: MatrixAxis
  /** the cell at row i, column j */
  cell: (i: number, j: number) => Cell
  /** value → fill (a d3 scale from the chart's useColourScheme stops) */
  color?: (v: number) => string
  /** value → printed text, and the value line of the default tooltip */
  format?: (v: number) => string
  /** 'auto' prints a cell's text where it fits, 'never' leaves cells plain */
  cellLabels?: 'auto' | 'always' | 'never'
  labelSize?: number
  labelWeight?: number
  /** the tooltip of a cell, built when it is pointed at (else the cell's `tip`, else row · column and the value) */
  tip?: (i: number, j: number) => TipBody | null

  /** the width to fill (default: the measured width of the matrix's box) */
  width?: number
  /** a fixed cell width, or the range the cells stretch over to fill the width */
  cellWidth?: number | [number, number]
  /** a fixed row height, or 'square' for cells as tall as wide */
  cellHeight?: number | 'square'
  /** in a fitted Dashboard tile (lib/chartFit), the row heights the matrix may take to fill it; false keeps cellHeight */
  fitRows?: [number, number] | false
  /** space between cells: one number, or [across, down] */
  gap?: number | [number, number]
  radius?: number
  crisp?: boolean

  rowLabelSize?: number
  /** 'start' sets row labels flush left (default when rows carry dots) */
  rowLabelAlign?: 'start' | 'end'
  /** row labels wider than this are cut, the full name in their tooltip */
  maxLabelWidth?: number
  colLabelSize?: number
  colLabelWeight?: number
  colLabelColor?: string
  /** auto: level when they fit, on two lines, else slanted; slant: level when they fit, else slanted; fit: level, each one only where it fits (months) */
  colLabels?: 'auto' | 'slant' | 'horizontal' | 'wrap' | 'rotate' | 'fit' | 'none'
  colLabelAngle?: number
  colLabelAlign?: 'middle' | 'start'
  colLabelPosition?: 'top' | 'bottom'
  /** column groups (years over months): a label at each group's start, and a rule there when colGroupRules */
  colGroupRules?: boolean
  /** a title over the row labels, and one over (or under) the columns */
  rowHeader?: string
  colHeader?: string
  extraColumns?: ExtraColumn[]
  /** px between row groups */
  groupGap?: number
  /** each row group's name in a gutter left of the row labels */
  groupLabels?: boolean

  onCell?: (i: number, j: number) => void
  /** a click on a row's label (or its cells, without onCell) */
  onRow?: (i: number) => void
  /** the row drawn tinted, its label bold */
  selectedRow?: number | null
  /** what follows the pointer: a ring on the cell, its row and column, or nothing */
  hover?: 'cell' | 'cross' | 'none'

  /** a heading over the matrix, inside the figure (a facet's season) */
  title?: ReactNode
  titleColor?: string
  /** the colour ramp under the matrix (components/ColorLegend), with any `legendItems` (components/Key) beside it */
  legend?: MatrixLegend | null
  legendItems?: KeyItem[]
  legendExtra?: ReactNode
  /** notes after the legend, inside the figure */
  footer?: ReactNode
  /** the cells (not cells + labels) in the middle of the panel when there is room */
  center?: boolean
  /** px kept above the header and under the matrix for overlays drawn by `children` */
  before?: number
  after?: number
  children?: (layout: MatrixLayout) => ReactNode
  ariaLabel?: string
}

const asItem = (a: string | MatrixItem): MatrixItem => (typeof a === 'string' ? { key: a } : a)
const labelOf = (it: MatrixItem) => it.label ?? it.key
const norm = (c: Cell): MatrixCell | undefined => (c === undefined ? undefined : c === null ? { value: null } : typeof c === 'number' ? { value: c } : c)
const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v))
/** the drawn width of mono text: FONT.mono is monospaced, so no canvas is needed per cell */
const monoWidth = (s: string, size: number) => s.length * size * 0.6
const PAD = 8

/** cut a label with an ellipsis to a width */
function cut(text: string, size: number, room: number, weight = 400, family: string = FONT.family) {
  if (textWidth(text, size, weight, family) <= room) return text
  let s = text
  while (s.length > 1 && textWidth(`${s}…`, size, weight, family) > room) s = s.slice(0, -1)
  return `${s.trimEnd()}…`
}
/** split a label at the space nearest its middle */
function twoLines(text: string): string[] {
  const spaces = [...text.matchAll(/ /g)].map((m) => m.index ?? 0)
  if (!spaces.length) return [text]
  const at = spaces.reduce((a, b) => (Math.abs(b - text.length / 2) < Math.abs(a - text.length / 2) ? b : a))
  return [text.slice(0, at), text.slice(at + 1)]
}

/**
 * A heat-map matrix: row and column labels in gutters measured from the text, HeatCell cells with
 * their value printed in readable ink where it fits, the ramp (components/ColorLegend, which is also
 * the palette picker) under it, a tooltip per cell, and a click per cell or row. Every heat map in the
 * gallery is one of these, from a 3 × 4 grid to a 300-site similarity matrix:
 *
 *   const colours = useColourScheme('my-chart', 'diverging')
 *   const color = d3.scaleQuantize<string>().domain([-1, 1]).range(colours.stops)
 *   <HeatMatrix rows={sites} columns={vars} cell={(i, j) => rho[i][j]} color={color} format={(v) => fmt(v, 2)}
 *     legend={{ scheme: colours.scheme, label: 'Spearman ρ' }} />
 *
 * One pointer handler serves every cell (no per-cell closures), and the hover ring and the tooltip
 * live in small stores, so pointing re-renders neither the cells nor the chart. The cells redraw only
 * when the caller re-renders; a dense caller passes a memoised `cell`. Site × month grids are
 * components/SiteMonthMatrix, built on this one.
 */
export function HeatMatrix(p: HeatMatrixProps) {
  useTheme() // cell label ink follows the theme the cells are seen in
  const wrapRef = useRef<HTMLDivElement>(null)
  const measured = useDimensions(wrapRef).width
  const tip = useTooltip(wrapRef)
  const fit = useFitHeight()
  const [hover] = useState(createHoverStore)
  const bgRef = useRef<SVGRectElement>(null)

  const R = useMemo(() => p.rows.map(asItem), [p.rows])
  const C = useMemo(() => p.columns.map(asItem), [p.columns])
  const nR = R.length, nC = C.length
  const avail = p.width ?? measured
  const [gx, gy] = Array.isArray(p.gap) ? p.gap : [p.gap ?? 1, p.gap ?? 1]
  const rowSize = p.rowLabelSize ?? 11, colSize = p.colLabelSize ?? 11, cellSize = p.labelSize ?? 10
  const maxLabel = p.maxLabelWidth ?? 220
  const pos = p.colLabelPosition ?? 'top'

  // ---- the left gutter: group labels, row labels (measured), extra columns
  const rowsHaveDots = R.some((r) => r.dot)
  const align = p.rowLabelAlign ?? (rowsHaveDots ? 'start' : 'end')
  const rowText = useMemo(() => R.map((r) => {
    const size = r.size ?? rowSize, weight = r.weight ?? 400, label = labelOf(r)
    const extra = (r.dot ? 14 : 0) + (r.suffix ? 4 + monoWidth(r.suffix, size * 0.9) : 0)
    const family = r.mono ? FONT.mono : FONT.family
    const full = label ? textWidth(label, size, weight, family) : 0
    const shown = full + extra > maxLabel ? cut(label, size, maxLabel - extra, weight, family) : label
    return { shown, cut: shown !== label, w: Math.min(maxLabel, full + extra) }
  }), [R, rowSize, maxLabel])
  const labelW = Math.max(0, ...rowText.map((t) => t.w), p.rowHeader ? textWidth(p.rowHeader, 11) : 0)
  const groupW = p.groupLabels ? Math.max(0, ...R.map((r) => (r.group ? textWidth(r.group, rowSize, 600) : 0))) + 10 : 0
  const extras = (p.extraColumns ?? []).map((e) => ({
    ...e, side: e.side ?? 'left',
    w: Math.max(e.header ? textWidth(e.header, 10.5) : 0, ...R.map((_, i) => monoWidth(e.text(i), 11))) + 12,
  }))
  const leftExtras = extras.filter((e) => e.side === 'left'), rightExtras = extras.filter((e) => e.side === 'right')
  const leftExtraW = leftExtras.reduce((a, e) => a + e.w, 0), rightExtraW = rightExtras.reduce((a, e) => a + e.w, 0)
  const labelRight = groupW + labelW
  const cellsX = labelRight + (labelW || groupW ? PAD : 0) + leftExtraW

  // ---- cell width, then how the column labels fit it
  const colW = useMemo(() => C.map((c) => (labelOf(c) ? textWidth(labelOf(c), c.size ?? colSize, c.weight ?? p.colLabelWeight ?? 400) : 0)), [C, colSize, p.colLabelWeight])
  const maxColW = Math.max(0, ...colW)
  const angle = ((p.colLabelAngle ?? -40) * Math.PI) / 180
  const widthFor = (overflow: number) => typeof p.cellWidth === 'number' ? p.cellWidth
    : clamp((avail - cellsX - rightExtraW - overflow - 4) / Math.max(1, nC), (p.cellWidth ?? [24, 64])[0], (p.cellWidth ?? [24, 64])[1])
  let cw = widthFor(0)
  const modeFor = (w: number): NonNullable<HeatMatrixProps['colLabels']> => {
    const m = p.colLabels ?? 'auto'
    if (m !== 'auto' && m !== 'slant') return m
    if (maxColW <= w - 4) return 'horizontal'
    if (m === 'slant') return 'rotate'
    if (C.every((c) => twoLines(labelOf(c)).every((l) => textWidth(l, c.size ?? colSize, c.weight ?? p.colLabelWeight ?? 400) <= w - 4))) return 'wrap'
    return 'rotate'
  }
  let mode = modeFor(cw)
  const overflowOf = (w: number, m: string) => (m === 'rotate' && pos === 'top' ? Math.max(0, maxColW * Math.cos(angle) - w / 2) : 0)
  if (typeof p.cellWidth !== 'number' && overflowOf(cw, mode) > 0) { cw = widthFor(overflowOf(cw, mode)); mode = modeFor(cw) }
  const rightOverflow = overflowOf(cw, mode)

  // ---- the header band (or the band under the cells)
  const colGroups = useMemo(() => groupStarts(C.map((c) => c.group)), [C])
  const labelBand = mode === 'none' ? 0 : mode === 'wrap' ? 2 * (colSize + 2) + 8 : mode === 'rotate' ? maxColW * Math.abs(Math.sin(angle)) + colSize * Math.cos(angle) + 8 : colSize + 9
  const groupBand = colGroups.length ? colSize + 7 : 0
  const headerBand = p.colHeader ? 16 : 0
  const titleBand = p.title != null ? 20 : 0
  const extraHeader = extras.some((e) => e.header) || p.rowHeader ? 16 : 0
  const before = p.before ?? 0
  const topBands = pos === 'top' ? Math.max(labelBand + groupBand + headerBand, extraHeader) : extraHeader
  const top = before + titleBand + topBands
  const bottomBands = pos === 'bottom' ? 4 + labelBand + groupBand + headerBand : 0

  // ---- row heights: fixed, square, or fitted to a tile
  let ch = p.cellHeight === 'square' ? cw : p.cellHeight ?? 26
  const groupGap = p.groupGap ?? 8
  const nGaps = useMemo(() => R.reduce((n, r, i) => n + (i > 0 && r.group !== R[i - 1].group ? 1 : 0), 0), [R])
  if (fit !== null && p.fitRows !== false && nR > 0) {
    const [mn, mx] = p.fitRows ?? [Math.min(10, ch), ch]
    ch = clamp((fit - top - bottomBands - (p.after ?? 0) - nGaps * groupGap) / nR, mn, mx)
  }
  const { tops, height: innerH } = useMemo(() => rowTops(R.map((r) => r.group), ch, groupGap), [R, ch, groupGap])
  const innerW = nC * cw
  const width = cellsX + innerW + rightExtraW + rightOverflow + 4
  const height = top + innerH + bottomBands + (p.after ?? 0) + 2
  const layout: MatrixLayout = { left: cellsX, top, cellW: cw, cellH: ch, innerW, innerH, width, height, x: (j) => cellsX + j * cw, y: (i) => top + (tops[i] ?? 0) }

  // ---- the cells: worked out and drawn again only when the caller re-renders (or the size changes)
  const { cell, color, format } = p
  const grid = useMemo(() => {
    const g: (MatrixCell | undefined)[][] = []
    for (let i = 0; i < nR; i++) { const row: (MatrixCell | undefined)[] = []; for (let j = 0; j < nC; j++) row.push(norm(cell(i, j))); g.push(row) }
    return g
  }, [cell, nR, nC])
  const cellLabels = p.cellLabels ?? 'auto'
  const cells = useMemo(() => {
    const out: ReactNode[] = []
    const w = cw - gx, h = ch - gy
    for (let i = 0; i < nR; i++) for (let j = 0; j < nC; j++) {
      const c = grid[i][j]
      if (!c) continue
      const fill = c.fill !== undefined ? c.fill : c.value == null || !color ? null : color(c.value)
      let label = c.label !== undefined ? c.label : c.value == null || !format ? null : format(c.value)
      let sub = c.sublabel ?? null
      if (cellLabels === 'never' || (cellLabels === 'auto' && label != null && (monoWidth(label, cellSize) > w - 4 || h < cellSize + 2))) label = null
      if (sub != null && (label == null || h < cellSize * 2.4 || (cellLabels === 'auto' && monoWidth(sub, cellSize * 0.82) > w - 4))) sub = null
      out.push(<HeatCellBody key={`${i}:${j}`} x={j * cw} y={tops[i]} width={w} height={h} fill={fill} label={label} sublabel={sub} fontSize={cellSize} fontWeight={p.labelWeight}
        outline={c.outline} stroke={c.stroke} strokeWidth={c.strokeWidth} opacity={c.opacity} ink={c.ink} radius={p.radius} rectProps={c.props} />)
    }
    return out
  }, [grid, cw, ch, gx, gy, tops, color, format, cellLabels, cellSize, p.labelWeight, p.radius, nR, nC]) // eslint-disable-line react-hooks/exhaustive-deps

  // ---- one pointer handler for every cell
  const at = (e: MouseEvent): [number, number] | null => {
    const box = bgRef.current?.getBoundingClientRect()
    if (!box) return null
    const i = rowAt(tops, ch, e.clientY - box.top), j = colAt(nC, cw, e.clientX - box.left)
    return i < 0 || j < 0 || !grid[i]?.[j] ? null : [i, j]
  }
  const tipOf = (i: number, j: number): TipBody => {
    const own = p.tip?.(i, j)
    if (own) return own
    const c = grid[i][j]!
    if (c.tip) return c.tip
    const r = R[i], k = C[j]
    return [`${r.title ?? labelOf(r)} · ${k.title ?? labelOf(k)}`, c.label ?? (c.value == null ? 'no value' : format ? format(c.value) : String(c.value))]
  }
  const onMove = (e: MouseEvent) => {
    const ij = at(e)
    if (!ij) { hover.set(null); tip.hide(); return }
    hover.set(ij)
    tip.show(e, tipOf(ij[0], ij[1]))
  }
  const onLeave = () => { hover.set(null); tip.hide() }
  const clickable = !!(p.onCell || p.onRow)
  const onClick = (e: MouseEvent) => {
    const ij = at(e)
    if (!ij) return
    tip.hide()
    if (p.onCell) p.onCell(ij[0], ij[1])
    else p.onRow?.(ij[0])
  }

  // ---- labels
  const rowLabelY = (i: number) => top + tops[i] + (ch - gy) / 2
  const colX = (j: number) => cellsX + j * cw + (p.colLabelAlign === 'start' ? 0 : cw / 2)
  const colAnchor = p.colLabelAlign === 'start' ? 'start' : 'middle'
  const bandTop = pos === 'top' ? before + titleBand : top + innerH + 4
  // top: header title, group labels, then labels sitting on the cells; bottom: labels under the cells, then groups, then the title
  const yHeader = pos === 'top' ? bandTop + 12 : bandTop + labelBand + groupBand + 12
  const yGroups = pos === 'top' ? bandTop + headerBand + colSize : bandTop + labelBand + colSize
  const yLabels = pos === 'top' ? top - 8 : bandTop + colSize
  const groupLabelsDrawn = useMemo(() => {
    let lastEnd = -Infinity
    return colGroups.flatMap((g) => {
      const x = cellsX + g.index * cw + 2
      if (x < lastEnd) return []
      lastEnd = x + textWidth(g.group, colSize) + 8
      return [{ ...g, x }]
    })
  }, [colGroups, cellsX, cw, colSize])
  const showTip = (body?: TipBody) => body ? { onMouseMove: (e: MouseEvent) => tip.show(e, body), onMouseLeave: tip.hide } : {}

  const colLabel = (c: MatrixItem, j: number) => {
    const text = labelOf(c)
    if (!text) return null
    const size = c.size ?? colSize, weight = c.weight ?? p.colLabelWeight ?? 400, fill = c.color ?? p.colLabelColor ?? INK.text
    const common = { fontSize: size, fontWeight: weight, fill, fontStyle: c.italic ? 'italic' : undefined, ...showTip(c.tip ?? (c.title ? [c.title] : undefined)) }
    const x = colX(j)
    if (mode === 'rotate') {
      const y = pos === 'top' ? top - 6 : bandTop + 2
      return <text key={c.key} transform={`translate(${x},${y}) rotate(${p.colLabelAngle ?? -40})`} textAnchor={pos === 'top' ? 'start' : 'end'} dy={pos === 'top' ? 0 : '0.7em'} {...common}>{text}</text>
    }
    if (mode === 'wrap') {
      const lines = colW[j] > cw - 4 ? twoLines(text) : [text]
      const y0 = pos === 'top' ? yLabels - (lines.length - 1) * (size + 2) : yLabels
      return <text key={c.key} x={x} y={y0} textAnchor={colAnchor} {...common}>{lines.map((l, k) => <tspan key={k} x={x} dy={k ? size + 2 : 0}>{l}</tspan>)}</text>
    }
    let shown = text
    if (mode === 'fit' && colW[j] > cw - 2) shown = c.short && textWidth(c.short, size, weight) <= cw - 2 ? c.short : ''
    return shown ? <text key={c.key} x={x} y={yLabels} textAnchor={colAnchor} {...common}>{shown}</text> : null
  }

  const svgStyle = p.center ? { display: 'block', marginLeft: `max(0px, calc(50% - ${cellsX + innerW / 2}px))` } : undefined
  const hasLegend = !!(p.legend || p.legendItems?.length || p.legendExtra)
  const legendScale = p.legend?.scale ?? (color && 'domain' in color ? color as unknown as MatrixLegend['scale'] : undefined)
  let extraX = labelRight + (labelW || groupW ? PAD : 0)
  const extraPos = extras.map((e) => {
    if (e.side === 'left') { const x = extraX; extraX += e.w; return x }
    return 0
  })
  let rightX = cellsX + innerW + PAD
  extras.forEach((e, k) => { if (e.side === 'right') { extraPos[k] = rightX; rightX += e.w } })

  return (
    <div ref={wrapRef} className={`chart-wrap${p.center ? ' centered' : ''}`}>
      {nR === 0 || nC === 0 ? null : (
        <svg width={width} height={height} fontFamily={FONT.family} style={svgStyle} role="img" aria-label={p.ariaLabel ?? `${nR} rows by ${nC} columns`}>
          {p.selectedRow != null && p.selectedRow >= 0 && p.selectedRow < nR && <rect x={0} y={top + tops[p.selectedRow]} width={width} height={ch - gy} fill={INK.tint} />}
          {p.title != null && <text x={cellsX} y={before + 13} fontSize={12} fontWeight={600} fill={p.titleColor ?? INK.text}>{p.title}</text>}
          {p.colHeader && <text x={cellsX + innerW / 2} y={yHeader} textAnchor="middle" fontSize={11} fill={INK.muted}>{p.colHeader}</text>}
          {p.rowHeader && <text x={labelRight} y={top - 8} textAnchor="end" fontSize={11} fill={INK.muted}>{p.rowHeader}</text>}
          {groupLabelsDrawn.map((g) => <text key={g.group + g.index} x={g.x} y={yGroups} fontSize={colSize} fill={INK.text}>{g.group}</text>)}
          {C.map(colLabel)}
          {extras.map((e, k) => e.header && <text key={e.key} x={extraPos[k] + 4} y={top - 8} fontSize={10.5} fill={INK.muted}>{e.header}</text>)}
          {R.map((r, i) => {
            const t = rowText[i], size = r.size ?? rowSize, on = p.selectedRow === i
            const x = align === 'start' ? groupW + (r.dot ? 14 : 0) : labelRight
            const rowTip = r.tip ?? (t.cut ? [r.title ?? labelOf(r)] : undefined)
            return <g key={r.key} style={p.onRow ? { cursor: 'pointer' } : undefined} onClick={p.onRow ? () => p.onRow!(i) : undefined} {...showTip(rowTip)}>
              {r.dot && <circle cx={groupW + 5} cy={rowLabelY(i)} r={5} fill={r.dot} />}
              {t.shown && <text x={x} y={rowLabelY(i)} dy="0.35em" textAnchor={align} fontSize={size} fontWeight={on ? 650 : r.weight} fontStyle={r.italic ? 'italic' : undefined} fontFamily={r.mono ? FONT.mono : undefined} fill={r.color ?? INK.text}>
                {t.shown}{r.suffix && <tspan fill={INK.muted} fontFamily={FONT.mono} fontSize={size * 0.9}> {r.suffix}</tspan>}
              </text>}
              {extras.map((e, k) => <text key={e.key} x={extraPos[k] + 4} y={rowLabelY(i)} dy="0.35em" fontSize={11} fontFamily={FONT.mono} fill={e.color ?? INK.muted}>{e.text(i)}</text>)}
            </g>
          })}
          {p.groupLabels && groupStarts(R.map((r) => r.group)).map((g) => <text key={g.group + g.index} x={0} y={top + tops[g.index] + Math.min(ch, 13) - 1} fontSize={rowSize} fontWeight={600} fill={INK.text}>{g.group}</text>)}
          <g transform={`translate(${cellsX},${top})`} onMouseMove={onMove} onMouseLeave={onLeave} onClick={clickable ? onClick : undefined}
            style={clickable ? { cursor: 'pointer' } : undefined} shapeRendering={p.crisp ? 'crispEdges' : undefined}>
            <rect ref={bgRef} width={innerW} height={innerH} fill="transparent" />
            {cells}
          </g>
          {p.colGroupRules && colGroups.filter((g) => g.index > 0).map((g) => <line key={g.index} x1={cellsX + g.index * cw} x2={cellsX + g.index * cw} y1={top} y2={top + innerH} stroke={INK.axis} strokeOpacity={0.45} pointerEvents="none" />)}
          {p.hover !== 'none' && <HoverRing store={hover} layout={layout} tops={tops} gx={gx} gy={gy} cross={p.hover === 'cross'} />}
          {p.children?.(layout)}
        </svg>
      )}
      {hasLegend && <Key items={p.legendItems ?? []}>
        {p.legend && legendScale && <ColorLegend scale={legendScale} scheme={p.legend.scheme} label={p.legend.label} format={p.legend.format} note={p.legend.note} ticks={p.legend.ticks} width={p.legend.width} />}
        {p.legendExtra}
      </Key>}
      {p.footer}
      {tip.node}
    </div>
  )
}

type HoverStore = ReturnType<typeof createHoverStore>
function createHoverStore() {
  let at: [number, number] | null = null
  const listeners = new Set<() => void>()
  return {
    get: () => at,
    set: (next: [number, number] | null) => {
      if (next === at || (next && at && next[0] === at[0] && next[1] === at[1])) return
      at = next
      for (const l of listeners) l()
    },
    subscribe: (l: () => void) => { listeners.add(l); return () => { listeners.delete(l) } },
  }
}

/** the ring on the pointed cell (or its row and column): the one part that redraws as the pointer moves */
function HoverRing({ store, layout, tops, gx, gy, cross }: { store: HoverStore; layout: MatrixLayout; tops: number[]; gx: number; gy: number; cross: boolean }) {
  const at = useSyncExternalStore(store.subscribe, store.get, store.get)
  if (!at) return null
  const [i, j] = at
  const x = layout.left + j * layout.cellW, y = layout.top + (tops[i] ?? 0)
  return <g fill="none" stroke={INK.text} strokeWidth={1} pointerEvents="none">
    {cross
      ? <><rect x={layout.left} y={y} width={layout.innerW} height={layout.cellH} /><rect x={x} y={layout.top} width={layout.cellW} height={layout.innerH} /></>
      : <rect x={x - 0.5} y={y - 0.5} width={layout.cellW - gx + 1} height={layout.cellH - gy + 1} strokeWidth={1.5} />}
  </g>
}
