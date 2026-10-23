import { useRef, type ReactNode } from 'react'
import * as d3 from 'd3'
import { useExport } from '@/lib/data'
import { ExportState } from '@/components/ExportState'
import { XAxis, YAxis } from '@/components/Axes'
import { RefLine } from '@/components/RefLines'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { FONT, INK } from '@/lib/theme'

/** Helpers shared by the Experimental subpages (gallery/data/export_experiments.py exports). */

export type Cols = Record<string, (number | string | boolean | null)[]>
export interface XpBase { schema_version: number; generated: string; source: string; settings: Record<string, any> }

/** one subpage's export, or the standard "run the exporter" panel while it loads or is missing */
export function useXp<T extends XpBase>(name: string, title: string): { data: T | null; state: ReactNode } {
  const { data, error } = useExport<T>(`experimental/${name}.json`, (d) => d.schema_version === 1 ? null : 'unsupported data version')
  return { data, state: data ? null : <ExportState title={title} what={`the ${title.toLowerCase()} results`} error={error} command="uv run --no-sync python gallery/data/export_experiments.py" /> }
}

/** columns → row objects */
export function rowsOf<R = Record<string, any>>(c: Cols): R[] {
  const keys = Object.keys(c), n = keys.length ? c[keys[0]].length : 0
  return d3.range(n).map((i) => Object.fromEntries(keys.map((k) => [k, c[k][i]])) as R)
}

export const runLine = (d: XpBase) => `experiments_20261022.ipynb run ${d.source.split('/').pop()} · bundle ${d.settings?.bundle ?? ''} · k ${JSON.stringify(d.settings?.k_selected ?? {}).replace(/[{}"]/g, '').replace(/,/g, ', ')}`
export const sourceOf = (file: string) => (fn: string) => `gallery/app/src/pages/experimental/${file}.tsx › ${fn}`
export const num = (v: unknown) => (v == null || v === '' ? NaN : +v)

export interface ForestRow { key: string; label: string; color?: string; hollow?: boolean; bold?: boolean; values: Record<string, { v: number; lo?: number; hi?: number } | undefined>; tip?: string[] }
export interface ForestPanel { key: string; label: string; ref?: number; fmt?: (v: number) => string; domain?: [number, number] }

/**
 * Rows of estimates, one panel per measure, side by side (a forest plot): a dot per row, an optional
 * interval, a reference line per panel. Used for model-by-model comparisons on each subpage.
 */
export function Forest({ rows, panels, labelWidth = 200, rowHeight = 28 }: { rows: ForestRow[]; panels: ForestPanel[]; labelWidth?: number; rowHeight?: number }) {
  const ref = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(ref)
  const tip = useTooltip(ref)
  const gap = 36
  const panelW = Math.max(150, (Math.max(520, width) - labelWidth - gap * (panels.length - 1) - 40) / panels.length)
  const h = rows.length * rowHeight + 52
  const yb = (i: number) => 10 + i * rowHeight + rowHeight / 2
  return (
    <div ref={ref} style={{ position: 'relative', overflowX: 'auto' }}>
      <svg width={labelWidth + panels.length * panelW + (panels.length - 1) * gap + 40} height={h} fontFamily={FONT.family}>
        {rows.map((r, i) => <g key={r.key}>
          {i % 2 === 0 && <rect x={0} y={yb(i) - rowHeight / 2} width={labelWidth + panels.length * (panelW + gap)} height={rowHeight} fill={INK.zebra} />}
          <text x={labelWidth - 10} y={yb(i)} dy="0.32em" textAnchor="end" fontSize={11} fontWeight={r.bold ? 700 : 400} fill={INK.text}>{r.label}</text>
        </g>)}
        {panels.map((p, pi) => {
          const vals = rows.flatMap((r) => { const x = r.values[p.key]; return x ? [x.v, x.lo ?? x.v, x.hi ?? x.v] : [] }).filter(Number.isFinite)
          const dom = p.domain ?? (d3.extent([...vals, ...(p.ref != null ? [p.ref] : [])]) as [number, number])
          const pad = (dom[1] - dom[0]) * 0.08 || 0.1
          const x = d3.scaleLinear().domain([dom[0] - pad, dom[1] + pad]).nice().range([0, panelW])
          const fmt = p.fmt ?? ((v: number) => String(+v.toPrecision(3)))
          return <g key={p.key} transform={`translate(${labelWidth + pi * (panelW + gap)},0)`}>
            <XAxis scale={x} y={rows.length * rowHeight + 10} label={p.label} tickCount={5} labelDy={32} format={(v) => fmt(+v)} />
            {p.ref != null && <RefLine orient="v" at={x(p.ref)} from={4} to={rows.length * rowHeight + 10} />}
            {rows.map((r, i) => {
              const e = r.values[p.key]
              if (!e || !Number.isFinite(e.v)) return null
              const c = r.color ?? INK.accent
              return <g key={r.key} onMouseEnter={(ev) => tip.show(ev, r.tip ?? [r.label, `${p.label}: ${fmt(e.v)}${e.lo != null && e.hi != null ? ` (${fmt(e.lo)} to ${fmt(e.hi)})` : ''}`])} onMouseLeave={tip.hide}>
                <rect x={0} y={yb(i) - rowHeight / 2} width={panelW} height={rowHeight} fill="transparent" />
                {e.lo != null && e.hi != null && <line x1={x(e.lo)} x2={x(e.hi)} y1={yb(i)} y2={yb(i)} stroke={c} strokeWidth={2} strokeOpacity={0.6} />}
                <circle cx={x(e.v)} cy={yb(i)} r={5.5} fill={r.hollow ? INK.panel : c} stroke={c} strokeWidth={2} />
                <text x={panelW + 4} y={yb(i)} dy="0.32em" fontSize={9.5} fill={INK.muted}>{fmt(e.v)}</text>
              </g>
            })}
          </g>
        })}
      </svg>
      {tip.node}
    </div>
  )
}

export interface ScatterPoint { x: number; y: number; color: string; r?: number; hollow?: boolean; tip?: string[]; key?: string }

/**
 * A plain scatter (cross plot) with optional log axes, identity line and extra layers drawn under the points.
 * The plot area is square by default (the gallery's preference for cross plots): `height` caps its size and
 * the figure is centred in the panel; `square={false}` stretches it to the panel's width instead.
 */
export function Scatter({ points, xLabel, yLabel, height = 340, xLog = false, yLog = false, identity = false, xDomain, yDomain, under, over, fixedWidth, square = true }: {
  points: ScatterPoint[]; xLabel: string; yLabel: string; height?: number; xLog?: boolean; yLog?: boolean; identity?: boolean
  xDomain?: [number, number]; yDomain?: [number, number]; fixedWidth?: number; square?: boolean
  under?: (s: { x: d3.ScaleContinuousNumeric<number, number>; y: d3.ScaleContinuousNumeric<number, number>; iw: number; ih: number }) => ReactNode
  over?: (s: { x: d3.ScaleContinuousNumeric<number, number>; y: d3.ScaleContinuousNumeric<number, number>; iw: number; ih: number }) => ReactNode
}) {
  const ref = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(ref)
  const tip = useTooltip(ref)
  const m = { top: 12, right: 18, bottom: 46, left: 62 }
  const avail = fixedWidth ?? Math.max(300, width)
  const side = Math.max(200, Math.min(avail - m.left - m.right, height - m.top - m.bottom))
  const iw = square ? side : avail - m.left - m.right, ih = square ? side : height - m.top - m.bottom
  const w = iw + m.left + m.right, h = ih + m.top + m.bottom
  const ok = points.filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y) && (!xLog || p.x > 0) && (!yLog || p.y > 0))
  const xd = xDomain ?? (d3.extent(ok, (p) => p.x) as [number, number])
  const yd = yDomain ?? (d3.extent(ok, (p) => p.y) as [number, number])
  const lo = Math.min(xd[0], yd[0]), hi = Math.max(xd[1], yd[1])
  const x = (xLog ? d3.scaleLog() : d3.scaleLinear()).domain(identity ? [xLog ? Math.max(lo, 1e-3) : Math.min(0, lo), hi] : xd).range([0, iw]).nice()
  const y = (yLog ? d3.scaleLog() : d3.scaleLinear()).domain(identity ? [yLog ? Math.max(lo, 1e-3) : Math.min(0, lo), hi] : yd).range([ih, 0]).nice()
  // log axes label only the 1-2-5 ticks of each decade, so minor ticks stay as marks without crowding text
  const fmt = (log: boolean) => (v: any) => {
    if (!log) return String(+(+v).toPrecision(3))
    const m = Math.round(+v / 10 ** Math.floor(Math.log10(+v)))
    return [1, 2, 5].includes(m) ? (+v >= 1 ? d3.format('~s')(+v) : String(+(+v).toPrecision(1))) : ''
  }
  return (
    <div ref={ref} style={{ position: 'relative' }}>
      <svg width={w} height={h} fontFamily={FONT.family} style={{ display: 'block', margin: '0 auto' }}>
        <g transform={`translate(${m.left},${m.top})`}>
          <YAxis scale={y} x={0} label={yLabel} tickCount={5} gridWidth={iw} labelX={-48} format={fmt(yLog)} />
          <XAxis scale={x} y={ih} label={xLabel} tickCount={Math.max(3, Math.floor(iw / 80))} labelDy={38} format={fmt(xLog)} />
          {identity && (() => { const [a, b] = x.domain(); return <line x1={x(a)} y1={y(a)} x2={x(b)} y2={y(b)} stroke={INK.identity} strokeDasharray="4 3" /> })()}
          {under?.({ x, y, iw, ih })}
          {ok.map((p, i) => <circle key={p.key ?? i} cx={x(p.x)} cy={y(p.y)} r={p.r ?? 3.5} fill={p.hollow ? 'none' : p.color} fillOpacity={0.8} stroke={p.hollow ? p.color : INK.panel} strokeWidth={p.hollow ? 1.5 : 0.6}
            onMouseEnter={(e) => p.tip && tip.show(e, p.tip)} onMouseLeave={tip.hide} />)}
          {over?.({ x, y, iw, ih })}
        </g>
      </svg>
      {tip.node}
    </div>
  )
}

export interface PlacedLabel { key: string; text: string; bold: boolean; tx: number; ty: number; anchor: 'start' | 'middle' | 'end'; leader: boolean; ax: number; ay: number; lx: number; ly: number }
/**
 * Greedy label placement for circles: in the order given, try right, left, above, below, then the four
 * diagonals and a ring further out (with a leader line), keeping the first spot that overlaps no
 * circle and no label already placed and stays inside the plot. An `optional` label with no free spot
 * is dropped; any other falls back to its first spot.
 */
export function placeLabels(items: { key: string; x: number; y: number; r: number; text: string; bold: boolean; optional?: boolean }[], w: number, h: number): PlacedLabel[] {
  const boxes: [number, number, number, number][] = []
  const hit = (a: [number, number, number, number], b: [number, number, number, number]) => a[0] < b[2] && a[2] > b[0] && a[1] < b[3] && a[3] > b[1]
  const circleHit = (bx: [number, number, number, number]) => items.some((c) => {
    const nx = Math.max(bx[0], Math.min(c.x, bx[2])), ny = Math.max(bx[1], Math.min(c.y, bx[3]))
    return Math.hypot(nx - c.x, ny - c.y) < c.r - 0.5
  })
  return items.map((it): PlacedLabel | null => {
    const tw = it.text.length * (it.bold ? 6.6 : 5.6), th = it.bold ? 15 : 12
    const tries: { dx: number; dy: number; anchor: 'start' | 'middle' | 'end'; leader: boolean }[] = []
    for (const k of [1, 2.2, 3.6]) {
      const g = it.r * k + 3 + (k > 1 ? 6 : 0), dg = g * 0.75
      tries.push({ dx: g, dy: 0, anchor: 'start', leader: k > 1 }, { dx: -g, dy: 0, anchor: 'end', leader: k > 1 },
        { dx: 0, dy: -g - 3, anchor: 'middle', leader: k > 1 }, { dx: 0, dy: g + 3, anchor: 'middle', leader: k > 1 },
        { dx: dg, dy: -dg, anchor: 'start', leader: true }, { dx: -dg, dy: -dg, anchor: 'end', leader: true },
        { dx: dg, dy: dg, anchor: 'start', leader: true }, { dx: -dg, dy: dg, anchor: 'end', leader: true })
    }
    const boxOf = (t: typeof tries[number]): [number, number, number, number] => {
      const tx = it.x + t.dx, ty = it.y + t.dy
      const x0 = t.anchor === 'start' ? tx : t.anchor === 'end' ? tx - tw : tx - tw / 2
      return [x0 - 1, ty - th / 2, x0 + tw + 1, ty + th / 2]
    }
    const ok = (b: [number, number, number, number]) => b[0] >= 0 && b[2] <= w && b[1] >= 0 && b[3] <= h && !boxes.some((o) => hit(o, b)) && !circleHit(b)
    const free = tries.find((t) => ok(boxOf(t)))
    if (!free && it.optional) return null                  // an optional label with no free spot is left to the tooltip
    const pick = free ?? tries[0]
    const b = boxOf(pick)
    boxes.push(b)
    const tx = it.x + pick.dx, ty = it.y + pick.dy
    // leader from the circle's edge toward the label's nearest edge
    const lx = pick.anchor === 'start' ? b[0] : pick.anchor === 'end' ? b[2] : tx, ly = pick.anchor === 'middle' ? (pick.dy < 0 ? b[3] : b[1]) : ty
    const ang = Math.atan2(ly - it.y, lx - it.x)
    return { key: it.key, text: it.text, bold: it.bold, tx, ty, anchor: pick.anchor, leader: pick.leader, ax: it.x + Math.cos(ang) * it.r, ay: it.y + Math.sin(ang) * it.r, lx, ly }
  }).filter((l): l is PlacedLabel => l !== null)
}
