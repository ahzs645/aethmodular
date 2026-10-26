import { useMemo, useRef, type ReactNode } from 'react'
import * as d3 from 'd3'
import { useExport } from '@/lib/data'
import { ExportState } from '@/components/ExportState'
import { XAxis, YAxis } from '@/components/Axes'
import { IdentityLine } from '@/components/RefLines'
import { IntervalStrip, type IntervalBlock } from '@/components/IntervalStrip'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { FONT, INK } from '@/lib/theme'
import { fidProps, useHandEdits } from '@/lib/filterEdits'
import { baseFilterId, useHighlight, type FilterRecord } from '@/lib/highlight'

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

export interface ForestRow { key: string; label: string; color?: string; hollow?: boolean; bold?: boolean; values: Record<string, { v: number; lo?: number; hi?: number } | undefined>; tip?: string[]; note?: string }
/** `fmt` writes the values in tooltips; `tick` the axis ticks (default: the strip's short numbers) */
export interface ForestPanel { key: string; label: string; ref?: number; fmt?: (v: number) => string; tick?: (v: number) => string; domain?: [number, number] }

/**
 * Rows of estimates, one panel per measure, side by side (a forest plot): a dot per row, an optional
 * interval, a reference line per panel. Drawn by the gallery's IntervalStrip, so it looks like every other
 * dot-interval chart and stacks its panels on a phone instead of scrolling sideways.
 */
export function Forest({ rows, panels, labelWidth = 200, rowHeight = 24, noteWidth = 0 }: { rows: ForestRow[]; panels: ForestPanel[]; labelWidth?: number; rowHeight?: number; noteWidth?: number }) {
  const fmt = (p: ForestPanel) => p.fmt ?? ((v: number) => String(+v.toPrecision(3)))
  const line = (p: ForestPanel, e: { v: number; lo?: number; hi?: number }) => `${p.label}: ${fmt(p)(e.v)}${e.lo != null && e.hi != null ? ` (${fmt(p)(e.lo)} to ${fmt(p)(e.hi)})` : ''}`
  const blocks: IntervalBlock[] = [{ key: 'rows', rows: rows.map((r) => ({
    key: r.key, label: r.label, strong: r.bold, note: r.note,
    tip: r.tip ?? [r.label, ...panels.flatMap((p) => { const e = r.values[p.key]; return e && Number.isFinite(e.v) ? [line(p, e)] : [] })],
    marks: panels.map((p) => { const e = r.values[p.key]
      return e && Number.isFinite(e.v) ? [{ mid: e.v, lo: e.lo, hi: e.hi, color: r.color ?? INK.accent, shape: r.hollow ? 'hollow' as const : 'dot' as const }] : [] }),
  })) }]
  return <IntervalStrip blocks={blocks} labelWidth={labelWidth} rowHeight={rowHeight} minPanelWidth={200} noteWidth={noteWidth}
    panels={panels.map((p) => ({ label: p.label, reference: p.ref, domain: p.domain, format: p.tick }))} />
}

/** `id` links hover and pin across charts and honours hand exclusions; `rec` opens the filter's record on click. */
export interface ScatterPoint { x: number; y: number; color: string; r?: number; hollow?: boolean; tip?: string[]; key?: string; id?: string; rec?: FilterRecord }
const NO_KEYS: ReadonlySet<string> = new Set()

/**
 * A plain scatter for two different quantities (log axes, extra layers under and over the points). The plot
 * area is square and centred, sized like the shared Crossplot (`maxSide` caps the side); a 1:1 comparison of
 * two measurements of the same thing uses components/Crossplot instead, for its fits. Points with an `id` behave
 * like the Crossplot's filters (hover, pin, record, right-click exclusion); `pick` makes the axis titles open the
 * nearest axis picker (components/AxisChoices).
 */
export function Scatter({ points, xLabel, yLabel, maxSide = 420, xLog = false, yLog = false, identity = false, xDomain, yDomain, under, over, pick = false }: {
  points: ScatterPoint[]; xLabel: string; yLabel: string; maxSide?: number; xLog?: boolean; yLog?: boolean; identity?: boolean; pick?: boolean
  xDomain?: [number, number]; yDomain?: [number, number]
  under?: (s: { x: d3.ScaleContinuousNumeric<number, number>; y: d3.ScaleContinuousNumeric<number, number>; iw: number; ih: number }) => ReactNode
  over?: (s: { x: d3.ScaleContinuousNumeric<number, number>; y: d3.ScaleContinuousNumeric<number, number>; iw: number; ih: number }) => ReactNode
}) {
  const ref = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(ref)
  const tip = useTooltip(ref)
  const hl = useHighlight(NO_KEYS)
  const edits = useHandEdits()
  const m = { top: 14, right: 18, bottom: 46, left: 62 }
  const side = Math.max(200, Math.min((width || maxSide + m.left + m.right) - m.left - m.right, maxSide))
  const iw = side, ih = side
  const w = iw + m.left + m.right, h = ih + m.top + m.bottom
  const drawable = points.filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y) && (!xLog || p.x > 0) && (!yLog || p.y > 0))
  const ok = drawable.filter((p) => !edits.isOut(p.id))
  const keys = useMemo(() => new Set(ok.flatMap((p) => (p.id ? [baseFilterId(p.id)] : []))), [ok])
  const { hoverId } = useHighlight(keys)
  const active = [hl.pinnedId, hoverId].some((k) => !!k && keys.has(k))
  const handNote = edits.note(drawable.length - ok.length)
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
          <YAxis scale={y} x={0} label={yLabel} tickCount={5} gridWidth={iw} labelX={-48} format={fmt(yLog)} pick={pick ? 'y' : undefined} />
          <XAxis scale={x} y={ih} label={xLabel} tickCount={Math.max(3, Math.floor(iw / 80))} labelDy={38} format={fmt(xLog)} pick={pick ? 'x' : undefined} />
          {keys.size > 0 && <rect width={iw} height={ih} fill="transparent" onClick={() => hl.setPinned(null)} />}
          {identity && (() => { const [a, b] = x.domain(); return <IdentityLine x1={x(a)} y1={y(a)} x2={x(b)} y2={y(b)} /> })()}
          {under?.({ x, y, iw, ih })}
          {ok.map((p, i) => {
            const key = p.id ? baseFilterId(p.id) : null
            const focus = !!key && (key === hl.pinnedId || key === hoverId)
            const r = p.r ?? 3.5
            return <circle key={p.key ?? i} cx={x(p.x)} cy={y(p.y)} r={focus ? Math.max(r, 6) : r} fill={p.hollow ? 'none' : p.color}
              fillOpacity={active && !focus ? 0.25 : 0.8} stroke={focus ? INK.text : p.hollow ? p.color : INK.panel} strokeWidth={focus ? 2 : p.hollow ? 1.5 : 0.6} style={{ cursor: p.rec ? 'pointer' : 'default' }}
              onMouseEnter={(e) => { if (p.tip) tip.show(e, p.rec ? [...p.tip, 'click to pin it on every chart and open its record'] : p.tip); if (key) hl.setHover(key) }}
              onMouseLeave={() => { tip.hide(); if (key) hl.setHover(null) }}
              {...(key ? fidProps(key) : {})}
              onClick={key ? () => { if (p.rec) hl.openRecord(p.rec); hl.setPinned(key) } : undefined} />
          })}
          {over?.({ x, y, iw, ih })}
        </g>
      </svg>
      {tip.node}
      {handNote && <p className="chart-note">{handNote}</p>}
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
