import { useMemo, type ComponentProps, type ReactNode } from 'react'
import * as d3 from 'd3'
import { useExport } from '@/lib/data'
import { ExportState } from '@/components/ExportState'
import { IntervalStrip, type IntervalBlock } from '@/components/IntervalStrip'
import { ScatterPlot } from '@/components/ScatterPlot'
import { INK } from '@/lib/theme'
import { useCountedFilter, useExclusionSplit } from '@/lib/markedRows'

/** Helpers shared by the Experimental subpages (gallery/data/export_experiments.py exports). */

export type Cols = Record<string, (number | string | boolean | null)[]>
export interface XpBase { schema_version: number; generated: string; source: string; settings: Record<string, any> }

/** one subpage's export, or the standard "run the exporter" panel while it loads or is missing */
export function useXp<T extends XpBase>(name: string, title: string): { data: T | null; state: ReactNode } {
  const { data, error } = useExport<T>(`experimental/${name}.json`, (d) => d.schema_version === 1 ? null : 'unsupported data version')
  return { data, state: data ? null : <ExportState title={title} what={`the ${title.toLowerCase()} results`} error={error} command="uv run --no-sync python gallery/data/export_experiments.py" /> }
}

/**
 * A study's own per-filter items under Filters → excluded (lib/markedRows): `counted` by the one rule
 * (useCountedFilter: the export's exclusions unless Include, hand exclusions, keep=) feed every statistic, line and
 * count; `marked` are the ones Mark draws as ×; `nOut` how many are left out. Ids the gallery does not know (IMPROVE,
 * blanks, the Adama pilot) always count. Memoised on `items`.
 */
export function useCounted<T>(items: readonly T[], idOf: (t: T) => string | null | undefined): { counted: T[]; marked: T[]; nOut: number } {
  const split = useExclusionSplit(items, idOf)
  const isCounted = useCountedFilter()
  return useMemo(() => {
    const counted = items.filter((t) => isCounted(idOf(t)))
    return { counted, marked: split.marked, nOut: items.length - counted.length }
  }, [items, split, isCounted]) // eslint-disable-line react-hooks/exhaustive-deps
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

/** `id` links hover and pin across charts and follows Filters → excluded; `rec` opens the filter's record on click. */
export type { ScatterPoint } from '@/components/ScatterPlot'

/**
 * A plain scatter for two different quantities: components/ScatterPlot with this page's names (`xLog`/`yLog`,
 * `under`/`over`). Square and centred like the Crossplot (`maxSide` caps the side); points with an `id` hover, pin,
 * open their record and follow Filters → excluded; `pick` makes the axis titles open the nearest axis picker.
 */
export function Scatter({ xLog = false, yLog = false, over, ...rest }: Omit<ComponentProps<typeof ScatterPlot>, 'xScale' | 'yScale' | 'children'> & {
  xLog?: boolean; yLog?: boolean; over?: ComponentProps<typeof ScatterPlot>['children']
}) {
  return <ScatterPlot {...rest} xScale={xLog ? 'log' : 'linear'} yScale={yLog ? 'log' : 'linear'} outside="draw">{over}</ScatterPlot>
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
