import { useMemo } from 'react'
import * as d3 from 'd3'
import { ScatterLayer } from '@/components/ScatterLayer'
import { StackedPanels, type StackMarker, type StackPanel } from '@/components/StackedPanels'
import type { TipBody, useTooltip } from '@/hooks/useTooltip'
import { fidProps } from '@/lib/filterEdits'
import { focusStyle } from '@/lib/highlight'
import { toMarks, useMarkReason, useMarkedRows } from '@/lib/markedRows'
import { nearestIndex } from '@/lib/stacked'
import { fmt } from '@/lib/stats'
import { useLoaded } from '@/lib/data'
import { namedColor } from '@/lib/siteColours'
import { INK } from '@/lib/theme'
import type { FilterRow, MetaFile, SeasonMeta } from '@/lib/types'

/**
 * One site's daily record (AERONET, ERA5 weather) in lanes above the filters' own measurements, on the filters'
 * dates: the AERONET timeline and the weather timeline. Built on components/StackedPanels, with what both
 * share defined here once: the loader (useSiteDaily), the window (the site's filters with a fortnight either
 * side), the daily lines and y ranges, the measurement lanes (FieldLane, through ScatterLayer) and the cursor
 * that snaps to the nearest day (dailyReadout).
 */

/** the days of a daily file inside the window: `i` indexes the file's arrays */
export interface Day { t: Date; i: number }
type Daily = { date: string[] }

/**
 * The site's daily file for the subset's filters: loaded once per site (and whatever `load` depends on, so
 * memoise it), a load that resolves after the site changed is dropped, and a missing file is `error`.
 * `domain` is the site's filters' span ± 14 days (null: no filters there), `days` the file's days inside it.
 */
export function useSiteDaily<D extends Daily>(rows: readonly FilterRow[], site: { name: string; code: string } | null,
  load: (code: string) => Promise<D>, missing: (site: string) => string) {
  // lib/data useLoaded: one load per site code (and per `load`), a late answer for another site dropped
  const loaded = useLoaded(site ? site.code : null, () => load(site!.code), { deps: [load] })
  const daily = loaded.data
  const error = loaded.error && site ? missing(site.name) : ''
  const domain = useMemo<[Date, Date] | null>(() => {
    const ds = rows.filter((r) => r.site === site?.name).map((r) => new Date(r.date).getTime())
    if (!ds.length) return null
    return [new Date(Math.min(...ds) - 14 * 864e5), new Date(Math.max(...ds) + 14 * 864e5)]
  }, [rows, site])
  const days = useMemo<Day[]>(() => {
    if (!daily || !domain) return []
    return daily.date.map((d, i) => ({ t: new Date(d), i })).filter((x) => x.t >= domain[0] && x.t <= domain[1])
  }, [daily, domain])
  return { daily, error, domain, days }
}

const val = (arr: readonly (number | null)[] | undefined, i: number) => arr?.[i] ?? null

/** a lane's y range over the days in view: from `lo` (else the smallest) to `hi` (else the largest); StackY nice rounds it */
export function dailyDomain(days: readonly Day[], arrs: readonly (readonly (number | null)[] | undefined)[], { lo, hi }: { lo?: number; hi?: number } = {}): [number, number] {
  const vs = days.flatMap(({ i }) => arrs.map((a) => val(a, i))).filter((v): v is number => v !== null)
  const a = lo ?? d3.min(vs) ?? 0, b = hi ?? d3.max(vs) ?? 1
  return [a, b === a ? a + 1 : b]
}

/** a daily series as one path, broken where a day has no value */
export function dailyLine(days: readonly Day[], arr: readonly (number | null)[] | undefined, x: d3.ScaleTime<number, number>, y: d3.ScaleLinear<number, number>) {
  return d3.line<Day>().defined(({ i }) => val(arr, i) !== null).x(({ t }) => x(t)).y(({ i }) => y(val(arr, i)!))(days as Day[]) ?? ''
}

/** the cursor snapped to the nearest day, with the chart's tooltip for it */
export const dailyReadout = (days: readonly Day[], body: (d: Day) => TipBody) => (px: number, x: d3.ScaleTime<number, number>) => {
  const d = days[nearestIndex(days, (q) => +q.t, +x.invert(px))]
  return d ? { at: x(d.t), tip: body(d) } : null
}

export type FieldPt = { row: FilterRow; t: Date; v: number }
export const NO_POINTS: FieldPt[] = []

/** one site's filters that carry `field`, as dated points */
export const fieldPoints = (rows: readonly FilterRow[], site: string | undefined, field: string): FieldPt[] =>
  rows.filter((r) => r.site === site && typeof r[field] === 'number').map((r) => ({ row: r, t: new Date(r.date), v: r[field] as number }))

/** a measurement lane's y: from 0 (or below) to just over its largest value; StackY nice rounds it */
export const fieldDomain = (pts: readonly FieldPt[]): [number, number] =>
  [Math.min(0, d3.min(pts, (p) => p.v) ?? 0), (d3.max(pts, (p) => p.v) ?? 1) * 1.05]

/**
 * A measurement lane's filters, coloured by season (components/ScatterLayer): drawn once per data, size or brush, so
 * the date cursor does not redraw them; the nearest filter answers a hover, a click opens its record, and the site's
 * excluded filters show as × (Filters → excluded: Mark).
 */
export function FieldLane({ pts, field, site, x, y, width, height, seasons, selected, tip, tipOf }: {
  pts: FieldPt[]
  field: string; site: string | undefined
  x: d3.ScaleTime<number, number>; y: d3.ScaleLinear<number, number>; width: number; height: number; seasons: MetaFile['seasons']
  selected: Set<string> | null; tip: ReturnType<typeof useTooltip>; tipOf: (p: FieldPt) => TipBody
}) {
  const colorOf = (p: FieldPt) => namedColor(seasons, p.row.season, INK.text)
  const xy = useMemo(() => pts.map((p) => [x(p.t), y(p.v)] as const), [pts, x, y])
  const dots = useMemo(() => pts.map((p, i) => {
    const st = focusStyle(p.row.id, null, { r: 3, level: 'sparse' }, selected)
    return <circle key={p.row.id} cx={xy[i][0]} cy={xy[i][1]} r={st.r} fill={colorOf(p)}
      fillOpacity={st.opacity} stroke={st.stroke} strokeWidth={st.strokeWidth} data-dot={st.dot} {...fidProps(p.row.id)} />
  }), [pts, xy, selected, seasons]) // eslint-disable-line react-hooks/exhaustive-deps
  const focused = (p: FieldPt) => focusStyle(p.row.id, p.row.id, { r: 3, level: 'sparse' })
  const markedAll = useMarkedRows()
  const markReason = useMarkReason()
  const marks = useMemo(() => {
    const [d0, d1] = x.domain(), [v0, v1] = y.domain()
    return toMarks(markedAll.filter((r) => r.site === site), (r) => {
      const v = r[field], t = new Date(r.date)
      return typeof v === 'number' && t >= d0 && t <= d1 && v >= v0 && v <= v1 ? [x(t), y(v)] : null
    }, markReason, (r) => [`${field}: ${fmt(r[field] as number)}`])
  }, [markedAll, site, field, x, y, markReason])
  return (
    <ScatterLayer items={pts} xy={xy} idOf={(p) => p.row.id} dots={dots} width={width} height={height} tip={tip} tipOf={tipOf} marks={marks}
      focusR={(p) => focused(p).r}
      focus={(p, [cx, cy]) => { const st = focused(p); return <circle cx={cx} cy={cy} r={st.r} fill={colorOf(p)} fillOpacity={st.opacity} stroke={st.stroke} strokeWidth={st.strokeWidth} /> }} />
  )
}

/** the lanes on the window's time axis, with the site's seasons shaded behind each and the cursor snapping to the nearest day */
export function DailyLanes({ domain, lanes, seasons, readout, left = 68, top = 8, gap = 14, xMarkers }: {
  domain: [Date, Date]
  lanes: readonly StackPanel<d3.ScaleTime<number, number>>[]
  seasons: readonly SeasonMeta[]
  readout: (px: number, x: d3.ScaleTime<number, number>) => { at: number; tip: TipBody } | null
  left?: number | ((heights: readonly number[]) => number)
  top?: number
  gap?: number
  xMarkers?: readonly StackMarker[]
}) {
  return <StackedPanels x={{ type: 'time', domain }} panels={lanes} seasons={seasons} readout={readout} xMarkers={xMarkers}
    margin={{ top, right: 20, bottom: 30, left }} gap={gap} panelHeight={92} />
}
