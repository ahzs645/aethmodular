import { useId, useMemo } from 'react'
import * as d3 from 'd3'
import { FacetGrid } from '@/components/FacetGrid'
import { PANEL_TITLE_H } from '@/components/Panels'
import { ScrubField } from '@/components/ScrubField'
import { UndoIcon } from '@/components/Icons'
import { XAxis, YAxis } from '@/components/Axes'
import { CategoryAxis, categoryAxisHeight, type CategoryLabel, type CategorySpans } from '@/components/CategoryAxis'
import { timeSeasonBands, type SeasonBand } from '@/components/SeasonBands'
import { seasonOfDate, seasonOnly } from '@/lib/seasons'
import { seasonParent, seasonSegmentLayout, seasonSegments, type SeasonOrder, type SeasonSegment } from '@/lib/seasonSegments'
import { fmt } from '@/lib/stats'
import { num } from '@/lib/table'
import { pct as pctOf } from '@/lib/format'
import { fitsWidth } from '@/lib/fitLabel'
import { rightAxisDomain } from '@/lib/axes'
import { dotRim } from '@/lib/highlight'
import { INK, MARGIN } from '@/lib/theme'
import { useFitHeight } from '@/lib/chartFit'
import { useSingleView } from '@/lib/singleView'
import type { FilterRow, SeasonMeta } from '@/lib/types'
import type { useTooltip } from '@/hooks/useTooltip'

export const TIME_SHAPES = ['Stacked area', 'Daily bars', 'Streamgraph'] as const
export type TimeShape = (typeof TIME_SHAPES)[number]
export const TIME_AXES = ['Calendar dates', 'Seasons side by side'] as const
export type TimeAxis = (typeof TIME_AXES)[number]
const DAY = 86_400_000
/** how a second variable is drawn over the stack, against the right axis */
export const RIGHT_MARKS = ['Points', 'Line'] as const
export type RightMark = (typeof RIGHT_MARKS)[number]
/** one more variable on each filter, drawn over the stack on its own axis: its field, its name and unit, the axis title, its colour */
export interface RightVariable { field: string; label: string; unit: string; title: string; mark: RightMark; color: string }
/** room for the right axis's ticks and title */
const RIGHT_MARGIN = 62
/** FacetGrid's default gap between two panels */
const FACET_GAP = 16
/** a right-axis tick: no trailing zeros, no exponent short of a million */
const RIGHT_TICK = (v: number) => d3.format('~g')(v)
/** the reader's panel size (Panel size): each panel's width and height in px, 0 for Auto */
export interface TimelineSize { w: number; h: number }
export const TIMELINE_W: [number, number] = [400, 12000]
export const TIMELINE_H: [number, number] = [140, 1600]

/**
 * A panel's height on Auto: its own height on the page (240 each when there are several, 440 alone), a share of
 * a Dashboard tile's fitted height, and in the one-chart view a share of the window but never below a readable
 * 320 (360 alone): four sites squeezed into one window left each plot about 60 px tall, so they scroll down instead.
 */
export function timelineAutoHeight(fit: number | null, n: number, single: boolean) {
  const several = n > 1
  if (fit === null) return several ? 240 : 440
  const share = Math.max(120, (fit - FACET_GAP * (Math.max(1, n) - 1)) / Math.max(1, n) - PANEL_TITLE_H)
  return single ? Math.max(several ? 320 : 360, share) : share
}

/**
 * Panel size: each panel's width and height, dragged or typed, Auto by default (the field shows the fitted size,
 * so a drag starts from what is drawn); ↺ goes back to Auto for both. Wider than the chart, the panels scroll
 * sideways together; taller than the window, the page scrolls down.
 */
export function TimelineSizeControl({ size, auto, onChange }: { size: TimelineSize; auto: TimelineSize; onChange: (s: TimelineSize) => void }) {
  const sized = !!(size.w || size.h)
  return (
    <span className="control panel-size"
      title="Each panel's width and height in px: drag sideways or click to type. Wider than the chart, the panels scroll sideways together; taller than the window, the page scrolls down. ↺ goes back to Auto, the panels fitted to the chart">
      Panel size
      <span className="panel-size-row">
        <ScrubField label="Panel width" value={size.w || Math.round(auto.w)} min={TIMELINE_W[0]} max={TIMELINE_W[1]} step={20} unit={size.w ? 'px' : undefined} width={96}
          onChange={(v) => onChange({ ...size, w: Math.round(v) })} />
        <span aria-hidden>×</span>
        <ScrubField label="Panel height" value={size.h || Math.round(auto.h)} min={TIMELINE_H[0]} max={TIMELINE_H[1]} step={10} unit={size.h ? 'px' : undefined} width={88}
          onChange={(v) => onChange({ ...size, h: Math.round(v) })} />
        {sized && <button type="button" className="btn quiet icon-btn panel-size-reset" aria-label="Panel size: back to Auto" title="Back to Auto: the panels fit the chart"
          onClick={() => onChange({ w: 0, h: 0 })}><UndoIcon /></button>}
      </span>
    </span>
  )
}

/** one filter on the time axis: its stacked values, and what it lacks of the species shown */
export interface TimePoint { row: FilterRow; d: Date; v: Record<string, number>; missing: string[]; seg?: number }
interface Panel { site: string; color: string; points: TimePoint[]; nRows: number; stacked: d3.Series<TimePoint, string>[]; seasons: readonly SeasonMeta[]; segs: SeasonSegment[] }

const seasonNameOf = (row: FilterRow, seasons: readonly SeasonMeta[]) =>
  seasonOnly(seasonOfDate(row.date, row.month as number | null, seasons)?.name ?? String(row.season ?? ''))

/**
 * The composition chart's "over time" view: every filter on a date axis, the shown species (or
 * PMF sources) stacked as an area, as one bar per sampling day, or as a streamgraph (absolute
 * scale only: a stack of shares sums to 100 % every day, so the wiggle would only move a flat
 * band). One panel per site; Panel axes Shared puts every panel on the same dates and range.
 *
 * Values are each filter's own: concentration in µg/m³, or its share of the shown total
 * (`share` gives a chart's own share instead, the PMF paper's normalised fractions). A filter
 * missing a shown species is left out unless `completeOnly` is off; then the missing species
 * adds nothing to its stack (the tooltip names it). Negative values stack as zero. An area is broken
 * across a sampling gap rather than drawn through it.
 *
 * `right` draws one more variable (the sampling window's precipitation, any measurement) over the stack on a
 * right-hand axis of its own, each filter's value at its own date; a filter without it draws nothing there. The
 * axis's range follows Panel axes, as the left one's does.
 *
 * Seasons side by side (`timeAxis`) cuts the date axis into one block per season occurrence (Belg 2023,
 * Belg 2024 …), grouped by year then season, or by season then year for annual comparisons.
 * Whole cross-year seasons belong to their ending year. Each panel keeps its own blocks.
 *
 * Daily bars are one path per species, not a rect per filter and species (all sites is tens of thousands of
 * rects, each with handlers and the `.animated` transition), and one transparent rect over the plot answers the
 * pointer with the nearest sampling day, as the areas always did.
 */
export function CompositionTimeline({ rows, sites, species, color, dim, pct, share, shape, timeAxis = 'Calendar dates', seasonOrder = 'Year, then season', completeOnly, sharedAxes, seasonsFor, strip = false, size = { w: 0, h: 0 }, right = null, tip, units, itemNoun, onRecord, extraTip }: {
  rows: FilterRow[]
  sites: { name: string; color: string }[]
  /** bottom of the stack first */
  species: string[]
  color: (name: string) => string
  dim: (name: string) => number
  pct: boolean
  share?: (row: FilterRow, field: string) => number | null
  shape: TimeShape
  timeAxis?: TimeAxis
  seasonOrder?: SeasonOrder
  completeOnly: boolean
  sharedAxes: boolean
  seasonsFor: (site: string) => readonly SeasonMeta[]
  /** draw the season strip along the top of each panel (the tooltip and record name the season either way) */
  strip?: boolean
  /** the reader's panel size (0 for Auto): wider than the chart, the panels scroll sideways together */
  size?: TimelineSize
  /** a second variable over the stack, on a right-hand axis */
  right?: RightVariable | null
  tip: ReturnType<typeof useTooltip>
  units: string
  itemNoun: string
  onRecord: (p: TimePoint, seasonName: string) => void
  extraTip?: (row: FilterRow) => string[]
}) {
  const uid = useId().replace(/:/g, '')
  const stream = shape === 'Streamgraph'
  const bars = shape === 'Daily bars'
  const bySeason = timeAxis === 'Seasons side by side'
  const fit = useFitHeight()
  const single = useSingleView().on

  const panels = useMemo(() => {
    return sites.flatMap((s): Panel[] => {
      const mine = rows.filter((r) => r.site === s.name && r.date)
      const dated = mine
        .map((row): TimePoint | null => {
          const raw = species.map((f) => [f, num(pct && share ? share(row, f) : row[f])] as const)
          const missing = raw.filter(([, v]) => v === null).map(([f]) => f)
          if (missing.length === species.length || (completeOnly && missing.length)) return null
          const v = Object.fromEntries(raw.map(([f, x]) => [f, Math.max(0, x ?? 0)]))
          if (pct) {
            const total = d3.sum(species, (f) => v[f]) || 1
            for (const f of species) v[f] /= total
          }
          return { row, d: new Date(row.date.slice(0, 10) + 'T00:00:00'), v, missing }
        })
        .filter((p): p is TimePoint => !!p)
        .sort((a, b) => +a.d - +b.d)
      if (!dated.length) return []
      const seasons = seasonsFor(s.name)
      const { points, segs } = bySeason ? seasonSegments(dated, seasons, seasonOrder) : { points: dated, segs: [] }
      const gen = d3.stack<TimePoint>().keys(species).value((p, k) => p.v[k] ?? 0)
      if (stream) gen.offset(d3.stackOffsetWiggle).order(d3.stackOrderInsideOut)
      return [{ site: s.name, color: s.color, points, nRows: mine.length, stacked: gen(points), seasons, segs }]
    })
  }, [rows, sites, species, pct, share, completeOnly, stream, seasonsFor, bySeason, seasonOrder])

  const yExtent = (p: Panel): [number, number] => [
    stream ? (d3.min(p.stacked, (l) => d3.min(l, (d) => d[0])) ?? 0) : 0,
    pct && !stream ? 1 : (d3.max(p.stacked, (l) => d3.max(l, (d) => d[1])) ?? 1),
  ]
  const xAll = d3.extent(panels.flatMap((p) => p.points), (p) => p.d) as [Date, Date]
  const yAll: [number, number] = [d3.min(panels, (p) => yExtent(p)[0]) ?? 0, d3.max(panels, (p) => yExtent(p)[1]) ?? 1]
  const rightOf = (q: TimePoint) => (right ? num(q.row[right.field]) : null)
  const rightValues = (ps: TimePoint[]) => ps.map(rightOf).filter((v): v is number => v !== null)
  const rightAll = rightValues(panels.flatMap((p) => p.points))
  const several = panels.length > 1
  const autoH = timelineAutoHeight(fit, panels.length, single)
  const minSeasonWidth = bySeason ? Math.max(0, ...panels.map(p => seasonSegmentLayout(p.segs, seasonOrder, 0).width)) : 0

  if (!panels.length) return <p className="chart-note">No filter in this subset has {completeOnly ? `every ${itemNoun.replace(/s$/, '')} shown` : `any ${itemNoun.replace(/s$/, '')} shown`}.</p>

  // the site's name alone above its panel: the marks are coloured by species, so the site's colour would name nothing here
  const grid = (
    <FacetGrid items={panels} maxCols={1} minCell={240}
      title={several ? (p) => p.site : undefined}
      label={(p) => p.site}
      note={(p) => {
        const n = p.points.length < p.nRows ? `${p.points.length} of ${p.nRows} filters` : `${p.points.length} filters`
        const withRight = rightValues(p.points).length
        return right && withRight < p.points.length ? `${n}, ${withRight} with ${right.label}` : n
      }}
      render={(p, cellW, i) => {
        const sideMargin = right ? RIGHT_MARGIN : MARGIN.right
        const width = Math.max(size.w || cellW, bySeason ? minSeasonWidth + MARGIN.left + sideMargin : 0)
        const height = size.h || autoH
        const innerW = Math.max(200, width - MARGIN.left - sideMargin)
        const segs = p.segs
        const segmentLayout = seasonSegmentLayout(segs, seasonOrder, innerW)
        const segX = segmentLayout.lefts
        const segW = (k: number) => segmentLayout.widths[k]
        const axisItems: CategoryLabel[] = segs.map((g, k) => {
          const ends = `${d3.timeFormat('%b %-d')(new Date(g.d0))} – ${d3.timeFormat('%b %-d')(new Date(g.d1))}`
          return { key: String(k), label: `${seasonOrder === 'Year, then season' ? g.label : g.year} (${ends})`,
            parent: seasonParent(g, seasonOrder), color: g.color || INK.muted,
            parentColor: seasonOrder === 'Year, then season' ? INK.text : g.color || INK.muted }
        })
        const axisSpans: CategorySpans = { span: key => { const k = Number(key); return [segX[k], segX[k] + segW(k)] }, range: () => [0, innerW], gap: 10 }
        const bottom = bySeason ? 12 + categoryAxisHeight(axisItems, axisSpans) : MARGIN.bottom
        const innerH = Math.max(60, height - MARGIN.top - bottom)
        const x = d3.scaleTime().domain(sharedAxes ? xAll : (d3.extent(p.points, (q) => q.d) as [Date, Date])).range([0, innerW])
        // A sparse block keeps enough width for its labels; dates remain proportional inside it.
        const dayWidth = (k: number) => segW(k) / ((segs[k].d1 - segs[k].d0) / DAY + 1)
        const pxPerDay = bySeason ? Math.min(...segs.map((_, k) => dayWidth(k))) : x(new Date(+p.points[0].d + DAY)) - x(p.points[0].d)
        const px = (q: TimePoint) => (bySeason ? segX[q.seg!] + ((+q.d - segs[q.seg!].d0) / DAY + 0.5) * dayWidth(q.seg!) : x(q.d))
        const [lo, hi] = sharedAxes ? yAll : yExtent(p)
        const y = d3.scaleLinear().domain([lo, hi * (stream || pct ? 1 : 1.02)]).range([innerH, 0]).nice()
        const area = d3.area<d3.SeriesPoint<TimePoint>>().x((d) => px(d.data)).y0((d) => y(d[0])).y1((d) => y(d[1])).curve(d3.curveMonotoneX)
        const rightDomain = rightAxisDomain(sharedAxes ? rightAll : rightValues(p.points))
        const yRight = rightDomain && d3.scaleLinear().domain(rightDomain).range([innerH, 0]).nice()
        const bands: SeasonBand[] = !strip ? [] : bySeason
          ? segs.map((g, k) => ({ x: segX[k], w: segW(k), color: g.color || INK.muted, name: g.name }))
          : timeSeasonBands(x, p.seasons, innerW)
        const seasonOf = (q: TimePoint) => seasonNameOf(q.row, p.seasons)
        // a bar per sampling day: three-quarters of the typical spacing between samples, so gaps read as gaps
        const spacing = d3.median(p.points.slice(1).flatMap((q, j) => q.seg === p.points[j].seg ? [px(q) - px(p.points[j])] : [])) ?? 6
        const barW = Math.max(1.5, Math.min(18, spacing * 0.75, 3 * pxPerDay))
        // a dot as big as the sampling leaves room for: years of filters in one panel would otherwise hide the stack under them
        const rightR = Math.max(1.6, Math.min(right?.mark === 'Line' ? 1.8 : 3, spacing * 0.6))
        const rightLine = yRight && d3.line<TimePoint>().defined((q) => rightOf(q) !== null).x(px).y((q) => yRight(rightOf(q) ?? 0))
        // the area breaks across a sampling gap (four times the usual spacing, a month at least) instead of drawing through it
        const gapMs = Math.max(30 * DAY, 4 * (d3.median(p.points.slice(1).map((q, j) => +q.d - +p.points[j].d)) ?? DAY))
        const runs: [number, number][] = []
        p.points.forEach((q, j) => { if (j === 0 || q.seg !== p.points[j - 1].seg || +q.d - +p.points[j - 1].d > gapMs) runs.push([j, j + 1]); else runs[runs.length - 1][1] = j + 1 })
        // the sampling day nearest the pointer; over bars, only within reach of one, so a gap reads as empty
        const nearestTo = (e: React.MouseEvent): TimePoint | null => {
          const [mx] = d3.pointer(e)
          const j = d3.bisector((q: TimePoint) => px(q)).center(p.points, mx)
          const q = p.points[j]
          return q && ((!bars && !bySeason) || Math.abs(px(q) - mx) <= Math.max(4, bars ? barW / 2 + 2 : 10)) ? q : null
        }
        const tipOf = (q: TimePoint) => [
          `${q.row.date.slice(0, 10)} · ${seasonOf(q)}${several ? ` · ${p.site}` : ''}`,
          ...species.slice().reverse().map((f) => {
            const c = typeof q.row[f] === 'number' ? (q.row[f] as number) : null
            const sh = share?.(q.row, f)
            const shown = pct ? pctOf(q.v[f], 1) : `${fmt(c, 2)} ${units}`
            const other = pct ? (c !== null ? ` (${fmt(c, 2)} ${units})` : '') : sh != null ? ` (${pctOf(sh, 1)})` : ''
            return q.missing.includes(f) ? `${f}: not measured` : `${f}: ${shown}${other}`
          }),
          ...(right ? [`${right.label}: ${rightOf(q) === null ? 'not available' : `${fmt(rightOf(q), 2)}${right.unit ? ` ${right.unit}` : ''}`}`] : []),
          ...(extraTip?.(q.row) ?? []),
          'click for the filter record',
        ]
        const clip = `${uid}-${i}`
        return (
          <svg width={width} height={Math.max(height, innerH + MARGIN.top + bottom)} data-composition-site={p.site} data-composition-filters={p.points.length} className={bars ? undefined : 'animated'}>
            <defs><clipPath id={clip}><rect width={innerW} height={innerH} /></clipPath></defs>
            <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
              {/* the season strip along the top, named where a band has room */}
              <g pointerEvents="none">
                {bands.map((b, j) => <rect key={j} data-export-season={b.name} data-export-color={b.color} x={b.x} y={-14} width={b.w} height={11} fill={b.color} />)}
                {bands.map((b, j) => {
                  // the full name where it fits, else its first word (Belg), else nothing (measured at 9 px)
                  const full = seasonOnly(b.name), short = full.split(' (')[0]
                  const text = fitsWidth(full, b.w - 8, 9) ? full : fitsWidth(short, b.w - 8, 9) ? short : ''
                  return text && <text key={j} x={b.x + b.w / 2} y={-5.5} textAnchor="middle" fontSize={9} fill={INK.text}>{text}</text>
                })}
              </g>
              <YAxis scale={y} x={0} gridWidth={innerW} wrapLabel
                label={stream ? `${units} (baseline free)` : pct ? `share of shown ${itemNoun}` : units}
                format={(v: number) => (pct ? `${Math.round(v * 100)}%` : String(v))} />
              {bySeason
                ? <g data-season-axis={seasonOrder}>
                  {segs.map((g, k) => <g key={k} data-season-block={g.label} data-season-year={g.year} data-season-start={new Date(g.d0).toISOString().slice(0, 10)} data-season-end={new Date(g.d1).toISOString().slice(0, 10)}>
                    {k > 0 && <line x1={segX[k] - segmentLayout.gaps[k] / 2} x2={segX[k] - segmentLayout.gaps[k] / 2} y2={innerH} stroke={INK.axis}
                      strokeOpacity={0.45} strokeDasharray={seasonParent(g, seasonOrder) === seasonParent(segs[k - 1], seasonOrder) ? '2 3' : '4 3'} />}
                  </g>)}
                  <CategoryAxis items={axisItems} scale={axisSpans} y={innerH} />
                </g>
                : <XAxis scale={x as any} y={innerH} format={(d: Date) => d3.timeFormat('%b %Y')(d)} tickCount={Math.max(2, Math.min(7, Math.floor(innerW / 90)))} />}
              <g clipPath={`url(#${clip})`}>
                <g pointerEvents="none">
                  {bars && p.stacked.map((layer) => (
                    <path key={layer.key} fill={color(layer.key)} fillOpacity={0.92 * dim(layer.key)}
                      d={layer.map((d) => {
                        const h = y(d[0]) - y(d[1])
                        return h > 0 ? `M${px(d.data) - barW / 2},${y(d[1])}h${barW}v${h}h${-barW}Z` : ''
                      }).join('')} />
                  ))}
                  {!bars && p.stacked.map((layer) => (
                    <path key={layer.key} d={runs.map(([a, b]) => area(layer.slice(a, b)) ?? '').join('')} fill={color(layer.key)} fillOpacity={0.88 * dim(layer.key)} stroke={INK.panel} strokeWidth={0.4}
                      style={{ transition: 'd 0.35s' }} />
                  ))}
                </g>
                <rect width={innerW} height={innerH} fill="transparent"
                  onMouseMove={(e) => {
                    const q = nearestTo(e)
                    e.currentTarget.style.cursor = q ? 'pointer' : ''
                    if (q) tip.show(e, tipOf(q)); else tip.hide()
                  }}
                  onMouseLeave={tip.hide}
                  onClick={(e) => { const q = nearestTo(e); if (q) { e.stopPropagation(); onRecord(q, seasonOf(q)) } }} />
              </g>
              {/* over the stack and outside its clip, so a value at either end of the axis keeps its whole dot; the hit rect still answers */}
              {right && yRight && rightLine && (
                <g pointerEvents="none">
                  <YAxis scale={yRight} x={innerW} side="right" wrapLabel label={right.title} format={RIGHT_TICK} />
                  {right.mark === 'Line' && <path d={runs.map(([a, b]) => rightLine(p.points.slice(a, b)) ?? '').join('')} fill="none" stroke={right.color} strokeWidth={1.4} strokeLinejoin="round" />}
                  {p.points.map((q, j) => {
                    const v = rightOf(q)
                    return v === null ? null : <circle key={j} cx={px(q)} cy={yRight(v)} r={rightR} fill={right.color} {...dotRim()} />
                  })}
                </g>
              )}
            </g>
          </svg>
        )
      }} />
  )
  // panels wider than the chart scroll sideways as one, so the sites stay on the same dates
  return <>
    {size.w || bySeason ? <div className="timeline-scroll" tabIndex={0} role="region" aria-label="Composition timeline, scroll horizontally for all seasons">{grid}</div> : grid}
    {bySeason && <p className="chart-note">Whole seasons are grouped by their ending year: Bega Oct 2022–Jan 2023 belongs to 2023. Each block shows its observed sampling dates.</p>}
  </>
}

/** the seasons a set of panels' strips show, each once, in order, for the strip's legend */
export function stripSeasons(seasonLists: (readonly SeasonMeta[])[]) {
  return [...new Map(seasonLists.flat().map((s) => [seasonOnly(s.name), s])).values()]
}
