import { useId, useMemo } from 'react'
import * as d3 from 'd3'
import { FacetGrid } from '@/components/FacetGrid'
import { XAxis, YAxis } from '@/components/Axes'
import { timeSeasonBands, type SeasonBand } from '@/components/SeasonBands'
import { seasonOfDate, seasonOnly } from '@/lib/seasons'
import { fmt } from '@/lib/stats'
import { FONT, INK, MARGIN } from '@/lib/theme'
import type { FilterRow, SeasonMeta } from '@/lib/types'
import type { useTooltip } from '@/hooks/useTooltip'

export const TIME_SHAPES = ['Stacked area', 'Daily bars', 'Streamgraph'] as const
export type TimeShape = (typeof TIME_SHAPES)[number]
export const TIME_AXES = ['Calendar dates', 'Seasons side by side'] as const
export type TimeAxis = (typeof TIME_AXES)[number]
const DAY = 86_400_000
/** a season's next occurrence starts after a sampling gap this long (a season is a few months; its next year's is ~a year on) */
const SEASON_BREAK = 150 * DAY
/** px between two season blocks */
const SEG_GAP = 10

/** one filter on the time axis: its stacked values, and what it lacks of the species shown */
export interface TimePoint { row: FilterRow; d: Date; v: Record<string, number>; missing: string[]; seg?: number }
/** one occurrence of a season (Belg 2024, Bega 2023/24) on the seasons-side-by-side axis: its points are points[a, b) */
interface Segment { name: string; label: string; year: string; color: string; a: number; b: number; d0: number; d1: number }
interface Panel { site: string; color: string; points: TimePoint[]; nRows: number; stacked: d3.Series<TimePoint, string>[]; seasons: readonly SeasonMeta[]; segs: Segment[] }

const seasonNameOf = (row: FilterRow, seasons: readonly SeasonMeta[]) =>
  seasonOnly(seasonOfDate(row.date, row.month as number | null, seasons)?.name ?? String(row.season ?? ''))

/**
 * Each run of filters in one season (a new run where the season changes or sampling stops for
 * SEASON_BREAK) is one occurrence; occurrences are ordered by the calendar's season order, then by
 * date, so one season's years sit side by side. Returns the points in that order, each with its block.
 */
function seasonSegments(points: TimePoint[], seasons: readonly SeasonMeta[]): { points: TimePoint[]; segs: Segment[] } {
  const runs: { name: string; pts: TimePoint[] }[] = []
  for (const q of points) {
    const name = seasonNameOf(q.row, seasons)
    const last = runs[runs.length - 1]
    const prev = last?.pts[last.pts.length - 1]
    if (last && last.name === name && +q.d - +prev!.d <= SEASON_BREAK) last.pts.push(q)
    else runs.push({ name, pts: [q] })
  }
  const order = (name: string) => { const i = seasons.findIndex((s) => seasonOnly(s.name) === name); return i < 0 ? seasons.length : i }
  runs.sort((a, b) => order(a.name) - order(b.name) || +a.pts[0].d - +b.pts[0].d)
  const out: TimePoint[] = [], segs: Segment[] = []
  runs.forEach((r, k) => {
    const d0 = +r.pts[0].d, d1 = +r.pts[r.pts.length - 1].d
    const y0 = r.pts[0].d.getFullYear(), y1 = r.pts[r.pts.length - 1].d.getFullYear()
    const year = y0 === y1 ? String(y0) : `${y0}/${String(y1).slice(2)}`
    const season = seasons.find((s) => seasonOnly(s.name) === r.name)
    segs.push({ name: r.name, label: `${r.name.split(' (')[0] || 'no season'} ${year}`, year, color: season?.color ?? INK.muted, a: out.length, b: out.length + r.pts.length, d0, d1 })
    for (const q of r.pts) out.push({ ...q, seg: k })
  })
  return { points: out, segs }
}

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
 * Seasons side by side (`timeAxis`) cuts the date axis into one block per season occurrence (Belg 2023,
 * Belg 2024 …), each as wide as its own dates, grouped by season and then by year, with the months between
 * dropped: filter a single season in the sidebar to read it year over year. Each panel keeps its own blocks.
 *
 * Daily bars are one path per species, not a rect per filter and species (all sites is tens of thousands of
 * rects, each with handlers and the `.animated` transition), and one transparent rect over the plot answers the
 * pointer with the nearest sampling day, as the areas always did.
 */
export function CompositionTimeline({ rows, sites, species, color, dim, pct, share, shape, timeAxis = 'Calendar dates', completeOnly, sharedAxes, seasonsFor, strip = false, tip, units, itemNoun, onRecord, extraTip }: {
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
  completeOnly: boolean
  sharedAxes: boolean
  seasonsFor: (site: string) => readonly SeasonMeta[]
  /** draw the season strip along the top of each panel (the tooltip and record name the season either way) */
  strip?: boolean
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

  const panels = useMemo(() => {
    const num = (v: unknown) => (typeof v === 'number' && Number.isFinite(v) ? v : null)
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
      const { points, segs } = bySeason ? seasonSegments(dated, seasons) : { points: dated, segs: [] }
      const gen = d3.stack<TimePoint>().keys(species).value((p, k) => p.v[k] ?? 0)
      if (stream) gen.offset(d3.stackOffsetWiggle).order(d3.stackOrderInsideOut)
      return [{ site: s.name, color: s.color, points, nRows: mine.length, stacked: gen(points), seasons, segs }]
    })
  }, [rows, sites, species, pct, share, completeOnly, stream, seasonsFor, bySeason])

  const yExtent = (p: Panel): [number, number] => [
    stream ? (d3.min(p.stacked, (l) => d3.min(l, (d) => d[0])) ?? 0) : 0,
    pct && !stream ? 1 : (d3.max(p.stacked, (l) => d3.max(l, (d) => d[1])) ?? 1),
  ]
  const xAll = d3.extent(panels.flatMap((p) => p.points), (p) => p.d) as [Date, Date]
  const yAll: [number, number] = [d3.min(panels, (p) => yExtent(p)[0]) ?? 0, d3.max(panels, (p) => yExtent(p)[1]) ?? 1]
  const several = panels.length > 1

  if (!panels.length) return <p className="chart-note">No filter in this subset has {completeOnly ? `every ${itemNoun.replace(/s$/, '')} shown` : `any ${itemNoun.replace(/s$/, '')} shown`}.</p>

  return (
    <FacetGrid items={panels} maxCols={1} minCell={240}
      title={several ? (p) => <><span className="swatch" style={{ background: p.color }} />{p.site}</> : undefined}
      label={(p) => p.site}
      note={(p) => p.points.length < p.nRows ? `${p.points.length} of ${p.nRows} filters` : `${p.points.length} filters`}
      render={(p, width, i, cell) => {
        const height = cell.height ?? (several ? 240 : 440)
        const innerW = Math.max(200, width - MARGIN.left - MARGIN.right)
        const innerH = Math.max(60, height - MARGIN.top - MARGIN.bottom)
        const x = d3.scaleTime().domain(sharedAxes ? xAll : (d3.extent(p.points, (q) => q.d) as [Date, Date])).range([0, innerW])
        // seasons side by side: each block as wide as its dates (half a day either side of its first and last), at one px per day
        // across blocks so a day is as wide in every year, SEG_GAP apart
        const segs = p.segs
        const pxPerDay = bySeason
          ? Math.max(0, innerW - SEG_GAP * (segs.length - 1)) / (d3.sum(segs, (g) => g.d1 - g.d0 + DAY) / DAY)
          : x(new Date(+p.points[0].d + DAY)) - x(p.points[0].d)
        const segX: number[] = []
        segs.forEach((g, k) => segX.push(k ? segX[k - 1] + ((segs[k - 1].d1 - segs[k - 1].d0 + DAY) / DAY) * pxPerDay + SEG_GAP : 0))
        const segW = (k: number) => ((segs[k].d1 - segs[k].d0 + DAY) / DAY) * pxPerDay
        const px = (q: TimePoint) => (bySeason ? segX[q.seg!] + ((+q.d - segs[q.seg!].d0) / DAY + 0.5) * pxPerDay : x(q.d))
        const [lo, hi] = sharedAxes ? yAll : yExtent(p)
        const y = d3.scaleLinear().domain([lo, hi * (stream || pct ? 1 : 1.02)]).range([innerH, 0]).nice()
        const area = d3.area<d3.SeriesPoint<TimePoint>>().x((d) => px(d.data)).y0((d) => y(d[0])).y1((d) => y(d[1])).curve(d3.curveMonotoneX)
        const bands: SeasonBand[] = !strip ? [] : bySeason
          ? segs.map((g, k) => ({ x: segX[k], w: segW(k), color: g.color, name: g.name }))
          : timeSeasonBands(x, p.seasons, innerW)
        const seasonOf = (q: TimePoint) => seasonNameOf(q.row, p.seasons)
        // a bar per sampling day: three-quarters of the typical spacing between samples, so gaps read as gaps
        const spacing = d3.median(p.points.slice(1).filter((q, j) => q.seg === p.points[j].seg).map((q, j) => px(q) - px(p.points[j]))) ?? 6
        const barW = Math.max(1.5, Math.min(18, spacing * 0.75, 3 * pxPerDay))
        // the area breaks across a sampling gap (four times the usual spacing, a month at least) instead of drawing through it
        const gapMs = Math.max(30 * DAY, 4 * (d3.median(p.points.slice(1).map((q, j) => +q.d - +p.points[j].d)) ?? DAY))
        const runs: [number, number][] = []
        p.points.forEach((q, j) => { if (j === 0 || q.seg !== p.points[j - 1].seg || +q.d - +p.points[j - 1].d > gapMs) runs.push([j, j + 1]); else runs[runs.length - 1][1] = j + 1 })
        // the sampling day nearest the pointer; over bars, only within reach of one, so a gap reads as empty
        const nearestTo = (e: React.MouseEvent): TimePoint | null => {
          const [mx] = d3.pointer(e)
          const j = d3.bisector((q: TimePoint) => px(q)).center(p.points, mx)
          const q = p.points[j]
          return q && ((!bars && !bySeason) || Math.abs(px(q) - mx) <= Math.max(4, bars ? barW / 2 + 2 : SEG_GAP)) ? q : null
        }
        const tipOf = (q: TimePoint) => [
          `${q.row.date.slice(0, 10)} · ${seasonOf(q)}${several ? ` · ${p.site}` : ''}`,
          ...species.slice().reverse().map((f) => {
            const c = typeof q.row[f] === 'number' ? (q.row[f] as number) : null
            const sh = share?.(q.row, f)
            const shown = pct ? `${(q.v[f] * 100).toFixed(1)} %` : `${fmt(c, 2)} ${units}`
            const other = pct ? (c !== null ? ` (${fmt(c, 2)} ${units})` : '') : sh != null ? ` (${(sh * 100).toFixed(1)} %)` : ''
            return q.missing.includes(f) ? `${f}: not measured` : `${f}: ${shown}${other}`
          }),
          ...(extraTip?.(q.row) ?? []),
          'click for the filter record',
        ]
        const clip = `${uid}-${i}`
        return (
          <svg width={width} height={height} className={bars ? undefined : 'animated'}>
            <defs><clipPath id={clip}><rect width={innerW} height={innerH} /></clipPath></defs>
            <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
              {/* the season strip along the top, named where a band has room */}
              <g pointerEvents="none">
                {bands.map((b, j) => <rect key={j} data-export-season={b.name} data-export-color={b.color} x={b.x} y={-14} width={b.w} height={11} fill={b.color} />)}
                {bands.map((b, j) => {
                  // the full name where it fits, else its first word (Belg), else nothing (about 5 px a character at 9 px)
                  const full = seasonOnly(b.name), short = full.split(' (')[0]
                  const text = full.length * 5 + 8 <= b.w ? full : short.length * 5 + 8 <= b.w ? short : ''
                  return text && <text key={j} x={b.x + b.w / 2} y={-5.5} textAnchor="middle" fontSize={9} fill={INK.text}>{text}</text>
                })}
              </g>
              <YAxis scale={y} x={0} gridWidth={innerW}
                label={stream ? `${units} (baseline free)` : pct ? `share of shown ${itemNoun}` : units}
                format={(v: number) => (pct ? `${Math.round(v * 100)}%` : String(v))} />
              {bySeason
                ? <SeasonBlockAxis segs={segs} segX={segX} segW={segW} y={innerH} />
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
            </g>
          </svg>
        )
      }} />
  )
}

/**
 * The seasons-side-by-side axis: a baseline under each block, its season and year below (the year
 * alone where the name does not fit, as the strip and the block's order name the season), and a
 * month tick at each block's first and last day where there is room.
 */
function SeasonBlockAxis({ segs, segX, segW, y }: { segs: Segment[]; segX: number[]; segW: (k: number) => number; y: number }) {
  const fits = (text: string, w: number) => text.length * 6.2 + 6 <= w
  return (
    <g transform={`translate(0,${y})`} fontFamily={FONT.family} pointerEvents="none">
      {segs.map((g, k) => {
        const x0 = segX[k], w = segW(k)
        const label = fits(g.label, w) ? g.label : fits(g.year, w) ? g.year : ''
        const ends = `${d3.timeFormat('%b %-d')(new Date(g.d0))} – ${d3.timeFormat('%b %-d')(new Date(g.d1))}`
        return (
          <g key={k}>
            <line x1={x0} x2={x0 + w} stroke={INK.axis} />
            <line x1={x0} x2={x0} y2={5} stroke={INK.axis} />
            <line x1={x0 + w} x2={x0 + w} y2={5} stroke={INK.axis} />
            {label && <text x={x0 + w / 2} y={17} textAnchor="middle" fontSize={11} fontWeight={600} fill={INK.text}>{label}</text>}
            {fits(ends, w) && <text x={x0 + w / 2} y={30} textAnchor="middle" fontSize={9.5} fill={INK.muted}>{ends}</text>}
          </g>
        )
      })}
    </g>
  )
}

/** the seasons a set of panels' strips show, each once, in order, for the strip's legend */
export function stripSeasons(seasonLists: (readonly SeasonMeta[])[]) {
  return [...new Map(seasonLists.flat().map((s) => [seasonOnly(s.name), s])).values()]
}
