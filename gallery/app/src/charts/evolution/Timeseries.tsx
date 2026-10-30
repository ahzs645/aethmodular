import { useMemo, useRef, useState, type ReactNode } from 'react'
import { seasonOnly } from '@/lib/seasons'
import { SeasonBands, timeSeasonBands } from '@/components/SeasonBands'
import * as d3 from 'd3'
import { ChartFrame, Empty, Segmented } from '@/components/ChartFrame'
import { Legend, toggleIn } from '@/components/Legend'
import { XAxis, YAxis } from '@/components/Axes'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { focusStyle, useHighlightActions } from '@/lib/highlight'
import { ScatterLayer } from '@/components/ScatterLayer'
import { fmt, splitAtGaps } from '@/lib/stats'
import { INK, MARGIN, FONT, withUnit, type DotLevel } from '@/lib/theme'
import type { FilterRow, MetaFile } from '@/lib/types'
import { calendarLabel, mixedCalendars, seasonsForSite } from '@/siteSeasons'
import { useChartOptions } from '@/lib/chartOptions'
import { useChartGrouping } from '@/components/GroupingDialog'
import { useSearchParam } from '@/lib/url'
import { useChartLabel, useFitHeight } from '@/lib/chartFit'
import { ExcludedNote, toMarks, useMarkReason, useMarkedRows } from '@/lib/markedRows'

const LAYOUTS = ['Overlay', 'Per site'] as const
const SERIES_BY = ['Site', 'Group'] as const
type Pt = { row: FilterRow; date: Date; value: number }
type Series = { name: string; color: string; pts: Pt[]; keep: (r: FilterRow) => boolean }

/**
 * Timeseries / area — 179 evolution figures in the estate (timeseries, line,
 * fill_between bands). Seasonal shading uses config.ETHIOPIA_SEASONS colours,
 * shipped through meta.json, so it matches the matplotlib axvspan overlays
 * rather than re-declaring a calendar (AGENTS.md is emphatic about this).
 *
 * "Per site" gives each site its own time axis — Beijing's 2013–2016 run and
 * Delhi's 2022–2024 run share almost no dates, so an overlay is mostly empty
 * space. Which sites appear is the subset bar's decision.
 *
 * A synchronized cursor (react-graph-gallery's line-chart pattern) follows
 * the pointer by *date* across every panel: hovering 2023-03 on the Addis
 * panel drops the same date line on Delhi's, with each series' nearest
 * value labelled, which is how "was Delhi high the same month" gets read.
 */
export function Timeseries({ rows, meta, field }: { rows: FilterRow[]; meta: MetaFile; field: string }) {
  const chartLabel = useChartLabel()
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
  // the brush only: the points read the hover themselves (components/ScatterLayer)
  const hl = useHighlightActions()

  const [hidden, setHidden] = useState<Set<string>>(new Set())
  const [hover, setHover] = useState<string | null>(null)
  const [cursor, setCursor] = useState<Date | null>(null)

  // series by site, or by the bands of the reader's grouping (components/GroupingDialog), every site pooled
  const measured = useMemo(() => rows.filter((r) => typeof r[field] === 'number' && Number.isFinite(r[field] as number)), [rows, field])
  // Filters → excluded: Mark (lib/markedRows): excluded filters with this measurement, drawn as × in their series' panel
  const markedAll = useMarkedRows()
  const marked = useMemo(() => markedAll.filter((r) => typeof r[field] === 'number' && Number.isFinite(r[field] as number)), [markedAll, field])
  const markReason = useMarkReason()
  const { grp, picker, missingNote } = useChartGrouping('timeseries', measured, () => setSeriesBy('Group'))
  const [seriesChoice, setSeriesBy] = useSearchParam<'Site' | 'Group'>('timeseries.series', 'Site', SERIES_BY)
  // Group is offered only once a grouping is picked
  const seriesBy = grp ? seriesChoice : 'Site'
  const byGroup = seriesBy === 'Group'
  const series = useMemo(
    () =>
      (byGroup ? grp!.labels.map((l, i) => ({ name: l, color: grp!.colors[i], keep: (r: FilterRow) => grp!.labelOf(r) === l }))
        : meta.sites.map((s) => ({ name: s.name, color: s.color, keep: (r: FilterRow) => r.site === s.name })))
        .filter((s) => !hidden.has(s.name))
        .map((s) => {
          const pts: Pt[] = measured
            .filter(s.keep)
            .map((r) => ({ row: r, date: new Date(r.date + 'T00:00:00'), value: r[field] as number }))
            .sort((a, b) => +a.date - +b.date)
          return { name: s.name, color: s.color, pts, keep: s.keep }
        })
        .filter((s) => s.pts.length > 1),
    [measured, field, meta.sites, hidden, byGroup, grp]
  )

  // the options and when each applies (lib/chartOptions): with one site, overlay and per-site
  // panels are the same chart; an overlay of sites on their own calendars has nothing to shade
  const o = useChartOptions('timeseries', [
    { key: 'series', node: () => <>
      {grp && <Segmented label="Series" value={seriesBy} options={SERIES_BY} onChange={setSeriesBy}
        title="One series per site, or per band of a grouping (every site pooled in each band)" />}
      {picker}
    </> },
    { key: 'layout', label: 'Layout', choices: LAYOUTS.map((v) => ({ value: v, label: v === 'Per site' && byGroup ? 'Per group' : v })), default: 'Per site', when: (c) => (c.nSeries as number) > 1, hidden: 'Per site' },
    { key: 'smooth', label: 'Rolling median', toggle: true, default: true },
    { key: 'shading', label: 'Season shading', toggle: true, default: true, hidden: false, when: (c) => !(c.mixed && c.layout === 'Overlay') },
    { key: 'cursor', when: () => !!cursor, node: () => <span className="control" style={{ fontFamily: 'var(--mono)' }}>{cursor && d3.timeFormat('%Y-%m-%d')(cursor)}</span> },
  ], { nSeries: series.length, mixed: mixedCalendars(meta) })
  const layout = o.values.layout as (typeof LAYOUTS)[number]
  const smooth = o.values.smooth as boolean
  const seasonShading = o.values.shading as boolean

  // a hovered site dims the other sites; a hovered season (a background key, not a series) deepens its own bands
  const siteHover = hover && series.some((s) => s.name === hover) ? hover : null
  const seasonHover = seasonShading && hover && meta.seasons.some((s) => s.name === hover) ? hover : null
  const dim = (name: string) => (siteHover === null || siteHover === name ? 1 : 0.15)

  const innerW = Math.max(240, width - MARGIN.left - MARGIN.right)
  const fit = useFitHeight()
  const panelCount = layout === 'Overlay' ? 1 : Math.max(1, series.length)
  const panelH = fit === null ? (layout === 'Overlay' ? 430 : 250) : Math.max(140, fit / panelCount)
  const innerH = panelH - MARGIN.top - MARGIN.bottom

  // shared y so panels compare; per-panel x so each site fills its own width. The same scales until the data or the
  // size change, so the dots survive the cursor (which re-renders the chart on every move)
  const panels = useMemo(() => (layout === 'Overlay' ? [{ key: 'all', title: null as string | null, color: INK.text, series }] : series.map((s) => ({ key: s.name, title: s.name, color: s.color, series: [s] })))
    .map((panel) => ({ ...panel, x: d3.scaleTime().domain(d3.extent(panel.series.flatMap((s) => s.pts), (p) => p.date) as [Date, Date]).range([0, innerW]) })),
  [layout, series, innerW])
  const y = useMemo(() => {
    const allPts = series.flatMap((s) => s.pts)
    return d3.scaleLinear().domain([Math.min(0, d3.min(allPts, (p) => p.value) ?? 0), (d3.max(allPts, (p) => p.value) ?? 1) * 1.08]).range([innerH, 0]).nice()
  }, [series, innerH])

  /** Centred rolling median — robust to the spikes these series carry. */
  const rolling = (pts: Pt[], w = 7) =>
    pts.map((p, i) => {
      const lo = Math.max(0, i - Math.floor(w / 2))
      const win = pts.slice(lo, lo + w).map((q) => q.value).sort(d3.ascending)
      return { date: p.date, value: d3.quantile(win, 0.5) ?? p.value }
    })
  /** Rolling median per sampling run: the line breaks at gaps over 45 days rather than bridging them. */
  const rollingPath = (pts: Pt[], line: d3.Line<{ date: Date; value: number }>) =>
    splitAtGaps(pts, (p) => p.date).map((seg) => line(rolling(seg)) ?? '').join('')

  const bisect = d3.bisector<Pt, Date>((p) => p.date).center
  /** The series point nearest the cursor date, or null when further than 45 days. */
  const nearest = (pts: Pt[], d: Date): Pt | null => {
    if (!pts.length) return null
    const p = pts[bisect(pts, d)]
    return p && Math.abs(+p.date - +d) <= 45 * 86400e3 ? p : null
  }

  // under per-site calendars each site panel shades its own seasons; an overlay of
  // several sites has no single calendar to shade, so it shows none
  const mixed = mixedCalendars(meta)
  const bandsFor = (x: d3.ScaleTime<number, number>, site: string | null) => {
    if (!seasonShading || (mixed && !site)) return []
    return timeSeasonBands(x, seasonsForSite(meta, site), innerW)
  }

  return (
    <ChartFrame
      id="timeseries"
      title="Timeseries — filter measurements over time"
      subtitle={`Every filter as a point, with an optional 7-sample rolling median that breaks at sampling gaps over 45 days. Season bands use the ${calendarLabel(meta)} selected in the filters${mixedCalendars(meta) ? ', one per site panel (the overlay has no single calendar, so it is left unshaded)' : ''}. Move the pointer across any panel and a date cursor follows on every other panel. Drag on the timeline in the subset bar to zoom every tab to a date range.`}
      provenance="stands in for 73 timeseries + 59 area + 46 line figures · plotting/timeseries.py · react-graph-gallery.com/timeseries · /line-chart-synchronized-cursors"
      controls={o.controls()}
    >
      <div ref={wrapRef} className="chart-wrap">
        {series.length === 0 ? (
          <Empty>No site has 2 or more filters with {field} in this subset.</Empty>
        ) : (
          panels.map((panel) => {
            const pts = panel.series.flatMap((s) => s.pts)
            const x = panel.x
            const line = d3.line<{ date: Date; value: number }>().x((p) => x(p.date)).y((p) => y(p.value)).curve(d3.curveMonotoneX)
            const [x0d, x1d] = x.domain() as [Date, Date]
            const cursorIn = cursor && cursor >= x0d && cursor <= x1d
            return (
              <div key={panel.key}>
                {panel.title && chartLabel(panel.title) && (
                  <p className="facet-title">
                    <span className="swatch" style={{ background: panel.color }} /> {chartLabel(panel.title!)}
                    <span style={{ color: 'var(--ink-muted)', fontWeight: 400, fontFamily: 'var(--mono)', fontSize: 11 }}>n={pts.length}</span>
                  </p>
                )}
                <svg width={width} height={panelH}>
                  <g
                    transform={`translate(${MARGIN.left},${MARGIN.top})`}
                    onMouseMove={(e) => {
                      const [mx] = d3.pointer(e, e.currentTarget)
                      if (mx >= 0 && mx <= innerW) setCursor(x.invert(mx))
                    }}
                    onMouseLeave={() => setCursor(null)}
                  >
                    {/* catches pointer moves in the empty parts of the panel (over the plot, the points' pointer layer does) */}
                    <rect x={0} y={0} width={innerW} height={innerH} fill="transparent" />
                    <SeasonBands bands={bandsFor(x, byGroup ? null : panel.title)} height={innerH} hover={seasonHover} />
                    <YAxis scale={y} x={0} label={withUnit(field, meta.field_units)} gridWidth={innerW} tickCount={layout === 'Overlay' ? 6 : 4} pick="field" />
                    <XAxis scale={x as any} y={innerH} format={(d: Date) => d3.timeFormat('%b %Y')(d)} tickCount={Math.max(2, Math.min(7, Math.floor(innerW / 80)))} />
                    <SeriesPoints series={panel.series} marked={marked} markReason={markReason} x={x} y={y} width={innerW} height={innerH} level={smooth ? 'dimmed' : 'standard'}
                      selected={hl.selected} dim={dim} dimKey={siteHover ?? ''} tip={tip} field={field}>
                      {panel.series.map((s) => (
                        <g key={s.name} opacity={dim(s.name)} data-export-site={s.name} data-export-color={s.color} pointerEvents="none">
                          {smooth && <path d={rollingPath(s.pts, line)} fill="none" stroke={s.color} strokeWidth={2.2} />}
                          {/* direct label at the end of each line in the overlay (the gallery's line-chart "design improvement": no legend lookup) */}
                          {layout === 'Overlay' && smooth && (() => {
                            const roll = rolling(s.pts)
                            const last = roll[roll.length - 1]
                            if (!last) return null
                            return (
                              <text x={x(last.date) - 4} y={y(last.value) - 8} textAnchor="end" fontSize={11} fontWeight={600} fill={s.color} stroke={INK.panel} strokeWidth={3} paintOrder="stroke" fontFamily={FONT.family}>
                                {s.name}
                              </text>
                            )
                          })()}
                        </g>
                      ))}
                    </SeriesPoints>
                    {cursorIn && cursor && (
                      <g pointerEvents="none" fontFamily={FONT.mono} fontSize={10.5}>
                        <line x1={x(cursor)} x2={x(cursor)} y1={0} y2={innerH} stroke={INK.text} strokeOpacity={0.5} strokeDasharray="3 3" />
                        {panel.series.map((s) => {
                          const p = nearest(s.pts, cursor)
                          if (!p) return null
                          const px = x(p.date)
                          const py = y(p.value)
                          const right = px < innerW - 90
                          return (
                            <g key={s.name} opacity={dim(s.name)} data-export-site={s.name} data-export-color={s.color}>
                              <circle cx={px} cy={py} r={5} fill={INK.panel} stroke={s.color} strokeWidth={2} />
                              <text x={right ? px + 9 : px - 9} y={py} dy="0.35em" textAnchor={right ? 'start' : 'end'} fill={s.color} stroke={INK.panel} strokeWidth={3} paintOrder="stroke" fontWeight={600}>
                                {fmt(p.value, 2)} · {d3.timeFormat('%d %b %y')(p.date)}
                              </text>
                            </g>
                          )
                        })}
                      </g>
                    )}
                  </g>
                </svg>
              </div>
            )
          })
        )}
        <Legend
          items={[
            ...meta.sites.filter((s) => rows.some((r) => r.site === s.name)).map((s) => ({ label: s.name, color: s.color })),
            ...(seasonShading ? meta.seasons.map((s) => ({ label: s.name, color: s.color, shape: 'band' as const })) : []),
          ]}
          hidden={hidden}
          onToggle={(l) => { if (meta.sites.some((s) => s.name === l)) setHidden((h) => toggleIn(h, l)) }}
          onHover={setHover}
          highlighted={siteHover ?? seasonHover}
          note="hover any panel: the dashed cursor is the same date on every panel"
        />
        {byGroup && missingNote('left out of the series')}
        <ExcludedNote fields={[field]} drawn />
        {tip.node}
      </div>
    </ChartFrame>
  )
}

/**
 * One panel's filters as points (components/ScatterLayer): drawn once per data, size or brush, so the date cursor
 * and a hover elsewhere do not redraw them; the nearest point within reach answers a hover or click. `children`
 * (the rolling medians and their labels) go over the points.
 */
function SeriesPoints({ series, marked, markReason, x, y, width, height, level, selected, dim, dimKey, tip, field, children }: {
  series: Series[]
  /** excluded filters under Filters → excluded: Mark, drawn as × where one of these series would hold them; never in the medians */
  marked: FilterRow[]; markReason: (id: string) => string | null
  x: d3.ScaleTime<number, number>; y: d3.ScaleLinear<number, number>; width: number; height: number
  level: DotLevel; selected: Set<string> | null
  /** a legend-hovered site dims the others; `dimKey` changes when it does (dim is a new function every render) */
  dim: (name: string) => number; dimKey: string
  tip: ReturnType<typeof useTooltip>; field: string; children?: ReactNode
}) {
  const items = useMemo(() => series.flatMap((s) => s.pts.map((p) => ({ p, s }))), [series])
  const xy = useMemo(() => items.map(({ p }) => [x(p.date), y(p.value)] as const), [items, x, y])
  const dots = useMemo(() => series.map((s) => (
    <g key={s.name} opacity={dim(s.name)} data-export-site={s.name} data-export-color={s.color}>
      {s.pts.map((p) => {
        const st = focusStyle(p.row.id, null, { r: 2.6, level }, selected)
        return <circle key={p.row.id} cx={x(p.date)} cy={y(p.value)} r={st.r} fill={s.color} fillOpacity={st.opacity} stroke={st.stroke} strokeWidth={st.strokeWidth} data-dot={st.dot} data-fid={p.row.id} />
      })}
    </g>
  )), [series, x, y, level, selected, dimKey]) // eslint-disable-line react-hooks/exhaustive-deps
  const marks = useMemo(() => {
    const [d0, d1] = x.domain(), [v0, v1] = y.domain()
    return toMarks(marked.filter((r) => series.some((s) => s.keep(r))), (r) => {
      const d = new Date(r.date + 'T00:00:00'), v = r[field] as number
      return d >= d0 && d <= d1 && v >= v0 && v <= v1 ? [x(d), y(v)] : null
    }, markReason, (r) => [`${field}: ${fmt(r[field] as number)}`])
  }, [marked, series, x, y, field, markReason])
  const focused = (id: string) => focusStyle(id, id, { r: 2.6, level })
  return (
    <ScatterLayer items={items} xy={xy} idOf={({ p }) => p.row.id} dots={dots} width={width} height={height} tip={tip} marks={marks}
      tipOf={({ p }) => [p.row.id, `${p.row.date} · ${seasonOnly(String(p.row.season ?? ''))}`, `${field}: ${fmt(p.value)}`, p.row.excluded ? `excluded: ${p.row.exclusion_reason}` : 'click for details']}
      focusR={({ p }) => focused(p.row.id).r}
      focus={({ p, s }, [cx, cy]) => {
        const st = focused(p.row.id)
        return <circle cx={cx} cy={cy} r={st.r} fill={s.color} fillOpacity={st.opacity} stroke={st.stroke} strokeWidth={st.strokeWidth} opacity={dim(s.name)} />
      }}>
      {children}
    </ScatterLayer>
  )
}
