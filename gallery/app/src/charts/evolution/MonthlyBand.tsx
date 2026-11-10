import { useMemo, useRef } from 'react'
import { seasonOnly } from '@/lib/seasons'
import { SeasonBands, monthSeasonBands } from '@/components/SeasonBands'
import * as d3 from 'd3'
import { ChartFrame, Empty } from '@/components/ChartFrame'
import { useChartOptions } from '@/lib/chartOptions'
import { Legend, useLegend } from '@/components/Legend'
import { XAxis, YAxis } from '@/components/Axes'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { fmt } from '@/lib/stats'
import { monthlyStats } from '@/lib/stats'
import { INK, MARGIN, withUnit, MONTHS } from '@/lib/theme'
import type { FilterRow, MetaFile } from '@/lib/types'
import { calendarLabel, mixedCalendars, seasonsForSite } from '@/siteSeasons'
import { useDashboardDetails } from '@/lib/dashboardDetails'
import { measuredRows } from '@/lib/metricDefinition'
import { useFitHeight } from '@/lib/chartFit'
import { ExcludedNote } from '@/lib/markedRows'


/**
 * Area chart with an uncertainty band — the `fill_between + plot` recipe,
 * 59 figures in the estate and the fourth-largest family.
 *
 * Monthly climatology: median line, interquartile band. February is drawn
 * with an explicit marker because the two published Ethiopian calendars
 * disagree about which season owns it, and it is a high-BC month — so the
 * seasonal means are not interchangeable between conventions (AGENTS.md).
 */
export function MonthlyBand({ rows, meta, field }: { rows: FilterRow[]; meta: MetaFile; field: string }) {
  const details = useDashboardDetails()
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
  const height = useFitHeight() ?? 420

  // sites are series; seasons are background keys (hover deepens their shading, they do not hide)
  const lg = useLegend(undefined, { background: meta.seasons.map((s) => s.name) })
  const hidden = lg.hidden
  const o = useChartOptions('monthly-band', [{ key: 'bands', label: 'IQR bands', toggle: true, default: true }])
  const bands = o.values.bands as boolean

  const allSeries = useMemo(
    () =>
      meta.sites
        .map((s) => {
          // median per month, the IQR band once a month has IQR_MIN_N filters (as the seasonal clock)
          const points = monthlyStats(rows.filter((r) => r.site === s.name), (r) => r.month, (r) => r[field])
          return { name: s.name, color: s.color, points }
        })
        .filter((s) => s.points.some((p) => p.median !== null)),
    [rows, field, meta.sites]
  )
  const series = allSeries.filter((s) => !hidden.has(s.name))
  // a hovered site dims the other sites; a hovered season (a background key, not a series) deepens its own shading
  const seasonHover = lg.backgroundHover
  const dim = lg.dim

  const innerW = Math.max(240, width - MARGIN.left - MARGIN.right)
  const innerH = height - MARGIN.top - MARGIN.bottom

  const x = d3.scalePoint<number>().domain(d3.range(1, 13)).range([0, innerW])
  const allHi = series.flatMap((s) => s.points.map((p) => (bands ? p.q3 : null) ?? p.median ?? 0))
  const allLo = series.flatMap((s) => s.points.map((p) => (bands ? p.q1 : null) ?? p.median ?? 0))
  const y = d3.scaleLinear().domain([Math.min(0, d3.min(allLo) ?? 0), (d3.max(allHi) ?? 1) * 1.08]).range([innerH, 0]).nice()

  // per-site calendars cannot share one month axis's shading: shade only when one calendar is in force
  const mixed = mixedCalendars(meta)
  const seasonOf = (m: number, site?: string) => seasonsForSite(meta, site ?? meta.seasons[0]?.site).find((s) => s.months.includes(m))
  const emphasizeFebruary = meta.season_convention !== 'local'

  return (
    <ChartFrame
      id="monthly"
      title="Area chart — monthly climatology with an IQR band"
      subtitle={`Median by calendar month with the interquartile range shaded. ${mixed ? 'Each site keeps its own calendar, so the month axis is left unshaded; hover a point for its local season.' : `Season shading uses the ${calendarLabel(meta)}.`}${emphasizeFebruary ? ' February is flagged because the two Ethiopian conventions assign it to different seasons.' : ''}`}
      provenance="stands in for 59 fill_between+plot figures · react-graph-gallery.com/area-plot"
      controls={o.controls()}
    >
      <div ref={wrapRef} className="chart-wrap">
        {series.length === 0 ? (
          <Empty>No site has {field} in this subset.</Empty>
        ) : (
          <svg width={width} height={height} className="animated">
            <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
              {!mixed && <SeasonBands bands={monthSeasonBands((m) => x(m) ?? 0, (m) => seasonOf(m), innerW / 11, innerW)} height={innerH} hover={seasonHover} />}
              <YAxis scale={y} x={0} label={withUnit(field, meta.field_units)} gridWidth={innerW} pick="field" />
              <XAxis scale={x as any} y={innerH} format={(m: number) => MONTHS[m - 1]} tickCount={12} />

              {series.map((s) => {
                const withBand = s.points.filter((p) => p.q1 !== null && p.q3 !== null)
                const area = d3.area<(typeof withBand)[number]>().x((p) => x(p.month) ?? 0).y0((p) => y(p.q1!)).y1((p) => y(p.q3!)).curve(d3.curveMonotoneX)
                const withMedian = s.points.filter((p) => p.median !== null)
                const line = d3.line<(typeof withMedian)[number]>().x((p) => x(p.month) ?? 0).y((p) => y(p.median!)).curve(d3.curveMonotoneX)
                return (
                  <g key={s.name} opacity={dim(s.name)} data-export-site={s.name} data-export-color={s.color}>
                    {bands && <path d={area(withBand) ?? ''} fill={s.color} fillOpacity={0.18} pointerEvents="none" />}
                    <path d={line(withMedian) ?? ''} fill="none" stroke={s.color} strokeWidth={2.4} pointerEvents="none" />
                    {withMedian.map((p) => (
                      <circle
                        key={p.month}
                        role={details ? 'button' : undefined} tabIndex={details ? 0 : undefined}
                        aria-label={details ? `${s.name} ${MONTHS[p.month - 1]}: show ${p.n} contributing filters` : undefined}
                        onClick={details ? () => { tip.hide(); details({ title: `${s.name} · ${MONTHS[p.month - 1]} · ${field}`, rows: measuredRows(rows.filter(r => r.site === s.name && r.month === p.month), [field]), fields: [field] }) } : undefined}
                        onKeyDown={details ? e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); details({ title: `${s.name} · ${MONTHS[p.month - 1]} · ${field}`, rows: measuredRows(rows.filter(r => r.site === s.name && r.month === p.month), [field]), fields: [field] }) } } : undefined}
                        cx={x(p.month) ?? 0} cy={y(p.median!)} r={emphasizeFebruary && p.month === 2 ? 5.5 : 3.6}
                        fill={emphasizeFebruary && p.month === 2 ? INK.panel : s.color}
                        stroke={p.month === 2 ? INK.text : s.color} strokeWidth={p.month === 2 ? 2 : 1}
                        {...tip.bind([
                          `${s.name} · ${MONTHS[p.month - 1]}`,
                          `median = ${fmt(p.median)}`,
                          p.q1 !== null ? `IQR ${fmt(p.q1)} – ${fmt(p.q3)}` : 'IQR n/a (n<4)',
                          `n = ${p.n}`,
                          `season: ${seasonOnly(seasonOf(p.month, s.name)?.name ?? '—')}`,
                        ])}
                      />
                    ))}
                  </g>
                )
              })}
            </g>
          </svg>
        )}
        {series.length > 0 && <Legend
          items={[
            ...allSeries.map((s) => ({ label: s.name, color: s.color, shape: 'line' as const })),
            ...(mixed ? [] : meta.seasons.map((s) => ({ label: s.name, color: s.color, shape: 'band' as const }))),
          ]}
          {...lg.props}
          note={emphasizeFebruary ? 'hollow marker = February (convention-dependent)' : calendarLabel(meta)}
        />}
        <ExcludedNote fields={[field]} />
        {tip.node}
      </div>
    </ChartFrame>
  )
}
