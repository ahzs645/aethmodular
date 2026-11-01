import { useId, useMemo, useRef } from 'react'
import { seasonOnly, shortSeason } from '@/lib/seasons'
import { seasonShade } from '@/components/SeasonBands'
import * as d3 from 'd3'
import { ChartFrame, Empty, Note } from '@/components/ChartFrame'
import { FacetGrid } from '@/components/FacetGrid'
import { Legend, useLegend } from '@/components/Legend'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { fmt } from '@/lib/stats'
import { IQR_MIN_N, monthlyStats, type MonthStats } from '@/lib/stats'
import { INK, FONT, withUnit, MONTHS } from '@/lib/theme'
import type { FilterRow, MetaFile, SeasonMeta } from '@/lib/types'
import { useChartOptions } from '@/lib/chartOptions'
import { calendarLabel, mixedCalendars, seasonsForSite } from '@/siteSeasons'
import { MeasurementControl } from '@/components/AxisFieldPopover'
import { useFitHeight } from '@/lib/chartFit'
import { ExcludedNote } from '@/lib/markedRows'

const STATS = ['Median', 'Mean'] as const
const LAYOUTS = ['Per site', 'Grouped', 'Rings'] as const
const MAX_RINGS = 8
/** A site-month with fewer filters than this is a stub, not a value: a median of two is a coin toss. */
const MIN_N = 3
const TAU = 2 * Math.PI

type Cell = MonthStats
type Series = { name: string; color: string; cells: Cell[] }
type Stat = (typeof STATS)[number]
const valueOf = (c: Cell, stat: Stat) => c.n >= MIN_N ? (stat === 'Median' ? c.median : c.mean) : null

/** Angle (radians, clockwise from 12 o'clock — d3.arc's convention) at which month m (1–12) starts. */
const monthStart = (m: number) => ((m - 1) * TAU) / 12

/** A point on the circle at radius r and clock angle a (radians from the top, clockwise). */
const polar = (r: number, a: number): [number, number] => [r * Math.sin(a), -r * Math.cos(a)]

/**
 * Splits a season's months into runs that are contiguous *around the circle*,
 * so Oct–Feb is one run crossing the Dec/Jan seam rather than two (Oct–Dec
 * and Jan–Feb). Returned as { start, length } with start in 1–12.
 */
function circularRuns(months: number[]): { start: number; length: number }[] {
  const set = new Set(months)
  if (set.size === 0) return []
  if (set.size === 12) return [{ start: 1, length: 12 }]
  const prev = (m: number) => (m === 1 ? 12 : m - 1)
  const next = (m: number) => (m === 12 ? 1 : m + 1)
  const runs: { start: number; length: number }[] = []
  // a run starts at any month whose predecessor is not in the season
  for (const m of [...set].sort(d3.ascending)) {
    if (set.has(prev(m))) continue
    let length = 1
    let cur = m
    while (set.has(next(cur))) {
      cur = next(cur)
      length++
    }
    runs.push({ start: m, length })
  }
  return runs
}

/**
 * Circular barplot as a seasonal clock — react-graph-gallery.com/circular-barplot.
 *
 * The monthly band chart lays the year on a straight axis, which puts
 * December and January at opposite ends although they are neighbours in the
 * same dry season. On a circle the dry season (Oct–Feb under `dry_feb`) is
 * one contiguous wedge and the eye reads it as such. The season sectors come
 * from meta.seasons, i.e. the calendar picked in the subset bar, so switching
 * to `belg_feb` visibly moves February out of the dry wedge.
 *
 * `Per site` gives each site its own clock and season calendar on a shared
 * value scale. `Grouped` puts sites side by side inside each month's sector;
 * `Rings` gives each site its own ring and baseline on that same value scale.
 */
export function SeasonalClock({ rows, meta, field }: { rows: FilterRow[]; meta: MetaFile; field: string }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)

  // sites are series; seasons are background keys (hover deepens their sectors) that also hide their months
  const lg = useLegend(undefined, { background: meta.seasons.map((s) => s.name), hideBackground: true })
  const hidden = lg.hidden

  const allSeries: Series[] = useMemo(
    () =>
      meta.sites
        .map((s) => {
          const cells: Cell[] = monthlyStats(rows.filter((r) => r.site === s.name), (r) => r.month, (r) => r[field])
          return { name: s.name, color: s.color, cells }
        })
        .filter((s) => s.cells.some((c) => c.n > 0)),
    [rows, field, meta.sites]
  )
  const series = useMemo(() => allSeries.filter(s => !hidden.has(s.name)).map(s => ({
    ...s,
    cells: s.cells.filter(c => !hidden.has(seasonsForSite(meta, s.name).find(season => season.months.includes(c.month))?.name ?? '')),
  })), [allSeries, hidden, meta])
  // Keep the layout choices stable while hiding sites in the legend. Only Rings
  // needs an upper site limit; the per-site grid can wrap to as many rows as needed.
  const o = useChartOptions('clock', [
    { key: 'seasons', label: 'Season overlay', toggle: true, default: false },
    { key: 'stat', label: 'Bar', choices: STATS, default: 'Median', title: "Statistic of the measurement across the month's filters" },
    { key: 'layout', label: 'Layout', default: 'Per site', hidden: 'Grouped',
      choices: LAYOUTS.map(value => ({ value, when: c => value !== 'Rings' || (c.nSeries as number) <= MAX_RINGS })),
      when: c => (c.nSeries as number) >= 2,
      title: 'Per site: a separate clock and season calendar for each site. Grouped: sites side by side in each month. Rings: one ring per site. All layouts share one value scale' },
  ], { nSeries: allSeries.length })
  const stat = o.values.stat as Stat
  const layout = o.values.layout as (typeof LAYOUTS)[number]
  // a hovered site dims the other sites' bars; a hovered season (a background key, not a series) deepens its own sectors
  const siteHover = lg.hover
  const seasonHover = lg.backgroundHover

  const fit = useFitHeight()
  const perSite = layout === 'Per site'
  const mixed = mixedCalendars(meta)
  const emphasizeFebruary = meta.season_convention !== 'local'
  const unit = withUnit(field, meta.field_units)
  const stubCount = series.reduce((acc, s) => acc + s.cells.filter(c => c.n > 0 && c.n < MIN_N).length, 0)
  const domain = useMemo<[number, number]>(() => {
    const valued = series.flatMap(s => s.cells.map(c => valueOf(c, stat))).filter((v): v is number => v !== null)
    // EC (FTIR) can be negative near the detection limit; retain those values.
    const min = Math.min(0, d3.min(valued) ?? 0)
    return [min, Math.max(min + 1e-9, (d3.max(valued) ?? 1) * 1.05)]
  }, [series, stat])
  const clock = (shown: Series[], cellWidth: number, cellHeight: number | null) => (
    <ClockPlot series={shown} meta={meta} stat={stat} layout={layout === 'Rings' ? 'Rings' : 'Grouped'}
      size={cellHeight === null ? Math.max(240, Math.min(cellWidth, 520)) : Math.max(160, Math.min(cellWidth, cellHeight))}
      domain={domain} seasons={!!o.values.seasons} hidden={hidden} siteHover={siteHover} seasonHover={seasonHover} unit={unit} tip={tip} />
  )

  return (
    <ChartFrame
      id="seasonal-clock"
      title="Seasonal clock — the year as a circle"
      subtitle={`Monthly statistic of the measurement, laid around a circle so December and January sit next to each other. ${perSite ? `Each site has its own clock; shaded sectors use ${calendarLabel(meta)}. All clocks share one value scale.` : mixed ? 'Season shading follows each site within its month sector or ring; use Per site to see a separate clock for each calendar.' : `Shaded sectors use ${calendarLabel(meta)}.`}`}
      provenance="react-graph-gallery.com/circular-barplot · not in the estate"
      controls={<><MeasurementControl />{o.controls()}</>}
    >
      <p className="facet-title">{stat} {unit}</p>
      <div ref={wrapRef} className={`chart-wrap centered${fit === null || perSite ? '' : ' fit-square'}`} style={fit === null ? undefined : { minHeight: fit }}>
        {allSeries.length === 0 ? (
          <Empty>No site has {field} in this subset.</Empty>
        ) : series.length === 0 ? (
          <Empty>All sites are hidden. Select a site in the legend to show its clock.</Empty>
        ) : perSite ? (
          <FacetGrid items={series} minCell={240} maxCell={520} maxCols={4}
            title={s => s.name} label={s => `${s.name} seasonal clock`}
            render={(s, cellWidth, _, cell) => clock([s], cellWidth, cell.height)} />
        ) : clock(series, width, fit)}
        {stubCount > 0 && (
          <Note>
            {stubCount} site-month{stubCount === 1 ? '' : 's'} with fewer than {MIN_N} filters {stubCount === 1 ? 'is' : 'are'} drawn as a hollow stub rather than a {stat.toLowerCase()}.
          </Note>
        )}
        <Legend
          items={[
            ...allSeries.map((s) => ({ label: s.name, color: s.color })),
            ...(o.values.seasons ? meta.seasons : []).map((s) => ({ label: s.name, color: s.color, shape: 'band' as const })),
          ]}
          {...lg.props}
          note={emphasizeFebruary ? `shared value scale · February changes bins across Ethiopian conventions` : `shared value scale · ${calendarLabel(meta)}`}
        />
        <ExcludedNote fields={[field]} />
        {tip.node}
      </div>
    </ChartFrame>
  )
}

/** The same radial plot serves combined layouts and each site facet. */
function ClockPlot({ series, meta, stat, layout, size, domain, seasons, hidden, siteHover, seasonHover, unit, tip }: {
  series: Series[]; meta: MetaFile; stat: Stat; layout: 'Grouped' | 'Rings'; size: number
  domain: [number, number]; seasons: boolean; hidden: Set<string>
  siteHover: string | null; seasonHover: string | null; unit: string
  tip: Pick<ReturnType<typeof useTooltip>, 'bind'>
}) {
  // Each facet needs its own textPath ids, including when two chart tiles are open.
  const uid = useId().replace(/:/g, '')
  const dim = (name: string) => siteHover === null || siteHover === name ? 1 : 0.15
  // ---- geometry: a square, with room outside the ring for month and season labels
  const cx = size / 2
  const cy = size / 2
  // the room outside the ring shrinks with the clock: season names need ~64 px, month names ~30,
  // single-letter months ~18; a small clock (a dashboard tile) keeps the ring and drops the rest
  const seasonNames = size >= 340
  const labelRoom = seasonNames ? 64 : size >= 220 ? 30 : 18
  const outerR = size / 2 - labelRoom
  const hole = Math.round(outerR * 0.24)
  const monthLabelR = outerR + 14
  const seasonLabelR = outerR + 38

  // Grouped: one radial scale from the hole to the rim. Rings: the same value
  // domain mapped into each ring's band, so a tall bar means the same thing
  // on every ring even though each ring has its own baseline.
  const ringGap = 5
  const band = layout === 'Rings' && series.length ? (outerR - hole) / series.length : outerR - hole
  const ringInner = (i: number) => (layout === 'Rings' ? hole + i * band : hole)
  const ringOuter = (i: number) => ringInner(i) + band - (layout === 'Rings' ? ringGap : 0)
  const rel = d3.scaleLinear().domain(domain).range([0, band - (layout === 'Rings' ? ringGap : 0)]).nice()
  const rOf = (i: number, v: number) => ringInner(i) + rel(v)
  const ticks = rel.ticks(3).filter((t) => t !== 0 && t >= rel.domain()[0] && t <= rel.domain()[1])

  // within a month's sector, the sub-sector for each site (Grouped only)
  const sectorPad = TAU / 12 / 14
  const subBand = d3.scaleBand<string>().domain(series.map((s) => s.name)).range([sectorPad, TAU / 12 - sectorPad]).paddingInner(0.14)

  const arc = d3.arc<{ r0: number; r1: number; a0: number; a1: number }>()
    .innerRadius((d) => d.r0)
    .outerRadius((d) => d.r1)
    .startAngle((d) => d.a0)
    .endAngle((d) => d.a1)

  // A separate clock can show the site's full season wedges and labels.
  const mixed = series.length > 1 && mixedCalendars(meta)
  const panelSeasons = series.length === 1 ? seasonsForSite(meta, series[0].name) : meta.seasons
  const wedgeSeasons = mixed || !seasons ? [] : panelSeasons.filter(s => !hidden.has(s.name))
  const seasonOf = (m: number, site?: string): SeasonMeta | undefined =>
    (site ? seasonsForSite(meta, site) : wedgeSeasons).find((s) => s.months.includes(m))
  const emphasizeFebruary = meta.season_convention !== 'local'

  /** The arc path a season label rides on. Top-half labels run clockwise and sit outside the path; bottom-half labels run the other way so they are not upside down, and hang inside it. */
  const labelPath = (start: number, length: number) => {
    const a0 = monthStart(start)
    const a1 = a0 + (length * TAU) / 12
    const midDeg = (((a0 + a1) / 2) * 180) / Math.PI % 360
    const bottom = midDeg > 90 && midDeg < 270
    const r = bottom ? seasonLabelR + 9 : seasonLabelR
    const [x0, y0] = polar(r, bottom ? a1 : a0)
    const [x1, y1] = polar(r, bottom ? a0 : a1)
    const large = a1 - a0 > Math.PI ? 1 : 0
    return `M${x0},${y0} A${r},${r} 0 ${large} ${bottom ? 0 : 1} ${x1},${y1}`
  }

  // Recreate the SVG on resize: WebKit can retain stale textPath geometry.
  return (
    <svg key={size} role="img" aria-label={`${series.map(s => s.name).join(', ')} seasonal clock`} width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="animated">
      <defs>
        {wedgeSeasons.flatMap((s) =>
          circularRuns(s.months).map((run, i) => (
            <path key={`${s.name}-${i}`} id={`${uid}-season-${s.name.replace(/\W+/g, '')}-${i}`} d={labelPath(run.start, run.length)} fill="none" />
          ))
        )}
      </defs>
      <g transform={`translate(${cx},${cy})`} fontFamily={FONT.family}>
        {/* season sectors behind everything, from just inside the hole to just past the rim */}
        {d3.range(1, 13).map((m) => {
          const s = seasonOf(m)
          if (!s || hidden.has(s.name)) return null
          return <path key={m} data-export-season={s.name} data-export-color={s.color} d={arc({ r0: Math.max(0, hole - 6), r1: outerR + 4, a0: monthStart(m), a1: monthStart(m + 1) }) ?? ''} fill={s.color} fillOpacity={seasonShade(s.name, seasonHover)} pointerEvents="none" />
        })}
        {mixed && seasons && series.flatMap((site, i) => d3.range(1, 13).map(m => {
          const season = seasonOf(m, site.name)
          const start = monthStart(m)
          const a0 = layout === 'Rings' ? start : start + (subBand(site.name) ?? 0)
          const a1 = layout === 'Rings' ? monthStart(m + 1) : a0 + subBand.bandwidth()
          return <path key={`${site.name}-${m}`} data-export-season={season?.name} data-export-color={season?.color} d={arc({ r0: ringInner(i), r1: ringOuter(i), a0, a1 }) ?? ''} fill={season?.color ?? INK.empty} fillOpacity={season && hidden.has(season.name) ? 0 : 0.2} pointerEvents="none" />
        }))}
        {/* month spokes */}
        {d3.range(1, 13).map((m) => {
          const [x0, y0] = polar(Math.max(0, hole - 6), monthStart(m))
          const [x1, y1] = polar(outerR + 4, monthStart(m))
          return <line key={m} x1={x0} y1={y0} x2={x1} y2={y1} stroke={INK.grid} strokeWidth={1} pointerEvents="none" />
        })}

        {/* radial axis: baseline per ring, faint tick rings, values printed once along the January spoke */}
        {(layout === 'Rings' ? series : series.slice(0, 1)).map((s, i) => (
          <g key={s.name} pointerEvents="none" data-export-site={s.name} data-export-color={s.color}>
            <circle r={ringInner(i)} fill="none" stroke={INK.axis} strokeOpacity={0.55} strokeWidth={1} />
            {ticks.map((t) => (
              <circle key={t} r={rOf(i, t)} fill="none" stroke={INK.axis} strokeOpacity={0.28} strokeWidth={0.8} strokeDasharray="2 3" />
            ))}
          </g>
        ))}
        {series.length > 0 &&
          ticks.map((t) => {
            // labels go on the outermost ring only; every ring shares the scale
            const i = layout === 'Rings' ? series.length - 1 : 0
            return (
              <text key={t} x={4} y={-rOf(i, t)} dy="0.35em" textAnchor="start" fontSize={9.5} fontFamily={FONT.mono} fill={INK.muted} stroke={INK.panel} strokeWidth={3} paintOrder="stroke" pointerEvents="none">
                {fmt(t, Math.abs(t) >= 10 ? 0 : 1)}
              </text>
            )
          })}

        {/* the bars */}
        {series.map((s, i) => (
          <g key={s.name} opacity={dim(s.name)} data-export-site={s.name} data-export-color={s.color}>
            {s.cells.map((c) => {
              const a0 = layout === 'Grouped' ? monthStart(c.month) + (subBand(s.name) ?? 0) : monthStart(c.month) + sectorPad
              const a1 = layout === 'Grouped' ? a0 + subBand.bandwidth() : monthStart(c.month + 1) - sectorPad
              const season = seasonOf(c.month, s.name)
              const seasonLine = `season: ${season ? seasonOnly(season.name) : '—'}`
              if (c.n === 0) return null
              const v = valueOf(c, stat)
              if (v === null) {
                // fewer than MIN_N filters: a hollow stub says "there is something here" without asserting a value
                const stub = Math.min(10, band * 0.12)
                return (
                  <path
                    key={c.month}
                    d={arc({ r0: ringInner(i), r1: ringInner(i) + stub, a0, a1 }) ?? ''}
                    fill="none" stroke={s.color} strokeWidth={1.2} strokeDasharray="2 2"
                    {...tip.bind([`${s.name} · ${MONTHS[c.month - 1]}`, `only ${c.n} filter${c.n === 1 ? '' : 's'} — no ${stat.toLowerCase()} drawn (needs ${MIN_N})`, seasonLine])}
                  />
                )
              }
              // bars grow from the ring's zero; a negative statistic grows inward from it
              const base = rOf(i, 0)
              const tipR = rOf(i, v)
              return (
                <path
                  key={c.month}
                  d={arc({ r0: Math.min(base, tipR), r1: Math.max(base, tipR), a0, a1 }) ?? ''}
                  fill={s.color} fillOpacity={0.88} stroke={INK.panel} strokeWidth={0.6}
                  style={{ cursor: 'default' }}
                  {...tip.bind([
                    `${s.name} · ${MONTHS[c.month - 1]}`,
                    `${stat.toLowerCase()} ${unit} = ${fmt(v, 2)}`,
                    `n = ${c.n} filters`,
                    c.q1 !== null ? `IQR ${fmt(c.q1, 2)} – ${fmt(c.q3, 2)}` : `IQR n/a (n<${IQR_MIN_N})`,
                    seasonLine,
                  ])}
                />
              )
            })}
          </g>
        ))}

        {/* month labels just outside the rim, January at the top */}
        {MONTHS.map((name, idx) => {
          const [x, y] = polar(monthLabelR, monthStart(idx + 1) + TAU / 24)
          return (
            <text key={name} x={x} y={y} dy="0.35em" textAnchor="middle" fontSize={11} fill={emphasizeFebruary && idx === 1 ? INK.text : INK.muted} fontWeight={emphasizeFebruary && idx === 1 ? 600 : 400} pointerEvents="none">
              {size >= 220 ? name : name[0]}
            </text>
          )
        })}

        {layout === 'Rings' && series.map((s, i) => <text key={`label-${s.name}`} x={-6} y={-(ringInner(i) + ringOuter(i)) / 2} dy="0.35em" textAnchor="end" fontSize={10.5} fontWeight={600} fill={s.color} opacity={dim(s.name)} stroke={INK.panel} strokeWidth={4} paintOrder="stroke" pointerEvents="none">{s.name}</text>)}
        {/* season names curved along the outside of their wedge */}
        {seasonNames && wedgeSeasons.flatMap((s) =>
          circularRuns(s.months).map((_, i) => (
            <text key={`${s.name}-${i}`} data-export-season={s.name} data-export-color={s.color} fontSize={10.5} fontWeight={600} fill={s.color} opacity={seasonHover === null || seasonHover === s.name ? 1 : 0.35} pointerEvents="none">
              <textPath href={`#${uid}-season-${s.name.replace(/\W+/g, '')}-${i}`} startOffset="50%" textAnchor="middle">
                {shortSeason(s.name)}
              </textPath>
            </text>
          ))
        )}
      </g>
    </svg>
  )
}
