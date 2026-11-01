import { useMemo, useRef } from 'react'
import { seasonOnly } from '@/lib/seasons'
import { SEASON_SHADE } from '@/components/SeasonBands'
import { robustDomain } from '@/lib/axes'
import * as d3 from 'd3'
import { mixedCalendars, seasonsForSite } from '@/siteSeasons'
import { ChartFrame, Empty, Note, Segmented, Toggle } from '@/components/ChartFrame'
import { SiteMonthMatrix } from '@/components/SiteMonthMatrix'
import { calendarMonths } from '@/lib/notebookAvailability'
import { useTooltip } from '@/hooks/useTooltip'
import { fmt } from '@/lib/stats'
import { INK, withUnit } from '@/lib/theme'
import type { FilterRow, MetaFile } from '@/lib/types'
import { useColourScheme } from '@/lib/colourSchemes'
import { fidsProps, useFilterEdits } from '@/lib/filterEdits'

import { MeasurementControl } from '@/components/AxisFieldPopover'
import { useTileSetting } from '@/lib/url'
import { ExcludedNote } from '@/lib/markedRows'

const STATS = ['Median', 'Mean', 'Count'] as const
type Stat = (typeof STATS)[number]
type Cell = { site: string; period: string; n: number; value: number | null; ids: string[] }

/**
 * Calendar heatmap — react-graph-gallery.com/heatmap with a continuous
 * colour legend. Site by year-month, coloured by the measurement.
 *
 * The coverage heatmap on the correlation tab answers "how many filters carry
 * this"; this one answers "how high was it", which is what the site×month
 * pivot tables in the notebooks tabulate. The columns are contiguous from the
 * first to the last date in the subset, so a gap in sampling is a run of
 * empty cells rather than a silently narrower table.
 *
 * The ramp spans the 2–98 % quantiles of the cell values and clamps, so one
 * hot month (a Delhi November, say) does not push every other cell to the
 * palest blue. Count uses a plain 0..max domain because counts have no tail
 * worth hiding.
 */
export function MonthlyHeatmap({ rows, meta, field }: { rows: FilterRow[]; meta: MetaFile; field: string }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const tip = useTooltip(wrapRef)

  const [stat, setStat] = useTileSetting<Stat>('monthly-heatmap.stat', 'Median')
  const [stripSetting, setStripSetting] = useTileSetting<string>('monthly-heatmap.seasons', '1')
  const seasonStrip = stripSetting === '1'
  const setSeasonStrip = (value: boolean) => setStripSetting(value ? '1' : '0')
  const [periodMode, setPeriodMode] = useTileSetting<'All time' | 'Month'>('monthly-heatmap.period', 'All time')
  const pooled = periodMode === 'Month'
  const periodOf = (date: string) => pooled ? date.slice(5, 7) : date.slice(0, 7)

  const { sites, periods, cells, filled } = useMemo(() => {
    const present = new Set(rows.map((r) => r.site))
    const sites = meta.sites.map((s) => s.name).filter((n) => present.has(n))
    // the month axis spans the field's own coverage, not every filter date: EC (FTIR)
    // starts in 2022, and a 2013 axis would leave most of the chart blank
    const dates = rows
      .filter((r) => typeof r[field] === 'number' && Number.isFinite(r[field]))
      .map((r) => r.date)
      .filter((d) => typeof d === 'string' && d.length >= 7)
      .sort()
    const periods = dates.length ? pooled ? d3.range(1, 13).map(m => String(m).padStart(2, '0')) : calendarMonths(dates[0].slice(0, 7), dates[dates.length - 1].slice(0, 7)) : []
    // values per site-month; only filters that actually carry the field count
    const vals = new Map<string, number[]>()
    const ids = new Map<string, string[]>()
    for (const r of rows) {
      const v = r[field]
      if (typeof v !== 'number' || !Number.isFinite(v)) continue
      const k = `${r.site}|${periodOf(r.date)}`
      const arr = vals.get(k)
      if (arr) arr.push(v)
      else vals.set(k, [v])
      ids.set(k, [...(ids.get(k) ?? []), r.id])
    }
    const cells: Cell[] = []
    for (const site of sites) {
      for (const period of periods) {
        const arr = (vals.get(`${site}|${period}`) ?? []).sort(d3.ascending)
        const n = arr.length
        const value = n === 0 ? null : stat === 'Count' ? n : stat === 'Median' ? (d3.quantile(arr, 0.5) ?? null) : (d3.mean(arr) ?? null)
        cells.push({ site, period, n, value, ids: ids.get(`${site}|${period}`) ?? [] })
      }
    }
    const filled = cells.filter((c): c is Cell & { value: number } => c.value !== null)
    return { sites, periods, cells, filled }
  }, [rows, field, stat, meta.sites, pooled])
  const nEmpty = cells.length - filled.length

  const colours = useColourScheme('monthly-heatmap', 'sequential')
  const edits = useFilterEdits()
  const color = useMemo(() => {
    const sorted = filled.map((c) => c.value).sort(d3.ascending)
    // the gallery blues start two steps in, so the faintest month still reads against white
    const interp = d3.interpolateRgbBasis(colours.name === 'Gallery' && !colours.reverse ? colours.stops.slice(2) : colours.stops)
    if (stat === 'Count') return d3.scaleSequential(interp).domain([0, Math.max(1, d3.max(sorted) ?? 1)]).clamp(true)
    const [lo, hi] = robustDomain(sorted, 0.02, 0.98).domain
    return d3.scaleSequential(interp).domain([lo, hi === lo ? lo + 1 : hi]).clamp(true)
  }, [filled, stat, colours.stops.join()]) // eslint-disable-line react-hooks/exhaustive-deps

  const stripH = 6

  // one strip over the columns needs one calendar; per-site calendars get a thin strip under each site's row instead
  const mixed = mixedCalendars(meta)
  const seasonOf = (period: string, site?: string) => seasonsForSite(meta, site ?? meta.seasons[0]?.site).find((s) => s.months.includes(+(pooled ? period : period.slice(5, 7))))
  const topStrip = seasonStrip && !mixed
  const unit = withUnit(field, meta.field_units)
  const statLabel = stat === 'Count' ? 'filters' : `${stat.toLowerCase()} ${unit}`
  const fmtCell = (v: number) => (stat === 'Count' ? v.toFixed(0) : fmt(v, 2))

  // a cell by site and period, for the matrix's rows and columns
  const byCell = useMemo(() => new Map(cells.map((c) => [`${c.site}|${c.period}`, c])), [cells])

  return (
    <ChartFrame
      id="monthly-heatmap"
      title="Monthly heatmap — site by month, coloured by the measurement"
      subtitle="Each cell is one site's month, coloured by the statistic of the measurement chosen in the bar. The coverage heatmap on the correlation tab counts filters; this one shows the value, which is what the notebooks' site×month pivot tables tabulate. Columns run without gaps from the first to the last date in the subset, so a pale cell is a low month and a blank one is a month with no filters."
      provenance="react-graph-gallery.com/heatmap · stands in for the site×month pivot tables in the notebooks"
      controls={
        <>
          <MeasurementControl />
          <Segmented label="Period" value={periodMode} options={['All time', 'Month'] as const} onChange={setPeriodMode} title="All time: consecutive year-months. Month: pool each calendar month across all years." />
          <Segmented label="Cell" value={stat} options={STATS} onChange={setStat} title="Statistic across the month's filters at that site" />
          <Toggle label="Season strip" checked={seasonStrip} onChange={setSeasonStrip} title="A strip above the columns coloured by the selected season calendar" />
        </>
      }
    >
      <p className="facet-title">{statLabel}{pooled ? " · calendar months pooled across years" : " · consecutive year-months"}</p>
      <div ref={wrapRef}>
        {sites.length === 0 || periods.length === 0 ? (
          <Empty>No filters with {field} in this subset. Choose another measurement or edit the filters.</Empty>
        ) : filled.length === 0 ? (
          <Empty>No site has {field} in this subset.</Empty>
        ) : (
          <SiteMonthMatrix rows={sites} months={periods} colLabelPosition="bottom" cellWidth={[pooled ? 28 : 8, 1000]} cellHeight={34} gap={[0.6, 4]} radius={1.5}
            rowLabelSize={12} cellLabels="never" color={color} fitRows={[10, 34]} before={topStrip ? stripH + 6 : 0} ariaLabel="Monthly heatmap, site by month"
            cell={(i, period) => {
              const c = byCell.get(`${sites[i]}|${period}`)!
              const s = seasonOf(period, c.site)
              // a cell is a site's month of filters: click lists them in the side panel, right-click excludes or keeps them
              return { value: c.value, props: c.ids.length ? fidsProps(c.ids, `${c.site} · ${c.period}`) : undefined,
                tip: [c.site, c.period, c.value === null ? 'no filters' : `${statLabel} = ${fmtCell(c.value)}`, `n = ${c.n} filter${c.n === 1 ? '' : 's'}`,
                  ...(c.n ? ['click to list them · right-click to exclude or keep'] : []), `season: ${s ? seasonOnly(s.name) : '—'}`] }
            }}
            onCell={(i, j) => {
              const c = byCell.get(`${sites[i]}|${periods[j]}`)
              if (c?.ids.length) edits.openGroup({ title: `${c.site} · ${c.period}`, ids: c.ids, fields: [field] })
            }}
            legend={{ scale: color, scheme: colours.scheme, label: stat === 'Count' ? 'filters per month' : unit, format: stat === 'Count' ? (v) => v.toFixed(0) : undefined,
              note: stat === 'Count' ? undefined : 'ramp spans the 2–98 % of cell values; the ends are clamped' }}
            footer={<>
              {nEmpty > 0 && <Note>{nEmpty} of {cells.length} site-months have no filters with {field} and are drawn blank.</Note>}
              <ExcludedNote fields={[field]} />
            </>}>
            {(l) => <g pointerEvents={topStrip ? undefined : 'none'}>
              {/* the season strip: one over the columns for one calendar, a thin one under each site's row for several */}
              {topStrip && periods.map((p, j) => {
                const s = seasonOf(p)
                return <rect key={p} x={l.x(j)} y={l.top - stripH - 6} width={Math.max(1, l.cellW - 0.6)} height={stripH} fill={s?.color ?? INK.neutral} fillOpacity={SEASON_SHADE.strip} rx={1}
                  onMouseEnter={(e) => tip.show(e, [p, `season: ${s ? seasonOnly(s.name) : '—'}`])} onMouseLeave={tip.hide} />
              })}
              {seasonStrip && mixed && sites.flatMap((site, i) => periods.map((p, j) => {
                const s = seasonOf(p, site)
                return <rect key={`${site}|${p}`} x={l.x(j)} y={l.y(i) + l.cellH - 4} width={Math.max(1, l.cellW - 0.6)} height={3} fill={s?.color ?? INK.neutral} fillOpacity={SEASON_SHADE.strip} />
              }))}
            </g>}
          </SiteMonthMatrix>
        )}
        {tip.node}
      </div>
    </ChartFrame>
  )
}
