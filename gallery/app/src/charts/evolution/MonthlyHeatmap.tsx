import { useMemo, useRef, useState } from 'react'
import * as d3 from 'd3'
import { calendarLabel, mixedCalendars, seasonsForSite } from '@/siteSeasons'
import { ChartFrame, Empty, Note, Segmented, Toggle } from '@/components/ChartFrame'
import { ColorLegend } from '@/components/ColorLegend'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { fmt } from '@/lib/stats'
import { INK, FONT, RAMP_SEQUENTIAL, withUnit } from '@/lib/theme'
import type { FilterRow, MetaFile } from '@/lib/types'

const STATS = ['Median', 'Mean', 'Count'] as const
type Stat = (typeof STATS)[number]
type Cell = { site: string; period: string; n: number; value: number | null }

/** Every 'YYYY-MM' from a to b inclusive, so a month with no filters is still a column. */
function monthRange(a: string, b: string): string[] {
  const out: string[] = []
  let y = +a.slice(0, 4)
  let m = +a.slice(5, 7)
  const yEnd = +b.slice(0, 4)
  const mEnd = +b.slice(5, 7)
  while (y < yEnd || (y === yEnd && m <= mEnd)) {
    out.push(`${y}-${String(m).padStart(2, '0')}`)
    m++
    if (m > 12) {
      m = 1
      y++
    }
  }
  return out
}

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
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)

  const [stat, setStat] = useState<Stat>('Median')
  const [seasonStrip, setSeasonStrip] = useState(true)

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
    const periods = dates.length ? monthRange(dates[0].slice(0, 7), dates[dates.length - 1].slice(0, 7)) : []
    // values per site-month; only filters that actually carry the field count
    const vals = new Map<string, number[]>()
    for (const r of rows) {
      const v = r[field]
      if (typeof v !== 'number' || !Number.isFinite(v)) continue
      const k = `${r.site}|${r.date.slice(0, 7)}`
      const arr = vals.get(k)
      if (arr) arr.push(v)
      else vals.set(k, [v])
    }
    const cells: Cell[] = []
    for (const site of sites) {
      for (const period of periods) {
        const arr = (vals.get(`${site}|${period}`) ?? []).sort(d3.ascending)
        const n = arr.length
        const value = n === 0 ? null : stat === 'Count' ? n : stat === 'Median' ? (d3.quantile(arr, 0.5) ?? null) : (d3.mean(arr) ?? null)
        cells.push({ site, period, n, value })
      }
    }
    const filled = cells.filter((c): c is Cell & { value: number } => c.value !== null)
    return { sites, periods, cells, filled }
  }, [rows, field, stat, meta.sites])
  const nEmpty = cells.length - filled.length

  const color = useMemo(() => {
    const sorted = filled.map((c) => c.value).sort(d3.ascending)
    const interp = d3.interpolateRgbBasis(RAMP_SEQUENTIAL.slice(2))
    if (stat === 'Count') return d3.scaleSequential(interp).domain([0, Math.max(1, d3.max(sorted) ?? 1)]).clamp(true)
    const lo = d3.quantile(sorted, 0.02) ?? 0
    const hi = d3.quantile(sorted, 0.98) ?? 1
    return d3.scaleSequential(interp).domain([lo, hi === lo ? lo + 1 : hi]).clamp(true)
  }, [filled, stat])

  // ---- geometry: the left gutter is measured from the longest site name so it never clips
  const labelPad = useMemo(() => {
    const longest = Math.max(4, ...sites.map((s) => s.length))
    return Math.min(200, Math.round(longest * 7) + 22)
  }, [sites])
  const stripH = 6
  const margin = { top: 12 + (seasonStrip && !mixedCalendars(meta) ? stripH + 6 : 0), right: 12, bottom: 40, left: labelPad }
  const rowH = 34
  const innerH = sites.length * rowH
  const innerW = Math.max(320, width - margin.left - margin.right)
  // the svg grows with the content rather than letting cells escape a narrow box
  const svgW = Math.max(width, margin.left + innerW + margin.right)
  const cellW = innerW / Math.max(1, periods.length)

  // one strip over the columns needs one calendar; per-site calendars get a thin strip under each site's row instead
  const mixed = mixedCalendars(meta)
  const seasonOf = (period: string, site?: string) => seasonsForSite(meta, site ?? meta.seasons[0]?.site).find((s) => s.months.includes(+period.slice(5, 7)))
  const topStrip = seasonStrip && !mixed
  const unit = withUnit(field, meta.field_units)
  const statLabel = stat === 'Count' ? 'filters' : `${stat.toLowerCase()} ${unit}`
  const fmtCell = (v: number) => (stat === 'Count' ? v.toFixed(0) : fmt(v, 2))

  // year labels: under the first column (if it is not a January) and under
  // every January, skipping any that would overprint the previous one
  const yearLabels = useMemo(() => {
    const out: { x: number; year: string }[] = []
    let lastX = -Infinity
    periods.forEach((p, i) => {
      const jan = p.endsWith('-01')
      if (!jan && i !== 0) return
      const x = i * cellW
      if (x - lastX < 34) return
      out.push({ x, year: p.slice(0, 4) })
      lastX = x
    })
    return out
  }, [periods, cellW])

  return (
    <ChartFrame
      id="monthly-heatmap"
      title="Monthly heatmap — site by month, coloured by the measurement"
      subtitle="Each cell is one site's month, coloured by the statistic of the measurement chosen in the bar. The coverage heatmap on the correlation tab counts filters; this one shows the value, which is what the notebooks' site×month pivot tables tabulate. Columns run without gaps from the first to the last date in the subset, so a pale cell is a low month and a blank one is a month with no filters."
      provenance="react-graph-gallery.com/heatmap · stands in for the site×month pivot tables in the notebooks"
      controls={
        <>
          <Segmented label="cell" value={stat} options={STATS} onChange={setStat} title="Statistic across the month's filters at that site" />
          <Toggle label="season strip" checked={seasonStrip} onChange={setSeasonStrip} title="A strip above the columns coloured by the selected season calendar" />
        </>
      }
    >
      <div ref={wrapRef} className="chart-wrap">
        {sites.length === 0 || periods.length === 0 ? (
          <Empty>No filters in this subset.</Empty>
        ) : filled.length === 0 ? (
          <Empty>No site has {field} in this subset.</Empty>
        ) : (
          <svg width={svgW} height={innerH + margin.top + margin.bottom} className="animated">
            <g transform={`translate(${margin.left},${margin.top})`} fontFamily={FONT.family}>
              {topStrip &&
                periods.map((p, i) => {
                  const s = seasonOf(p)
                  return (
                    <rect
                      key={p} x={i * cellW} y={-stripH - 6} width={Math.max(1, cellW - 0.6)} height={stripH}
                      fill={s?.color ?? INK.neutral} rx={1}
                      onMouseEnter={(e) => tip.show(e, [p, `season: ${s?.name ?? '—'} (${calendarLabel(meta)})`])}
                      onMouseLeave={tip.hide}
                    />
                  )
                })}
              {cells.map((c) => {
                const yi = sites.indexOf(c.site)
                const xi = periods.indexOf(c.period)
                const s = seasonOf(c.period, c.site)
                return (
                  <g key={`${c.site}|${c.period}`}>
                  {seasonStrip && mixed && <rect x={xi * cellW} y={yi * rowH + rowH - 4} width={Math.max(1, cellW - 0.6)} height={3} fill={s?.color ?? INK.neutral} pointerEvents="none" />}
                  <rect
                    x={xi * cellW} y={yi * rowH}
                    width={Math.max(1, cellW - 0.6)} height={rowH - 4}
                    fill={c.value === null ? INK.empty : color(c.value)} rx={1.5}
                    onMouseEnter={(e) =>
                      tip.show(e, [
                        c.site,
                        c.period,
                        c.value === null ? 'no filters' : `${statLabel} = ${fmtCell(c.value)}`,
                        `n = ${c.n} filter${c.n === 1 ? '' : 's'}`,
                        `season: ${s?.name ?? '—'} (${calendarLabel(meta)})`,
                      ])
                    }
                    onMouseLeave={tip.hide}
                  />
                  </g>
                )
              })}
              {/* a rule at every January so the years read even where the cells are a few px wide */}
              {periods.map((p, i) =>
                p.endsWith('-01') ? (
                  <line key={p} x1={i * cellW} x2={i * cellW} y1={topStrip ? -stripH - 6 : 0} y2={innerH - 4} stroke={INK.axis} strokeOpacity={0.45} strokeWidth={1} pointerEvents="none" />
                ) : null
              )}
              {sites.map((s, i) => (
                <text key={s} x={-10} y={i * rowH + (rowH - 4) / 2} dy="0.32em" textAnchor="end" fontSize={12} fill={INK.text}>
                  {s}
                </text>
              ))}
              {yearLabels.map((l) => (
                <text key={l.year + l.x} x={l.x + 3} y={innerH + 12} fontSize={10.5} fill={INK.muted} fontFamily={FONT.mono} textAnchor="start">
                  {l.year}
                </text>
              ))}
            </g>
          </svg>
        )}
        {filled.length > 0 && (
          <div className="legend">
            <ColorLegend
              scale={color}
              label={stat === 'Count' ? 'filters per month' : unit}
              format={stat === 'Count' ? (v) => v.toFixed(0) : undefined}
              note={stat === 'Count' ? undefined : 'ramp spans the 2–98 % of cell values; the ends are clamped'}
            />
          </div>
        )}
        {filled.length > 0 && nEmpty > 0 && (
          <Note>
            {nEmpty} of {cells.length} site-months have no filters with {field} and are drawn blank.
          </Note>
        )}
        {tip.node}
      </div>
    </ChartFrame>
  )
}
