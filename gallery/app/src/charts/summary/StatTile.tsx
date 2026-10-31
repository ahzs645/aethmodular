import { useMemo, useRef } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty } from '@/components/ChartFrame'
import { prettyUnit } from '@/components/FieldSelect'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { useFitHeight } from '@/lib/chartFit'
import { chemText, fmt } from '@/lib/format'
import { useDashboardDetails } from '@/lib/dashboardDetails'
import { measuredRows } from '@/lib/metricDefinition'
import type { FilterRow, MetaFile } from '@/lib/types'

/** sites drawn as their own line; more than this and the sparkline is the network's median */
const MAX_LINES = 8

/**
 * A headline number for a dashboard: one field's median over the filters in view, its
 * interquartile range and count, and a sparkline of the monthly medians (one line per site,
 * in the site's colour, while few sites are in view).
 */
export function StatTile({ rows, meta, field }: { rows: FilterRow[]; meta: MetaFile; field: string }) {
  const details = useDashboardDetails()
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
  const fit = useFitHeight()
  const sparkH = Math.max(24, fit ?? 56)

  const { values, sites, lines, months } = useMemo(() => {
    const ok = rows.filter((r) => typeof r[field] === 'number' && Number.isFinite(r[field] as number) && r.date)
    const values = ok.map((r) => r[field] as number).sort(d3.ascending)
    const bySite = d3.group(ok, (r) => r.site)
    const sites = [...bySite.keys()]
    const monthly = (rs: FilterRow[]) => d3.rollups(rs, (g) => d3.median(g, (r) => r[field] as number) ?? 0, (r) => r.date.slice(0, 7))
      .map(([m, v]) => ({ t: new Date(`${m}-15`), v })).sort((a, b) => +a.t - +b.t)
    const lines = sites.length <= MAX_LINES
      ? sites.map((s) => ({ name: s, color: bySite.get(s)![0].color, pts: monthly(bySite.get(s)!) }))
      : [{ name: 'All sites', color: 'var(--accent)', pts: monthly(ok) }]
    const months = new Set(ok.map((r) => r.date.slice(0, 7))).size
    return { values, sites, lines, months }
  }, [rows, field])

  const unit = prettyUnit(meta.field_units?.[field])
  const med = d3.quantileSorted(values, 0.5)
  const q1 = d3.quantileSorted(values, 0.25)
  const q3 = d3.quantileSorted(values, 0.75)
  const all = lines.flatMap((l) => l.pts)
  const x = d3.scaleTime().domain(d3.extent(all, (p) => p.t) as [Date, Date]).range([2, Math.max(10, width - 2)])
  const y = d3.scaleLinear().domain([0, d3.max(all, (p) => p.v) ?? 1]).nice().range([sparkH - 2, 2])
  const path = d3.line<{ t: Date; v: number }>().x((p) => x(p.t)).y((p) => y(p.v))

  return (
    <ChartFrame id="stat" title={`${field} — median over the filters in view, with the monthly medians`} exportable={false}>
      <div ref={wrapRef} className="stat-tile">
        {values.length === 0 ? <Empty>No {field} in this subset.</Empty> : <>
          <div className="stat-label" title={field}>{chemText(field)}</div>
          <div className="stat-value">{details ? <button className="stat-detail-value" type="button" aria-label={`Show contributing filters for ${field}`} onClick={() => details({ title: `${field}: headline median`, rows: measuredRows(rows, [field], true), fields: [field] })}><strong>{fmt(med, 3)}</strong></button> : <strong>{fmt(med, 3)}</strong>}{unit && <span className="stat-unit">{unit}</span>}</div>
          <div className="stat-sub">IQR {fmt(q1, 2)}–{fmt(q3, 2)} · n = {values.length}{sites.length > 1 ? ` · ${sites.length} sites` : ''} · {months} months</div>
          {all.length > 1 && <svg width={width} height={sparkH} className="stat-spark" role="img" aria-label={`Monthly medians of ${field}`}>
            {lines.map((l) => <path key={l.name} d={path(l.pts) ?? ''} fill="none" stroke={l.color} strokeWidth={1.5} strokeOpacity={0.9}
              onMouseMove={(e) => tip.show(e, <><strong>{l.name}</strong><br />{l.pts.length} months · latest {fmt(l.pts[l.pts.length - 1]?.v, 3)} {unit}</>)}
              onMouseLeave={tip.hide} />)}
          </svg>}
        </>}
        {tip.node}
      </div>
    </ChartFrame>
  )
}
