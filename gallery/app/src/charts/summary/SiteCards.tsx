import { useMemo } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty } from '@/components/ChartFrame'
import { prettyUnit } from '@/components/FieldSelect'
import { useChartOptions } from '@/lib/chartOptions'
import { useFitHeight } from '@/lib/chartFit'
import { fmt } from '@/lib/format'
import type { FilterRow, MetaFile } from '@/lib/types'

const SORTS = ['Name', 'First field ↓', 'Filters ↓'] as const

interface Card {
  site: string
  code: string
  color: string
  n: number
  from: string
  to: string
  stats: { field: string; median: number | null; n: number; spark: { t: Date; v: number }[] }[]
}

/**
 * One small card per site, the same few numbers on each, so 4 or 38 sites read the same
 * way (the OpenAQ / PurpleAir location card): each field's median over the filters in view,
 * a bar for where it sits among the sites shown, and a sparkline of its monthly medians.
 * A click opens the site on its own (the Dashboard's One site scope).
 */
export function SiteCards({ rows, meta, fields, onOpenSite }: { rows: FilterRow[]; meta: MetaFile; fields: string[]; onOpenSite?: (code: string) => void }) {
  const fit = useFitHeight()
  const o = useChartOptions('sitecards', [
    { key: 'sort', label: 'Sort', choices: [...SORTS], default: 'First field ↓', persist: true },
  ])
  const cards = useMemo<Card[]>(() => {
    const bySite = d3.group(rows, (r) => r.site)
    return [...bySite].map(([site, rs]) => ({
      site, code: rs[0].code, color: rs[0].color, n: rs.length,
      from: d3.min(rs, (r) => r.date) ?? '', to: d3.max(rs, (r) => r.date) ?? '',
      stats: fields.map((f) => {
        const ok = rs.filter((r) => typeof r[f] === 'number' && Number.isFinite(r[f] as number))
        const spark = d3.rollups(ok, (g) => d3.median(g, (r) => r[f] as number) ?? 0, (r) => r.date.slice(0, 7))
          .map(([m, v]) => ({ t: new Date(`${m}-15`), v })).sort((a, b) => +a.t - +b.t)
        return { field: f, median: d3.median(ok, (r) => r[f] as number) ?? null, n: ok.length, spark }
      }),
    }))
  }, [rows, fields])
  const sort = o.values.sort as (typeof SORTS)[number]
  const sorted = [...cards].sort(sort === 'Name' ? (a, b) => a.site.localeCompare(b.site)
    : sort === 'Filters ↓' ? (a, b) => b.n - a.n
      : (a, b) => (b.stats[0]?.median ?? -Infinity) - (a.stats[0]?.median ?? -Infinity))
  // each field's largest site median, for the bars
  const maxOf = fields.map((_, i) => d3.max(cards, (c) => c.stats[i]?.median ?? 0) ?? 1)

  return (
    <ChartFrame id="sitecards" title="Site cards — the same numbers for every site, sorted" exportable={false} controls={o.controls()}>
      <div className="site-cards" style={fit === null ? undefined : { maxHeight: Math.max(120, fit), overflowY: 'auto' }}>
        {sorted.length === 0 && <Empty>No sites in this subset.</Empty>}
        {sorted.map((c) => (
          <button key={c.site} type="button" className="site-card" style={{ ['--site' as string]: c.color }}
            onClick={() => onOpenSite?.(c.code)} title={onOpenSite ? `Open ${c.site} on its own` : c.site}>
            <span className="site-card-head"><span className="swatch" style={{ background: c.color }} /><strong>{c.site}</strong>
              <span className="site-card-n">{c.n}</span></span>
            <span className="site-card-dates">{c.from.slice(0, 7)} → {c.to.slice(0, 7)}</span>
            {c.stats.map((s, i) => (
              <span key={s.field} className="site-card-row">
                <span className="site-card-field" title={s.field}>{s.field}</span>
                <span className="site-card-value">{s.median === null ? '—' : fmt(s.median, 3)}<small> {prettyUnit(meta.field_units?.[s.field])}</small></span>
                <span className="site-card-bar"><span style={{ width: `${s.median === null || !maxOf[i] ? 0 : Math.max(2, (100 * Math.max(0, s.median)) / maxOf[i])}%` }} /></span>
                <Spark pts={s.spark} color={c.color} />
              </span>
            ))}
          </button>
        ))}
      </div>
    </ChartFrame>
  )
}

function Spark({ pts, color }: { pts: { t: Date; v: number }[]; color: string }) {
  if (pts.length < 2) return <svg className="site-card-spark" />
  const w = 64, h = 16
  const x = d3.scaleTime().domain(d3.extent(pts, (p) => p.t) as [Date, Date]).range([1, w - 1])
  const y = d3.scaleLinear().domain(d3.extent(pts, (p) => p.v) as [number, number]).range([h - 1, 1])
  return <svg className="site-card-spark" width={w} height={h} aria-hidden>
    <path d={d3.line<{ t: Date; v: number }>().x((p) => x(p.t)).y((p) => y(p.v))(pts) ?? ''} fill="none" stroke={color} strokeWidth={1.3} />
  </svg>
}
