import { useMemo, useRef, useState } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Segmented } from '@/components/ChartFrame'
import { Legend, useLegend } from '@/components/Legend'
import { XAxis, YAxis } from '@/components/Axes'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { fmt } from '@/lib/stats'
import { INK, MARGIN, FONT, withUnit } from '@/lib/theme'
import type { FilterRow, MetaFile } from '@/lib/types'

const STATS = ['Median', 'Mean'] as const

/**
 * Grouped barplot with IQR whiskers — 70 bar figures in the estate.
 * Bars carry the median rather than the mean: these distributions are
 * right-skewed and a handful of high-loading filters drag the mean somewhere
 * no sample actually sits.
 */
export function Barplot({ rows, meta, field }: { rows: FilterRow[]; meta: MetaFile; field: string }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
  const height = 430

  const [stat, setStat] = useState<(typeof STATS)[number]>('Median')
  const seasons = meta.seasons.map((s) => s.name)
  const lg = useLegend()

  const data = useMemo(() => {
    const out: { site: string; season: string; value: number; q1: number; q3: number; n: number; color: string }[] = []
    for (const s of meta.sites) {
      for (const season of seasons) {
        const vals = rows
          .filter((r) => r.site === s.name && r.season === season)
          .map((r) => r[field])
          .filter((v): v is number => typeof v === 'number' && Number.isFinite(v))
          .sort(d3.ascending)
        if (vals.length < 3) continue
        out.push({
          site: s.name,
          season,
          value: stat === 'Mean' ? (d3.mean(vals) ?? 0) : (d3.quantile(vals, 0.5) ?? 0),
          q1: d3.quantile(vals, 0.25) ?? 0,
          q3: d3.quantile(vals, 0.75) ?? 0,
          n: vals.length,
          color: meta.seasons.find((x) => x.name === season)?.color ?? s.color,
        })
      }
    }
    return out
  }, [rows, field, stat, meta, seasons])

  const sites = [...new Set(data.map((d) => d.site))]
  const shown = data.filter((d) => lg.show(d.season))
  const innerW = Math.max(240, width - MARGIN.left - MARGIN.right)
  const innerH = height - MARGIN.top - MARGIN.bottom

  const x0 = d3.scaleBand<string>().domain(sites).range([0, innerW]).paddingInner(0.24)
  // each site's group holds only the seasons it has (under per-site calendars they differ by site)
  const x1For = (site: string) =>
    d3.scaleBand<string>().domain(seasons.filter((se) => lg.show(se) && shown.some((d) => d.site === site && d.season === se))).range([0, x0.bandwidth()]).padding(0.1)
  const x1BySite = new Map(sites.map((site) => [site, x1For(site)]))
  const yLo = Math.min(0, d3.min(shown, (d) => Math.min(d.value, d.q1)) ?? 0)
  const y = d3.scaleLinear().domain([yLo, (d3.max(shown, (d) => Math.max(d.value, d.q3)) ?? 1) * 1.08]).range([innerH, 0]).nice()

  return (
    <ChartFrame
      id="barplot"
      title="Barplot — site by season, with the spread shown"
      subtitle="Median per site per selected season (each site on its own calendar unless shared Ethiopian bins are chosen), whiskered to the interquartile range. A bar alone would imply a precision these n≈10–90 groups don't have, so the IQR travels with it."
      provenance="stands in for 70 bar/barh figures · comparisons.summary_bars · react-graph-gallery.com/barplot"
      controls={
        <>
          <Segmented label="statistic" value={stat} options={STATS} onChange={setStat} />
        </>
      }
    >
      <div ref={wrapRef} className="chart-wrap">
        {data.length === 0 ? (
          <Empty>No site-season group has 3 or more filters with {field}.</Empty>
        ) : (
          <svg width={width} height={height} className="animated">
            <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
              <YAxis scale={y} x={0} label={withUnit(field, meta.field_units)} gridWidth={innerW} />
              <XAxis scale={x0} y={innerH} />
              {shown.map((d, i) => {
                const x1 = x1BySite.get(d.site)!
                const bx = (x0(d.site) ?? 0) + (x1(d.season) ?? 0)
                const bw = x1.bandwidth()
                return (
                  <g
                    key={i}
                    opacity={lg.dim(d.season)}
                    onMouseEnter={(e) => tip.show(e, [`${d.site} · ${d.season}`, `${stat.toLowerCase()} = ${fmt(d.value)}`, `IQR ${fmt(d.q1)} – ${fmt(d.q3)}`, `n = ${d.n}`])}
                    onMouseLeave={tip.hide}
                  >
                    <rect x={bx} y={Math.min(y(d.value), y(0))} width={bw} height={Math.abs(y(d.value) - y(0))} fill={d.color} fillOpacity={0.78} rx={2} />
                    <line x1={bx + bw / 2} x2={bx + bw / 2} y1={y(d.q1)} y2={y(d.q3)} stroke={INK.text} strokeWidth={1.3} />
                    <line x1={bx + bw / 2 - 4} x2={bx + bw / 2 + 4} y1={y(d.q3)} y2={y(d.q3)} stroke={INK.text} strokeWidth={1.3} />
                    <line x1={bx + bw / 2 - 4} x2={bx + bw / 2 + 4} y1={y(d.q1)} y2={y(d.q1)} stroke={INK.text} strokeWidth={1.3} />
                    {bw > 22 && (
                      <text x={bx + bw / 2} y={innerH + 32} textAnchor="middle" fontSize={9} fill={INK.muted} fontFamily={FONT.mono}>
                        {d.n}
                      </text>
                    )}
                  </g>
                )
              })}
            </g>
          </svg>
        )}
        <Legend items={meta.seasons.map((s) => ({ label: s.name, color: s.color, shape: 'square' as const }))} {...lg.props} note="small numbers under the site names are n per bar" />
        {tip.node}
      </div>
    </ChartFrame>
  )
}
