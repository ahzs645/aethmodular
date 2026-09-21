import { useMemo, useRef, useState } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Toggle } from '@/components/ChartFrame'
import { Legend, toggleIn } from '@/components/Legend'
import { XAxis, YAxis } from '@/components/Axes'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { fmt } from '@/lib/stats'
import { INK, MARGIN, withUnit } from '@/lib/theme'
import type { FilterRow, MetaFile } from '@/lib/types'

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

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
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
  const height = 420

  const [hidden, setHidden] = useState<Set<string>>(new Set())
  const [bands, setBands] = useState(true)

  const allSeries = useMemo(
    () =>
      meta.sites
        .map((s) => {
          const points = d3.range(1, 13).map((m) => {
            const vals = rows
              .filter((r) => r.site === s.name && r.month === m)
              .map((r) => r[field])
              .filter((v): v is number => typeof v === 'number' && Number.isFinite(v))
              .sort(d3.ascending)
            return {
              month: m,
              n: vals.length,
              median: vals.length ? (d3.quantile(vals, 0.5) ?? null) : null,
              q1: vals.length >= 4 ? (d3.quantile(vals, 0.25) ?? null) : null,
              q3: vals.length >= 4 ? (d3.quantile(vals, 0.75) ?? null) : null,
            }
          })
          return { name: s.name, color: s.color, points }
        })
        .filter((s) => s.points.some((p) => p.median !== null)),
    [rows, field, meta.sites]
  )
  const series = allSeries.filter((s) => !hidden.has(s.name))

  const innerW = Math.max(240, width - MARGIN.left - MARGIN.right)
  const innerH = height - MARGIN.top - MARGIN.bottom

  const x = d3.scalePoint<number>().domain(d3.range(1, 13)).range([0, innerW])
  const allHi = series.flatMap((s) => s.points.map((p) => (bands ? p.q3 : null) ?? p.median ?? 0))
  const allLo = series.flatMap((s) => s.points.map((p) => (bands ? p.q1 : null) ?? p.median ?? 0))
  const y = d3.scaleLinear().domain([Math.min(0, d3.min(allLo) ?? 0), (d3.max(allHi) ?? 1) * 1.08]).range([innerH, 0]).nice()

  const seasonOf = (m: number) => meta.seasons.find((s) => s.months.includes(m))

  return (
    <ChartFrame
      id="monthly"
      title="Area chart — monthly climatology with an IQR band"
      subtitle="Median by calendar month with the interquartile range shaded. February is flagged: the two published Ethiopian calendars assign it to different seasons and it is a high-BC month, so seasonal means are not interchangeable between conventions."
      provenance="stands in for 59 fill_between+plot figures · react-graph-gallery.com/area-plot"
      controls={<Toggle label="IQR bands" checked={bands} onChange={setBands} />}
    >
      <div ref={wrapRef} className="chart-wrap">
        {series.length === 0 ? (
          <Empty>No site has {field} in this subset.</Empty>
        ) : (
          <svg width={width} height={height} className="animated">
            <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
              {d3.range(1, 13).map((m) => {
                const s = seasonOf(m)
                if (!s) return null
                const step = innerW / 11
                const x0 = Math.max(0, (x(m) ?? 0) - step / 2)
                const x1 = Math.min(innerW, (x(m) ?? 0) + step / 2)
                return <rect key={m} x={x0} y={0} width={Math.max(0, x1 - x0)} height={innerH} fill={s.color} fillOpacity={0.08} />
              })}
              <YAxis scale={y} x={0} label={withUnit(field, meta.field_units)} gridWidth={innerW} />
              <XAxis scale={x as any} y={innerH} format={(m: number) => MONTHS[m - 1]} tickCount={12} />

              {series.map((s) => {
                const withBand = s.points.filter((p) => p.q1 !== null && p.q3 !== null)
                const area = d3.area<(typeof withBand)[number]>().x((p) => x(p.month) ?? 0).y0((p) => y(p.q1!)).y1((p) => y(p.q3!)).curve(d3.curveMonotoneX)
                const withMedian = s.points.filter((p) => p.median !== null)
                const line = d3.line<(typeof withMedian)[number]>().x((p) => x(p.month) ?? 0).y((p) => y(p.median!)).curve(d3.curveMonotoneX)
                return (
                  <g key={s.name}>
                    {bands && <path d={area(withBand) ?? ''} fill={s.color} fillOpacity={0.18} pointerEvents="none" />}
                    <path d={line(withMedian) ?? ''} fill="none" stroke={s.color} strokeWidth={2.4} pointerEvents="none" />
                    {withMedian.map((p) => (
                      <circle
                        key={p.month}
                        cx={x(p.month) ?? 0} cy={y(p.median!)} r={p.month === 2 ? 5.5 : 3.6}
                        fill={p.month === 2 ? '#fff' : s.color}
                        stroke={p.month === 2 ? INK.text : s.color} strokeWidth={p.month === 2 ? 2 : 1}
                        onMouseEnter={(e) =>
                          tip.show(e, [
                            `${s.name} · ${MONTHS[p.month - 1]}`,
                            `median = ${fmt(p.median)}`,
                            p.q1 !== null ? `IQR ${fmt(p.q1)} – ${fmt(p.q3)}` : 'IQR n/a (n<4)',
                            `n = ${p.n}`,
                            p.month === 2 ? `February: ${seasonOf(2)?.name} under ${meta.season_convention}` : `season: ${seasonOf(p.month)?.name ?? '—'}`,
                          ])
                        }
                        onMouseLeave={tip.hide}
                      />
                    ))}
                  </g>
                )
              })}
            </g>
          </svg>
        )}
        <Legend
          items={[
            ...allSeries.map((s) => ({ label: s.name, color: s.color, shape: 'line' as const })),
            ...meta.seasons.map((s) => ({ label: s.name, color: s.color, shape: 'band' as const })),
          ]}
          hidden={hidden}
          onToggle={(l) => { if (allSeries.some((s) => s.name === l)) setHidden((h) => toggleIn(h, l)) }}
          note="hollow marker = February (convention-dependent)"
        />
        {tip.node}
      </div>
    </ChartFrame>
  )
}
