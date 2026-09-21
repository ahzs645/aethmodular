import { useMemo, useRef, useState } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Select } from '@/components/ChartFrame'
import { Legend } from '@/components/Legend'
import { XAxis } from '@/components/Axes'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { byConstruction } from '@/lib/derived'
import { regression, fmt } from '@/lib/stats'
import { INK, MARGIN, FONT } from '@/lib/theme'
import type { FilterRow, MetaFile } from '@/lib/types'

const POS = INK.positive
const NEG = INK.negative

/**
 * Lollipop — what actually predicts a chosen measurement, ranked.
 *
 * AGENTS.md: "Prefer R² over raw r when ranking or describing relationship
 * strength." So the bar length is R² and the sign of r is carried by colour
 * plus the tooltip, never by the length.
 *
 * This is the chart the estate is missing. 116 notebooks draw crossplots one
 * pair at a time; nothing ranks all the pairs at once. It keeps a per-site
 * select on purpose: pooling sites manufactures correlation out of the site
 * offsets alone (Simpson's), so "one site" is the honest default.
 */
export function CorrelationRanking({ rows, meta, target }: { rows: FilterRow[]; meta: MetaFile; target: string }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)

  const [siteFilter, setSiteFilter] = useState('Addis Ababa')
  const [minN, setMinN] = useState('30')

  const siteOptions = useMemo(
    () => ['All sites (pooled)', ...meta.sites.filter((s) => rows.some((r) => r.site === s.name)).map((s) => s.name)],
    [meta.sites, rows]
  )
  const site = siteOptions.includes(siteFilter) ? siteFilter : siteOptions[1] ?? siteOptions[0]

  const ranked = useMemo(() => {
    const src = site.startsWith('All') ? rows : rows.filter((r) => r.site === site)
    const out: { field: string; r2: number; r: number; n: number; slope: number; trivial: boolean }[] = []
    for (const f of meta.fields) {
      if (f === target) continue
      const xs: number[] = []
      const ys: number[] = []
      for (const r of src) {
        const a = r[f]
        const b = r[target]
        if (typeof a === 'number' && typeof b === 'number' && Number.isFinite(a) && Number.isFinite(b)) {
          xs.push(a)
          ys.push(b)
        }
      }
      if (xs.length < Number(minN)) continue
      const s = regression(xs, ys)
      if (s) out.push({ field: f, r2: s.r2, r: s.r, n: s.n, slope: s.slope, trivial: byConstruction(f, target) })
    }
    return out.sort((a, b) => b.r2 - a.r2)
  }, [rows, target, site, minN, meta.fields])

  const rowH = 22
  // right gutter holds the "0.939 · n=190" annotation, which used to run
  // off the canvas for the top-ranked rows
  const LABEL_GUTTER = 96
  const innerW = Math.max(160, width - MARGIN.left - 60 - MARGIN.right - LABEL_GUTTER)
  const innerH = ranked.length * rowH
  const height = innerH + MARGIN.top + MARGIN.bottom
  const x = d3.scaleLinear().domain([0, 1]).range([0, innerW])

  return (
    <ChartFrame
      id="ranking"
      title={`Lollipop — everything that predicts ${target}, ranked`}
      subtitle="R² of every other measured quantity against the measurement chosen in the bar, at one site, ranked. Length is R² (never signed); colour carries the sign of r. This is the view that tells you where to point a crossplot, instead of drawing 281 of them one pair at a time."
      provenance="stands in for 70 barplot + 11 lollipop figures · react-graph-gallery.com/lollipop"
      controls={
        <>
          <Select label="site" value={site} options={siteOptions} onChange={setSiteFilter} title="Pooled sites can rank a species highly on site offsets alone" />
          <Select label="min n" value={minN} options={['10', '30', '50', '100']} onChange={setMinN} />
          <span className="control">{ranked.length} pairs</span>
        </>
      }
    >
      <div ref={wrapRef} className="chart-wrap">
        {ranked.length === 0 ? (
          <Empty>No pair reaches n ≥ {minN} at {site}. Lower the minimum or widen the subset.</Empty>
        ) : (
          <svg width={width} height={height} className="animated">
            <g transform={`translate(${MARGIN.left + 60},${MARGIN.top})`}>
              {x.ticks(6).map((t) => (
                <line key={t} x1={x(t)} x2={x(t)} y1={0} y2={innerH} stroke={INK.grid} />
              ))}
              {ranked.map((d, i) => {
                const cy = i * rowH + rowH / 2
                const col = d.trivial ? INK.identity : d.r >= 0 ? POS : NEG
                return (
                  <g
                    key={d.field}
                    onMouseEnter={(e) =>
                      tip.show(e, [
                        d.field,
                        `R² = ${fmt(d.r2, 4)}`,
                        `r = ${fmt(d.r, 3)}`,
                        `OLS slope = ${fmt(d.slope, 4)}`,
                        `n = ${d.n}`,
                        ...(d.trivial ? ['derived from the target by arithmetic — not a finding'] : []),
                      ])
                    }
                    onMouseLeave={tip.hide}
                  >
                    <line x1={0} x2={x(d.r2)} y1={cy} y2={cy} stroke={col} strokeWidth={2} strokeOpacity={0.55} strokeDasharray={d.trivial ? '3 3' : undefined} />
                    <circle cx={x(d.r2)} cy={cy} r={5} fill={d.trivial ? '#fff' : col} stroke={col} strokeWidth={d.trivial ? 1.5 : 0} />
                    <text x={-10} y={cy} dy="0.32em" textAnchor="end" fontSize={11.5} fill={d.trivial ? INK.muted : INK.text} fontFamily={FONT.family}>
                      {d.field}
                    </text>
                    <text x={x(d.r2) + 10} y={cy} dy="0.32em" fontSize={10.5} fill={INK.muted} fontFamily={FONT.mono}>
                      {d.r2.toFixed(3)} · n={d.n}
                    </text>
                  </g>
                )
              })}
              <XAxis scale={x} y={innerH} label={`R² against ${target}`} />
            </g>
          </svg>
        )}
        <Legend
          items={[
            { label: 'positive r', color: POS },
            { label: 'negative r', color: NEG },
            { label: 'by construction (unit conversion of the target)', color: INK.identity },
          ]}
        />
        {tip.node}
      </div>
    </ChartFrame>
  )
}
