import { useMemo, useRef, useState } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Segmented } from '@/components/ChartFrame'
import { Legend } from '@/components/Legend'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { fmt } from '@/lib/stats'
import { INK, FONT } from '@/lib/theme'
import type { FilterRow, MetaFile } from '@/lib/types'

const MODES = ['Treemap', 'Donut'] as const
const STATS = ['Mean', 'Median'] as const

/**
 * Treemap / Donut — mean PM2.5 composition per site.
 *
 * The estate has almost nothing in this category (1 stackplot, 2 pie calls),
 * because matplotlib makes it awkward. It is the natural reading of the
 * ChemSpec columns, so it goes in.
 */
export function CompositionTreemap({ rows, meta }: { rows: FilterRow[]; meta: MetaFile }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
  const [mode, setMode] = useState<(typeof MODES)[number]>('Treemap')
  const [stat, setStat] = useState<(typeof STATS)[number]>('Mean')

  const panelW = Math.max(240, Math.floor((width - 26) / 2))
  const panelH = 260
  const UNITS = 'µg/m³'

  // Species that make up mass, in the order the exporter groups them. PM2.5
  // mass is the denominator, not a slice, so it is excluded from the parts.
  const speciesFields = useMemo(() => {
    const wanted = new Set(['Ions', 'Metals & crustal'])
    const fromGroups = meta.field_groups.filter((g) => wanted.has(g.label)).flatMap((g) => g.fields)
    return ['OC (TOR)', 'EC (TOR)', ...fromGroups].filter((f) => meta.fields.includes(f))
  }, [meta])

  // Built from the live subset so the season/site filters above apply here too.
  const composition = useMemo(() => {
    const sites = meta.sites
      .filter((s) => rows.some((r) => r.site === s.name))
      .map((s) => {
        const sub = rows.filter((r) => r.site === s.name)
        const children = speciesFields
          .map((f) => {
            const vals = sub.map((r) => r[f]).filter((v): v is number => typeof v === 'number' && Number.isFinite(v)).sort(d3.ascending)
            if (!vals.length) return null
            return { name: f, value: d3.mean(vals) ?? 0, median: d3.quantile(vals, 0.5) ?? 0, n: vals.length }
          })
          .filter((c): c is { name: string; value: number; median: number; n: number } => !!c)
        const pm = sub.map((r) => r['PM2.5 mass']).filter((v): v is number => typeof v === 'number' && Number.isFinite(v))
        return { code: s.code, name: s.name, color: s.color, pm25_mean: pm.length ? (d3.mean(pm) ?? null) : null, children }
      })
    return { units: UNITS, sites }
  }, [rows, meta.sites, speciesFields])

  const color = useMemo(() => d3.scaleOrdinal<string>().domain(speciesFields).range(d3.schemeTableau10.concat(d3.schemeSet3 as any)), [speciesFields])
  const val = (c: { value: number; median: number }) => (stat === 'Mean' ? c.value : c.median)

  return (
    <ChartFrame
      id="composition"
      title="Treemap · Donut — what the PM2.5 is made of, per site"
      subtitle="Speciated mass per site from the ChemSpec columns, as a share of the speciated total. PM2.5 mass is the denominator, not a slice; the gap between speciated and measured PM2.5 is unmeasured organic matter and water."
      provenance="1 stackplot + 2 pie calls in the estate · react-graph-gallery.com/treemap · /donut"
      controls={
        <>
          <Segmented label="encoding" value={mode} options={MODES} onChange={setMode} />
          <Segmented label="statistic" value={stat} options={STATS} onChange={setStat} />
        </>
      }
    >
      <div ref={wrapRef} className="chart-wrap" style={{ display: 'flex', flexWrap: 'wrap', gap: 24 }}>
        {composition.sites.length === 0 && <Empty>No sites in this subset.</Empty>}
        {composition.sites.map((site) => {
          const kids = site.children.filter((c) => val(c) > 0)
          const total = d3.sum(kids, val)
          return (
            <div key={site.code} style={{ width: panelW }}>
              <p className="facet-title">
                <span className="swatch" style={{ background: site.color }} />
                {site.name}
                <span style={{ fontWeight: 400, color: 'var(--ink-muted)', fontSize: 11.5 }}>
                  PM2.5 {fmt(site.pm25_mean, 1)} µg/m³ · speciated {fmt(total, 1)}
                </span>
              </p>
              <svg width={panelW} height={panelH} className="animated">
                {mode === 'Treemap'
                  ? (() => {
                      const root = d3
                        .hierarchy<{ name: string; children?: any[]; value?: number }>({ name: site.name, children: kids as any })
                        .sum((d: any) => (d.name === site.name ? 0 : val(d)))
                        .sort((a, b) => (b.value ?? 0) - (a.value ?? 0))
                      d3.treemap<any>().size([panelW - 1, panelH - 1]).paddingInner(2).round(true)(root)
                      return root.leaves().map((leaf: any, i) => {
                        const w = leaf.x1 - leaf.x0
                        const h = leaf.y1 - leaf.y0
                        if (w <= 0 || h <= 0) return null
                        const pct = ((leaf.value ?? 0) / total) * 100
                        return (
                          <g
                            key={i}
                            onMouseEnter={(e) => tip.show(e, [leaf.data.name, `${fmt(leaf.value, 3)} ${UNITS}`, `${pct.toFixed(1)} % of speciated mass`, `n = ${leaf.data.n}`])}
                            onMouseLeave={tip.hide}
                          >
                            <rect x={leaf.x0} y={leaf.y0} width={w} height={h} fill={color(leaf.data.name)} fillOpacity={0.85} stroke="#fff" rx={2} />
                            {w > 54 && h > 20 && (
                              <text x={leaf.x0 + 5} y={leaf.y0 + 14} fontSize={10.5} fill="#fff" fontFamily={FONT.family} pointerEvents="none">
                                {leaf.data.name}
                              </text>
                            )}
                            {w > 54 && h > 33 && (
                              <text x={leaf.x0 + 5} y={leaf.y0 + 27} fontSize={9.5} fill="#fff" fillOpacity={0.85} fontFamily={FONT.mono} pointerEvents="none">
                                {pct.toFixed(1)}%
                              </text>
                            )}
                          </g>
                        )
                      })
                    })()
                  : (() => {
                      const r = Math.min(panelW, panelH) / 2 - 8
                      const arcs = d3.pie<any>().value(val).sort(null)(kids as any)
                      const arc = d3.arc<any>().innerRadius(r * 0.55).outerRadius(r)
                      return (
                        <g transform={`translate(${panelW / 2},${panelH / 2})`}>
                          {arcs.map((a, i) => (
                            <path
                              key={i} d={arc(a) ?? ''} fill={color(a.data.name)} fillOpacity={0.88} stroke="#fff" strokeWidth={1}
                              onMouseEnter={(e) => tip.show(e, [a.data.name, `${fmt(val(a.data), 3)} ${UNITS}`, `${((val(a.data) / total) * 100).toFixed(1)} %`, `n = ${a.data.n}`])}
                              onMouseLeave={tip.hide}
                            />
                          ))}
                          <text textAnchor="middle" dy="0.35em" fontSize={12} fill={INK.muted} fontFamily={FONT.mono}>
                            {fmt(total, 1)}
                          </text>
                        </g>
                      )
                    })()}
              </svg>
            </div>
          )
        })}
        {tip.node}
      </div>
      <Legend items={speciesFields.map((f) => ({ label: f, color: color(f), shape: 'square' as const }))} />
    </ChartFrame>
  )
}
