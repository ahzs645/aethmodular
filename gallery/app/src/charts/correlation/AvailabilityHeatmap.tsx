import { useMemo, useRef, useState } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Segmented } from '@/components/ChartFrame'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { INK, FONT } from '@/lib/theme'
import type { FilterRow, MetaFile } from '@/lib/types'
import { useColourScheme } from '@/lib/colourSchemes'
import { fidsProps, useFilterEdits } from '@/lib/filterEdits'

/**
 * Heatmap — sample coverage by site and month. The matplotlib equivalents are
 * the data-completeness / availability strip charts
 * (notebooks/analysis/data_availability/, timeseries.data_completeness).
 *
 * "both" is the one that matters for the crossplot above: a filter only
 * enters the regression if it carries x *and* y, and the two instruments'
 * coverage windows do not line up.
 */
export function AvailabilityHeatmap({ rows, meta, xField, yField }: { rows: FilterRow[]; meta: MetaFile; xField: string; yField: string }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
  const [which, setWhich] = useState<'x' | 'y' | 'both'>('both')
  const colours = useColourScheme('coverage', 'sequential')
  const edits = useFilterEdits()

  const has = (r: FilterRow, f: string) => typeof r[f] === 'number' && Number.isFinite(r[f] as number)
  const label = which === 'x' ? xField : which === 'y' ? yField : `both ${xField} and ${yField}`

  const { cells, periods, sites, maxCount } = useMemo(() => {
    const sites = meta.sites.map((s) => s.name)
    const byKey = new Map<string, number>()
    const idsOf = new Map<string, string[]>()
    const periodSet = new Set<string>()
    for (const r of rows) {
      const ok = which === 'x' ? has(r, xField) : which === 'y' ? has(r, yField) : has(r, xField) && has(r, yField)
      if (!ok) continue
      const period = r.date.slice(0, 7) // YYYY-MM
      periodSet.add(period)
      const k = `${r.site}|${period}`
      byKey.set(k, (byKey.get(k) ?? 0) + 1)
      idsOf.set(k, [...(idsOf.get(k) ?? []), r.id])
    }
    const periods = [...periodSet].sort()
    const cells = [...byKey.entries()].map(([k, count]) => {
      const [site, period] = k.split('|')
      return { site, period, count, ids: idsOf.get(k) ?? [] }
    })
    return { cells, periods, sites, maxCount: d3.max(cells, (c) => c.count) ?? 1 }
  }, [rows, which, xField, yField, meta.sites])

  const margin = { top: 14, right: 20, bottom: 56, left: 110 }
  const innerW = Math.max(320, width - margin.left - margin.right)
  const rowH = 34
  const innerH = sites.length * rowH
  const cellW = innerW / Math.max(1, periods.length)
  const color = d3.scaleQuantize<string>().domain([0, maxCount]).range(colours.stops)

  return (
    <ChartFrame
      id="coverage"
      title="Heatmap — sample coverage by site and month"
      subtitle="How many filters carry the measurement per site per month. Gaps here are the reason several notebooks exist at all: Beijing reaches back to 2013, Delhi only starts mid-2022. 'Both' shows what the crossplot can actually use."
      provenance="stands in for the data-availability strip charts · react-graph-gallery.com/heatmap"
      controls={<>
        <Segmented
          label="Count filters with"
          value={which}
          options={['x', 'y', 'both'] as const}
          optionLabel={(o) => (o === 'both' ? 'Both' : `${o} field`)}
          onChange={setWhich}
          title={`x = ${xField} · y = ${yField}`}
        />
        {colours.picker}
      </>}
    >
      <div ref={wrapRef} className="chart-wrap">
        <svg width={width} height={innerH + margin.top + margin.bottom}>
          <g transform={`translate(${margin.left},${margin.top})`}>
            {cells.map((c, i) => {
              const yi = sites.indexOf(c.site)
              const xi = periods.indexOf(c.period)
              if (yi < 0 || xi < 0) return null
              return (
                <rect
                  key={i}
                  x={xi * cellW} y={yi * rowH}
                  width={Math.max(1, cellW - 0.6)} height={rowH - 4}
                  fill={color(c.count)} rx={1.5}
                  // a cell is a site's month of filters: click lists them in the side panel, right-click excludes or keeps them
                  {...fidsProps(c.ids, `${c.site} · ${c.period}`)} style={{ cursor: 'pointer' }}
                  onClick={() => { tip.hide(); edits.openGroup({ title: `${c.site} · ${c.period}`, ids: c.ids, fields: [xField, yField] }) }}
                  onMouseEnter={(e) => tip.show(e, [c.site, c.period, `${c.count} filters with ${label}`, 'click to list them · right-click to exclude or keep'])}
                  onMouseLeave={tip.hide}
                />
              )
            })}
            {sites.map((s, i) => (
              <text key={s} x={-10} y={i * rowH + (rowH - 4) / 2} dy="0.32em" textAnchor="end" fontSize={12} fill={INK.text} fontFamily={FONT.family}>
                {s}
              </text>
            ))}
            {periods.map((p, i) =>
              p.endsWith('-01') ? (
                <text key={p} transform={`translate(${i * cellW + cellW / 2},${innerH + 6}) rotate(-58)`} fontSize={10} fill={INK.muted} textAnchor="end" fontFamily={FONT.family}>
                  {p.slice(0, 4)}
                </text>
              ) : null
            )}
          </g>
        </svg>
        <div className="legend">
          <span>0 filters</span>
          <span style={{ display: 'inline-block', width: 160, height: 10, borderRadius: 2, background: `linear-gradient(to right, ${colours.stops.join(',')})` }} />
          <span>{maxCount}</span>
          <span className="legend-note">counting {label}</span>
        </div>
        {tip.node}
      </div>
    </ChartFrame>
  )
}
