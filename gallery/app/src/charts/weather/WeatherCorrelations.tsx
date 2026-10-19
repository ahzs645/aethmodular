import { useMemo, useRef, useState } from 'react'
import { ExpandIcon } from '@/components/Icons'
import { SpectrumModal } from '@/components/FilterSpectrumPlot'
import * as d3 from 'd3'
import { ChartFrame, Empty, Select } from '@/components/ChartFrame'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { fmt } from '@/lib/stats'
import { chemText } from '@/lib/format'
import { FONT, INK, inkOn } from '@/lib/theme'
import { WEATHER_VARS, spearman, weatherField, type WeatherSource } from '@/lib/weather'
import type { FilterRow, MetaFile } from '@/lib/types'
import { useColourScheme } from '@/lib/colourSchemes'
import { ColorLegend } from '@/components/ColorLegend'
import { useFitHeight } from '@/lib/chartFit'
import { useTileSetting } from '@/lib/url'

/** species worth a column when they are measured; the rest of the field list would only add empty columns */
const PREFERRED = ['PM2.5 mass', 'HIPS BC', 'HIPS Fabs', 'EC (FTIR)', 'OC (FTIR)', 'OM (FTIR)', 'Sulfate', 'Nitrate', 'Ammonium', 'Potassium', 'Sodium', 'Calcium', 'Iron', 'Silicon', 'Aluminum', 'Zinc', 'Lead']
const MIN_N = 10

/**
 * Weather × chemistry correlation matrices at one site: Spearman ρ between each window-mean
 * weather variable (rows) and each measured species (columns), for every filter and then
 * season by season, because a correlation pooled across seasons is mostly the seasonal
 * cycle (meteorology_revisions R5, which drew three of these for the PMF sources).
 */
export function WeatherCorrelations({ rows, meta, source, site }: { rows: FilterRow[]; meta: MetaFile; source: WeatherSource; site: { name: string; code: string } | null }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
  const fit = useFitHeight()
  // a fitted tile shows one matrix, sized to it (its ⋯ menu picks which); the page shows every season,
  // and a click opens one larger
  const [pick, setPick] = useTileSetting<string>('season', 'All filters')
  const [open, setOpen] = useState<string | null>(null)

  const at = useMemo(() => rows.filter((r) => r.site === site?.name), [rows, site])
  const vars: { label: string; field: string }[] = WEATHER_VARS.flatMap((v) => {
    const field = weatherField(v.key, source)
    return field && at.some((r) => typeof r[field] === 'number') ? [{ label: v.label as string, field }] : []
  })
  const species = PREFERRED.filter((f) => at.filter((r) => typeof r[f] === 'number').length >= MIN_N)
  const facets = useMemo(() => {
    const seasons = d3.groups(at, (r) => r.season).sort((a, b) => meta.seasons.findIndex((s) => s.name === a[0]) - meta.seasons.findIndex((s) => s.name === b[0]))
    return [['All filters', at] as const, ...seasons].map(([name, rs]) => ({
      name: String(name).replace(`${site?.name} · `, ''), color: meta.seasons.find((s) => s.name === name)?.color ?? INK.text,
      cells: vars.map((v) => species.map((f) => {
        const ok = rs.filter((r) => typeof r[v.field] === 'number' && typeof r[f] === 'number')
        return { v, f, n: ok.length, rho: ok.length >= MIN_N ? spearman(ok.map((r) => r[v.field] as number), ok.map((r) => r[f] as number)) : null }
      })),
      n: rs.length,
    }))
  }, [at, vars.map((v) => v.field).join(), species.join(), meta.seasons, site]) // eslint-disable-line react-hooks/exhaustive-deps

  const colours = useColourScheme('weather-correlations', 'diverging')
  const color = d3.scaleQuantize<string>().domain([-1, 1]).range(colours.stops)
  const EVERY = 'Every season'
  const one = fit !== null && pick !== EVERY
  const chosen = facets.find((f) => f.name === pick) ?? facets[0]
  const shownFacets = one && chosen ? [chosen] : facets
  const labelW = one ? 120 : 150
  const cell = one
    ? Math.max(12, Math.min(48, (width - labelW - 14) / Math.max(1, species.length), (fit! - 80) / Math.max(1, vars.length)))
    : Math.max(22, Math.min(34, (Math.min(width, 900) - labelW - 20) / Math.max(1, species.length)))
  const facetPicker = fit !== null && facets.length > 1
    ? <Select label="Show" value={one && chosen ? chosen.name : EVERY} options={[EVERY, ...facets.map((f) => f.name)]} onChange={setPick} />
    : null

  return (
    <ChartFrame id="weather-correlations" controls={facetPicker} title={`${site ? `${site.name} · ` : ''}weather × chemistry correlations, by season`}
      subtitle={`Spearman ρ between each weather variable over the filter's window and each species, for every filter at the site and then within each season (blank: fewer than ${MIN_N} filters). Pooled correlations mostly follow the seasonal cycle; the per-season ones are closer to "does this weather change the day's air".`}
      provenance="meteorology_revisions R5 (seasonal correlation matrices) · meteorology_source_interaction §3 (temperature range as an inversion proxy)">
      <div ref={wrapRef} className="chart-wrap" style={{ display: 'flex', flexWrap: 'wrap', gap: 18, justifyContent: 'center' }}>
        {!site || species.length === 0 || vars.length === 0 ? <Empty>No site with weather and at least {MIN_N} measured filters in this subset.</Empty> : shownFacets.map((fc) => matrix(fc, cell, labelW, fit === null ? () => setOpen(fc.name) : undefined))}
        {tip.node}
      </div>
      {open && facets.find((f) => f.name === open) && <SpectrumModal id={`${site?.name ?? ''} · ${open}`} title="Weather × chemistry correlations" onClose={() => setOpen(null)}>
        <div style={{ display: 'flex', justifyContent: 'center', overflowX: 'auto' }}>
          {matrix(facets.find((f) => f.name === open)!, Math.max(22, Math.min(46, (window.innerWidth - 300) / Math.max(1, species.length), (window.innerHeight - 320) / Math.max(1, vars.length))), 170)}
        </div>
      </SpectrumModal>}
      <div className="legend">
        <ColorLegend scale={color} scheme={colours.scheme} label="Spearman ρ" width={160} ticks={4} format={(v) => (v > 0 ? `+${v}` : v < 0 ? `−${-v}` : '0')} />
        <span className="legend-item">values shown where |ρ| ≥ 0.3{fit === null ? ' · ⤢ opens a matrix larger' : ''}</span>
      </div>
    </ChartFrame>
  )

  function matrix(fc: (typeof facets)[number], cell: number, labelW: number, onOpen?: () => void) {
    const fw = labelW + species.length * cell + 10, fh = 70 + vars.length * cell + 10
    return (
          <div key={fc.name} className="expandable-panel">
          {onOpen && <button type="button" className="btn quiet panel-expand" style={{ right: 10 }} onClick={onOpen} aria-label={`Open ${fc.name} larger`} title="Open larger"><ExpandIcon /></button>}
          <svg width={fw} height={fh} fontFamily={FONT.family}>
            <text x={labelW} y={14} fontSize={12} fontWeight={600} fill={fc.color}>{fc.name} <tspan fill={INK.muted} fontWeight={400} fontSize={10.5}>n={fc.n}</tspan></text>
            {species.map((f, j) => <text key={f} transform={`translate(${labelW + j * cell + cell / 2},${66}) rotate(-50)`} fontSize={10} fill={INK.muted}>{chemText(f)}</text>)}
            {vars.map((v, i) => <text key={v.field} x={labelW - 6} y={70 + i * cell + cell / 2} dy="0.35em" textAnchor="end" fontSize={10.5} fill={INK.text}>{v.label}</text>)}
            {fc.cells.flatMap((row, i) => row.map((c, j) => {
              const fill = c.rho === null ? '#f3f4f6' : color(c.rho)
              return <g key={`${i}-${j}`} transform={`translate(${labelW + j * cell},${70 + i * cell})`}
                onMouseEnter={(e) => tip.show(e, [`${fc.name}: ${c.v.label} vs ${c.f}`, c.rho === null ? `n = ${c.n} (too few)` : `ρ = ${fmt(c.rho, 2)} · n = ${c.n}`])} onMouseLeave={tip.hide}>
                <rect width={cell - 1} height={cell - 1} fill={fill} />
                {c.rho !== null && Math.abs(c.rho) >= 0.3 && cell >= 18 && <text x={cell / 2} y={cell / 2} dy="0.35em" textAnchor="middle" fontSize={9} fill={inkOn(fill)} fontFamily={FONT.mono}>{fmt(c.rho, 1)}</text>}
              </g>
            }))}
          </svg>
          </div>
    )
  }
}
