import { useMemo, useRef } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty } from '@/components/ChartFrame'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { fmt } from '@/lib/stats'
import { FONT, INK, inkOn } from '@/lib/theme'
import { WEATHER_VARS, spearman } from '@/lib/weather'
import type { FilterRow, MetaFile } from '@/lib/types'
import { useColourScheme } from '@/lib/colourSchemes'
import { ColorLegend } from '@/components/ColorLegend'

const MIN_N = 10

/**
 * The comparison across sites: for every site in the pick, Spearman ρ between the chosen
 * measurement and each window-mean weather variable (ERA5), with the site's own weather
 * level beside it. Same question at every site, so the rows read as "where does rain,
 * humidity or a shallow boundary layer go with higher concentrations". Clicking a site narrows
 * the Filters panel's pick to it (the one-site view).
 */
export function SiteWeatherSummary({ rows, meta, field, current, onSite }: {
  rows: FilterRow[]; meta: MetaFile; field: string; current: string | null
  onSite: (code: string) => void
}) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)

  const table = useMemo(() => d3.groups(rows, (r) => r.site).map(([site, rs]) => {
    const code = rs[0]?.code ?? ''
    const withField = rs.filter((r) => typeof r[field] === 'number')
    return {
      site, code, n: withField.length,
      median: d3.median(withField, (r) => r[field] as number) ?? null,
      cells: WEATHER_VARS.map((v) => {
        const ok = withField.filter((r) => typeof r[v.era5] === 'number')
        return { v, n: ok.length, level: d3.mean(ok, (r) => r[v.era5] as number) ?? null,
          rho: ok.length >= MIN_N ? spearman(ok.map((r) => r[v.era5] as number), ok.map((r) => r[field] as number)) : null }
      }),
    }
  }).filter((t) => t.n > 0 && t.cells.some((c) => c.n > 0)).sort((a, b) => a.site.localeCompare(b.site)), [rows, field])

  const colours = useColourScheme('weather-sites', 'diverging')
  const color = d3.scaleQuantize<string>().domain([-1, 1]).range(colours.stops)
  const labelW = 190, numW = 70
  const cell = Math.max(46, Math.min(90, (width - labelW - numW * 2 - 10) / WEATHER_VARS.length))
  const rowH = 26, head = 48
  const w = labelW + numW * 2 + WEATHER_VARS.length * cell, h = head + table.length * rowH + 6

  return (
    <ChartFrame id="weather-sites" title="Sites compared — which weather goes with the measurement where"
      subtitle={`Spearman ρ between ${field} and each weather variable over the filters' windows (ERA5), one row per site in the Filters panel's pick; blank below ${MIN_N} filters. Hover a cell for the site's mean of that variable; click a site to look at it alone.`}
      provenance="the per-site meteorology questions of addis_03_meteorology and meteorology_source_interaction, side by side">
      <div ref={wrapRef} className="chart-wrap" style={{ overflowX: 'auto' }}>
        {table.length === 0 ? <Empty>No site in the pick has both {field} and weather.</Empty> : (
          <svg width={w} height={h} fontFamily={FONT.family}>
            <text x={labelW + 4} y={head - 8} fontSize={10.5} fill={INK.muted}>filters</text>
            <text x={labelW + numW + 4} y={head - 8} fontSize={10.5} fill={INK.muted}>median</text>
            {WEATHER_VARS.map((v, j) => <text key={v.key} x={labelW + numW * 2 + j * cell + cell / 2} y={head - 8} textAnchor="middle" fontSize={10.5} fill={INK.text}>
              {v.label.replace('Daily temperature range', 'Temp. range').replace('Boundary-layer height', 'BLH').replace('Relative humidity', 'RH')}</text>)}
            {table.map((t, i) => {
              const y = head + i * rowH
              const on = t.code === current
              return (
                <g key={t.site} transform={`translate(0,${y})`} style={{ cursor: 'pointer' }} onClick={() => onSite(t.code)}>
                  <rect width={w} height={rowH - 2} fill={on ? '#eef4fb' : 'transparent'} />
                  <circle cx={10} cy={rowH / 2 - 1} r={5} fill={meta.sites.find((s) => s.name === t.site)?.color ?? INK.muted} />
                  <text x={22} y={rowH / 2} dy="0.1em" fontSize={12} fontWeight={on ? 650 : 400} fill={INK.text}>{t.site}</text>
                  <text x={labelW + 4} y={rowH / 2} dy="0.1em" fontSize={11} fontFamily={FONT.mono} fill={INK.muted}>{t.n}</text>
                  <text x={labelW + numW + 4} y={rowH / 2} dy="0.1em" fontSize={11} fontFamily={FONT.mono} fill={INK.muted}>{fmt(t.median, 2)}</text>
                  {t.cells.map((c, j) => {
                    const fill = c.rho === null ? '#f3f4f6' : color(c.rho)
                    return <g key={c.v.key} transform={`translate(${labelW + numW * 2 + j * cell},0)`}
                      onMouseEnter={(e) => tip.show(e, [`${t.site}: ${field} vs ${c.v.label}`, c.rho === null ? `n = ${c.n} (too few)` : `ρ = ${fmt(c.rho, 2)} · n = ${c.n}`,
                        `site mean ${c.v.label.toLowerCase()}: ${fmt(c.level, 1)}`, 'click to look at this site alone'])} onMouseLeave={tip.hide}>
                      <rect x={1} width={cell - 2} height={rowH - 2} fill={fill} />
                      {c.rho !== null && <text x={cell / 2} y={rowH / 2} dy="0.1em" textAnchor="middle" fontSize={10.5} fontFamily={FONT.mono} fill={inkOn(fill)}>{fmt(c.rho, 2)}</text>}
                    </g>
                  })}
                </g>
              )
            })}
          </svg>
        )}
        {tip.node}
      </div>
      <div className="legend">
        <ColorLegend scale={color} scheme={colours.scheme} label="Spearman ρ" width={160} ticks={4} format={(v) => (v > 0 ? `+${v}` : v < 0 ? `−${-v}` : '0')} />
      </div>
    </ChartFrame>
  )
}
