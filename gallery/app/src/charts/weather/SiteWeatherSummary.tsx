import { useMemo } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty } from '@/components/ChartFrame'
import { fmt } from '@/lib/stats'
import { WEATHER_MIN_PAIRS, WEATHER_VARS } from '@/lib/weather'
import { spearman } from '@/lib/stats'
import type { FilterRow, MetaFile } from '@/lib/types'
import { useColourScheme } from '@/lib/colourSchemes'
import { siteColor } from '@/lib/siteColours'
import { HeatMatrix } from '@/components/HeatMatrix'

const MIN_N = 10
/** short names for the columns (two lines where a column is too narrow for one), the full name in the tooltip */
const short = (label: string) => label.replace('Daily temperature range', 'Temp. range').replace('Boundary-layer height', 'BLH').replace('Relative humidity', 'RH').replace('Precipitation', 'Precip.').replace('Temperature', 'Temp.')

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
  const table = useMemo(() => d3.groups(rows, (r) => r.site).map(([site, rs]) => {
    const code = rs[0]?.code ?? ''
    const withField = rs.filter((r) => typeof r[field] === 'number')
    return {
      site, code, n: withField.length,
      median: d3.median(withField, (r) => r[field] as number) ?? null,
      cells: WEATHER_VARS.map((v) => {
        const ok = withField.filter((r) => typeof r[v.era5] === 'number')
        return { v, n: ok.length, level: d3.mean(ok, (r) => r[v.era5] as number) ?? null,
          rho: ok.length >= MIN_N ? spearman(ok.map((r) => r[v.era5] as number), ok.map((r) => r[field] as number), WEATHER_MIN_PAIRS) : null }
      }),
    }
  }).filter((t) => t.n > 0 && t.cells.some((c) => c.n > 0)).sort((a, b) => a.site.localeCompare(b.site)), [rows, field])

  const colours = useColourScheme('weather-sites', 'diverging')
  const color = d3.scaleQuantize<string>().domain([-1, 1]).range(colours.stops)
  const cols = useMemo(() => WEATHER_VARS.map((v) => ({ key: v.key, label: short(v.label), title: v.label, tip: [v.label] })), [])
  const siteRows = useMemo(() => table.map((t) => ({ key: t.site, label: t.site, dot: siteColor(meta, t.site), size: 12 })), [table, meta])
  const selected = table.findIndex((t) => t.code === current)

  return (
    <ChartFrame id="weather-sites" title="Sites compared — which weather goes with the measurement where"
      subtitle={`Spearman ρ between ${field} and each weather variable over the filters' windows (ERA5), one row per site in the Filters panel's pick; blank below ${MIN_N} filters. Hover a cell for the site's mean of that variable; click a site to look at it alone.`}
      provenance="the per-site meteorology questions of addis_03_meteorology and meteorology_source_interaction, side by side">
      {table.length === 0 ? <Empty>No site in the pick has both {field} and weather.</Empty> : (
        <HeatMatrix rows={siteRows} columns={cols} cellWidth={[46, 90]} cellHeight={26} gap={2} maxLabelWidth={190} colLabels="wrap" colLabelSize={10.5} labelSize={10.5}
          extraColumns={[{ key: 'n', header: 'filters', text: (i) => String(table[i].n) }, { key: 'median', header: 'median', text: (i) => fmt(table[i].median, 2) }]}
          selectedRow={selected >= 0 ? selected : null} onRow={(i) => onSite(table[i].code)} color={color} format={(v) => fmt(v, 2)}
          cell={(i, j) => {
            const t = table[i], c = t.cells[j]
            return { value: c.rho, tip: [`${t.site}: ${field} vs ${c.v.label}`, c.rho === null ? `n = ${c.n} (too few)` : `ρ = ${fmt(c.rho, 2)} · n = ${c.n}`,
              `site mean ${c.v.label.toLowerCase()}: ${fmt(c.level, 1)}`, 'click to look at this site alone'] }
          }}
          legend={{ scheme: colours.scheme, label: 'Spearman ρ', width: 160, ticks: 4, format: (v) => (v > 0 ? `+${v}` : v < 0 ? `−${-v}` : '0') }} />
      )}
    </ChartFrame>
  )
}
