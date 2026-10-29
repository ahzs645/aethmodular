import { useMemo, useRef } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Note } from '@/components/ChartFrame'
import { ColorLegend } from '@/components/ColorLegend'
import { FieldSelect, prettyUnit } from '@/components/FieldSelect'
import { Legend } from '@/components/Legend'
import { ScrubField } from '@/components/ScrubField'
import { useTooltip } from '@/hooks/useTooltip'
import { useDimensions } from '@/hooks/useGalleryData'
import { useFitHeight } from '@/lib/chartFit'
import { useChartOptions } from '@/lib/chartOptions'
import { useColourScheme } from '@/lib/colourSchemes'
import { fmt } from '@/lib/stats'
import { FONT, INK } from '@/lib/theme'
import { useSearchParam } from '@/lib/url'
import { concentrationEdges, pollutionRose, type FilterWind } from '@/lib/windRose'
import type { WeatherSource } from '@/lib/weather'
import type { FilterRow, MetaFile } from '@/lib/types'
import { RosePanel, roseRadius, roseSize, useWindHours } from './roseParts'

const SPLITS = ['Site', 'Season'] as const
const STATS = ['Mean', 'Median', 'Percentile', 'Frequency'] as const
const MAX_PANELS = 16
/** a sector's statistic needs this many filters with air from it */
const MIN_FILTERS = 3

/**
 * Pollution rose (openair pollutionRose / percentileRose): the measurement by the direction the
 * air came from while each filter sampled. A filter's hours from each sector carry its
 * concentration (lib/windRose), so a wedge is the hour-weighted mean, median or percentile over
 * the hours from that sector; or, as openair draws it, a wind rose (wedge = share of hours) stacked
 * by concentration class, the classes at the filters' quartiles and 90th percentile. 16 sectors from
 * weather/wind.json, 8 from the wind roses until export_weather.py has written it. One rose per site
 * (or per season at one site), all on one radial scale.
 */
export function PollutionRose({ rows, meta, field, source, site, splits = SPLITS, pickField = true, id = 'weather-pollution-rose', title = 'Pollution rose — the measurement by where the air came from' }: {
  rows: FilterRow[]; meta: MetaFile
  /** the tab's measurement: the chart starts on it until the reader picks another */
  field: string
  source: WeatherSource
  site: { name: string; code: string } | null
  splits?: readonly (typeof SPLITS)[number][]
  /** false in a Dashboard tile, whose own picker sets `field` */
  pickField?: boolean
  id?: string; title?: string
}) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const tip = useTooltip(wrapRef)
  const { width } = useDimensions(wrapRef)
  const fit = useFitHeight()
  const [own, setOwn] = useSearchParam<string>(`${id}.field`, field)
  const f = pickField ? own : field
  const [pParam, setP] = useSearchParam<string>(`${id}.p`, '75')
  const p = Math.min(99, Math.max(50, Number(pParam) || 75))
  const colours = useColourScheme(id, 'sequential')
  // the options and when each applies (lib/chartOptions)
  const o = useChartOptions(id, [
    { key: 'field', when: () => pickField, node: () => <FieldSelect label="Measurement" value={own} meta={meta} onChange={setOwn} ratios={false} /> },
    { key: 'split', label: 'Split by', default: splits[0], choices: SPLITS.map((v) => ({ value: v, when: () => splits.includes(v) })),
      title: 'Site: one rose per site. Season: the chosen site, one rose per season.' },
    { key: 'stat', label: 'Wedges', default: 'Mean', choices: STATS.map((v) => ({ value: v })),
      title: "Mean, median or a percentile of the concentration over the hours from each sector; or Frequency: each sector's share of the hours, stacked by concentration class (openair pollutionRose)" },
    { key: 'p', when: (c) => c.stat === 'Percentile', node: () => <span className="control" title="Which percentile of the hours from each sector">Percentile
      <ScrubField label="Percentile" value={p} onChange={(v) => setP(String(Math.round(v)))} min={50} max={99} step={1} unit="%" width={90} /></span> },
    { key: 'colours', when: (c) => c.stat === 'Frequency', node: () => colours.picker },
  ])
  const split = o.values.split as (typeof SPLITS)[number]
  const stat = o.values.stat as (typeof STATS)[number]
  const { wind, fine, loading } = useWindHours(source)
  const nSectors = wind?.sectors.length ?? 16
  const unit = prettyUnit(meta.field_units?.[f])

  // each group's filters with the measurement and their window's wind
  const groups = useMemo(() => {
    if (!wind) return []
    const usable = rows.filter((r) => typeof r[f] === 'number' && Number.isFinite(r[f] as number))
    const one = site ? usable.filter((r) => r.site === site.name) : usable
    const keyed: [string, string, FilterRow[]][] = split === 'Site'
      ? d3.groups(usable, (r) => r.site).map(([k, rs]) => [k, meta.sites.find((s) => s.name === k)?.color ?? INK.muted, rs])
      : d3.groups(one, (r) => r.season).map(([k, rs]) => [k, meta.seasons.find((s) => s.name === k)?.color ?? INK.muted, rs])
    return keyed.map(([name, color, rs]) => {
      const filters: FilterWind[] = rs.flatMap((r) => { const w = wind.of(r.id); return w ? [{ id: r.id, value: r[f] as number, ...w }] : [] })
      return { name, color, filters }
    }).filter((g) => g.filters.length > 0).sort((a, b) => b.filters.length - a.filters.length)
  }, [wind, rows, f, split, site, meta])
  const shown = useMemo(() => groups.slice(0, MAX_PANELS), [groups])
  // one set of concentration classes for every panel, so a colour means one range everywhere
  const edges = useMemo(() => concentrationEdges(shown.flatMap((g) => g.filters.map((x) => x.value))), [shown])
  const roses = useMemo(() => shown.map((g) => ({ ...g, rose: pollutionRose(g.filters, nSectors, {
    stat: stat === 'Frequency' ? 'mean' : (stat.toLowerCase() as 'mean' | 'median' | 'percentile'), p, edges, minFilters: MIN_FILTERS }) })), [shown, nSectors, stat, p, edges])

  const size = roseSize(shown.length, width, fit)
  const R = roseRadius(size)
  const freq = stat === 'Frequency'
  // Frequency: % of each panel's hours (calm included), area-true like the wind roses; a statistic: linear from 0
  const pctOf = (h: number, rose: (typeof roses)[number]['rose']) => (rose.hours + rose.calm ? (100 * h) / (rose.hours + rose.calm) : 0)
  const rMax = freq
    ? d3.max(roses, (g) => d3.max(g.rose.sectors, (s) => pctOf(s.hours, g.rose))) ?? 10
    : d3.max(roses, (g) => d3.max(g.rose.sectors, (s) => s.value ?? 0)) ?? 1
  const r = (freq ? d3.scaleSqrt() : d3.scaleLinear()).domain([0, rMax || 1]).range([16, R]).nice()
  const half = Math.PI / nSectors
  const arc = d3.arc<{ a: number; r0: number; r1: number }>().startAngle((d) => d.a - half + 0.02).endAngle((d) => d.a + half - 0.02).innerRadius((d) => d.r0).outerRadius((d) => d.r1)
  const ramp = d3.interpolateRgbBasis(colours.stops)
  // the stacked classes: the ramp's palest step would vanish on the panel, so they start a fifth in
  const classColors = d3.quantize((t) => ramp(0.2 + 0.8 * t), edges.length + 1)
  const classLabel = (k: number) => k === 0 ? `< ${fmt(edges[0], 2)}` : k === edges.length ? `≥ ${fmt(edges[k - 1], 2)}` : `${fmt(edges[k - 1], 2)}–${fmt(edges[k], 2)}`
  const valueColor = d3.scaleSequential(ramp).domain([0, r.domain()[1]])
  const statName = stat === 'Percentile' ? `${p}th percentile` : stat.toLowerCase()
  const label = (name: string) => name.replace(`${site?.name} · `, '')
  const sectorName = (s: number) => wind?.sectors[s] ?? ''

  return (
    <ChartFrame id={id} title={title}
      subtitle={`${f} by the direction of each hour of wind inside the filters' sampling windows (${nSectors} sectors): ${freq
        ? `each wedge is the share of hours from that sector, stacked by the concentration of the filter those hours belong to; the middle is calm (${wind?.calm ?? '< 0.5 m/s'}).`
        : `each wedge is the ${statName} over the hours from that sector, every hour carrying its filter's concentration (sectors with fewer than ${MIN_FILTERS} filters are left out); the middle is the calm hours'.`} ${source === 'station' ? 'Airport station winds.' : 'ERA5 10 m winds.'} Every panel shares one radial scale.`}
      provenance="openair pollutionRose / percentileRose · meteorology_source_interaction §2"
      controls={o.controls()}>
      <div ref={wrapRef} className="chart-wrap">
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, justifyContent: 'center' }}>
          {loading ? <Empty>Loading the wind…</Empty> : !wind ? <Empty>No wind export: re-run gallery/data/export_weather.py (it writes weather/roses.json and wind.json).</Empty>
            : shown.length === 0 ? <Empty>No filters carry both {f} and wind in this subset.</Empty> : roses.map((g) => (
              <RosePanel key={g.name} size={size} title={label(g.name)} titleColor={g.color === INK.muted ? INK.text : g.color} nSectors={nSectors}
                rings={r.ticks(3).filter((t) => t > 0).map((t) => ({ r: r(t), label: freq ? `${t}%` : `${t}` }))}
                foot={`${g.rose.filters} filters · ${g.rose.hours + g.rose.calm} h`}>
                {g.rose.sectors.map((sec, s) => {
                  const a = (s * 2 * Math.PI) / nSectors
                  const head = [`${label(g.name)} · from ${sectorName(s)}`, `${sec.hours} h · ${sec.filters} filters`]
                  if (!freq) {
                    if (sec.value === null) return null
                    return <path key={s} d={arc({ a, r0: 16, r1: r(sec.value) }) ?? ''} fill={valueColor(sec.value)} stroke={INK.panel} strokeWidth={0.5}
                      onMouseEnter={(e) => tip.show(e, [...head, `${statName}: ${fmt(sec.value, 2)} ${unit}`])} onMouseLeave={tip.hide} />
                  }
                  let acc = 0
                  return sec.classHours.map((h, k) => {
                    const v = pctOf(h, g.rose)
                    const r0 = r(acc), r1 = r(acc + v)
                    acc += v
                    return v > 0 ? <path key={`${s}-${k}`} d={arc({ a, r0, r1 }) ?? ''} fill={classColors[k]} stroke={INK.panel} strokeWidth={0.5}
                      onMouseEnter={(e) => tip.show(e, [...head, `${classLabel(k)} ${unit}: ${fmt(v, 1)} % of hours`, `all from ${sectorName(s)}: ${fmt(pctOf(sec.hours, g.rose), 1)} %`])}
                      onMouseLeave={tip.hide} /> : null
                  })
                })}
                <circle r={14} fill={INK.panel} stroke={INK.grid} />
                <text dy="0.35em" textAnchor="middle" fontSize={9} fill={INK.muted} fontFamily={FONT.mono}>
                  {freq ? `${fmt(pctOf(g.rose.calm, g.rose), 0)}%` : g.rose.calmValue === null ? '—' : fmt(g.rose.calmValue, g.rose.calmValue >= 100 ? 0 : 1)}</text>
              </RosePanel>
            ))}
        </div>
        {shown.length > 0 && (freq
          ? <Legend items={classColors.map((c, k) => ({ label: `${classLabel(k)} ${unit}`, color: c, shape: 'square' as const }))} note="classes at the filters' quartiles and 90th percentile" />
          : <div className="legend"><ColorLegend scale={valueColor} label={`${statName} ${f}${unit ? ` (${unit})` : ''}`} scheme={colours.scheme} /></div>)}
        {tip.node}
      </div>
      {wind && !fine && <Note>8 sectors: weather/wind.json (16 sectors × 0.5 m/s) is not exported yet. Re-run gallery/data/export_weather.py for the finer rose.</Note>}
      {groups.length > MAX_PANELS && <Note>Showing the {MAX_PANELS} groups with the most filters of {groups.length}.</Note>}
    </ChartFrame>
  )
}
