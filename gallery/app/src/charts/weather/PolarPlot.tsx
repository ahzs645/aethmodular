import { useMemo, useRef } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Note } from '@/components/ChartFrame'
import { ColorLegend } from '@/components/ColorLegend'
import { FieldSelect, prettyUnit } from '@/components/FieldSelect'
import { ScrubField } from '@/components/ScrubField'
import { useTooltip } from '@/hooks/useTooltip'
import { useDimensions } from '@/hooks/useGalleryData'
import { useFitHeight } from '@/lib/chartFit'
import { useChartOptions } from '@/lib/chartOptions'
import { useColourScheme } from '@/lib/colourSchemes'
import { fmt } from '@/lib/stats'
import { INK } from '@/lib/theme'
import { useSearchParam } from '@/lib/url'
import { polarGrid, type FilterWind } from '@/lib/windRose'
import type { WeatherSource } from '@/lib/weather'
import type { FilterRow, MetaFile } from '@/lib/types'
import { RosePanel, roseRadius, roseSize, useWindHours } from './roseParts'

const SPLITS = ['Site', 'Season'] as const
const MAX_PANELS = 16
/** an unsmoothed cell needs this many filters, a smoothed one this many in its neighbourhood */
const MIN_FILTERS = 3

/**
 * Bivariate polar plot (openair polarPlot): the measurement as colour over wind direction (angle)
 * and wind speed (radius), each cell the hour-weighted mean concentration of the hours that blew
 * from that direction at that speed (lib/windRose polarGrid), smoothed over neighbouring cells
 * unless the reader turns it off. A source upwind shows as a hot patch at the speed that brings
 * it; a local one as a hot centre (low speeds). Needs weather/wind.json (16 sectors × 0.5 m/s bins,
 * export_weather.py): the wind roses' 3 speed classes are too coarse for a surface.
 */
export function PolarPlot({ rows, meta, field, source, site, splits = SPLITS, pickField = true, id = 'weather-polar', title = 'Polar plot — the measurement by wind direction and speed' }: {
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
  const [minParam, setMin] = useSearchParam<string>(`${id}.minh`, '10')
  const minHours = Math.max(1, Number(minParam) || 10)
  const colours = useColourScheme(id, 'sequential')
  // the options and when each applies (lib/chartOptions)
  const o = useChartOptions(id, [
    { key: 'field', when: () => pickField, node: () => <FieldSelect label="Measurement" value={own} meta={meta} onChange={setOwn} ratios={false} /> },
    { key: 'split', label: 'Split by', default: splits[0], choices: SPLITS.map((v) => ({ value: v, when: () => splits.includes(v) })),
      title: 'Site: one plot per site. Season: the chosen site, one plot per season.' },
    { key: 'smooth', label: 'Smooth', toggle: true, default: true, title: 'A Gaussian-kernel mean over neighbouring cells (one sector, one speed bin), weighted by their hours' },
    { key: 'minh', node: () => <span className="control" title="A cell is drawn only with at least this many hours (smoothed: kernel-weighted hours)">Min hours
      <ScrubField label="Minimum hours per cell" value={minHours} onChange={(v) => setMin(String(Math.round(v)))} min={1} max={200} step={1} width={80} /></span> },
  ])
  const split = o.values.split as (typeof SPLITS)[number]
  const smooth = o.values.smooth as boolean
  const { wind, fine, loading } = useWindHours(source)
  const edges = wind?.speedEdges ?? []
  const nSectors = wind?.sectors.length ?? 16, nSpeed = edges.length
  const unit = prettyUnit(meta.field_units?.[f])

  const panels = useMemo(() => {
    if (!wind || !fine) return []
    const usable = rows.filter((r) => typeof r[f] === 'number' && Number.isFinite(r[f] as number))
    const one = site ? usable.filter((r) => r.site === site.name) : usable
    const keyed: [string, string, FilterRow[]][] = split === 'Site'
      ? d3.groups(usable, (r) => r.site).map(([k, rs]) => [k, meta.sites.find((s) => s.name === k)?.color ?? INK.muted, rs])
      : d3.groups(one, (r) => r.season).map(([k, rs]) => [k, meta.seasons.find((s) => s.name === k)?.color ?? INK.muted, rs])
    return keyed.map(([name, color, rs]) => {
      const filters: FilterWind[] = rs.flatMap((r) => { const w = wind.of(r.id); return w ? [{ id: r.id, value: r[f] as number, ...w }] : [] })
      const calmH = d3.sum(filters, (x) => x.calm)
      return { name, color, filters, cells: polarGrid(filters, nSectors, nSpeed, { smooth, minHours, minFilters: MIN_FILTERS }),
        calm: calmH, calmValue: calmH >= minHours ? d3.sum(filters, (x) => x.calm * x.value) / calmH : null }
    }).filter((g) => g.filters.length > 0).sort((a, b) => b.filters.length - a.filters.length)
  }, [wind, fine, rows, f, split, site, meta, nSectors, nSpeed, smooth, minHours])
  const shown = useMemo(() => panels.slice(0, MAX_PANELS), [panels])

  // one colour scale and one speed axis for every panel; the colours clipped at the 2nd and 98th percentiles
  const values = useMemo(() => shown.flatMap((g) => g.cells.flatMap((c) => (c.value === null ? [] : [c.value]))).sort(d3.ascending), [shown])
  const lo = d3.quantile(values, 0.02) ?? 0, hi = d3.quantile(values, 0.98) ?? 1
  const color = d3.scaleSequential(d3.interpolateRgbBasis(colours.stops)).domain([lo, hi > lo ? hi : lo + 1]).clamp(true)
  const upper = (k: number) => edges[k + 1] ?? edges[k] + 1
  const topBin = d3.max(shown, (g) => d3.max(g.cells, (c) => (c.value === null ? -1 : c.speed))) ?? -1
  const size = roseSize(shown.length, width, fit)
  const R = roseRadius(size)
  const rs = d3.scaleLinear().domain([0, topBin >= 0 ? upper(topBin) : 4]).range([0, R]).nice()
  const half = Math.PI / nSectors
  const arc = d3.arc<{ a: number; r0: number; r1: number }>().startAngle((d) => d.a - half).endAngle((d) => d.a + half).innerRadius((d) => d.r0).outerRadius((d) => d.r1)
  const label = (name: string) => name.replace(`${site?.name} · `, '')
  const speedText = (k: number) => (k === nSpeed - 1 ? `≥ ${edges[k]} m/s` : `${edges[k]}–${upper(k)} m/s`)

  return (
    <ChartFrame id={id} title={title}
      subtitle={`${f} as colour over the direction (angle, ${nSectors} sectors) and speed (radius) of each hour of wind inside the filters' sampling windows; each cell is the mean over its hours, every hour carrying its filter's concentration${smooth ? ', smoothed over the neighbouring cells' : ''}. Cells with fewer than ${minHours} h or ${MIN_FILTERS} filters are blank; the centre disc is the calm hours (${wind?.calm ?? '< 0.5 m/s'}). ${source === 'station' ? 'Airport station winds.' : 'ERA5 10 m winds.'}`}
      provenance="openair polarPlot · meteorology_source_interaction §2"
      controls={o.controls()}>
      <div ref={wrapRef} className="chart-wrap">
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, justifyContent: 'center' }}>
          {loading ? <Empty>Loading the wind…</Empty>
            : !fine ? <Empty>The polar plot needs each filter's hours by wind direction and speed (weather/wind.json), which is not exported yet. Re-run gallery/data/export_weather.py to write it; the pollution rose meanwhile uses the wind roses' 8 sectors.</Empty>
              : shown.length === 0 ? <Empty>No filters carry both {f} and wind in this subset.</Empty> : shown.map((g) => (
                <RosePanel key={g.name} size={size} title={label(g.name)} titleColor={g.color === INK.muted ? INK.text : g.color} nSectors={nSectors}
                  rings={rs.ticks(3).filter((t) => t > 0).map((t) => ({ r: rs(t), label: `${t} m/s` }))}
                  foot={`${g.filters.length} filters · ${d3.sum(g.filters, (x) => x.calm + d3.sum(x.cells, (c) => c.hours))} h`}>
                  {g.cells.map((c) => c.value === null || c.speed > topBin ? null : (
                    <path key={`${c.sector}-${c.speed}`} d={arc({ a: (c.sector * 2 * Math.PI) / nSectors, r0: rs(edges[c.speed]), r1: rs(upper(c.speed)) }) ?? ''}
                      fill={color(c.value)} stroke={color(c.value)} strokeWidth={0.4}
                      onMouseEnter={(e) => tip.show(e, [`${label(g.name)} · from ${wind!.sectors[c.sector]} at ${speedText(c.speed)}`,
                        `mean ${f}: ${fmt(c.value, 2)} ${unit}`, `${smooth ? 'kernel-weighted ' : ''}${fmt(c.hours, smooth ? 1 : 0)} h · ${c.filters} filters`])}
                      onMouseLeave={tip.hide} />
                  ))}
                  <circle r={rs(edges[0] ?? 0.5)} fill={g.calmValue === null ? INK.panel : color(g.calmValue)} stroke={INK.grid}
                    onMouseEnter={(e) => tip.show(e, [`${label(g.name)} · calm (${wind!.calm})`, g.calmValue === null ? `${g.calm} h: too few` : `mean ${f}: ${fmt(g.calmValue, 2)} ${unit}`])}
                    onMouseLeave={tip.hide} />
                </RosePanel>
              ))}
        </div>
        {shown.length > 0 && <div className="legend"><ColorLegend scale={color} label={`mean ${f}${unit ? ` (${unit})` : ''}`} scheme={colours.scheme} note="clipped at the 2nd and 98th percentiles" /></div>}
        {tip.node}
      </div>
      {panels.length > MAX_PANELS && <Note>Showing the {MAX_PANELS} groups with the most filters of {panels.length}.</Note>}
    </ChartFrame>
  )
}
