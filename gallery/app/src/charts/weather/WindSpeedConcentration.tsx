import { useMemo, useRef, useState } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty } from '@/components/ChartFrame'
import { XAxis, YAxis } from '@/components/Axes'
import { FieldSelect } from '@/components/FieldSelect'
import { Legend, toggleIn } from '@/components/Legend'
import { ScrubField } from '@/components/ScrubField'
import { useTooltip } from '@/hooks/useTooltip'
import { useDimensions } from '@/hooks/useGalleryData'
import { useFitHeight } from '@/lib/chartFit'
import { useChartOptions } from '@/lib/chartOptions'
import { fmt } from '@/lib/stats'
import { INK, MARGIN, withUnit } from '@/lib/theme'
import { useSearchParam } from '@/lib/url'
import { speedBins } from '@/lib/windRose'
import type { FilterRow, MetaFile } from '@/lib/types'

/** a bin's median is drawn from this many filters (its IQR from 4, lib/windRose speedBins) */
const MIN_N = 3

/**
 * The measurement by wind speed: each filter's window-mean speed binned (0.5 m/s by default), the
 * median and interquartile range of the filters in each bin, one line per site. Dilution reads as
 * a falling line: stronger wind, more mixing, lower concentration; a rising one points at a source
 * the wind brings in (or resuspended dust).
 */
export function WindSpeedConcentration({ rows, meta, field, wspdField, pickField = true, id = 'weather-wind-speed', title = 'The measurement by wind speed — dilution or transport' }: {
  rows: FilterRow[]; meta: MetaFile
  /** the tab's measurement: the chart starts on it until the reader picks another */
  field: string
  /** the window-mean wind speed field (ERA5 or the station) */
  wspdField: string
  /** false in a Dashboard tile, whose own picker sets `field` */
  pickField?: boolean
  id?: string; title?: string
}) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const tip = useTooltip(wrapRef)
  const { width } = useDimensions(wrapRef)
  const height = useFitHeight() ?? 360
  const [own, setOwn] = useSearchParam<string>(`${id}.field`, field)
  const f = pickField ? own : field
  const [binParam, setBin] = useSearchParam<string>(`${id}.bin`, '0.5')
  const bin = Math.min(2, Math.max(0.25, Number(binParam) || 0.5))
  const [hidden, setHidden] = useState<Set<string>>(new Set())
  const [hover, setHover] = useState<string | null>(null)
  // the options and when each applies (lib/chartOptions)
  const o = useChartOptions(id, [
    { key: 'field', when: () => pickField, node: () => <FieldSelect label="Measurement" value={own} meta={meta} onChange={setOwn} ratios={false} /> },
    { key: 'bin', node: () => <span className="control" title="Width of each wind-speed bin">Bin
      <ScrubField label="Wind-speed bin width" value={bin} onChange={(v) => setBin(String(Math.round(v * 4) / 4))} min={0.25} max={2} step={0.25} unit="m/s" width={96} /></span> },
    { key: 'bands', label: 'IQR bands', toggle: true, default: true },
  ])
  const bands = o.values.bands as boolean

  const allSeries = useMemo(() => meta.sites.map((s) => {
    const pts = rows.filter((r) => r.site === s.name && typeof r[f] === 'number' && typeof r[wspdField] === 'number')
      .map((r) => ({ speed: r[wspdField] as number, value: r[f] as number }))
    return { name: s.name, color: s.color, n: pts.length, bins: speedBins(pts, bin).filter((b) => b.n >= MIN_N) }
  }).filter((s) => s.bins.length > 0), [rows, meta.sites, f, wspdField, bin])
  const series = allSeries.filter((s) => !hidden.has(s.name))
  const dim = (name: string) => (hover === null || hover === name || !series.some((s) => s.name === hover) ? 1 : 0.15)

  const innerW = Math.max(240, width - MARGIN.left - MARGIN.right)
  const innerH = height - MARGIN.top - MARGIN.bottom
  const mid = (b: { lo: number; hi: number }) => (b.lo + b.hi) / 2
  const x = d3.scaleLinear().domain([0, d3.max(series, (s) => d3.max(s.bins, (b) => b.hi)) ?? 4]).range([0, innerW]).nice()
  const top = d3.max(series, (s) => d3.max(s.bins, (b) => (bands ? b.q3 : null) ?? b.median ?? 0)) ?? 1
  const y = d3.scaleLinear().domain([0, top * 1.08 || 1]).range([innerH, 0]).nice()

  return (
    <ChartFrame id={id} title={title}
      subtitle={`${f} against each filter's mean wind speed over its sampling window (${wspdField}), in ${bin} m/s bins: the line is each site's median, the band its interquartile range (from 4 filters); bins with fewer than ${MIN_N} filters are left out.`}
      provenance="openair polarPlot's speed axis, one dimension · meteorology_source_interaction §2"
      controls={o.controls()}>
      <div ref={wrapRef} className="chart-wrap">
        {series.length === 0 ? <Empty>No site has both {f} and {wspdField} in this subset.</Empty> : (
          <svg width={width} height={height} className="animated">
            <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
              <YAxis scale={y} x={0} label={withUnit(f, meta.field_units)} gridWidth={innerW} />
              <XAxis scale={x} y={innerH} label={withUnit(wspdField, { [wspdField]: 'm/s' })} />
              {series.map((s) => {
                const withBand = s.bins.filter((b) => b.q1 !== null && b.q3 !== null)
                const area = d3.area<(typeof withBand)[number]>().x((b) => x(mid(b))).y0((b) => y(b.q1!)).y1((b) => y(b.q3!)).curve(d3.curveMonotoneX)
                const line = d3.line<(typeof s.bins)[number]>().x((b) => x(mid(b))).y((b) => y(b.median!)).curve(d3.curveMonotoneX)
                return (
                  <g key={s.name} opacity={dim(s.name)} data-export-site={s.name} data-export-color={s.color}>
                    {bands && <path d={area(withBand) ?? ''} fill={s.color} fillOpacity={0.18} pointerEvents="none" />}
                    <path d={line(s.bins) ?? ''} fill="none" stroke={s.color} strokeWidth={2.4} pointerEvents="none" />
                    {s.bins.map((b) => (
                      <circle key={b.lo} cx={x(mid(b))} cy={y(b.median!)} r={3.6} fill={s.color} stroke={INK.panel} strokeWidth={0.8}
                        onMouseEnter={(e) => tip.show(e, [`${s.name} · ${fmt(b.lo, 2)}–${fmt(b.hi, 2)} m/s`, `median = ${fmt(b.median)}`,
                          b.q1 !== null ? `IQR ${fmt(b.q1)} – ${fmt(b.q3)}` : 'IQR n/a (n<4)', `n = ${b.n}`])}
                        onMouseLeave={tip.hide} />
                    ))}
                  </g>
                )
              })}
            </g>
          </svg>
        )}
        <Legend items={allSeries.map((s) => ({ label: s.name, color: s.color, shape: 'line' as const, detail: `n=${s.n}` }))}
          hidden={hidden} onToggle={(l) => setHidden((h) => toggleIn(h, l))} onHover={setHover} highlighted={hover} />
        {tip.node}
      </div>
    </ChartFrame>
  )
}
