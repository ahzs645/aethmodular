import { useMemo, useRef } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty } from '@/components/ChartFrame'
import { XAxis, YAxis } from '@/components/Axes'
import { IdentityLine } from '@/components/RefLines'
import { RegressionLines, useRegressionSummary } from '@/components/RegressionOverlay'
import { FacetGrid } from '@/components/FacetGrid'
import { useTooltip } from '@/hooks/useTooltip'
import { fmt, regression } from '@/lib/stats'
import { FONT } from '@/lib/theme'
import { dotFill, dotRim } from '@/lib/highlight'
import type { StationCompare } from '@/lib/weather'

const VARS = [
  { k: 't', label: 'Mean temperature', unit: '°C' }, { k: 'tmin', label: 'Minimum temperature', unit: '°C' },
  { k: 'tmax', label: 'Maximum temperature', unit: '°C' }, { k: 'p', label: 'Precipitation', unit: 'mm/day' },
  { k: 'rh', label: 'Relative humidity', unit: '%' }, { k: 'ws', label: 'Wind speed', unit: 'm/s' },
] as const
type Var = (typeof VARS)[number]
// room for the crossplots' own axes: tick labels and titles on x and y (the title is the facet's)
const M = { top: 12, right: 14, bottom: 46, left: 56 }
const DOT = '#2c7fb8'

/**
 * ERA5 against the Addis Ababa airport station (Meteostat, WMO 63450), day by day over the
 * station record (Dec 2022 – Sep 2024): how far the reanalysis can stand in for the station
 * the Addis notebooks used. Mean bias is ERA5 minus station. Every variable shows as a square
 * crossplot drawn like the gallery's others (axes, 1:1 line, the sidebar's fits and their
 * readout), laid out as facets (components/FacetGrid); ⤢ opens one larger.
 */
export function StationCheck({ data }: { data: StationCompare | null }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const tip = useTooltip(wrapRef)
  // the title row's readout: mean bias (and rain totals), from the same days each panel draws
  const notes = useMemo(() => Object.fromEntries(VARS.map((v) => {
    if (!data) return [v.k, null]
    const ps = pairsOf(data, v.k), bias = d3.mean(ps, (p) => p.e - p.s)
    const tot = v.k === 'p' ? ` · totals ${fmt(d3.sum(ps, (p) => p.s), 0)} / ${fmt(d3.sum(ps, (p) => p.e), 0)} mm` : ''
    return [v.k, `bias ${bias !== undefined && bias >= 0 ? '+' : ''}${fmt(bias, 2)} ${v.unit}${tot}`]
  })), [data])

  return (
    <ChartFrame id="weather-station" title="ERA5 against the Addis Ababa airport station — day by day"
      subtitle="Each dot is one local day with at least 18 station hours. Dashed: 1:1; the fit lines and readout follow the sidebar's fit choice; ⤢ opens a panel larger. ERA5 is a 0.25° grid-box average, so it smooths the station's extremes; precipitation is the weakest match (ERA5 is known to over-rain the Ethiopian highlands), so rain-washout results from ERA5 at Addis deserve the station check."
      provenance="Meteostat hourly, WMO 63450 (the station behind addis_03_meteorology and the meteorology notebooks) · ERA5 via Open-Meteo"
      // six small crossplots: each keeps its readout above it in the one-chart view too (the display menu can move them to the options)
      statsPlace="plot">
      <div ref={wrapRef} className="chart-wrap">
        {!data ? <Empty>Loading the station comparison…</Empty>
          : <FacetGrid items={[...VARS]} minCell={300} maxCols={3} maxCell={360} gap={12} center expandable
              title={(v) => v.label} note={(v) => notes[v.k]} label={(v) => `${v.label} (${v.unit})`}
              // square: the cell's width, or in a tile its row's share of the height (less the readout and axes)
              render={(v, w, _, cell) => <StationPanel v={v} data={data} tip={tip}
                side={Math.max(140, Math.min(w - M.left - M.right, cell.height === null ? Infinity : cell.height - 110))} />} />}
        {tip.node}
      </div>
    </ChartFrame>
  )
}

const pairsOf = (data: StationCompare, k: Var['k']) => data.date.map((d, i) => ({ d, s: data.station[k][i], e: data.era5[k][i] }))
  .filter((p): p is { d: string; s: number; e: number } => p.s !== null && p.e !== null)

function StationPanel({ v, data, side, tip }: {
  v: Var; data: StationCompare; side: number; tip: ReturnType<typeof useTooltip>
}) {
  // memoised: the fit's confidence band is cached per stats object (lib/regressionCI), so a fresh fit on
  // every hover would bootstrap it again in all six panels
  const pairs = useMemo(() => pairsOf(data, v.k), [data, v.k])
  const hi = d3.max(pairs, (p) => Math.max(p.s, p.e)) ?? 1, lo = Math.min(0, d3.min(pairs, (p) => Math.min(p.s, p.e)) ?? 0)
  // station and ERA5 share one scale, so the panel is square and the 1:1 line is the diagonal
  const x = d3.scaleLinear().domain([lo, hi]).range([0, side]).nice()
  const y = d3.scaleLinear().domain(x.domain()).range([side, 0])
  const stats = useMemo(() => regression(pairs.map((p) => p.s), pairs.map((p) => p.e), { errorsInVariables: true }), [pairs])
  const [d0, d1] = x.domain()
  const r = side > 300 ? 3.2 : 2.6
  const summary = useRegressionSummary({ stats, width: side + M.left + M.right - 24, panel: { title: v.label } })
  const top = M.top + summary.height + (summary.height ? 12 : 0)
  return (
    <svg width={side + M.left + M.right} height={side + top + M.bottom} fontFamily={FONT.family}>
      <g transform="translate(12,6)">{summary.node}</g>
      <g className="regression-plot" transform={`translate(${M.left},${top})`}>
        <YAxis scale={y} x={0} label={`ERA5 (${v.unit})`} gridWidth={side} tickCount={4} labelX={-40} />
        <XAxis scale={x} y={side} label={`station (${v.unit})`} tickCount={4} labelDy={36} />
        <IdentityLine x1={x(d0)} y1={y(d0)} x2={x(d1)} y2={y(d1)} />
        <RegressionLines stats={stats} x={x} y={y} domain={x.domain()} width={side} height={side} />
        {pairs.map((p) => <circle key={p.d} cx={x(p.s)} cy={y(p.e)} r={r} fill={DOT} {...dotFill('dense')} {...dotRim()}
          onMouseEnter={(e) => tip.show(e, [p.d, `station ${fmt(p.s, 1)} · ERA5 ${fmt(p.e, 1)} ${v.unit}`])} onMouseLeave={tip.hide} />)}
      </g>
    </svg>
  )
}
