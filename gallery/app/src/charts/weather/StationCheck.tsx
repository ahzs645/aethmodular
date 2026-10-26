import { useRef, useState } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty } from '@/components/ChartFrame'
import { XAxis, YAxis } from '@/components/Axes'
import { IdentityLine } from '@/components/RefLines'
import { RegressionLines, useRegressionSummary } from '@/components/RegressionOverlay'
import { SpectrumModal } from '@/components/FilterSpectrumPlot'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { fmt, regression } from '@/lib/stats'
import { FONT, INK } from '@/lib/theme'
import { ExpandIcon } from '@/components/Icons'
import type { StationCompare } from '@/lib/weather'

const VARS = [
  { k: 't', label: 'Mean temperature', unit: '°C' }, { k: 'tmin', label: 'Minimum temperature', unit: '°C' },
  { k: 'tmax', label: 'Maximum temperature', unit: '°C' }, { k: 'p', label: 'Precipitation', unit: 'mm/day' },
  { k: 'rh', label: 'Relative humidity', unit: '%' }, { k: 'ws', label: 'Wind speed', unit: 'm/s' },
] as const
type Var = (typeof VARS)[number]
// room for the crossplots' own axes: tick labels and titles on x and y, a title line above
const M = { top: 30, right: 14, bottom: 46, left: 56 }
const DOT = '#2c7fb8'

/**
 * ERA5 against the Addis Ababa airport station (Meteostat, WMO 63450), day by day over the
 * station record (Dec 2022 – Sep 2024): how far the reanalysis can stand in for the station
 * the Addis notebooks used. Mean bias is ERA5 minus station. Every variable shows as a square
 * crossplot drawn like the gallery's others (axes, 1:1 line, the sidebar's fits and their
 * readout); click one to open it larger.
 */
export function StationCheck({ data }: { data: StationCompare | null }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
  const [open, setOpen] = useState<Var | null>(null)
  const cols = Math.max(1, Math.min(3, Math.floor(((width || 900) + 12) / (300 + 12))))
  const pw = Math.min(360, ((width || 900) - (cols - 1) * 12) / cols)
  const side = Math.max(140, pw - M.left - M.right)

  return (
    <ChartFrame id="weather-station" title="ERA5 against the Addis Ababa airport station — day by day"
      subtitle="Each dot is one local day with at least 18 station hours. Dashed: 1:1; the fit lines and readout follow the sidebar's fit choice; ⤢ opens a panel larger. ERA5 is a 0.25° grid-box average, so it smooths the station's extremes; precipitation is the weakest match (ERA5 is known to over-rain the Ethiopian highlands), so rain-washout results from ERA5 at Addis deserve the station check."
      provenance="Meteostat hourly, WMO 63450 (the station behind addis_03_meteorology and the meteorology notebooks) · ERA5 via Open-Meteo">
      <div ref={wrapRef} className="chart-wrap" style={{ display: 'flex', flexWrap: 'wrap', gap: 12, justifyContent: 'center' }}>
        {!data ? <Empty>Loading the station comparison…</Empty> : VARS.map((v) => (
          <StationPanel key={v.k} v={v} data={data} side={side} tip={tip} onOpen={() => setOpen(v)} />
        ))}
        {tip.node}
      </div>
      {open && data && <SpectrumModal id={`${open.label} (${open.unit})`} title="ERA5 against the Addis Ababa airport station" onClose={() => setOpen(null)}>
        <div style={{ display: 'flex', justifyContent: 'center' }}>
          <StationPanel v={open} data={data} side={Math.max(260, Math.min(620, window.innerWidth - 220, window.innerHeight - 260))} tip={tip} />
        </div>
      </SpectrumModal>}
    </ChartFrame>
  )
}

function StationPanel({ v, data, side, tip, onOpen }: {
  v: Var; data: StationCompare; side: number; tip: ReturnType<typeof useTooltip>; onOpen?: () => void
}) {
  const pairs = data.date.map((d, i) => ({ d, s: data.station[v.k][i], e: data.era5[v.k][i] }))
    .filter((p): p is { d: string; s: number; e: number } => p.s !== null && p.e !== null)
  const hi = d3.max(pairs, (p) => Math.max(p.s, p.e)) ?? 1, lo = Math.min(0, d3.min(pairs, (p) => Math.min(p.s, p.e)) ?? 0)
  // station and ERA5 share one scale, so the panel is square and the 1:1 line is the diagonal
  const x = d3.scaleLinear().domain([lo, hi]).range([0, side]).nice()
  const y = d3.scaleLinear().domain(x.domain()).range([side, 0])
  const stats = regression(pairs.map((p) => p.s), pairs.map((p) => p.e), { errorsInVariables: true })
  const bias = d3.mean(pairs, (p) => p.e - p.s)
  const tot = v.k === 'p' ? [d3.sum(pairs, (p) => p.s), d3.sum(pairs, (p) => p.e)] : null
  const [d0, d1] = x.domain()
  const r = side > 300 ? 3.2 : 2.6
  const summary = useRegressionSummary({ stats, width: side + M.left + M.right - 24 })
  const top = M.top + summary.height + (summary.height ? 12 : 0)
  return (
    <div className="expandable-panel">
    {onOpen && <button type="button" className="btn quiet panel-expand" style={{ right: M.right }} onClick={onOpen} aria-label={`Open ${v.label} larger`} title="Open larger"><ExpandIcon /></button>}
    <svg width={side + M.left + M.right} height={side + top + M.bottom} fontFamily={FONT.family}>
      <text x={M.left} y={13} fontSize={12} fontWeight={600} fill={INK.text}>{v.label}
        <tspan fontWeight={400} fill={INK.muted} fontSize={10.5}> · bias {bias !== undefined && bias >= 0 ? '+' : ''}{fmt(bias, 2)} {v.unit}{tot ? ` · totals ${fmt(tot[0], 0)} / ${fmt(tot[1], 0)} mm` : ''}</tspan></text>
      <g transform="translate(12,24)">{summary.node}</g>
      <g className="regression-plot" transform={`translate(${M.left},${top})`}>
        <YAxis scale={y} x={0} label={`ERA5 (${v.unit})`} gridWidth={side} tickCount={4} labelX={-40} />
        <XAxis scale={x} y={side} label={`station (${v.unit})`} tickCount={4} labelDy={36} />
        <IdentityLine x1={x(d0)} y1={y(d0)} x2={x(d1)} y2={y(d1)} />
        <RegressionLines stats={stats} x={x} y={y} domain={x.domain()} width={side} height={side} />
        {pairs.map((p) => <circle key={p.d} cx={x(p.s)} cy={y(p.e)} r={r} fill={DOT} fillOpacity={0.55} stroke={INK.panel} strokeWidth={0.6}
          onMouseEnter={(e) => tip.show(e, [p.d, `station ${fmt(p.s, 1)} · ERA5 ${fmt(p.e, 1)} ${v.unit}`])} onMouseLeave={tip.hide} />)}
      </g>
    </svg>
    </div>
  )
}
