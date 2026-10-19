import { useRef } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Select } from '@/components/ChartFrame'
import { useSearchParam } from '@/lib/url'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { fmt, pearson } from '@/lib/stats'
import { FONT, INK } from '@/lib/theme'
import type { StationCompare } from '@/lib/weather'

const VARS = [
  { k: 't', label: 'Mean temperature', unit: '°C' }, { k: 'tmin', label: 'Minimum temperature', unit: '°C' },
  { k: 'tmax', label: 'Maximum temperature', unit: '°C' }, { k: 'p', label: 'Precipitation', unit: 'mm/day' },
  { k: 'rh', label: 'Relative humidity', unit: '%' }, { k: 'ws', label: 'Wind speed', unit: 'm/s' },
] as const
const S = 220, M = { top: 24, right: 10, bottom: 34, left: 40 }

/**
 * ERA5 against the Addis Ababa airport station (Meteostat, WMO 63450), day by day over the
 * station record (Dec 2022 – Sep 2024): how far the reanalysis can stand in for the station
 * the Addis notebooks used. Mean bias is ERA5 minus station.
 */
export function StationCheck({ data }: { data: StationCompare | null }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
  // every variable as a small square, or one larger; x and y share one scale, so each panel is square
  const [pick, setPick] = useSearchParam<string>('stationvar', 'All', ['All', ...VARS.map((v) => v.k)])
  const shownVars = pick === 'All' ? VARS : VARS.filter((v) => v.k === pick)
  const side = pick === 'All' ? S - M.top - M.bottom : Math.max(160, Math.min(440, (width || 600) - M.left - M.right - 10))
  const iw = side, ih = side, pw = side + M.left + M.right, ph = side + M.top + M.bottom

  return (
    <ChartFrame id="weather-station" title="ERA5 against the Addis Ababa airport station — day by day"
      subtitle="Each dot is one local day with at least 18 station hours. Dashed: 1:1. ERA5 is a 0.25° grid-box average, so it smooths the station's extremes; precipitation is the weakest match (ERA5 is known to over-rain the Ethiopian highlands), so rain-washout results from ERA5 at Addis deserve the station check."
      provenance="Meteostat hourly, WMO 63450 (the station behind addis_03_meteorology and the meteorology notebooks) · ERA5 via Open-Meteo"
      controls={<Select label="Show" value={pick} options={['All', ...VARS.map((v) => v.k)]} optionLabel={(k) => (k === 'All' ? 'Every variable' : VARS.find((v) => v.k === k)?.label ?? k)} onChange={setPick} />}>
      <div ref={wrapRef} className="chart-wrap" style={{ display: 'flex', flexWrap: 'wrap', gap: 10, justifyContent: 'center' }}>
        {!data ? <Empty>Loading the station comparison…</Empty> : shownVars.map((v) => {
          const pairs = data.date.map((d, i) => ({ d, s: data.station[v.k][i], e: data.era5[v.k][i] })).filter((p): p is { d: string; s: number; e: number } => p.s !== null && p.e !== null)
          const hi = d3.max(pairs, (p) => Math.max(p.s, p.e)) ?? 1, lo = Math.min(0, d3.min(pairs, (p) => Math.min(p.s, p.e)) ?? 0)
          const x = d3.scaleLinear().domain([lo, hi]).range([0, iw]).nice(), y = d3.scaleLinear().domain(x.domain()).range([ih, 0])
          const bias = d3.mean(pairs, (p) => p.e - p.s), r = pearson(pairs.map((p) => p.s), pairs.map((p) => p.e))
          const tot = v.k === 'p' ? [d3.sum(pairs, (p) => p.s), d3.sum(pairs, (p) => p.e)] : null
          return (
            <svg key={v.k} width={pw} height={ph} fontFamily={FONT.family}>
              <text x={M.left} y={13} fontSize={11.5} fontWeight={600} fill={INK.text}>{v.label} <tspan fontWeight={400} fill={INK.muted}>({v.unit})</tspan></text>
              <g transform={`translate(${M.left},${M.top})`}>
                {x.ticks(4).map((t) => <g key={t}>
                  <line x1={x(t)} x2={x(t)} y1={0} y2={ih} stroke={INK.grid} /><line x1={0} x2={iw} y1={y(t)} y2={y(t)} stroke={INK.grid} />
                  <text x={x(t)} y={ih + 12} textAnchor="middle" fontSize={9} fill={INK.muted}>{t}</text>
                  <text x={-4} y={y(t)} dy="0.32em" textAnchor="end" fontSize={9} fill={INK.muted}>{t}</text></g>)}
                <line x1={x(lo)} y1={y(lo)} x2={x(hi)} y2={y(hi)} stroke={INK.muted} strokeDasharray="4 4" />
                {pairs.map((p) => <circle key={p.d} cx={x(p.s)} cy={y(p.e)} r={2} fill="#2c7fb8" fillOpacity={0.35}
                  onMouseEnter={(e) => tip.show(e, [p.d, `station ${fmt(p.s, 1)} · ERA5 ${fmt(p.e, 1)} ${v.unit}`])} onMouseLeave={tip.hide} />)}
                <text x={iw / 2} y={ih + 28} textAnchor="middle" fontSize={9.5} fill={INK.muted}>station</text>
                <text transform={`translate(-30,${ih / 2}) rotate(-90)`} textAnchor="middle" fontSize={9.5} fill={INK.muted}>ERA5</text>
                <g fontFamily={FONT.mono} fontSize={9.5} fill={INK.text}>
                  <text x={4} y={10}>bias {bias !== undefined && bias >= 0 ? '+' : ''}{fmt(bias, 2)} · r {fmt(r, 2)} · n {pairs.length}</text>
                  {tot && <text x={4} y={22}>totals: station {fmt(tot[0], 0)} · ERA5 {fmt(tot[1], 0)} mm</text>}
                </g>
              </g>
            </svg>
          )
        })}
        {tip.node}
      </div>
    </ChartFrame>
  )
}
