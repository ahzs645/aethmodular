import { useMemo, useRef } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Segmented } from '@/components/ChartFrame'
import { ColorLegend } from '@/components/ColorLegend'
import { FacetGrid } from '@/components/FacetGrid'
import { HeatCell } from '@/components/HeatCell'
import { useTooltip } from '@/hooks/useTooltip'
import { fmt } from '@/lib/format'
import { FONT, INK, RAMP_DIVERGING, RAMP_SEQUENTIAL } from '@/lib/theme'
import { seasonOnly } from '@/lib/seasons'
import { useSearchParam } from '@/lib/url'
import type { MetaFile } from '@/lib/types'
import { seasonsForSite } from '@/siteSeasons'
import { DOW } from './data'
import { weekHourGrid } from './hourly'
import type { HourlySeries } from './useHourly'

type Scale = 'rel' | 'abs'
const M = { top: 22, right: 6, bottom: 22, left: 34 }

/** one season's week: weekday rows (Mon first), hour columns */
function WeekGrid({ title, color, grid, width, fill, tip }: {
  title: string; color?: string; grid: ReturnType<typeof weekHourGrid>; width: number
  fill: (v: number | null, overall: number | null) => string | null
  tip: ReturnType<typeof useTooltip>
}) {
  const iw = width - M.left - M.right, cw = iw / 24, ch = 16
  return (
    <svg width={width} height={M.top + 7 * ch + M.bottom} fontFamily={FONT.family}>
      <text x={M.left} y={13} fontSize={12} fontWeight={600} fill={color ?? INK.text}>{title}</text>
      <g transform={`translate(${M.left},${M.top})`}>
        {DOW.map((d, i) => <g key={d}>
          <text x={-4} y={i * ch + ch / 2} dy="0.32em" textAnchor="end" fontSize={10} fill={i >= 5 ? INK.negative : INK.muted}>{d}</text>
          {grid.mean[i].map((v, h) => <HeatCell key={h} x={h * cw} y={i * ch} width={cw - 0.5} height={ch - 0.5} fill={fill(v, grid.overall)}
            onEnter={(e) => tip.show(e, [`${title} · ${d} ${String(h).padStart(2, '0')}:00`, `IR BCc ${fmt(v, 2)} µg/m³ (${grid.n[i][h]} hours)`,
              `${fmt(v !== null && grid.overall ? v / grid.overall : null, 2)} × the season's mean hour (${fmt(grid.overall, 2)})`])} onLeave={tip.hide} />)}
        </g>)}
        {[0, 6, 12, 18].map((h) => <text key={h} x={h * cw} y={7 * ch + 13} fontSize={10} fill={INK.muted}>{String(h).padStart(2, '0')}</text>)}
      </g>
    </svg>
  )
}

/**
 * addis_02_temporal_patterns Task 2's weekly heatmap from the hourly record: mean IR BCc for each
 * weekday and local hour, one panel per season. The Diurnal page has the hour-of-day cycles; this is
 * how the week bends them (the weekend morning, a Friday evening).
 */
export function WeekHour({ hs, meta }: { hs: HourlySeries; meta: MetaFile }) {
  const ref = useRef<HTMLDivElement>(null)
  const tip = useTooltip(ref)
  const [scale, setScale] = useSearchParam<Scale>('aeowh', 'rel', ['rel', 'abs'])
  const panels = useMemo(() => [
    ...seasonsForSite(meta, hs.site).map((se) => ({ key: se.name, title: seasonOnly(se.name), color: se.color as string | undefined, grid: weekHourGrid(hs.hours.filter((h) => h.season?.name === se.name)) })),
    { key: 'all', title: 'All hours', color: undefined, grid: weekHourGrid(hs.hours) },
  ].filter((p) => p.grid.overall !== null), [hs, meta])
  const max = d3.max(panels.flatMap((p) => p.grid.mean.flat()), (v) => v ?? undefined) ?? 1
  const rel = d3.scaleDiverging<string>(d3.interpolateRgbBasis(RAMP_DIVERGING)).domain([-1, 0, 1]).clamp(true)
  const abs = d3.scaleSequential(d3.interpolateRgbBasis(RAMP_SEQUENTIAL)).domain([0, max])
  const fill = (v: number | null, overall: number | null) => (v === null ? null : scale === 'abs' ? abs(v) : overall ? rel(Math.log2(v / overall)) : null)
  const res = hs.info.resolution === 'minute' ? `hourly means of every minute (${hs.info.minute_file?.split('/').pop()})` : `the hourly means file (${hs.info.hourly_file.split('/').pop()})`
  return (
    <ChartFrame id="aeth-overview-weekhour" title={`${hs.site} · the week by the hour — IR BCc for each weekday and local hour, by season`}
      subtitle={`Mean IR BCc over the hours of each weekday and local standard hour, from ${res}, ${hs.hours.length.toLocaleString()} valid hours in the span. ${scale === 'rel' ? "Coloured against the panel's mean hour (log scale, ½× to 2×), so each season's own week shows" : 'Coloured on one µg/m³ scale for every panel'}. Weekends in red. The Diurnal page draws the plain hour-of-day cycles.`}
      provenance="addis_02_temporal_patterns.ipynb Task 2 (weekly heatmap by Ethiopian season), for every site with an hourly record · gallery/data/export_aeth_overview.py"
      controls={<Segmented label="Colour" value={scale} options={['rel', 'abs'] as Scale[]} onChange={setScale} optionLabel={(o) => (o === 'rel' ? 'Against the season' : 'µg/m³')} />}>
      <div ref={ref} className="chart-wrap">
        {!panels.length ? <Empty>No hourly data in this span.</Empty>
          : <FacetGrid items={panels} minCell={300} maxCols={2} render={(p, w) => <WeekGrid title={p.title} color={p.color} grid={p.grid} width={w} fill={fill} tip={tip} />} />}
        <div className="legend">
          {scale === 'rel' ? <ColorLegend scale={rel} width={160} label="× the season's mean hour" format={(v) => `${fmt(2 ** v, v === 0 ? 0 : 1)}×`} />
            : <ColorLegend scale={abs} width={160} label="IR BCc (µg/m³)" format={(v) => fmt(v, 0)} />}
        </div>
        {tip.node}
      </div>
    </ChartFrame>
  )
}
