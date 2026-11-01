import { useMemo } from 'react'
import { useColourScheme } from '@/lib/colourSchemes'
import * as d3 from 'd3'
import { ChartFrame, Empty, Segmented } from '@/components/ChartFrame'
import { ColorLegend } from '@/components/ColorLegend'
import { FacetGrid } from '@/components/FacetGrid'
import { HeatMatrix, type MatrixItem } from '@/components/HeatMatrix'
import { fmt } from '@/lib/format'
import { INK } from '@/lib/theme'
import { seasonOnly } from '@/lib/seasons'
import { useSearchParam } from '@/lib/url'
import type { MetaFile } from '@/lib/types'
import { seasonsForSite } from '@/siteSeasons'
import { DOW } from './data'
import { weekHourGrid } from './hourly'
import type { HourlySeries } from './useHourly'

type Scale = 'rel' | 'abs'
const HOUR_COLUMNS: MatrixItem[] = Array.from({ length: 24 }, (_, h) => ({ key: String(h), label: h % 6 === 0 ? String(h).padStart(2, '0') : '' }))
const DAY_ROWS: MatrixItem[] = DOW.map((d, i) => ({ key: d, color: i >= 5 ? INK.negative : INK.muted }))

/** one season's week: weekday rows (Mon first), hour columns (components/HeatMatrix) */
function WeekGrid({ title, color, grid, width, fill }: {
  title: string; color?: string; grid: ReturnType<typeof weekHourGrid>; width: number
  fill: (v: number | null, overall: number | null) => string | null
}) {
  return <HeatMatrix width={width} title={title} titleColor={color ?? INK.text} rows={DAY_ROWS} columns={HOUR_COLUMNS} rowLabelSize={10}
    colLabels="horizontal" colLabelPosition="bottom" colLabelAlign="start" colLabelSize={10} colLabelColor={INK.muted}
    cellWidth={[1, 1000]} cellHeight={16} gap={0.5} fitRows={false} cellLabels="never" legend={null}
    cell={(i, h) => ({ fill: fill(grid.mean[i][h], grid.overall) })}
    tip={(i, h) => {
      const v = grid.mean[i][h]
      return [`${title} · ${DOW[i]} ${String(h).padStart(2, '0')}:00`, `IR BCc ${fmt(v, 2)} µg/m³ (${grid.n[i][h]} hours)`,
        `${fmt(v !== null && grid.overall ? v / grid.overall : null, 2)} × the season's mean hour (${fmt(grid.overall, 2)})`]
    }} />
}

/**
 * addis_02_temporal_patterns Task 2's weekly heatmap from the hourly record: mean IR BCc for each
 * weekday and local hour, one panel per season. The Diurnal page has the hour-of-day cycles; this is
 * how the week bends them (the weekend morning, a Friday evening).
 */
export function WeekHour({ hs, meta }: { hs: HourlySeries; meta: MetaFile }) {
  const [scale, setScale] = useSearchParam<Scale>('aeowh', 'rel', ['rel', 'abs'])
  const panels = useMemo(() => [
    ...seasonsForSite(meta, hs.site).map((se) => ({ key: se.name, title: seasonOnly(se.name), color: se.color as string | undefined, grid: weekHourGrid(hs.hours.filter((h) => h.season?.name === se.name)) })),
    { key: 'all', title: 'All hours', color: undefined, grid: weekHourGrid(hs.hours) },
  ].filter((p) => p.grid.overall !== null), [hs, meta])
  const max = d3.max(panels.flatMap((p) => p.grid.mean.flat()), (v) => v ?? undefined) ?? 1
  const relColours = useColourScheme('aeth-overview-weekhour.rel', 'diverging')
  const absColours = useColourScheme('aeth-overview-weekhour.abs', 'sequential')
  const rel = d3.scaleDiverging<string>(d3.interpolateRgbBasis(relColours.stops)).domain([-1, 0, 1]).clamp(true)
  const abs = d3.scaleSequential(d3.interpolateRgbBasis(absColours.stops)).domain([0, max])
  const fill = (v: number | null, overall: number | null) => (v === null ? null : scale === 'abs' ? abs(v) : overall ? rel(Math.log2(v / overall)) : null)
  const res = hs.info.resolution === 'minute' ? `hourly means of every minute (${hs.info.minute_file?.split('/').pop()})` : `the hourly means file (${hs.info.hourly_file.split('/').pop()})`
  return (
    <ChartFrame id="aeth-overview-weekhour" title={`${hs.site} · the week by the hour — IR BCc for each weekday and local hour, by season`}
      subtitle={`Mean IR BCc over the hours of each weekday and local standard hour, from ${res}, ${hs.hours.length.toLocaleString()} valid hours in the span. ${scale === 'rel' ? "Coloured against the panel's mean hour (log scale, ½× to 2×), so each season's own week shows" : 'Coloured on one µg/m³ scale for every panel'}. Weekends in red. The Diurnal page draws the plain hour-of-day cycles.`}
      provenance="addis_02_temporal_patterns.ipynb Task 2 (weekly heatmap by Ethiopian season), for every site with an hourly record · gallery/data/export_aeth_overview.py"
      controls={<Segmented label="Colour" value={scale} options={['rel', 'abs'] as Scale[]} onChange={setScale} optionLabel={(o) => (o === 'rel' ? 'Against the season' : 'µg/m³')} />}>
      <div className="chart-wrap">
        {!panels.length ? <Empty>No hourly data in this span.</Empty>
          : <FacetGrid items={panels} minCell={300} maxCols={2} render={(p, w) => <WeekGrid title={p.title} color={p.color} grid={p.grid} width={w} fill={fill} />} />}
        <div className="legend">
          {scale === 'rel' ? <ColorLegend scale={rel} scheme={relColours.scheme} width={160} label="× the season's mean hour" format={(v) => `${fmt(2 ** v, v === 0 ? 0 : 1)}×`} />
            : <ColorLegend scale={abs} scheme={absColours.scheme} width={160} label="IR BCc (µg/m³)" format={(v) => fmt(v, 0)} />}
        </div>
      </div>
    </ChartFrame>
  )
}
