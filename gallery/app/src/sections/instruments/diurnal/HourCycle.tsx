import type { ReactNode } from 'react'
import { SweepPlot, type SweepSeries } from '@/components/SweepPlot'
import { fmt } from '@/lib/format'
import { INK } from '@/lib/theme'
import type { SeasonMeta } from '@/lib/types'
import { byHour, hourLabel, hourSpans, utcLabel } from './data'

/** a set of days drawn as one line: a season of the site's calendar, or the whole year */
export interface DayGroup { name: string; color: string }
export const ALL_YEAR = 'All year'
/** the whole year as one group (its ink read when drawn, so it follows the theme) */
export const allYear = (): DayGroup => ({ name: ALL_YEAR, color: INK.text })

/** the groups a chart draws: the site's seasons when split (those with any day), else the whole year */
export function dayGroups(split: boolean, seasons: readonly SeasonMeta[], present: readonly string[]): DayGroup[] {
  if (!split) return [allYear()]
  const have = new Set(present)
  return seasons.filter((s) => have.has(s.name)).map((s) => ({ name: s.name, color: s.color }))
}
export const inGroup = (g: DayGroup, season: string) => g.name === ALL_YEAR || season === g.name

/**
 * One line per group of a quantity's hour-of-day median, its interquartile range as a band. `values`,
 * `hours` and `seasons` are parallel columns (one entry per day-hour); `keep` drops rows (too few minutes).
 */
export function cycleSeries({ id, label, hours, values, seasons, groups, keep, digits = 2, color }: {
  id: string; label: string; hours: number[]; values: (number | null)[]; seasons: string[]; groups: DayGroup[]
  keep?: (i: number) => boolean; digits?: number; color?: string
}): SweepSeries[] {
  return groups.map((g) => {
    const stats = byHour(hours, values, (i) => inGroup(g, seasons[i]) && (keep ? keep(i) : true))
    const name = g.name === ALL_YEAR ? label : `${label} · ${g.name}`
    return {
      id: `${id}-${g.name}`, label: name, color: color ?? g.color, group: g.name, curve: 'monotone' as const,
      points: stats.map((s) => ({ x: s.hour, y: s.median, lo: s.lo, hi: s.hi,
        tip: [name, hourLabel(s.hour), s.median == null ? `${s.n} day-hours (too few)` : `median ${fmt(s.median, digits)} (IQR ${fmt(s.lo, digits)}–${fmt(s.hi, digits)})`, `${s.n} day-hours`] })),
    }
  })
}

/** a quantity against the hour of day, with the photometer's hours shaded */
export function HourPlot({ series, yLabel, offset, shade = [], height = 260, width, title, include, yMarkers, children, digits = 2 }: {
  series: SweepSeries[]; yLabel: string; offset: number
  /** hours to shade (AERONET's inversion hours) */
  shade?: readonly number[]
  height?: number; width?: number; title?: ReactNode; include?: number[]
  yMarkers?: { at: number; label?: string; color?: string }[]
  children?: Parameters<typeof SweepPlot>[0]['children']
  digits?: number
}) {
  return <SweepPlot series={series} xScale="linear" xDomain={[0, 23]} xLabel={`Hour of day (local standard time, ${utcLabel(offset)})`} yLabel={yLabel}
    xFormat={(v) => hourLabel(+v)} yFormat={(v) => fmt(v, digits)} height={height} width={width} title={title} include={include} yMarkers={yMarkers}
    minWidth={width != null ? undefined : 300}
    background={({ x, ih }) => hourSpans(shade).map(([a, b]) => {
      const sx = x as (v: number) => number, x0 = sx(Math.max(0, a - 0.5)), x1 = sx(Math.min(23, b + 0.5))
      return <rect key={a} x={x0} width={x1 - x0} y={0} height={ih} fill={INK.shadeWarm} fillOpacity={0.85} />
    })}>
    {children}
  </SweepPlot>
}
