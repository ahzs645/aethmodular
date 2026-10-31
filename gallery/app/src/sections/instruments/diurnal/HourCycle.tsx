import type { ReactNode } from 'react'
import { SweepPlot, type SweepSeries } from '@/components/SweepPlot'
import { fmt } from '@/lib/format'
import { INK } from '@/lib/theme'
import type { SeasonMeta } from '@/lib/types'
import { byHour, hourLabel, hourSpans, seasonsOf, utcLabel } from './data'

/** a set of days drawn as one line: a season of the site's calendar, a day of the week, weekdays or weekends, or every day */
export interface DayGroup { name: string; color: string }
export const ALL_YEAR = 'All year'
/** the whole year as one group (its ink read when drawn, so it follows the theme) */
export const allYear = (): DayGroup => ({ name: ALL_YEAR, color: INK.text })

/** how the page splits the days: not at all, by the site's seasons, by day of the week, or weekdays against weekends */
export const SPLITS = ['together', 'seasons', 'dow', 'weekend'] as const
export type SplitBy = (typeof SPLITS)[number]
export const SPLIT_LABEL: Record<SplitBy, string> = { together: 'All days', seasons: 'Season', dow: 'Day of week', weekend: 'Weekday vs weekend' }
/** a split: its groups in order, and each local date's group (dates are the tables' local days, so the weekday is local) */
export interface DaySplit { by: SplitBy; groups: DayGroup[]; labels: (days: string[]) => string[] }

const DOW: DayGroup[] = [
  { name: 'Monday', color: '#1f6fae' }, { name: 'Tuesday', color: '#2a9d8f' }, { name: 'Wednesday', color: '#7a9a2e' },
  { name: 'Thursday', color: '#c28a14' }, { name: 'Friday', color: '#c2562e' }, { name: 'Saturday', color: '#a43f8a' }, { name: 'Sunday', color: '#6a4fb3' },
]
const WEEK: DayGroup[] = [{ name: 'Weekday (Mon–Fri)', color: '#1f6fae' }, { name: 'Weekend (Sat–Sun)', color: '#c2562e' }]
/** 0 Monday … 6 Sunday */
const dowOf = (day: string) => (new Date(`${day}T00:00:00Z`).getUTCDay() + 6) % 7

/** the split the page asked for (null: every day together; seasons need a calendar with two or more) */
export function daySplit(by: SplitBy, seasons: readonly SeasonMeta[]): DaySplit | null {
  if (by === 'seasons') return seasons.length > 1
    ? { by, groups: seasons.map((s) => ({ name: s.name, color: s.color })), labels: (days) => seasonsOf(days, seasons) } : null
  if (by === 'dow') return { by, groups: DOW, labels: (days) => days.map((d) => DOW[dowOf(d)].name) }
  if (by === 'weekend') return { by, groups: WEEK, labels: (days) => days.map((d) => WEEK[dowOf(d) >= 5 ? 1 : 0].name) }
  return null
}
/** each day's group under a split ('' for every day when there is none) */
export const splitLabels = (days: string[], split: DaySplit | null): string[] => (split ? split.labels(days) : days.map(() => ''))

/** the groups a chart draws: the split's groups that have any day, else the whole year */
export function dayGroups(split: DaySplit | null, present: readonly string[]): DayGroup[] {
  if (!split) return [allYear()]
  const have = new Set(present)
  return split.groups.filter((g) => have.has(g.name))
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

/**
 * The hours a plot needs: those where any series has a value, and the shaded ones, an hour of margin either side.
 * A full-day record keeps 0–23; one drawn only at the photometer's hours (AERONET is daylight only) is not padded
 * out to midnight.
 */
export function hourDomain(series: readonly SweepSeries[], shade: readonly number[] = []): [number, number] {
  const hs = [...series.flatMap((s) => s.points.filter((p) => p.y != null && Number.isFinite(p.y)).map((p) => +p.x)), ...shade]
  return hs.length ? [Math.max(0, Math.min(...hs) - 1), Math.min(23, Math.max(...hs) + 1)] : [0, 23]
}

/** a quantity against the hour of day, with the photometer's hours shaded */
export function HourPlot({ series, yLabel, offset, shade = [], height = 260, width, title, include, yMarkers, children, digits = 2, hours }: {
  series: SweepSeries[]; yLabel: string; offset: number
  /** hours to shade (AERONET's inversion hours) */
  shade?: readonly number[]
  /** the hour range (several panels share one); else hourDomain of this plot's series */
  hours?: [number, number]
  height?: number; width?: number; title?: ReactNode; include?: number[]
  yMarkers?: { at: number; label?: string; color?: string }[]
  children?: Parameters<typeof SweepPlot>[0]['children']
  digits?: number
}) {
  const [h0, h1] = hours ?? hourDomain(series, shade)
  return <SweepPlot series={series} xScale="linear" xDomain={[h0, h1]} xLabel={`Hour of day (local standard time, ${utcLabel(offset)})`} yLabel={yLabel}
    xFormat={(v) => hourLabel(+v)} yFormat={(v) => fmt(v, digits)} height={height} width={width} title={title} include={include} yMarkers={yMarkers}
    minWidth={width != null ? undefined : 300}
    background={({ x, ih }) => hourSpans(shade).map(([a, b]) => {
      const sx = x as (v: number) => number, x0 = sx(Math.max(h0, a - 0.5)), x1 = sx(Math.min(h1, b + 0.5))
      return <rect key={a} x={x0} width={x1 - x0} y={0} height={ih} fill={INK.shadeWarm} fillOpacity={0.85} />
    })}>
    {children}
  </SweepPlot>
}
