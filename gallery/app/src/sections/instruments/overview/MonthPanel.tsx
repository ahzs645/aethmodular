import * as d3 from 'd3'
import { SeasonBands, monthSeasonBands } from '@/components/SeasonBands'
import { SweepPlot, type SweepSeries } from '@/components/SweepPlot'
import { fmt } from '@/lib/format'
import { seasonOnly } from '@/lib/seasons'
import type { SeasonMeta } from '@/lib/types'
import { MONTH_LETTER, MONTH_NAME } from './data'
import type { Summary } from './stats'

export type MonthStat = 'mean' | 'median'
export const MONTH_STAT_LABEL: Record<MonthStat, string> = { mean: 'Mean ± SD', median: 'Median and IQR' }
export interface MonthSeries { key: string; label: string; color: string; months: (Summary | null)[]; opacity?: number }

/**
 * One month-of-year panel: a line per series through each month's mean (band ± 1 SD, as the
 * addis_02 notebooks drew it) or median (band: interquartile range), the site's season calendar
 * shaded behind and as a strip under the months. Months with no days are left open. The lines set
 * the y range; a band far wider (Addis UV's SD) is clipped at 1.5× the highest line, its values in the tooltip.
 */
export function MonthPanel({ series, seasons, stat, width, height = 240, title, unit, digits = 2 }: {
  series: MonthSeries[]; seasons: readonly SeasonMeta[]; stat: MonthStat; width: number; height?: number
  /** a heading inside the panel (small multiples) */
  title?: string
  unit: string
  digits?: number
}) {
  const mid = (s: Summary) => (stat === 'mean' ? s.mean : s.median)
  const lo = (s: Summary) => (stat === 'mean' ? s.mean - (s.sd ?? 0) : s.q25)
  const hi = (s: Summary) => (stat === 'mean' ? s.mean + (s.sd ?? 0) : s.q75)
  // the lines set the scale; a wild band (a noisy channel's SD) is cut at 1.5× the highest line, its edge clamped there
  const mids = series.flatMap((se) => se.months.flatMap((s) => (s ? [mid(s)] : [])))
  const vals = series.flatMap((se) => se.months.flatMap((s) => (s ? [lo(s), hi(s)] : [])))
  const top = Math.max(d3.max(mids) ?? 1, Math.min(d3.max(vals) ?? 1, (d3.max(mids) ?? 1) * 1.5))
  const bottom = Math.min(0, d3.min(mids) ?? 0, Math.max(d3.min(vals) ?? 0, -0.2 * top))
  const yDomain = d3.scaleLinear().domain([bottom, top * 1.04]).nice().domain() as [number, number]
  const seasonOf = (m: number) => seasons.find((s) => s.months.includes(m))
  const opacity = new Map(series.map((se) => [se.key, se.opacity ?? 1]))
  // every month a point (an open month a missing value, so the line breaks there and the axis keeps all twelve)
  const sweep: SweepSeries[] = series.map((se) => ({
    id: se.key, label: se.label, group: se.key, color: se.color, width: 2, marks: true, gaps: true, fill: 0.14,
    points: MONTH_NAME.map((name, i) => {
      const v = se.months[i], season = seasonOf(i + 1)
      return { x: name, y: v ? mid(v) : null, lo: v ? lo(v) : null, hi: v ? hi(v) : null,
        tip: [`${name}${season ? ` · ${seasonOnly(season.name)}` : ''}`, v ? `${se.label}: ${stat === 'mean' ? `${fmt(v.mean, digits)} ± ${fmt(v.sd, digits)}` : `${fmt(v.median, digits)} [${fmt(v.q25, digits)}–${fmt(v.q75, digits)}]`} (n=${v.n})` : `${se.label}: no days`] }
    }),
  }))
  // the season calendar shaded behind the months and as a strip above them
  const shade = (x: d3.ScalePoint<string>, iw: number) => monthSeasonBands((m) => x(MONTH_NAME[m - 1]) ?? 0, seasonOf, x.step(), iw)
  return <SweepPlot series={sweep} xScale="point" yDomain={yDomain} yLabel={unit} yFormat={(v) => String(+v.toPrecision(3))} title={title}
    xFormat={(m: string) => (width < 360 ? MONTH_LETTER[MONTH_NAME.indexOf(m)] : m)} width={width} height={title ? height - 18 : height} dim={(k) => opacity.get(k) ?? 1}
    background={({ x, ih, iw }) => { const b = shade(x as d3.ScalePoint<string>, iw); return <><SeasonBands bands={b} height={ih} /><SeasonBands bands={b} height={ih} strip /></> }} />
}
