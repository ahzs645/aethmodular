import { useId, useRef } from 'react'
import * as d3 from 'd3'
import { XAxis, YAxis } from '@/components/Axes'
import { SeasonBands, monthSeasonBands } from '@/components/SeasonBands'
import { useTooltip } from '@/hooks/useTooltip'
import { fmt } from '@/lib/format'
import { FONT, INK } from '@/lib/theme'
import { seasonOnly } from '@/lib/seasons'
import type { SeasonMeta } from '@/lib/types'
import { MONTH_LETTER, MONTH_NAME } from './data'
import type { Summary } from './stats'

export type MonthStat = 'mean' | 'median'
export const MONTH_STAT_LABEL: Record<MonthStat, string> = { mean: 'Mean ± SD', median: 'Median and IQR' }
export interface MonthSeries { key: string; label: string; color: string; months: (Summary | null)[]; opacity?: number }

const M = { top: 22, right: 14, bottom: 40, left: 58 }

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
  const ref = useRef<HTMLDivElement>(null)
  const tip = useTooltip(ref)
  const iw = Math.max(120, width - M.left - M.right), ih = height - M.top - M.bottom
  const x = d3.scaleBand<string>().domain(MONTH_NAME).range([0, iw]).padding(0.1)
  const cx = (m: number) => (x(MONTH_NAME[m - 1]) ?? 0) + x.bandwidth() / 2
  const mid = (s: Summary) => (stat === 'mean' ? s.mean : s.median)
  const lo = (s: Summary) => (stat === 'mean' ? s.mean - (s.sd ?? 0) : s.q25)
  const hi = (s: Summary) => (stat === 'mean' ? s.mean + (s.sd ?? 0) : s.q75)
  // the lines set the scale; a wild band (a noisy channel's SD) is cut at 1.5× the highest line and clipped
  const mids = series.flatMap((se) => se.months.flatMap((s) => (s ? [mid(s)] : [])))
  const vals = series.flatMap((se) => se.months.flatMap((s) => (s ? [lo(s), hi(s)] : [])))
  const top = Math.max(d3.max(mids) ?? 1, Math.min(d3.max(vals) ?? 1, (d3.max(mids) ?? 1) * 1.5))
  const bottom = Math.min(0, d3.min(mids) ?? 0, Math.max(d3.min(vals) ?? 0, -0.2 * top))
  const y = d3.scaleLinear().domain([bottom, top * 1.04]).range([ih, 0]).nice()
  const clip = `mp${useId().replace(/:/g, '')}`
  const seasonOf = (m: number) => seasons.find((s) => s.months.includes(m))
  const bands = monthSeasonBands(cx, seasonOf, x.step(), iw)
  const idx = d3.range(1, 13)
  const line = (se: MonthSeries) => d3.line<number>().defined((m) => !!se.months[m - 1]).x(cx).y((m) => y(mid(se.months[m - 1]!)))(idx) ?? ''
  const area = (se: MonthSeries) => d3.area<number>().defined((m) => !!se.months[m - 1]).x(cx).y0((m) => y(lo(se.months[m - 1]!))).y1((m) => y(hi(se.months[m - 1]!)))(idx) ?? ''
  const compact = width < 360
  return (
    <div ref={ref} style={{ position: 'relative' }}>
      <svg width={width} height={height} fontFamily={FONT.family}>
        {title && <text x={M.left} y={14} fontSize={12} fontWeight={600} fill={INK.text}>{title}</text>}
        <g transform={`translate(${M.left},${M.top})`}>
          <clipPath id={clip}><rect width={iw} height={ih} /></clipPath>
          <SeasonBands bands={bands} height={ih} />
          <YAxis scale={y} x={0} gridWidth={iw} tickCount={4} label={unit} labelX={-42} />
          {series.map((se) => <g key={se.key} opacity={se.opacity ?? 1}>
            <path d={area(se)} fill={se.color} fillOpacity={0.14} clipPath={`url(#${clip})`} />
            <path d={line(se)} fill="none" stroke={se.color} strokeWidth={2} />
            {idx.map((m) => se.months[m - 1] && <circle key={m} cx={cx(m)} cy={y(mid(se.months[m - 1]!))} r={3} fill={se.color} />)}
          </g>)}
          {idx.map((m) => {
            const se = seasonOf(m)
            return <g key={m}>
              <rect x={cx(m) - x.step() / 2} width={x.step()} y={ih + 3} height={6} fill={se?.color ?? INK.neutral} fillOpacity={0.7} />
              <rect x={cx(m) - x.step() / 2} width={x.step()} height={ih + 10} fill="transparent"
                onMouseEnter={(e) => tip.show(e, [`${MONTH_NAME[m - 1]}${se ? ` · ${seasonOnly(se.name)}` : ''}`,
                  ...series.map((s) => { const v = s.months[m - 1]; return v ? `${s.label}: ${stat === 'mean' ? `${fmt(v.mean, digits)} ± ${fmt(v.sd, digits)}` : `${fmt(v.median, digits)} [${fmt(v.q25, digits)}–${fmt(v.q75, digits)}]`} (n=${v.n})` : `${s.label}: no days` })])}
                onMouseLeave={tip.hide} />
            </g>
          })}
          <XAxis scale={x} y={ih + 10} format={(m: string) => (compact ? MONTH_LETTER[MONTH_NAME.indexOf(m)] : m)} />
        </g>
      </svg>
      {tip.node}
    </div>
  )
}
