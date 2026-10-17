import { useMemo, useRef } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty } from '@/components/ChartFrame'
import { XAxis } from '@/components/Axes'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { useChartOptions } from '@/lib/chartOptions'
import { useFitHeight } from '@/lib/chartFit'
import { fmt } from '@/lib/format'
import { INK, FONT, withUnit } from '@/lib/theme'
import type { FilterRow, MetaFile } from '@/lib/types'

const MIN_N = 3

/**
 * One row per site, ranked: the median as a dot, the interquartile range as a bar, the count
 * at the end. Where a box plot per site turns to noise past a dozen sites, this stays legible
 * at 38: every row reads left to right, the order is the answer, and a click opens the site
 * (the Dashboard's One site scope). Sites with fewer than three values are left out.
 */
export function SiteRank({ rows, meta, field, onOpenSite }: { rows: FilterRow[]; meta: MetaFile; field: string; onOpenSite?: (code: string) => void }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
  const fit = useFitHeight()
  const o = useChartOptions('siterank', [
    { key: 'order', label: 'Order', choices: ['Highest first', 'Lowest first', 'Name'], default: 'Highest first' },
    { key: 'scale', label: 'Scale', choices: ['Linear', 'Log'], default: 'Linear' },
  ])

  const sites = useMemo(() => {
    const out = [...d3.group(rows, (r) => r.site)].map(([site, rs]) => {
      const v = rs.map((r) => r[field]).filter((x): x is number => typeof x === 'number' && Number.isFinite(x)).sort(d3.ascending)
      return { site, code: rs[0].code, color: rs[0].color, n: v.length,
        q1: d3.quantileSorted(v, 0.25) ?? 0, med: d3.quantileSorted(v, 0.5) ?? 0, q3: d3.quantileSorted(v, 0.75) ?? 0 }
    }).filter((s) => s.n >= MIN_N)
    return out
  }, [rows, field])
  const order = o.values.order as string
  const sorted = [...sites].sort(order === 'Name' ? (a, b) => a.site.localeCompare(b.site)
    : order === 'Lowest first' ? (a, b) => a.med - b.med : (a, b) => b.med - a.med)

  const log = o.values.scale === 'Log' && sorted.every((s) => s.q1 > 0)
  const labelW = Math.min(180, 12 + 6.2 * Math.max(6, ...sorted.map((s) => s.site.length)))
  const margin = { top: 6, right: 54, bottom: 42, left: labelW }
  const innerW = Math.max(120, width - margin.left - margin.right)
  const rowH = fit === null ? 20 : Math.max(11, Math.min(24, (fit - margin.top - margin.bottom) / Math.max(1, sorted.length)))
  const innerH = rowH * sorted.length
  const hi = d3.max(sorted, (s) => s.q3) ?? 1
  const lo = d3.min(sorted, (s) => s.q1) ?? 0
  const x = log ? d3.scaleLog().domain([Math.max(1e-3, lo * 0.9), hi * 1.1]).range([0, innerW]).nice()
    : d3.scaleLinear().domain([Math.min(0, lo), hi]).range([0, innerW]).nice()
  const font = Math.min(12, Math.max(9, rowH * 0.62))

  return (
    <ChartFrame id="siterank" title={`Sites ranked — median and interquartile range of ${field} per site`} controls={o.controls()}>
      <div ref={wrapRef} className="chart-wrap">
        {sorted.length === 0 ? <Empty>No site has {MIN_N} or more values of {field} in this subset.</Empty> : (
          <svg width={width} height={innerH + margin.top + margin.bottom}>
            <g transform={`translate(${margin.left},${margin.top})`} fontFamily={FONT.family}>
              {x.ticks(5).map((t) => <line key={t} x1={x(t)} x2={x(t)} y1={0} y2={innerH} stroke={INK.grid} />)}
              {sorted.map((s, i) => {
                const y = i * rowH + rowH / 2
                return (
                  <g key={s.site} style={onOpenSite ? { cursor: 'pointer' } : undefined} onClick={onOpenSite ? () => onOpenSite(s.code) : undefined}
                    onMouseEnter={(e) => tip.show(e, [`${s.site} (${s.code})`, `median ${fmt(s.med, 3)} · IQR ${fmt(s.q1, 2)}–${fmt(s.q3, 2)}`, `n = ${s.n}${onOpenSite ? ' · click to open the site' : ''}`])}
                    onMouseLeave={tip.hide}>
                    <rect x={-margin.left} y={i * rowH} width={width} height={rowH} fill={i % 2 ? '#f7f8fa' : 'transparent'} />
                    <text x={-8} y={y} dy="0.35em" textAnchor="end" fontSize={font} fill={INK.text}>{s.site}</text>
                    <line x1={x(Math.max(x.domain()[0], s.q1))} x2={x(s.q3)} y1={y} y2={y} stroke={s.color} strokeWidth={Math.max(2, rowH * 0.3)} strokeOpacity={0.4} strokeLinecap="round" />
                    <circle cx={x(Math.max(x.domain()[0], s.med))} cy={y} r={Math.max(2.5, Math.min(5, rowH * 0.28))} fill={s.color} />
                    <text x={innerW + 6} y={y} dy="0.35em" fontSize={Math.min(10.5, font)} fill={INK.muted} fontFamily={FONT.mono}>{s.n}</text>
                  </g>
                )
              })}
              <XAxis scale={x as d3.ScaleLinear<number, number>} y={innerH} label={withUnit(field, meta.field_units)} tickCount={Math.max(2, Math.floor(innerW / 80))} format={(v: number) => fmt(v, v >= 10 ? 0 : 1)} />
            </g>
          </svg>
        )}
        {tip.node}
      </div>
    </ChartFrame>
  )
}
