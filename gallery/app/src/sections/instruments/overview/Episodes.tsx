import { useMemo, useRef } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Segmented } from '@/components/ChartFrame'
import { XAxis, YAxis } from '@/components/Axes'
import { FacetGrid } from '@/components/FacetGrid'
import { useTooltip } from '@/hooks/useTooltip'
import { fmt, signed } from '@/lib/format'
import { FONT, INK } from '@/lib/theme'
import { seasonOnly } from '@/lib/seasons'
import type { MetaFile } from '@/lib/types'
import { seasonsForSite } from '@/siteSeasons'
import { CHANNELS, DOW, MONTH_NAME, SCREEN_TEXT, channelName, type Channel, type Screen, type Series } from './data'
import { dayToDay, quantileSorted, summarize } from './stats'

interface Bar { key: string; label: string; value: number; color: string; tip: string[]; x0?: number; x1?: number }
const M = { top: 24, right: 12, bottom: 44, left: 52 }

/** A small bar panel, bars up or down from zero: on a category axis, or on a number line when every bar has x0–x1 (a histogram). */
function BarPanel({ bars, width, height = 220, title, yLabel, xLabel }: { bars: Bar[]; width: number; height?: number; title: string; yLabel: string; xLabel?: string }) {
  const ref = useRef<HTMLDivElement>(null)
  const tip = useTooltip(ref)
  const iw = Math.max(120, width - M.left - M.right), ih = height - M.top - M.bottom
  const numeric = bars.length > 0 && bars.every((b) => b.x0 !== undefined && b.x1 !== undefined)
  const band = d3.scaleBand<string>().domain(bars.map((b) => b.key)).range([0, iw]).padding(0.18)
  const lin = d3.scaleLinear().domain(numeric ? [bars[0].x0!, bars[bars.length - 1].x1!] : [0, 1]).range([0, iw])
  const left = (b: Bar) => (numeric ? lin(b.x0!) + 0.5 : band(b.key) ?? 0)
  const bw = (b: Bar) => (numeric ? Math.max(1, lin(b.x1!) - lin(b.x0!) - 1) : band.bandwidth())
  const vs = bars.map((b) => b.value)
  const y = d3.scaleLinear().domain([Math.min(0, d3.min(vs) ?? 0), Math.max(0, d3.max(vs) ?? 1) || 1]).range([ih, 0]).nice()
  const labelOf = new Map(bars.map((b) => [b.key, b.label]))
  return (
    <div ref={ref} style={{ position: 'relative' }}>
      <svg width={width} height={height} fontFamily={FONT.family}>
        <text x={M.left} y={14} fontSize={12} fontWeight={600} fill={INK.text}>{title}</text>
        <g transform={`translate(${M.left},${M.top})`}>
          <YAxis scale={y} x={0} gridWidth={iw} tickCount={4} label={yLabel} labelX={-38} />
          {bars.map((b) => <rect key={b.key} x={left(b)} width={bw(b)} y={y(Math.max(0, b.value))} height={Math.abs(y(b.value) - y(0))} fill={b.color} fillOpacity={0.8}
            onMouseEnter={(e) => tip.show(e, b.tip)} onMouseLeave={tip.hide} />)}
          <line x1={0} x2={iw} y1={y(0)} y2={y(0)} stroke={INK.axis} />
          {numeric ? <XAxis scale={lin} y={ih} tickCount={5} label={xLabel} labelDy={34} /> : <XAxis scale={band} y={ih} format={(k: string) => labelOf.get(k) ?? k} />}
        </g>
      </svg>
      {tip.node}
    </div>
  )
}

/** Days per value bin, the outer bins collecting everything beyond ±3 SD (as the notebook clipped). */
function histogram(values: number[], bins = 24): Bar[] {
  const s = summarize(values)
  if (!s || s.sd === null || s.sd === 0) return []
  const lim = 3 * s.sd, w = (2 * lim) / bins
  const counts = new Array(bins).fill(0)
  for (const v of values) counts[Math.min(bins - 1, Math.max(0, Math.floor((v + lim) / w)))]++
  return counts.map((n, i) => {
    const a = -lim + i * w
    return { key: String(i), label: fmt(a + w / 2, 1), value: n, x0: a, x1: a + w, color: a + w / 2 >= 0 ? INK.negative : INK.positive,
      tip: [`${signed(a, 2)} to ${signed(a + w, 2)} µg/m³ a day${i === 0 || i === bins - 1 ? ' (and beyond)' : ''}`, `${n} days`] }
  })
}

/**
 * High-BC days and the day-to-day change at one site (addis_02_daily Tasks 4 and 5): when the days
 * above the 95th percentile fall, by season and weekday, and how much BCc moves from one day to the next.
 */
export function Episodes({ s, meta, screen, channel, setChannel }: {
  s: Series; meta: MetaFile; screen: Screen; channel: Channel; setChannel: (c: Channel) => void
}) {
  const panels = useMemo(() => {
    const get = (d: (typeof s.days)[number]) => d.bc[channel]
    const vals = s.days.map(get).filter((v): v is number => v !== null).sort((a, b) => a - b)
    if (vals.length < 20) return null
    const p95 = quantileSorted(vals, 0.95)
    const high = s.days.filter((d) => (get(d) ?? -Infinity) > p95)
    const seasons = seasonsForSite(meta, s.inst.site)
    const bySeason: Bar[] = seasons.map((se) => {
      const n = high.filter((d) => d.season?.name === se.name).length, all = s.days.filter((d) => d.season?.name === se.name && get(d) !== null).length
      return { key: se.name, label: seasonOnly(se.name).replace(/ \(.*\)$/, ''), value: n, color: se.color, tip: [seasonOnly(se.name), `${n} of the ${high.length} high days`, `${fmt(all ? (100 * n) / all : null, 1)} % of the season's ${all} days`] }
    })
    const byDow: Bar[] = DOW.map((name, i) => {
      const n = high.filter((d) => d.dow === i).length
      return { key: name, label: name, value: n, color: i >= 5 ? INK.negative : INK.accent, tip: [name, `${n} of the ${high.length} high days`] }
    })
    const changes = dayToDay(s.days, get)
    const byMonth: Bar[] = MONTH_NAME.map((name, i) => {
      const st = summarize(changes.filter((c) => c.day.month === i + 1).map((c) => c.dv))
      return { key: name, label: name.slice(0, 1), value: st?.mean ?? 0, color: (st?.mean ?? 0) >= 0 ? INK.negative : INK.positive,
        tip: [name, st ? `mean change ${signed(st.mean, 3)} µg/m³ a day` : 'no consecutive days', st ? `${st.n} day pairs, SD ${fmt(st.sd, 2)}` : ''] }
    })
    return { p95, nHigh: high.length, n: vals.length, bySeason, byDow, byMonth, hist: histogram(changes.map((c) => c.dv)), nPairs: changes.length }
  }, [s, meta, channel])
  return (
    <ChartFrame id="aeth-overview-episodes" title={`${s.inst.site} · high days and day-to-day change — ${channel} BCc`}
      subtitle={panels ? `Top: the ${panels.nHigh} days whose daily ${channelName(channel, meta)} BCc is above its 95th percentile (${fmt(panels.p95, 2)} µg/m³ over ${panels.n} days in the span), counted by season and by day of the week. Bottom: the change from one day to the next (only between consecutive bins, ${panels.nPairs} pairs), its mean by month and its distribution (bins beyond ±3 SD folded into the end bars, as the notebook clipped). The daily files of Beijing, Delhi and JPL hold only days near a filter sample, so their pairs come from short runs. Screen: ${SCREEN_TEXT[screen]}.` : undefined}
      provenance="addis_02_temporal_patterns_daily.ipynb Tasks 4 (extreme days) and 5 (day-to-day rate of change) for every site · gallery/data/export_aethalometer.py"
      controls={<Segmented label="Wavelength" value={channel} options={CHANNELS} onChange={setChannel} />}>
      <div className="chart-wrap">
        {!panels ? <Empty>Too few days with {channel} BCc in this span.</Empty> : (
          <FacetGrid items={[
            { title: 'High days by season', bars: panels.bySeason, y: 'Days', x: undefined },
            { title: 'High days by weekday', bars: panels.byDow, y: 'Days' },
            { title: 'Mean day-to-day change by month', bars: panels.byMonth, y: 'µg/m³ per day' },
            { title: 'Day-to-day change', bars: panels.hist, y: 'Day pairs', x: 'Change (µg/m³ per day)' },
          ]} minCell={250} maxCols={2} render={(p, w) => <BarPanel bars={p.bars} width={w} title={p.title} yLabel={p.y} xLabel={p.x} />} />
        )}
      </div>
    </ChartFrame>
  )
}
