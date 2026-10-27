import { useMemo, useRef, useState } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Segmented, Toggle } from '@/components/ChartFrame'
import { XAxis, YAxis } from '@/components/Axes'
import { Legend, useLegend } from '@/components/Legend'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { fmt } from '@/lib/format'
import { FONT } from '@/lib/theme'
import { useSearchParam } from '@/lib/url'
import type { MetaFile } from '@/lib/types'
import { CHANNELS, SCREEN_TEXT, channelName, instColor, logTick, type Channel, type Screen, type Series } from './data'
import { movingMean, splitRuns } from './stats'

type Avg = 'daily' | '30d'
const M = { top: 10, right: 20, bottom: 34, left: 64 }
const H = 300

/**
 * Every instrument's daily BCc on one time axis (Multi_Site_Analysis §1's combined plot, _Fixed §3),
 * one wavelength at a time; a log axis by default, since Delhi runs ten times JPL.
 */
export function SitesTimeline({ series, meta, screen, channel, setChannel }: {
  series: Series[]; meta: MetaFile; screen: Screen; channel: Channel; setChannel: (c: Channel) => void
}) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
  const [avg, setAvg] = useSearchParam<Avg>('aeost', '30d', ['daily', '30d'])
  const [log, setLog] = useState(true)
  const leg = useLegend()
  const label = (s: Series) => (s.inst.code === s.inst.siteCode ? s.inst.site : s.inst.label)
  const lines = useMemo(() => series.filter((s) => s.days.length).map((s) => {
    const raw = s.days.map((d) => d.bc[channel])
    return { s, label: label(s), color: instColor(meta, s.inst), dashed: s.inst.code !== s.inst.siteCode, vals: avg === '30d' ? movingMean(s.days, (d) => d.bc[channel]) : raw, runs: splitRuns(s.days, avg === '30d' ? 15 : 1) }
  }), [series, channel, avg, meta])
  const shown = lines.filter((l) => leg.show(l.label))
  const all = shown.flatMap((l) => l.s.days.map((d) => d.t))
  const iw = Math.max(240, width - M.left - M.right)
  const x = d3.scaleUtc().domain(all.length ? (d3.extent(all) as [Date, Date]) : [new Date(0), new Date(1)]).range([0, iw])
  const vs = shown.flatMap((l) => l.vals).filter((v): v is number => v !== null && (!log || v > 0))
  const y = log ? d3.scaleLog().domain([Math.max(1e-3, (d3.min(vs) ?? 0.1) * 0.9), (d3.max(vs) ?? 1) * 1.1]).range([H, 0]).nice()
    : d3.scaleLinear().domain([Math.min(0, d3.min(vs) ?? 0), (d3.max(vs) ?? 1) * 1.04]).range([H, 0]).nice()
  const path = (l: (typeof lines)[number], idx: number[]) => d3.line<number>().defined((i) => l.vals[i] !== null && (!log || l.vals[i]! > 0))
    .x((i) => x(new Date(l.s.days[i].t.getTime() + 864e5 / 2))).y((i) => y(l.vals[i]!))(idx) ?? ''
  return (
    <ChartFrame id="aeth-overview-sites-timeline" title={`Black carbon at every instrument — daily ${channel} BCc on one time axis`}
      subtitle={`${avg === '30d' ? 'A centred 30-day mean (at least 7 days with data) of each instrument\'s' : 'Each instrument\'s'} daily ${channelName(channel, meta)} BCc, broken where days are missing${log ? ', on a log axis' : ''}. Bins are labelled at their start (Beijing, Delhi, JPL 15:00; Addis 09:00). Dashed: Jacros, the second Addis instrument. Screen: ${SCREEN_TEXT[screen]}.`}
      provenance="Multi_Site_Analysis.ipynb §1 (combined IR time series); Multi_Site_Analysis_Fixed.ipynb §3 · gallery/data/export_aethalometer.py"
      controls={<>
        <Segmented label="Wavelength" value={channel} options={CHANNELS} onChange={setChannel} />
        <Segmented label="Show" value={avg} options={['daily', '30d'] as Avg[]} onChange={setAvg} optionLabel={(o) => (o === 'daily' ? 'Daily' : '30-day mean')} />
        <Toggle label="Log scale" checked={log} onChange={setLog} />
      </>}>
      <div ref={wrapRef} className="chart-wrap">
        {!shown.length ? <Empty>No daily bins in this span.</Empty> : (
          <svg width={width} height={H + M.top + M.bottom} fontFamily={FONT.family} onMouseLeave={tip.hide}>
            <g transform={`translate(${M.left},${M.top})`}>
              <YAxis scale={y as d3.ScaleLinear<number, number>} x={0} gridWidth={iw} tickCount={log ? 12 : 6} label={`${channel} BCc (µg/m³)`} labelX={-48} format={(v) => (log ? logTick(v) : String(v))} />
              {shown.map((l) => <g key={l.label} opacity={leg.dim(l.label)}>
                {l.runs.map((idx, i) => <path key={i} d={path(l, idx)} fill="none" stroke={l.color}
                  strokeWidth={avg === '30d' ? 2 : 1} strokeOpacity={avg === '30d' ? 1 : 0.8} strokeDasharray={l.dashed ? '5 3' : undefined} />)}
              </g>)}
              <rect width={iw} height={H} fill="transparent" onMouseMove={(e) => {
                const t = x.invert(d3.pointer(e)[0])
                tip.show(e, [d3.utcFormat('%d %b %Y')(t), ...shown.map((l) => {
                  const i = d3.bisector((d: { t: Date }) => d.t).center(l.s.days, t)
                  const d = l.s.days[i]
                  return d && Math.abs(d.t.getTime() - t.getTime()) < 2 * 864e5 ? `${l.label}: ${fmt(l.vals[i], 2)} µg/m³` : `${l.label}: —`
                })])
              }} />
              <XAxis scale={x} y={H} tickCount={8} format={d3.utcFormat('%b %Y')} />
            </g>
          </svg>
        )}
        <Legend items={lines.map((l) => ({ label: l.label, color: l.color, shape: l.dashed ? 'dashed' as const : 'line' as const }))} {...leg.props} />
        {tip.node}
      </div>
    </ChartFrame>
  )
}
