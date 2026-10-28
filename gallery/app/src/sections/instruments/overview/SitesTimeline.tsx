import { useMemo, useState } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Segmented, Toggle } from '@/components/ChartFrame'
import { Legend, useLegend } from '@/components/Legend'
import { SweepPlot, type SweepPoint, type SweepSeries } from '@/components/SweepPlot'
import { fmt } from '@/lib/format'
import { useSearchParam } from '@/lib/url'
import type { MetaFile } from '@/lib/types'
import { CHANNELS, SCREEN_TEXT, channelName, instColor, logTick, type Channel, type Screen, type Series } from './data'
import { movingMean, splitRuns } from './stats'

type Avg = 'daily' | '30d'
const H = 300
const DAY = 864e5

/**
 * Every instrument's daily BCc on one time axis (Multi_Site_Analysis §1's combined plot, _Fixed §3),
 * one wavelength at a time; a log axis by default, since Delhi runs ten times JPL.
 */
export function SitesTimeline({ series, meta, screen, channel, setChannel }: {
  series: Series[]; meta: MetaFile; screen: Screen; channel: Channel; setChannel: (c: Channel) => void
}) {
  const [avg, setAvg] = useSearchParam<Avg>('aeost', '30d', ['daily', '30d'])
  const [log, setLog] = useState(true)
  const leg = useLegend()
  const label = (s: Series) => (s.inst.code === s.inst.siteCode ? s.inst.site : s.inst.label)
  const lines = useMemo(() => series.filter((s) => s.days.length).map((s) => {
    const raw = s.days.map((d) => d.bc[channel])
    return { s, label: label(s), color: instColor(meta, s.inst), dashed: s.inst.code !== s.inst.siteCode, vals: avg === '30d' ? movingMean(s.days, (d) => d.bc[channel]) : raw, runs: splitRuns(s.days, avg === '30d' ? 15 : 1) }
  }), [series, channel, avg, meta])
  const shown = lines.filter((l) => leg.show(l.label))
  const sweep = useMemo<SweepSeries[]>(() => shown.map((l) => {
    // a bin at its centre; a missing value point between runs (days absent from the file) breaks the line there
    const at = (i: number) => new Date(l.s.days[i].t.getTime() + DAY / 2)
    const points: SweepPoint[] = l.runs.flatMap((idx, r) => [
      ...(r ? [{ x: new Date((l.s.days[l.runs[r - 1][l.runs[r - 1].length - 1]].t.getTime() + l.s.days[idx[0]].t.getTime()) / 2 + DAY / 2), y: null }] : []),
      ...idx.map((i) => ({ x: at(i), y: l.vals[i], tip: [l.label, d3.utcFormat('%d %b %Y')(l.s.days[i].t), `${channel} BCc${avg === '30d' ? ', 30-day mean' : ''}: ${fmt(l.vals[i], 2)} µg/m³`] })),
    ])
    return { id: l.label, label: l.label, group: l.label, color: l.color, points, gaps: true, marks: false, interval: 'none' as const, width: avg === '30d' ? 2 : 1, dash: l.dashed ? '5 3' : undefined }
  }), [shown, channel, avg]) // eslint-disable-line react-hooks/exhaustive-deps
  const vs = shown.flatMap((l) => l.vals).filter((v): v is number => v !== null && (!log || v > 0))
  // log: the data's own range, not rounded out to whole decades (0.38–20 drawn as 0.1–100); linear: from 0, niced
  const yDomain: [number, number] = log ? [Math.max(1e-3, (d3.min(vs) ?? 0.1) * 0.9), (d3.max(vs) ?? 1) * 1.1]
    : d3.scaleLinear().domain([Math.min(0, d3.min(vs) ?? 0), (d3.max(vs) ?? 1) * 1.04]).nice().domain() as [number, number]
  return (
    <ChartFrame id="aeth-overview-sites-timeline" title={`Black carbon at every instrument — daily ${channel} BCc on one time axis`}
      subtitle={`${avg === '30d' ? 'A centred 30-day mean (at least 7 days with data) of each instrument\'s' : 'Each instrument\'s'} daily ${channelName(channel, meta)} BCc, broken where days are missing${log ? ', on a log axis' : ''}. Bins are labelled at their start (Beijing, Delhi, JPL 15:00; Addis 09:00). Dashed: a site's longer daily file of the same instrument (Addis: the Jacros file). Screen: ${SCREEN_TEXT[screen]}.`}
      provenance="Multi_Site_Analysis.ipynb §1 (combined IR time series); Multi_Site_Analysis_Fixed.ipynb §3 · gallery/data/export_aethalometer.py"
      controls={<>
        <Segmented label="Wavelength" value={channel} options={CHANNELS} onChange={setChannel} />
        <Segmented label="Show" value={avg} options={['daily', '30d'] as Avg[]} onChange={setAvg} optionLabel={(o) => (o === 'daily' ? 'Daily' : '30-day mean')} />
        <Toggle label="Log scale" checked={log} onChange={setLog} />
      </>}>
      <div className="chart-wrap">
        {!shown.length ? <Empty>No daily bins in this span.</Empty> : (
          <SweepPlot series={sweep} xScale="time" yScale={log ? 'log' : 'linear'} yDomain={yDomain} yLabel={`${channel} BCc (µg/m³)`}
            yFormat={(v) => (log ? logTick(v) : String(+v.toPrecision(3)))} height={H + 58} dim={leg.dim} />
        )}
        <Legend items={lines.map((l) => ({ label: l.label, color: l.color, shape: l.dashed ? 'dashed' as const : 'line' as const }))} {...leg.props} />
      </div>
    </ChartFrame>
  )
}
