import { useMemo, useRef, useState } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Note } from '@/components/ChartFrame'
import { XAxis, YAxis } from '@/components/Axes'
import { IntervalStrip, type IntervalBlock } from '@/components/IntervalStrip'
import { Key } from '@/components/Key'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { fmt } from '@/lib/format'
import { FONT, INK } from '@/lib/theme'
import { seasonOnly } from '@/lib/seasons'
import type { MetaFile } from '@/lib/types'
import { seasonsForSite } from '@/siteSeasons'
import { DOW, channelColor, loadTraces, siteColor, useSiteFiles } from './data'
import { episodeRate, type Episode } from './hourly'
import type { HourlySeries } from './useHourly'

const M = { top: 10, right: 20, bottom: 34, left: 60 }
const H = 3600e3
const hhmm = d3.utcFormat('%d %b %Y %H:%M')

/**
 * The high-BC episodes at one site (addis_02_temporal_patterns Task 4, from minutes rather than
 * single extreme minutes): runs of at least two hours above the record's 95th-percentile hour. Top:
 * every episode in the span by start and peak (size: length; filled: a minute trace exists). Below:
 * the picked episode minute by minute where the minute file covers it, its hourly means, and the
 * 24 h means of the local days around it, which is all a daily filter or the daily file would keep.
 */
export function HighEpisodes({ hs, meta }: { hs: HourlySeries; meta: MetaFile }) {
  const ref = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(ref)
  const tip = useTooltip(ref)
  const traces = useSiteFiles([hs.code], loadTraces)[hs.code]
  const traced = useMemo(() => new Map((traces?.traces ?? []).map((t) => [t.episode, t])), [traces])
  const eps = hs.episodes
  const [pick, setPick] = useState<number | null>(null)
  const fallback = useMemo(() => [...eps].sort((a, b) => Number(traced.has(b.i)) - Number(traced.has(a.i)) || b.peak - a.peak)[0], [eps, traced])
  const sel = eps.find((e) => e.i === pick) ?? fallback
  const seasons = seasonsForSite(meta, hs.site)

  const iw = Math.max(240, width - M.left - M.right)
  const topH = 150, trH = 230
  const ext = d3.extent(hs.hours, (h) => h.t) as [Date, Date] | [undefined, undefined]
  const x = d3.scaleUtc().domain(ext[0] ? ext as [Date, Date] : [new Date(0), new Date(1)]).range([0, iw])
  const yTop = d3.scaleLinear().domain([hs.threshold, (d3.max(eps, (e) => e.peak) ?? hs.threshold + 1) * 1.05]).range([topH, 0]).nice()
  const color = (e: Episode) => e.season?.color ?? INK.muted

  // the picked episode's window
  const view = useMemo(() => {
    if (!sel) return null
    const tr = traced.get(sel.i)
    const a = tr ? new Date(`${tr.t0.replace(' ', 'T')}:00Z`) : new Date(sel.start.getTime() - 6 * H)
    const b = tr ? new Date(a.getTime() + tr.IR.length * 60e3) : new Date(sel.start.getTime() + (sel.hours + 6) * H)
    const hours = hs.hours.filter((h) => h.t >= a && h.t < b)
    const dayMean = new Map(hs.days.map((d) => [d.date, d.mean]))
    const dates = [...new Set(d3.utcDays(d3.utcDay.floor(a), b).map((d) => d.toISOString().slice(0, 10)))]
    const minutes = tr ? tr.IR.map((v, k) => ({ t: new Date(a.getTime() + k * 60e3), v: v === null ? null : v / 1000 })) : []
    return { a, b, tr, hours, minutes, days: dates.map((d) => ({ date: d, mean: dayMean.get(d) ?? null })) }
  }, [sel, traced, hs])
  const xs = d3.scaleUtc().domain(view ? [view.a, view.b] : [new Date(0), new Date(1)]).range([0, iw])
  const minVals = view?.minutes.map((m) => m.v).filter((v): v is number => v !== null).sort(d3.ascending) ?? []
  const yMax = Math.max(d3.max(view?.hours ?? [], (h) => h.v) ?? 1, minVals.length ? d3.quantileSorted(minVals, 0.995)! : 0, hs.threshold) * 1.08
  const ys = d3.scaleLinear().domain([0, yMax]).range([trH, 0]).nice()
  const clip = `aeo-ep-${hs.code}`

  const blocks = useMemo<IntervalBlock[]>(() => {
    const groups = [...seasons.map((se) => ({ key: se.name, label: seasonOnly(se.name), color: se.color, f: (e: Episode) => e.season?.name === se.name, hf: (h: { season: { name: string } | null }) => h.season?.name === se.name })),
      { key: 'all', label: 'All', color: INK.text, f: () => true, hf: () => true }]
    return [{ key: 'e', rows: groups.map((g) => {
      const r = episodeRate(eps.filter(g.f), hs.hours.filter(g.hf).length)
      return { key: g.key, label: g.label, labelColor: g.color === INK.text ? undefined : g.color, note: `${r.n} · ${fmt(r.per30, 1)}/30 d`,
        marks: [r.hours ? [{ mid: r.hours.median, lo: r.hours.q25, hi: r.hours.q75, color: g.color }] : [], r.peak ? [{ mid: r.peak.median, lo: r.peak.q25, hi: r.peak.q75, color: g.color }] : []],
        tip: [`${g.label}: ${r.n} episodes, ${fmt(r.per30, 2)} per 30 days of valid hours`, r.hours ? `length median ${fmt(r.hours.median, 0)} h [IQR ${fmt(r.hours.q25, 0)}–${fmt(r.hours.q75, 0)}], longest ${fmt(r.hours.max, 0)} h` : '',
          r.peak ? `peak hour median ${fmt(r.peak.median, 1)} µg/m³, highest ${fmt(r.peak.max, 1)}` : ''].filter(Boolean) }
    }) }]
  }, [eps, hs, seasons])

  return (
    <ChartFrame id="aeth-overview-episodes-hourly" title={`${hs.site} · high-BC episodes — hours above the record's 95th percentile, minute by minute`}
      subtitle={`An episode is a run of at least two consecutive hours with IR BCc above ${fmt(hs.threshold, 1)} µg/m³, the 95th-percentile hour of the whole record (local standard time; a missing hour ends a run). Top: the ${eps.length} episodes starting in the span, by start and highest hour; size grows with length, filled ones have a minute trace (the ${hs.info.traced} highest the minute file covers${hs.info.minute_file ? `, ${hs.info.minute_file.split('/').pop()}` : ''}). Click one. Middle: the picked episode ±6 h — the minutes (thin; noise included, nothing screened), hourly means (steps), the threshold (dashed) and each local day's 24 h mean (grey bars), all a daily value keeps. Bottom: by season, how many, per 30 days of valid hours, how long and how high.`}
      provenance="addis_02_temporal_patterns.ipynb Task 4 (extreme events) and Task 5 (buildup and decay), from the minute record; the episode rule is this page's · gallery/data/export_aeth_overview.py">
      <div ref={ref} className="chart-wrap">
        {!eps.length ? <Empty>No episodes start in this span.</Empty> : <>
          <svg width={width} height={topH + M.top + M.bottom} fontFamily={FONT.family}>
            <g transform={`translate(${M.left},${M.top})`}>
              <YAxis scale={yTop} x={0} gridWidth={iw} tickCount={4} label="Peak hour (µg/m³)" labelX={-46} />
              {eps.map((e) => {
                const on = e.i === sel?.i, has = traced.has(e.i)
                return <circle key={e.i} cx={x(e.start)} cy={yTop(e.peak)} r={2 + Math.sqrt(e.hours) * 1.6 + (on ? 1.5 : 0)} fill={has ? color(e) : INK.panel} fillOpacity={0.8}
                  stroke={on ? INK.text : color(e)} strokeWidth={on ? 2 : 1.2} style={{ cursor: 'pointer' }} onClick={() => setPick(e.i)}
                  onMouseEnter={(ev) => tip.show(ev, [`${hhmm(e.start)} (${DOW[e.dow]})${e.season ? ` · ${seasonOnly(e.season.name)}` : ''}`, `${e.hours} h above ${fmt(hs.threshold, 1)} µg/m³`,
                    `peak ${fmt(e.peak, 1)} µg/m³ in hour ${e.peakAt + 1}, mean ${fmt(e.mean, 1)}`, `rise ${e.peakAt} h, fall ${e.hours - 1 - e.peakAt} h`, has ? 'minute trace: click' : 'no minute trace: click for the hourly view'])}
                  onMouseLeave={tip.hide} />
              })}
              <XAxis scale={x} y={topH} tickCount={8} format={d3.utcFormat('%b %Y')} />
            </g>
          </svg>
          {view && sel && <svg width={width} height={trH + M.top + M.bottom + 16} fontFamily={FONT.family}>
            <text x={M.left} y={12} fontSize={12} fontWeight={600} fill={INK.text}>{hhmm(sel.start)} · {sel.hours} h, peak {fmt(sel.peak, 1)} µg/m³{view.tr ? '' : ' · hourly only (no minute trace)'}</text>
            <g transform={`translate(${M.left},${M.top + 16})`}>
              <clipPath id={clip}><rect width={iw} height={trH} /></clipPath>
              <rect x={xs(sel.start)} width={xs(new Date(sel.start.getTime() + sel.hours * H)) - xs(sel.start)} height={trH} fill={color(sel)} fillOpacity={0.12} />
              <YAxis scale={ys} x={0} gridWidth={iw} tickCount={5} label="IR BCc (µg/m³)" labelX={-46} />
              <g clipPath={`url(#${clip})`}>
                {view.days.map((d) => d.mean === null ? null : <line key={d.date} x1={Math.max(0, xs(new Date(`${d.date}T00:00:00Z`)))} x2={Math.min(iw, xs(new Date(`${d.date}T00:00:00Z`).getTime() + 24 * H))}
                  y1={ys(d.mean)} y2={ys(d.mean)} stroke={INK.neutral} strokeWidth={6} strokeOpacity={0.8} />)}
                <path d={d3.line<{ t: Date; v: number | null }>().defined((m) => m.v !== null).x((m) => xs(m.t)).y((m) => ys(m.v!))(view.minutes) ?? ''} fill="none" stroke={channelColor('IR')} strokeWidth={0.7} strokeOpacity={0.55} />
                <path d={d3.line<{ t: Date; v: number }>().curve(d3.curveStepAfter).x((h) => xs(h.t)).y((h) => ys(h.v))(view.hours) ?? ''} fill="none" stroke={siteColor(meta, hs.site)} strokeWidth={2.2} />
              </g>
              <line x1={0} x2={iw} y1={ys(hs.threshold)} y2={ys(hs.threshold)} stroke={INK.negative} strokeDasharray="5 4" strokeWidth={1.2} />
              <rect width={iw} height={trH} fill="transparent" onMouseMove={(e) => {
                const t = xs.invert(d3.pointer(e)[0])
                const h = view.hours.find((q) => q.t <= t && t.getTime() < q.t.getTime() + H)
                const m = view.minutes.length ? view.minutes[Math.max(0, Math.min(view.minutes.length - 1, Math.round((t.getTime() - view.a.getTime()) / 60e3)))] : null
                const date = t.toISOString().slice(0, 10)
                tip.show(e, [hhmm(t), ...(m ? [`minute ${fmt(m.v, 2)} µg/m³`] : []), `hour ${fmt(h?.v, 2)} µg/m³`, `24 h mean of ${date}: ${fmt(view.days.find((d) => d.date === date)?.mean, 2)} µg/m³`])
              }} onMouseLeave={tip.hide} />
              <XAxis scale={xs} y={trH} tickCount={8} format={d3.utcFormat('%d %b %H:%M')} />
            </g>
          </svg>}
          {sel && !view?.tr && hs.info.minute_file && <Note>The minute file covers only the {hs.info.traced} highest episodes it spans; pick a filled one for the minute trace.</Note>}
          <IntervalStrip blocks={blocks} labelWidth={width < 600 ? 136 : 200} rowHeight={20} noteWidth={width < 600 ? 0 : 92} minPanelWidth={200}
            panels={[{ label: 'Episode length (h)', heading: 'How long' }, { label: 'Peak hour (µg/m³)', heading: 'How high' }]} />
        </>}
        <Key items={[{ label: 'minutes', color: channelColor('IR'), kind: 'line' }, { label: 'hourly mean', color: siteColor(meta, hs.site), kind: 'line' },
          { label: '24 h mean of the local day', color: INK.neutral, kind: 'bar' }, { label: 'threshold (P95 hour)', color: INK.negative, kind: 'dash' },
          ...seasons.map((se) => ({ label: seasonOnly(se.name), color: se.color, kind: 'dot' as const })), { label: 'no minute trace', color: INK.muted, kind: 'hollow' as const }]} />
        {tip.node}
      </div>
    </ChartFrame>
  )
}

/** Episodes at every site, by season: how often, how long, how high. */
export function EpisodesBySite({ series, meta }: { series: HourlySeries[]; meta: MetaFile }) {
  const blocks = useMemo<IntervalBlock[]>(() => series.map((hs) => {
    const seasons = seasonsForSite(meta, hs.site)
    const groups = [...seasons.map((se) => ({ key: se.name, label: seasonOnly(se.name), color: se.color as string | undefined, f: (s: { name: string } | null) => s?.name === se.name })),
      { key: 'all', label: 'All', color: undefined, f: () => true }]
    return { key: hs.code, title: `${hs.site} (above ${fmt(hs.threshold, 1)} µg/m³)`, color: siteColor(meta, hs.site), rows: groups.map((g) => {
      const r = episodeRate(hs.episodes.filter((e) => g.f(e.season)), hs.hours.filter((h) => g.f(h.season)).length)
      const c = siteColor(meta, hs.site)
      return { key: g.key, label: g.label, labelColor: g.color, note: `${r.n} · ${fmt(r.per30, 1)}/30 d`,
        marks: [r.hours ? [{ mid: r.hours.median, lo: r.hours.q25, hi: r.hours.q75, color: c }] : [], r.peak ? [{ mid: r.peak.median / hs.threshold, lo: r.peak.q25 / hs.threshold, hi: r.peak.q75 / hs.threshold, color: c }] : []],
        tip: [`${hs.site} · ${g.label}: ${r.n} episodes, ${fmt(r.per30, 2)} per 30 days of valid hours`, r.hours ? `length median ${fmt(r.hours.median, 0)} h, longest ${fmt(r.hours.max, 0)} h` : '',
          r.peak ? `peak median ${fmt(r.peak.median, 1)} µg/m³ (${fmt(r.peak.median / hs.threshold, 2)}× the threshold)` : ''].filter(Boolean) }
    }) }
  }), [series, meta])
  return (
    <ChartFrame id="aeth-overview-episodes-sites" title="High-BC episodes by site — how often, how long and how high, by season"
      subtitle="Episodes are runs of at least two consecutive hours above each record's own 95th-percentile hour (local standard time), so every site has the same share of high hours and the comparison is of their structure: how many runs they form per 30 days of valid hours, how long the runs are (median and IQR) and how far their peaks rise above the threshold."
      provenance="addis_02_temporal_patterns.ipynb Task 4 (extreme events by season), for every site with an hourly record · gallery/data/export_aeth_overview.py">
      <div className="chart-wrap">
        {!blocks.length ? <Empty>No hourly records at the sites in view.</Empty>
          : <IntervalStrip blocks={blocks} labelWidth={170} rowHeight={18} noteWidth={92} minPanelWidth={200}
            panels={[{ label: 'Episode length (h)', heading: 'How long' }, { label: 'Peak / threshold', heading: 'How high', reference: 1, format: (v) => `${fmt(v, 1)}×` }]} />}
      </div>
    </ChartFrame>
  )
}
