import { useMemo, useRef, useState } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Note, Segmented, Toggle } from '@/components/ChartFrame'
import { robustDomain } from '@/lib/axes'
import { XAxis, YAxis } from '@/components/Axes'
import { Legend, useLegend } from '@/components/Legend'
import { Key } from '@/components/Key'
import { SeasonBands, timeSeasonBands } from '@/components/SeasonBands'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { focusLast, focusStyle, useHighlight } from '@/lib/highlight'
import { fidProps } from '@/lib/filterEdits'
import { fmt } from '@/lib/format'
import { INK } from '@/lib/theme'
import { seasonOnly } from '@/lib/seasons'
import { useSearchParam } from '@/lib/url'
import type { MetaFile } from '@/lib/types'
import { seasonsForSite } from '@/siteSeasons'
import { CHANNELS, FILTER_METHODS, SCREEN_TEXT, channelColor, channelName, logTick, type Channel, type Exclusion, type Screen, type Series } from './data'
import { movingMean, splitRuns } from './stats'

const M = { top: 10, right: 20, bottom: 30, left: 68 }
const GAP = 16
type Avg = 'daily' | 'both' | '30d'
const AVG_LABEL: Record<Avg, string> = { daily: 'Daily', both: 'Daily + 30-day mean', '30d': '30-day mean' }
const t0 = (date: string) => new Date(`${date}T00:00:00Z`)

/**
 * One instrument's daily BCc at the five wavelengths (Multi_Site_Analysis §1–2, _Fixed §3 and §13,
 * _FollowUp §7; addis_02_daily Task 3's 30-day mean over the season shading), above the same
 * filters measured three ways (addis_02.5 Task 4) and, where the file has it, each day's minute
 * coverage (Multi_Site_Analysis §4).
 */
export function DailyTimeline({ s, meta, screen, excluded = [] }: { s: Series; meta: MetaFile; screen: Screen; excluded?: Exclusion[] }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
  const hl = useHighlight()
  const [avg, setAvg] = useSearchParam<Avg>('aeot', 'both', ['daily', 'both', '30d'])
  const [log, setLog] = useState(false)
  const [full, setFull] = useState(false)
  const [cursor, setCursor] = useState<number | null>(null)
  const leg = useLegend()
  const days = s.days

  const hasCov = days.some((d) => d.coverage !== null)
  const panels = ['bc', 'filters', ...(hasCov ? ['cov'] : [])] as const
  const H: Record<string, number> = { bc: 230, filters: 110, cov: 56 }
  const innerW = Math.max(240, width - M.left - M.right)
  const tops: Record<string, number> = {}
  let y0 = M.top
  for (const p of panels) { tops[p] = y0; y0 += H[p] + GAP }
  const height = y0 - GAP + M.bottom

  const shown = CHANNELS.filter((c) => leg.show(c))
  const smooth = useMemo(() => Object.fromEntries(CHANNELS.map((c) => [c, movingMean(days, (d) => d.bc[c])])) as Record<Channel, (number | null)[]>, [days])
  const filterPts = useMemo(() => s.rows.flatMap((r) => FILTER_METHODS.map((m) => ({ row: r, m, t: t0(r.date), v: typeof r[m.field] === 'number' ? (r[m.field] as number) : null })))
    .filter((p) => p.v !== null), [s.rows])

  const domain = useMemo<[Date, Date] | null>(() => {
    if (!days.length) return null
    return [days[0].t, new Date(days[days.length - 1].t.getTime() + 864e5)]
  }, [days])
  const x = d3.scaleUtc().domain(domain ?? [new Date(0), new Date(1)]).range([0, innerW])

  const yVals = shown.flatMap((c) => [...(avg !== '30d' ? days.map((d) => d.bc[c]) : []), ...(avg !== 'daily' ? smooth[c] : [])])
    .filter((v): v is number => v !== null && (!log || v > 0))
  // the bulk of the days (to the 99.5th percentile) sets the scale unless the full range is asked for; spikes run off the top, clipped
  const rd = robustDomain(yVals, full ? 0 : 0.002, full ? 1 : 0.995, { include: log ? [] : [0] })
  const yBc = log
    ? d3.scaleLog().domain([Math.max(1e-3, rd.domain[0] * 0.9), rd.domain[1] * 1.1]).range([H.bc, 0]).nice()
    : d3.scaleLinear().domain([rd.domain[0], rd.domain[1] * 1.04]).range([H.bc, 0]).nice()
  const offTop = yVals.filter((v) => v > yBc.domain()[1]).length
  const fVals = filterPts.filter((p) => leg.show(p.m.label) && p.t >= x.domain()[0] && p.t <= x.domain()[1]).map((p) => p.v!)
  const yF = d3.scaleLinear().domain([Math.min(0, d3.min(fVals) ?? 0), (d3.max(fVals) ?? 1) * 1.05]).range([H.filters, 0]).nice()
  const yC = d3.scaleLinear().domain([0, 100]).range([H.cov, 0])

  const runs = useMemo(() => splitRuns(days), [days])
  const smoothRuns = useMemo(() => splitRuns(days, 15), [days])
  // a bin's value sits at its middle
  const xm = (t: Date) => x(new Date(t.getTime() + 864e5 / 2))
  const path = (vals: (number | null)[], idx: number[]) =>
    d3.line<number>().defined((i) => vals[i] !== null && (!log || vals[i]! > 0)).x((i) => xm(days[i].t)).y((i) => yBc(vals[i]!))(idx) ?? ''
  const seasons = seasonsForSite(meta, s.inst.site)
  const bands = domain ? timeSeasonBands(x, seasons, innerW) : []
  const at = cursor === null || !days.length ? null : days[d3.bisector((d: { t: Date }) => d.t).center(days, new Date(x.invert(cursor).getTime() - 864e5 / 2))]
  const inX = (t: Date) => !domain || (t >= domain[0] && t <= domain[1])

  const legendItems = [
    ...CHANNELS.map((c) => ({ label: c, color: channelColor(c), shape: 'line' as const, detail: `${meta.wavelengths_nm[c]} nm` })),
    ...FILTER_METHODS.map((m) => ({ label: m.label, color: m.color, shape: 'dot' as const })),
  ]

  return (
    <ChartFrame id="aeth-overview-timeline" title={`${s.inst.site} · daily black carbon at five wavelengths — and the filters on the same dates`}
      subtitle={`Each 24 h bin of ${s.inst.label}'s daily file (${s.inst.info.file}), labelled at its start: BCc at 375, 470, 528, 625 and 880 nm (thin lines, broken where days are missing), with a centred 30-day mean (thick; at least 7 days with data). Below, the site's filters in view measured three ways: the aethalometer's IR BCc averaged over each filter's sampling window, FTIR EC, and HIPS Fabs / MAC (${meta.mac_value} m²/g); click a filter for its record. ${hasCov ? 'Bottom: the minutes with data, % of each day. ' : ''}Shaded: the site's seasons. Screen: ${SCREEN_TEXT[screen]}.`}
      provenance="Multi_Site_Analysis.ipynb §1, §2 and §4; Multi_Site_Analysis_Fixed.ipynb §3 and §13; Multi_Site_Analysis_FollowUp.ipynb §7; addis_02_temporal_patterns_daily.ipynb Task 3; addis_02.5_temporal_patterns_filter.ipynb Task 4 · gallery/data/export_aethalometer.py"
      controls={<>
        <Segmented label="Show" value={avg} options={['daily', 'both', '30d'] as Avg[]} onChange={setAvg} optionLabel={(o) => AVG_LABEL[o]} />
        <Toggle label="Log scale" checked={log} onChange={setLog} />
        <Toggle label="Full range" checked={full} onChange={setFull} title="Scale to every value, spikes included (otherwise the top 0.5 % of values run off the top)" />
      </>}>
      <div ref={wrapRef} className="chart-wrap">
        {!domain ? <Empty>No daily bins for {s.inst.label} in this span.</Empty> : (
          <svg width={width} height={height} onMouseLeave={() => { setCursor(null); tip.hide() }}>
            <g transform={`translate(${M.left},0)`}>
              {panels.map((k) => (
                <g key={k} transform={`translate(0,${tops[k]})`}>
                  <SeasonBands bands={bands} height={H[k]} hover={leg.hover && seasons.some((se) => se.name === leg.hover) ? leg.hover : null} />
                  {k === 'bc' && <>
                    <clipPath id="aeth-overview-timeline-clip"><rect width={innerW} height={H.bc} /></clipPath>
                    <YAxis scale={yBc as d3.ScaleLinear<number, number>} x={0} gridWidth={innerW} tickCount={log ? 12 : 5} label="BCc (µg/m³)" labelX={-50} format={(v) => (log ? logTick(v) : String(v))} />
                    {shown.map((c) => <g key={c} opacity={leg.dim(c)} clipPath="url(#aeth-overview-timeline-clip)">
                      {avg !== '30d' && runs.map((idx, i) => <path key={i} d={path(days.map((d) => d.bc[c]), idx)} fill="none" stroke={channelColor(c)}
                        strokeWidth={avg === 'both' ? 0.8 : 1.2} strokeOpacity={avg === 'both' ? 0.45 : 0.85} />)}
                      {avg !== 'daily' && smoothRuns.map((idx, i) => <path key={`m${i}`} d={path(smooth[c], idx)} fill="none" stroke={channelColor(c)} strokeWidth={2.2} />)}
                    </g>)}
                    {s.rows.map((r) => inX(t0(r.date)) && <line key={r.id} x1={x(t0(r.date))} x2={x(t0(r.date))} y1={H.bc - 5} y2={H.bc} stroke={INK.muted} strokeWidth={0.8} />)}
                    {excluded.map((xd) => inX(t0(xd.date)) && <line key={xd.date} x1={x(t0(xd.date)) + 0.5} x2={x(t0(xd.date)) + 0.5} y1={0} y2={H.bc} stroke={INK.negative} strokeWidth={1.4} strokeDasharray="4 3" pointerEvents="none" />)}
                    <rect width={innerW} height={H.bc} fill="transparent"
                      onMouseMove={(e) => {
                        const mx = d3.pointer(e)[0]
                        setCursor(mx)
                        const d = days[d3.bisector((q: { t: Date }) => q.t).center(days, new Date(x.invert(mx).getTime() - 864e5 / 2))]
                        if (!d) return
                        const i = days.indexOf(d)
                        const ex = excluded.filter((xd) => xd.date === d.date)
                        tip.show(e, [`${d.date} (bin from ${d.start})`, ...(d.season ? [seasonOnly(d.season.name)] : []), ...ex.map((xd) => `excluded sample${xd.filter_id ? ` ${xd.filter_id}` : ''}: ${xd.reason}`),
                          ...CHANNELS.map((c) => `${channelName(c, meta)}: ${fmt(d.bc[c], 2)} µg/m³ · 30-day ${fmt(smooth[c][i], 2)}`),
                          ...(d.coverage !== null ? [`minute coverage ${fmt(d.coverage, 0)} %`] : []), ...(d.flow ? [`flow period: ${d.flow}`] : [])])
                      }} />
                  </>}
                  {k === 'filters' && <>
                    <YAxis scale={yF} x={0} gridWidth={innerW} tickCount={3} label="Filters (µg/m³)" labelX={-50} />
                    {focusLast(filterPts.filter((p) => leg.show(p.m.label) && inX(p.t)), (p) => p.row.id, hl.focusId).map(([p]) => {
                      const st = focusStyle(p.row.id, hl.focusId, { r: 2.8, level: 'sparse' }, hl.selected)
                      return <circle key={`${p.row.id}|${p.m.key}`} cx={x(p.t)} cy={yF(p.v!)} r={st.r} fill={p.m.color} fillOpacity={st.opacity * leg.dim(p.m.label)} data-dot={leg.dim(p.m.label) === 1 ? st.dot : undefined}
                        stroke={st.stroke} strokeWidth={st.strokeWidth} style={{ cursor: 'pointer' }} {...fidProps(p.row.id)}
                        onMouseEnter={(e) => { hl.setHover(p.row.id); tip.show(e, [p.row.id, `${p.row.site} · ${p.row.date} · ${seasonOnly(String(p.row.season ?? ''))}`,
                          ...FILTER_METHODS.map((m) => `${m.label}: ${fmt(p.row[m.field], 2)} µg/m³`), 'click for details · right-click to exclude']) }}
                        onMouseLeave={() => { hl.setHover(null); tip.hide() }} onClick={() => hl.openSample(p.row.id)} />
                    })}
                  </>}
                  {k === 'cov' && <>
                    <YAxis scale={yC} x={0} gridWidth={innerW} tickCount={2} label="Coverage %" labelX={-50} />
                    {days.map((d) => d.coverage !== null && d.coverage > 0 && <rect key={d.start} x={x(d.t)} width={Math.max(1, x(new Date(d.t.getTime() + 864e5)) - x(d.t) - 0.3)}
                      y={yC(Math.min(100, d.coverage))} height={H.cov - yC(Math.min(100, d.coverage))} fill={d.bc.IR === null ? INK.neutral : INK.axis} fillOpacity={0.75} />)}
                  </>}
                </g>
              ))}
              {at && <line x1={xm(at.t)} x2={xm(at.t)} y1={M.top} y2={height - M.bottom} stroke={INK.text} strokeOpacity={0.35} strokeDasharray="3 3" pointerEvents="none" />}
              <XAxis scale={x} y={height - M.bottom} tickCount={8} format={d3.utcFormat('%b %Y')} />
            </g>
          </svg>
        )}
        {offTop > 0 && <Note>{offTop} plotted values above {fmt(yBc.domain()[1], 1)} µg/m³ run off the top (largest {fmt(d3.max(yVals), 1)}); tick Full range to see them.</Note>}
        <Legend items={legendItems} {...leg.props} />
        <Key items={[...seasons.map((se) => ({ label: seasonOnly(se.name), color: se.color, kind: 'faded' as const })), ...(excluded.length ? [{ label: 'excluded sample (outliers.py)', color: INK.negative, kind: 'dash' as const }] : [])]} />
        {tip.node}
      </div>
    </ChartFrame>
  )
}
