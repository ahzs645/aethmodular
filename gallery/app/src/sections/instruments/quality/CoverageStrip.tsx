import { useMemo, useRef, useState } from 'react'
import * as d3 from 'd3'
import { ChartFrame } from '@/components/ChartFrame'
import { ColorLegend } from '@/components/ColorLegend'
import { XAxis } from '@/components/Axes'
import { Key } from '@/components/Key'
import { Marker } from '@/components/RefLines'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { useColourScheme } from '@/lib/colourSchemes'
import { fmt } from '@/lib/format'
import { FONT, INK } from '@/lib/theme'
import type { MetaFile } from '@/lib/types'
import { finite, instrumentColor, instrumentName, localDate, PERIOD_COLOR, type Day, type QualityIndex } from './data'

const ROW = 22, LABEL = 132
const DAY = 864e5

/** per instrument: the bins, and the calendar days between the first and last bin that have no IR value */
export function coverageSummary(days: Day[]) {
  const withIr = days.filter((d) => finite(d.bc.IR))
  if (!days.length) return { bins: 0, withIr: 0, span: 0, longestGap: 0, gapStart: null as Date | null }
  const first = days[0].t, last = days[days.length - 1].t
  const span = Math.round((+last - +first) / DAY) + 1
  // the longest stretch of calendar days with no IR value, between the first and the last IR day
  let longestGap = 0, gapStart: Date | null = null
  for (let k = 1; k < withIr.length; k++) {
    const gap = Math.round((+withIr[k].t - +withIr[k - 1].t) / DAY) - 1
    if (gap > longestGap) { longestGap = gap; gapStart = new Date(+withIr[k - 1].t + DAY) }
  }
  return { bins: days.length, withIr: withIr.length, span, longestGap, gapStart }
}

/**
 * Which days have black carbon, per instrument, on one calendar: every daily bin the pickle holds, coloured by
 * the input availability its builder recorded (minutes with data over 1440, not verified observed coverage),
 * a solid tick where the bin has IR BCc but no availability field (Addis, Jacros), orange where the bin is
 * there but its BC is missing, and nothing where the pickle has no bin at all (the builder kept only days near
 * filters at some sites, so a blank is not necessarily an instrument outage). The flow-fix boundaries are marked
 * on each row. This is the "gap calendar" of the flow-fix notebooks and ftir_49's missingness section, at the
 * daily resolution the repo holds.
 */
export function CoverageStrip({ all, meta, index }: { all: Record<string, Day[]>; meta: MetaFile; index: QualityIndex | null }) {
  const ref = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(ref)
  const tip = useTooltip(ref)
  const colours = useColourScheme('aq-coverage', 'sequential')
  const [hover, setHover] = useState<{ code: string; d: Day } | null>(null)
  const codes = Object.keys(all)
  const summaries = useMemo(() => Object.fromEntries(codes.map((c) => [c, coverageSummary(all[c])])), [all]) // eslint-disable-line react-hooks/exhaustive-deps
  const t0 = d3.min(codes, (c) => all[c][0]?.t), t1 = d3.max(codes, (c) => all[c][all[c].length - 1]?.t)
  const iw = Math.max(240, width - LABEL - 16)
  const x = d3.scaleTime().domain(t0 && t1 ? [t0, new Date(+t1 + DAY)] : [new Date(), new Date()]).range([0, iw])
  const ramp = d3.scaleSequential(d3.interpolateRgbBasis(colours.stops)).domain([0, 100])
  const cw = Math.max(1, x(new Date(+x.domain()[0] + DAY)) - x(x.domain()[0]))
  const height = codes.length * ROW + 34
  const MISSING = INK.warn
  const fillOf = (d: Day) => !finite(d.bc.IR) ? MISSING : finite(d.coverage) ? ramp(Math.min(100, d.coverage)) : INK.muted

  if (!codes.length) return null
  return (
    <ChartFrame id="aq-coverage" title="Daily coverage — which days have black carbon, per instrument"
      subtitle="Every 24 h bin in the legacy daily pickles, one tick a day, labelled at its start. Colour is the input availability the builder recorded (minutes with data / 1440), which is not verified observed coverage; grey ticks have IR BCc but no availability field (Addis, Jacros); orange ticks are bins the pickle kept whose black carbon is missing; a blank is a day with no bin at all, which at Beijing, Delhi and JPL may only mean the builder did not keep a day that far from a filter. Dashed ticks mark each site's flow-fix boundaries (config.FLOW_FIX_PERIODS). The note counts bins with IR BCc and the longest run of days without it."
      provenance="FlowFix_BeforeAfter_Analysis (data by period), plotting/timeseries.data_completeness, ftir_49 (missingness, daily fallback) · processed_sites/*.pkl, data_completeness_pct">
      <div ref={ref} className="chart-wrap" style={{ position: 'relative' }}>
        <svg width={LABEL + iw + 16} height={height} fontFamily={FONT.family} onMouseLeave={() => { setHover(null); tip.hide() }}>
          <g transform={`translate(${LABEL},4)`}>
            {codes.map((code, k) => {
              const days = all[code], s = summaries[code], p = index?.flow_periods[code]
              const y = k * ROW
              return (
                <g key={code} transform={`translate(0,${y})`}>
                  <rect x={0} y={2} width={iw} height={ROW - 6} fill={INK.empty} />
                  <text x={-10} y={ROW / 2 - 1} dy="0.32em" textAnchor="end" fontSize={11.5} fontWeight={600} fill={instrumentColor(code, meta)}>{instrumentName(code, meta)}</text>
                  {days.map((d) => <rect key={d.i} x={x(d.t)} y={2} width={cw} height={ROW - 6} fill={fillOf(d)} />)}
                  {p?.before_end && <Marker orient="v" at={x(localDate(p.before_end))} from={-1} to={ROW - 3} color={PERIOD_COLOR.before} />}
                  {p?.after_start && <Marker orient="v" at={x(localDate(p.after_start))} from={-1} to={ROW - 3} color={PERIOD_COLOR.after} />}
                  <rect x={0} y={0} width={iw} height={ROW} fill="transparent"
                    onMouseMove={(e) => {
                      const t = x.invert(d3.pointer(e)[0])
                      const d = days[d3.bisector((q: Day) => q.t).center(days, t)]
                      if (!d) return
                      setHover({ code, d })
                      const near = Math.abs(+d.t - +t) < DAY
                      tip.show(e, [instrumentName(code, meta),
                        near ? `bin from ${d.start}` : `no bin on ${d3.utcFormat('%d %b %Y')(t)} (nearest: ${d.start})`,
                        ...(near ? [`IR BCc ${fmt(d.bc.IR, 2)} µg/m³ · availability ${finite(d.coverage) ? `${fmt(d.coverage, 1)} %` : 'not recorded'}`] : []),
                        `${s.withIr} of ${s.bins} bins with IR BCc over ${s.span} days; longest run without: ${s.longestGap} d${s.gapStart ? ` from ${d3.utcFormat('%d %b %Y')(s.gapStart)}` : ''}`])
                    }} />
                  {hover?.code === code && <rect x={x(hover.d.t) - 1} y={0} width={cw + 2} height={ROW - 2} fill="none" stroke={INK.text} strokeWidth={1} pointerEvents="none" />}
                </g>
              )
            })}
            <XAxis scale={x} y={codes.length * ROW + 2} tickCount={10} format={d3.utcFormat('%b %Y')} />
          </g>
        </svg>
        <Key items={[{ label: 'IR BCc, availability not recorded', color: INK.muted, kind: 'square' }, { label: 'bin kept, IR BCc missing', color: MISSING, kind: 'square' }]}>
          <ColorLegend scale={ramp} label="Input availability (%)" width={150} scheme={colours.scheme} format={(v) => v.toFixed(0)} />
        </Key>
        <div style={{ overflowX: 'auto' }}><table className="census-table">
          <thead><tr><th>Instrument</th><th>Bins</th><th>With IR BCc</th><th>Calendar span (d)</th><th>Longest run without IR (d)</th></tr></thead>
          <tbody>{codes.map((c) => { const s = summaries[c]; return (
            <tr key={c}><td>{instrumentName(c, meta)}</td><td>{s.bins}</td><td>{s.withIr} ({Math.round((100 * s.withIr) / Math.max(1, s.span))} % of days)</td><td>{s.span}</td>
              <td>{s.longestGap}{s.gapStart ? ` from ${d3.utcFormat('%d %b %Y')(s.gapStart)}` : ''}</td></tr>) })}</tbody>
        </table></div>
        {tip.node}
      </div>
    </ChartFrame>
  )
}

