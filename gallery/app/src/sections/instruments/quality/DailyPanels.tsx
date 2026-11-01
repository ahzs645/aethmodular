import { useMemo } from 'react'
import * as d3 from 'd3'
import { Key, type KeyItem } from '@/components/Key'
import { Marker } from '@/components/RefLines'
import { StackedPanels, type StackMarker, type StackPanel } from '@/components/StackedPanels'
import { robustDomain } from '@/lib/axes'
import { fmt } from '@/lib/format'
import { lineRuns, nearestIndex } from '@/lib/stacked'
import { INK } from '@/lib/theme'
import { finite, localDate, PERIOD_COLOR, PERIOD_LABEL, type Day, type FlowPeriods, type Period } from './data'

export interface PanelLine {
  key: string; label: string; color: string
  value: (d: Day) => number | null
  dash?: string
  /** decimals in the tooltip (default 3) */
  digits?: number
}
export interface PanelSpec {
  key: string
  /** the panel's y title */
  title: string
  lines: PanelLine[]
  /** reference values (an ideal ratio, a cleaning bound) */
  refs?: { at: number; label: string; color?: string }[]
  /** a fixed y range; otherwise the 1st–99th percentile of the panel's values, widened to its references */
  domain?: [number, number]
  /** points coloured by the day's flow period instead of lines (the BC panel of the flow-fix notebooks) */
  byPeriod?: boolean
}

const M = { top: 10, right: 18, bottom: 32, left: 72 }
const GAP = 16
/** consecutive bins further apart than this are not joined: a missing day breaks the line */
const JOIN_DAYS = 1.5

/**
 * Daily series in stacked panels on one date axis (components/StackedPanels), the way the flow-fix and QC notebooks
 * drew them, with the site's flow-fix boundaries marked (plotting_gaps_scenarios scenario 3: a dashed line at each
 * boundary and the gap between them shaded) and each panel's reference values dashed (scenario 2). A line joins only
 * bins on consecutive days (lib/stacked lineRuns), so every missing day shows as a break, never a straight bridge;
 * a day with no neighbour is a dot. The y range is each panel's 1st–99th percentile, so a few spikes run off the
 * panel instead of flattening it; the note under the chart counts them.
 */
export function DailyPanels({ days, panels, periods, panelHeight = 96, extraTip }: {
  days: Day[]
  panels: PanelSpec[]
  periods?: FlowPeriods
  panelHeight?: number
  extraTip?: (d: Day) => string[]
}) {
  const ext = d3.extent(days, (d) => d.t) as [Date, Date]
  const domain: [Date, Date] = ext[0] ? [new Date(+ext[0] - 3 * 864e5), new Date(+ext[1] + 4 * 864e5)] : [new Date(0), new Date(864e5)]
  const ranges = useMemo(() => panels.map((p) => {
    const vals = days.flatMap((d) => p.lines.map((l) => l.value(d)))
    const rd = p.domain ? { domain: p.domain, below: 0, above: 0 } : robustDomain(vals, 0.01, 0.99, { include: (p.refs ?? []).map((r) => r.at) })
    const [a, b] = rd.domain
    return { domain: (a === b ? [a - 1, b + 1] : [a, b]) as [number, number], nice: !p.domain, off: rd.below + rd.above }
  }), [days, panels])
  const offPanel = ranges.reduce((n, s) => n + s.off, 0)

  const be = periods?.before_end ? localDate(periods.before_end) : null
  const as = periods?.after_start ? localDate(periods.after_start) : null
  const inView = (t: Date | null) => !!t && t >= domain[0] && t <= domain[1]
  const markers: StackMarker[] = [
    ...(be ? [{ at: be, label: 'before ends', color: PERIOD_COLOR.before, labelSide: 'left' as const }] : []),
    ...(as ? [{ at: as, label: 'after starts', color: PERIOD_COLOR.after }] : []),
  ]
  const keyItems: KeyItem[] = []
  for (const p of panels) for (const l of p.lines) if (!p.byPeriod && !keyItems.some((k) => k.label === l.label)) keyItems.push({ label: l.label, color: l.color, kind: l.dash ? 'dash' : 'line' })
  const periodsShown = panels.some((p) => p.byPeriod) ? [...new Set(days.map((d) => d.flow as Period | null).filter((p): p is Period => !!p))] : []

  const stack: StackPanel<d3.ScaleTime<number, number>>[] = panels.map((p, k) => ({
    key: p.key,
    y: { domain: ranges[k].domain, nice: ranges[k].nice, label: p.title, labelX: -54, wrapLabel: true, format: (v) => String(+(+v).toPrecision(3)) },
    background: ({ x, height }) => inView(be) && inView(as) && <rect x={x(be!)} width={Math.max(0, x(as!) - x(be!))} height={height} fill={INK.shade} />,
    children: ({ x, y, width, clip }) => <g clipPath={clip}>
      {(p.refs ?? []).map((r) => <Marker key={r.label} orient="h" at={y(r.at)} from={0} to={width} label={r.label} color={r.color ?? INK.muted} />)}
      {p.byPeriod ? p.lines.flatMap((l) => days.map((d) => {
        const v = l.value(d)
        if (!finite(v)) return null
        return <circle key={`${l.key}${d.i}`} cx={x(d.t)} cy={y(v)} r={1.9} fill={PERIOD_COLOR[(d.flow as Period) ?? 'no_fix'] ?? l.color} fillOpacity={0.75} />
      })) : p.lines.map((l) => lineRuns(days, (d) => d.t, l.value, JOIN_DAYS).map((seg, j) => seg.length > 1
        ? <path key={`${l.key}${j}`} d={d3.line<{ t: Date; v: number }>().x((q) => x(q.t)).y((q) => y(q.v))(seg) ?? ''} fill="none" stroke={l.color} strokeWidth={1.2} strokeDasharray={l.dash} />
        : <circle key={`${l.key}${j}`} cx={x(seg[0].t)} cy={y(seg[0].v)} r={1.6} fill={l.color} />))}
    </g>,
  }))
  const readout = (px: number, x: d3.ScaleTime<number, number>) => {
    const d = days[nearestIndex(days, (q) => +q.t, +x.invert(px))]
    if (!d) return null
    const lines: string[] = [`bin from ${d.start}${d.flow ? ` · ${PERIOD_LABEL[d.flow as Period] ?? d.flow}` : ''}`]
    for (const p of panels) for (const l of p.lines) lines.push(`${l.label}: ${fmt(l.value(d), l.digits ?? 3)}`)
    return { at: x(d.t), tip: [...lines, ...(extraTip?.(d) ?? [])] }
  }

  return (
    <div className="chart-wrap" style={{ position: 'relative' }}>
      <StackedPanels x={{ type: 'time', domain }} panels={stack} margin={M} gap={GAP} panelHeight={panelHeight} xMarkers={markers} readout={readout} />
      <Key items={[...periodsShown.map((p) => ({ label: PERIOD_LABEL[p], color: PERIOD_COLOR[p] } as KeyItem)), ...keyItems,
        ...(be && as ? [{ label: 'between the flow-fix boundaries', color: INK.shade, kind: 'bar' } as KeyItem] : [])]} />
      {offPanel > 0 && <p className="chart-note">{offPanel} daily value{offPanel === 1 ? '' : 's'} outside the panels' 1st–99th percentile range run off the edge.</p>}
    </div>
  )
}

/** a panel line for one QC series */
export const qLine = (key: string, label: string, color: string, extra: Partial<PanelLine> = {}): PanelLine =>
  ({ key, label, color, value: (d) => d.q(key), ...extra })

