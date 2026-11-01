import { useMemo } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty } from '@/components/ChartFrame'
import { Key, type KeyItem } from '@/components/Key'
import { PanelPicker } from '@/components/PanelPicker'
import { Marker, RefLine } from '@/components/RefLines'
import { useRolling } from '@/components/RollingControls'
import { ScatterLayer } from '@/components/ScatterLayer'
import { StackedPanels, type StackMarker, type StackPanel } from '@/components/StackedPanels'
import type { useTooltip } from '@/hooks/useTooltip'
import { robustDomain } from '@/lib/axes'
import { fmt } from '@/lib/format'
import { fidProps } from '@/lib/filterEdits'
import { ExcludedNote, toMarks, useMarkReason, useMarkedRows } from '@/lib/markedRows'
import { usePanelPick } from '@/lib/panelPick'
import { rollingBy, rollingByTime, type RollSpec } from '@/lib/rolling'
import { lineRuns, nearestIndex } from '@/lib/stacked'
import { pearson } from '@/lib/stats'
import { INK } from '@/lib/theme'
import type { FilterRow, MetaFile } from '@/lib/types'
import { finite, instrumentName, localDate, PERIOD_COLOR, PERIOD_LABEL, periodOf, ratio, type Period, type QualityIndex } from './data'
import { AETH_IR } from './FlowFixCompare'

const HIPS = 'HIPS BC', FTIR = 'EC (FTIR)'
/** a rolling line is broken wherever the filters stop for more than this many days (no bridge across a sampling gap) */
const GAP_DAYS = 45
/** the ratios' rolling statistic: the explorer's trailing 30-day mean of at least 3 filters (the reader may change it) */
const ROLL: RollSpec = { days: 30, stat: 'mean', align: 'trailing', minN: 3 }
const R_COLOR = '#2ca02c'
const PAIRS = { 'aeth-hips': [AETH_IR, HIPS], 'aeth-ftir': [AETH_IR, FTIR], 'hips-ftir': [HIPS, FTIR] } as const
type RatioKey = keyof typeof PAIRS
const PANEL_IDS = ['aeth-hips', 'aeth-ftir', 'hips-ftir', 'r'] as const
const PANEL_LABEL: Record<string, string> = { 'aeth-hips': 'IR BCc / HIPS', 'aeth-ftir': 'IR BCc / FTIR EC', 'hips-ftir': 'HIPS / FTIR EC', r: '60-day r' }
const Y_LABEL: Record<string, string> = { 'aeth-hips': 'IR BCc / HIPS BC', 'aeth-ftir': 'IR BCc / FTIR EC', 'hips-ftir': 'HIPS BC / FTIR EC', r: '60-day r with HIPS' }
const R_MARKERS = [{ at: 0.8, label: 'r = 0.8' }, { at: 0.5, label: 'r = 0.5' }]

type RatioPt = { r: FilterRow; t: Date; v: number; period: Period }
type Roll = { d: Date; v: number | null }

/**
 * When did the methods' relationship change? One site's filters over time: the aethalometer over HIPS, the
 * aethalometer over FTIR EC and HIPS over FTIR EC per filter (coloured by flow period) with their rolling mean, and
 * the 60-day rolling correlation of the aethalometer with HIPS, stacked on one date axis (components/StackedPanels).
 * The flow-fix boundaries are marked, so a step in a ratio can be read against them (flow_fix_explorer: stacked
 * series, rolling correlation, ratio time series). The rolling windows are lib/rolling's.
 */
export function AgreementOverTime({ code, rows, meta, index }: { code: string; rows: FilterRow[]; meta: MetaFile; index: QualityIndex }) {
  const periods = index.flow_periods[code]
  const rolling = useRolling('aq-agreement-time', ROLL)
  const pick = usePanelPick('aq-agreement-time', PANEL_IDS, PANEL_IDS)
  const site = useMemo(() => rows.filter((r) => r.code === code).sort((a, b) => a.date.localeCompare(b.date)), [rows, code])
  // Filters → excluded: Mark (lib/markedRows): this instrument's excluded filters, × on the ratio panels, in no rolling line
  const markedAll = useMarkedRows()
  const markedSite = useMemo(() => markedAll.filter((r) => r.code === code), [markedAll, code])
  const ratioOk = (r: FilterRow, num: string, den: string) => { const v = ratio(r[num] as number, r[den] as number); return finite(v) && (r[den] as number) > 0.01 ? v : null }
  const { days, align, minN } = rolling.spec
  const ratios = useMemo(() => Object.fromEntries((Object.keys(PAIRS) as RatioKey[]).map((k) => {
    const [num, den] = PAIRS[k]
    // the explorer drops denominators under 0.01 µg/m³
    const pts: RatioPt[] = site.flatMap((r) => { const v = ratioOk(r, num, den); return v === null ? [] : [{ r, t: localDate(r.date), v, period: periodOf(r.date, periods) }] })
    const roll = rollingByTime(pts, (p) => p.t, (p) => p.v, { days, stat: rolling.spec.stat, align, minN })
    // the bulk of the ratios (2–98 %, and 1), so a few extreme ones sit hollow at the edge
    const rd = robustDomain([...pts.map((p) => p.v), ...roll.flatMap((q) => (q.v === null ? [] : [q.v]))], 0.02, 0.98, { include: [1] })
    return [k, { pts, roll, domain: rd.domain }]
  })) as Record<RatioKey, { pts: RatioPt[]; roll: Roll[]; domain: [number, number] }>, [site, periods, days, rolling.spec.stat, align, minN]) // eslint-disable-line react-hooks/exhaustive-deps
  const corr = useMemo(() => {
    const pts = site.filter((r) => finite(r[AETH_IR] as number) && finite(r[HIPS] as number)).map((r) => ({ t: localDate(r.date), a: r[AETH_IR] as number, b: r[HIPS] as number }))
    return rollingBy(pts, (p) => p.t, { days: 60, align: 'trailing' }, (win) => (win.length >= 5 ? pearson(win.map((p) => p.a), win.map((p) => p.b)) : null))
      .filter((q): q is { d: Date; v: number } => q.v !== null && Number.isFinite(q.v))
  }, [site])
  const markers: StackMarker[] = [
    ...(periods?.before_end ? [{ at: localDate(periods.before_end), label: 'before ends', color: PERIOD_COLOR.before, labelSide: 'left' as const }] : []),
    ...(periods?.after_start ? [{ at: localDate(periods.after_start), label: 'after starts', color: PERIOD_COLOR.after }] : []),
  ]
  const xDomain: [Date, Date] | null = site.length ? [localDate(site[0].date), localDate(site[site.length - 1].date)] : null
  const hasData = (k: string) => (k === 'r' ? corr.length > 0 : ratios[k as RatioKey].pts.length > 0)

  const panels: StackPanel<d3.ScaleTime<number, number>>[] = pick.ids.filter(hasData).map((k) => {
    if (k === 'r') return {
      key: k, y: { domain: [-0.2, 1.05], label: Y_LABEL.r, labelX: -50, tickCount: 5 },
      children: ({ x, y, width }) => <g pointerEvents="none">
        {R_MARKERS.map((m) => <Marker key={m.at} orient="h" at={y(m.at)} from={0} to={width} label={m.label} />)}
        {lineRuns(corr, (q) => q.d, (q) => q.v, GAP_DAYS).map((run, i) => <RunPath key={i} run={run} x={x} y={y} color={R_COLOR} />)}
      </g>,
    }
    const { pts, roll, domain } = ratios[k as RatioKey]
    const [num, den] = PAIRS[k as RatioKey]
    return {
      key: k, y: { domain, nice: true, label: Y_LABEL[k], labelX: -50, tickCount: 5 }, pointer: false,
      background: ({ y, width }) => y.domain()[0] <= 1 && y.domain()[1] >= 1 && <RefLine orient="h" at={y(1)} from={0} to={width} />,
      children: ({ x, y, width, height, tip }) => <RatioPoints pts={pts} roll={roll} num={num} den={den} marked={markedSite} x={x} y={y} width={width} height={height} tip={tip} />,
    }
  })
  // the cursor reads the rolling correlation (the ratio panels answer with their nearest filter instead)
  const readout = (px: number, x: d3.ScaleTime<number, number>) => {
    const q = corr[nearestIndex(corr, (c) => +c.d, +x.invert(px))]
    return q ? { at: x(q.d), tip: [d3.timeFormat('%Y-%m-%d')(q.d), `60-day rolling r with HIPS: ${fmt(q.v, 2)}`] } : null
  }

  return (
    <ChartFrame id="aq-agreement-time" title={`${instrumentName(code, meta)} · agreement over time — when did the relationship change?`}
      subtitle={`Per filter: the aethalometer's IR BCc over HIPS BC (HIPS Fabs / MAC), over FTIR EC, and HIPS over FTIR EC, coloured by flow period, with a ${rolling.label} (at least ${minN} filters); and the trailing 60-day correlation of IR BCc with HIPS (at least 5 filters). Denominators under 0.01 µg/m³ are left out, as in the explorer. The dashed verticals are the flow-fix boundaries; the y range is the bulk of the ratios, so a few extreme ones sit hollow at the edge. Choose the panels, and the rolling window, in the options.`}
      provenance="flow_fix_explorer (plot_stacked_timeseries, plot_rolling_correlation, plot_ratio_timeseries) · FTIR EC is a spectral prediction"
      controls={<><PanelPicker pick={pick} mode="toggle" min={1} labelOf={(k) => PANEL_LABEL[k]} />{rolling.controls}</>}>
      {!site.length || !xDomain ? <Empty>No filters from {instrumentName(code, meta)} in the current subset.</Empty> : <>
        <div className="chart-wrap">
          {panels.length ? <StackedPanels x={{ type: 'time', domain: xDomain }} panels={panels} xMarkers={markers} readout={readout}
            margin={{ top: 14, right: 16, bottom: 30, left: 62 }} gap={18} panelHeight={112} />
            : <Empty>None of the chosen panels has data for this instrument.</Empty>}
        </div>
        <ExcludedNote count={markedSite.filter((r) => Object.values(PAIRS).some(([a, b]) => ratioOk(r, a, b) !== null)).length} drawn />
        <Key items={[...new Set(site.map((r) => periodOf(r.date, periods)))].map((k): KeyItem => ({ label: PERIOD_LABEL[k], color: PERIOD_COLOR[k] }))
          .concat([{ label: rolling.label, color: INK.text, kind: 'line' as const }, { label: '60-day rolling r', color: R_COLOR, kind: 'line' as const }])} />
      </>}
    </ChartFrame>
  )
}

function RunPath({ run, x, y, color, width = 1.7 }: { run: { t: Date; v: number }[]; x: d3.ScaleTime<number, number>; y: d3.ScaleLinear<number, number>; color: string; width?: number }) {
  return <path d={d3.line<{ t: Date; v: number }>().x((q) => x(q.t)).y((q) => y(q.v))(run) ?? ''} fill="none" stroke={color} strokeWidth={width} />
}

/**
 * A ratio panel: each filter's ratio coloured by flow period, through components/ScatterLayer (linked hover, pin,
 * click to open, right-click to exclude), with the rolling mean over it. Ratios beyond the panel's range sit hollow
 * at its edge and are left out of the line; the panel's corner counts them.
 */
function RatioPoints({ pts, roll, num, den, marked, x, y, width, height, tip }: {
  pts: RatioPt[]; roll: Roll[]; num: string; den: string; marked: readonly FilterRow[]
  x: d3.ScaleTime<number, number>; y: d3.ScaleLinear<number, number>; width: number; height: number; tip: ReturnType<typeof useTooltip>
}) {
  const [y0, y1] = y.domain()
  const out = (v: number) => v < y0 || v > y1
  const edge = (v: number) => y(Math.max(y0, Math.min(y1, v)))
  const xy = useMemo(() => pts.map((p) => [x(p.t), edge(p.v)] as const), [pts, x, y]) // eslint-disable-line react-hooks/exhaustive-deps
  const dots = useMemo(() => pts.map((p, i) => {
    const c = PERIOD_COLOR[p.period], o = out(p.v)
    return <circle key={p.r.id} cx={xy[i][0]} cy={xy[i][1]} r={2.6} fill={o ? INK.panel : c} stroke={c} strokeWidth={o ? 1.6 : 0} {...fidProps(p.r.id)} />
  }), [pts, xy]) // eslint-disable-line react-hooks/exhaustive-deps
  const reasonOf = useMarkReason()
  const marks = useMemo(() => toMarks(marked, (r) => {
    const v = ratio(r[num] as number, r[den] as number)
    return finite(v) && (r[den] as number) > 0.01 && !out(v) ? [x(localDate(r.date)), y(v)] : null
  }, reasonOf, (r) => [`${num} / ${den}: ${fmt(ratio(r[num] as number, r[den] as number), 2)}`]), [marked, num, den, x, y, reasonOf]) // eslint-disable-line react-hooks/exhaustive-deps
  const nOut = pts.filter((p) => out(p.v)).length
  return <>
    <ScatterLayer items={pts} xy={xy} idOf={(p) => p.r.id} dots={dots} width={width} height={height} tip={tip} marks={marks} focusR={4}
      tipOf={(p) => [p.r.id, p.r.date, `${num} / ${den}: ${fmt(p.v, 2)}`, PERIOD_LABEL[p.period], 'click for details · right-click to exclude']}
      focus={(p, [cx, cy]) => <circle cx={cx} cy={cy} r={4} fill={PERIOD_COLOR[p.period]} stroke={INK.text} strokeWidth={1.2} />}>
      <g pointerEvents="none">
        {/* the line joins the drawable values (a thin window or an extreme one is skipped), breaking only at a gap in the filters */}
        {lineRuns(roll.filter((q) => q.v !== null && !out(q.v)), (q) => q.d, (q) => q.v, GAP_DAYS).map((run, i) => <RunPath key={i} run={run} x={x} y={y} color={INK.text} />)}
      </g>
    </ScatterLayer>
    {nOut > 0 && <text x={width} y={-3} textAnchor="end" fontSize={10} fill={INK.muted} pointerEvents="none">{nOut} point{nOut > 1 ? 's' : ''} beyond the range, drawn hollow at the edge</text>}
  </>
}
