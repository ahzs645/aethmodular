import { useMemo } from 'react'
import { ChartFrame, Empty } from '@/components/ChartFrame'
import { Key, type KeyItem } from '@/components/Key'
import { SweepPlot, type SweepMarker, type SweepSeries } from '@/components/SweepPlot'
import { fmt } from '@/lib/format'
import { pearson, splitAtGaps } from '@/lib/stats'
import { INK } from '@/lib/theme'
import { useHighlightActions } from '@/lib/highlight'
import type { FilterRow, MetaFile } from '@/lib/types'
import { finite, instrumentName, localDate, PERIOD_COLOR, PERIOD_LABEL, periodOf, ratio, type Period, type QualityIndex } from './data'
import { AETH_IR } from './FlowFixCompare'


const HIPS = 'HIPS BC', FTIR = 'EC (FTIR)'
const DAY = 864e5

/** a trailing time-window statistic at each point (pandas rolling('30D', min_periods)) */
export function trailing<T>(pts: { t: Date; v: T }[], days: number, minN: number, stat: (vs: T[]) => number | null) {
  return pts.map((p, i) => {
    const vs: T[] = []
    for (let j = i; j >= 0 && +p.t - +pts[j].t < days * DAY; j--) vs.push(pts[j].v)
    return { t: p.t, v: vs.length >= minN ? stat(vs) : null }
  })
}
const mean = (vs: number[]) => vs.reduce((a, b) => a + b, 0) / vs.length
/** a rolling line, broken wherever the filters stop for more than 45 days (no bridge across a sampling gap) */
const lines = (id: string, label: string, color: string, pts: { t: Date; v: number | null }[]): SweepSeries[] =>
  splitAtGaps(pts.filter((p) => p.v !== null), (p) => p.t).map((seg, i) =>
    ({ id: `${id}${i}`, label, color, width: 1.7, marks: false, points: seg.map((p) => ({ x: p.t, y: p.v })) }))

/**
 * When did the methods' relationship change? One site's filters over time: the aethalometer over HIPS, the
 * aethalometer over FTIR EC and HIPS over FTIR EC per filter (coloured by flow period) with their 30-day rolling
 * mean, and the 60-day rolling correlation of the aethalometer with HIPS. The flow-fix boundaries are marked,
 * so a step in a ratio can be read against them (flow_fix_explorer: stacked series, rolling correlation, ratio
 * time series).
 */
export function AgreementOverTime({ code, rows, meta, index }: { code: string; rows: FilterRow[]; meta: MetaFile; index: QualityIndex }) {
  const hl = useHighlightActions()
  const periods = index.flow_periods[code]
  const site = useMemo(() => rows.filter((r) => r.code === code).sort((a, b) => a.date.localeCompare(b.date)), [rows, code])
  const markers: SweepMarker[] = [
    ...(periods?.before_end ? [{ at: localDate(periods.before_end), label: 'before ends', color: PERIOD_COLOR.before, labelSide: 'left' as const }] : []),
    ...(periods?.after_start ? [{ at: localDate(periods.after_start), label: 'after starts', color: PERIOD_COLOR.after }] : []),
  ]
  const xDomain = site.length ? [localDate(site[0].date), localDate(site[site.length - 1].date)] as [Date, Date] : undefined
  const ratioSeries = (num: string, den: string): SweepSeries[] => {
    const pts = site.flatMap((r) => {
      const v = ratio(r[num] as number, r[den] as number)
      // the explorer drops denominators under 0.01 µg/m³
      return finite(v) && (r[den] as number) > 0.01 ? [{ r, t: localDate(r.date), v }] : []
    })
    const byPeriod = new Map<Period, typeof pts>()
    for (const p of pts) { const k = periodOf(p.r.date, periods); byPeriod.set(k, [...(byPeriod.get(k) ?? []), p]) }
    const roll = trailing(pts.map((p) => ({ t: p.t, v: p.v })), 30, 3, mean)
    return [
      ...[...byPeriod].map(([k, ps]) => ({ id: k, label: PERIOD_LABEL[k], color: PERIOD_COLOR[k], width: 0, marks: true, r: 2.6,
        points: ps.map((p) => ({ x: p.t, y: p.v, id: p.r.id, tip: [p.r.id, p.r.date, `${num} / ${den}: ${fmt(p.v, 2)}`, PERIOD_LABEL[k]] })) })),
      ...lines('roll', '30-day rolling mean', INK.text, roll),
    ]
  }
  const corr = useMemo(() => {
    const pts = site.filter((r) => finite(r[AETH_IR] as number) && finite(r[HIPS] as number)).map((r) => ({ t: localDate(r.date), v: [r[AETH_IR] as number, r[HIPS] as number] as [number, number] }))
    return trailing(pts, 60, 5, (vs) => pearson(vs.map((p) => p[0]), vs.map((p) => p[1])))
  }, [site])
  const panels: { key: string; y: string; series: SweepSeries[]; reference?: number; yDomain?: [number, number]; markers?: { at: number; label: string }[] }[] = [
    { key: 'aeth-hips', y: 'IR BCc / HIPS BC', series: ratioSeries(AETH_IR, HIPS), reference: 1 },
    { key: 'aeth-ftir', y: 'IR BCc / FTIR EC', series: ratioSeries(AETH_IR, FTIR), reference: 1 },
    { key: 'hips-ftir', y: 'HIPS BC / FTIR EC', series: ratioSeries(HIPS, FTIR), reference: 1 },
    { key: 'r', y: '60-day r with HIPS', yDomain: [-0.2, 1.05], markers: [{ at: 0.8, label: 'r = 0.8' }, { at: 0.5, label: 'r = 0.5' }],
      series: lines('r', '60-day rolling r', '#2ca02c', corr) },
  ]
  const shown = panels.filter((p) => p.series.some((s) => s.points.some((q) => finite(q.y))))
  return (
    <ChartFrame id="aq-agreement-time" title={`${instrumentName(code, meta)} · agreement over time — when did the relationship change?`}
      subtitle="Per filter: the aethalometer's IR BCc over HIPS BC (HIPS Fabs / MAC), over FTIR EC, and HIPS over FTIR EC, coloured by flow period, with a trailing 30-day mean (at least 3 filters); and the trailing 60-day correlation of IR BCc with HIPS (at least 5 filters). Denominators under 0.01 µg/m³ are left out, as in the explorer. The dashed verticals are the flow-fix boundaries; the y range is the bulk of the ratios, so a few extreme ones sit hollow at the edge."
      provenance="flow_fix_explorer (plot_stacked_timeseries, plot_rolling_correlation, plot_ratio_timeseries) · FTIR EC is a spectral prediction">
      {!site.length ? <Empty>No filters from {instrumentName(code, meta)} in the current subset.</Empty> : <>
        {shown.map((p) => <SweepPlot key={p.key} series={p.series} onPoint={(_, pt) => pt.id && hl.openSample(pt.id)} xScale="time" xDomain={xDomain} yLabel={p.y} reference={p.reference} yDomain={p.yDomain}
          yClip={p.yDomain ? undefined : [0.02, 0.98]} xMarkers={markers} yMarkers={p.markers} height={170} />)}
        <Key items={[...new Set(site.map((r) => periodOf(r.date, periods)))].map((k): KeyItem => ({ label: PERIOD_LABEL[k], color: PERIOD_COLOR[k] }))
          .concat([{ label: '30-day rolling mean', color: INK.text, kind: 'line' as const }, { label: '60-day rolling r', color: '#2ca02c', kind: 'line' as const }])} />
      </>}
    </ChartFrame>
  )
}
