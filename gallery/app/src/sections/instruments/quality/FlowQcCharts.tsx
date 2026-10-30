/**
 * The figures of notebooks/qc/flow_qc_analysis (and its saved all-sites and per-site flow-ratio figures in
 * notebooks/qc/figures) that the minute flow QC panel (MinuteCharts.MinuteFlowQC) does not already draw: the daily
 * flow ratio at every instrument, one instrument's record with its distribution, and on the two minute records the
 * hourly ratio, the flow fractions, the flow differences, the ratio's box plot and density, and Flow1 against Flow2
 * by QC class. Everything is drawn from the exports the page already loads (aethalometer/quality/*.json and
 * minute/*.json) plus minute/<CODE>.flow.json (export_aeth_quality_minute.flow_detail); the rolling windows, the
 * box plots, the KDE and the classes are computed here (flowQc.ts).
 */
import { useMemo, useRef } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Segmented } from '@/components/ChartFrame'
import { XAxis, YAxis } from '@/components/Axes'
import { FacetGrid } from '@/components/FacetGrid'
import { IntervalStrip, type IntervalBlock } from '@/components/IntervalStrip'
import { Key, type KeyItem } from '@/components/Key'
import { SweepPlot, type SweepPoint, type SweepSeries } from '@/components/SweepPlot'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { useExport } from '@/lib/data'
import { fmt, pct } from '@/lib/format'
import { FONT, INK } from '@/lib/theme'
import { useSearchParam } from '@/lib/url'
import type { MetaFile } from '@/lib/types'
import { finite, instrumentColor, instrumentName, localDate, ratio, type Day } from './data'
import { breakGaps, dailyGaps, expandHours, kdeFromHist, quantileLinear, ratioClass, rollingCentred, tukeyBox, unionDays, type DayValues, type RatioClass, type TukeyBox } from './flowQc'
import type { MinuteFlow, MinuteSite } from './minute'
import { NoMinute } from './MinuteCharts'

const NOTEBOOK = 'notebooks/qc/flow_qc_analysis'
const MINUTE_SOURCE = 'gallery/data/export_aeth_quality_minute.py (flow_qc, flow_detail) · Drive "Aethalometry Data/Raw" MA350 1-min CSVs'
const DAILY_SOURCE = 'gallery/data/export_aeth_quality.py · processed_sites/df_<Site>_9am_resampled.pkl (daily mean Flow1, Flow2)'
/** the notebook's nominal split, 75/25: Flow1/Flow2 = 3 */
const NOMINAL_R = 3
/** a daily line breaks across a gap of more than this many days */
const MAX_GAP_DAYS = 7
const CLASS_COLOR: Record<RatioClass, () => string> = { pass: () => INK.good, warn: () => '#e69f00', fail: () => INK.negative }
const CLASS_LABEL: Record<RatioClass, string> = { pass: 'pass (within 5 %)', warn: 'warn (5–10 %)', fail: 'fail (beyond 10 %)' }

// ------------------------------------------------------------------------------------------ the daily records

/** one daily flow-ratio record: an instrument's daily bins, its minute record, or both */
export interface FlowRecord { key: string; code: string; label: string; color: string; dash?: string; days: DayValues; source: string; box: TukeyBox | null }
interface QualityFile { start: string[]; series: Record<string, (number | null)[]> }

/** Flow1/Flow2 of each daily bin's mean flows, by the bin's 9am day */
const binRatios = (days: readonly Day[]): DayValues =>
  new Map(days.map((d) => [d.start.slice(0, 10), ratio(d.q('flow1'), d.q('flow2'))] as const).filter((e): e is [string, number] => finite(e[1])))
const fileRatios = (q: QualityFile | null): DayValues =>
  new Map((q?.start ?? []).map((s, i) => [s.slice(0, 10), ratio(q!.series.flow1?.[i], q!.series.flow2?.[i])] as const).filter((e): e is [string, number] => finite(e[1])))
const minuteRatios = (f: MinuteFlow | null): DayValues =>
  new Map((f?.daily9.day ?? []).map((d, i) => [d, f!.daily9.R_med[i]] as const).filter((e): e is [string, number] => finite(e[1])))

/**
 * The daily flow ratio of every record, as the saved all-sites figures group them. 'instrument': one per instrument,
 * Addis from its fuller Jacros file (the same MA350-0238; R is identical on shared days) and Pasadena as the union
 * of the JPL pickle and the minute record (MA350-0229; the minute record's 9am-day median wherever both have the
 * day). 'record': the six records as filed — the Addis and Jacros files, the JPL pickle, Beijing, Delhi and the
 * Pasadena minute record. Each list is sorted by median, as the figures are.
 */
export function useFlowRecords(all: Record<string, Day[]>, meta: MetaFile, uspaFlow: MinuteFlow | null) {
  const addisFile = useExport<QualityFile>('aethalometer/quality/ETAD.json').data
  return useMemo(() => {
    const name = (c: string) => instrumentName(c, meta)
    const rec = (key: string, code: string, label: string, days: DayValues, source: string, dash?: string): FlowRecord =>
      ({ key, code, label, color: instrumentColor(code, meta), dash, days, source, box: tukeyBox([...days.values()]) })
    const pkl = (c: string) => binRatios(all[c] ?? [])
    const minute = minuteRatios(uspaFlow)
    const byMedian = (a: FlowRecord, b: FlowRecord) => (a.box?.median ?? Infinity) - (b.box?.median ?? Infinity)
    const instrument = [
      rec('ETAD', 'ETAD', `${name('ETAD')} (Jacros)`, pkl('ETAD'), 'df_Jacros_9am_resampled.pkl, daily mean flows'),
      rec('CHTS', 'CHTS', name('CHTS'), pkl('CHTS'), 'daily mean flows of the pickle'),
      rec('INDH', 'INDH', name('INDH'), pkl('INDH'), 'daily mean flows of the pickle'),
      rec('USPA', 'USPA', 'Pasadena (JPL)', unionDays(minute, pkl('USPA')), 'the minute record\'s 9am-day median R, else the JPL pickle\'s daily mean flows'),
    ].filter((r) => r.days.size).sort(byMedian)
    const record = [
      rec('ETAD-file', 'ETAD', `${name('ETAD')} file`, fileRatios(addisFile), 'df_Addis_Ababa_9am_resampled.pkl'),
      rec('ETAD-JACROS', 'ETAD', 'Jacros file', pkl('ETAD'), 'df_Jacros_9am_resampled.pkl', '5 3'),
      rec('USPA-pkl', 'USPA', 'JPL pickle', pkl('USPA'), 'df_JPL_9am_resampled.pkl'),
      rec('CHTS', 'CHTS', name('CHTS'), pkl('CHTS'), 'df_Beijing_9am_resampled.pkl'),
      rec('INDH', 'INDH', name('INDH'), pkl('INDH'), 'df_Delhi_9am_resampled.pkl'),
      rec('USPA-minute', 'USPA', 'Pasadena minute record', minute, 'Pasadena_MA350_1-min, median R per 9am day', '5 3'),
    ].filter((r) => r.days.size).sort(byMedian)
    return { instrument, record }
  }, [all, meta, uspaFlow, addisFile])
}

const span = (r: FlowRecord) => { const k = [...r.days.keys()]; return `${k[0]} → ${k[k.length - 1]}` }
const boxTip = (r: FlowRecord) => r.box ? [r.label, `${r.box.n} days, ${span(r)}`, `median ${fmt(r.box.median, 3)}, IQR ${fmt(r.box.q25, 3)}–${fmt(r.box.q75, 3)}`,
  `whiskers ${fmt(r.box.lo, 3)}–${fmt(r.box.hi, 3)} (${r.box.below + r.box.above} days beyond)`, `mean ${fmt(r.box.mean, 3)}`, r.source] : [r.label]

/** all_sites_flow_ratio(_4sites, _boxplot, _timeseries).png: the daily flow ratio by site, box plots and over time */
export function FlowRatioSites({ records }: { records: { instrument: FlowRecord[]; record: FlowRecord[] } }) {
  const [group, setGroup] = useSearchParam<'instrument' | 'record'>('aq.frs', 'instrument', ['instrument', 'record'])
  const list = records[group]
  const blocks = useMemo<IntervalBlock[]>(() => [{
    key: 'all', rows: list.filter((r) => r.box).map((r) => ({
      key: r.key, label: r.label, labelColor: r.color, note: `n=${r.box!.n}d`, tip: boxTip(r),
      marks: [[{ mid: null, lo: r.box!.lo, hi: r.box!.hi, color: r.color, faint: true }, { mid: r.box!.median, lo: r.box!.q25, hi: r.box!.q75, color: r.color }]],
    })),
  }], [list])
  const series = useMemo<SweepSeries[]>(() => list.map((r) => ({
    id: r.key, label: `${r.label} (median ${fmt(r.box?.median, 2)})`, color: r.color, dash: r.dash, width: 1.2, marks: false, gaps: true,
    points: breakGaps(r.days, MAX_GAP_DAYS).map((p) => ({ ...p, tip: p.y == null ? undefined : [r.label, d3.utcFormat('%Y-%m-%d')(p.x), `R ${fmt(p.y, 3)}`] })),
  })), [list])
  return (
    <ChartFrame id="aq-flow-ratio-sites" title="Flow ratio by site — the daily Flow1/Flow2 at every instrument, as a box plot and over time"
      subtitle={`R = Flow1/Flow2 per 9am day: from the daily pickles, the ratio of the day's mean flows; from the Pasadena minute record, the day's median minute ratio (the two agree on the days both have). ${group === 'instrument' ? 'One record per instrument: Addis is its Jacros file (the same MA350-0238 with the longer record), Pasadena (JPL) the union of the JPL pickle and the minute record (MA350-0229), the minute value where both have a day.' : 'The six records as filed: the Addis and Jacros files of MA350-0238 (identical where they overlap), the JPL pickle and the Pasadena minute record of MA350-0229, Beijing and Delhi.'} Box: the interquartile range and median; whiskers to the furthest day within 1.5 IQR (no fliers). Dashed: the nominal R = ${NOMINAL_R} of a 75/25 split. Lines break across gaps of more than ${MAX_GAP_DAYS} days.`}
      provenance={`${NOTEBOOK} figures all_sites_flow_ratio, all_sites_flow_ratio_4sites, _boxplot and _timeseries · ${DAILY_SOURCE} · ${MINUTE_SOURCE}`}
      tip={<>No instrument runs near the nominal 3.0; each sits at its own split, and JPL/Pasadena and Delhi move a long way within their records. The saved 4-site figure counts 629 days at Pasadena (JPL); this union has 628 (the minute record's 9am days are taken from its local clock).</>}
      controls={<Segmented label="Records" value={group} options={['instrument', 'record'] as const} onChange={setGroup} optionLabel={(g) => (g === 'instrument' ? 'One per instrument' : 'Six records')} />}>
      {!list.length ? <Empty>No flow columns in the daily bins.</Empty> : <div className="chart-wrap">
        <IntervalStrip blocks={blocks} labelWidth={190} noteWidth={60} panels={[{ label: 'Daily R = Flow1 / Flow2', marker: { value: NOMINAL_R, label: `nominal ${NOMINAL_R}.0` }, format: (v) => v.toFixed(1) }]} />
        <SweepPlot series={series} xScale="time" height={300} yLabel="R (Flow1 / Flow2)" xLabel="9am day" yMarkers={[{ at: NOMINAL_R, label: `nominal ${NOMINAL_R}.0`, color: INK.muted }]} />
        <Key items={list.map((r) => ({ label: `${r.label} (median ${fmt(r.box?.median, 2)}, ${r.box?.n ?? 0} d)`, color: r.color, kind: r.dash ? 'dash' : 'line' }))} />
      </div>}
    </ChartFrame>
  )
}

/** site_<site>_flow_ratio.png: one instrument's daily R with its IQR band and median, and the distribution of its days */
export function FlowRatioSite({ code, records, meta }: { code: string; records: FlowRecord[]; meta: MetaFile }) {
  const r = records.find((x) => x.code === code) ?? null
  const ref = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(ref)
  const height = 300, histW = width < 480 ? 96 : width < 720 ? 130 : 180, m = { top: 12, bottom: 46 }
  const vals = r ? [...r.days.values()] : []
  const dom: [number, number] = vals.length ? (() => { const lo = d3.min(vals)!, hi = d3.max(vals)!, pad = 0.03 * (hi - lo || 1); return [lo - pad, hi + pad] })() : [0, 1]
  const y = d3.scaleLinear().domain(dom).range([height - m.top - m.bottom, 0])
  const bins = d3.bin().domain(dom).thresholds(d3.ticks(dom[0], dom[1], 40))(vals)
  const x = d3.scaleLinear().domain([0, d3.max(bins, (b) => b.length) ?? 1]).nice().range([0, histW - 44])
  const tip = useTooltip(ref)
  const b = r?.box
  return (
    <ChartFrame id="aq-flow-ratio-site" title={`${r?.label ?? instrumentName(code, meta)} · flow ratio over the record — daily R, its interquartile range and distribution`}
      subtitle={r && b ? `${b.n} days, ${span(r)}. R = Flow1/Flow2 per 9am day (${r.source}); the shaded band is the interquartile range (${fmt(b.q25, 2)}–${fmt(b.q75, 2)}), the dashed line the median (${fmt(b.median, 2)}). Right: the days by R on the same axis. Lines break across gaps of more than ${MAX_GAP_DAYS} days.` : undefined}
      provenance={`${NOTEBOOK} figures site_addis_ababa, site_beijing, site_delhi and site_pasadena_flow_ratio · ${DAILY_SOURCE}${code === 'USPA' ? ` · ${MINUTE_SOURCE}` : ''}`}>
      {!r || !b ? <Empty>No daily flow ratio for this instrument.</Empty> : <div ref={ref} style={{ display: 'flex', alignItems: 'flex-start', position: 'relative' }}>
        <div style={{ flex: 1, minWidth: 0 }}>
          <SweepPlot height={height} minWidth={150} xScale="time" yDomain={dom} yLabel="R (Flow1 / Flow2)" xLabel="9am day"
            series={[{ id: r.key, label: r.label, color: r.color, width: 1.2, marks: false, gaps: true, points: breakGaps(r.days, MAX_GAP_DAYS) }]}
            yMarkers={[{ at: b.median, label: `median ${fmt(b.median, 2)}`, color: INK.text }]}
            background={({ y: sy, iw }) => <rect x={0} width={iw} y={sy(b.q75)} height={Math.max(0, sy(b.q25) - sy(b.q75))} fill={r.color} fillOpacity={0.14} />} />
        </div>
        <svg width={histW} height={height} fontFamily={FONT.family} style={{ flex: 'none' }}>
          <g transform={`translate(8,${m.top})`}>
            <rect y={y(b.q75)} width={histW - 44} height={Math.max(0, y(b.q25) - y(b.q75))} fill={r.color} fillOpacity={0.1} />
            {bins.map((bn, i) => bn.length > 0 && <rect key={i} x={0} y={y(bn.x1!)} width={x(bn.length)} height={Math.max(0.5, y(bn.x0!) - y(bn.x1!) - 0.6)} fill={r.color} fillOpacity={0.8}
              onMouseMove={(e) => tip.show(e, [`R ${fmt(bn.x0, 3)}–${fmt(bn.x1, 3)}`, `${bn.length} days`])} onMouseLeave={tip.hide} />)}
            <line x1={0} x2={histW - 44} y1={y(b.median)} y2={y(b.median)} stroke={INK.text} strokeDasharray="4 3" />
            <XAxis scale={x} y={height - m.top - m.bottom} tickCount={3} label="Days" labelDy={38} />
          </g>
        </svg>
        {tip.node}
      </div>}
    </ChartFrame>
  )
}

// ------------------------------------------------------------------------------------------ the minute records

/** one point per stored hour, with a break (null) wherever a run of consecutive hours ends */
function hourlyPoints(hours: Date[], runs: [number, number][], values: (number | null)[], scale = 1): SweepPoint[] {
  const out: SweepPoint[] = []
  let i = 0
  for (const [, len] of runs) {
    if (i > 0) out.push({ x: new Date(+hours[i - 1] + 1800e3), y: null })
    for (let k = 0; k < len; k++, i++) out.push({ x: hours[i], y: finite(values[i]) ? values[i]! * scale : null })
  }
  return out
}
const bandRect = (sy: d3.ScaleContinuousNumeric<number, number>, iw: number, lo: number, hi: number, color: string, key: string, opacity = 0.14) =>
  <rect key={key} x={0} width={iw} y={sy(hi)} height={Math.max(0, sy(lo) - sy(hi))} fill={color} fillOpacity={opacity} />

/** jacros_/pasadena_ratio_timeseries.png: the hourly median R with its 24-hour rolling median, and the hourly pass rate */
export function MinuteFlowHourly({ flow, name }: { flow: MinuteFlow | null; name: string }) {
  const d = useMemo(() => {
    if (!flow) return null
    const h = flow.hourly, hours = expandHours(h), T = flow.R_target
    const roll = rollingCentred(h.R, 24, 'median'), pass7 = rollingCentred(h.pass, 24 * 7, 'mean')
    const lo = Math.max(quantileLinear(h.R, 0.01) ?? T * 0.8, T * 0.8), hi = Math.min(quantileLinear(h.R, 0.99) ?? T * 1.2, T * 1.2)
    return { hours, T, dom: [lo, hi] as [number, number], R: hourlyPoints(hours, h.runs, h.R), roll: hourlyPoints(hours, h.runs, roll),
      pass: hourlyPoints(hours, h.runs, h.pass).map((p) => ({ ...p, lo: finite(p.y) ? 0 : null, hi: p.y })), pass7: hourlyPoints(hours, h.runs, pass7),
      n: h.R.filter(finite).length, passAll: d3.mean(h.pass.filter(finite)) }
  }, [flow])
  return (
    <ChartFrame id="aq-min-flow-hourly" title={`${name} · flow ratio by the hour — hourly median R, its 24-hour rolling median and the pass rate`}
      subtitle={d ? `${d.n.toLocaleString()} hours with flows. Top: the median minute R = Flow1/Flow2 of each hour and its centred 24-hour rolling median (over 24 consecutive hourly rows, as pandas rolls them), with the notebook's bands around the instrument's own median R ${fmt(d.T, 3)}: ±5 % passes (green), 5–10 % warns (orange). The y range is the notebook's: the 1–99 % of hours, within ±20 % of the median. Bottom: the share of each hour's minutes whose R passes, and its centred 7-day (168-hour) rolling mean. Lines break where hours are missing.` : undefined}
      provenance={`${NOTEBOOK} §3 plot_ratio_timeseries_improved (create_hourly_average) · ${MINUTE_SOURCE}`}>
      {!d ? <NoMinute name={name} /> : <>
        <SweepPlot height={260} xScale="time" yLabel="R (Flow1 / Flow2)" yDomain={d.dom}
          yMarkers={[{ at: d.T, label: `median ${fmt(d.T, 3)}`, color: INK.text }]}
          background={({ y, iw }) => [bandRect(y, iw, d.T * 0.95, d.T * 1.05, INK.good, 'p'), bandRect(y, iw, d.T * 0.9, d.T * 0.95, '#e69f00', 'w1'), bandRect(y, iw, d.T * 1.05, d.T * 1.1, '#e69f00', 'w2')]}
          series={[
            { id: 'h', label: 'hourly median R', color: INK.accent, width: 0.7, marks: false, gaps: true, points: d.R },
            { id: 'r', label: '24-hour rolling median', color: INK.negative, width: 1.4, marks: false, gaps: true, points: d.roll }]} />
        <SweepPlot height={170} xScale="time" yLabel="Minutes passing (%)" yDomain={[0, 105]}
          series={[
            { id: 'p', label: 'hourly pass rate', color: INK.good, width: 0, marks: false, gaps: true, fill: 0.22, points: d.pass },
            { id: 'p7', label: '7-day rolling mean', color: INK.accent, width: 1.6, marks: false, gaps: true, points: d.pass7 }]} />
        <Key items={[{ label: 'hourly median R', color: INK.accent, kind: 'line' }, { label: '24-hour rolling median', color: INK.negative, kind: 'line' },
          { label: '±5 % pass band', color: INK.good, kind: 'square' }, { label: '5–10 % warn band', color: '#e69f00', kind: 'square' },
          { label: `hourly pass rate (mean ${fmt(d.passAll, 0)} %)`, color: INK.good, kind: 'square' }, { label: 'pass rate, 7-day rolling mean', color: INK.accent, kind: 'line' }]} />
      </>}
    </ChartFrame>
  )
}

/** jacros_/pasadena_flow_fractions.png: f1 = Flow1/total and f2 = Flow2/total by the hour, against the 0.75/0.25 targets */
export function MinuteFlowFractions({ flow, name }: { flow: MinuteFlow | null; name: string }) {
  const d = useMemo(() => {
    if (!flow) return null
    const h = flow.hourly, hours = expandHours(h)
    const f2 = h.f1.map((v) => (finite(v) ? 1 - v : null))
    return { f1: hourlyPoints(hours, h.runs, h.f1), f1r: hourlyPoints(hours, h.runs, rollingCentred(h.f1, 24 * 7, 'median')),
      f2: hourlyPoints(hours, h.runs, f2), f2r: hourlyPoints(hours, h.runs, rollingCentred(f2, 24 * 7, 'median')),
      m1: quantileLinear(h.f1, 0.5) }
  }, [flow])
  const panel = (k: 'f1' | 'f2') => {
    const target = k === 'f1' ? 0.75 : 0.25, bounds: [number, number] = k === 'f1' ? [0.73, 0.77] : [0.23, 0.27]
    const color = k === 'f1' ? INK.accent : INK.good
    return <SweepPlot key={k} height={200} xScale="time" yLabel={k === 'f1' ? 'f1 = Flow1 / total' : 'f2 = Flow2 / total'} include={[bounds[0], bounds[1]]}
      yMarkers={[{ at: target, label: `target ${target}`, color: INK.negative }]}
      background={({ y, iw }) => bandRect(y, iw, bounds[0], bounds[1], INK.good, 'b', 0.1)}
      series={[
        { id: k, label: `${k}, hourly median`, color, width: 0.7, marks: false, gaps: true, points: d![k] },
        { id: `${k}r`, label: '7-day rolling median', color: INK.text, width: 1.5, marks: false, gaps: true, points: d![`${k}r`] }]} />
  }
  return (
    <ChartFrame id="aq-min-flow-fractions" title={`${name} · flow fractions by the hour — f1 and f2 against the 75/25 split`}
      subtitle={d ? `The hourly median of each minute's f1 = Flow1/total and f2 = Flow2/total, with their centred 7-day (168-hour) rolling medians, the notebook's targets (0.75, 0.25) and its acceptable bands (0.73–0.77, 0.23–0.27, shaded). S = (Flow1 + Flow2)/total is 1 on every minute of this record (the table above), so f2 = 1 − f1. The notebook fixed the axes at 0.65–0.85 and 0.15–0.35, which leaves Jacros (f1 ≈ ${fmt(d.m1, 2)} here) off its chart; these axes follow the data and keep the targets in view.` : undefined}
      provenance={`${NOTEBOOK} §3 plot_flow_fractions_improved · ${MINUTE_SOURCE}`}>
      {!d ? <NoMinute name={name} /> : <>
        {panel('f1')}{panel('f2')}
        <Key items={[{ label: 'f1, hourly median', color: INK.accent, kind: 'line' }, { label: 'f2, hourly median', color: INK.good, kind: 'line' },
          { label: '7-day rolling median', color: INK.text, kind: 'line' }, { label: 'target', color: INK.negative, kind: 'dash' }, { label: 'acceptable band', color: INK.good, kind: 'square' }]} />
      </>}
    </ChartFrame>
  )
}

/** jacros_/pasadena_flow_differences.png: (F1 + F2) − total and total − setpoint, as daily envelopes of the minutes */
export function MinuteFlowDiffs({ site, flow, name }: { site: MinuteSite | null; flow: MinuteFlow | null; name: string }) {
  const d = useMemo(() => {
    if (!flow || !site) return null
    const days = flow.diffs.day.map(localDate)
    const nByDay = new Map(site.flow_qc.daily.day.map((k, i) => [k, site.flow_qc.daily.n[i]]))
    const over = (key: 'sum_diff' | 'set_diff') => {
      let a = 0, n = 0
      flow.diffs.day.forEach((k, i) => { const v = flow.diffs[key].over2[i], w = nByDay.get(k) ?? 0; if (finite(v)) { a += v * w; n += w } })
      return n ? a / n : null
    }
    const pts = (key: 'sum_diff' | 'set_diff') => {
      const s = flow.diffs[key]
      return {
        range: dailyGaps(days.map((t, i) => ({ x: t, y: s.p50[i], lo: s.min[i], hi: s.max[i] }))),
        mid: dailyGaps(days.map((t, i) => ({ x: t, y: s.p50[i], lo: s.p0_5[i], hi: s.p99_5[i],
          tip: [d3.utcFormat('%Y-%m-%d')(t), `median ${fmt(s.p50[i], 3)} mL/min`, `0.5–99.5 %: ${fmt(s.p0_5[i], 2)} to ${fmt(s.p99_5[i], 2)}`, `range ${fmt(s.min[i], 2)} to ${fmt(s.max[i], 2)}`, `beyond ±2: ${pct(s.over2[i])}`] }))),
      }
    }
    return { sum: pts('sum_diff'), set: pts('set_diff'), overSum: over('sum_diff'), overSet: over('set_diff') }
  }, [site, flow])
  const panel = (k: 'sum' | 'set') => (
    <SweepPlot key={k} height={200} xScale="time" include={[-2.2, 2.2]} reference={0}
      yLabel={k === 'sum' ? '(Flow1 + Flow2) − total' : 'Total − setpoint'} yMarkers={[{ at: 2, label: '+2 mL/min', color: INK.negative }, { at: -2, label: '−2 mL/min', color: INK.negative }]}
      series={[
        { id: 'r', label: 'daily range', color: k === 'sum' ? INK.accent : INK.good, width: 0, marks: false, gaps: true, fill: 0.12, points: d![k].range },
        { id: 'm', label: 'daily median, 0.5–99.5 %', color: k === 'sum' ? INK.accent : INK.good, width: 1.2, marks: false, gaps: true, fill: 0.35, points: d![k].mid }]} />
  )
  return (
    <ChartFrame id="aq-min-flow-diffs" title={`${name} · flow consistency — does Flow1 + Flow2 equal the total, and does the total track its setpoint?`}
      subtitle={d ? `Per local day, over every minute with all four flows: the median, the 0.5–99.5 % range (darker) and the full range (pale) of (Flow1 + Flow2) − total and of total − setpoint, in mL/min, with the notebook's ±2 mL/min lines (2 % of 100 mL/min). The notebook drew every minute as a dot; the daily envelope keeps the same spread and its outliers. Minutes beyond ±2 mL/min: ${pct(d.overSum)} of the sum check, ${pct(d.overSet)} of the setpoint check.` : undefined}
      provenance={`${NOTEBOOK} §3 plot_flow_differences · ${MINUTE_SOURCE}`}>
      {!d ? <NoMinute name={name} /> : <>
        {panel('sum')}{panel('set')}
        <Key items={[{ label: 'sum check: daily median, 0.5–99.5 % and full range', color: INK.accent, kind: 'square' }, { label: 'setpoint check: the same', color: INK.good, kind: 'square' }, { label: '±2 mL/min', color: INK.negative, kind: 'dash' }]} />
      </>}
    </ChartFrame>
  )
}

/** jacros_/pasadena_ratio_distribution.png: the density of every minute's R with its KDE, and the box plot */
export function MinuteRDistribution({ site, flow, name }: { site: MinuteSite | null; flow: MinuteFlow | null; name: string }) {
  const [range, setRange] = useSearchParam<'data' | 'nominal'>('aq.rdx', 'data', ['data', 'nominal'])
  const f = site?.flow_qc, b = flow?.R_box
  const d = useMemo(() => {
    if (!f || !b) return null
    const h = f.R_hist, nb = h.counts.length, w = (h.hi - h.lo) / nb
    const n = d3.sum(h.counts) + h.below + h.above
    const lo = range === 'nominal' ? Math.min(h.lo, NOMINAL_R * 0.9 - 0.05) : h.lo, hi = range === 'nominal' ? Math.max(h.hi, NOMINAL_R * 1.1 + 0.05) : h.hi
    const grid = d3.range(301).map((i) => lo + ((hi - lo) * i) / 300)
    const kde = kdeFromHist(h, f.stats.R.std, grid)
    const edges = d3.range(nb + 1).map((i) => h.lo + i * w)
    const bars: SweepPoint[] = h.counts.map((c, i) => {
      const v = c / (n * w)
      return { x: (edges[i] + edges[i + 1]) / 2, y: v, lo: 0, hi: v, tip: [`R ${fmt(edges[i], 3)}–${fmt(edges[i + 1], 3)}`, `${c.toLocaleString()} minutes`, `density ${fmt(v, 2)}`] }
    })
    return { dom: [lo, hi] as [number, number], bars, kde: grid.map((x, i) => ({ x, y: kde[i] })) }
  }, [f, b, range])
  const nominal = range === 'nominal'
  return (
    <ChartFrame id="aq-min-flow-rdist" title={`${name} · distribution of the flow ratio — density, KDE and box plot of every minute's R`}
      subtitle={f && b ? `Every minute's R = Flow1/Flow2 (${b.n.toLocaleString()} minutes) as a density over the exported 80-bin histogram (its 0.5–99.5 % range; the 1 % beyond it counts in the density's total but is not drawn), with a Gaussian KDE (Scott's bandwidth, on the same bins), the median ${fmt(f.stats.R.median, 3)} and the 5th/95th percentiles ${fmt(f.stats.R.p5, 3)}/${fmt(f.stats.R.p95, 3)}. Below, the notebook's box plot: quartiles ${fmt(b.q25, 3)}–${fmt(b.q75, 3)}, whiskers to the furthest minute within 1.5 IQR (${fmt(b.whisker_lo, 3)}–${fmt(b.whisker_hi, 3)}; ${b.below.toLocaleString()} minutes below, ${b.above.toLocaleString()} above), mean ${fmt(b.mean, 3)} (open circle), SD ${fmt(f.stats.R.std, 3)}, IQR ${fmt(b.q75 - b.q25, 3)}.${nominal ? ` The nominal ${NOMINAL_R}.0 and its warn bounds ${NOMINAL_R * 0.9}/${NOMINAL_R * 1.1} are drawn as the notebook drew them.` : ''}` : undefined}
      provenance={`${NOTEBOOK} §4 plot_ratio_distribution · ${MINUTE_SOURCE}`}
      controls={<Segmented label="Range" value={range} options={['data', 'nominal'] as const} onChange={setRange} optionLabel={(o) => (o === 'data' ? 'The data' : `With the nominal ${NOMINAL_R}.0`)} />}>
      {!d || !f || !b ? <NoMinute name={name} /> : <>
        <SweepPlot height={240} xScale="linear" xDomain={d.dom} xLabel="R = Flow1 / Flow2" yLabel="Density" include={[0]}
          xMarkers={[{ at: f.stats.R.median, label: `median ${fmt(f.stats.R.median, 3)}`, color: INK.negative }, { at: f.stats.R.p5, label: '5 %', color: INK.good }, { at: f.stats.R.p95, label: '95 %', color: INK.good },
            ...(nominal ? [{ at: NOMINAL_R, label: `target ${NOMINAL_R}.0`, color: INK.text }, { at: NOMINAL_R * 0.9, label: 'warn', color: '#e69f00' }, { at: NOMINAL_R * 1.1, label: 'warn', color: '#e69f00' }] : [])]}
          series={[
            { id: 'h', label: 'minutes (density)', color: INK.accent, width: 1, marks: false, curve: 'mid-step', fill: 0.45, points: d.bars },
            { id: 'k', label: 'KDE', color: INK.negative, width: 1.8, marks: false, points: d.kde }]} />
        <IntervalStrip labelWidth={62} noteWidth={0} blocks={[{ key: 'box', rows: [{ key: 'R', label: 'R', tip: [name, `quartiles ${fmt(b.q25, 3)}–${fmt(b.q75, 3)}`, `whiskers ${fmt(b.whisker_lo, 3)}–${fmt(b.whisker_hi, 3)}`, `range ${fmt(b.min, 3)}–${fmt(b.max, 3)}`, `mean ${fmt(b.mean, 3)}`],
          marks: [[{ mid: null, lo: b.whisker_lo, hi: b.whisker_hi, color: INK.accent, faint: true }, { mid: b.median, lo: b.q25, hi: b.q75, color: INK.accent }, { mid: b.mean, color: INK.accent, shape: 'hollow', r: 3 }]] }] }]}
          panels={[{ label: 'R = Flow1 / Flow2 (box plot)', domain: d.dom, ...(nominal ? { marker: { value: NOMINAL_R, label: `target ${NOMINAL_R}.0` } } : {}), format: (v) => v.toFixed(2) }]} />
        <Key items={[{ label: 'density of minutes', color: INK.accent, kind: 'square' }, { label: 'KDE', color: INK.negative, kind: 'line' }, { label: 'median and quartiles', color: INK.accent, kind: 'dot' }, { label: 'mean', color: INK.accent, kind: 'hollow' }, { label: 'whiskers (1.5 IQR)', color: INK.muted, kind: 'line' }]} />
      </>}
    </ChartFrame>
  )
}

/** the left panel of jacros_/pasadena_flow_scatter.png: Flow1 against Flow2, coloured by the R class, with iso-ratio lines */
export function MinuteFlowClasses({ site, name }: { site: MinuteSite | null; name: string }) {
  const f = site?.flow_qc
  return (
    <ChartFrame id="aq-min-flow-classes" title={`${name} · Flow1 against Flow2 by QC class — where the passing, warning and failing minutes lie`}
      subtitle={f ? `The 2-D histogram of the flow QC panel above, each cell coloured by the class of the ratio at its centre (R within 5 % of the median ${fmt(f.R_target, 3)} passes, 5–10 % warns, beyond fails) and shaded by its count (log). Lines: R at the median (solid), ±5 % (dashed) and ±10 % (dotted); the grey cross marks the flows a 100 mL/min total would split into at the median ratio. A cell that straddles a class boundary takes its centre's class; the shares in the key are of the minutes themselves.` : undefined}
      provenance={`${NOTEBOOK} §5 plot_flow_scatter_improved (scatter coloured by flag, iso-ratio lines, expected flows, statistics box) · ${MINUTE_SOURCE}`}>
      {!f ? <NoMinute name={name} /> : <>
        <FacetGrid items={[f]} minCell={320} maxCols={1} render={(q, cell) => <ClassHeat f={q} width={cell} />} />
        <Key items={(['pass', 'warn', 'fail'] as RatioClass[]).map((c): KeyItem => ({ label: `${CLASS_LABEL[c]}: ${pct(f.rates[c === 'pass' ? 'R_pass' : c === 'warn' ? 'R_warn' : 'R_fail'])} of minutes`, color: CLASS_COLOR[c](), kind: 'square' }))} />
      </>}
    </ChartFrame>
  )
}

function ClassHeat({ f, width }: { f: MinuteSite['flow_qc']; width: number }) {
  const ref = useRef<HTMLDivElement>(null)
  const tip = useTooltip(ref)
  const h = f.flows_2d, T = f.R_target
  const m = { top: 8, right: 12, bottom: 42, left: 52 }
  const side = Math.max(200, Math.min(420, width - m.left - m.right))
  const nx = h.counts[0]?.length ?? 1, ny = h.counts.length
  const x = d3.scaleLinear().domain(h.x).range([0, side]), y = d3.scaleLinear().domain(h.y).range([side, 0])
  const max = d3.max(h.counts.flat()) ?? 1
  const shade = d3.scaleLog().domain([1, Math.max(2, max)]).range([0.25, 1]).clamp(true)
  const cw = side / nx, ch = side / ny, dx = (h.x[1] - h.x[0]) / nx, dy = (h.y[1] - h.y[0]) / ny
  const e2 = 100 / (1 + T), e1 = (100 * T) / (1 + T)
  const lines = [{ k: T, color: INK.accent, dash: undefined, w: 2 }, { k: T * 0.95, color: INK.good, dash: '6 3', w: 1.4 }, { k: T * 1.05, color: INK.good, dash: '6 3', w: 1.4 },
    { k: T * 0.9, color: '#e69f00', dash: '2 3', w: 1.4 }, { k: T * 1.1, color: '#e69f00', dash: '2 3', w: 1.4 }]
  const clip = `aq-class-${Math.round(T * 1000)}`
  return (
    <div ref={ref} style={{ position: 'relative' }}>
      <svg width={side + m.left + m.right} height={side + m.top + m.bottom} fontFamily={FONT.family}>
        <g transform={`translate(${m.left},${m.top})`}>
          <defs><clipPath id={clip}><rect width={side} height={side} /></clipPath></defs>
          <rect width={side} height={side} fill={INK.empty} />
          {h.counts.map((row, j) => row.map((c, i) => {
            if (c <= 0) return null
            const cx = h.x[0] + (i + 0.5) * dx, cy = h.y[0] + (j + 0.5) * dy, cls = ratioClass(cy / cx, T)
            return <rect key={`${i}-${j}`} x={i * cw} y={side - (j + 1) * ch} width={cw + 0.3} height={ch + 0.3} fill={CLASS_COLOR[cls]()} fillOpacity={shade(c)}
              onMouseMove={(e) => tip.show(e, [`Flow2 ${fmt(h.x[0] + i * dx, 2)}–${fmt(h.x[0] + (i + 1) * dx, 2)} mL/min`, `Flow1 ${fmt(h.y[0] + j * dy, 2)}–${fmt(h.y[0] + (j + 1) * dy, 2)} mL/min`,
                `R at the centre ${fmt(cy / cx, 3)}: ${CLASS_LABEL[cls]}`, `${c.toLocaleString()} minutes`])} onMouseLeave={tip.hide} />
          }))}
          <g clipPath={`url(#${clip})`} pointerEvents="none">
            {lines.map((l, k) => <line key={k} x1={x(h.x[0])} y1={y(l.k * h.x[0])} x2={x(h.x[1])} y2={y(l.k * h.x[1])} stroke={l.color} strokeWidth={l.w} strokeDasharray={l.dash} />)}
            <line x1={x(e2)} x2={x(e2)} y1={0} y2={side} stroke={INK.muted} strokeDasharray="1 3" />
            <line x1={0} x2={side} y1={y(e1)} y2={y(e1)} stroke={INK.muted} strokeDasharray="1 3" />
          </g>
          <g transform={`translate(${side - 6},${side - 8})`} fontSize={11} fill={INK.text} textAnchor="end" pointerEvents="none">
            <text y={-28}>{`r(Flow1, Flow2) = ${fmt(f.corr_F1_F2, 3)}`}</text>
            <text y={-14}>{`${f.n.toLocaleString()} minutes`}</text>
            <text>{`R median ${fmt(T, 3)}; expected ${fmt(e2, 1)} / ${fmt(e1, 1)} mL/min`}</text>
          </g>
          <YAxis scale={y} x={0} tickCount={5} label="Flow1 (mL/min)" labelX={-40} />
          <XAxis scale={x} y={side} tickCount={5} label="Flow2 (mL/min)" labelDy={34} />
        </g>
      </svg>
      {tip.node}
    </div>
  )
}
