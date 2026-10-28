import { useMemo } from 'react'
import { ChartFrame, Empty } from '@/components/ChartFrame'
import { IntervalStrip, type IntervalBlock } from '@/components/IntervalStrip'
import { fmt } from '@/lib/format'
import { INK } from '@/lib/theme'
import type { MetaFile } from '@/lib/types'
import { DailyPanels, qLine, type PanelSpec } from './DailyPanels'
import { finite, instrumentColor, instrumentName, PERIOD_COLOR, PERIOD_LABEL, quantile, ratio, type Day, type Period, type QualityIndex } from './data'

/** Flow1/Flow2 lines drawn by research/ftir_hips_chem/scripts/plotting/timeseries.flow_ratio */
const FLOW_RATIO_LINES = [{ at: 1, label: 'ideal 1.0' }, { at: 2, label: 'degraded 2.0' }]
const flowRatio = (d: Day) => ratio(d.q('flow1'), d.q('flow2'))
const setpointRatio = (d: Day) => ratio(d.q('flowTotal'), d.q('flowSetpoint'))

/**
 * One instrument's day by day: IR BCc coloured by flow period (FlowFix_BeforeAfter_Analysis §5), the spot-flow
 * ratio Flow1/Flow2 with the plotting module's ideal and degraded lines (flow_qc_analysis's R, at daily
 * resolution), the two spot flows, total flow over its setpoint with the minute cleaning's ±10 % band, and the
 * sample humidity and internal temperature the dual-spot diagnostics bin by (ftir_49 A3).
 */
export function FlowTimeline({ code, days, meta, index }: { code: string; days: Day[]; meta: MetaFile; index: QualityIndex }) {
  const periods = index.flow_periods[code]
  const band = index.cleaning_bounds.flow_range.flow_threshold
  const panels: PanelSpec[] = [
    { key: 'ir', title: 'IR BCc (µg/m³)', byPeriod: !!periods?.before_end, lines: [{ key: 'ir', label: 'IR BCc', color: instrumentColor(code, meta), value: (d) => d.bc.IR, digits: 2 }] },
    { key: 'ratio', title: 'Flow1 / Flow2', refs: FLOW_RATIO_LINES, lines: [{ key: 'ratio', label: 'Flow1 / Flow2', color: INK.text, value: flowRatio }] },
    { key: 'spots', title: 'Spot flow (mL/min)', lines: [qLine('flow1', 'Flow1', '#2c7fb8', { digits: 1 }), qLine('flow2', 'Flow2', '#d95f0e', { digits: 1 })] },
    { key: 'total', title: 'Total / setpoint', refs: [{ at: 1 - band, label: `−${Math.round(band * 100)} %` }, { at: 1 + band, label: `+${Math.round(band * 100)} %` }],
      lines: [{ key: 'total', label: 'Total flow / setpoint', color: '#41ab5d', value: setpointRatio, digits: 4 }] },
    { key: 'rh', title: 'Sample RH (%)', lines: [qLine('rhSample', 'Sample RH', '#6a51a3', { digits: 1 })] },
    { key: 'temp', title: 'Temperature (°C)', lines: [qLine('tInternal', 'Internal temperature', '#8c2d04', { digits: 2 }), qLine('tSample', 'Sample temperature', '#8c2d04', { dash: '4 3', digits: 2 })] },
  ]
  const shown = days.filter((d) => finite(flowRatio(d)))
  return (
    <ChartFrame id="aq-flow-timeline" title={`${instrumentName(code, meta)} · flow and instrument timeline — the flow fix, day by day`}
      subtitle={`Daily means of the minute record: IR BCc (coloured by flow period where config.FLOW_FIX_PERIODS sets one), the spot-flow ratio Flow1/Flow2 with the lines plotting/timeseries.flow_ratio draws (1.0 ideal, 2.0 degraded), the two spot flows, total flow over its setpoint with the ±${Math.round(band * 100)} % band the minute cleaning keeps (src/data/qc/pkl_cleaning.py), sample RH and the instrument temperatures. ${periods ? `Flow-fix periods: ${periods.description.toLowerCase()} (${periods.notes.toLowerCase()}).` : 'No flow-fix period is defined for this instrument.'} A line breaks wherever a day is missing.`}
      provenance="FlowFix_BeforeAfter_Analysis §5, flow_fix_explorer, notebooks/qc/flow_qc_analysis (R, S, T at daily resolution), plotting_gaps_scenarios scenarios 2–3 · processed_sites/*.pkl"
      tip={<>The nominal flow split is not agreed in the sources: flow_qc_analysis targets Flow1/Flow2 = 3.0 (a 75/25 split) and finds medians of 1.2–2.5; plotting/timeseries.flow_ratio draws 1.0 as ideal and 2.0 as degraded; the minute cleaning keeps ratios between {index.cleaning_bounds.flow_ratio.lower} and {index.cleaning_bounds.flow_ratio.upper}. The chart shows the lines and leaves the verdict open. Ratios are of daily mean flows.</>}>
      {shown.length ? <DailyPanels days={days} panels={panels} periods={periods} panelHeight={84} /> : <Empty>No flow record in this instrument's daily bins.</Empty>}
    </ChartFrame>
  )
}

/**
 * Every instrument's spot-flow ratio, total flow and UV/IR BCc ratio by flow period: median and 10–90 % of the
 * daily values (FlowFix_BeforeAfter_Analysis §11's UV/IR comparison and the flow-ratio notes, on one strip).
 */
export function FlowSummary({ all, meta }: { all: Record<string, Day[]>; meta: MetaFile }) {
  const blocks = useMemo<IntervalBlock[]>(() => Object.entries(all).map(([code, days]) => {
    const order: Period[] = ['before', 'gap', 'after', 'no_fix']
    const groups = new Map<string, Day[]>()
    for (const d of days) {
      const k = d.flow ?? 'all'
      groups.set(k, [...(groups.get(k) ?? []), d])
    }
    const keys = [...groups.keys()].sort((a, b) => order.indexOf(a as Period) - order.indexOf(b as Period))
    const color = instrumentColor(code, meta)
    return {
      key: code, title: instrumentName(code, meta), color,
      rows: keys.map((k) => {
        const g = groups.get(k)!
        const stat = (f: (d: Day) => number | null) => {
          const v = g.map(f).filter(finite)
          return { mid: quantile(v, 0.5), lo: quantile(v, 0.1), hi: quantile(v, 0.9), n: v.length }
        }
        const fr = stat(flowRatio), tot = stat(setpointRatio), uvir = stat((d) => ratio(d.bc.UV, d.bc.IR))
        const c = PERIOD_COLOR[k as Period] ?? color
        if (!fr.n && !uvir.n) return null
        return {
          key: k, label: PERIOD_LABEL[k as Period] ?? 'All days',
          labelColor: k in PERIOD_COLOR ? c : undefined,
          marks: [[{ mid: fr.mid, lo: fr.lo, hi: fr.hi, color: c }], [{ mid: tot.mid, lo: tot.lo, hi: tot.hi, color: c }], [{ mid: uvir.mid, lo: uvir.lo, hi: uvir.hi, color: c }]],
          note: `n=${fr.n}`,
          tip: [`${instrumentName(code, meta)} · ${PERIOD_LABEL[k as Period] ?? 'all days'}`,
            `Flow1/Flow2 ${fmt(fr.mid, 2)} (10–90 %: ${fmt(fr.lo, 2)}–${fmt(fr.hi, 2)}), ${fr.n} days`,
            `total / setpoint ${fmt(tot.mid, 4)} (${fmt(tot.lo, 4)}–${fmt(tot.hi, 4)})`,
            `UV/IR BCc ${fmt(uvir.mid, 2)} (${fmt(uvir.lo, 2)}–${fmt(uvir.hi, 2)}), ${uvir.n} days`],
        }
      }).filter((r) => r !== null),
    }
  }), [all, meta])
  return (
    <ChartFrame id="aq-flow-summary" title="Flow by site and period — spot-flow ratio, total flow and UV/IR before and after the fix"
      subtitle="Median and 10–90 % of the daily values per instrument and flow period (config.FLOW_FIX_PERIODS; Addis has none defined). The flow ratio is Flow1/Flow2 of daily mean flows; the reference line is the plotting module's 1.0 and the dashed marker its 2.0 'degraded'. UV/IR is the BCc ratio the flow-fix notebook compared before and after (above 1: stronger UV absorption)."
      provenance="FlowFix_BeforeAfter_Analysis §11 (UV/IR before vs after) and its flow-ratio notes · flow_qc_analysis (R, T) · processed_sites/*.pkl">
      <IntervalStrip blocks={blocks} labelWidth={150} noteWidth={48} minPanelWidth={200}
        panels={[
          { label: 'Flow1 / Flow2', reference: 1, marker: { value: 2, label: 'degraded 2.0' }, format: (v) => v.toFixed(1) },
          { label: 'Total flow / setpoint', reference: 1, format: (v) => v.toFixed(3) },
          { label: 'UV / IR BCc', reference: 1, format: (v) => v.toFixed(2) },
        ]} />
    </ChartFrame>
  )
}
