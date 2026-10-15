import { useMemo } from 'react'
import { FIT_STYLE, useEstimator, estimatorText } from '@/lib/regressionPrefs'
import * as d3 from 'd3'
import { ChartFrame, Empty } from '@/components/ChartFrame'
import { useChartOptions } from '@/lib/chartOptions'
import { FacetGrid } from '@/components/FacetGrid'
import { Legend, useLegend } from '@/components/Legend'
import { SweepPlot } from '@/components/SweepPlot'
import { INK } from '@/lib/theme'
import type { CalibFile } from '@/lib/types'
import { configKey, configLabel, sameConfig, type Config } from './common'
import { COHORT, CV_PROTOCOL, METRIC, PREPROCESSING, SELECTION_SPACE, label } from '@/lib/labels'

// Saved coefficient fields are projected to the active estimator by CalibrationPage.
const PANELS: { id: 'intercept' | 'slope' | 'r2'; title: string; keys: { key: 'db' | 'dm' | 'r2' | 'ho'; label: string; color: string; dash?: string }[]; ref?: number }[] = [
  { id: 'intercept', title: METRIC.testIntercept, keys: [{ key: 'db', label: 'Deming', color: INK.deming }], ref: 0 },
  { id: 'slope', title: METRIC.testSlope, keys: [{ key: 'dm', label: 'Deming', color: INK.deming }], ref: 1 },
  { id: 'r2', title: 'R²', keys: [{ key: 'r2', label: METRIC.testR2, color: INK.accent }, { key: 'ho', label: METRIC.cvR2, color: INK.good, dash: '5 3' }] },
]

/**
 * k sweep — the component-count "optimization" for one configuration, read
 * out on Addis under site-grouped cross-validation. The explorer's rule k is marked. The trap
 * this chart exists to show: intercept usually keeps shrinking with k while
 * slope and the IMPROVE cross-validation R² deteriorate, so the k with the smallest
 * intercept is rarely the k to report (CONVERSATION_OPTIMIZATION_AUDIT).
 */
export function KSweep({ calib, selected, onSelect }: { calib: CalibFile; selected: Config; onSelect: (c: Config) => void }) {
  const lg = useLegend()
  const method = useEstimator()
  const panels = PANELS.map(panel => ({ ...panel, keys: panel.keys.map(key => key.key === 'dm' || key.key === 'db' ? { ...key, color: FIT_STYLE[method].color, dash: FIT_STYLE[method].dash } : key) }))

  // the options and when each applies (lib/chartOptions): cohort, then cohort size, selection and
  // preprocessing, each offered only where a sweep exists for the choices before it, so the chart
  // never asks for a configuration the batch did not run. Bound to the page's shared selection.
  const sweeps = calib.sweeps
  const cohorts = useMemo(() => [...new Set(sweeps.map((s) => s.co))], [sweeps])
  const allCuts = useMemo(() => [...new Set(sweeps.map((s) => s.cut))].filter((c): c is number => c !== null).sort((a, b) => a - b), [sweeps])
  const allSels = useMemo(() => [...new Set(sweeps.map((s) => s.sel ?? ''))], [sweeps])
  const allSps = useMemo(() => [...new Set(sweeps.map((s) => s.sp))], [sweeps])
  const cutOf = (v: unknown) => (v === '' || v == null ? null : Number(v))
  const at = (c: Record<string, unknown>) => sweeps.filter((s) => s.co === c.co
    && ('cut' in c ? s.cut === cutOf(c.cut) : true) && ('sel' in c ? (s.sel ?? '') === c.sel : true))
  const pick = (patch: Partial<Config>) => onSelect({ ...eff(), ...patch })
  const o = useChartOptions('k-sweep', [
    { key: 'co', label: 'Cohort', style: 'select', default: selected.co, choices: cohorts.map((c) => ({ value: c, label: label(COHORT, c) })),
      bind: [selected.co, ((co: string) => { const first = sweeps.find((s) => s.co === co); onSelect({ co, cut: first?.cut ?? null, sel: first?.sel ?? null, sp: first?.sp ?? 'raw' }) }) as (v: never) => void] },
    { key: 'cut', label: 'Calibration cohort size', style: 'select', default: String(selected.cut ?? ''), hidden: '',
      when: (c: Record<string, unknown>) => sweeps.some((s) => s.co === c.co && s.cut !== null),
      choices: allCuts.map((k) => ({ value: String(k), when: (c: Record<string, unknown>) => sweeps.some((s) => s.co === c.co && s.cut === k) })),
      bind: [String(selected.cut ?? ''), ((v: string) => pick({ cut: cutOf(v) })) as (v: never) => void] },
    { key: 'sel', label: 'Selected on', style: 'select', default: selected.sel ?? '',
      choices: allSels.map((v) => ({ value: v, label: label(SELECTION_SPACE, v || 'raw'), when: (c: Record<string, unknown>) => at(c).some((s) => (s.sel ?? '') === v) })),
      bind: [selected.sel ?? '', ((v: string) => pick({ sel: v || null })) as (v: never) => void] },
    { key: 'sp', label: 'Preprocessing', style: 'select', default: selected.sp,
      choices: allSps.map((v) => ({ value: v, label: label(PREPROCESSING, v), when: (c: Record<string, unknown>) => at(c).some((s) => s.sp === v) })),
      bind: [selected.sp, ((v: string) => pick({ sp: v })) as (v: never) => void] },
    { key: 'n', node: () => <span className="control">{rows.length} PLS-factor counts{autoK !== null ? ` · rule: ${autoK} factors` : ''}</span> },
  ])
  // the configuration drawn: what the options resolved to, which always exists in the batch
  const eff = (): Config => ({ co: o.values.co as string, cut: cutOf(o.values.cut), sel: (o.values.sel as string) || null, sp: o.values.sp as string })
  const config = eff()
  const rows = useMemo(
    () => sweeps.filter((s) => sameConfig({ co: s.co, cut: s.cut, sel: s.sel, sp: s.sp }, config)).sort((a, b) => a.k - b.k),
    [sweeps, configKey(config)] // eslint-disable-line react-hooks/exhaustive-deps
  )
  const autoK = rows[0]?.ak ?? null

  const tipOf = (k: number) => {
    const r = rows.find((q) => q.k === k)!
    return [
      `${METRIC.k}: ${k}${k === autoK ? ' (rule)' : ''}`,
      `Test set (Addis): Deming ${r.dm?.toFixed(3) ?? '—'}x ${(r.db ?? 0) < 0 ? '−' : '+'} ${Math.abs(r.db ?? 0).toFixed(3)} · R² ${r.r2?.toFixed(3) ?? '—'}`,
      `${METRIC.cvR2}: ${r.ho?.toFixed(3) ?? '—'}`,
    ]
  }

  return (
    <ChartFrame
      id="k-sweep"
      title="PLS-factor sweep — choosing the number of PLS factors for one configuration"
      subtitle={`Every number of PLS factors the batch tried for the configuration chosen here (or clicked in the cutoff sweep above), read out on the Addis test set, with the calibration cohort cross-validated by ${CV_PROTOCOL.site_heldout}. The dashed vertical line is the number the explorer's rule would choose on its own.`}
      provenance="calibration_explorer batch results · Addis test set · site-grouped CV · MAC 10 · fixed set"
      controls={o.controls()}
    >
      <div className="chart-wrap">
        {rows.length < 2 ? (
          <Empty>No PLS-factor sweep saved for {configLabel(selected, calib)} on Addis.</Empty>
        ) : (
          <FacetGrid items={panels} minCell={250} maxCols={3} gap={12} render={(panel, cw) => {
            const keys = panel.keys.filter((k) => lg.show(k.label))
            const hot = panel.keys.some((k) => k.label === lg.hover) ? lg.hover : null
            const vals = keys.flatMap((k) => rows.map((r) => r[k.key]).filter((v): v is number => v !== null))
            const ext = (vals.length ? d3.extent(vals) : [panel.ref ?? 0, panel.ref ?? 1]) as [number, number]
            const lo = panel.ref !== undefined ? Math.min(panel.ref, ext[0]) : Math.min(0, ext[0])
            const hi = panel.ref !== undefined ? Math.max(panel.ref, ext[1]) : Math.max(1, ext[1])
            return <>
              <SweepPlot width={cw} minWidth={200} height={258} title={estimatorText(panel.title, method)} xLabel={METRIC.k} yLabel=""
                xDomain={[1, d3.max(rows, (r) => r.k) ?? 30]} yDomain={d3.scaleLinear().domain([lo, hi]).nice().domain() as [number, number]}
                reference={panel.ref} dim={(g) => (hot && hot !== g ? 0.15 : 1)}
                xMarkers={autoK !== null ? [{ at: autoK, label: `rule: ${autoK}` }] : []}
                yMarkers={panel.id === 'r2' ? [{ at: calib.heldout_floor, color: INK.good }] : []}
                chosen={autoK !== null ? keys.map((k) => ({ series: k.key, x: autoK })) : []}
                series={keys.map((k) => ({ id: k.key, label: k.label, group: k.label, color: k.color, dash: k.dash, marks: true,
                  points: rows.filter((r) => r[k.key] !== null).map((r) => ({ x: r.k, y: r[k.key], tip: tipOf(r.k) })) }))}
                background={({ y, iw, ih }) => panel.id === 'slope' && (() => {
                  const top = Math.max(0, y(calib.slope_box[1])), bot = Math.min(ih, y(calib.slope_box[0]))
                  return <rect x={0} y={top} width={iw} height={Math.max(0, bot - top)} fill={INK.accent} fillOpacity={0.06} />
                })()} />
              <Legend items={panel.keys.map((k) => ({ label: k.label, color: k.color, shape: k.dash ? ('dashed' as const) : ('line' as const) }))} {...lg.props} highlighted={hot} />
            </>
          }} />
        )}
        <p className="chart-note">{configLabel(selected, calib)}</p>
      </div>
    </ChartFrame>
  )
}
