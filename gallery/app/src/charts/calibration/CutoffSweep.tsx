import { useMemo, useRef } from 'react'
import { useChartOptions } from '@/lib/chartOptions'
import { robustDomain } from '@/lib/axes'
import { Marker } from '@/components/RefLines'
import * as d3 from 'd3'
import { ChartFrame, Empty } from '@/components/ChartFrame'
import { Legend, useLegend } from '@/components/Legend'
import { XAxis, YAxis } from '@/components/Axes'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { INK, MARGIN, FONT } from '@/lib/theme'
import type { CalibFile, CalibRow } from '@/lib/types'
import { METRICS, fmtFit, spectraColor, isInterceptMetric, isSlopeMetric, metricLabel, metricOf, sameConfig, type Config, type Metric } from './common'
import { COHORT, COHORT_SHORT, CV_PROTOCOL, CV_PROTOCOL_SHORT, METRIC, PREPROCESSING, SELECTION_SPACE, label } from '@/lib/labels'
import { calibrationReadout, cutoffPosition, fittingCounts, cutoffConfig, type CalibrationReadout, type CutoffAxis } from '@/lib/cutoffSweep'
import { AxisChoices } from '@/components/AxisChoices'
import { ChartAside } from '@/lib/singleView'

/**
 * Cutoff sweep — the "try somewhat more and somewhat less" experiment from
 * the July-17 meeting, as a line chart. One line per spectral preprocessing,
 * x = how many IMPROVE filters the calibration cohort keeps, y = the readout at
 * the rule number of PLS factors. The 2026-08-20 dense sweep found that the
 * locked ocec-800 cutoff was never the optimum: a basin at 440–490 × spline
 * baseline beats it on the test-set intercept *and* on IMPROVE cross-validation
 * R², which the coarse 600–1000 ladder could not see.
 *
 * Agreed with Ann, 23 Sep 2026: the calibration-set score (IMPROVE
 * cross-validation) sits in its own panel, apart from the test-set (Addis)
 * readouts, so the two are never read as the same kind of number.
 */
/** Test-set readouts only; IMPROVE cross-validation has its own panel below. */
const TEST_METRICS = METRICS.filter((m) => m !== 'held-out TOR R²')
const CV_METRIC: Metric = 'held-out TOR R²'
type SweepMetric = Metric | CalibrationReadout
const CV_LABELS: Record<CalibrationReadout, string> = {
  ho: 'IMPROVE cross-validation R² (held-out sites)',
  cv_rmse: 'IMPROVE cross-validation RMSE (held-out sites, µg/filter)',
  rmsecv: 'IMPROVE internal cross-validation RMSE (µg/filter)',
}
const CV_AXIS_LABELS: Record<CalibrationReadout, string> = {
  ho: 'IMPROVE cross-validation R²',
  cv_rmse: 'IMPROVE RMSE (µg/filter)',
  rmsecv: 'IMPROVE RMSECV (µg/filter)',
}
const X_LABELS: Record<CutoffAxis, string> = {
  fitted: 'IMPROVE filters used in fitting',
  cutoff: 'Requested IMPROVE cohort cutoff (filters)',
}
const isCvMetric = (m: SweepMetric): m is CalibrationReadout => m === 'ho' || m === 'cv_rmse' || m === 'rmsecv'
const fmtCount = (v: number | null) => v === null ? 'unavailable' : v.toLocaleString()

interface SweepPoint { r: CalibRow; x: number; y: number }
interface SweepSeries { key: string; sel: string; sp: string; label: string; pts: SweepPoint[]; color: string; dash?: string }

function sweepSeries(rows: CalibRow[], metric: SweepMetric, allPairs: boolean, axis: CutoffAxis): SweepSeries[] {
  const groups = d3.group(rows, (r) => `${r.sel ?? 'raw'}|${r.sp}`)
  return [...groups.entries()]
    .map(([key, rs]) => {
      const [sel, sp] = key.split('|')
      const pts = rs
        .map((r) => ({ r, x: cutoffPosition(r, axis), y: isCvMetric(metric) ? calibrationReadout(r, metric) : metricOf(r, metric, allPairs) }))
        .filter((p): p is SweepPoint => p.x !== null && p.y !== null && Number.isFinite(p.y))
        .sort((a, b) => a.x - b.x || (a.r.cut ?? 0) - (b.r.cut ?? 0))
      return { key, sel, sp, label: `${label(PREPROCESSING, sp)}${sel !== 'raw' ? ` · ${label(SELECTION_SPACE, sel)}` : ''}`, pts, color: spectraColor(sp), dash: sel !== 'raw' ? '6 4' : undefined }
    })
    .filter((s) => s.pts.length > 0)
    .sort((a, b) => a.label.localeCompare(b.label))
}

// readouts with no all-pairs version: the toggle would change nothing
const NO_ALL_PAIRS = new Set<Metric>(['Deming intercept @ MAC 6'])

export function CutoffSweep({ calib, selected, onSelect }: { calib: CalibFile; selected: Config; onSelect: (c: Config) => void }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)

  const rankedCohorts = Object.keys(calib.default_cutoff)
  const { hidden, dim, props: legendProps } = useLegend()

  const targets = useMemo(() => [...new Set(calib.grid.map((r) => r.tg))].sort(), [calib.grid])
  const modes = useMemo(() => Object.keys(calib.modes).filter((m) => calib.grid.some((r) => r.mo === m)), [calib])
  // the swept rows for a cohort / protocol / test set (any of them may be left out)
  const swept = (c: Record<string, unknown>) => calib.grid.filter((r) => r.lot === 'all' && r.el === 'all' && r.cut !== null
    && r.co === c.cohort && (!('mode' in c) || r.mo === c.mode) && (!('target' in c) || r.tg === c.target))
  // the options and when each applies (lib/chartOptions): protocols and test sets are offered only
  // where the batch swept them for the chosen cohort; the fitted-count axis only where the rows carry
  // fitted counts; "all pairs" only for a readout that has an all-pairs version
  const o = useChartOptions('cutoff-sweep', [
    { key: 'cohort', label: 'Cohort', style: 'select', default: rankedCohorts.includes('ocec') ? 'ocec' : rankedCohorts[0], choices: rankedCohorts.map((c) => ({ value: c, label: label(COHORT, c) })) },
    { key: 'mode', label: 'Cross-validation protocol', style: 'select', default: 'site_heldout',
      choices: modes.map((m) => ({ value: m, label: label(CV_PROTOCOL, m), when: (c: Record<string, unknown>) => swept({ cohort: c.cohort, mode: m }).length > 0 })) },
    { key: 'target', label: 'Test set', style: 'select', default: 'addis',
      choices: targets.map((t) => ({ value: t, label: calib.targets[t]?.label ?? t, when: (c: Record<string, unknown>) => swept({ cohort: c.cohort, mode: c.mode, target: t }).length > 0 })) },
    { key: 'metric', axis: 'y', label: 'Test-set readout', style: 'select', default: 'Deming intercept', choices: TEST_METRICS.map((m) => ({ value: m, label: metricLabel(m) })) },
    { key: 'cvMetric', axis: 'y', label: 'Calibration cross-validation readout', style: 'select', default: 'cv_rmse', choices: (['cv_rmse', 'ho', 'rmsecv'] as const).map((m) => ({ value: m, label: CV_LABELS[m] })) },
    { key: 'axis', axis: 'x', label: 'Horizontal axis', style: 'select', default: 'fitted',
      choices: (['fitted', 'cutoff'] as const).map((a) => ({ value: a, label: X_LABELS[a], when: a === 'fitted' ? (c: Record<string, unknown>) => swept(c).some((r) => cutoffPosition(r, 'fitted') !== null) : undefined })) },
    { key: 'allPairs', label: 'All pairs (not the fixed set)', toggle: true, default: false, hidden: false, when: (c) => !NO_ALL_PAIRS.has(c.metric as Metric) },
  ])
  const cohort = o.values.cohort as string
  const mode = o.values.mode as string
  const target = o.values.target as string
  const metric = o.values.metric as Metric
  const cvMetric = o.values.cvMetric as CalibrationReadout
  const allPairs = o.values.allPairs as boolean
  const axis = o.values.axis as CutoffAxis

  const rows = useMemo(
    () => calib.grid.filter((r) => r.co === cohort && r.mo === mode && r.tg === target && r.lot === 'all' && r.el === 'all' && r.cut !== null),
    [calib.grid, cohort, mode, target]
  )
  const hasFitCounts = rows.some((r) => cutoffPosition(r, 'fitted') !== null)
  const missingFitCounts = rows.filter((r) => cutoffPosition(r, 'fitted') === null).length
  const series = useMemo(() => sweepSeries(rows, metric, allPairs, axis), [rows, metric, allPairs, axis])
  const cvSeries = useMemo(() => sweepSeries(rows, cvMetric, false, axis), [rows, cvMetric, axis])

  const siteName = calib.targets[target]?.site ?? target
  const testLabel = (m: Metric) => (target === 'addis' ? metricLabel(m) : metricLabel(m).replace('Addis', siteName))
  const shown = series.filter((s) => !hidden.has(s.label))
  const cvShown = cvSeries.filter((s) => !hidden.has(s.label))
  const innerW = Math.max(240, width - MARGIN.left - MARGIN.right)
  const xAll = [...shown, ...cvShown].flatMap((s) => s.pts.map((p) => p.x))
  const [x0, x1] = d3.extent(xAll)
  const x = d3.scaleLinear().domain(x0 === undefined || x1 === undefined ? [0, 1] : [x0, x1]).range([0, innerW]).nice()
  const defaultCut = calib.default_cutoff[cohort]
  const legendSeries = [...new Map([...series, ...cvSeries].map((s) => [s.key, s])).values()]
  const shortMode = label(CV_PROTOCOL_SHORT, mode)

  const panel = (ss: SweepSeries[], m: SweepMetric, height: number, yLabel: string) => (
    <SweepPanel
      series={ss} metric={m} height={height} width={width} innerW={innerW} x={x} calib={calib} yLabel={yLabel}
      defaultCut={defaultCut} axis={axis} selected={selected} dim={dim}
      onEnter={(e, s, p) =>
        tip.show(e, [
          `${label(COHORT_SHORT, cohort)} ${p.r.cut} × ${label(PREPROCESSING, s.sp)}${s.sel !== 'raw' ? ` (${label(SELECTION_SPACE, s.sel)})` : ''}`,
          `Selected cohort: ${fmtCount(fittingCounts(p.r).cohort)} · fitted: ${fmtCount(fittingCounts(p.r).fitted)} · held-out: ${fmtCount(fittingCounts(p.r).heldout)} filters`,
          `${METRIC.k}: ${p.r.k ?? '—'} (rule)`,
          `Test set (${siteName}): Deming ${fmtFit(allPairs ? p.r.adm : p.r.dm, allPairs ? p.r.adb : p.r.db)} · R² ${(allPairs ? p.r.ar2 : p.r.r2)?.toFixed(3) ?? '—'}`,
          `Calibration set: IMPROVE cross-validation R² ${p.r.ho?.toFixed(3) ?? '—'}`,
          `Calibration set: IMPROVE cross-validation RMSE ${calibrationReadout(p.r, 'cv_rmse')?.toFixed(3) ?? 'unavailable'} µg/filter`,
          `Internal cross-validation RMSE: ${calibrationReadout(p.r, 'rmsecv')?.toFixed(3) ?? 'unavailable'} µg/filter`,
          'click to load into the PLS-factor sweep',
        ])
      }
      onLeave={tip.hide}
      onSelect={onSelect}
    />
  )
  const cvPanelTitle: React.CSSProperties = { margin: '14px 0 2px', fontSize: 12.5, fontWeight: 600, color: 'var(--ink-text)' }
  const countCut = selected.co === cohort && selected.cut !== null && rows.some((r) => r.cut === selected.cut) ? selected.cut : defaultCut
  const countRows = rows.filter((r) => r.cut === countCut).sort((a, b) => a.sp.localeCompare(b.sp) || (a.sel ?? '').localeCompare(b.sel ?? ''))

  return (
    <ChartFrame
      id="cutoff-sweep"
      title="Cutoff sweep — how big should the IMPROVE calibration cohort be?"
      subtitle="Readouts at the explorer's rule number of PLS factors, one line per spectral preprocessing (colour); a dashed line is a cohort selected on baseline-corrected rather than raw spectra. The top panel scores the test set (the evaluation site, never used in fitting); the bottom panel scores IMPROVE cross-validation. IMPROVE RMSE measures TOR loading in µg/filter. Fitted counts exclude the whole-site holdout and can differ at the same requested cutoff. Vertical markers locate the locked cutoff. Click a point to load its original cutoff into the PLS-factor sweep below."
      provenance={`calibration_explorer batch results · ${calib.n_source_rows.toLocaleString()} rows · DENSE_SWEEP_RESULTS_2026-08-20.md`}
      controls={o.controls()}
    >
      <div ref={wrapRef} className="chart-wrap">
        {!hasFitCounts && <ChartAside><p className="chart-note">Actual fitting counts are unavailable for these saved rows; the axis shows requested cutoffs. No test-set pair count or cutoff is substituted for a fit count.</p></ChartAside>}
        {axis === 'fitted' && missingFitCounts > 0 && <ChartAside><p className="chart-note">{missingFitCounts} saved rows have no fitting counts and are omitted on this axis. Choose requested cutoff to inspect their available scores.</p></ChartAside>}
        <p style={{ ...cvPanelTitle, marginTop: 0 }}>Test set: {siteName}</p>
        {shown.flatMap((s) => s.pts).length === 0 ? (
          <Empty>No rows for {label(COHORT, cohort)} · {shortMode} · {siteName}.{mode !== 'site_heldout' ? ' Interleaved cross-validation covers spline-baselined spectra at every site (Colab batch, 2026-09-23); raw and 2nd-derivative spectra only at Addis and Bishoftu, on the 5-point cutoff ladder.' : ''}</Empty>
        ) : (
          <AxisChoices {...o.axisChoices({ x: 'axis', y: 'metric' })}>{panel(shown, metric, 340, testLabel(metric))}</AxisChoices>
        )}
        <p style={{ ...cvPanelTitle, marginTop: 18, paddingTop: 12, borderTop: '1px solid var(--ink-border)' }}>
          Calibration set: IMPROVE cross-validation
          <span style={{ fontWeight: 400, color: 'var(--ink-muted)' }}> · scored on IMPROVE filters only, not on {siteName}</span>
        </p>
        <ChartAside><p className="chart-note">{cvMetric === 'rmsecv'
          ? 'Internal cross-validation: fold predictions within the fitted IMPROVE set, at the selected number of PLS factors. RMSECV scores TOR loading in µg/filter.'
          : 'Held-out-site cross-validation: predictions for IMPROVE sites excluded from the final fit. RMSE scores TOR loading in µg/filter; it is distinct from test-set concentration error.'}</p></ChartAside>
        {cvShown.flatMap((s) => s.pts).length === 0 ? (
          <Empty>
            {mode !== 'site_heldout' && cvMetric !== 'rmsecv'
              // calibration_explorer/app.py only scores held-out IMPROVE sites under site-grouped CV;
              // interleaved CV fits on every site, so there is nothing left out to score against
              ? <>No {CV_LABELS[cvMetric]} is saved for these rows. Interleaved protocols keep every site in the final fit, so no whole-site holdout score exists. Select internal cross-validation RMSE to inspect any saved fold scores.</>
              : <>No {CV_LABELS[cvMetric]} is saved for {label(COHORT, cohort)} at these cutoffs. Missing RMSE is not replaced by the test-set RMSE.</>}
          </Empty>
        ) : (
          <AxisChoices {...o.axisChoices({ x: 'axis', y: 'cvMetric' })}>{panel(cvShown, cvMetric, 250, CV_AXIS_LABELS[cvMetric])}</AxisChoices>
        )}
        <Legend
          items={legendSeries.map((s) => ({ label: s.label, color: s.color, shape: s.dash ? ('dashed' as const) : ('line' as const) }))}
          {...legendProps}
          note={[
            isSlopeMetric(metric) ? `shaded = slope box ${calib.slope_box[0]}–${calib.slope_box[1]}` : null,
            cvMetric === 'ho' ? `bottom panel dashed = R² floor ${calib.heldout_floor}` : null,
          ].filter(Boolean).join(' · ')}
        />
        {countRows.length > 0 && <div style={{ overflowX: 'auto', marginTop: 12 }}>
          <ChartAside><p className="chart-note">Cutoff {countCut}: actual calibration counts and scores. Click a chart point to inspect another cutoff. Fitted and held-out counts describe the final fit split; internal CV repeatedly holds out folds within the fitted set.</p></ChartAside>
          <div style={{ overflowX: 'auto' }}>
          <table className="census-table" style={{ width: '100%', minWidth: 800, fontSize: 12 }}>
            <thead><tr><th>Spectra / selection</th><th>Selected cohort</th><th>Fitted</th><th>Held-out</th><th>Site-grouped cross-validation R²</th><th>Site-grouped cross-validation RMSE (µg/filter)</th><th>Internal cross-validation RMSE (µg/filter)</th></tr></thead>
            <tbody>{countRows.map((r) => {
              const counts = fittingCounts(r)
              return <tr key={`${r.sp}|${r.sel}`}><td>{label(PREPROCESSING, r.sp)}{r.sel && r.sel !== 'raw' ? ` · ${label(SELECTION_SPACE, r.sel)}` : ''}</td><td>{fmtCount(counts.cohort)}</td><td>{fmtCount(counts.fitted)}</td><td>{fmtCount(counts.heldout)}</td><td>{r.ho?.toFixed(3) ?? 'unavailable'}</td><td>{calibrationReadout(r, 'cv_rmse')?.toFixed(3) ?? 'unavailable'}</td><td>{calibrationReadout(r, 'rmsecv')?.toFixed(3) ?? 'unavailable'}</td></tr>
            })}</tbody>
          </table>
          </div>
        </div>}
        {tip.node}
      </div>
    </ChartFrame>
  )
}

function SweepPanel({ series, metric, height, width, innerW, x, calib, yLabel, defaultCut, axis, selected, dim, onEnter, onLeave, onSelect }: {
  series: SweepSeries[]
  metric: SweepMetric
  height: number
  width: number
  innerW: number
  x: d3.ScaleLinear<number, number>
  calib: CalibFile
  yLabel: string
  defaultCut: number | undefined
  axis: CutoffAxis
  selected: Config
  dim: (label: string) => number
  onEnter: (e: React.MouseEvent, s: SweepSeries, p: SweepPoint) => void
  onLeave: () => void
  onSelect: (c: Config) => void
}) {
  const innerH = height - MARGIN.top - MARGIN.bottom
  const all = series.flatMap((s) => s.pts)
  // Tiny cohorts (cutoff ≈ 100) produce wild fits that would squash the
  // readable range into a few pixels; the axis follows the 2–98 % bulk and the
  // extremes are pinned to the edge, with a count below the chart.
  const yExt = robustDomain(all.map((p) => p.y), 0.02, 0.98).domain
  const slope = !isCvMetric(metric) && isSlopeMetric(metric)
  const intercept = !isCvMetric(metric) && isInterceptMetric(metric)
  const yLo = intercept ? Math.min(0, yExt[0]) : slope ? Math.min(0.5, yExt[0]) : Math.min(0, yExt[0])
  const y = d3.scaleLinear().domain([yLo, Math.max(slope ? 1.2 : 0, yExt[1])]).range([innerH, 0]).nice()
  const clampY = (v: number) => Math.max(y.domain()[0], Math.min(y.domain()[1], v))
  const isClamped = (v: number) => v < y.domain()[0] || v > y.domain()[1]
  const nClamped = all.filter((p) => isClamped(p.y)).length
  // extremes are not joined into the line: a spike to the edge of the axis is
  // a lie about the neighbouring cutoffs; they are drawn as hollow edge markers
  const line = d3.line<{ x: number; y: number }>().defined((p) => !isClamped(p.y)).x((p) => x(p.x)).y((p) => y(p.y))
  const locked = axis === 'cutoff'
    ? (defaultCut === undefined ? [] : [{ x: defaultCut, label: `locked ${defaultCut}`, color: INK.muted }])
    : [...new Map(series.flatMap((s) => s.pts.filter((p) => p.r.cut === defaultCut).map((p) => [p.x, { x: p.x, label: `locked ${defaultCut}: ${p.x} fitted`, color: s.color }] as const))).values()]
  return (
    <>
      <svg width={width} height={height} className="animated">
        <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
          {slope && (
            <rect x={0} y={Math.max(0, y(calib.slope_box[1]))} width={innerW} height={Math.max(0, Math.min(innerH, y(calib.slope_box[0])) - Math.max(0, y(calib.slope_box[1])))} fill={INK.accent} fillOpacity={0.06} />
          )}
          {(metric === CV_METRIC || metric === 'ho') && <Marker orient="h" at={y(calib.heldout_floor)} from={0} to={innerW} />}
          <YAxis scale={y} x={0} label={yLabel} gridWidth={innerW} pick="y" />
          <XAxis scale={x} y={innerH} label={X_LABELS[axis]} pick="x" />
          {(intercept || slope) && (
            <line x1={0} x2={innerW} y1={y(slope ? 1 : 0)} y2={y(slope ? 1 : 0)} stroke={INK.axis} strokeWidth={1} strokeOpacity={0.7} />
          )}
          {locked.filter((l) => x(l.x) >= 0 && x(l.x) <= innerW).map((l, i) => (
            <g key={l.x}>
              <Marker orient="v" at={x(l.x)} from={0} to={innerH} color={l.color} />
              <text x={Math.min(innerW - 5, x(l.x) + 4)} y={12 + i * 12} textAnchor={x(l.x) > innerW * 0.7 ? 'end' : 'start'} fontSize={10} fill={l.color} fontFamily={FONT.mono}>{l.label}</text>
            </g>
          ))}
          {series.map((s) => (
            <g key={s.key} opacity={dim(s.label)}>
              <path d={line(s.pts) ?? ''} fill="none" stroke={s.color} strokeWidth={1.8} strokeDasharray={s.dash} strokeOpacity={0.85} pointerEvents="none" />
              {s.pts.map((p) => {
                const isSel = sameConfig(cutoffConfig(p.r), selected)
                return (
                  <circle
                    key={p.r.cut}
                    cx={x(p.x)} cy={y(clampY(p.y))} r={isSel ? 6 : isClamped(p.y) ? 2.5 : 3}
                    fill={isSel || isClamped(p.y) ? INK.panel : s.color} stroke={isSel ? INK.text : s.color} strokeWidth={isSel ? 2 : 1}
                    style={{ cursor: 'pointer' }}
                    onMouseEnter={(e) => onEnter(e, s, p)}
                    onMouseLeave={onLeave}
                    onClick={() => onSelect(cutoffConfig(p.r))}
                  />
                )
              })}
            </g>
          ))}
        </g>
      </svg>
      {nClamped > 0 && <ChartAside><p className="chart-note">{nClamped} extreme readouts (tiny cohorts) are drawn hollow at the edge of the axis and left out of the line.</p></ChartAside>}
    </>
  )
}
