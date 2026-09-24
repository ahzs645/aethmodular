import { useMemo, useRef, useState } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Select, Toggle } from '@/components/ChartFrame'
import { Legend, useLegend } from '@/components/Legend'
import { XAxis, YAxis } from '@/components/Axes'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { INK, MARGIN, FONT } from '@/lib/theme'
import type { CalibFile, CalibRow } from '@/lib/types'
import { METRICS, fmtFit, spectraColor, isInterceptMetric, isSlopeMetric, metricLabel, metricOf, sameConfig, type Config, type Metric } from './common'
import { COHORT, CV_PROTOCOL, CV_PROTOCOL_SHORT, METRIC, PREPROCESSING, SELECTION_SPACE, label } from '@/lib/labels'

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
/** Test-set readouts only; the IMPROVE cross-validation R² gets its own panel below. */
const TEST_METRICS = METRICS.filter((m) => m !== 'held-out TOR R²')
const CV_METRIC: Metric = 'held-out TOR R²'

interface SweepPoint { r: CalibRow; x: number; y: number }
interface SweepSeries { key: string; sel: string; sp: string; label: string; pts: SweepPoint[]; color: string; dash?: string }

function sweepSeries(rows: CalibRow[], metric: Metric, allPairs: boolean): SweepSeries[] {
  const groups = d3.group(rows, (r) => `${r.sel ?? 'raw'}|${r.sp}`)
  return [...groups.entries()]
    .map(([key, rs]) => {
      const [sel, sp] = key.split('|')
      const pts = rs
        .map((r) => ({ r, x: r.cut as number, y: metricOf(r, metric, allPairs) }))
        .filter((p): p is SweepPoint => p.y !== null && Number.isFinite(p.y))
        .sort((a, b) => a.x - b.x)
      return { key, sel, sp, label: `${label(PREPROCESSING, sp)}${sel !== 'raw' ? ` · ${label(SELECTION_SPACE, sel)}` : ''}`, pts, color: spectraColor(sp), dash: sel !== 'raw' ? '6 4' : undefined }
    })
    .filter((s) => s.pts.length > 1)
    .sort((a, b) => a.label.localeCompare(b.label))
}

export function CutoffSweep({ calib, selected, onSelect }: { calib: CalibFile; selected: Config; onSelect: (c: Config) => void }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)

  const rankedCohorts = Object.keys(calib.default_cutoff)
  const [cohort, setCohort] = useState(rankedCohorts.includes('ocec') ? 'ocec' : rankedCohorts[0])
  const [mode, setMode] = useState('site_heldout')
  const [target, setTarget] = useState('addis')
  const [metric, setMetric] = useState<Metric>('Deming intercept')
  const [allPairs, setAllPairs] = useState(false)
  const { hidden, dim, props: legendProps } = useLegend()

  const targets = useMemo(() => [...new Set(calib.grid.map((r) => r.tg))].sort(), [calib.grid])
  const modes = useMemo(() => Object.keys(calib.modes).filter((m) => calib.grid.some((r) => r.mo === m)), [calib])

  const rows = useMemo(
    () => calib.grid.filter((r) => r.co === cohort && r.mo === mode && r.tg === target && r.lot === 'all' && r.el === 'all' && r.cut !== null),
    [calib.grid, cohort, mode, target]
  )
  const series = useMemo(() => sweepSeries(rows, metric, allPairs), [rows, metric, allPairs])
  // The calibration-set panel. calibration.json carries only the cross-validation R²
  // (`ho`, heldout_R2) per grid row; the `rmse` field is the test-set (evaluation site)
  // RMSE, and no RMSECV is exported, so this panel shows R² only.
  const cvSeries = useMemo(() => sweepSeries(rows, CV_METRIC, false), [rows])

  const siteName = calib.targets[target]?.site ?? target
  const testLabel = (m: Metric) => (target === 'addis' ? metricLabel(m) : metricLabel(m).replace('Addis', siteName))
  const shown = series.filter((s) => !hidden.has(s.label))
  const cvShown = cvSeries.filter((s) => !hidden.has(s.label))
  const innerW = Math.max(240, width - MARGIN.left - MARGIN.right)
  const xAll = [...shown, ...cvShown].flatMap((s) => s.pts.map((p) => p.x))
  const [x0, x1] = d3.extent(xAll)
  const x = d3.scaleLinear().domain(x0 === undefined || x1 === undefined ? [0, 1] : [x0, x1]).range([0, innerW]).nice()
  const defaultCut = calib.default_cutoff[cohort]
  const legendSeries = series.length ? series : cvSeries
  const shortMode = label(CV_PROTOCOL_SHORT, mode)

  const panel = (ss: SweepSeries[], m: Metric, height: number, yLabel: string) => (
    <SweepPanel
      series={ss} metric={m} height={height} width={width} innerW={innerW} x={x} calib={calib} yLabel={yLabel}
      defaultCut={defaultCut} selected={selected} dim={dim}
      onEnter={(e, s, p) =>
        tip.show(e, [
          `${cohort}-${p.x} × ${label(PREPROCESSING, s.sp)}${s.sel !== 'raw' ? ` (${label(SELECTION_SPACE, s.sel)})` : ''}`,
          `${METRIC.k}: ${p.r.k ?? '—'} (rule)`,
          `Test set (${siteName}): Deming ${fmtFit(allPairs ? p.r.adm : p.r.dm, allPairs ? p.r.adb : p.r.db)} · R² ${(allPairs ? p.r.ar2 : p.r.r2)?.toFixed(3) ?? '—'}`,
          `Calibration set: IMPROVE cross-validation R² ${p.r.ho?.toFixed(3) ?? '—'}`,
          'click to load into the PLS-factor sweep',
        ])
      }
      onLeave={tip.hide}
      onSelect={onSelect}
    />
  )
  const cvPanelTitle: React.CSSProperties = { margin: '14px 0 2px', fontSize: 12.5, fontWeight: 600, color: 'var(--ink-text)' }

  return (
    <ChartFrame
      id="cutoff-sweep"
      title="Cutoff sweep — how big should the IMPROVE calibration cohort be?"
      subtitle="Readouts at the explorer's rule number of PLS factors as the calibration cohort grows, one line per spectral preprocessing (colour); a dashed line is a cohort selected on baseline-corrected rather than raw spectra. The top panel is the test set (the evaluation site, never used in fitting); the bottom panel is the calibration set, scored by IMPROVE cross-validation. The vertical line is the locked cutoff. Click a point to load that configuration into the PLS-factor sweep below."
      provenance={`calibration_explorer batch results · ${calib.n_source_rows.toLocaleString()} rows · DENSE_SWEEP_RESULTS_2026-08-20.md`}
      controls={
        <>
          <Select label="cohort" value={cohort} options={rankedCohorts} onChange={setCohort} optionLabel={(c) => label(COHORT, c)} />
          <Select label="cross-validation protocol" value={mode} options={modes} onChange={setMode} optionLabel={(m) => label(CV_PROTOCOL, m)} />
          <Select label="test set" value={target} options={targets} onChange={setTarget} optionLabel={(t) => calib.targets[t]?.label ?? t} />
          <Select label="test-set readout" value={metric} options={[...TEST_METRICS]} onChange={(v) => setMetric(v as Metric)} optionLabel={metricLabel} />
          <Toggle label="all pairs (not the fixed set)" checked={allPairs} onChange={setAllPairs} />
        </>
      }
    >
      <div ref={wrapRef} className="chart-wrap">
        <p style={{ ...cvPanelTitle, marginTop: 0 }}>Test set: {siteName}</p>
        {shown.flatMap((s) => s.pts).length < 2 ? (
          <Empty>No rows for {label(COHORT, cohort)} · {shortMode} · {siteName}.{mode !== 'site_heldout' ? ' Interleaved CV covers AIRSpec spectra at every site (Colab batch, 2026-09-23); raw and 2nd-derivative spectra only at Addis and Bishoftu, on the 5-point cutoff ladder.' : ''}</Empty>
        ) : (
          panel(shown, metric, 340, testLabel(metric))
        )}
        <p style={{ ...cvPanelTitle, marginTop: 18, paddingTop: 12, borderTop: '1px solid var(--ink-border)' }}>
          Calibration set: IMPROVE cross-validation
          <span style={{ fontWeight: 400, color: 'var(--ink-muted)' }}> · scored on IMPROVE filters only, not on {siteName}</span>
        </p>
        {cvShown.flatMap((s) => s.pts).length < 2 ? (
          <Empty>
            {mode !== 'site_heldout'
              // calibration_explorer/app.py only scores held-out IMPROVE sites under site-grouped CV;
              // interleaved CV fits on every site, so there is nothing left out to score against
              ? <>{METRIC.cvR2} is exported only for {label(CV_PROTOCOL, 'site_heldout')}. {label(CV_PROTOCOL, mode)} keeps every IMPROVE site in the fit, so no whole site is left out to score. Switch the cross-validation protocol to see this panel.</>
              : <>No IMPROVE cross-validation readouts for {label(COHORT, cohort)} at these cutoffs.</>}
          </Empty>
        ) : (
          panel(cvShown, CV_METRIC, 220, 'IMPROVE cross-validation R²')
        )}
        <Legend
          items={legendSeries.map((s) => ({ label: s.label, color: s.color, shape: s.dash ? ('dashed' as const) : ('line' as const) }))}
          {...legendProps}
          note={[
            isSlopeMetric(metric) ? `shaded = slope box ${calib.slope_box[0]}–${calib.slope_box[1]}` : null,
            `bottom panel dashed = R² floor ${calib.heldout_floor}`,
          ].filter(Boolean).join(' · ')}
        />
        {tip.node}
      </div>
    </ChartFrame>
  )
}

function SweepPanel({ series, metric, height, width, innerW, x, calib, yLabel, defaultCut, selected, dim, onEnter, onLeave, onSelect }: {
  series: SweepSeries[]
  metric: Metric
  height: number
  width: number
  innerW: number
  x: d3.ScaleLinear<number, number>
  calib: CalibFile
  yLabel: string
  defaultCut: number | undefined
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
  const sortedY = all.map((p) => p.y).sort(d3.ascending)
  const yExt: [number, number] = [d3.quantile(sortedY, 0.02) ?? 0, d3.quantile(sortedY, 0.98) ?? 1]
  const yLo = isInterceptMetric(metric) ? Math.min(0, yExt[0]) : isSlopeMetric(metric) ? Math.min(0.5, yExt[0]) : Math.min(0, yExt[0])
  const y = d3.scaleLinear().domain([yLo, Math.max(isSlopeMetric(metric) ? 1.2 : 0, yExt[1])]).range([innerH, 0]).nice()
  const clampY = (v: number) => Math.max(y.domain()[0], Math.min(y.domain()[1], v))
  const isClamped = (v: number) => v < y.domain()[0] || v > y.domain()[1]
  const nClamped = all.filter((p) => isClamped(p.y)).length
  // extremes are not joined into the line: a spike to the edge of the axis is
  // a lie about the neighbouring cutoffs; they are drawn as hollow edge markers
  const line = d3.line<{ x: number; y: number }>().defined((p) => !isClamped(p.y)).x((p) => x(p.x)).y((p) => y(p.y))
  return (
    <>
      <svg width={width} height={height} className="animated">
        <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
          {isSlopeMetric(metric) && (
            <rect x={0} y={Math.max(0, y(calib.slope_box[1]))} width={innerW} height={Math.max(0, Math.min(innerH, y(calib.slope_box[0])) - Math.max(0, y(calib.slope_box[1])))} fill={INK.accent} fillOpacity={0.06} />
          )}
          {metric === CV_METRIC && <line x1={0} x2={innerW} y1={y(calib.heldout_floor)} y2={y(calib.heldout_floor)} stroke={INK.axis} strokeDasharray="4 3" />}
          <YAxis scale={y} x={0} label={yLabel} gridWidth={innerW} />
          <XAxis scale={x} y={innerH} label={METRIC.cohortSize} />
          {(isInterceptMetric(metric) || isSlopeMetric(metric)) && (
            <line x1={0} x2={innerW} y1={y(isSlopeMetric(metric) ? 1 : 0)} y2={y(isSlopeMetric(metric) ? 1 : 0)} stroke={INK.axis} strokeWidth={1} strokeOpacity={0.7} />
          )}
          {defaultCut !== undefined && x(defaultCut) >= 0 && x(defaultCut) <= innerW && (
            <g>
              <line x1={x(defaultCut)} x2={x(defaultCut)} y1={0} y2={innerH} stroke={INK.muted} strokeDasharray="3 3" />
              <text x={x(defaultCut) + 4} y={12} fontSize={10} fill={INK.muted} fontFamily={FONT.mono}>locked {defaultCut}</text>
            </g>
          )}
          {series.map((s) => (
            <g key={s.key} opacity={dim(s.label)}>
              <path d={line(s.pts) ?? ''} fill="none" stroke={s.color} strokeWidth={1.8} strokeDasharray={s.dash} strokeOpacity={0.85} pointerEvents="none" />
              {s.pts.map((p) => {
                const isSel = sameConfig({ co: p.r.co, cut: p.r.cut, sel: p.r.sel, sp: p.r.sp }, selected)
                return (
                  <circle
                    key={p.x}
                    cx={x(p.x)} cy={y(clampY(p.y))} r={isSel ? 6 : isClamped(p.y) ? 2.5 : 3}
                    fill={isSel || isClamped(p.y) ? '#fff' : s.color} stroke={isSel ? INK.text : s.color} strokeWidth={isSel ? 2 : 1}
                    style={{ cursor: 'pointer' }}
                    onMouseEnter={(e) => onEnter(e, s, p)}
                    onMouseLeave={onLeave}
                    onClick={() => onSelect({ co: p.r.co, cut: p.r.cut, sel: p.r.sel, sp: p.r.sp })}
                  />
                )
              })}
            </g>
          ))}
        </g>
      </svg>
      {nClamped > 0 && <p className="chart-note">{nClamped} extreme readouts (tiny cohorts) are drawn hollow at the edge of the axis and left out of the line.</p>}
    </>
  )
}
