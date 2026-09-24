import { useRef } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty } from '@/components/ChartFrame'
import { Legend, useLegend } from '@/components/Legend'
import { XAxis, YAxis } from '@/components/Axes'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { fmt } from '@/lib/stats'
import { INK, MARGIN, FONT } from '@/lib/theme'
import type { CalibRun, CalibRunsFile } from '@/lib/types'
import { CV_PROTOCOL, METRIC, label } from '@/lib/labels'

/**
 * The cross-validation curve that chose k: RMSECV against PLS component
 * count, with the per-fold standard error where the protocol produces one,
 * and the rule's choice ringed. This is the explorer's Calibrate curve; the
 * k sweep above shows what each k does to the target, this shows why the
 * rule stopped where it did (first major minimum under protocol A).
 */
export function CVCurve({ run, runs }: { run: CalibRun; runs: CalibRunsFile }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
  const lg = useLegend()
  const height = 320

  const pts = run.curve.filter((c) => c.rmsecv !== null && Number.isFinite(c.rmsecv)) as { k: number; rmsecv: number; se: number | null }[]
  const innerW = Math.max(240, width - MARGIN.left - MARGIN.right)
  const innerH = height - MARGIN.top - MARGIN.bottom
  const x = d3.scaleLinear().domain([1, d3.max(pts, (p) => p.k) ?? 30]).range([0, innerW])
  const hasSe = pts.some((p) => p.se !== null && Number.isFinite(p.se))
  const SE = '± 1 SE across folds'
  const seOn = hasSe && lg.show(SE)
  const yMax = d3.max(pts, (p) => p.rmsecv + (seOn ? p.se ?? 0 : 0)) ?? 1
  const yMin = d3.min(pts, (p) => p.rmsecv - (seOn ? p.se ?? 0 : 0)) ?? 0
  const y = d3.scaleLinear().domain([Math.max(0, yMin - (yMax - yMin) * 0.1), yMax * 1.04]).range([innerH, 0]).nice()
  const line = d3.line<(typeof pts)[number]>().x((p) => x(p.k)).y((p) => y(p.rmsecv))
  const band = d3.area<(typeof pts)[number]>().x((p) => x(p.k)).y0((p) => y(p.rmsecv - (p.se ?? 0))).y1((p) => y(p.rmsecv + (p.se ?? 0)))
  const target = runs.targets[run.target]

  return (
    <ChartFrame
      id="cv-curve"
      title="Cross-validation curve — why the rule stopped at this number of PLS factors"
      subtitle={`IMPROVE cross-validation RMSE (RMSECV, calibration set) in µg per filter against the number of PLS factors for this cohort, under the cross-validation protocol ${label(CV_PROTOCOL, runs.mode)}${runs.mode === 'site_heldout' ? ', first major minimum' : ''}. The curve does not depend on the test set; the ringed point is the rule's choice. The floor is quoted as a fraction of the mean training loading.`}
      provenance={`calibration_explorer /api/run curve · cohort ${run.n_cohort} filters, ${run.n_train_sites} sites`}
      controls={
        <span className="control">
          rule: {run.auto_k} PLS factors · RMSECV floor {fmt(run.rmsecv_floor, 3)} µg ({run.pct_rmsecv_floor} % of mean loading)
          {/* the exported held-out slope is a y-on-x fit, so only R² and RMSE are shown */}
          {run.heldout ? ` · ${METRIC.cvR2}: ${fmt(run.heldout.R2, 3)}, RMSE ${fmt(run.heldout.RMSE, 3)} µg` : ' · no whole-site cross-validation score under this protocol'}
        </span>
      }
    >
      <div ref={wrapRef} className="chart-wrap">
        {pts.length < 2 ? (
          <Empty>No curve saved for this configuration at {target?.site ?? run.target}.</Empty>
        ) : (
          <svg width={width} height={height}>
            <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
              <YAxis scale={y} x={0} label="IMPROVE cross-validation RMSE (µg / filter)" gridWidth={innerW} />
              <XAxis scale={x} y={innerH} label={METRIC.k} tickCount={8} />
              {seOn && <path d={band(pts) ?? ''} fill={INK.accent} fillOpacity={0.12 * lg.dim(SE, 0.3)} />}
              {lg.show('RMSECV') && <path d={line(pts) ?? ''} fill="none" stroke={INK.accent} strokeWidth={2} strokeOpacity={lg.dim('RMSECV')} />}
              <line x1={x(run.auto_k)} x2={x(run.auto_k)} y1={0} y2={innerH} stroke={INK.muted} strokeDasharray="3 3" />
              {lg.show('RMSECV') && pts.map((p) => (
                <circle key={p.k} opacity={lg.dim('RMSECV')} cx={x(p.k)} cy={y(p.rmsecv)} r={p.k === run.auto_k ? 6 : 3} fill={p.k === run.auto_k ? '#fff' : INK.accent} stroke={p.k === run.auto_k ? INK.text : INK.accent} strokeWidth={p.k === run.auto_k ? 2 : 1}
                  onMouseEnter={(e) => tip.show(e, [`${p.k} PLS factors${p.k === run.auto_k ? ' (rule choice)' : ''}`, `RMSECV ${fmt(p.rmsecv, 4)} µg${p.se !== null ? ` ± ${fmt(p.se, 4)}` : ''}`])}
                  onMouseLeave={tip.hide} />
              ))}
              <text x={x(run.auto_k) + 5} y={12} fontSize={10} fill={INK.muted} fontFamily={FONT.mono}>rule: {run.auto_k}</text>
            </g>
          </svg>
        )}
        <Legend items={[{ label: 'RMSECV', color: INK.accent, shape: 'line' }, ...(hasSe ? [{ label: SE, color: INK.accent, shape: 'band' as const }] : [])]} {...lg.props} />
        {tip.node}
      </div>
    </ChartFrame>
  )
}
