import { useRef } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty } from '@/components/ChartFrame'
import { Legend } from '@/components/Legend'
import { XAxis, YAxis } from '@/components/Axes'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { fmt } from '@/lib/stats'
import { INK, MARGIN, FONT } from '@/lib/theme'
import type { CalibRun, CalibRunsFile } from '@/lib/types'

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
  const height = 320

  const pts = run.curve.filter((c) => c.rmsecv !== null && Number.isFinite(c.rmsecv)) as { k: number; rmsecv: number; se: number | null }[]
  const innerW = Math.max(240, Math.min(720, width) - MARGIN.left - MARGIN.right)
  const innerH = height - MARGIN.top - MARGIN.bottom
  const x = d3.scaleLinear().domain([1, d3.max(pts, (p) => p.k) ?? 30]).range([0, innerW])
  const hasSe = pts.some((p) => p.se !== null && Number.isFinite(p.se))
  const yMax = d3.max(pts, (p) => p.rmsecv + (hasSe ? p.se ?? 0 : 0)) ?? 1
  const yMin = d3.min(pts, (p) => p.rmsecv - (hasSe ? p.se ?? 0 : 0)) ?? 0
  const y = d3.scaleLinear().domain([Math.max(0, yMin - (yMax - yMin) * 0.1), yMax * 1.04]).range([innerH, 0]).nice()
  const line = d3.line<(typeof pts)[number]>().x((p) => x(p.k)).y((p) => y(p.rmsecv))
  const band = d3.area<(typeof pts)[number]>().x((p) => x(p.k)).y0((p) => y(p.rmsecv - (p.se ?? 0))).y1((p) => y(p.rmsecv + (p.se ?? 0)))
  const target = runs.targets[run.target]

  return (
    <ChartFrame
      id="cv-curve"
      title="Cross-validation curve — why the rule stopped at this k"
      subtitle={`RMSECV in µg per filter against PLS component count for this cohort under protocol ${runs.mode === 'site_heldout' ? 'A (site-grouped 5-fold, first major minimum)' : runs.mode}. The curve is target-independent; the ringed k is the rule's choice. The floor is quoted as a fraction of the mean training loading.`}
      provenance={`calibration_explorer /api/run curve · cohort ${run.n_cohort} filters, ${run.n_train_sites} sites`}
      controls={
        <span className="control">
          rule k = {run.auto_k} · RMSECV floor {fmt(run.rmsecv_floor, 3)} µg ({run.pct_rmsecv_floor} % of mean loading)
          {run.heldout ? ` · held-out TOR: R² ${fmt(run.heldout.R2, 3)}, slope ${fmt(run.heldout.slope, 3)}, RMSE ${fmt(run.heldout.RMSE, 3)}` : ' · no held-out test under this protocol'}
        </span>
      }
    >
      <div ref={wrapRef} className="chart-wrap">
        {pts.length < 2 ? (
          <Empty>No curve saved for this configuration at {target?.site ?? run.target}.</Empty>
        ) : (
          <svg width={Math.min(720, width)} height={height}>
            <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
              <YAxis scale={y} x={0} label="RMSECV (µg / filter)" gridWidth={innerW} />
              <XAxis scale={x} y={innerH} label="PLS components k" tickCount={8} />
              {hasSe && <path d={band(pts) ?? ''} fill={INK.accent} fillOpacity={0.12} />}
              <path d={line(pts) ?? ''} fill="none" stroke={INK.accent} strokeWidth={2} />
              <line x1={x(run.auto_k)} x2={x(run.auto_k)} y1={0} y2={innerH} stroke={INK.muted} strokeDasharray="3 3" />
              {pts.map((p) => (
                <circle key={p.k} cx={x(p.k)} cy={y(p.rmsecv)} r={p.k === run.auto_k ? 6 : 3} fill={p.k === run.auto_k ? '#fff' : INK.accent} stroke={p.k === run.auto_k ? INK.text : INK.accent} strokeWidth={p.k === run.auto_k ? 2 : 1}
                  onMouseEnter={(e) => tip.show(e, [`k = ${p.k}${p.k === run.auto_k ? ' (rule choice)' : ''}`, `RMSECV ${fmt(p.rmsecv, 4)} µg${p.se !== null ? ` ± ${fmt(p.se, 4)}` : ''}`])}
                  onMouseLeave={tip.hide} />
              ))}
              <text x={x(run.auto_k) + 5} y={12} fontSize={10} fill={INK.muted} fontFamily={FONT.mono}>rule k = {run.auto_k}</text>
            </g>
          </svg>
        )}
        <Legend items={[{ label: 'RMSECV', color: INK.accent, shape: 'line' }, ...(hasSe ? [{ label: '± 1 SE across folds', color: INK.accent, shape: 'band' as const }] : [])]} />
        {tip.node}
      </div>
    </ChartFrame>
  )
}
