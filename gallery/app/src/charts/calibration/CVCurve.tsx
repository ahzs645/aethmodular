import * as d3 from 'd3'
import { ChartFrame, Empty } from '@/components/ChartFrame'
import { Legend, useLegend } from '@/components/Legend'
import { SweepPlot } from '@/components/SweepPlot'
import { fmt } from '@/lib/stats'
import { INK } from '@/lib/theme'
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
  const lg = useLegend()
  const pts = run.curve.filter((c) => c.rmsecv !== null && Number.isFinite(c.rmsecv)) as { k: number; rmsecv: number; se: number | null }[]
  const hasSe = pts.some((p) => p.se !== null && Number.isFinite(p.se))
  const SE = '± 1 SE across folds'
  const seOn = hasSe && lg.show(SE), lineOn = lg.show('RMSECV')
  const yMax = d3.max(pts, (p) => p.rmsecv + (seOn ? p.se ?? 0 : 0)) ?? 1
  const yMin = d3.min(pts, (p) => p.rmsecv - (seOn ? p.se ?? 0 : 0)) ?? 0
  const yDomain = d3.scaleLinear().domain([Math.max(0, yMin - (yMax - yMin) * 0.1), yMax * 1.04]).nice().domain() as [number, number]
  const lo = (p: (typeof pts)[number]) => (seOn && p.se !== null ? p.rmsecv - p.se : null)
  const hi = (p: (typeof pts)[number]) => (seOn && p.se !== null ? p.rmsecv + p.se : null)
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
      <div className="chart-wrap">
        {pts.length < 2 ? (
          <Empty>No curve saved for this configuration at {target?.site ?? run.target}.</Empty>
        ) : (
          <SweepPlot height={320} xDomain={[1, d3.max(pts, (p) => p.k) ?? 30]} yDomain={yDomain}
            xLabel={METRIC.k} yLabel="IMPROVE cross-validation RMSE (µg / filter)"
            // the SE band keeps full strength while its own legend entry is hovered
            dim={(g) => (lg.hover === SE ? 1 : lg.dim(g))}
            xMarkers={[{ at: run.auto_k, label: `rule: ${run.auto_k}` }]}
            chosen={lineOn ? [{ series: 'rmsecv', x: run.auto_k }] : []}
            series={lineOn ? [{ id: 'rmsecv', label: 'RMSECV', group: 'RMSECV', color: INK.accent, width: 2,
              points: pts.map((p) => ({ x: p.k, y: p.rmsecv, lo: lo(p), hi: hi(p),
                tip: [`${p.k} PLS factors${p.k === run.auto_k ? ' (rule choice)' : ''}`, `RMSECV ${fmt(p.rmsecv, 4)} µg${p.se !== null ? ` ± ${fmt(p.se, 4)}` : ''}`] })) }] : []}>
            {/* the band alone, when the line is toggled off */}
            {({ x, y }) => !lineOn && seOn && <path fill={INK.accent} fillOpacity={0.15 * lg.dim(SE, 0.3)}
              d={d3.area<(typeof pts)[number]>().x((p) => (x as d3.ScaleLinear<number, number>)(p.k)).y0((p) => y(lo(p) ?? p.rmsecv)).y1((p) => y(hi(p) ?? p.rmsecv))(pts) ?? ''} />}
          </SweepPlot>
        )}
        <Legend items={[{ label: 'RMSECV', color: INK.accent, shape: 'line' }, ...(hasSe ? [{ label: SE, color: INK.accent, shape: 'band' as const }] : [])]} {...lg.props} />
      </div>
    </ChartFrame>
  )
}
