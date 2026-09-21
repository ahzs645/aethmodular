import { useId, useMemo, useRef, useState } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Segmented } from '@/components/ChartFrame'
import { Legend, toggleIn } from '@/components/Legend'
import { XAxis, YAxis } from '@/components/Axes'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { focusStyle, useHighlight } from '@/lib/highlight'
import { fmt } from '@/lib/stats'
import { INK, MARGIN, FONT } from '@/lib/theme'
import type { CalibRun, CalibRunsFile, MetaFile } from '@/lib/types'
import { groupColorFn, metricRow } from './common'

const XMODES = ['reference', 'deployed EC'] as const
const ESTIMATORS = ['Deming', 'OLS'] as const
const SETS = ['fixed', 'all'] as const

/**
 * The explorer's Calibrate crossplot and its residual panel, for one preset
 * at one evaluation site: predicted FTIR EC against the HIPS reference
 * (Fabs ÷ MAC), coloured by season, with the fitted line from the same
 * metrics row the explorer quotes. The residual panel is the meeting's check:
 * does the correction remove the curve, or only shift it?
 *
 * "deployed EC" swaps the x axis for the deployed SPARTAN EC on the same
 * filters (Addis only), which is the explorer's vs-deployed panel.
 * Clicking an Addis point opens the gallery's sample drawer for that filter.
 */
export function RunCrossplot({ run, runs, meta, knownIds }: { run: CalibRun; runs: CalibRunsFile; meta: MetaFile; knownIds: Set<string> }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
  const hl = useHighlight()
  const clipId = useId()

  const [xmode, setXmode] = useState<(typeof XMODES)[number]>('reference')
  const [est, setEst] = useState<(typeof ESTIMATORS)[number]>('Deming')
  const [evalSet, setEvalSet] = useState<(typeof SETS)[number]>('fixed')
  const [mac, setMac] = useState<number>(runs.mac_value)
  const [hidden, setHidden] = useState<Set<string>>(new Set())

  const macs = useMemo(() => [...new Set(run.metrics.map((m) => m.MAC).filter((v): v is number => v !== null))].sort((a, b) => a - b), [run.metrics])
  const isFabs = run.ref_kind === 'fabs'
  const hasDeployed = !!run.eval.deployed?.some((v) => v !== null)
  const hasFixed = run.eval.fixed.some(Boolean)
  const useDeployed = xmode === 'deployed EC' && hasDeployed
  const color = useMemo(() => groupColorFn(meta, run.eval.group), [meta, run.eval.group])

  const pts = useMemo(() => {
    const e = run.eval
    const out: { i: number; x: number; y: number; g: string; id: string | null; fixed: boolean }[] = []
    for (let i = 0; i < e.pred.length; i++) {
      if (evalSet === 'fixed' && hasFixed && !e.fixed[i]) continue
      if (hidden.has(e.group[i])) continue
      const x = useDeployed ? e.deployed?.[i] ?? null : isFabs ? e.ref[i] / mac : e.ref[i]
      if (x === null || !Number.isFinite(x) || !Number.isFinite(e.pred[i])) continue
      out.push({ i, x, y: e.pred[i], g: e.group[i], id: e.id[i], fixed: e.fixed[i] })
    }
    return out
  }, [run.eval, evalSet, hasFixed, hidden, useDeployed, isFabs, mac])

  const m = metricRow(run.metrics, hasFixed ? evalSet : 'all', isFabs ? mac : null)
  const slope = m ? (est === 'Deming' ? m.deming_slope : m.ols_slope) : null
  const icpt = m ? (est === 'Deming' ? m.deming_intercept : m.ols_intercept) : null

  const cols = width > 760 ? 2 : 1
  const panelW = Math.floor((width - (cols - 1) * 12) / cols)
  const S = Math.max(220, Math.min(460, panelW - MARGIN.left - MARGIN.right))
  const hi = Math.max(d3.max(pts, (p) => p.x) ?? 1, d3.max(pts, (p) => p.y) ?? 1) * 1.06
  const lo = Math.min(0, d3.min(pts, (p) => p.y) ?? 0) * 1.05
  const x = d3.scaleLinear().domain([Math.min(0, lo), hi]).range([0, S]).nice()
  const y = d3.scaleLinear().domain([lo, hi]).range([S, 0]).nice()
  const res = pts.map((p) => ({ ...p, r: p.y - p.x }))
  const rExt = d3.extent(res, (p) => p.r) as [number, number]
  const rPad = Math.max(0.2, ((rExt[1] ?? 1) - (rExt[0] ?? 0)) * 0.08)
  const ry = d3.scaleLinear().domain([Math.min(0, rExt[0] ?? 0) - rPad, Math.max(0, rExt[1] ?? 1) + rPad]).range([S * 0.6, 0]).nice()
  const xLabel = useDeployed ? 'deployed SPARTAN EC (µg/m³)' : isFabs ? `HIPS Fabs ÷ MAC ${mac} (µg/m³)` : 'reference EC (µg/m³)'
  const groups = [...new Set(run.eval.group)]
  const target = runs.targets[run.target]

  const enter = (e: React.MouseEvent, p: (typeof pts)[number]) => {
    if (p.id) hl.setHover(p.id)
    tip.show(e, [
      p.id ?? `filter #${p.i + 1}`,
      `${target?.site ?? run.target} · ${run.eval.date?.[p.i] ?? 'no date'} · ${p.g}`,
      `${xLabel}: ${fmt(p.x, 3)}`,
      `predicted EC: ${fmt(p.y, 3)} µg/m³ · residual ${fmt(p.y - p.x, 3)}`,
      p.fixed ? 'fixed evaluation set' : 'outside the fixed set',
      p.id && knownIds.has(p.id) ? 'click for the filter record' : '',
    ].filter(Boolean))
  }
  const leave = () => { hl.setHover(null); tip.hide() }
  const click = (p: (typeof pts)[number]) => { if (p.id && knownIds.has(p.id)) hl.openSample(p.id) }

  return (
    <ChartFrame
      id="run-crossplot"
      title={`Crossplot — predicted EC vs ${useDeployed ? 'deployed EC' : 'HIPS reference'} at ${target?.site ?? run.target}`}
      subtitle="Every evaluation filter under this configuration, coloured by season. The fitted line is the explorer's own metrics row for the chosen estimator, evaluation set and MAC. The residual panel shows whether the calibration's error is a constant offset (a flat band) or tracks loading (a slope)."
      provenance={`calibration_explorer /api/run · ${run.n_cohort} cohort filters, k = ${run.k}${run.k !== run.auto_k ? ` (rule ${run.auto_k})` : ''} · protocol A`}
      controls={
        <>
          {hasDeployed && <Segmented label="x" value={xmode} options={XMODES} onChange={setXmode} />}
          <Segmented label="estimator" value={est} options={ESTIMATORS} onChange={setEst} />
          {hasFixed && <Segmented label="evaluation set" value={evalSet} options={SETS} onChange={setEvalSet} title="fixed = the filters with a deployed EC (Addis fixed-190); all = every filter with a reference" />}
          {isFabs && macs.length > 1 && <Segmented label="MAC" value={String(mac)} options={macs.map(String)} onChange={(v) => setMac(Number(v))} />}
          <span className="control">n = {pts.length}{run.extrap_pct !== null ? ` · score OOD ${run.extrap_pct}%` : ''}{run.q_residual_pct !== null ? ` · Q OOD ${run.q_residual_pct}%` : ''}</span>
        </>
      }
    >
      <div ref={wrapRef} className="chart-wrap">
        {pts.length < 3 ? (
          <Empty>Fewer than 3 filters in this view.</Empty>
        ) : (
          <div className="facet-grid" style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }}>
            <div>
              <svg width={S + MARGIN.left + MARGIN.right} height={S + MARGIN.top + MARGIN.bottom} className="animated">
                <defs><clipPath id={clipId}><rect width={S} height={S} /></clipPath></defs>
                <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
                  <YAxis scale={y} x={0} label="predicted FTIR EC (µg/m³)" gridWidth={S} />
                  <XAxis scale={x} y={S} label={xLabel} />
                  <g clipPath={`url(#${clipId})`}>
                    <line x1={x(0)} y1={y(0)} x2={x(hi)} y2={y(hi)} stroke={INK.identity} strokeWidth={1.5} strokeDasharray="5 4" />
                    {slope !== null && icpt !== null && !useDeployed && (
                      <line x1={x(x.domain()[0])} y1={y(slope * x.domain()[0] + icpt)} x2={x(x.domain()[1])} y2={y(slope * x.domain()[1] + icpt)} stroke={est === 'Deming' ? INK.deming : INK.fit} strokeWidth={2} strokeDasharray={est === 'Deming' ? '7 3' : undefined} />
                    )}
                  </g>
                  {pts.map((p) => {
                    const st = focusStyle(p.id ?? `#${p.i}`, hl.focusId)
                    return (
                      <circle key={p.i} cx={x(p.x)} cy={y(p.y)} r={st.r} fill={color(p.g)} fillOpacity={st.opacity} stroke={st.stroke} strokeWidth={st.strokeWidth}
                        style={{ cursor: p.id && knownIds.has(p.id) ? 'pointer' : 'default' }}
                        onMouseEnter={(e) => enter(e, p)} onMouseLeave={leave} onClick={() => click(p)} />
                    )
                  })}
                  {m && !useDeployed && (
                    <g transform="translate(10,10)" fontFamily={FONT.mono} fontSize={11} pointerEvents="none">
                      <rect width={196} height={62} rx={5} fill="#fff" fillOpacity={0.93} stroke={INK.border} />
                      <text x={9} y={18} fill={INK.text}>n = {m.n} · {m.evaluation_set} set{isFabs ? ` · MAC ${m.MAC}` : ''}</text>
                      <text x={9} y={33} fill={est === 'Deming' ? INK.deming : INK.fit}>{est}: y = {fmt(slope, 3)}x {icpt !== null && icpt < 0 ? '−' : '+'} {fmt(Math.abs(icpt ?? 0), 3)}</text>
                      <text x={9} y={48} fill={INK.muted}>R² = {fmt(m.R2 ?? null, 3)}{run.heldout ? ` · held-out TOR R² ${fmt(run.heldout.R2, 2)}` : ''}</text>
                    </g>
                  )}
                </g>
              </svg>
            </div>
            <div>
              <p className="facet-title">residuals (predicted − {useDeployed ? 'deployed' : 'reference'})</p>
              <svg width={S + MARGIN.left + MARGIN.right} height={S * 0.6 + MARGIN.top + MARGIN.bottom} className="animated">
                <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
                  <YAxis scale={ry} x={0} label="residual (µg/m³)" gridWidth={S} tickCount={5} />
                  <XAxis scale={x} y={S * 0.6} label={xLabel} />
                  <line x1={0} x2={S} y1={ry(0)} y2={ry(0)} stroke={INK.text} strokeWidth={1.4} />
                  {res.map((p) => {
                    const st = focusStyle(p.id ?? `#${p.i}`, hl.focusId, { r: 3.2, opacity: 0.7 })
                    return <circle key={p.i} cx={x(p.x)} cy={ry(p.r)} r={st.r} fill={color(p.g)} fillOpacity={st.opacity} stroke={st.stroke} strokeWidth={st.strokeWidth} onMouseEnter={(e) => enter(e, p)} onMouseLeave={leave} onClick={() => click(p)} style={{ cursor: p.id && knownIds.has(p.id) ? 'pointer' : 'default' }} />
                  })}
                </g>
              </svg>
            </div>
          </div>
        )}
        <Legend items={groups.map((g) => ({ label: g, color: color(g), detail: `n=${run.eval.group.filter((x) => x === g).length}` }))} hidden={hidden} onToggle={(l) => setHidden((h) => toggleIn(h, l))} note="dashed grey = 1:1" />
        {tip.node}
      </div>
    </ChartFrame>
  )
}
