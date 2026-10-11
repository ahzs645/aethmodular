import { demingLambda } from './seasonView'
import { useMemo, useRef, useState } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Segmented } from '@/components/ChartFrame'
import { Legend, useLegend } from '@/components/Legend'
import { XAxis, YAxis } from '@/components/Axes'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { baseFilterId, focusStyle, useHighlight, type FilterRecord } from '@/lib/highlight'
import { Crossplot } from '@/components/Crossplot'
import { fmt } from '@/lib/stats'
import { INK, MARGIN } from '@/lib/theme'
import type { CalibRun, CalibRunsFile, MetaFile } from '@/lib/types'
import { groupColorFn, presetName } from './common'

const XMODES = ['reference', 'deployed EC'] as const
const SETS = ['fixed', 'all'] as const

/**
 * The explorer's Calibrate crossplot and its residual panel, for one preset
 * at one evaluation site: predicted FTIR EC against the HIPS reference
 * (Fabs ÷ MAC), coloured by season, with the fitted line from the same
 * metrics row the explorer quotes (Deming only, agreed with Ann 23 Sep 2026:
 * both axes carry measurement error). The residual panel is the meeting's check:
 * does the correction remove the curve, or only shift it?
 *
 * "deployed EC" swaps the x axis for the deployed SPARTAN EC on the same
 * filters (Addis only), which is the explorer's vs-deployed panel.
 * Clicking an Addis point opens the gallery's sample drawer for that filter.
 */
export function RunCrossplot({ run, runs, meta, knownIds, lambda10 = 2.96 }: { run: CalibRun; runs: CalibRunsFile; meta: MetaFile; knownIds: Set<string>; lambda10?: number }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
  const hl = useHighlight()

  const [xmode, setXmode] = useState<(typeof XMODES)[number]>('reference')
  const [evalSet, setEvalSet] = useState<(typeof SETS)[number]>('fixed')
  const [mac, setMac] = useState<number>(runs.mac_value)
  const { hidden, dim, props: legendProps } = useLegend()

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


  const cols = width > 760 ? 2 : 1
  const panelW = Math.floor((width - (cols - 1) * 12) / cols)
  const S = Math.max(220, Math.min(460, panelW - MARGIN.left - MARGIN.right))
  const hi = Math.max(d3.max(pts, (p) => p.x) ?? 1, d3.max(pts, (p) => p.y) ?? 1) * 1.06
  const lo = Math.min(0, d3.min(pts, (p) => p.y) ?? 0) * 1.05
  // one domain for both axes (the shared crossplot is square); the residual panel reuses its x
  const x = d3.scaleLinear().domain([Math.min(0, lo), hi]).range([0, S]).nice()
  const res = pts.map((p) => ({ ...p, r: p.y - p.x }))
  const rExt = d3.extent(res, (p) => p.r) as [number, number]
  const rPad = Math.max(0.2, ((rExt[1] ?? 1) - (rExt[0] ?? 0)) * 0.08)
  // the residual panel matches the crossplot's size so the pair reads side by side on one x scale
  const ry = d3.scaleLinear().domain([Math.min(0, rExt[0] ?? 0) - rPad, Math.max(0, rExt[1] ?? 1) + rPad]).range([S, 0]).nice()
  const xLabel = useDeployed ? 'deployed SPARTAN EC (µg/m³)' : isFabs ? `HIPS Fabs ÷ MAC ${mac} (µg/m³)` : 'reference EC (µg/m³)'
  const groups = [...new Set(run.eval.group)]
  const target = runs.targets[run.target]

  const tipLines = (p: (typeof pts)[number]) => [
    p.id ?? `filter #${p.i + 1}`,
    `${target?.site ?? run.target} · ${run.eval.date?.[p.i] ?? 'no date'} · ${p.g}`,
    `${xLabel}: ${fmt(p.x, 3)}`,
    `predicted EC: ${fmt(p.y, 3)} µg/m³ · residual ${fmt(p.y - p.x, 3)}`,
    p.fixed ? 'fixed test set' : 'outside the fixed test set',
  ]
  const enter = (e: React.MouseEvent, p: (typeof pts)[number]) => {
    if (p.id) hl.setHover(p.id)
    tip.show(e, [...tipLines(p), p.id && knownIds.has(baseFilterId(p.id)) ? 'click for the full filter record' : 'click for this filter\'s values'])
  }
  const leave = () => { hl.setHover(null); tip.hide() }
  const record = (p: (typeof pts)[number]): FilterRecord => ({
    id: p.id ?? `${run.target} filter #${p.i + 1}`,
    source: 'the calibration crossplot',
    site: target?.site ?? run.target,
    date: run.eval.date?.[p.i] ?? undefined,
    fields: [
      ['configuration', (() => { const q = runs.presets.find((p) => p.key === run.preset); return q ? presetName(q) : run.preset })()],
      ['season / group', p.g],
      [xLabel, p.x],
      ['predicted FTIR EC (µg/m³)', p.y],
      ['residual, predicted − x (µg/m³)', p.y - p.x],
      ['deployed SPARTAN EC (µg/m³)', run.eval.deployed?.[p.i] ?? null],
      ['in the fixed test set', p.fixed ? 'yes' : 'no'],
    ],
  })
  const click = (p: (typeof pts)[number]) => hl.openRecord(record(p))

  return (
    <ChartFrame
      id="run-crossplot"
      title={`Crossplot — predicted EC vs ${useDeployed ? 'deployed EC' : 'HIPS reference'} at ${target?.site ?? run.target}`}
      subtitle="Every test-set filter under this configuration, coloured by season. The regression fits selected in the sidebar use the displayed filters and MAC. The residual panel shows whether the calibration's error is a constant offset (a flat band) or tracks loading (a slope)."
      provenance={`calibration_explorer /api/run · ${run.n_cohort} calibration cohort filters, ${run.k} PLS factors${run.k !== run.auto_k ? ` (rule ${run.auto_k})` : ''} · site-grouped CV`}
      controls={
        <>
          {hasDeployed && <Segmented label="x" value={xmode} options={XMODES} onChange={setXmode} />}
          {hasFixed && <Segmented label="Test set" value={evalSet} options={SETS} onChange={setEvalSet} title="fixed = the filters with a deployed EC (Addis fixed-190); all = every filter with a reference" />}
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
              <Crossplot title={`predicted vs ${useDeployed ? 'deployed' : 'reference'}`} maxSide={460} domain={x.domain() as [number, number]}
                errorRatio={demingLambda(isFabs && !useDeployed ? mac : null, lambda10)} xLabel={xLabel} yLabel="predicted FTIR EC (µg/m³)"
                points={pts.map((p) => ({ x: p.x, y: p.y, color: color(p.g), opacity: dim(p.g), id: p.id ?? undefined, tip: tipLines(p), rec: record(p) }))} />
            </div>
            <div>
              <p className="facet-title">residuals (predicted − {useDeployed ? 'deployed' : 'reference'})</p>
              <svg width={S + MARGIN.left + MARGIN.right} height={S + MARGIN.top + MARGIN.bottom} className="animated">
                <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
                  <YAxis scale={ry} x={0} label="residual (µg/m³)" gridWidth={S} tickCount={5} />
                  <XAxis scale={x} y={S} label={xLabel} />
                  <line x1={0} x2={S} y1={ry(0)} y2={ry(0)} stroke={INK.text} strokeWidth={1.4} />
                  {res.map((p) => {
                    const st = focusStyle(p.id ?? `#${p.i}`, hl.focusId, { r: 3.2, opacity: 0.7 })
                    return <circle key={p.i} cx={x(p.x)} cy={ry(p.r)} r={st.r} fill={color(p.g)} fillOpacity={st.opacity * dim(p.g)} stroke={st.stroke} strokeWidth={st.strokeWidth} onMouseEnter={(e) => enter(e, p)} onMouseLeave={leave} onClick={() => click(p)} style={{ cursor: 'pointer' }} />
                  })}
                </g>
              </svg>
            </div>
          </div>
        )}
        <Legend items={groups.map((g) => ({ label: g, color: color(g), detail: `n=${run.eval.group.filter((x) => x === g).length}` }))} {...legendProps} note="dashed grey = 1:1 · regression fits selected in the sidebar" />
        {tip.node}
      </div>
    </ChartFrame>
  )
}
