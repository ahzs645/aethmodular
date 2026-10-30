import { demingLambda, useMac } from '@/lib/mac'
import { useMemo, useRef } from 'react'
import { useChartOptions } from '@/lib/chartOptions'
import { useColourScheme } from '@/lib/colourSchemes'
import * as d3 from 'd3'
import { ChartFrame, Empty } from '@/components/ChartFrame'
import { Legend, useLegend } from '@/components/Legend'
import { XAxis, YAxis } from '@/components/Axes'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { baseFilterId, focusLast, focusStyle, useHighlight, type FilterRecord } from '@/lib/highlight'
import { Crossplot } from '@/components/Crossplot'
import { fmt } from '@/lib/stats'
import { INK, MARGIN, withUnit } from '@/lib/theme'
import { AxisChoices, type AxisOption } from '@/components/AxisChoices'
import { useSearchParam } from '@/lib/url'
import type { CalibRun, CalibRunsFile, FilterRow, MetaFile } from '@/lib/types'
import { deployedEC, groupColorFn, presetName, useNetworkRows } from './common'
import { useFitHeight } from '@/lib/chartFit'

const XMODES = ['reference', 'deployed EC'] as const
const SETS = ['fixed', 'all'] as const
// the residual panel's own axes (components/AxisChoices): this calibration's values, or any gallery field
const REF = 'Crossplot x (reference)', PRED = 'Predicted FTIR EC', RESID = 'Residual (predicted − x)'

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
 *
 * The right panel is the explorer: residual against the reference by default, but either axis
 * (click its title) can be the predicted EC, the residual, or any gallery measurement, ratio or
 * weather field of the same filter (residual against iron, against EC / PM2.5 …), from
 * `galleryRows` under the gallery's view.
 */
export function RunCrossplot({ run, runs, meta, knownIds, lambda10: lambdaProp, galleryRows = [], sourceRows, galleryMeta }: {
  run: CalibRun; runs: CalibRunsFile; meta: MetaFile; knownIds: Set<string>; lambda10?: number
  galleryRows?: FilterRow[]; galleryMeta?: MetaFile
  /** every filter's row before the view's options, for its deployed SPARTAN EC (default: galleryRows) */
  sourceRows?: FilterRow[]
}) {
  const fit = useFitHeight()
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
  const hl = useHighlight()

  const { hidden, dim, props: legendProps } = useLegend()

  // the MAC: the one set under Filters (lib/mac) by default, or one the explorer saved fits at
  const gallery = useMac()
  const lambda10 = lambdaProp ?? gallery.lambda10
  const macs = useMemo(() => [...new Set(run.metrics.map((m) => m.MAC).filter((v): v is number => v !== null))].sort((a, b) => a - b), [run.metrics])
  const isFabs = run.ref_kind === 'fabs'
  // the deployed SPARTAN EC: the run's own column (Addis fixed set) or the gallery's EC (FTIR) for the filter
  const byId = useMemo(() => new Map(galleryRows.map((r) => [r.id, r])), [galleryRows])
  const sourceById = useMemo(() => (sourceRows ? new Map(sourceRows.map((r) => [r.id, r])) : byId), [sourceRows, byId])
  const network = useNetworkRows([runs.targets[run.target]?.code ?? ''].filter(Boolean))
  const deployed = (i: number) => deployedEC(run, i, sourceById, network)
  const hasDeployed = run.eval.pred.some((_, i) => deployed(i) !== null)
  const hasFixed = run.eval.fixed.some(Boolean)
  // the options and when each applies (lib/chartOptions): on the deployed-EC axis only the fixed
  // filters carry a value, so the test-set switch is moot, and the MAC (which converts Fabs) is unused
  const o = useChartOptions('run-crossplot', [
    // the crossplot's x (the HIPS reference or the deployed SPARTAN EC), picked from its axis title
    { key: 'x', axis: 'x', label: 'Crossplot x', choices: XMODES.map((v) => ({ value: v, label: v === 'reference' ? 'HIPS reference (Fabs ÷ MAC)' : 'Deployed SPARTAN EC' })), default: 'reference', hidden: 'reference', when: (c) => !!c.hasDeployed },
    { key: 'set', label: 'Test set', choices: SETS, default: 'fixed', hidden: 'fixed', when: (c) => !!c.hasFixed && c.x !== 'deployed EC',
      title: 'fixed = the filters with a deployed EC (Addis fixed-190); all = every filter with a reference' },
    { key: 'mac', label: 'MAC', choices: [{ value: 'gallery', label: `Filters (${gallery.mac})` }, ...macs.map((m) => ({ value: String(m), label: String(m) }))], default: 'gallery', hidden: 'gallery',
      when: (c) => !!c.isFabs && c.x !== 'deployed EC', title: 'Filters: the MAC set under Filters → units; or one the explorer saved fits at' },
    // only groups that are not seasons take the set's colours: no picker when every group is a season
    { key: 'colours', node: () => groupColours.others.length > 0 ? colours.picker : null },
    { key: 'n', node: () => <span className="control">n = {pts.length}{run.extrap_pct !== null ? ` · score OOD ${run.extrap_pct}%` : ''}{run.q_residual_pct !== null ? ` · Q OOD ${run.q_residual_pct}%` : ''}</span> },
  ], { hasDeployed, hasFixed, isFabs })
  const xmode = o.values.x as (typeof XMODES)[number]
  const evalSet = o.values.set as (typeof SETS)[number]
  const mac = o.values.mac === 'gallery' ? gallery.mac : Number(o.values.mac)
  const useDeployed = xmode === 'deployed EC' && hasDeployed
  const colours = useColourScheme('run-crossplot', 'categorical', 'Tableau 10')
  const groupColours = useMemo(() => groupColorFn(meta, run.eval.group, colours.name, colours.reverse), [meta, run.eval.group, colours.name, colours.reverse])
  const color = groupColours.color

  const pts = useMemo(() => {
    const e = run.eval
    const out: { i: number; x: number; y: number; g: string; id: string | null; fixed: boolean }[] = []
    for (let i = 0; i < e.pred.length; i++) {
      if (evalSet === 'fixed' && hasFixed && !e.fixed[i]) continue
      if (hidden.has(e.group[i])) continue
      const x = useDeployed ? deployed(i) : isFabs ? e.ref[i] / mac : e.ref[i]
      if (x === null || !Number.isFinite(x) || !Number.isFinite(e.pred[i])) continue
      out.push({ i, x, y: e.pred[i], g: e.group[i], id: e.id[i], fixed: e.fixed[i] })
    }
    return out
  }, [run.eval, evalSet, hasFixed, hidden, useDeployed, isFabs, mac, sourceById, network]) // eslint-disable-line react-hooks/exhaustive-deps


  // the two square panels (crossplot, residuals) side by side or stacked; in a fitted tile, whichever
  // gives the larger squares, sized to the tile (a facet title of ~22 px above each)
  const margins = MARGIN.left + MARGIN.right
  const sideBy = (w: number, h: number) => Math.min((w - 12) / 2 - margins, h - 22 - MARGIN.top - MARGIN.bottom)
  const stacked = (w: number, h: number) => Math.min(w - margins, (h - 12) / 2 - 22 - MARGIN.top - MARGIN.bottom)
  const cols = fit === null ? (width > 760 ? 2 : 1) : sideBy(width, fit) >= stacked(width, fit) ? 2 : 1
  const panelW = Math.floor((width - (cols - 1) * 12) / cols)
  const fitSide = fit === null ? null : Math.max(140, Math.min(460, cols === 2 ? sideBy(width, fit) : stacked(width, fit)))
  const S = fitSide ?? Math.max(220, Math.min(460, panelW - margins))
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

  // the residual panel's axes: this calibration's three values, then the gallery's fields for these filters
  const options = useMemo<AxisOption[]>(() => {
    const ids = new Set(pts.flatMap((p) => (p.id ? [baseFilterId(p.id)] : [])))
    const mine = galleryRows.filter((r) => ids.has(r.id))
    return [
      { key: REF, group: 'This calibration', unit: 'µg/m³' }, { key: PRED, group: 'This calibration', unit: 'µg/m³' }, { key: RESID, group: 'This calibration', unit: 'µg/m³' },
      ...(galleryMeta?.field_groups ?? []).flatMap((g) => g.fields.filter((f) => galleryMeta!.fields.includes(f) && mine.some((r) => typeof r[f] === 'number'))
        .map((f) => ({ key: f, group: g.label, unit: galleryMeta!.field_units?.[f] }))),
    ]
  }, [pts, galleryRows, galleryMeta])
  const keys = options.map((o) => o.key)
  const [rx, setRx] = useSearchParam<string>('rcx', REF, keys)
  const [rY, setRy] = useSearchParam<string>('rcy', RESID, keys)
  const valueOf = (k: string, p: (typeof pts)[number]) => {
    if (k === REF) return p.x
    if (k === PRED) return p.y
    if (k === RESID) return p.y - p.x
    const v = p.id ? byId.get(baseFilterId(p.id))?.[k] : undefined
    return typeof v === 'number' ? v : NaN
  }
  const explore = res.flatMap((p) => { const ex = valueOf(rx, p), ey = valueOf(rY, p); return Number.isFinite(ex) && Number.isFinite(ey) ? [{ ...p, ex, ey }] : [] })
  const span = (v: number[]) => { const e = d3.extent(v) as [number, number]; const pad = Math.max(1e-9, ((e[1] ?? 1) - (e[0] ?? 0)) * 0.06); return [Math.min(0, (e[0] ?? 0) - pad), (e[1] ?? 1) + pad] as [number, number] }
  // the default pair keeps the crossplot's x scale and the residual scale, so the two panels read side by side
  const ex = rx === REF ? x : d3.scaleLinear().domain(span(explore.map((p) => p.ex))).range([0, S]).nice()
  const ey = rY === RESID ? ry : d3.scaleLinear().domain(span(explore.map((p) => p.ey))).range([S, 0]).nice()
  const axisLabel = (k: string) => (k === REF ? xLabel : k === PRED ? 'predicted FTIR EC (µg/m³)' : k === RESID ? 'residual (µg/m³)' : withUnit(k, galleryMeta?.field_units ?? {}))

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
      ['deployed SPARTAN EC (µg/m³)', deployed(p.i)],
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
      controls={o.controls()}
    >
      <div ref={wrapRef} className="chart-wrap">
        {pts.length < 3 ? (
          <Empty>Fewer than 3 filters in this view.</Empty>
        ) : (
          <div className="facet-grid" style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }}>
            <div>
              <AxisChoices {...o.axisChoices({ x: 'x' })}>
              <Crossplot title={`predicted vs ${useDeployed ? 'deployed' : 'reference'}`} maxSide={fitSide ?? 460} domain={x.domain() as [number, number]} pick={hasDeployed ? 'x' : undefined}
                errorRatio={demingLambda(isFabs && !useDeployed ? mac : null, lambda10)} xLabel={xLabel} yLabel="predicted FTIR EC (µg/m³)"
                points={pts.map((p) => ({ x: p.x, y: p.y, color: color(p.g), opacity: dim(p.g), id: p.id ?? undefined, tip: tipLines(p), rec: record(p) }))} />
              </AxisChoices>
            </div>
            <div>
              <p className="facet-title">{rx === REF && rY === RESID ? `residuals (predicted − ${useDeployed ? 'deployed' : 'reference'})` : `${rY} against ${rx}`}
                {explore.length < pts.length && ` · ${explore.length} of ${pts.length} filters carry both`}</p>
              <AxisChoices noun="measurements" options={options} x={[rx, setRx]} y={[rY, setRy]}>
              <svg width={S + MARGIN.left + MARGIN.right} height={S + MARGIN.top + MARGIN.bottom} className="animated">
                <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
                  <YAxis scale={ey} x={0} label={axisLabel(rY)} gridWidth={S} tickCount={5} pick="y" />
                  <XAxis scale={ex} y={S} label={axisLabel(rx)} pick="x" />
                  {rY === RESID && <line x1={0} x2={S} y1={ey(0)} y2={ey(0)} stroke={INK.text} strokeWidth={1.4} />}
                  {focusLast(explore, (p) => p.id ?? `#${p.i}`, hl.focusId).map(([p]) => {
                    const st = focusStyle(p.id ?? `#${p.i}`, hl.focusId, { r: 3.2 })
                    return <circle key={p.i} cx={ex(p.ex)} cy={ey(p.ey)} r={st.r} fill={color(p.g)} fillOpacity={st.opacity * dim(p.g)} stroke={st.stroke} strokeWidth={st.strokeWidth} data-dot={dim(p.g) === 1 ? st.dot : undefined} onMouseEnter={(e) => enter(e, p)} onMouseLeave={leave} onClick={() => click(p)} style={{ cursor: 'pointer' }} />
                  })}
                </g>
              </svg>
              </AxisChoices>
            </div>
          </div>
        )}
        <Legend items={groups.map((g) => ({ label: g, color: color(g), detail: `n=${run.eval.group.filter((x) => x === g).length}` }))} {...legendProps} note="dashed grey = 1:1 · regression fits selected in the sidebar" />
        {tip.node}
      </div>
    </ChartFrame>
  )
}
