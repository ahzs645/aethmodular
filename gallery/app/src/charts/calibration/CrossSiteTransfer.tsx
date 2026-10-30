import { RegressionBand } from '@/components/RegressionOverlay'
import { IdentityLine, RefLine } from '@/components/RefLines'
import { confidence, monthBlocks } from '@/lib/regressionCI'
import { regression } from '@/lib/stats'
import { demingLambda, useMac } from '@/lib/mac'
import { EstimatorName, FIT_STYLE, useEstimator, useRegressions, fitsFor, type Estimator } from '@/lib/regressionPrefs'
import { useChartOptions } from '@/lib/chartOptions'
import { useMemo, useRef, useState } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty } from '@/components/ChartFrame'
import { AxisTitle, XAxis, YAxis } from '@/components/Axes'
import { prettyUnit } from '@/components/FieldSelect'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { fmt } from '@/lib/stats'
import { INK, FONT } from '@/lib/theme'
import type { CalibRun, CalibRunsFile, FilterRow, MetaFile } from '@/lib/types'
import { cohortColor, deployedECFrom, fmtFit, presetName, useNetworkRows } from './common'
import { metricRowLive, type KeepIndex } from './runFit'
import { COHORT, PREPROCESSING, label } from '@/lib/labels'
import { AxisChoices, type AxisOption } from '@/components/AxisChoices'
import { useSearchParam } from '@/lib/url'
import { baseFilterId, dotFill, dotRim, useHighlightActions, type FilterRecord } from '@/lib/highlight'
import { ScatterLayer } from '@/components/ScatterLayer'
import { fidProps } from '@/lib/filterEdits'
import { PanelAxesControl, usePanelAxes } from '@/lib/panelAxes'
import { collapsePanelAxes } from '@/lib/axes'
import { ChartAside } from '@/lib/singleView'
import { ExcludedNote, useCountedFilter, useExclMode, useExclStatus, useMarkReason, type MarkPoint } from '@/lib/markedRows'
import { markTip } from '@/lib/exclusionMode'

/** Ethiopian sites first (the calibration's own target, then its neighbour), then the other SPARTAN sites. */
const SITE_ORDER = ['addis', 'etbi', 'indh', 'chts', 'uspa']
const DOT_METRICS = ['intercept', 'slope', 'R²'] as const
/** The explorer's own threshold for "partially out of domain" (CROSS_SITE_EVALUATION_2026-08-22.md). */
const EXTRAP_FLAG = 20
const SPECTRA_SHAPE: Record<string, d3.SymbolType> = { raw: d3.symbolCircle, airspec: d3.symbolSquare, deriv2: d3.symbolDiamond }

const TRANSFER_TIP = (
  <>
    <p>Every calibration here is trained on IMPROVE filters with thermal (TOR) EC, then applied unchanged to each SPARTAN site. No SPARTAN site has thermal EC, so the site readouts compare the prediction with HIPS Fabs ÷ MAC, an optical equivalent. They show how the calibration transfers, not EC accuracy at the site.</p>
    <p>A slope mixes the site's true MAC with calibration error, so read the intercept ordering before the slopes. Score OOD is the share of site filters beyond the training 95th percentile in the model's score space (Reggente 2016). Q OOD is the share with unmodelled spectral variation beyond that percentile. Above {EXTRAP_FLAG}% the site is partly outside the training domain.</p>
    <p>For citation, the per-filter weighted York fits in research/ftir_ec_phase3/OFFSET_ADJUDICATION_2026-08-23.md supersede these Deming numbers. The HIPS diagnostics chart on this tab shows them for ocec-450 × Spline baseline.</p>
    <p>The sidebar selects the displayed estimator. Saved York views use constant errors with the saved Deming ratio. The IMPROVE cross-validation row reports R² and RMSE only.</p>
  </>
)

type Preset = CalibRunsFile['presets'][number]
const sizeOf = (p: Preset) => (p.cutoff == null ? 'complete' : String(p.cutoff))
/** a preset's own note, e.g. 'Lowest-OC/EC (800) × AIRSpec (locked)' → 'locked' */
const noteOf = (p: Preset) => p.label.match(/×[^(]*\(([^)]*)\)\s*$/)?.[1]

/**
 * One IMPROVE-trained calibration read out at every SPARTAN site with FTIR
 * spectra and HIPS Fabs: the explorer's Sites tab. Three views share the
 * picker: the per-site crossplots, a summary table against the IMPROVE
 * cross-validation score (calibration set), and whether the site ordering holds across all the
 * exported calibrations.
 *
 * Every fit is rebuilt from the runs' per-filter readouts (runFit metricRowLive: with every filter counted, the
 * explorer's saved row) on the filters that count under Filters → excluded (lib/markedRows useCountedFilter, by
 * filter id): an excluded filter leaves the panels' points, their fits, the table and the ordering; under Mark the
 * panels draw it as ×. The IMPROVE cross-validation row is the explorer's (training filters, not the gallery's).
 */
export function CrossSiteTransfer({ runs, galleryRows, sourceRows, galleryMeta }: {
  runs: CalibRunsFile
  galleryRows?: FilterRow[]; galleryMeta?: MetaFile
  /** every filter's row before the view's options, for its deployed SPARTAN EC */
  sourceRows?: FilterRow[]
}) {
  const presets = runs.presets.filter((p) => runs.runs.some((r) => r.preset === p.key))

  const sites = useMemo(() => {
    const present = [...new Set(runs.runs.map((r) => r.target))]
    return [...SITE_ORDER.filter((s) => present.includes(s)), ...present.filter((s) => !SITE_ORDER.includes(s)).sort()]
  }, [runs.runs])
  // the MAC: the one set under Filters (lib/mac) by default, or one the explorer saved fits at
  const gallery = useMac()
  const macs = useMemo(() => [...new Set(runs.runs.flatMap((r) => r.metrics.map((m) => m.MAC)).filter((v): v is number => v !== null))].sort((a, b) => a - b), [runs.runs])
  // the options and when each applies (lib/chartOptions): the MAC converts HIPS Fabs, so it is
  // offered only for a calibration referenced to Fabs (and only with more than one MAC)
  // the calibration as three pickers (cohort, its size, the spectra); each offers only what the exported
  // presets have under the choices before it, so every combination drawn is one that was run
  const start = presets.find((p) => p.key === 'ocec800_airspec') ?? presets[0]
  const o = useChartOptions('cross-site', [
    { key: 'cohort', label: 'Cohort', style: 'select', default: start?.cohort ?? '',
      choices: [...new Set(presets.map((p) => p.cohort))].map((c) => ({ value: c, label: label(COHORT, c) })) },
    { key: 'size', label: 'Cohort size', style: 'select', default: start ? sizeOf(start) : '',
      choices: [...new Set(presets.map(sizeOf))].map((z) => ({ value: z, label: z === 'complete' ? 'Complete cohort' : `${z} filters`,
        when: (c) => presets.some((p) => p.cohort === c.cohort && sizeOf(p) === z) })) },
    { key: 'spectra', label: 'Preprocessing', style: 'select', default: start?.spectra ?? '',
      choices: [...new Set(presets.map((p) => p.spectra))].map((sp) => ({ value: sp,
        label: (c) => { const q = presets.find((p) => p.cohort === c.cohort && sizeOf(p) === c.size && p.spectra === sp); const n = q && noteOf(q); return label(PREPROCESSING, sp) + (n ? ` (${n})` : '') },
        when: (c) => presets.some((p) => p.cohort === c.cohort && sizeOf(p) === c.size && p.spectra === sp) })) },
    { key: 'mac', label: 'MAC', choices: [{ value: 'gallery', label: `Filters (${gallery.mac})` }, ...macs.map((m) => ({ value: String(m), label: String(m) }))], default: 'gallery', hidden: 'gallery',
      when: (c) => runs.runs.some((r) => r.ref_kind === 'fabs' && presets.some((p) => p.key === r.preset && p.cohort === c.cohort && sizeOf(p) === c.size && p.spectra === c.spectra)), title: 'Converts HIPS Fabs to BC for calibrations referenced to Fabs. Filters: the MAC set under Filters → units; or one the explorer saved fits at' },
  ])
  const presetKey = presets.find((p) => p.cohort === o.values.cohort && sizeOf(p) === o.values.size && p.spectra === o.values.spectra)?.key ?? start?.key ?? ''
  const mac = o.values.mac === 'gallery' ? gallery.mac : Number(o.values.mac)
  // the fits' MAC context: the saved row at this MAC, or the export's re-expressed (lib/mac); λ ∝ MAC²
  const at: MacAt = { mac, exportMac: runs.mac_value, lambda10: gallery.lambda10 }
  const setPresetKey = (k: string) => { const p = presets.find((q) => q.key === k); if (p) { o.set('cohort', p.cohort); o.set('size', sizeOf(p)); o.set('spectra', p.spectra) } }
  const siteRuns = sites.map((s) => runs.runs.find((r) => r.preset === presetKey && r.target === s)).filter((r): r is CalibRun => !!r)
  const preset = presets.find((p) => p.key === presetKey)

  if (!presets.length) return null
  const controls = o.controls()
  return (
    <>
      <SiteMultiples runs={runs} siteRuns={siteRuns} mac={mac} at={at} controls={controls} presetLabel={preset ? presetName(preset) : presetKey}
        galleryRows={galleryRows} sourceRows={sourceRows} galleryMeta={galleryMeta} />
      <TransferTable runs={runs} siteRuns={siteRuns} mac={mac} at={at} presetLabel={preset ? presetName(preset) : presetKey} />
      <OrderingAcrossCalibrations runs={runs} sites={sites} presetKey={presetKey} mac={mac} at={at} onPick={setPresetKey} />
    </>
  )
}

/** where the fits are read: the chart's MAC, the export's, and λ at MAC 10 */
interface MacAt { mac: number; exportMac: number; lambda10: number }

/**
 * The Deming fit (agreed with Ann, 23 Sep 2026: no y-on-x least squares in reader-facing views; the sidebar's OLS view
 * reads the OLS line, as projectRun does for saved rows), at the chart's MAC, on every pair `keep` counts.
 */
function lineOf(run: CalibRun, at: MacAt, method: Estimator, keep: KeepIndex) {
  const m = metricRowLive(run.eval, 'all', run.ref_kind === 'fabs' ? at.mac : null, at.lambda10, keep)
  if (!m) return null
  const ols = method === 'ols'
  return { m, slope: ols ? m.ols_slope : m.deming_slope, intercept: ols ? m.ols_intercept : m.deming_intercept }
}

/** The same fit with month-block bootstrap groups, for its intervals: the same pairs as lineOf, so they agree. */
function statsFor(run: CalibRun, at: MacAt, method: Estimator, keep: KeepIndex) {
  const mac = at.mac
  const pairs = run.eval.pred.flatMap((y, i) => {
    const x = run.ref_kind === 'fabs' ? run.eval.ref[i] / mac : run.eval.ref[i]
    return keep(i) && Number.isFinite(x) && Number.isFinite(y) ? [{ x, y, i }] : []
  })
  const st = regression(pairs.map(p => p.x), pairs.map(p => p.y), { errorsInVariables: true,
    lambda: demingLambda(run.ref_kind === 'fabs' ? mac : null, at.lambda10),
    bootstrapGroups: monthBlocks(pairs.map(p => ({ site: run.target, date: run.eval.date?.[p.i] }))),
  })
  return fitsFor(st, [method]).length ? st : null
}

/** Filters → excluded's one rule (lib/markedRows useCountedFilter) as each run's KeepIndex, by filter id */
function useRunKeep(): (run: CalibRun) => KeepIndex {
  const isCounted = useCountedFilter()
  return useMemo(() => {
    const cache = new WeakMap<CalibRun, KeepIndex>()
    return (run) => {
      let k = cache.get(run)
      if (!k) { const ok = run.eval.id.map((id) => isCounted(id)); k = (i) => ok[i]; cache.set(run, k) }
      return k
    }
  }, [isCounted])
}

// the readouts this calibration gives every panel; gallery measurements join by filter id
const REF = 'reference', PRED = 'predicted', RESID = 'residual', DEP = 'deployed'
const EC_KEYS = [REF, PRED, DEP]

function SiteMultiples({ runs, siteRuns, mac, at, controls, presetLabel, galleryRows = [], sourceRows, galleryMeta }: {
  runs: CalibRunsFile; siteRuns: CalibRun[]; mac: number; at: MacAt; controls: React.ReactNode; presetLabel: string
  galleryRows?: FilterRow[]; galleryMeta?: MetaFile; sourceRows?: FilterRow[]
}) {
  const estimator = useEstimator()
  const { showCI } = useRegressions()
  const fitStyle = FIT_STYLE[estimator]
  const fitName = fitStyle.label
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
  // Panel axes is the gallery's one setting (lib/panelAxes: the sidebar's, and every multi-panel chart's)
  const [axes] = usePanelAxes()
  // Filters → excluded: the counted filters feed the points and fits; under Mark the excluded are ×
  const keepOf = useRunKeep()
  const status = useExclStatus()
  const reasonOf = useMarkReason()
  const markMode = useExclMode() === 'mark'

  // x and y picked from the axis titles (the gallery's axis popover, as on the correlation crossplots):
  // this calibration's readouts, then any gallery measurement these filters carry
  const byId = useMemo(() => new Map(galleryRows.map((r) => [r.id, r])), [galleryRows])
  const sourceById = useMemo(() => (sourceRows ? new Map(sourceRows.map((r) => [r.id, r])) : byId), [sourceRows, byId])
  // filters the gallery's own rows lack (Bishoftu, much of Delhi) take the public SPARTAN release's value
  const network = useNetworkRows(siteRuns.map((r) => runs.targets[r.target]?.code ?? '').filter(Boolean))
  const deployedOf = (run: CalibRun, i: number) => deployedECFrom(run, i, sourceById, network)
  const hasDeployed = siteRuns.some((r) => r.eval.pred.some((_, i) => deployedOf(r, i) !== null))
  const options = useMemo<AxisOption[]>(() => {
    const ids = new Set(siteRuns.flatMap((r) => r.eval.id.flatMap((id) => (id ? [baseFilterId(id)] : []))))
    const mine = galleryRows.filter((r) => ids.has(r.id))
    return [
      { key: REF, label: `HIPS Fabs ÷ MAC ${mac}`, group: 'This calibration', unit: 'µg/m³' },
      { key: PRED, label: 'Predicted FTIR EC', group: 'This calibration', unit: 'µg/m³' },
      { key: RESID, label: 'Residual, predicted − reference', group: 'This calibration', unit: 'µg/m³' },
      ...(hasDeployed ? [{ key: DEP, label: 'Deployed SPARTAN EC', group: 'This calibration', unit: 'µg/m³' }] : []),
      ...(galleryMeta?.field_groups ?? []).flatMap((g) => g.fields.filter((f) => galleryMeta!.fields.includes(f) && mine.some((r) => typeof r[f] === 'number'))
        .map((f) => ({ key: f, group: g.label, unit: prettyUnit(galleryMeta!.field_units?.[f]) || undefined }))),
    ]
  }, [siteRuns, galleryRows, galleryMeta, mac, hasDeployed])
  const keys = options.map((o) => o.key)
  const [xk, setXk] = useSearchParam<string>('csx', REF, keys)
  const [yk, setYk] = useSearchParam<string>('csy', PRED, keys)
  const titleOf = (k: string) => { const o = options.find((x) => x.key === k); return o ? `${o.label ?? o.key}${o.unit ? ` (${o.unit})` : ''}` : k }
  // the calibration's own pair: square panels, the 1:1 line and the saved fit; any other pair is a plain scatter
  const ecPair = EC_KEYS.includes(xk) && EC_KEYS.includes(yk)
  const savedPair = xk === REF && yk === PRED

  const valueOf = (run: CalibRun, i: number, k: string): number => {
    const e = run.eval
    const ref = run.ref_kind === 'fabs' ? e.ref[i] / mac : e.ref[i]
    if (k === REF) return ref
    if (k === PRED) return e.pred[i]
    if (k === RESID) return e.pred[i] - ref
    if (k === DEP) return deployedOf(run, i)?.v ?? NaN
    const v = e.id[i] ? byId.get(baseFilterId(e.id[i]!))?.[k] : undefined
    return typeof v === 'number' ? v : NaN
  }
  // the same arrays until the readouts change, so a tooltip's re-render leaves every panel's dots alone
  const { pts, out } = useMemo(() => {
    const pts: TransferPoint[][] = [], out: TransferPoint[][] = []
    for (const run of siteRuns) {
      const keep = keepOf(run), mine: TransferPoint[] = [], marked: TransferPoint[] = []
      for (let i = 0; i < run.eval.pred.length; i++) {
        const x = valueOf(run, i, xk), y = valueOf(run, i, yk)
        if (!Number.isFinite(x) || !Number.isFinite(y)) continue
        if (keep(i)) mine.push({ x, y, i })
        else if (status(run.eval.id[i]) === 'marked') marked.push({ x, y, i })
      }
      pts.push(mine); out.push(marked)
    }
    return { pts, out }
  }, [siteRuns, xk, yk, mac, byId, sourceById, network, keepOf, status]) // eslint-disable-line react-hooks/exhaustive-deps
  const nMarked = out.reduce((a, o) => a + o.length, 0)
  const nOut = siteRuns.reduce((a, run) => a + run.eval.pred.filter((_, i) => !keepOf(run)(i)).length, 0)
  const pad = (e: [number, number]): [number, number] => { const d = Math.max(1e-9, (e[1] - e[0]) * 0.06); return [Math.min(0, e[0] - d), e[1] + d] }
  const square = (p: { x: number; y: number }[]): [number, number] => [
    Math.min(0, d3.min(p, (d) => Math.min(d.x, d.y)) ?? 0),
    Math.max(d3.max(p, (d) => d.x) ?? 1, d3.max(p, (d) => d.y) ?? 1) * 1.06,
  ]
  const ext = (v: number[]): [number, number] => pad([d3.min(v) ?? 0, d3.max(v) ?? 1])
  const all = pts.flat()
  const sharedSq = square(all), sharedX = ext(all.map((p) => p.x)), sharedY = ext(all.map((p) => p.y))

  const gap = 18
  // as many columns as fit, then balanced so five sites wrap 3 + 2 rather than 4 + 1
  // every panel plots the same two quantities, so the axis titles are said once for the grid (y down
  // the left, x underneath), not per panel; with shared axes the inner panels also drop their
  // repeated tick numbers (gridlines stay), as in a small-multiples figure
  const titleW = 22
  const gridW = Math.max(0, width - titleW)
  const maxCols = Math.max(1, Math.min(siteRuns.length, Math.floor((gridW + gap) / 230)))
  const cols = Math.ceil(siteRuns.length / Math.ceil(siteRuns.length / maxCols))
  const rows = Math.ceil(siteRuns.length / cols)
  const cellW = (gridW - (cols - 1) * gap) / cols
  const m = { l: 42, r: 8, t: 40, b: 44 }
  const S = Math.max(150, Math.min(300, cellW - m.l - m.r))
  const gridH = rows * (S + m.t + m.b) + (rows - 1) * gap
  const sharedTicks = collapsePanelAxes(axes, 'site') === 'Shared'   // one panel per site
  const caret = <tspan className="axis-pick-caret" dx={4} data-export="skip">▾</tspan>

  return (
    <ChartFrame
      id="cross-site-multiples"
      title={`Every site under one calibration — ${presetLabel}`}
      subtitle="One panel per SPARTAN site: by default the calibration's predicted FTIR EC against HIPS Fabs ÷ MAC, with the 1:1 line (dashed) and the Deming fit on that site's filters (each site is a test set the model never saw), refitted here from each filter's prediction at the MAC set under Filters → units (λ scales with MAC²), on the filters that count under Filters → excluded. With every filter counted it is the explorer's saved fit. Click an axis title for another readout or any gallery measurement these filters carry."
      provenance="calibration_explorer /api/run eval for each target (per-filter predictions) · test set 'all' · fits rebuilt in the browser (charts/calibration/runFit)"
      tip={TRANSFER_TIP}
      controls={<>{controls}{siteRuns.length > 1 && <PanelAxesControl dims={['site']} />}</>}
    >
      <div ref={wrapRef} className="chart-wrap">
        {!siteRuns.length ? (
          <Empty>No site readouts exported for this calibration.</Empty>
        ) : (
          <AxisChoices noun="measurements" options={options} x={[xk, setXk]} y={[yk, setYk]}>
          <div style={{ display: 'flex', alignItems: 'flex-start' }}>
          <svg width={titleW} height={gridH} fontFamily={FONT.family} style={{ flex: 'none', overflow: 'visible' }}>
            <AxisTitle pick="y">
              <text transform={`translate(13,${(m.t + gridH - m.b) / 2}) rotate(-90)`} textAnchor="middle" fontSize={12} fontWeight={600} fill={INK.text}>{titleOf(yk)}{caret}</text>
            </AxisTitle>
          </svg>
          <div style={{ display: 'grid', gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))`, gap, justifyItems: 'center', flex: 1, minWidth: 0 }}>
            {siteRuns.map((run, k) => {
              const t = runs.targets[run.target]
              const xDom = ecPair ? (sharedTicks ? sharedSq : square(pts[k])) : sharedTicks ? sharedX : ext(pts[k].map((p) => p.x))
              const x = d3.scaleLinear().domain(xDom).range([0, S]).nice()
              const y = ecPair ? d3.scaleLinear().domain(x.domain()).range([S, 0])
                : d3.scaleLinear().domain(sharedTicks ? sharedY : ext(pts[k].map((p) => p.y))).range([S, 0]).nice()
              const [d0, d1] = x.domain()
              const fit = savedPair ? lineOf(run, at, estimator, keepOf(run)) : null
              const st = savedPair && showCI ? statsFor(run, at, estimator, keepOf(run)) : null
              const ci = savedPair && showCI ? confidence(st, estimator) : null
              // clip the fitted line to the square
              const seg = fit && fit.slope !== null && fit.intercept !== null ? (() => {
                const f = (v: number) => fit.slope! * v + fit.intercept!
                const xs = [d0, d1, (d0 - fit.intercept!) / fit.slope!, (d1 - fit.intercept!) / fit.slope!]
                  .filter((v) => Number.isFinite(v) && v >= d0 && v <= d1 && f(v) >= d0 - 1e-9 && f(v) <= d1 + 1e-9)
                  .sort((a, b) => a - b)
                return xs.length >= 2 ? { x1: x(xs[0]), y1: y(f(xs[0])), x2: x(xs[xs.length - 1]), y2: y(f(xs[xs.length - 1])) } : null
              })() : null
              const flagged = (run.extrap_pct ?? 0) > EXTRAP_FLAG
              const color = t?.color ?? INK.neutral
              return (
                <svg key={run.target} width={S + m.l + m.r} height={S + m.t + m.b} fontFamily={FONT.family}>
                  <text x={m.l} y={14} fontSize={12.5} fontWeight={600} fill={color}>{t?.site ?? run.target}<tspan fill={INK.muted} fontWeight={400}>{`  ${t?.code ?? ''} · n = ${pts[k].length}`}</tspan></text>
                  <text x={m.l} y={29} fontSize={10.5} fill={flagged ? INK.warn : INK.muted}>
                    {`score OOD ${fmt(run.extrap_pct, 1)} %${flagged ? ' · partly out of domain' : ''}`}
                  </text>
                  <g transform={`translate(${m.l},${m.t})`}>
                    <rect width={S} height={S} fill="none" stroke={flagged ? '#f59e0b' : 'none'} strokeDasharray="4 3" />
                    {/* shared axes: tick numbers only on the left column and the bottom panel of each column */}
                    <YAxis scale={y} x={0} gridWidth={S} tickCount={4} format={sharedTicks && k % cols !== 0 ? () => '' : undefined} />
                    <XAxis scale={x} y={S} tickCount={4} format={sharedTicks && k + cols < siteRuns.length ? () => '' : undefined} />
                    {ecPair && <IdentityLine x1={x(d0)} y1={y(d0)} x2={x(d1)} y2={y(d1)} />}
                    {yk === RESID && y.domain()[0] < 0 && y.domain()[1] > 0 && <line x1={0} x2={S} y1={y(0)} y2={y(0)} stroke={INK.text} strokeWidth={1.2} />}
                    {pts[k].length === 0 && <text x={S / 2} y={S / 2} textAnchor="middle" fontSize={11} fill={INK.muted}>no filters carry both</text>}
                    {/* a dot is a filter: hover and click work as on every other crossplot (highlight on every chart, pin, open its record) */}
                    <TransferPoints pts={pts[k]} x={x} y={y} side={S} color={color} tip={tip}
                      marks={out[k].flatMap((p): MarkPoint[] => {
                        const id = run.eval.id[p.i], [x0, x1] = x.domain(), [y0, y1] = y.domain()
                        if (!id || p.x < x0 || p.x > x1 || p.y < y0 || p.y > y1) return []
                        return [{ id: baseFilterId(id), at: [x(p.x), y(p.y)], tip: markTip({ id, site: t?.site ?? run.target, date: run.eval.date?.[p.i] ?? undefined }, reasonOf(id) ?? 'excluded',
                          [`${titleOf(xk)}: ${fmt(p.x, 3)}`, `${titleOf(yk)}: ${fmt(p.y, 3)}`]) }]
                      })}
                      idOf={(p) => (run.eval.id[p.i] ? baseFilterId(run.eval.id[p.i]!) : null)}
                      tipOf={(p) => [
                        run.eval.id[p.i] ?? `filter #${p.i + 1}`,
                        `${t?.site ?? run.target} · ${run.eval.date?.[p.i] ?? 'no date'} · ${run.eval.group[p.i]}`,
                        `${titleOf(xk)}: ${fmt(p.x, 3)}${xk === DEP && deployedOf(run, p.i)?.from === 'public release' ? ' (public SPARTAN release)' : ''}`,
                        `${titleOf(yk)}: ${fmt(p.y, 3)}${yk === DEP && deployedOf(run, p.i)?.from === 'public release' ? ' (public SPARTAN release)' : ''}`,
                        'click to pin it on every chart and open its record',
                      ]}
                      recordOf={(p): FilterRecord => ({
                        id: run.eval.id[p.i] ?? `${run.target} filter #${p.i + 1}`, source: `Every site under one calibration (${presetLabel})`, site: t?.site ?? run.target,
                        date: run.eval.date?.[p.i] ?? undefined,
                        fields: [['season / group', run.eval.group[p.i]], [titleOf(xk), p.x], [titleOf(yk), p.y]],
                      })}>
                      {seg && <svg width={S} height={S} overflow="hidden" pointerEvents="none"><RegressionBand stats={st} method={estimator} color={fitStyle.color} x={x} y={y} domain={x.domain()} /></svg>}
                      {seg && <line {...seg} pointerEvents="none" stroke={fitStyle.color} strokeWidth={1.8} strokeDasharray={fitStyle.dash} />}
                    </TransferPoints>
                    {savedPair && <text x={S / 2} y={S + 36} textAnchor="middle" fontSize={11} fontFamily={FONT.mono} fill={fitStyle.color}>
                      {`${fitName} ${fmtFit(fit?.slope ?? null, fit?.intercept ?? null)} · R² ${fmt(fit?.m.R2 ?? null, 2)}`}
                      <title>{showCI ? ci ? `${ci.description}; slope [${fmt(ci.slope[0], 3)}, ${fmt(ci.slope[1], 3)}]; intercept [${fmt(ci.intercept[0], 3)}, ${fmt(ci.intercept[1], 3)}]` : '95% CI unavailable: too few filters' : ''}</title>
                    </text>}
                  </g>
                </svg>
              )
            })}
          </div>
          </div>
          <svg width={width} height={22} fontFamily={FONT.family} style={{ display: 'block', overflow: 'visible' }}>
            <AxisTitle pick="x">
              <text x={titleW + gridW / 2 + (m.l - m.r) / 2} y={15} textAnchor="middle" fontSize={12} fontWeight={600} fill={INK.text}>{titleOf(xk)}{caret}</text>
            </AxisTitle>
          </svg>
          </AxisChoices>
        )}
        {nOut > 0 && !markMode && <ChartAside><p className="chart-note">{nOut} excluded filter{nOut === 1 ? '' : 's'} (Filters → excluded) {nOut === 1 ? 'is' : 'are'} hidden here and left out of every site's fit.</p></ChartAside>}
        <ExcludedNote count={nMarked} drawn />
        {tip.node}
      </div>
    </ChartFrame>
  )
}

type TransferPoint = { x: number; y: number; i: number }

/**
 * One site panel's filters (components/ScatterLayer): drawn once per readout and range; a hover draws the focused
 * filter over them. `children` (the saved fit and its band) sit over the dots.
 */
function TransferPoints({ pts, x, y, side, color, tip, idOf, tipOf, recordOf, marks, children }: {
  pts: TransferPoint[]; x: d3.ScaleLinear<number, number>; y: d3.ScaleLinear<number, number>; side: number; color: string
  /** the excluded filters under Filters → excluded: Mark, as × (never in the fit) */
  marks?: MarkPoint[]
  tip: ReturnType<typeof useTooltip>; idOf: (p: TransferPoint) => string | null; tipOf: (p: TransferPoint) => string[]
  recordOf: (p: TransferPoint) => FilterRecord; children: React.ReactNode
}) {
  const hl = useHighlightActions()
  const xy = useMemo(() => pts.map((p) => [x(p.x), y(p.y)] as const), [pts, x.domain().join(), y.domain().join(), side]) // eslint-disable-line react-hooks/exhaustive-deps
  const dots = useMemo(() => pts.map((p, i) => {
    const key = idOf(p)
    return <circle key={p.i} cx={xy[i][0]} cy={xy[i][1]} r={2.6} fill={color} {...dotFill('dense')} stroke={dotRim().stroke} strokeWidth={dotRim().strokeWidth} {...(key ? fidProps(key) : {})} />
  }), [pts, xy, color]) // eslint-disable-line react-hooks/exhaustive-deps
  return <ScatterLayer items={pts} xy={xy} idOf={idOf} dots={dots} width={side} height={side} tip={tip} tipOf={tipOf} focusR={5} marks={marks}
    focus={(_p, [cx, cy]) => <circle cx={cx} cy={cy} r={5} fill={color} fillOpacity={0.9} stroke={INK.text} strokeWidth={1.5} />}
    onPick={(p) => { hl.openRecord(recordOf(p)); const key = idOf(p); if (key) hl.setPinned(key) }}>
    {children}
  </ScatterLayer>
}

function TransferTable({ runs, siteRuns, mac, at, presetLabel }: { runs: CalibRunsFile; siteRuns: CalibRun[]; mac: number; at: MacAt; presetLabel: string }) {
  const { showCI } = useRegressions()
  const estimator = useEstimator()
  const keepOf = useRunKeep()
  const nOut = siteRuns.reduce((a, run) => a + run.eval.pred.filter((_, i) => !keepOf(run)(i)).length, 0)
  const nMarked = useExclusionCount(siteRuns)
  const held = siteRuns.find((r) => r.heldout)?.heldout ?? null
  const train = siteRuns[0]
  return (
    <ChartFrame
      id="cross-site-table"
      title={`Transfer summary — ${presetLabel}`}
      subtitle="The calibration-set check first (IMPROVE cross-validation: whole IMPROVE sites held out of training, scored against thermal EC), then the same calibration on each SPARTAN test set against HIPS Fabs ÷ MAC."
      tip={TRANSFER_TIP}
      exportable={false}
    >
      <div style={{ overflowX: 'auto' }}>
        <table className="census-table">
          <thead>
            <tr><th>Applied to</th><th>Reference</th><th>n</th><th><EstimatorName /> slope</th><th><EstimatorName /> intercept</th><th>R²</th><th>RMSE</th><th>Score OOD</th><th>Q OOD</th><th>Median prediction</th><th>Negative</th></tr>
          </thead>
          <tbody>
            {held && train && (
              <tr style={{ background: 'var(--accent-soft)' }}>
                <td><strong>IMPROVE cross-validation (calibration set)</strong><div style={{ fontSize: 11, color: 'var(--ink-muted)' }}>{`trained on ${train.n_train.toLocaleString()} filters, ${train.n_train_sites} sites · ${train.k} PLS factors`}</div></td>
                <td>TOR EC (µg/filter)</td>
                <td>—</td>
                {/* the explorer exports this row's slope/intercept from a y-on-x fit only; no Deming values to show */}
                <td title="not exported as a Deming fit">—</td>
                <td title="not exported as a Deming fit">—</td>
                <td>{fmt(held.R2, 3)}</td>
                <td>{fmt(held.RMSE, 3)}</td>
                <td>—</td><td>—</td><td>—</td><td>—</td>
              </tr>
            )}
            {siteRuns.map((run) => {
              const t = runs.targets[run.target]
              const fit = lineOf(run, at, estimator, keepOf(run))
              const ci = showCI ? confidence(statsFor(run, at, estimator, keepOf(run)), estimator) : null
              const flagged = (run.extrap_pct ?? 0) > EXTRAP_FLAG
              return (
                <tr key={run.target}>
                  <td><span style={{ color: t?.color, fontWeight: 600 }}>{t?.site ?? run.target}</span> <span style={{ color: 'var(--ink-muted)' }}>{t?.code}</span></td>
                  <td>{run.ref_kind === 'fabs' ? `HIPS Fabs ÷ ${mac}` : 'EC'}</td>
                  <td>{fit?.m.n ?? run.eval.id.length}</td>
                  <td>{fmt(fit?.slope ?? null, 3)}{showCI && <div className="chart-note">{ci ? `95% [${fmt(ci.slope[0], 3)}, ${fmt(ci.slope[1], 3)}]` : '95% CI unavailable'}</div>}</td>
                  <td style={{ fontWeight: (fit?.intercept ?? 0) < -1 ? 600 : 400 }}>{fmt(fit?.intercept ?? null, 3)}{showCI && <div className="chart-note">{ci ? `95% [${fmt(ci.intercept[0], 3)}, ${fmt(ci.intercept[1], 3)}]` : '95% CI unavailable'}</div>}</td>
                  <td>{fmt(fit?.m.R2 ?? null, 3)}</td>
                  <td>{fmt(fit?.m.RMSE ?? null, 3)}</td>
                  <td style={{ color: flagged ? INK.warn : undefined, fontWeight: flagged ? 600 : 400 }}>{`${fmt(run.extrap_pct, 1)} %`}</td>
                  <td>{`${fmt(run.q_residual_pct, 1)} %`}</td>
                  <td>{fmt(run.plausibility?.median ?? null, 2)}</td>
                  <td>{run.plausibility?.negative_pct != null ? `${fmt(run.plausibility.negative_pct, 1)} %` : '—'}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      <ChartAside><p className="chart-note">Intercept and RMSE are in the reference's units: µg/filter for the IMPROVE row, µg/m³ at the SPARTAN sites. Bold intercepts are below −1 µg/m³; an amber score OOD is above {EXTRAP_FLAG} %.
        The site rows are refitted here on the filters that count under Filters → excluded{nOut > 0 ? ` (${nOut} left out)` : ''}; score OOD, Q OOD, the median prediction and the negative share are the explorer's, on every filter.</p></ChartAside>
      <ExcludedNote count={nMarked} />
    </ChartFrame>
  )
}

function OrderingAcrossCalibrations({ runs, sites, presetKey, mac, at, onPick }: {
  runs: CalibRunsFile; sites: string[]; presetKey: string; mac: number; at: MacAt; onPick: (k: string) => void
}) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
  const { showCI } = useRegressions()
  const estimator = useEstimator()
  const [metric, setMetric] = useState<(typeof DOT_METRICS)[number]>('intercept')
  const keepOf = useRunKeep()
  const siteRuns = runs.runs.filter((r) => sites.includes(r.target) && r.preset === presetKey)
  const nMarked = useExclusionCount(siteRuns)

  const rows = runs.presets.flatMap((p) => sites.map((s) => {
    const run = runs.runs.find((r) => r.preset === p.key && r.target === s)
    const fit = run ? lineOf(run, at, estimator, keepOf(run)) : null
    const v = !fit ? null : metric === 'intercept' ? fit.intercept : metric === 'slope' ? fit.slope : fit.m.R2
    const ci = showCI && run && metric !== 'R²' ? confidence(statsFor(run, at, estimator, keepOf(run)), estimator) : null
    return { preset: p, site: s, v, interval: ci && metric !== 'R²' ? ci[metric] : null }
  })).filter((r): r is typeof r & { v: number } => r.v !== null && Number.isFinite(r.v))

  const left = 110, right = 24, rowH = 44
  const iw = Math.max(200, width - left - right)
  const ih = sites.length * rowH
  const ext = d3.extent(rows.flatMap(r => r.interval ? [r.v, ...r.interval] : [r.v])) as [number, number]
  const ref = metric === 'slope' ? 1 : metric === 'intercept' ? 0 : null
  const x = d3.scaleLinear().domain([Math.min(ext[0] ?? 0, ref ?? Infinity), Math.max(ext[1] ?? 1, ref ?? -Infinity)]).range([0, iw]).nice()
  const y = d3.scaleBand().domain(sites).range([0, ih]).padding(0.3)
  const spectra = [...new Set(runs.presets.map((p) => p.spectra))]
  const cohorts = [...new Set(runs.presets.map((p) => p.cohort))]

  return (
    <ChartFrame
      id="cross-site-ordering"
      title="Does the site ordering hold across calibrations?"
      subtitle="Each dot is one exported calibration read out at one site, coloured by cohort, shaped by preprocessing; the ringed dot is the calibration picked above. Click a dot to pick that calibration. A site whose dots stay together is robust to the calibration choice."
      tip={TRANSFER_TIP}
    >
      <div ref={wrapRef} className="chart-wrap">
        {/* the readout is what the x axis shows: picked from its title (components/AxisChoices) */}
        <AxisChoices noun="readouts" x={[metric, (v) => setMetric(v as (typeof DOT_METRICS)[number])]}
          options={DOT_METRICS.map((m) => ({ key: m, label: m === 'R²' ? 'Test set R²' : `Test set Deming ${m}`, group: 'Readout', roles: ['x'] }))}>
        <svg width={width} height={ih + 96} fontFamily={FONT.family}>
          <g transform={`translate(${left},12)`}>
            <XAxis scale={x} y={ih} label={`Test set ${metric === 'R²' ? 'R²' : `Deming ${metric}`}${metric === 'intercept' ? ' (µg/m³)' : ''} · MAC ${mac}`} pick="x" />
            {x.ticks(6).map((t) => <line key={t} x1={x(t)} x2={x(t)} y1={0} y2={ih} stroke={INK.grid} />)}
            {ref !== null && <RefLine orient="v" at={x(ref)} from={0} to={ih} />}
            {sites.map((s) => {
              const t = runs.targets[s]
              return (
                <g key={s}>
                  <text x={-12} y={(y(s) ?? 0) + y.bandwidth() / 2} dy="0.35em" textAnchor="end" fontSize={12} fontWeight={600} fill={t?.color ?? INK.text}>{t?.site ?? s}</text>
                  <line x1={0} x2={iw} y1={(y(s) ?? 0) + y.bandwidth() / 2} y2={(y(s) ?? 0) + y.bandwidth() / 2} stroke={INK.grid} />
                </g>
              )
            })}
            {rows.map((r, i) => {
              const picked = r.preset.key === presetKey
              // spread the nine calibrations across the row so overlapping dots stay visible
              const jitter = ((runs.presets.indexOf(r.preset) / Math.max(1, runs.presets.length - 1)) - 0.5) * y.bandwidth() * 0.8
              const cy = (y(r.site) ?? 0) + y.bandwidth() / 2 + jitter
              const path = d3.symbol().type(SPECTRA_SHAPE[r.preset.spectra] ?? d3.symbolTriangle).size(picked ? 110 : 55)() ?? ''
              return (
                <g key={i}>
                {r.interval && <line data-fit-uncertainty="site-coefficient" x1={x(r.interval[0])} x2={x(r.interval[1])} y1={cy} y2={cy} stroke={cohortColor(r.preset.cohort)} strokeOpacity={0.45}><title>95% {metric} CI [{fmt(r.interval[0], 3)}, {fmt(r.interval[1], 3)}]</title></line>}
                <path
                  key={i} d={path} transform={`translate(${x(r.v)},${cy})`}
                  fill={cohortColor(r.preset.cohort)} fillOpacity={picked ? 1 : 0.7}
                  stroke={picked ? INK.text : INK.panel} strokeWidth={picked ? 2 : 0.8}
                  style={{ cursor: 'pointer' }}
                  onClick={() => onPick(r.preset.key)}
                  onMouseEnter={(e) => tip.show(e, [presetName(r.preset), `${runs.targets[r.site]?.site ?? r.site}: ${metric} ${fmt(r.v, 3)}`, picked ? 'picked above' : 'click to pick this calibration'])}
                  onMouseLeave={tip.hide}
                />
                </g>
              )
            })}
          </g>
          <g transform={`translate(${left},${ih + 66})`} fontSize={11} fill={INK.text}>
            {cohorts.map((c, i) => (
              <g key={c} transform={`translate(${i * 105},0)`}><circle cx={5} cy={-4} r={5} fill={cohortColor(c)} /><text x={14}>{c.replace('_', ' ')}</text></g>
            ))}
            {spectra.map((sp, i) => (
              <g key={sp} transform={`translate(${cohorts.length * 105 + 20 + i * 125},0)`}>
                <path d={d3.symbol().type(SPECTRA_SHAPE[sp] ?? d3.symbolTriangle).size(55)() ?? ''} transform="translate(5,-4)" fill={INK.muted} />
                <text x={14}>{label(PREPROCESSING, sp)}</text>
              </g>
            ))}
          </g>
        </svg>
        </AxisChoices>
        {tip.node}
      </div>
      <ExcludedNote count={nMarked} />
    </ChartFrame>
  )
}

/** how many of these runs' filters Filters → excluded: Mark leaves out (for a summary chart's ExcludedNote) */
function useExclusionCount(siteRuns: readonly CalibRun[]): number {
  const status = useExclStatus()
  return useMemo(() => siteRuns.reduce((a, run) => a + run.eval.id.filter((id) => status(id) === 'marked').length, 0), [siteRuns, status])
}
