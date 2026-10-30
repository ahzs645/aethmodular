import { useMemo, type ReactNode } from 'react'
import { ChartFrame, Segmented } from '@/components/ChartFrame'
import { Crossplot, sharedCrossDomains, type CrossPoint, type CrossLine } from '@/components/Crossplot'
import { FacetGrid } from '@/components/FacetGrid'
import { Key } from '@/components/Key'
import { baseFilterId } from '@/lib/filterId'
import { ExcludedNote, useCountedFilter, useExclusionSplit } from '@/lib/markedRows'
import { f2, f3 } from '@/lib/format'
import { useMac } from '@/lib/mac'
import { num, type Row } from '@/lib/table'
import { INK } from '@/lib/theme'
import type { CalibRunsFile } from '@/lib/types'
import { useSearchParam } from '@/lib/url'
import { transferMetrics, type TransferMetrics } from './hips'

/** the three predictions ftir_08 [9] wrote, the notebook's metric-table row for each, and its colour */
const MODELS = [
  { col: 'improve', name: 'IMPROVE HIPS model → Addis', model: 'IMPROVE HIPS → Addis external test', color: '#c0392b' },
  { col: 'addis', name: 'Addis-only HIPS model (nested CV)', model: 'Addis-only HIPS', color: '#2b6cb0' },
  { col: 'naive', name: 'Naive direct Fabs (unit mismatched)', model: 'Naive direct-Fabs sensitivity (unit mismatched)', color: '#7b8794' },
] as const
const PANELS = ['notebook pair', 'with naive'] as const
const UNITS = ['Fabs', 'HIPS BC'] as const
const SETS = ['all 239', 'fixed 190'] as const
const METRIC_KEYS = ['slope', 'intercept', 'R2', 'RMSE', 'bias', 'median_bias'] as const

const line = (m: TransferMetrics | null, color: string): CrossLine[] =>
  m ? [{ slope: m.slope, intercept: m.intercept, color, label: 'OLS of predicted on observed (ftir_08)', method: 'ols', stats: null }] : []

/**
 * ftir_08 [10]: each Addis filter's observed HIPS Fabs against the Fabs predicted by the IMPROVE HIPS model and by
 * the Addis-only model (nested CV), from the notebook's per-filter predictions. The points and the metrics are
 * drawn here: they follow the filter set, Filters → excluded (by filter id: under Mark the excluded are ×) and the units; OLS of predicted on observed as the
 * notebook's regression_metrics. The models were trained on HIPS optical depth τ = Fabs·V/(100·A) and converted
 * back, so the comparison is Fabs against Fabs and MAC-free; in HIPS BC units both axes are divided by the
 * reader's MAC, which rescales intercept and errors and leaves slope and R² as they are.
 */
export function HipsTransfer({ filters, shown, runs, provenance }: { filters: Row[]; shown: Row[]; runs: CalibRunsFile | null; provenance: ReactNode }) {
  const mac = useMac()
  const [panels, setPanels] = useSearchParam<(typeof PANELS)[number]>('ml-model-form-hips-crossplot.panels', 'notebook pair', PANELS)
  const [units, setUnits] = useSearchParam<(typeof UNITS)[number]>('ml-model-form-hips-crossplot.units', 'Fabs', UNITS)
  const [setName, setSet] = useSearchParam<(typeof SETS)[number]>('ml-model-form-hips-crossplot.set', 'all 239', SETS)

  // calibration_runs' Addis evaluation set: the fixed-190 flag, the season and the date by base filter id
  const meta = useMemo(() => {
    const run = runs?.runs.find((r) => r.target === 'addis')
    const out = new Map<string, { fixed: boolean; season: string; date: string }>()
    run?.eval.id.forEach((id, i) => out.set(String(id), { fixed: !!run.eval.fixed?.[i], season: String(run.eval.group?.[i] ?? ''), date: String(run.eval.date?.[i] ?? '') }))
    return out
  }, [runs])
  const fixedKnown = meta.size > 0
  const set = fixedKnown ? setName : 'all 239'
  const scale = units === 'Fabs' ? 1 : 1 / mac.mac
  const unit = units === 'Fabs' ? 'Mm⁻¹' : 'µg/m³'

  const inSet = filters.filter((r) => set === 'all 239' || meta.get(baseFilterId(String(r.id)))?.fixed)
  // the counted filters by the one rule (lib/markedRows useCountedFilter: the export's exclusions unless Include, hand
  // exclusions, keep=); the split gives the × under Mark
  const split = useExclusionSplit(inSet, (r) => baseFilterId(String(r.id)))
  const isCounted = useCountedFilter()
  const kept = inSet.filter((r) => isCounted(baseFilterId(String(r.id))))
  const models = MODELS.filter((m) => panels === 'with naive' || m.col !== 'naive')

  const panelsOut = models.map((m) => {
    const metrics = transferMetrics(kept.map((r) => (num(r.Fabs) ?? NaN) * scale), kept.map((r) => (num(r[m.col]) ?? NaN) * scale))
    const toPoints = (rs: Row[]): CrossPoint[] => rs.flatMap((r) => {
      const fabs = num(r.Fabs), pred = num(r[m.col])
      if (fabs == null || pred == null) return []
      const id = baseFilterId(String(r.id)), info = meta.get(id)
      return [{ x: fabs * scale, y: pred * scale, color: m.color, id,
        tip: [String(r.id), ...(info ? [`${info.date} · ${info.season}`] : []), `observed ${units === 'Fabs' ? 'Fabs' : 'HIPS BC'} ${f2(fabs * scale)} ${unit}`, `${m.name}: ${f2(pred * scale)} ${unit}`] }]
    })
    return { m, metrics, points: toPoints(kept), marked: toPoints(split.marked) }
  })
  const domains = sharedCrossDomains(panelsOut.map((p) => p.points))

  // the live metrics against ftir_08's table, on the notebook's own set (all 239, nothing excluded, Fabs)
  const fullCheck = MODELS.map((m) => {
    const live = transferMetrics(filters.map((r) => num(r.Fabs) ?? NaN), filters.map((r) => num(r[m.col]) ?? NaN))
    const nb = shown.find((r) => r.model === m.model)
    return live && nb ? Math.max(...METRIC_KEYS.map((k) => Math.abs(live[k] - (num(nb[k]) ?? NaN)) / Math.max(1, Math.abs(num(nb[k]) ?? 1)))) : NaN
  })
  const worst = Math.max(...fullCheck)
  const imp = panelsOut.find((p) => p.m.col === 'improve')?.metrics, add = panelsOut.find((p) => p.m.col === 'addis')?.metrics
  const median = (v: number[]) => { const s = [...v].sort((a, b) => a - b); return s.length ? (s[(s.length - 1) >> 1] + s[s.length >> 1]) / 2 : NaN }
  const medObs = median(kept.map((r) => (num(r.Fabs) ?? NaN) * scale).filter(Number.isFinite))
  const medImp = median(kept.map((r) => (num(r.improve) ?? NaN) * scale).filter(Number.isFinite))
  const axis = (what: string) => units === 'Fabs' ? `${what} HIPS Fabs (Mm⁻¹)` : `${what} HIPS BC = Fabs / MAC (µg/m³, ${mac.label})`
  const excluded = inSet.length - kept.length

  return (
    <ChartFrame id="ml-model-form-hips-crossplot"
      title={`The IMPROVE HIPS model predicts a median Addis ${units === 'Fabs' ? 'Fabs' : 'HIPS BC'} of ${f2(medImp)} against ${f2(medObs)} observed (slope ${f2(imp?.slope)}); an Addis-only model reaches R² ${f3(add?.R2)} — ftir_08 [10], HIPS as the PLS target`}
      subtitle={`Each of the ${kept.length} Addis filters${set === 'fixed 190' ? ' in the fixed 190' : ' with spectra, HIPS and a sampled volume'}: observed HIPS against the HIPS predicted from its FTIR spectrum. Left, a PLS model trained on 916 IMPROVE filters' HIPS optical depth (τ = Fabs × volume / (100 × deposit area), a filter-loading basis like FTIR absorbance; k chosen by holding out whole sites) and converted back to Fabs per filter; right, a PLS model trained on Addis itself, each prediction from nested 5-fold CV.${panels === 'with naive' ? ' The grey panel regresses Fabs directly on IMPROVE spectra, ignoring volume and area: a deliberate unit mismatch, diagnostic only.' : ''} The solid line is the OLS of predicted on observed, with its live metrics in the corner as the notebook printed them; dashed is 1:1. ${units === 'Fabs' ? 'Fabs against Fabs: no MAC enters anywhere, so the chart does not move with the MAC setting.' : `Both axes are divided by the reader's MAC (${mac.label}): the slope and R² are the same as in Fabs, the intercept, RMSE and bias scale by 1/MAC.`} The IMPROVE model's shortfall survives the τ normalisation, so it is not the naive unit mismatch; the Addis-only model is a mechanism probe, not an EC calibration.`}
      provenance={provenance}
      controls={<>
        <Segmented label="Panels" value={panels} options={PANELS} optionLabel={(o) => (o === 'notebook pair' ? 'The notebook’s two' : '+ naive direct Fabs')} onChange={setPanels} />
        {fixedKnown && <Segmented label="Filters" value={set} options={SETS} onChange={setSet} />}
        <Segmented label="Units" value={units} options={UNITS} optionLabel={(o) => (o === 'Fabs' ? 'Fabs (MAC-free)' : `HIPS BC (${mac.label})`)} onChange={setUnits} />
      </>}>
      <FacetGrid items={panelsOut} minCell={300} maxCols={3} maxCell={480} render={(p, w) => (
        <Crossplot title={p.m.name} points={p.points} marked={p.marked} fit={false} grouping="none" extraLines={line(p.metrics, p.m.color)} maxSide={w - 70} domain={domains.domain}
          xLabel={axis('Observed')} yLabel={axis('Predicted')}
          background={(x, y) => p.metrics && <text x={x(domains.domain[0]) + 8} y={y(domains.domain[1]) + 16} fontSize={11} fill={INK.text} pointerEvents="none">
            {/* the notebook's in-panel box: the live OLS line and its metrics */}
            {[`y = ${f2(p.metrics.slope)}x ${p.metrics.intercept < 0 ? '−' : '+'} ${f2(Math.abs(p.metrics.intercept))}`,
              `R² = ${f3(p.metrics.R2)}   RMSE = ${f2(p.metrics.RMSE)}`, `n = ${p.metrics.n}`].map((t, i) => <tspan key={i} x={x(domains.domain[0]) + 8} dy={i ? 14 : 0}>{t}</tspan>)}
          </text>} />
      )} />
      <Key items={[...panelsOut.map((p) => ({ label: p.m.name, color: p.m.color, kind: 'dot' as const })),
        { label: 'OLS of predicted on observed', color: INK.text, kind: 'line' }, { label: '1:1', color: INK.identity, kind: 'dash' }]} />
      <div className="legend-note" style={{ marginTop: 6 }}>
        {panelsOut.map((p) => p.metrics && <span key={p.m.col}>{p.m.name}: bias {f2(p.metrics.bias)} {unit} (median {f2(p.metrics.median_bias)}), MAE {f2(p.metrics.MAE)}. </span>)}
        {excluded > 0 && <>{excluded} filter{excluded === 1 ? '' : 's'} excluded under Filters → excluded {excluded === 1 ? 'is' : 'are'} left out of these metrics. </>}
        {Number.isFinite(worst) && <>On the notebook's set (all 239, nothing excluded, Fabs) the browser's metrics reproduce ftir_08's table for all three models {worst < 1e-3 ? `to its stored digits (largest relative difference ${worst.toExponential(0)})` : `within ${(100 * worst).toFixed(2)} %`}.</>}
      </div>
      <ExcludedNote count={split.marked.length} drawn />
    </ChartFrame>
  )
}
