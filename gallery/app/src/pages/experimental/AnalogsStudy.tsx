import { useMemo } from 'react'
import * as d3 from 'd3'
import { ChartFrame, ExplainOnly, Note, Segmented, Select } from '@/components/ChartFrame'
import { Legend, useLegend } from '@/components/Legend'
import { SweepPlot } from '@/components/SweepPlot'
import { IntervalStrip } from '@/components/IntervalStrip'
import { useSearchParam } from '@/lib/url'
import { ADDIS_ALL, INK } from '@/lib/theme'
import { coef, coefSigned, f2, signed } from '@/lib/format'
import { demingLambda, useMac } from '@/lib/mac'
import { AxisChoices } from '@/components/AxisChoices'
import { Crossplot, CrossplotBrushControls, sharedCrossDomains, useCrossplotBrush, type CrossPoint } from '@/components/Crossplot'
import { FacetGrid } from '@/components/FacetGrid'
import { ExcludedNote, FixedSetNote, useExclusionSplit } from '@/lib/markedRows'
import { Forest, num, rowsOf, runLine, sourceOf, useXp, type Cols, type XpBase } from './shared'

/** Test 3: calibration neighbours chosen by deposited mass as well as spectral shape (experiments_20261022.ipynb §3). */

interface AnalogsData extends XpBase {
  improve: Record<string, any>[]
  addis: Record<string, any>[]
  addis_filters: Cols
  improve_filters: Cols
}
const RULES_ALL = ['shape', 'shape + loading', 'loading only', 'shape + HIPS loading']
const RULE_COLOR: Record<string, string> = { shape: '#2171b5', 'shape + loading': '#d94801', 'loading only': '#6a51a3', 'shape + HIPS loading': '#1b7837' }
/** the rules this run has (older runs lack the HIPS-loading rule) */
const rulesOf = (d: AnalogsData) => RULES_ALL.filter((r) => d.addis.some((x) => x.model === `local: ${r}`))
const MODEL_COLOR: Record<string, string> = { get 'SPARTAN published'() { return INK.text }, 'full calibration (raw)': '#8a8f98', ...Object.fromEntries(RULES_ALL.map((r) => [`local: ${r}`, RULE_COLOR[r]])) }
const source = sourceOf('AnalogsStudy')
/** the notebook's Deming error ratio for Addis against HIPS at MAC 10 (σ from the HIPS uncertainty); older exports lack it */
const lambdaOf = (d: AnalogsData) => Number.isFinite(num(d.settings?.LAMBDA_ADDIS)) ? num(d.settings.LAMBDA_ADDIS) : 2.96

export function AnalogsStudy() {
  const { data, state } = useXp<AnalogsData>('analogs', 'Loading-matched analogs')
  if (!data) return <>{state}</>
  return <>
    <Findings d={data} />
    <AddisForest d={data} />
    <MatchCheck d={data} />
    <ImproveHolds d={data} />
    <AddisScatter d={data} />
  </>
}

function Findings({ d }: { d: AnalogsData }) {
  const { label } = useMac()
  const A = Object.fromEntries(d.addis.map((r) => [r.model, r]))
  const I = (model: string, g: string) => d.improve.find((r) => r.model === model && r.group === g)
  const full = I('full calibration (raw)', 'heaviest tenth'), sl = I('local: shape + loading', 'heaviest tenth'), sh = I('local: shape', 'heaviest tenth')
  const closer = Math.abs(A['local: shape + loading'].intercept) < Math.abs(A['local: shape'].intercept)
  return (
    <ExplainOnly>
      <ChartFrame id="xa-findings" title="Loading-matched analogs — short answers" exportable={false}>
        <ol style={{ margin: '4px 0 0 18px', lineHeight: 1.55 }}>
          <li><strong>Matching worked on loading:</strong> Addis filters are {f2(A['local: shape']['PM ÷ neighbours (median)'])}× heavier than their shape neighbours (weighed PM2.5) and {f2(A['local: shape + loading']['PM ÷ neighbours (median)'])}× with shape + loading neighbours.</li>
          <li><strong>Addis against HIPS:</strong> intercept {coefSigned(A['local: shape + loading'].intercept)} µg/m³ with shape + loading neighbours, {coefSigned(A['local: shape'].intercept)} with shape alone, {coefSigned(A['local: loading only'].intercept)} with loading alone; published {coefSigned(A['SPARTAN published'].intercept)}. {closer ? 'Loading matching moves Addis toward HIPS.' : 'Loading matching does not move Addis toward HIPS, so loading extrapolation alone does not explain the gap.'}</li>
          {A['local: shape + HIPS loading'] && <li><strong>Matching on absorption instead</strong> (shape + HIPS loading): Addis is still {f2(A['local: shape + HIPS loading']['HIPS ÷ neighbours (median)'])}× more absorbing than its neighbours (IMPROVE has no heavier absorbers of that shape), and the intercept is {coefSigned(A['local: shape + HIPS loading'].intercept)} µg/m³.</li>}
          <li><strong>IMPROVE's heaviest tenth against TOR:</strong> RMSE {f2(full?.rmse)} µg (full), {f2(sh?.rmse)} (shape), {f2(sl?.rmse)} (shape + loading){I('local: shape + HIPS loading', 'heaviest tenth') ? `, ${f2(I('local: shape + HIPS loading', 'heaviest tenth')?.rmse)} (shape + HIPS loading)` : ''}.</li>
        </ol>
        <Note>{runLine(d)}. Addis is read against HIPS BC ({label}, set under Samples), a cross-check, not the truth. The intercepts do not depend on the MAC.</Note>
      </ChartFrame>
    </ExplainOnly>
  )
}

function AddisForest({ d }: { d: AnalogsData }) {
  // the export's fits are against HIPS BC at MAC 10 (λ at MAC 10): at the reader's MAC the slope and the
  // FTIR ÷ HIPS ratio scale by MAC/10 and the intercept stays (lib/mac)
  const m = useMac()
  const rows = d.addis.map((r) => ({
    key: r.model, label: r.model.replace('local: ', 'local · '), color: MODEL_COLOR[r.model], bold: r.model === 'local: shape + loading',
    values: { intercept: { v: r.intercept }, slope: { v: m.slope(r.slope) as number }, ratio: { v: m.slope(r['ratio (median)']) as number } },
    tip: [r.model, `Deming slope ${coef(m.slope(r.slope))}, intercept ${coefSigned(r.intercept)} µg/m³`, `median FTIR ÷ HIPS ${f2(m.slope(r['ratio (median)']))}`, `n ${r.n}`,
      ...(r['PM ÷ neighbours (median)'] != null ? [`PM ÷ neighbours ${f2(r['PM ÷ neighbours (median)'])}×, HIPS ÷ neighbours ${f2(r['HIPS ÷ neighbours (median)'])}×`] : [])],
  }))
  return (
    <ChartFrame id="xa-addis" source={source('AddisForest')} provenance={runLine(d)}
      title="Addis against HIPS — does matching on loading close the gap?"
      subtitle={`Deming fit (λ ${f2(demingLambda(m.mac, lambdaOf(d)))}) of each model's Addis FTIR EC against HIPS BC (${m.label}, set under Samples) on the same filters; the export fitted at MAC 10, and another MAC scales the slope and ratio by MAC/10 and leaves the intercept. The local models are refitted per filter on 500 IMPROVE neighbours chosen by shape, by shape and weighed PM2.5, by PM2.5 alone, or by shape and HIPS absorption loading. A model on the HIPS line has intercept 0 and slope 1. The published n (189) is smaller: SPARTAN's EC covers fewer of the filters.`}>
      <Forest rows={rows} panels={[{ key: 'intercept', label: 'Deming intercept (µg/m³)', ref: 0, fmt: coefSigned }, { key: 'slope', label: 'Deming slope', ref: 1, fmt: coef }, { key: 'ratio', label: 'median FTIR ÷ HIPS', ref: 1, fmt: f2 }]} />
      <FixedSetNote what="the run's saved Deming fits on its own Addis filters" />
    </ChartFrame>
  )
}

function MatchCheck({ d }: { d: AnalogsData }) {
  const RULES = rulesOf(d)
  const [what, setWhat] = useSearchParam<'PM2.5' | 'HIPS'>('xa-match.what', 'PM2.5', ['PM2.5', 'HIPS'])
  const lg = useLegend()
  const f = useMemo(() => rowsOf(d.addis_filters), [d])
  const bins = d3.range(-1.2, 1.51, 0.1)
  const series = RULES.filter((r) => lg.show(r)).map((r) => {
    const v = f.map((x) => Math.log10(num(x[`${what === 'PM2.5' ? 'PM' : 'HIPS'} ratio ${r}`]))).filter(Number.isFinite)
    const h = d3.bin().domain([bins[0], bins[bins.length - 1]]).thresholds(bins)(v)
    return { id: r, label: r, group: r, color: RULE_COLOR[r], marks: false, curve: 'monotone' as const,
      points: h.map((b) => ({ x: ((b.x0 ?? 0) + (b.x1 ?? 0)) / 2, y: b.length / v.length, tip: [r, `${what} ÷ neighbours ${f2(10 ** (b.x0 ?? 0))}–${f2(10 ** (b.x1 ?? 0))}×`, `${b.length} Addis filters`] })) }
  })
  return (
    <ChartFrame id="xa-match" source={source('MatchCheck')}
      title="Did matching work? — how much heavier each Addis filter is than its neighbours"
      subtitle="For every Addis filter, its loading divided by the median of its 500 calibration neighbours (log scale; 0 = as heavy as its neighbours). Shape neighbours are much lighter than Addis; shape + loading neighbours should centre near 0 if IMPROVE has heavy filters of that shape."
      controls={<Segmented label="Loading" value={what} options={['PM2.5', 'HIPS'] as const} onChange={setWhat} optionLabel={(o) => o === 'PM2.5' ? 'Weighed PM2.5' : 'HIPS BC'} />}>
      <SweepPlot series={series} xDomain={[-1.2, 1.5]} yLabel="share of Addis filters" xLabel={`log10 (${what === 'PM2.5' ? 'weighed PM2.5' : 'HIPS BC'} ÷ neighbours' median)`} height={280} dim={lg.dim}
        xMarkers={[{ at: 0, label: 'as heavy as neighbours' }]} yFormat={(v) => `${Math.round(v * 100)} %`} xFormat={(v) => `${f2(10 ** +v)}×`} />
      <Legend {...lg.props} items={RULES.map((r) => ({ label: r, color: RULE_COLOR[r], shape: 'line' as const }))} />
      <FixedSetNote what="every Addis filter of the run, with the neighbours the run chose for it" />
    </ChartFrame>
  )
}

function ImproveHolds({ d }: { d: AnalogsData }) {
  const [metric, setMetric] = useSearchParam<'rmse' | 'bias' | 'rel_bias'>('xa-improve.metric', 'rmse', ['rmse', 'bias', 'rel_bias'])
  const groups = ['heaviest tenth', 'other filters', 'all']
  const color: Record<string, string> = { 'heaviest tenth': '#b2182b', 'other filters': '#2171b5', get all() { return INK.text } }
  const lg = useLegend()
  const models = [...new Set(d.improve.map((r) => r.model as string))]
  const fmt = (v: number) => metric === 'rel_bias' ? `${Math.round(v * 100)} %` : f2(v)
  // one row per model, the three groups stacked in it: the models are categories, so no line joins them
  const rows = models.map((m) => ({
    key: m, label: m.replace('local: ', 'local · '),
    tip: [m, ...groups.map((g) => { const r = d.improve.find((x) => x.model === m && x.group === g)
      return r ? `${g}: RMSE ${f2(r.rmse)} µg, bias ${signed(r.bias)} µg (${Math.round(r.rel_bias * 100)} %), n ${r.n}` : `${g}: —` })],
    marks: [groups.flatMap((g, gi) => { const r = d.improve.find((x) => x.model === m && x.group === g)
      return r && lg.show(g) ? [{ mid: r[metric] as number, color: color[g], dy: (gi - 1) * 6, r: 4 }] : [] })],
  }))
  return (
    <ChartFrame id="xa-improve" source={source('ImproveHolds')} provenance={runLine(d)}
      title="Does IMPROVE hold? — the same rules against TOR on held-out sites"
      subtitle="600 IMPROVE filters (300 from the heaviest tenth by weighed PM2.5, 300 others); neighbours come only from other site folds. A rule that helps Addis must not make IMPROVE worse, above all its heaviest filters."
      controls={<Segmented label="Read" value={metric} options={['rmse', 'bias', 'rel_bias'] as const} onChange={setMetric} optionLabel={(o) => ({ rmse: 'RMSE', bias: 'Bias', rel_bias: 'Relative bias' }[o])} />}>
      <IntervalStrip blocks={[{ key: 'models', rows }]} labelWidth={200} rowHeight={26}
        panels={[{ label: { rmse: 'RMSE against TOR (µg/filter)', bias: 'mean error (µg/filter)', rel_bias: 'mean error ÷ mean TOR' }[metric], reference: metric === 'rmse' ? undefined : 0, format: fmt }]} />
      <Legend {...lg.props} items={groups.map((g) => ({ label: g, color: color[g] }))} />
      <FixedSetNote what="the run's saved errors on its 600 held-out IMPROVE filters" />
    </ChartFrame>
  )
}

type ScatterView = 'one' | 'panels' | 'overlay'
const SCATTER_VIEWS = ['one', 'panels', 'overlay'] as const

function AddisScatter({ d }: { d: AnalogsData }) {
  const RULES = rulesOf(d)
  const models = ['published', 'full raw', ...RULES.map((r) => `local ${r}`)]
  /** the export's model name for a picker key, for its saved fit and colour */
  const exportName = (key: string) => key === 'published' ? 'SPARTAN published' : key === 'full raw' ? 'full calibration (raw)' : key.replace('local ', 'local: ')
  const [view, setView] = useSearchParam<ScatterView>('xa-scatter.view', 'one', SCATTER_VIEWS)
  const [model, setModel] = useSearchParam<string>('xa-scatter.model', 'local shape + loading', models)
  const [brush, setBrush] = useCrossplotBrush()
  const lg = useLegend()
  // HIPS BC at the reader's MAC (the export's is Fabs / 10)
  const mac = useMac()
  const all = useMemo(() => rowsOf(d.addis_filters).map((r): Record<string, any> => ({ ...r, hips_bc: mac.bc(num(r.hips_bc)) })), [d, mac])
  // the gallery's exclusions by filter id (lib/markedRows): `f` feeds every point and fit, `out` is drawn as × under Mark
  const { counted: f, marked: out, nOut } = useExclusionSplit(all, (r) => String(r.filter_id))
  const fit = d.addis.find((r) => r.model === exportName(model))
  const choices = ['hips_bc', ...models]
  const [xField, setX] = useSearchParam<string>('xa-scatter.x', 'hips_bc', choices)
  const [yField, setY] = useSearchParam<string>('xa-scatter.y', model, choices)
  const standard = xField === 'hips_bc' && yField === model
  const pickModel = (value: string) => { setModel(value); setX('hips_bc'); setY(value) }
  const labelOf = (key: string) => key === 'hips_bc' ? `HIPS BC (µg/m³, ${mac.label})` : `FTIR EC, ${key} (µg/m³)`
  const record = (r: Record<string, any>) => ({
    id: String(r.filter_id), source: 'Loading-matched analogs', site: 'Addis Ababa', date: r.date == null ? undefined : String(r.date), season: String(r.season ?? ''), fields: choices.map(k => [labelOf(k), Number.isFinite(num(r[k])) ? String(r[k]) : null] as [string, string | null]),
  })
  const pointsOf = (xKey: string, yKey: string, color = ADDIS_ALL, extra: Partial<CrossPoint> = {}, rows = f): CrossPoint[] => rows.map(r => ({ x: num(r[xKey]), y: num(r[yKey]), color, id: String(r.filter_id),
    tip: [String(r.filter_id), String(r.season), `${labelOf(xKey)}: ${f2(r[xKey])}`, `${labelOf(yKey)}: ${f2(r[yKey])}`], rec: record(r), ...extra }))
  const colorOf = (key: string) => MODEL_COLOR[exportName(key)] ?? ADDIS_ALL
  const lambda = demingLambda(mac.mac, lambdaOf(d))
  // every model on one range, so the panels compare directly
  const panels = useMemo(() => view === 'panels' ? models.map((m) => ({ key: m, points: pointsOf('hips_bc', m), marked: pointsOf('hips_bc', m, ADDIS_ALL, {}, out) })) : [], [view, f, out, models.join()]) // eslint-disable-line react-hooks/exhaustive-deps
  const shared = sharedCrossDomains(panels.map((p) => p.points))
  const overlayOf = (rows: Record<string, any>[]) => models.filter((m) => lg.show(m)).flatMap((m) => pointsOf('hips_bc', m, colorOf(m), { site: m, opacity: 0.3 * lg.dim(m) }, rows))
  const overlay = view === 'overlay' ? overlayOf(f) : []
  const brushControls = <CrossplotBrushControls brush={[brush, setBrush]} />
  return <ChartFrame id="xa-scatter" source={source('AddisScatter')} provenance={runLine(d)}
    title={view === 'one' ? 'Addis filters — FTIR EC against HIPS BC, one model at a time' : 'Addis filters — FTIR EC against HIPS BC, every model'}
    subtitle={view === 'one' ? 'Pick a model to plot its Addis predictions against HIPS BC, or show every model side by side or overlaid. Click either axis title to plot other measurements. Points follow the gallery\'s selection, exclusions (Filters → excluded), filter records and the estimators chosen in the side panel.'
      : view === 'panels' ? 'One graph per model, every graph on the same range: each model\'s Addis predictions against HIPS BC with its own Deming fit. Points follow the gallery\'s selection, exclusions (Filters → excluded), filter records and the estimators chosen in the side panel.'
      : 'Every model\'s Addis predictions against HIPS BC on one graph, each with its own Deming fit in the model\'s colour. Click a legend entry to hide a model; hover one to bring it forward. Points follow the gallery\'s selection, exclusions (Filters → excluded), filter records and the estimators chosen in the side panel.'}
    controls={<>
      <Segmented label="View" value={view} options={SCATTER_VIEWS} onChange={setView} optionLabel={(o) => ({ one: 'One model', panels: 'Graph per model', overlay: 'Lines overlaid' }[o])} />
      {view === 'one' && <Select label="Model" value={standard ? model : 'Custom'} options={[...models, ...(!standard ? ['Custom'] : [])]} onChange={v => { if (v !== 'Custom') pickModel(v) }} />}
      {brushControls}
    </>}>
    {view === 'one' && <AxisChoices noun="measurements" x={[xField, setX]} y={[yField, setY]} options={choices.map(key => ({ key, label: key === 'hips_bc' ? `HIPS BC, ${mac.label}` : `FTIR EC, ${key}`, group: key === 'hips_bc' ? 'Optical measurement' : 'Model predictions', unit: 'µg/m³' }))}>
      <Crossplot points={pointsOf(xField, yField)} marked={pointsOf(xField, yField, ADDIS_ALL, {}, out)} title={standard ? model : 'Custom axes'} xLabel={labelOf(xField)} yLabel={labelOf(yField)} pick brushable brush={brush} maxSide={480} errorRatio={lambda} />
    </AxisChoices>}
    {view === 'panels' && <FacetGrid items={panels} minCell={300} maxCols={3} render={(p, w) =>
      <Crossplot points={p.points} marked={p.marked} title={p.key} xLabel={labelOf('hips_bc')} yLabel="FTIR EC (µg/m³)" brushable brush={brush} maxSide={w - 70} errorRatio={lambda} domain={shared.domain} />} />}
    {view === 'overlay' && <>
      <Crossplot points={overlay} marked={view === 'overlay' && out.length ? overlayOf(out) : undefined} title="Every model" xLabel={labelOf('hips_bc')} yLabel="FTIR EC (µg/m³)" brushable brush={brush} maxSide={520} errorRatio={lambda} grouping="site" groupNoun="model" />
      <Legend {...lg.props} items={models.map((m) => ({ label: m, color: colorOf(m) }))} />
    </>}
    {view === 'one' && standard && fit && <Note>With every filter shown, the Deming line here is the one in "Addis against HIPS" above (slope {coef(mac.slope(fit.slope))}, intercept {coefSigned(fit.intercept)} µg/m³, n {fit.n}, λ {f2(lambda)}). Excluding filters or choosing another estimator changes it here only.</Note>}
    {view !== 'one' && <Note>With every filter shown, each model's Deming line is the one in "Addis against HIPS" above (λ {f2(lambda)}). Excluding filters or choosing another estimator changes them here only.</Note>}
    <ExcludedNote count={nOut} drawn />
  </ChartFrame>
}
