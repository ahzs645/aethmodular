import { useMemo } from 'react'
import * as d3 from 'd3'
import { ChartFrame, ExplainOnly, Note, Segmented } from '@/components/ChartFrame'
import { Legend, useLegend } from '@/components/Legend'
import { SweepPlot } from '@/components/SweepPlot'
import { useSearchParam } from '@/lib/url'
import { ADDIS_ALL, INK } from '@/lib/theme'
import { f2, signed } from '@/lib/format'
import { AxisChoices } from '@/components/AxisChoices'
import { Crossplot } from '@/components/Crossplot'
import { Select } from '@/components/ChartFrame'
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
  const A = Object.fromEntries(d.addis.map((r) => [r.model, r]))
  const I = (model: string, g: string) => d.improve.find((r) => r.model === model && r.group === g)
  const full = I('full calibration (raw)', 'heaviest tenth'), sl = I('local: shape + loading', 'heaviest tenth'), sh = I('local: shape', 'heaviest tenth')
  const closer = Math.abs(A['local: shape + loading'].intercept) < Math.abs(A['local: shape'].intercept)
  return (
    <ExplainOnly>
      <ChartFrame id="xa-findings" title="Loading-matched analogs — short answers" exportable={false}>
        <ol style={{ margin: '4px 0 0 18px', lineHeight: 1.55 }}>
          <li><strong>Matching worked on loading:</strong> Addis filters are {f2(A['local: shape']['PM ÷ neighbours (median)'])}× heavier than their shape neighbours (weighed PM2.5) and {f2(A['local: shape + loading']['PM ÷ neighbours (median)'])}× with shape + loading neighbours.</li>
          <li><strong>Addis against HIPS:</strong> intercept {signed(A['local: shape + loading'].intercept)} µg/m³ with shape + loading neighbours, {signed(A['local: shape'].intercept)} with shape alone, {signed(A['local: loading only'].intercept)} with loading alone; published {signed(A['SPARTAN published'].intercept)}. {closer ? 'Loading matching moves Addis toward HIPS.' : 'Loading matching does not move Addis toward HIPS, so loading extrapolation alone does not explain the gap.'}</li>
          {A['local: shape + HIPS loading'] && <li><strong>Matching on absorption instead</strong> (shape + HIPS loading): Addis is still {f2(A['local: shape + HIPS loading']['HIPS ÷ neighbours (median)'])}× more absorbing than its neighbours (IMPROVE has no heavier absorbers of that shape), and the intercept is {signed(A['local: shape + HIPS loading'].intercept)} µg/m³.</li>}
          <li><strong>IMPROVE's heaviest tenth against TOR:</strong> RMSE {f2(full?.rmse)} µg (full), {f2(sh?.rmse)} (shape), {f2(sl?.rmse)} (shape + loading){I('local: shape + HIPS loading', 'heaviest tenth') ? `, ${f2(I('local: shape + HIPS loading', 'heaviest tenth')?.rmse)} (shape + HIPS loading)` : ''}.</li>
        </ol>
        <Note>{runLine(d)}. Addis is read against HIPS BC (MAC 10), a cross-check, not the truth.</Note>
      </ChartFrame>
    </ExplainOnly>
  )
}

function AddisForest({ d }: { d: AnalogsData }) {
  const rows = d.addis.map((r) => ({
    key: r.model, label: r.model.replace('local: ', 'local · '), color: MODEL_COLOR[r.model], bold: r.model === 'local: shape + loading',
    values: { intercept: { v: r.intercept }, slope: { v: r.slope }, ratio: { v: r['ratio (median)'] } },
    tip: [r.model, `Deming slope ${f2(r.slope)}, intercept ${signed(r.intercept)} µg/m³`, `median FTIR ÷ HIPS ${f2(r['ratio (median)'])}`, `n ${r.n}`,
      ...(r['PM ÷ neighbours (median)'] != null ? [`PM ÷ neighbours ${f2(r['PM ÷ neighbours (median)'])}×, HIPS ÷ neighbours ${f2(r['HIPS ÷ neighbours (median)'])}×`] : [])],
  }))
  return (
    <ChartFrame id="xa-addis" source={source('AddisForest')}
      title="Addis against HIPS — does matching on loading close the gap?"
      subtitle="Deming fit of each model's Addis FTIR EC against HIPS BC (MAC 10) on the same filters. The local models are refitted per filter on 500 IMPROVE neighbours chosen by shape, by shape and weighed PM2.5, by PM2.5 alone, or by shape and HIPS absorption loading. A model on the HIPS line has intercept 0 and slope 1. Note the published n (189) is smaller: SPARTAN's EC covers fewer of the filters.">
      <Forest rows={rows} panels={[{ key: 'intercept', label: 'Deming intercept (µg/m³)', ref: 0, fmt: (v) => signed(v) }, { key: 'slope', label: 'Deming slope', ref: 1, fmt: f2 }, { key: 'ratio', label: 'median FTIR ÷ HIPS', ref: 1, fmt: f2 }]} />
      <Note>{runLine(d)}.</Note>
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
    </ChartFrame>
  )
}

function ImproveHolds({ d }: { d: AnalogsData }) {
  const [metric, setMetric] = useSearchParam<'rmse' | 'bias' | 'rel_bias'>('xa-improve.metric', 'rmse', ['rmse', 'bias', 'rel_bias'])
  const groups = ['heaviest tenth', 'other filters', 'all']
  const color: Record<string, string> = { 'heaviest tenth': '#b2182b', 'other filters': '#2171b5', all: INK.text }
  const lg = useLegend()
  const models = [...new Set(d.improve.map((r) => r.model as string))]
  const series = groups.filter((g) => lg.show(g)).map((g) => ({
    id: g, label: g, group: g, color: color[g], interval: 'none' as const,
    points: models.map((m) => { const r = d.improve.find((x) => x.model === m && x.group === g)!
      return { x: m.replace('local: ', 'local · '), y: r[metric], tip: [m, g, `RMSE ${f2(r.rmse)} µg, bias ${signed(r.bias)} µg (${Math.round(r.rel_bias * 100)} %)`, `n ${r.n}`] } }),
  }))
  return (
    <ChartFrame id="xa-improve" source={source('ImproveHolds')}
      title="Does IMPROVE hold? — the same rules against TOR on held-out sites"
      subtitle="600 IMPROVE filters (300 from the heaviest tenth by weighed PM2.5, 300 others); neighbours come only from other site folds. A rule that helps Addis must not make IMPROVE worse, above all its heaviest filters."
      controls={<Segmented label="Read" value={metric} options={['rmse', 'bias', 'rel_bias'] as const} onChange={setMetric} optionLabel={(o) => ({ rmse: 'RMSE', bias: 'Bias', rel_bias: 'Relative bias' }[o])} />}>
      <SweepPlot series={series} yLabel={{ rmse: 'RMSE against TOR (µg/filter)', bias: 'mean error (µg/filter)', rel_bias: 'mean error ÷ mean TOR' }[metric]} height={300} dim={lg.dim}
        reference={metric === 'rmse' ? undefined : 0} yFormat={(v) => metric === 'rel_bias' ? `${Math.round(v * 100)} %` : f2(v)} />
      <Legend {...lg.props} items={groups.map((g) => ({ label: g, color: color[g] }))} />
    </ChartFrame>
  )
}

function AddisScatter({ d }: { d: AnalogsData }) {
  const RULES = rulesOf(d)
  const models = ['published', 'full raw', ...RULES.map((r) => `local ${r}`)]
  const [model, setModel] = useSearchParam<string>('xa-scatter.model', 'local shape + loading', models)
  const f = useMemo(() => rowsOf(d.addis_filters), [d])
  const fit = d.addis.find((r) => (r.model as string).replace('local: ', 'local ') === model || (model === 'published' && r.model === 'SPARTAN published') || (model === 'full raw' && r.model === 'full calibration (raw)'))
  const choices = ['hips_bc', ...models]
  const [xField, setX] = useSearchParam<string>('xa-scatter.x', 'hips_bc', choices)
  const [yField, setY] = useSearchParam<string>('xa-scatter.y', model, choices)
  const standard = xField === 'hips_bc' && yField === model
  const pickModel = (value: string) => { setModel(value); setX('hips_bc'); setY(value) }
  const labelOf = (key: string) => key === 'hips_bc' ? 'HIPS BC (µg/m³, MAC 10)' : `FTIR EC, ${key} (µg/m³)`
  const pts = f.map(r => ({ x: num(r[xField]), y: num(r[yField]), color: ADDIS_ALL, id: String(r.filter_id), tip: [String(r.filter_id), String(r.season), `${labelOf(xField)}: ${f2(r[xField])}`, `${labelOf(yField)}: ${f2(r[yField])}`], rec: {
    id: String(r.filter_id), source: 'Loading-matched analogs', site: 'Addis Ababa', date: r.date == null ? undefined : String(r.date), season: String(r.season ?? ''), fields: choices.map(k => [labelOf(k), Number.isFinite(num(r[k])) ? String(r[k]) : null] as [string, string | null]),
  } }))
  return <ChartFrame id="xa-scatter" source={source('AddisScatter')}
    title="Addis filters — FTIR EC against HIPS BC, one model at a time"
    subtitle="Pick a model to restore HIPS BC versus that prediction. Click either axis title for custom measurements. Points use the shared selection, exclusion, record and regression controls."
    controls={<><Select label="Model preset" value={standard ? model : 'Custom'} options={[...models, ...(!standard ? ['Custom'] : [])]} onChange={v => { if (v !== 'Custom') pickModel(v) }} /><span className="control">{standard ? 'HIPS BC × model prediction' : 'Custom axes'}</span></>}>
    <AxisChoices noun="measurements" x={[xField, setX]} y={[yField, setY]} options={choices.map(key => ({ key, label: key === 'hips_bc' ? 'HIPS BC, MAC 10' : `FTIR EC, ${key}`, group: key === 'hips_bc' ? 'Optical measurement' : 'Model predictions', unit: 'µg/m³' }))}>
      <Crossplot points={pts} title={standard ? model : 'Custom axes'} xLabel={labelOf(xField)} yLabel={labelOf(yField)} pick brushable maxSide={480} errorRatio={typeof d.settings?.deming_lambda_mac10 === 'number' ? d.settings.deming_lambda_mac10 : 2.96} />
    </AxisChoices>
    {standard && fit && <Note>Saved Deming: slope {f2(fit.slope)}, intercept {signed(fit.intercept)} µg/m³, n {fit.n}. Live fits above use the selected estimators and visible filters.</Note>}
  </ChartFrame>
}
