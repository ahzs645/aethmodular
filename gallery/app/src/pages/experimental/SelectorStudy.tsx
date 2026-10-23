import { ChartFrame, ExplainOnly, Note } from '@/components/ChartFrame'
import { Legend } from '@/components/Legend'
import { SweepPlot } from '@/components/SweepPlot'
import { INK } from '@/lib/theme'
import { f2, pct } from '@/lib/format'
import { Forest, Scatter, runLine, sourceOf, useXp, type Cols, type XpBase } from './shared'

/** Test 5: pick the model with the smallest predicted error, or abstain (experiments_20261022.ipynb §5). */

interface Strategy { strategy: string; 'kept share': number; RMSE: number; MAE: number; bias: number; '99th pct |error|': number; 'max |error|': number; 'share |error| > 10 µg': number }
interface SelectorData extends XpBase {
  strategies: Strategy[]
  coverage: { 'kept share': number; 'RMSE selector': number; 'RMSE oracle order': number }[]
  calibration: { decile: number; 'predicted |error|': number; 'actual |error|': number }[]
  picks: { model: string; 'chosen share (selector)': number; 'chosen share (winner)': number; 'best share (oracle)': number }[]
  filters: Cols
}
const source = sourceOf('SelectorStudy')
const COLOR: Record<string, string> = {
  'full spline': '#2171b5', 'full raw': '#525252', 'local raw': '#d94801', 'average of three': '#8a8f98', 'winner classifier': '#6a51a3',
  get 'error-predicting selector'() { return INK.good }, get 'selector, abstain on riskiest tenth'() { return INK.good }, get 'oracle (best per filter)'() { return INK.fit },
}

export function SelectorStudy() {
  const { data, state } = useXp<SelectorData>('selector', 'Error-predicting selector')
  if (!data) return <>{state}</>
  return <>
    <Findings d={data} />
    <Strategies d={data} />
    <Picks d={data} />
    <CoverageCurve d={data} />
    <RiskCalibration d={data} />
  </>
}

function Findings({ d }: { d: SelectorData }) {
  const S = Object.fromEntries(d.strategies.map((s) => [s.strategy, s]))
  const singles = ['full spline', 'full raw', 'local raw'].map((k) => S[k]).sort((a, b) => a.RMSE - b.RMSE)
  const sel = S['error-predicting selector'], ab = S['selector, abstain on riskiest tenth'], win = S['winner classifier']
  return (
    <ExplainOnly>
      <ChartFrame id="xs-findings" title="Error-predicting selector — short answers" exportable={false}>
        <ol style={{ margin: '4px 0 0 18px', lineHeight: 1.55 }}>
          <li><strong>Against the best single model</strong> ({singles[0].strategy}, RMSE {f2(singles[0].RMSE)} µg), the selector reaches {f2(sel.RMSE)} and the winner classifier {f2(win.RMSE)}; the oracle (knowing TOR) would reach {f2(S['oracle (best per filter)'].RMSE)}.</li>
          <li><strong>Worst errors:</strong> largest |error| {f2(sel['max |error|'])} µg for the selector against {f2(win['max |error|'])} for the winner classifier and {f2(singles[0]['max |error|'])} for {singles[0].strategy}.</li>
          <li><strong>Abstaining</strong> on the riskiest tenth (threshold set on the training folds) keeps {pct(ab['kept share'])} of filters at RMSE {f2(ab.RMSE)} µg; {pct(ab['share |error| > 10 µg'])} of kept filters are still off by more than 10 µg.</li>
        </ol>
        <Note>{runLine(d)}. 1,500 IMPROVE filters, nested by site fold: the error models and the abstention threshold never see the fold they score.</Note>
      </ChartFrame>
    </ExplainOnly>
  )
}

function Strategies({ d }: { d: SelectorData }) {
  const rows = d.strategies.map((s) => ({
    key: s.strategy, label: s.strategy, color: COLOR[s.strategy] ?? INK.accent, hollow: s.strategy.startsWith('oracle'), bold: s.strategy.startsWith('error-predicting'),
    values: { RMSE: { v: s.RMSE }, p99: { v: s['99th pct |error|'] }, max: { v: s['max |error|'] } },
    tip: [s.strategy, `kept ${pct(s['kept share'])}`, `RMSE ${f2(s.RMSE)}, MAE ${f2(s.MAE)}, bias ${f2(s.bias)} µg`, `99th pct |error| ${f2(s['99th pct |error|'])}, max ${f2(s['max |error|'])} µg`, `${pct(s['share |error| > 10 µg'])} off by > 10 µg`],
  }))
  return (
    <ChartFrame id="xs-strategies" source={source('Strategies')}
      title="Strategies — typical error and the worst errors, side by side"
      subtitle="Each candidate alone, their average, the old idea (a classifier that predicts which model wins), the error-predicting selector, the selector that abstains on its riskiest tenth, and the oracle that knows TOR (hollow: not achievable). A good selector lowers the worst errors, not only the RMSE.">
      <Forest rows={rows} labelWidth={230} panels={[{ key: 'RMSE', label: 'RMSE against TOR (µg)', fmt: f2 }, { key: 'p99', label: '99th percentile |error| (µg)', fmt: f2 }, { key: 'max', label: 'largest |error| (µg)', fmt: f2 }]} />
    </ChartFrame>
  )
}

function Picks({ d }: { d: SelectorData }) {
  const cols = [['chosen share (selector)', 'selector'], ['chosen share (winner)', 'winner classifier'], ['best share (oracle)', 'oracle (best)']] as const
  const series = d.picks.map((p) => ({ id: p.model, label: p.model, group: p.model, color: COLOR[p.model], interval: 'none' as const,
    points: cols.map(([k, lab]) => ({ x: lab, y: (p as any)[k], tip: [p.model, `${lab}: ${pct((p as any)[k])} of filters`] })) }))
  return (
    <ChartFrame id="xs-picks" source={source('Picks')}
      title="Who gets chosen — each rule's share of filters per model"
      subtitle="How often the selector and the winner classifier pick each candidate, beside how often each is actually the best (oracle). A selector that always picks one model is that model.">
      <SweepPlot series={series} yLabel="share of filters" height={260} yFormat={(v) => pct(v)} yDomain={[0, 1]} />
      <Legend items={d.picks.map((p) => ({ label: p.model, color: COLOR[p.model] }))} />
    </ChartFrame>
  )
}

function CoverageCurve({ d }: { d: SelectorData }) {
  const series = [
    { id: 'sel', label: 'selector, kept by predicted risk', color: INK.good, marks: false, points: d.coverage.map((c) => ({ x: c['kept share'], y: c['RMSE selector'] })) },
    { id: 'or', label: 'oracle order (true error)', color: INK.fit, dash: '5 4', marks: false, points: d.coverage.map((c) => ({ x: c['kept share'], y: c['RMSE oracle order'] })) },
  ]
  return (
    <ChartFrame id="xs-coverage" source={source('CoverageCurve')}
      title="Abstaining — error left as the selector sets its riskiest filters aside"
      subtitle="Filters ranked by the selector's own predicted error; keep the safest share and read the RMSE of the chosen predictions among them. The dashed line keeps by the true error (the best any ranking could do).">
      <SweepPlot series={series} xDomain={[1, 0.3]} xLabel="share of filters kept" yLabel="RMSE among kept (µg)" height={300} xFormat={(v) => pct(+v)} yFormat={f2} xMarkers={[{ at: 0.9, label: 'abstain 10 %' }]} />
      <Legend items={series.map((s) => ({ label: s.label, color: s.color, shape: s.dash ? 'dashed' as const : 'line' as const }))} />
    </ChartFrame>
  )
}

function RiskCalibration({ d }: { d: SelectorData }) {
  const pts = d.calibration.map((c) => ({ x: c['predicted |error|'], y: c['actual |error|'], color: INK.good, r: 6, key: String(c.decile), tip: [`predicted-risk decile ${c.decile + 1}`, `predicted |error| ${f2(c['predicted |error|'])} µg`, `actual |error| ${f2(c['actual |error|'])} µg`] }))
  return (
    <ChartFrame id="xs-calibration" source={source('RiskCalibration')}
      title="Is the predicted error honest? — predicted against actual, by decile"
      subtitle="Filters grouped into tenths by the selector's predicted error for the model it chose; mean predicted against mean actual |error|. On the dashed line the prediction is honest; above it the selector is overconfident.">
      <Scatter points={pts} xLabel="mean predicted |error| (µg)" yLabel="mean actual |error| (µg)" identity height={300} />
    </ChartFrame>
  )
}
