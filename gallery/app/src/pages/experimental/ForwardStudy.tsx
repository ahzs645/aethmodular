import * as d3 from 'd3'
import { ChartFrame, ExplainOnly, Note, Segmented } from '@/components/ChartFrame'
import { Legend, useLegend } from '@/components/Legend'
import { SweepPlot } from '@/components/SweepPlot'
import { useSearchParam } from '@/lib/url'
import { f2, signed } from '@/lib/format'
import { Forest, runLine, sourceOf, useXp, type XpBase } from './shared'

/** Test 6: train on earlier years, test on later ones (experiments_20261022.ipynb §6). */

interface Curve { preprocessing: string; split: string; repeat: number; k: number; 'n test': number; RMSE: number; bias: number; 'R²': number }
interface Summary { preprocessing: string; split: string; n_test: number; RMSE: number; RMSE_sd: number | null; bias: number; R2: number }
interface ForwardData extends XpBase { curves: Curve[]; summary: Summary[]; years: { year: number; filters: number; sites: number; median_tor_ug: number }[] }
const source = sourceOf('ForwardStudy')
const SPLIT_COLOR: Record<string, string> = {
  'later year (train 2020–22, test 2023)': '#d94801', 'new sites (same test size)': '#2171b5', 'random filters (same test size)': '#8a8f98', 'new sites in the later year': '#6a51a3',
}

export function ForwardStudy() {
  const { data, state } = useXp<ForwardData>('forward', 'Earlier → later years')
  if (!data) return <>{state}</>
  return <>
    <Findings d={data} />
    <SummaryForest d={data} />
    <Curves d={data} />
    <Years d={data} />
  </>
}

function Findings({ d }: { d: ForwardData }) {
  const s = (prep: string, start: string) => d.summary.find((x) => x.preprocessing === prep && x.split.startsWith(start))
  return (
    <ExplainOnly>
      <ChartFrame id="xf-findings" title="Earlier → later years — short answers" exportable={false}>
        <ol style={{ margin: '4px 0 0 18px', lineHeight: 1.55 }}>
          {['spline', 'raw'].map((p) => <li key={p}><strong>{p}:</strong> RMSE {f2(s(p, 'later')?.RMSE)} µg in the later year, {f2(s(p, 'new sites (')?.RMSE)} at new sites, {f2(s(p, 'random')?.RMSE)} on random filters, {f2(s(p, 'new sites in')?.RMSE)} at new sites in the later year (bias {signed(s(p, 'later')?.bias)} in the later year).</li>)}
        </ol>
        <Note>{runLine(d)}. Only 2023 can be a "later year" here (2022 holds most filters), so the later-year test is one split, not five.</Note>
      </ChartFrame>
    </ExplainOnly>
  )
}

function SummaryForest({ d }: { d: ForwardData }) {
  const [prep, setPrep] = useSearchParam<'spline' | 'raw'>('xf-sum.prep', 'spline', ['spline', 'raw'])
  const rows = d.summary.filter((s) => s.preprocessing === prep).map((s) => ({
    key: s.split, label: s.split, color: SPLIT_COLOR[s.split], bold: s.split.startsWith('later'),
    values: { RMSE: { v: s.RMSE, lo: s.RMSE_sd != null ? s.RMSE - s.RMSE_sd : undefined, hi: s.RMSE_sd != null ? s.RMSE + s.RMSE_sd : undefined }, bias: { v: s.bias }, R2: { v: s.R2 } },
    tip: [s.split, `${Math.round(s.n_test).toLocaleString()} test filters`, `RMSE ${f2(s.RMSE)}${s.RMSE_sd != null ? ` ± ${f2(s.RMSE_sd)}` : ''} µg, bias ${signed(s.bias)} µg, R² ${f2(s.R2)}`],
  }))
  return (
    <ChartFrame id="xf-summary" source={source('SummaryForest')} provenance={runLine(d)}
      title="Time against space — the same calibration, four ways to hold out"
      subtitle={`At the one-SE k for ${prep}. Whiskers: ± 1 SD over five repeats (new-site and random splits); the later-year split is a single fixed split. If the later year is as hard as new sites, time transfer is no worse than site transfer.`}
      controls={<Segmented label="Preprocessing" value={prep} options={['spline', 'raw'] as const} onChange={setPrep} />}>
      <Forest rows={rows} labelWidth={260} panels={[{ key: 'RMSE', label: 'RMSE against TOR (µg/filter)', fmt: f2 }, { key: 'bias', label: 'mean error (µg/filter)', ref: 0, fmt: (v) => signed(v) }, { key: 'R2', label: 'predictive R²', fmt: f2 }]} />
    </ChartFrame>
  )
}

function Curves({ d }: { d: ForwardData }) {
  const [prep, setPrep] = useSearchParam<'spline' | 'raw'>('xf-curves.prep', 'spline', ['spline', 'raw'])
  const [metric, setMetric] = useSearchParam<'RMSE' | 'bias' | 'R²'>('xf-curves.metric', 'RMSE', ['RMSE', 'bias', 'R²'])
  const lg = useLegend()
  const splits = Object.keys(SPLIT_COLOR)
  const series = splits.filter((s) => lg.show(s)).map((s) => {
    const rows = d.curves.filter((c) => c.preprocessing === prep && c.split === s)
    const byK = d3.groups(rows, (c) => c.k).sort((a, b) => a[0] - b[0])
    return { id: s, label: s, group: s, color: SPLIT_COLOR[s], marks: false,
      points: byK.map(([k, cs]) => { const v = cs.map((c) => c[metric]); return { x: k, y: d3.mean(v)!, lo: cs.length > 1 ? d3.min(v) : undefined, hi: cs.length > 1 ? d3.max(v) : undefined } }) }
  })
  const kSel = d.settings?.k_selected?.[prep]
  return (
    <ChartFrame id="xf-curves" source={source('Curves')} provenance={runLine(d)}
      title="By number of components — does time transfer need a different k?"
      subtitle="Mean over repeats (band: range over the five repeats). The dotted line is the one-SE k chosen by site-grouped CV in the decision workbook."
      controls={<><Segmented label="Preprocessing" value={prep} options={['spline', 'raw'] as const} onChange={setPrep} /><Segmented label="Read" value={metric} options={['RMSE', 'bias', 'R²'] as const} onChange={setMetric} /></>}>
      <SweepPlot series={series} xLabel="PLS components k" yLabel={metric === 'RMSE' ? 'RMSE (µg/filter)' : metric === 'bias' ? 'mean error (µg/filter)' : 'predictive R²'} height={300} dim={lg.dim}
        xMarkers={kSel ? [{ at: kSel, label: `one-SE k = ${kSel}` }] : []} reference={metric === 'bias' ? 0 : undefined} yFormat={f2} />
      <Legend {...lg.props} items={splits.map((s) => ({ label: s, color: SPLIT_COLOR[s], shape: 'line' as const }))} />
    </ChartFrame>
  )
}

function Years({ d }: { d: ForwardData }) {
  return (
    <ChartFrame id="xf-years" source={source('Years')} title="IMPROVE calibration filters by year" exportable={false}>
      <table className="census-table">
        <thead><tr><th>Year</th><th>Filters</th><th>Sites</th><th>Median TOR EC (µg/filter)</th></tr></thead>
        <tbody>{d.years.map((y) => <tr key={y.year}><td>{y.year}</td><td>{y.filters.toLocaleString()}</td><td>{y.sites}</td><td>{f2(y.median_tor_ug)}</td></tr>)}</tbody>
      </table>
      <Note>Most of the calibration set is 2022, so a "train on earlier, test on later" split can only hold out 2023. A later year also changes which sites and lots are present; this split mixes all of that.</Note>
    </ChartFrame>
  )
}
