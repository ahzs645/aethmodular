/**
 * The per-quarter charts: held-out MAE by model (stability 01 and 06, proportionality 03), signed error
 * (stability 02, proportionality 02), the intercept's paired MAE gain (proportionality 01) and each quarter's
 * share of the later-period test filters (proportionality 08). One facet per site, each on its own y scale.
 */
import type { ScalePoint } from 'd3'
import { ChartFrame, Note, Segmented } from '@/components/ChartFrame'
import { FacetGrid } from '@/components/FacetGrid'
import { Key } from '@/components/Key'
import { SweepPlot, type SweepSeries } from '@/components/SweepPlot'
import { fmt, signed } from '@/lib/format'
import { INK } from '@/lib/theme'
import type { MetaFile } from '@/lib/types'
import { useSearchParam } from '@/lib/url'
import { summarize, type Block, type Model, type Scheme } from '../../common/filterOnly'
import { FoExcludedNote, MODEL_LABEL, NB_PROP, NB_STAB, SCHEMES, SOURCE, barLayer, blocksOf, popLabel, schemeLabel, shortQ, siteColor, statusText, type Settings, type SiteRun } from './data'

const GREY = '#8a8a8a'
const MODEL_STYLE: Record<Model, { color?: string; dash?: string }> = { median: { color: '#b8b8b8', dash: '2 3' }, proportional: { color: GREY, dash: '6 3' }, ols: {} }

/** a quarter's tooltip: block, test n, status and the chosen values */
const tipOf = (b: Block, lines: string[]) => [b.block, `${b.testN} test filters · ${b.trainN} training`, ...(b.status === 'evaluated' ? lines : [statusText(b.status)])]

/** "NA" under each quarter with test filters but no fit, as the notebooks mark them */
const naMarks = (blocks: Block[]) => ({ x, ih }: { x: unknown; ih: number }) => {
  const sx = x as ScalePoint<string>
  return <g>{blocks.filter((b) => b.status !== 'evaluated').map((b) => <text key={b.block} x={sx(b.block) ?? 0} y={ih - 4} textAnchor="middle" fontSize={9} fill={INK.muted}>{b.testN ? 'NA' : '–'}</text>)}</g>
}

function SchemePick({ id, value, onChange }: { id: string; value: Scheme; onChange: (s: Scheme) => void }) {
  return <Segmented label="Split" value={value} options={SCHEMES} onChange={onChange} optionLabel={schemeLabel} title={`${id}: leave-quarter-out trains on every other quarter; later-period only on filters reported before the quarter`} />
}

/** stability 01 / 06 and proportionality 03: held-out MAE by quarter, OLS against the median (and the proportional fit) */
export function QuarterMae({ runs, s, meta }: { runs: SiteRun[]; s: Settings; meta: MetaFile }) {
  const [scheme, setScheme] = useSearchParam<Scheme>('fo.mae.split', 'leave_quarter_out', SCHEMES)
  const [models, setModels] = useSearchParam<'two' | 'three'>('fo.mae.models', 'two', ['two', 'three'])
  const shown: Model[] = models === 'three' ? ['median', 'proportional', 'ols'] : ['median', 'ols']
  return (
    <ChartFrame id="fo-mae" title={`${schemeLabel(scheme)} MAE by quarter — does the linear model beat a constant?`}
      subtitle={`${popLabel(s.population)} population · HIPS predicted from FTIR EC, scored on withheld filters · ${scheme === 'later_period' ? 'training only on filters reported before the quarter' : 'training on every other quarter, earlier and later'}`}
      provenance={<>{NB_STAB} figures 01 (leave quarter out) and 06 (later period); {NB_PROP} figure 03 (three models, later period)</>}
      source={`${SOURCE}/Quarters.tsx › QuarterMae`}
      tip={<>At the release's settings: Addis improves on the training median in all eight withheld quarters, MAE 3.94 against 8.89 Mm⁻¹; in the later-period scheme it evaluates 155 filters at 4.27 against 9.33 Mm⁻¹ with mean error +2.26, and Delhi keeps a strong negative bias. With three models, Addis's intercept model wins all six supported later quarters. The two schemes evaluate different filters, so their MAEs are not an algorithm comparison.</>}
      controls={<><SchemePick id="MAE" value={scheme} onChange={setScheme} />
        <Segmented label="Models" value={models} options={['two', 'three'] as const} onChange={setModels} optionLabel={(o) => (o === 'two' ? 'OLS and median' : 'Add proportional')} /></>}>
      <Key items={shown.map((m) => ({ label: MODEL_LABEL[m], color: m === 'ols' ? INK.text : MODEL_STYLE[m].color!, kind: m === 'ols' ? 'line' : 'dash' }))} />
      <FacetGrid items={runs} minCell={290} maxCols={2} title={(r) => r.site}
        note={(r) => { const sm = summarize(blocksOf(r, scheme)); return `MAE ${fmt(sm.filters.mae.ols * s.k, 2)} vs ${fmt(sm.filters.mae.median * s.k, 2)} · n ${sm.filters.n}` }}
        render={(r, w) => {
          const blocks = blocksOf(r, scheme), color = siteColor(meta, r.site)
          const series: SweepSeries[] = shown.map((m) => ({
            id: m, label: MODEL_LABEL[m], color: MODEL_STYLE[m].color ?? color, dash: MODEL_STYLE[m].dash, width: m === 'ols' ? 2 : 1.5, gaps: true,
            points: blocks.map((b) => ({ x: b.block, y: b.status === 'evaluated' ? b.mae[m] * s.k : null, tip: tipOf(b, shown.map((q) => `${MODEL_LABEL[q]} MAE ${fmt(b.mae[q] * s.k, 2)} ${s.u}`)) })),
          }))
          return <SweepPlot series={series} xScale="point" width={w} height={230} xFormat={shortQ} yLabel={`Held-out MAE (${s.u})`} include={[0]} children={naMarks(blocks)} />
        }} />
      <Note>Every model in a fold uses the same training filters and is scored on the same withheld filters. NA marks a quarter with filters but fewer than {s.minTrain} training filters; – a quarter with none.</Note>
      <FoExcludedNote s={s} />
    </ChartFrame>
  )
}

/** stability 02 and proportionality 02: mean signed error (prediction − reported HIPS) by quarter */
export function QuarterBias({ runs, s, meta }: { runs: SiteRun[]; s: Settings; meta: MetaFile }) {
  const [scheme, setScheme] = useSearchParam<Scheme>('fo.bias.split', 'leave_quarter_out', SCHEMES)
  return (
    <ChartFrame id="fo-bias" title="Signed error by quarter — the bias that remains after choosing the model form"
      subtitle={`${popLabel(s.population)} population · ${schemeLabel(scheme).toLowerCase()} · mean of prediction − reported HIPS; positive = overprediction`}
      provenance={<>{NB_STAB} figure 02 (OLS bars); {NB_PROP} figure 02 (proportional against intercept)</>}
      source={`${SOURCE}/Quarters.tsx › QuarterBias`}
      tip={<>At the release's settings, Addis's aggregate bias is near zero but its quarter biases run from −4.11 to +2.30 Mm⁻¹: opposing errors cancel in the overall mean. Bias persists with either model form. Beijing 2024Q3 stays in and has no test EC beyond its training range; the five non-positive EC predictions are kept and no prediction is clipped.</>}
      controls={<SchemePick id="Bias" value={scheme} onChange={setScheme} />}>
      <Key items={[{ label: 'OLS with intercept (bars)', color: INK.text, kind: 'bar' }, { label: MODEL_LABEL.proportional, color: GREY, kind: 'dash' }]} />
      <FacetGrid items={runs} minCell={290} maxCols={2} title={(r) => r.site}
        note={(r) => { const sm = summarize(blocksOf(r, scheme)); return `overall ${signed(sm.filters.mse.ols * s.k, 2)} ${s.u}` }}
        render={(r, w) => {
          const blocks = blocksOf(r, scheme), color = siteColor(meta, r.site)
          const tip = (b: Block) => tipOf(b, [`OLS ${signed(b.mse.ols * s.k, 2)} ${s.u}`, `Proportional ${signed(b.mse.proportional * s.k, 2)} ${s.u}`])
          const v = (b: Block, m: Model) => (b.status === 'evaluated' ? b.mse[m] * s.k : null)
          const series: SweepSeries[] = [
            { id: 'ols', label: 'OLS', color, width: 0, r: 0, points: blocks.map((b) => ({ x: b.block, y: v(b, 'ols'), tip: tip(b) })) },
            { id: 'prop', label: 'Proportional', color: GREY, dash: '6 3', width: 1.5, gaps: true, points: blocks.map((b) => ({ x: b.block, y: v(b, 'proportional'), tip: tip(b) })) },
          ]
          return <SweepPlot series={series} xScale="point" width={w} height={230} xFormat={shortQ} yLabel={`Mean signed error (${s.u})`} reference={0}
            background={barLayer(blocks.map((b) => ({ x: b.block, y: v(b, 'ols'), color })))} children={naMarks(blocks)} />
        }} />
      <FoExcludedNote s={s} />
    </ChartFrame>
  )
}

/** proportionality 01: MAE(proportional) − MAE(intercept) on identical withheld filters */
export function PairedProportional({ runs, s, meta }: { runs: SiteRun[]; s: Settings; meta: MetaFile }) {
  const [scheme, setScheme] = useSearchParam<Scheme>('fo.pair.split', 'leave_quarter_out', SCHEMES)
  return (
    <ChartFrame id="fo-paired" title="Intercept against proportionality — does an intercept improve prediction beyond Fabs ∝ EC?"
      subtitle={`${popLabel(s.population)} population · ${schemeLabel(scheme).toLowerCase()} · MAE of the through-origin fit minus MAE of OLS on the same filters; positive favours the intercept`}
      provenance={<>{NB_PROP} figure 01</>} source={`${SOURCE}/Quarters.tsx › PairedProportional`}
      tip={<>At the release's settings, Addis improves by 7.64 Mm⁻¹ overall and wins all eight quarters. Other sites are less consistent; Delhi's aggregate favours proportional prediction despite three quarter-level intercept wins. These are statistical coefficients, not physical MAC estimates.</>}
      controls={<SchemePick id="Paired" value={scheme} onChange={setScheme} />}>
      <FacetGrid items={runs} minCell={290} maxCols={2} title={(r) => r.site}
        note={(r) => { const sm = summarize(blocksOf(r, scheme)); return `${signed((sm.filters.mae.proportional - sm.filters.mae.ols) * s.k, 2)} overall · ${sm.interceptBetterBlocks}/${sm.evaluatedBlocks} quarters` }}
        render={(r, w) => {
          const blocks = blocksOf(r, scheme), color = siteColor(meta, r.site)
          const v = (b: Block) => (b.status === 'evaluated' ? (b.mae.proportional - b.mae.ols) * s.k : null)
          const series: SweepSeries[] = [{ id: 'd', label: 'Δ MAE', color, width: 0, r: 0,
            points: blocks.map((b) => ({ x: b.block, y: v(b), tip: tipOf(b, [`Proportional MAE ${fmt(b.mae.proportional * s.k, 2)} ${s.u}`, `OLS MAE ${fmt(b.mae.ols * s.k, 2)} ${s.u}`, `difference ${signed(v(b), 2)}`]) })) }]
          return <SweepPlot series={series} xScale="point" width={w} height={230} xFormat={shortQ} yLabel={`MAE prop. − MAE OLS (${s.u})`} reference={0}
            background={barLayer(blocks.map((b) => ({ x: b.block, y: v(b), color })))} children={naMarks(blocks)} />
        }} />
      <FoExcludedNote s={s} />
    </ChartFrame>
  )
}

/** proportionality 08: each evaluated later-period quarter's share of the supported test filters */
export function TestWeights({ runs, s, meta }: { runs: SiteRun[]; s: Settings; meta: MetaFile }) {
  return (
    <ChartFrame id="fo-weights" title="Quarter contributions — unequal quarter sizes weight the overall later-period error"
      subtitle={`${popLabel(s.population)} population · later period · each bar is a quarter's share of the evaluated test filters`}
      provenance={<>{NB_PROP} figure 08</>} source={`${SOURCE}/Quarters.tsx › TestWeights`}
      tip={<>At the release's settings Delhi's 2024Q2 supplies 26 of 38 later-period test filters (68.4 %). Filter-weighted errors lean on that quarter; equal-quarter errors give each supported quarter the same weight. Neither is chosen for a preferred sign.</>}>
      <FacetGrid items={runs} minCell={290} maxCols={2} title={(r) => r.site}
        note={(r) => { const g = r.later.filter((b) => b.status === 'evaluated'); return `${g.reduce((a, b) => a + b.testN, 0)} filters in ${g.length} quarters` }}
        render={(r, w) => {
          const g = r.later.filter((b) => b.status === 'evaluated'), total = g.reduce((a, b) => a + b.testN, 0), color = siteColor(meta, r.site)
          const share = (b: Block) => (100 * b.testN) / Math.max(1, total)
          const series: SweepSeries[] = [{ id: 'w', label: 'share', color, width: 0, r: 0, points: g.map((b) => ({ x: b.block, y: share(b), tip: [b.block, `${fmt(share(b), 1)} % of evaluated filters`, `n ${b.testN}`] })) }]
          return <SweepPlot series={series} xScale="point" width={w} height={210} xFormat={shortQ} yLabel="Share of evaluated filters (%)" include={[0, Math.max(...g.map(share), 0) * 1.18]}
            background={barLayer(g.map((b) => ({ x: b.block, y: share(b), color })))}
            children={({ x, y }) => <g>{g.map((b) => <text key={b.block} x={(x as ScalePoint<string>)(b.block) ?? 0} y={y(share(b)) - 4} textAnchor="middle" fontSize={9.5} fill={INK.muted}>{fmt(share(b), 0)}%</text>)}</g>} />
        }} />
      <FoExcludedNote s={s} />
    </ChartFrame>
  )
}
