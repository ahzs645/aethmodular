import { useMemo } from 'react'
import * as d3 from 'd3'
import { ChartFrame, ExplainOnly, Note, Segmented } from '@/components/ChartFrame'
import { HeatMatrix } from '@/components/HeatMatrix'
import { BarRows, type BarRow } from '@/components/BarRows'
import { Legend } from '@/components/Legend'
import { RefLine } from '@/components/RefLines'
import { useColourScheme } from '@/lib/colourSchemes'
import { useSearchParam } from '@/lib/url'
import { INK, PREPROCESSING_COLOR } from '@/lib/theme'
import { coef, coefSigned, f2, fitSummary, signed } from '@/lib/format'
import { useMac, type MacView } from '@/lib/mac'
import { Scatter, runLine, sourceOf, useXp, type XpBase } from './shared'
import { FixedSetNote } from '@/lib/markedRows'

/** Test 4: preprocessing × window × pool × k on the same filters (experiments_20261022.ipynb §4). */

interface Cell { preprocessing: string; window: string; pool: string; 'k rule': string; k: number; 'IMPROVE RMSE': number; 'IMPROVE bias': number; 'IMPROVE R²': number; 'IMPROVE RMSE heaviest tenth': number; 'Addis slope': number; 'Addis intercept': number; 'Addis n': number }
interface Effect { term: string; contrast: string; grid: string; 'IMPROVE RMSE': number; 'IMPROVE bias': number; 'Addis intercept': number; 'Addis slope': number }
interface AttributionData extends XpBase { grid: Cell[]; effects: Effect[]; published: { slope: number; intercept: number; n: number } }

const source = sourceOf('AttributionStudy')
const OUTCOMES = ['Addis intercept', 'Addis slope', 'IMPROVE RMSE', 'IMPROVE RMSE heaviest tenth', 'IMPROVE bias'] as const
type Outcome = typeof OUTCOMES[number]
const PREP_COLOR: Record<string, string> = { raw: PREPROCESSING_COLOR.raw, spline: PREPROCESSING_COLOR.spline, vibes: PREPROCESSING_COLOR.vibes }
const rowKey = (c: Cell) => `${c.preprocessing} · ${c.window === 'full' ? '4000–500' : '4000–1425'}`
const colKey = (c: Cell) => `${c.pool} · ${c['k rule']}`

/**
 * The run's Addis fits are against HIPS BC at MAC 10. At the reader's MAC (lib/mac) every Addis slope, and so every
 * slope effect (a difference of slopes), scales by MAC/10; intercepts and the IMPROVE outcomes do not move.
 */
function atMac(d: AttributionData, mac: MacView): AttributionData {
  if (!mac.changed) return d
  const k = (v: number) => mac.slope(v) as number
  return { ...d, grid: d.grid.map((c) => ({ ...c, 'Addis slope': k(c['Addis slope']) })), effects: d.effects.map((e) => ({ ...e, 'Addis slope': k(e['Addis slope']) })),
    published: { ...d.published, slope: k(d.published.slope) } }
}

export function AttributionStudy() {
  const { data: raw, state } = useXp<AttributionData>('attribution', 'Attribution grid')
  const mac = useMac()
  const data = useMemo(() => (raw ? atMac(raw, mac) : null), [raw, mac])
  if (!data) return <>{state}</>
  return <>
    <Findings d={data} />
    <GridHeat d={data} />
    <Effects d={data} />
    <TradeOff d={data} />
  </>
}

function Findings({ d }: { d: AttributionData }) {
  const e = [...d.effects].sort((a, b) => Math.abs(b['Addis intercept']) - Math.abs(a['Addis intercept']))
  const best = [...d.grid].sort((a, b) => a['IMPROVE RMSE'] - b['IMPROVE RMSE'])[0]
  const closest = [...d.grid].sort((a, b) => Math.abs(a['Addis intercept']) - Math.abs(b['Addis intercept']))[0]
  return (
    <ExplainOnly>
      <ChartFrame id="xt-findings" title="Attribution grid — short answers" exportable={false}>
        <ol style={{ margin: '4px 0 0 18px', lineHeight: 1.55 }}>
          {e.slice(0, 3).map((x) => <li key={x.grid + x.term}><strong>{x.term}</strong> ({x.grid}; {x.contrast}): Addis intercept {coefSigned(x['Addis intercept'])} µg/m³, IMPROVE RMSE {signed(x['IMPROVE RMSE'])} µg.</li>)}
          <li><strong>Lowest IMPROVE error:</strong> {rowKey(best)}, {colKey(best)} (RMSE {f2(best['IMPROVE RMSE'])} µg; Addis intercept {coefSigned(best['Addis intercept'])}). <strong>Addis closest to HIPS:</strong> {rowKey(closest)}, {colKey(closest)} (intercept {coefSigned(closest['Addis intercept'])}; IMPROVE RMSE {f2(closest['IMPROVE RMSE'])}).</li>
        </ol>
        <Note>{runLine(d)}. Effects are differences between factor levels averaged over the other two factors of a complete 2 × 2 × 2 sub-grid; an interaction is half the change of one factor's effect across the other's levels.</Note>
      </ChartFrame>
    </ExplainOnly>
  )
}

function GridHeat({ d }: { d: AttributionData }) {
  const { label } = useMac()
  const [out, setOut] = useSearchParam<Outcome>('xt-grid.outcome', 'Addis intercept', OUTCOMES)
  const div = useColourScheme('xt-grid-div', 'diverging')
  const seq = useColourScheme('xt-grid-seq', 'sequential')
  const rows = [...new Set(d.grid.map(rowKey))], cols = [...new Set(d.grid.map(colKey))]
  const vals = d.grid.map((c) => c[out])
  const diverging = out === 'Addis intercept' || out === 'IMPROVE bias' || out === 'Addis slope'
  const centre = out === 'Addis slope' ? 1 : 0
  const ext = Math.max(...vals.map((v) => Math.abs(v - centre)))
  const colour = diverging
    ? d3.scaleDiverging(d3.interpolateRgbBasis(div.stops)).domain([centre - ext, centre, centre + ext])
    : d3.scaleSequential(d3.interpolateRgbBasis(seq.stops)).domain(d3.extent(vals) as [number, number])
  const fmtOut = (v: number) => (out.includes('RMSE') ? f2(v) : out === 'IMPROVE bias' ? signed(v) : out === 'Addis slope' ? coef(v) : coefSigned(v))
  const byKey = new Map(d.grid.map((c) => [`${rowKey(c)}|${colKey(c)}`, c]))
  return (
    <ChartFrame id="xt-grid" source={source('GridHeat')} provenance={runLine(d)}
      title="The grid — every combination on the same filters"
      subtitle={`Rows: preprocessing and spectral window (spline exists only above 1425 cm⁻¹, so its full-range cell is missing). Columns: calibration pool (all IMPROVE, or the 500 lowest TOR OC/EC filters) and k (fixed 8, as on the poster, or the one-SE choice; the cell shows the k used). SPARTAN published on the same Addis filters: ${fitSummary({ slope: d.published.slope, intercept: d.published.intercept }, { sep: ', ', signedIntercept: true, unit: 'µg/m³' })}.`}
      controls={<Segmented label="Colour" value={out} options={OUTCOMES} onChange={setOut} />}>
      {/* columns: the pool over its two k rules (a column group), the k rule under it */}
      <HeatMatrix rows={rows.map((r) => ({ key: r, color: PREP_COLOR[r.split(' · ')[0]], weight: 600 }))}
        columns={cols.map((c) => ({ key: c, label: c.split(' · ')[1], group: c.split(' · ')[0], weight: 600, size: 10 }))}
        cellWidth={[90, 150]} cellHeight={40} gap={2} labelSize={11} labelWeight={600} colLabelSize={10}
        cell={(i, j) => {
          const c = byKey.get(`${rows[i]}|${cols[j]}`)
          if (!c) return undefined
          return { value: c[out], fill: colour(c[out]), label: fmtOut(c[out]), sublabel: `k ${c.k}`,
            tip: [`${rowKey(c)} · ${colKey(c)} (k ${c.k})`, `IMPROVE out-of-site RMSE ${f2(c['IMPROVE RMSE'])} µg (heaviest tenth ${f2(c['IMPROVE RMSE heaviest tenth'])}), bias ${signed(c['IMPROVE bias'])}, R² ${f2(c['IMPROVE R²'])}`,
              `Addis vs HIPS: ${fitSummary({ slope: c['Addis slope'], intercept: c['Addis intercept'] }, { sep: ', ', signedIntercept: true, unit: 'µg/m³' })} (n ${c['Addis n']})`] }
        }}
        legend={{ scale: colour, label: out, scheme: diverging ? div.scheme : seq.scheme, format: (v) => out.includes('RMSE') ? f2(v) : out === 'Addis slope' ? coef(v) : signed(v) }} />
      <Note>IMPROVE: every filter predicted by a model trained without its site fold (the common test set); Addis: models fitted on all of IMPROVE, against HIPS BC; slopes at {label} (set under Filters; the run used MAC 10, and the MAC moves only the slopes).</Note>
      <FixedSetNote what="the run's grid, each cell scored on the run's own IMPROVE and Addis filters" />
    </ChartFrame>
  )
}

function Effects({ d }: { d: AttributionData }) {
  const [out, setOut] = useSearchParam<Outcome>('xt-effects.outcome', 'Addis intercept', OUTCOMES.filter((o) => o !== 'IMPROVE RMSE heaviest tenth'))
  const ext = Math.max(...d.effects.map((e) => Math.abs(e[out as keyof Effect] as number))) || 1
  const domain = d3.scaleLinear().domain([-ext, ext]).nice().domain() as [number, number]
  // grouped by sub-grid, in the export's order
  const grids = [...new Set(d.effects.map((e) => e.grid))]
  const bars: BarRow[] = grids.flatMap((g) => d.effects.filter((e) => e.grid === g)).map((e) => {
    const v = e[out as keyof Effect] as number, inter = e.term.includes('×')
    return { key: `${e.grid}|${e.term}`, label: e.term, group: e.grid, value: v, valueLabel: signed(v), color: v < 0 ? INK.positive : INK.negative, opacity: inter ? 0.45 : 0.85,
      italic: inter, labelColor: inter ? INK.muted : INK.text,
      tip: [e.term, e.contrast, `${out}: ${signed(v, 3)}`, `IMPROVE RMSE ${signed(e['IMPROVE RMSE'], 3)} · Addis intercept ${signed(e['Addis intercept'], 3)} · Addis slope ${signed(e['Addis slope'], 3)}`] }
  })
  return (
    <ChartFrame id="xt-effects" source={source('Effects')} provenance={runLine(d)}
      title="Effects and interactions — which choice moves the result, alone and together"
      subtitle="Bars: the average change in the chosen outcome when a factor moves from its first level to its second (main effects), and how much one factor's effect depends on another (interactions, marked ×). Two complete 2 × 2 × 2 sub-grids: raw vs spline at 1425–4000, and full vs 1425 window on raw spectra."
      controls={<Segmented label="Outcome" value={out} options={OUTCOMES.filter((o) => o !== 'IMPROVE RMSE heaviest tenth')} onChange={setOut} />}>
      <BarRows rows={bars} domain={domain} rowHeight={24} barHeight={14} labelSize={10.5} maxLabelWidth={230} groupLabels
        axis={{ label: `effect on ${out}`, ticks: 6, format: (v) => signed(+v) }} />
      <Legend items={[{ label: 'raises the outcome', color: INK.negative, shape: 'square' }, { label: 'lowers it', color: INK.positive, shape: 'square' }]} />
      <FixedSetNote what="the run's grid, each cell scored on the run's own IMPROVE and Addis filters" />
    </ChartFrame>
  )
}

function TradeOff({ d }: { d: AttributionData }) {
  const pts = d.grid.map((c) => ({ x: c['IMPROVE RMSE'], y: c['Addis intercept'], color: PREP_COLOR[c.preprocessing], r: c.pool === 'all IMPROVE' ? 6 : 4, hollow: c['k rule'] !== 'k = 8',
    key: `${rowKey(c)}${colKey(c)}`, tip: [`${rowKey(c)} · ${colKey(c)} (k ${c.k})`, `IMPROVE RMSE ${f2(c['IMPROVE RMSE'])} µg`, `Addis intercept ${coefSigned(c['Addis intercept'])} µg/m³, slope ${coef(c['Addis slope'])}`] }))
  return (
    <ChartFrame id="xt-tradeoff" source={source('TradeOff')} provenance={runLine(d)}
      title="Trade-off — IMPROVE error against the Addis intercept, every cell"
      subtitle="A cell that is good on both sits low-left with an intercept near 0. If the cells line up, what helps Addis costs IMPROVE accuracy, and IMPROVE cannot choose the setting for Addis.">
      <Scatter points={pts} xLabel="IMPROVE out-of-site RMSE against TOR (µg/filter)" yLabel="Addis Deming intercept vs HIPS (µg/m³)" maxSide={400}
        under={({ y, iw }) => <><RefLine orient="h" at={y(0)} from={0} to={iw} /><RefLine orient="h" at={y(d.published.intercept)} from={0} to={iw} /><text x={iw - 4} y={y(d.published.intercept) - 4} textAnchor="end" fontSize={10} fill={INK.muted}>SPARTAN published</text></>} />
      <Legend items={[...Object.entries(PREP_COLOR).map(([k, c]) => ({ label: k, color: c })), { label: 'large = all IMPROVE, small = 500 lowest OC/EC', color: INK.muted }, { label: 'hollow = one-SE k', color: INK.muted, shape: 'ring' as const }]} />
      <FixedSetNote what="the run's grid, each cell scored on the run's own IMPROVE and Addis filters" />
    </ChartFrame>
  )
}
