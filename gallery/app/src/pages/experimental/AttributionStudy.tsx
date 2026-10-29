import { useMemo, useRef } from 'react'
import * as d3 from 'd3'
import { ChartFrame, ExplainOnly, Note, Segmented } from '@/components/ChartFrame'
import { ColorLegend } from '@/components/ColorLegend'
import { HeatCell } from '@/components/HeatCell'
import { Legend } from '@/components/Legend'
import { XAxis } from '@/components/Axes'
import { RefLine } from '@/components/RefLines'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { useColourScheme } from '@/lib/colourSchemes'
import { useSearchParam } from '@/lib/url'
import { FONT, INK, inkOn, themed } from '@/lib/theme'
import { coef, coefSigned, f2, signed } from '@/lib/format'
import { useMac, type MacView } from '@/lib/mac'
import { Scatter, runLine, sourceOf, useXp, type XpBase } from './shared'

/** Test 4: preprocessing × window × pool × k on the same filters (experiments_20261022.ipynb §4). */

interface Cell { preprocessing: string; window: string; pool: string; 'k rule': string; k: number; 'IMPROVE RMSE': number; 'IMPROVE bias': number; 'IMPROVE R²': number; 'IMPROVE RMSE heaviest tenth': number; 'Addis slope': number; 'Addis intercept': number; 'Addis n': number }
interface Effect { term: string; contrast: string; grid: string; 'IMPROVE RMSE': number; 'IMPROVE bias': number; 'Addis intercept': number; 'Addis slope': number }
interface AttributionData extends XpBase { grid: Cell[]; effects: Effect[]; published: { slope: number; intercept: number; n: number } }

const source = sourceOf('AttributionStudy')
const OUTCOMES = ['Addis intercept', 'Addis slope', 'IMPROVE RMSE', 'IMPROVE RMSE heaviest tenth', 'IMPROVE bias'] as const
type Outcome = typeof OUTCOMES[number]
const PREP_COLOR: Record<string, string> = { get raw() { return themed('#525252', '#9aa3ad') }, spline: '#2171b5', vibes: '#e6550d' }
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
  const ref = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(ref)
  const tip = useTooltip(ref)
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
  const labelW = 150, cw = Math.max(90, Math.min(150, (width - labelW - 20) / cols.length)), ch = 40
  return (
    <ChartFrame id="xt-grid" source={source('GridHeat')} provenance={runLine(d)}
      title="The grid — every combination on the same filters"
      subtitle={`Rows: preprocessing and spectral window (spline exists only above 1425 cm⁻¹, so its full-range cell is missing). Columns: calibration pool (all IMPROVE, or the 500 lowest TOR OC/EC filters) and k (fixed 8, as on the poster, or the one-SE choice; the cell shows the k used). SPARTAN published on the same Addis filters: slope ${coef(d.published.slope)}, intercept ${coefSigned(d.published.intercept)} µg/m³.`}
      controls={<Segmented label="Colour" value={out} options={OUTCOMES} onChange={setOut} />}>
      <div ref={ref} style={{ position: 'relative', overflowX: 'auto' }}>
        <svg width={labelW + cols.length * cw + 10} height={rows.length * ch + 60} fontFamily={FONT.family}>
          {cols.map((c, j) => <text key={c} x={labelW + j * cw + cw / 2} y={14} textAnchor="middle" fontSize={10} fill={INK.muted}>{c.split(' · ')[0]}</text>)}
          {cols.map((c, j) => <text key={`k${c}`} x={labelW + j * cw + cw / 2} y={27} textAnchor="middle" fontSize={10} fill={INK.text} fontWeight={600}>{c.split(' · ')[1]}</text>)}
          {rows.map((r, i) => <text key={r} x={labelW - 8} y={36 + i * ch + ch / 2} dy="0.32em" textAnchor="end" fontSize={11} fill={PREP_COLOR[r.split(' · ')[0]]} fontWeight={600}>{r}</text>)}
          {d.grid.map((c) => {
            const i = rows.indexOf(rowKey(c)), j = cols.indexOf(colKey(c)), v = c[out], fill = colour(v)
            return <g key={`${i}-${j}`}>
              <HeatCell x={labelW + j * cw + 1} y={36 + i * ch + 1} width={cw - 2} height={ch - 2} fill={fill}
                onEnter={(e) => tip.show(e, [`${rowKey(c)} · ${colKey(c)} (k ${c.k})`, `IMPROVE out-of-site RMSE ${f2(c['IMPROVE RMSE'])} µg (heaviest tenth ${f2(c['IMPROVE RMSE heaviest tenth'])}), bias ${signed(c['IMPROVE bias'])}, R² ${f2(c['IMPROVE R²'])}`,
                  `Addis vs HIPS: slope ${coef(c['Addis slope'])}, intercept ${coefSigned(c['Addis intercept'])} µg/m³ (n ${c['Addis n']})`])} onLeave={tip.hide} />
              <text x={labelW + j * cw + cw / 2} y={36 + i * ch + ch / 2 - 4} textAnchor="middle" fontSize={11} fontWeight={600} fill={inkOn(fill)} pointerEvents="none">{out.includes('RMSE') ? f2(v) : out === 'IMPROVE bias' ? signed(v) : out === 'Addis slope' ? coef(v) : coefSigned(v)}</text>
              <text x={labelW + j * cw + cw / 2} y={36 + i * ch + ch / 2 + 10} textAnchor="middle" fontSize={9} fill={inkOn(fill)} opacity={0.8} pointerEvents="none">k {c.k}</text>
            </g>
          })}
        </svg>
        {tip.node}
      </div>
      <div className="legend"><ColorLegend scale={colour} label={out} format={(v) => out.includes('RMSE') ? f2(v) : out === 'Addis slope' ? coef(v) : signed(v)} scheme={diverging ? div.scheme : seq.scheme} /></div>
      <Note>IMPROVE: every filter predicted by a model trained without its site fold (the common test set); Addis: models fitted on all of IMPROVE, against HIPS BC; slopes at {label} (set under Filters; the run used MAC 10, and the MAC moves only the slopes).</Note>
    </ChartFrame>
  )
}

function Effects({ d }: { d: AttributionData }) {
  const ref = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(ref)
  const tip = useTooltip(ref)
  const [out, setOut] = useSearchParam<Outcome>('xt-effects.outcome', 'Addis intercept', OUTCOMES.filter((o) => o !== 'IMPROVE RMSE heaviest tenth'))
  const grids = [...new Set(d.effects.map((e) => e.grid))]
  const labelW = 230, rowH = 24
  const pw = Math.max(240, width - labelW - 60)
  const ext = Math.max(...d.effects.map((e) => Math.abs(e[out as keyof Effect] as number))) || 1
  const x = d3.scaleLinear().domain([-ext, ext]).nice().range([0, pw])
  let yy = 0
  const blocks = grids.map((g) => { const es = d.effects.filter((e) => e.grid === g); const top = yy; yy += es.length * rowH + 34; return { g, es, top } })
  return (
    <ChartFrame id="xt-effects" source={source('Effects')} provenance={runLine(d)}
      title="Effects and interactions — which choice moves the result, alone and together"
      subtitle="Bars: the average change in the chosen outcome when a factor moves from its first level to its second (main effects), and how much one factor's effect depends on another (interactions, marked ×). Two complete 2 × 2 × 2 sub-grids: raw vs spline at 1425–4000, and full vs 1425 window on raw spectra."
      controls={<Segmented label="Outcome" value={out} options={OUTCOMES.filter((o) => o !== 'IMPROVE RMSE heaviest tenth')} onChange={setOut} />}>
      <div ref={ref} style={{ position: 'relative' }}>
        <svg width={labelW + pw + 60} height={yy + 30} fontFamily={FONT.family}>
          {blocks.map(({ g, es, top }) => <g key={g} transform={`translate(0,${top})`}>
            <text x={0} y={14} fontSize={11} fontWeight={700} fill={INK.text}>{g}</text>
            <RefLine orient="v" at={labelW + x(0)} from={20} to={20 + es.length * rowH} />
            {es.map((e, i) => {
              const v = e[out as keyof Effect] as number, cy = 20 + i * rowH + rowH / 2, inter = e.term.includes('×')
              return <g key={e.term} onMouseEnter={(ev) => tip.show(ev, [e.term, e.contrast, `${out}: ${signed(v, 3)}`, `IMPROVE RMSE ${signed(e['IMPROVE RMSE'], 3)} · Addis intercept ${signed(e['Addis intercept'], 3)} · Addis slope ${signed(e['Addis slope'], 3)}`])} onMouseLeave={tip.hide}>
                <text x={labelW - 8} y={cy} dy="0.32em" textAnchor="end" fontSize={10.5} fill={inter ? INK.muted : INK.text} fontStyle={inter ? 'italic' : 'normal'}>{e.term}</text>
                <rect x={labelW + Math.min(x(0), x(v))} y={cy - 7} width={Math.abs(x(v) - x(0))} height={14} fill={v < 0 ? INK.positive : INK.negative} fillOpacity={inter ? 0.45 : 0.85} />
                <text x={labelW + x(v) + (v < 0 ? -4 : 4)} y={cy} dy="0.32em" textAnchor={v < 0 ? 'end' : 'start'} fontSize={9.5} fill={INK.muted}>{signed(v)}</text>
              </g>
            })}
          </g>)}
          <g transform={`translate(${labelW},${yy})`}><XAxis scale={x} y={0} label={`effect on ${out}`} tickCount={6} labelDy={28} format={(v) => signed(+v)} /></g>
        </svg>
        {tip.node}
      </div>
      <Legend items={[{ label: 'raises the outcome', color: INK.negative, shape: 'square' }, { label: 'lowers it', color: INK.positive, shape: 'square' }]} />
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
    </ChartFrame>
  )
}
