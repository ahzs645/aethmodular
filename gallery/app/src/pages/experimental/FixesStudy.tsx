import { useRef } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Note, Segmented } from '@/components/ChartFrame'
import { Legend } from '@/components/Legend'
import { XAxis } from '@/components/Axes'
import { RefLine } from '@/components/RefLines'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { useSearchParam } from '@/lib/url'
import { FONT, INK } from '@/lib/theme'
import { f2, signed } from '@/lib/format'
import { runLine, sourceOf, useXp, type XpBase } from './shared'

/** Quick fixes: the ETAD-0241/0248 exclusion and the February season calendar (experiments_20261022.ipynb §7). */

interface SeasonRow { calendar: string; model: string; season: string; n: number; slope: number; 'slope lo': number; 'slope hi': number; intercept: number }
interface FixesData extends XpBase { seasons: SeasonRow[]; february: Record<string, any>[]; etad: Record<string, Record<string, boolean>> }
const source = sourceOf('FixesStudy')
const CAL_COLOR: Record<string, string> = { 'Feb in Dry (historical)': '#8a8f98', 'Feb in Belg (site calendar)': '#d94801' }
const SEASONS = ['Dry', 'Belg', 'Kiremt']

export function FixesStudy() {
  const { data, state } = useXp<FixesData>('fixes', 'Quick fixes')
  if (!data) return <>{state}</>
  return <>
    <Etad d={data} />
    <SeasonSlopes d={data} />
  </>
}

function Etad({ d }: { d: FixesData }) {
  const exports = Object.entries(d.etad)
  const cell = (v: boolean | undefined) => v === undefined ? <span style={{ color: INK.muted }}>not in this export</span>
    : v ? <span style={{ color: INK.negative, fontWeight: 600 }}>excluded</span> : <span style={{ color: INK.good, fontWeight: 600 }}>kept</span>
  return (
    <ChartFrame id="xq-etad" source={source('Etad')} exportable={false}
      title="ETAD-0241 and ETAD-0248 — one exclusion, two filters on the same day"
      subtitle="ETAD-0248 is a PM10 filter (it has HIPS Fabs) and belongs outside every PM2.5 comparison; ETAD-0241, sampled the same day (2024-08-07), is an ordinary PM2.5 filter. The exclusion registry names ETAD-0248, but the gallery's network exporters fell back to matching by date and excluded ETAD-0241 too (or instead). Since 22 Oct the registry matches the filter ID however the table stores it.">
      <table className="census-table">
        <thead><tr><th>Export</th><th>ETAD-0241 (PM2.5)</th><th>ETAD-0248 (PM10)</th></tr></thead>
        <tbody>{exports.map(([name, v]) => <tr key={name}><td><code>{name}</code></td><td>{cell(v['ETAD-0241'])}</td><td>{cell(v['ETAD-0248'])}</td></tr>)}</tbody>
      </table>
      <Note>Read from the current exports when this page's data was built. PM2.5-only exports never contain ETAD-0248. Fix: research/ftir_hips_chem/scripts/outliers.py apply_exclusion_flags.</Note>
    </ChartFrame>
  )
}

function SeasonSlopes({ d }: { d: FixesData }) {
  const ref = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(ref)
  const tip = useTooltip(ref)
  const models = [...new Set(d.seasons.map((s) => s.model))]
  const [what, setWhat] = useSearchParam<'slope' | 'intercept'>('xq-season.what', 'slope', ['slope', 'intercept'])
  const cals = Object.keys(CAL_COLOR)
  const labelW = 170, rowH = 22, groupGap = 14
  const pw = Math.max(260, width - labelW - 140)
  const vals = d.seasons.flatMap((s) => what === 'slope' ? [s['slope lo'], s['slope hi']] : [s.intercept])
  const x = d3.scaleLinear().domain(d3.extent([...vals, what === 'slope' ? 1 : 0]) as [number, number]).nice().range([0, pw])
  const blocks = models.map((m, mi) => ({ m, top: mi * (SEASONS.length * cals.length * rowH + groupGap + 20) }))
  const h = blocks.length * (SEASONS.length * cals.length * rowH + groupGap + 20) + 40
  return (
    <ChartFrame id="xq-season" source={source('SeasonSlopes')}
      title="February: Dry or Belg? — the Addis seasonal slopes under both calendars"
      subtitle="Deming fits of FTIR EC against HIPS BC (MAC 10) per season, for SPARTAN's published EC and the full spline and raw calibrations, with 95 % bootstrap intervals for the slope. The question is whether Dry and Belg stay low against Kiremt when February moves."
      controls={<Segmented label="Read" value={what} options={['slope', 'intercept'] as const} onChange={setWhat} />}>
      <div ref={ref} style={{ position: 'relative' }}>
        <svg width={labelW + pw + 140} height={h} fontFamily={FONT.family}>
          {blocks.map(({ m, top }) => <g key={m} transform={`translate(0,${top})`}>
            <text x={0} y={14} fontSize={11.5} fontWeight={700} fill={INK.text}>{m}</text>
            <RefLine orient="v" at={labelW + x(what === 'slope' ? 1 : 0)} from={20} to={20 + SEASONS.length * cals.length * rowH} />
            {SEASONS.flatMap((s, si) => cals.map((c, ci) => {
              const r = d.seasons.find((z) => z.model === m && z.season === s && z.calendar === c)
              if (!r) return null
              const cy = 20 + (si * cals.length + ci) * rowH + rowH / 2, v = what === 'slope' ? r.slope : r.intercept
              return <g key={`${s}${c}`} onMouseEnter={(e) => tip.show(e, [`${m} · ${s}`, c, `n ${r.n}`, `slope ${f2(r.slope)} (95 % ${f2(r['slope lo'])} to ${f2(r['slope hi'])})`, `intercept ${signed(r.intercept)} µg/m³`])} onMouseLeave={tip.hide}>
                {ci === 0 && <text x={labelW - 10} y={cy + rowH / 2} dy="0.32em" textAnchor="end" fontSize={11} fill={INK.text}>{s}</text>}
                {what === 'slope' && <line x1={labelW + x(r['slope lo'])} x2={labelW + x(r['slope hi'])} y1={cy} y2={cy} stroke={CAL_COLOR[c]} strokeWidth={2} strokeOpacity={0.6} />}
                <circle cx={labelW + x(v)} cy={cy} r={5} fill={CAL_COLOR[c]} />
                <text x={labelW + pw + 8} y={cy} dy="0.32em" fontSize={10} fill={INK.muted}>n {r.n} · {what === 'slope' ? f2(v) : signed(v)}</text>
              </g>
            }))}
          </g>)}
          <g transform={`translate(${labelW},${h - 34})`}><XAxis scale={x} y={0} label={what === 'slope' ? 'Deming slope vs HIPS BC' : 'Deming intercept (µg/m³)'} tickCount={6} labelDy={28} format={(v) => what === 'slope' ? f2(+v) : signed(+v)} /></g>
        </svg>
        {tip.node}
      </div>
      <Legend items={cals.map((c) => ({ label: c, color: CAL_COLOR[c] }))} />
      <Note>{runLine(d)}. These are the 233-filter evaluation set's counts ({d.february.length} February filters with HIPS move between seasons); the poster's 189-filter counts (73/51/65 → 55/69/65) are a subset. The seasonal analog cohorts of the 28 Oct meeting are not refitted here.</Note>
    </ChartFrame>
  )
}
