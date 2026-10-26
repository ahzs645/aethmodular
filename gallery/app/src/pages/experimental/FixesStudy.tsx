import { ChartFrame, Note, Segmented } from '@/components/ChartFrame'
import { Legend } from '@/components/Legend'
import { IntervalStrip, type IntervalBlock } from '@/components/IntervalStrip'
import { useSearchParam } from '@/lib/url'
import { INK } from '@/lib/theme'
import { coef, coefSigned } from '@/lib/format'
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
    <ChartFrame id="xq-etad" source={source('Etad')} exportable={false} provenance={runLine(d)}
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
  const models = [...new Set(d.seasons.map((s) => s.model))]
  const [what, setWhat] = useSearchParam<'slope' | 'intercept'>('xq-season.what', 'slope', ['slope', 'intercept'])
  const cals = Object.keys(CAL_COLOR)
  // a block per model, a row per season, the two calendars stacked in the row
  const blocks: IntervalBlock[] = models.map((m) => ({ key: m, title: m, rows: SEASONS.map((s) => {
    const rs = cals.map((c) => d.seasons.find((z) => z.model === m && z.season === s && z.calendar === c))
    return {
      key: s, label: s, note: rs.map((r) => r ? `n ${r.n}` : '—').join(' → '),
      tip: [`${m} · ${s}`, ...rs.flatMap((r, ci) => r ? [`${cals[ci]}: n ${r.n}, slope ${coef(r.slope)} (95 % ${coef(r['slope lo'])} to ${coef(r['slope hi'])}), intercept ${coefSigned(r.intercept)} µg/m³`] : [])],
      marks: [rs.flatMap((r, ci) => r ? [{ mid: what === 'slope' ? r.slope : r.intercept, lo: what === 'slope' ? r['slope lo'] : undefined, hi: what === 'slope' ? r['slope hi'] : undefined,
        color: CAL_COLOR[cals[ci]], dy: (ci - 0.5) * 8, r: 4 }] : [])],
    }
  }) }))
  return (
    <ChartFrame id="xq-season" source={source('SeasonSlopes')} provenance={runLine(d)}
      title="February: Dry or Belg? — the Addis seasonal slopes under both calendars"
      subtitle="Deming fits of FTIR EC against HIPS BC (MAC 10) per season, for SPARTAN's published EC and the full spline and raw calibrations, with 95 % bootstrap intervals for the slope. The question is whether Dry and Belg stay low against Kiremt when February moves. On the right: filters per season, historical → site calendar."
      controls={<Segmented label="Read" value={what} options={['slope', 'intercept'] as const} onChange={setWhat} />}>
      <IntervalStrip blocks={blocks} labelWidth={120} rowHeight={26} noteWidth={78}
        panels={[{ label: what === 'slope' ? 'Deming slope vs HIPS BC' : 'Deming intercept (µg/m³)', reference: what === 'slope' ? 1 : 0 }]} />
      <Legend items={cals.map((c) => ({ label: c, color: CAL_COLOR[c] }))} />
      <Note>These are the 233-filter evaluation set's counts ({d.february.length} February filters with HIPS move between seasons); the poster's 189-filter counts (73/51/65 → 55/69/65) are a subset. The seasonal analog cohorts of the 28 Oct meeting are not refitted here.</Note>
    </ChartFrame>
  )
}
