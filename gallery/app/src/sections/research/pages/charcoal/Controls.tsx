/**
 * char_06's controls and char_07's exclusion test, recomputed on the filters shown: is it the filter lot
 * (char_06 §1), is it the loading (char_06 [20], against EC and HIPS Fabs or Fabs/MAC), and does char_05's result
 * survive without the dry season or without the anomalous filters (char_07 [7])?
 */
import { ChartFrame, Empty, Note, Segmented } from '@/components/ChartFrame'
import { Crossplot } from '@/components/Crossplot'
import { IntervalStrip, type IntervalRow } from '@/components/IntervalStrip'
import { Key } from '@/components/Key'
import { fmt, pct } from '@/lib/format'
import { useMac } from '@/lib/mac'
import { INK } from '@/lib/theme'
import { num } from '@/lib/table'
import type { FilterRow } from '@/lib/types'
import { useSearchParam } from '@/lib/url'
import { headline, mannWhitney, spread, type Scored as ScoredOf } from '../../common/charcoal'
import { spearman } from '../../common/stats'
import { BASIS_NOTE, DataTable, Grid, line, pText, shortSeason } from '../../common/ui'
import { provenance, rows, type SpectralExport } from '../spectralCommon'
import { NB, type Charcoal } from './data'
import { DELTA, R_CHAR, R_RAW, scorePoint } from './Scores'

type Scored = ScoredOf<FilterRow>
const med = (v: number[]) => spread(v).mid

export function LotControl({ c, data, setPick }: { c: Charcoal; data: SpectralExport; setPick: React.ReactNode }) {
  const withLot = c.scored.filter((s) => s.lot !== null)
  const lots = [...new Set(withLot.map((s) => s.lot!))].sort((a, b) => a - b)
  const crosstab = lots.map((lot) => ({ lot, ...Object.fromEntries(c.seasons.map((se) => [se.name, withLot.filter((s) => s.lot === lot && s.season === se.name).length])) }))
  const main = lots.length ? lots.reduce((a, b) => (withLot.filter((s) => s.lot === b).length > withLot.filter((s) => s.lot === a).length ? b : a)) : null
  const inMain = withLot.filter((s) => s.lot === main)
  const t1 = c.seasons.map((se) => ({ season: se.name, ...headline(inMain.filter((s) => s.season === se.name)) })).filter((r) => r.n)
  const u1 = mannWhitney(inMain.filter(c.isDry).map((s) => s.delta), inMain.filter((s) => !c.isDry(s)).map((s) => s.delta))
  const dry = withLot.filter(c.isDry)
  const t2 = lots.map((lot) => ({ lot, ...headline(dry.filter((s) => s.lot === lot)) })).filter((r) => r.n)
  const u2 = lots.length === 2 ? mannWhitney(dry.filter((s) => s.lot === lots[0]).map((s) => s.delta), dry.filter((s) => s.lot === lots[1]).map((s) => s.delta)) : null
  return (
    <ChartFrame id="ch-lot" controls={setPick} exportable={false}
      title={u1 && u2 ? `The lot is not the explanation: within lot ${main} the ${c.dryName} season still stands out (p ${pText(u1.p)}); within the ${c.dryName} season the lots do not differ (p ${pText(u2.p)})` : 'Is it the filter lot?'}
      subtitle={`ETAD spectra come from two PTFE lots, and lot 248 is mostly dry-season, so lot and season are confounded. char_06 separates them: within the lot that spans every season, and within the dry season split by lot. Mann–Whitney on delta (two-sided, scipy's normal approximation). The lot is known for the filters the gallery has (char_06 joins it from the SPARTAN filter list); the newer filters have none.`}
      provenance={<>{NB.c06} §1. {provenance(data, ['per_filter', 'lot_within_251', 'lot_within_dry'])}</>}>
      {!withLot.length ? <Empty>No filter in view has a known lot.</Empty> : <Grid min={300}>
        <DataTable rows={crosstab} rowKey={(r) => String(r.lot)} columns={[
          { key: 'l', label: 'Lot', value: (r) => r.lot },
          ...c.seasons.map((se) => ({ key: se.name, label: <span style={{ color: se.color }}>{shortSeason(se.name)}</span>, num: true, value: (r: Record<string, number>) => r[se.name] })),
        ]} />
        <div>
          <DataTable rows={t1} rowKey={(r) => r.season} columns={[
            { key: 's', label: `Lot ${main}, by season`, value: (r) => <span style={{ color: c.colorOf(r.season) }}>{shortSeason(r.season)}</span> },
            { key: 'n', label: 'n', num: true, value: (r) => r.n },
            { key: 'c', label: 'Median r_char', num: true, value: (r) => fmt(r.charMedian, 3) },
            { key: 'd', label: 'Median delta', num: true, value: (r) => fmt(r.deltaMedian, 3) },
          ]} />
          <Note>{c.dryName} vs the rest within lot {main}: U = {fmt(u1?.u, 0)}, p {pText(u1?.p ?? NaN)}. char_06 (all 239, its seasons): U = 4284, p = 9.4e-12.</Note>
        </div>
        <div>
          <DataTable rows={t2} rowKey={(r) => String(r.lot)} columns={[
            { key: 'l', label: `${c.dryName} season, by lot`, value: (r) => `lot ${r.lot}` },
            { key: 'n', label: 'n', num: true, value: (r) => r.n },
            { key: 'c', label: 'Median r_char', num: true, value: (r) => fmt(r.charMedian, 3) },
            { key: 'd', label: 'Median delta', num: true, value: (r) => fmt(r.deltaMedian, 3) },
          ]} />
          <Note>Lot {lots[0]} vs {lots[1]} within the {c.dryName} season: U = {fmt(u2?.u, 0)}, p {pText(u2?.p ?? NaN)}. char_06: U = 634, p = 0.886. char_12 finds the same for its envelope escape (Research → Applicability domain).</Note>
        </div>
      </Grid>}
    </ChartFrame>
  )
}

export function LoadingCheck({ c, data, setPick }: { c: Charcoal; data: SpectralExport; setPick: React.ReactNode }) {
  const m = useMac()
  const [axis, setAxis] = useSearchParam<'fabs' | 'bc'>('ch.abs', 'fabs', ['fabs', 'bc'])
  const absLabel = axis === 'fabs' ? 'HIPS Fabs (Mm⁻¹)' : `HIPS BC = Fabs / ${m.mac} (µg/m³)`
  const absOf = (s: Scored) => (axis === 'fabs' ? s.fabs : m.fromFabs(s.fabs))
  const ecPts = c.scored.filter((s) => s.ec !== null), abPts = c.scored.filter((s) => absOf(s) !== null)
  const rho = (pts: Scored[], x: (s: Scored) => number | null) => spearman(pts.map((s) => x(s)!), pts.map((s) => s.rChar))
  const dryRho = (pts: Scored[], x: (s: Scored) => number | null) => { const d = pts.filter(c.isDry); return d.length > 5 ? rho(d, x) : null }
  // within the dry season, do the two mixture groups differ in loading? (filters with both, as char_06's model frame)
  const model = c.scored.filter((s) => c.isDry(s) && s.fabs !== null && s.ec !== null)
  const tests = (['fabs', 'ec'] as const).map((k) => {
    const a = model.filter((s) => s.low).map((s) => s[k]!), b = model.filter((s) => !s.low).map((s) => s[k]!)
    const saved = rows(data, 'loading_within_dry').find((r) => r.measure === (k === 'fabs' ? 'Fabs' : 'EC'))
    return { k, label: k === 'fabs' ? 'HIPS Fabs (Mm⁻¹)' : 'FTIR EC (µg/m³)', nLow: a.length, nNormal: b.length, low: med(a), normal: med(b), u: mannWhitney(a, b), saved }
  })
  const competing = rows(data, 'competing')
  return (
    <ChartFrame id="ch-loading" controls={<>{setPick}<Segmented label="Absorption" value={axis} options={['fabs', 'bc'] as const} onChange={setAxis} optionLabel={(o) => (o === 'fabs' ? 'HIPS Fabs' : `Fabs / MAC (${m.mac})`)} /></>}
      title={`Not the loading: within the ${c.dryName} season the low-similarity filters carry the same Fabs (p ${pText(tests[0].u?.p ?? NaN)}) and EC (p ${pText(tests[1].u?.p ?? NaN)}) as the rest`}
      subtitle={`Thin deposits give noisier spectra, and noise lowers correlation with any reference, so low loading could fake the anomaly. Left: r_char against the gallery's FTIR EC; right: against HIPS ${axis === 'fabs' ? 'Fabs' : `BC at the MAC set under Filters → units (${m.mac})`}. A gallery row's own EC and Fabs are used; newer filters take the notebook's Fabs and have no EC. Spearman ρ does not depend on the MAC.`}
      provenance={<>{NB.c06} [20], §5. {provenance(data, ['per_filter', 'loading_within_dry', 'competing'])}</>}>
      {!c.scored.length ? <Empty>No scored Addis filters in view.</Empty> : <Grid>
        {ecPts.length >= 3 ? <Crossplot points={ecPts.map((s) => scorePoint(s, s.ec!, s.rChar, c.colorOf(s.season), 'EC', 'r_char', !s.low))} title="r_char vs EC" xLabel="FTIR EC (µg/m³)" yLabel={R_CHAR} fit={false} sameScale={false} maxSide={380} />
          : <Empty>No filter in view carries FTIR EC.</Empty>}
        <Crossplot points={abPts.map((s) => scorePoint(s, absOf(s)!, s.rChar, c.colorOf(s.season), axis === 'fabs' ? 'Fabs' : 'HIPS BC', 'r_char', !s.low))} title={`r_char vs ${axis === 'fabs' ? 'Fabs' : 'Fabs / MAC'}`} xLabel={absLabel} yLabel={R_CHAR} fit={false} sameScale={false} maxSide={380} />
      </Grid>}
      <Key items={[...c.seasons.map((s) => ({ label: shortSeason(s.name), color: s.color })), { label: 'low-similarity component', color: INK.text, kind: 'dot' }, { label: 'normal', color: INK.text, kind: 'hollow' }]} />
      <Note>Spearman ρ of r_char with EC: {fmt(rho(ecPts, (s) => s.ec), 2)} (n {ecPts.length}; {c.dryName} only {fmt(dryRho(ecPts, (s) => s.ec), 2)}); with {axis === 'fabs' ? 'Fabs' : 'HIPS BC'}: {fmt(rho(abPts, absOf), 2)} (n {abPts.length}; {c.dryName} only {fmt(dryRho(abPts, absOf), 2)}). {BASIS_NOTE}</Note>
      <DataTable rows={tests} rowKey={(r) => r.k} columns={[
        { key: 'm', label: `Within the ${c.dryName} season`, value: (r) => r.label },
        { key: 'l', label: 'Low-similarity median (n)', num: true, value: (r) => `${fmt(r.low, 2)} (${r.nLow})` },
        { key: 'o', label: 'Normal median (n)', num: true, value: (r) => `${fmt(r.normal, 2)} (${r.nNormal})` },
        { key: 'p', label: 'Mann–Whitney p', num: true, value: (r) => pText(r.u?.p ?? NaN) },
        { key: 'nb', label: 'char_06 printed', num: true, value: (r) => r.saved ? `${fmt(num(r.saved['low-similarity median']), 2)} vs ${fmt(num(r.saved['normal median']), 2)}, p ${fmt(num(r.saved['Mann–Whitney p']), 3)}` : '—' },
      ]} />
      {competing.length > 0 && <>
        <p className="subtitle" style={{ margin: '10px 0 4px' }}>char_06's logistic regressions of the low-similarity label on season and loading (its 190 filters with EC and Fabs; static, a model fit):</p>
        <DataTable rows={competing} rowKey={(r) => String(r.model)} columns={['model', 'pseudo R²', 'AIC', 'p(is_dry)', 'p(Fabs)', 'p(EC)'].map((k) => ({ key: k, label: k === 'model' ? 'Model' : k, num: k !== 'model', value: (r: (typeof competing)[number]) => (k === 'model' ? String(r[k]) : fmt(num(r[k]), k === 'AIC' ? 1 : 3)) }))} />
      </>}
    </ChartFrame>
  )
}

type Rule = 'calendar' | 'anomaly' | 'both'
const RULE_LABEL: Record<Rule, string> = { calendar: 'Dry season (calendar)', anomaly: 'char_06 anomalies', both: 'Both' }

export function Exclusion({ c, data, setPick }: { c: Charcoal; data: SpectralExport; setPick: React.ReactNode }) {
  const [rule, setRule] = useSearchParam<Rule>('ch.rule', 'calendar', ['calendar', 'anomaly', 'both'])
  const out = (s: Scored) => (rule === 'calendar' ? c.isDry(s) : rule === 'anomaly' ? s.low : c.isDry(s) || s.low)
  const kept = c.scored.filter((s) => !out(s)), dropped = c.scored.filter(out)
  const all = headline(c.scored), k = headline(kept), x = headline(dropped)
  const pts = [...dropped.map((s) => scorePoint(s, s.rChar, s.rRaw, INK.neutral, 'r_char', 'r_raw', true)), ...kept.map((s) => scorePoint(s, s.rChar, s.rRaw, c.colorOf(s.season), 'r_char', 'r_raw'))]
  const strip = (key: string, label: string, set: Scored[], color: string): IntervalRow => {
    const d = spread(set.map((s) => s.delta))
    return { key, label, note: `n=${d.n}`, marks: [[{ mid: d.mid, lo: d.lo, hi: d.hi, color }]], tip: [label, `median delta ${fmt(d.mid, 3)} (IQR ${fmt(d.lo, 3)} to ${fmt(d.hi, 3)})`, `share raw > char ${pct(headline(set).rawWins)}`] }
  }
  const saved = [...rows(data, 'headline_07'), ...rows(data, 'rules_07')]
  const table = [
    { subset: 'All filters shown', ...all },
    { subset: `Kept (${RULE_LABEL[rule].toLowerCase()} excluded)`, ...k },
    { subset: 'Excluded', ...x },
  ]
  return (
    <ChartFrame id="ch-exclusion" controls={<>{setPick}<Segmented label="Exclude" value={rule} options={['calendar', 'anomaly', 'both'] as const} onChange={setRule} optionLabel={(o) => RULE_LABEL[o]} /></>}
      title={`Without the excluded filters raw feedstock still wins for ${pct(k.rawWins)}, with a smaller margin (median delta ${fmt(all.deltaMedian, 3)} → ${fmt(k.deltaMedian, 3)})`}
      subtitle={`char_07 repeats char_05 with the dry season excluded by the calendar (its primary rule), by char_06's low-similarity component, or both. Left: the char_05 scatter with the excluded filters grey. Right: delta = r_raw − r_char, median and interquartile range, for every filter shown, the kept and the excluded.`}
      provenance={<>{NB.c07} [7], §1 and §5. {provenance(data, ['per_filter', 'headline_07', 'rules_07'])}</>}>
      {c.scored.length < 3 ? <Empty>No scored Addis filters in view.</Empty> : <Grid>
        <Crossplot points={pts} title="r_raw vs r_char" xLabel={R_CHAR} yLabel={R_RAW} fit={false} extraLines={line({ slope: 1, intercept: 0 }, INK.identity, '1:1', '5 4')} maxSide={380} />
        <div>
          <IntervalStrip panels={[{ label: DELTA, reference: 0 }]} labelWidth={150} noteWidth={50}
            blocks={[{ key: 'd', rows: [strip('all', 'All shown', c.scored, INK.text), strip('kept', 'Kept', kept, INK.accent), strip('out', 'Excluded', dropped, INK.muted)] }]} />
          <DataTable rows={table} rowKey={(r) => r.subset} columns={[
            { key: 's', label: 'Subset', value: (r) => r.subset },
            { key: 'n', label: 'n', num: true, value: (r) => r.n },
            { key: 'r', label: 'Median r_raw', num: true, value: (r) => fmt(r.rawMedian, 3) },
            { key: 'c', label: 'Median r_char', num: true, value: (r) => fmt(r.charMedian, 3) },
            { key: 'w', label: 'Share raw > char', num: true, value: (r) => fmt(r.rawWins, 3) },
            { key: 'd', label: 'Median delta', num: true, value: (r) => fmt(r.deltaMedian, 3) },
          ]} />
        </div>
      </Grid>}
      <Key items={[...c.seasons.map((s) => ({ label: shortSeason(s.name), color: s.color })), { label: 'excluded', color: INK.neutral, kind: 'hollow' }, { label: '1:1', color: INK.identity, kind: 'dash' }]} />
      {saved.length > 0 && <>
        <p className="subtitle" style={{ margin: '10px 0 4px' }}>char_07's own table (all 239 filters, its seasons). The mean-r columns need each filter's r against every reference, which no script saves:</p>
        <DataTable rows={saved} rowKey={(r, i) => `${r.subset}-${i}`} columns={Object.keys(saved[0]).map((key) => ({ key, label: key, num: key !== 'subset', value: (r: (typeof saved)[number]) => (key === 'subset' ? String(r[key]) : fmt(num(r[key]), key === 'filters' ? 0 : 3)) }))} />
      </>}
    </ChartFrame>
  )
}
