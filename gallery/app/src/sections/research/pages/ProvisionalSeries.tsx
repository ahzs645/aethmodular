import { useMemo } from 'react'
import * as d3 from 'd3'
import { Figure } from '@/components/Figure'
import { RunSeries } from '@/charts/calibration/RunSeries'
import { groupColorFn } from '@/charts/calibration/common'
import { f2, fmt } from '@/lib/format'
import { useMac } from '@/lib/mac'
import { ExcludedNote, FixedSetNote, useCountedFilter, useExclStatus } from '@/lib/markedRows'
import { num, type Row } from '@/lib/table'
import type { CalibRun, CalibRunsFile } from '@/lib/types'
import type { SectionPageProps } from '../../types'
import { plausibilityOf } from '../common/explorerFit'
import { PageIntro, presetName, provenance, rows, useCalibRuns, useSpectral, Waiting } from './spectralCommon'

const PROVISIONAL = 'ocec800_airspec', RAW = 'ocec800_raw'
const SERIES = { prov: 'provisional (Lowest-OC/EC 800 × spline baseline)', raw: 'raw-spectra model (MAC 6 branch)', dep: 'deployed SPARTAN EC' }
/** the HIPS series at the MAC set under Filters → units */
const hipsSeries = (mac: number) => `HIPS Fabs / ${mac}`
const finite = (v: number | null | undefined): v is number => v != null && Number.isFinite(v)
const runAt = (runs: CalibRunsFile, preset: string): CalibRun | undefined => runs.runs.find((r) => r.preset === preset && r.target === 'addis')
const seasonOf = (g: string) => g.split(' (')[0]

/**
 * ftir_29 and ftir_30: what the locked corrected calibration says Addis EC is, as a product. The
 * explorer's per-filter readouts (calibration_runs.json) reproduce ftir_29's series exactly; the
 * bootstrap CI on the HIPS-axis offset and ftir_51's variant ratios come from the notebooks' outputs.
 */
export function ProvisionalSeries(props: SectionPageProps) {
  const { data, error } = useSpectral('provisional_series')
  const { runs, error: runsError } = useCalibRuns()
  const knownIds = useMemo(() => new Set(props.rows.map((r) => r.id)), [props.rows])
  const { mac } = useMac()
  // Filters → excluded acts on the explorer's dated Addis filters by id: the series, its summaries, the season
  // intervals and the presets' Kiremt/Dry ratios use the filters that count; the crossplot crosses the rest under Mark
  const counts = useCountedFilter()
  const status = useExclStatus()

  const view = useMemo(() => {
    if (!runs) return null
    const prov = runAt(runs, PROVISIONAL), raw = runAt(runs, RAW)
    if (!prov || !raw) return null
    const byId = new Map(raw.eval.id.map((id, i) => [id, raw.eval.pred[i]]))
    const perFilter: Row[] = [], pairs: Row[] = []
    let marked = 0
    prov.eval.pred.forEach((p, i) => {
      if (!prov.eval.date?.[i]) return
      const id = prov.eval.id[i]
      // the crossplot gets every filter with its id (Crossplot applies the setting itself); the summaries the counted
      const season = seasonOf(prov.eval.group[i]), dep = prov.eval.deployed?.[i], r = byId.get(id)
      if (finite(r)) pairs.push({ against: SERIES.raw, x: r, y: p, id, season })
      if (finite(dep)) pairs.push({ against: SERIES.dep, x: dep, y: p, id, season })
      if (!counts(id)) { if (status(id) === 'marked') marked++; return }
      perFilter.push({ season, series: SERIES.prov, value: p }, { season, series: hipsSeries(mac), value: prov.eval.ref[i] / mac })
      if (finite(r)) perFilter.push({ season, series: SERIES.raw, value: r })
      if (finite(dep)) perFilter.push({ season, series: SERIES.dep, value: dep })
    })
    const values = prov.eval.pred.filter((v, i) => finite(v) && prov.eval.date?.[i] && counts(prov.eval.id[i])).sort(d3.ascending)
    // the explorer's plausibility card per preset, recomputed on its dated Addis filters that count (common/explorerFit;
    // on all of them it is the saved card)
    const cardOf = (r: CalibRun) => plausibilityOf(r.eval, (i) => counts(r.eval.id[i]))
    const ratios: Row[] = runs.presets.flatMap((pre) => {
      const r = runAt(runs, pre.key)
      const g = r?.plausibility ? cardOf(r).group_medians : null
      return g && g['Kiremt (Jun-Sep)'] && g['Dry (Oct-Feb)'] ? [{ block: 'Gallery explorer presets (site-held-out, rule k)', row: `${presetName(pre)} (k = ${r!.k})`, ratio: g['Kiremt (Jun-Sep)'] / g['Dry (Oct-Feb)'] }] : []
    })
    const readout: Row[] = runs.presets.flatMap((pre) => {
      const r = runAt(runs, pre.key)
      const card = r?.plausibility ? cardOf(r) : null
      return r ? [{ calibration: presetName(pre), k: r.k, tor_slope: r.heldout?.slope ?? null, tor_r2: r.heldout?.R2 ?? null, extrap: r.extrap_pct, q: r.q_residual_pct,
        median: card?.median ?? null, negative: card?.negative_pct ?? null, above8: card?.above_8_pct ?? null }] : []
    })
    const card = cardOf(prov)
    return { prov, raw, perFilter, pairs, ratios, readout, card, marked, dated: prov.eval.date?.filter(Boolean).length ?? 0,
      n: values.length, median: d3.quantile(values, 0.5)!, q1: d3.quantile(values, 0.25)!, q3: d3.quantile(values, 0.75)!, max: values[values.length - 1] }
  }, [runs, mac, counts, status])

  if (!data || !runs) return <Waiting error={error || runsError} />
  if (!view) return <Waiting error="calibration_runs.json has no Addis run for the locked Lowest-OC/EC 800 presets" />
  const committed = rows(data, 'summary')[0]
  const ratios = [...view.ratios, ...rows(data, 'kiremt_dry_ftir51').map((r): Row => ({ block: 'ftir_51 variants (Option A, site-grouped 5-fold CV)', row: `${r.variant} (k = ${r.k})`, ratio: num(r['Kiremt/Dry']) }))]
  const seasons = view.card.group_medians
  // the offset c is an x-intercept on the HIPS axis: fixed in Fabs (Mm⁻¹), so in µg/m³ it is c(Mm⁻¹) / MAC
  const offsets = rows(data, 'offset_ci').map((r): Row => ({ ...r, c_median: num(r['c_median_Mm-1_MAC10']) != null ? num(r['c_median_Mm-1_MAC10'])! / mac : null,
    c_lo: num(r['c_lo95_Mm-1']) != null ? num(r['c_lo95_Mm-1'])! / mac : null, c_hi: num(r['c_hi95_Mm-1']) != null ? num(r['c_hi95_Mm-1'])! / mac : null }))
  const offset = offsets.find((r) => String(r.model).startsWith('AIRSpec'))
  // the Ethiopian season colours from config (meta), keyed by the short season name these figures use
  const seasonColor = groupColorFn(props.meta, view.prov.eval.group).color
  const colors = Object.fromEntries([...new Set(view.prov.eval.group)].map((g) => [seasonOf(g), seasonColor(g)]))

  return <>
    <PageIntro id="provisional-intro" title="The provisional Addis EC series — what the corrected calibration gives" data={data}
      notebooks="research/ftir_ec_phase3/ftir_29_provisional_addis_ec_series, ftir_30_composition_case_and_its_limits; ftir_53 §4 (Kiremt/Dry across variants)">
      <p>The locked Lowest-OC/EC 800 × spline-baseline calibration (k = {view.prov.k}) over {view.n === view.dated ? `all ${view.n}` : `${view.n} of the ${view.dated}`} dated Addis filters{view.n === view.dated ? '' : ' (the rest excluded under Filters → excluded)'}: median {f2(view.median)} µg/m³ (IQR {f2(view.q1)}–{f2(view.q3)}), maximum {f2(view.max)},
        {' '}{fmt(view.card.negative_pct, 2)} % negative and {fmt(view.card.above_8_pct, 2)} % above 8 µg/m³. ftir_29's committed summary gives median {f2(committed?.median_ugm3)} and maximum {f2(committed?.max)} on its {committed?.n_dated} filters: {view.n === committed?.n_dated ? 'the explorer readouts reproduce it on the same filters' : 'on the same filters (Filters → excluded: Include, nothing excluded by hand) the explorer readouts reproduce it'}.
        {' '}No MAC enters the series (predicted µg per filter ÷ sampled volume). Season medians: {Object.entries(seasons).map(([g, v]) => `${seasonOf(g)} ${f2(v)}`).join(', ')}.</p>
      <p>Held out at IMPROVE sites the model never saw, at Addis-like composition, it reproduces TOR EC at slope {f2(view.prov.heldout?.slope)}, R² {f2(view.prov.heldout?.R2)}; {fmt(view.prov.extrap_pct, 2)} % of Addis filters sit beyond its training score envelope, against {fmt(view.raw.extrap_pct, 2)} % for the raw-spectra model (ftir_30).
        {' '}What it cannot claim: no Addis filter has thermal EC, so the validation is the right composition on the wrong continent. Say “provisional” with every number.</p>
      <FixedSetNote subject="the offset figure" what="ftir_15's saved bootstrap intervals (the series, its summaries here, the season intervals, the crossplot and the explorer's Kiremt/Dry ratios follow it)" />
    </PageIntro>

    <RunSeries run={view.prov} runs={runs} meta={props.meta} knownIds={knownIds} />

    <Figure kind="intervals" id="provisional-seasons" data={view.perFilter} row="season" series="series"
      panels={[{ value: 'value', label: 'µg/m³ (median and IQR over dated filters)', reference: 0 }]} labelWidth={110}
      title="Season by season — the provisional series, the raw-model envelope, deployed EC and HIPS"
      subtitle={`Dry (Oct–Feb), Belg (Mar–May), Kiremt (Jun–Sep): the explorer's Ethiopian calendar, February in Dry. The provisional series keeps the Kiremt ≈ 2 × Dry ordering at about half the deployed level; the raw-spectra model is the upper envelope (the unresolved MAC-6 branch). HIPS Fabs / ${mac} (the MAC set under Filters → units) is the optical axis the calibration disagrees with by a near-constant offset.`}
      provenance={<>Stands in for ftir_29 §3 and Figure 3. Per filter: <code>calibration_runs.json</code> (explorer eval of the two locked Lowest-OC/EC 800 presets at Addis).</>}
      aside={<ExcludedNote count={view.marked} />} />

    <Figure kind="crossplot" id="provisional-vs" data={view.pairs} x="x" y="y" color="season" colors={colors} idColumn="id" choose={[{ column: 'against', label: 'Against' }]}
      xLabel="Other series, same filter (µg/m³)" yLabel="Provisional EC (µg/m³)" maxSide={440}
      title="A rescaling, not a reshuffle — the provisional series against deployed EC and the raw model"
      subtitle="Same filters. ftir_29 quotes OLS: provisional = 0.45 × deployed + 0.29 (R² 0.85) and slope 0.55 against the raw model; the fit drawn here is the gallery's estimator. The ordering of days survives; the absolute scale is the conditional part."
      provenance={<>Stands in for ftir_29 Figure 2. Per filter: <code>calibration_runs.json</code>.</>} />

    <Figure kind="intervals" id="provisional-offset" data={offsets} row="model"
      panels={[{ value: 'c_median', lo: 'c_lo', hi: 'c_hi', label: `HIPS-axis offset c = −intercept/slope (µg/m³ at MAC ${mac}; median and 95 % interval)`, reference: 0 }]} labelWidth={170}
      title={`The offset is a number with an interval — ${f2(offset?.c_median)} µg/m³ for the corrected model at MAC ${mac}`}
      subtitle={`Site-cluster bootstrap draws from ftir_15 (B = 200 per model), no refitting. The offset is where the fit meets the HIPS axis, so it is fixed in Fabs units: ${f2(offset?.['c_lo95_Mm-1'])}–${f2(offset?.['c_hi95_Mm-1'])} Mm⁻¹ (95 %) for the corrected model, divided here by the MAC set under Filters → units (the notebook quoted µg/m³ at MAC 10). An interval that excludes zero is an offset, not noise.`}
      provenance={provenance(data, ['offset_ci'])} />

    <Figure kind="intervals" id="provisional-kiremt" data={ratios} row="row" block="block"
      panels={[{ value: 'ratio', label: 'Kiremt / Dry ratio of median predicted EC, Addis', reference: 1 }]} aggregate={{ stat: 'median', interval: 'none' }} labelWidth={390}
      title="Kiremt / Dry under every calibration — rainy-season EC stays about twice the dry season"
      subtitle="The gallery's nine explorer presets (season medians from the explorer's plausibility card) and ftir_51's seven variants as committed in ftir_53. The ratio does not depend on the calibration chosen; the level does."
      provenance={provenance(data, ['kiremt_dry_ftir51'], 'Presets: the season medians of calibration_runs.json runs[].eval, recomputed on the filters that count (on all of them, runs[].plausibility.group_medians).')}
      aside={<><ExcludedNote count={view.marked} /><FixedSetNote subject="the ftir_51 rows" what="ftir_53's saved ratios" /></>} />

    <Figure kind="table" id="provisional-readout" data={view.readout}
      columns={[{ key: 'calibration', label: 'Calibration' }, { key: 'k', label: 'k', format: 'int' }, { key: 'tor_slope', label: 'Held-out TOR slope', format: 'f2' }, { key: 'tor_r2', label: 'Held-out TOR R²', format: 'f3' },
        { key: 'extrap', label: 'Addis beyond score envelope (%)', format: 'f2' }, { key: 'q', label: 'Addis Q-residual OOD (%)', format: 'f2' },
        { key: 'median', label: 'Median Addis EC', format: 'f2' }, { key: 'negative', label: 'Negative (%)', format: 'f2' }, { key: 'above8', label: '> 8 µg/m³ (%)', format: 'f2' }]}
      title="Which calibration the series comes from — validated on IMPROVE, in-domain at Addis, plausible as a record"
      subtitle="ftir_30's argument as a table: the corrected Lowest-OC/EC 800 model holds TOR at slope ≈ 1 on unseen IMPROVE sites and treats almost no Addis filter as exotic, where the raw model puts about a quarter outside its envelope. In-domain is necessary, not sufficient (see Applicability domain)."
      provenance={<>Stands in for ftir_30 Figures 2–4. <code>calibration_runs.json</code> runs[] (heldout, extrap_pct, q_residual_pct; median, negative and above 8 µg/m³ recomputed from runs[].eval on the filters that count).</>}
      aside={<><ExcludedNote count={view.marked} /><FixedSetNote subject="each held-out TOR and OOD column" what="the explorer's saved readout (IMPROVE held-out sites; the share of all its Addis filters)" /></>} />
  </>
}
