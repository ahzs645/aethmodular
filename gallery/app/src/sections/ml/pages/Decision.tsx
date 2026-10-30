import { useMemo } from 'react'
import { ChartFrame, Note } from '@/components/ChartFrame'
import { Figure } from '@/components/Figure'
import { f2 } from '@/lib/format'
import { demingLambda } from '@/lib/macMath'
import { useMac } from '@/lib/mac'
import { num, type Row } from '@/lib/table'
import type { SectionPageProps } from '../../types'
import { deming } from '../../research/common/stats'
import { MlWaiting, PageIntro, provenance, rows, useMl } from '../common'
import { FlagOnSpartan } from './decision/FlagOnSpartan'
import { FixedSetNote, useCountedFilter } from '@/lib/markedRows'

const NOTEBOOKS = 'research/ftir_hips_chem/decision_workbook_20261022 §2, §3, §4, §6, §7, §8 (results run on the Drive, or the executed notebook)'
const PREP_COLORS = { spline: '#4e79a7', raw: '#f28e2b', vibes: '#59a14f', 'SPARTAN published': '#555555' }
const RULE_COLORS = { 'same kind': '#76b7b2', 'target analogs': '#b07aa1', local: '#e15759' }
/** a percentage to the nearest whole number, signed when asked */
const pct0 = (v: number | null) => (v == null || !Number.isFinite(v) ? '—' : Math.round(v).toString())
const ALL = 'All IMPROVE filters (§2)', HARD = 'Hard groups, union of the five', REST = 'Other IMPROVE filters'

/**
 * decision_workbook_20261022: what to do about FTIR EC where the IMPROVE calibration does not transfer. Inside
 * IMPROVE (TOR the truth, whole sites held out): where the reference model fails (§3), whether specialist
 * calibrations beat it (§4, option B), one model over k (§7, option A); then the flag on every SPARTAN filter (§6)
 * and the decision board (§8). Static where the notebook fitted models (PLS, the classifier); the §6 site tables
 * are rebuilt from the per-filter export and follow the MAC and hand exclusions; every slope against HIPS
 * (Fabs ÷ MAC) is re-expressed exactly at the reader's MAC. §5/§6b and §7b are on the Experimental tab.
 */
export function Decision(_props: SectionPageProps) {
  const { data, error } = useMl('decision')
  const m = useMac()

  const settings = rows(data, 'settings')[0] ?? {}
  const lambda10 = num(settings.lambda_addis) ?? 3.34
  const kSel = useMemo(() => new Map(rows(data, 'cv_choice').map((r) => [String(r.preprocessing), num(r.k) ?? 0])), [data])
  const chosen = [...kSel].map(([series, x]) => ({ series, x }))

  // §2 and §7: out-of-site RMSE over k, on every filter and split into the hard groups and the rest
  const trade = rows(data, 'tradeoff')
  const sweep = useMemo(() => trade.flatMap((r): Row[] => [[ALL, 'RMSE all'], [HARD, 'RMSE hard groups'], [REST, 'RMSE rest']]
    .map(([measure, col]) => ({ measure, preprocessing: r.preprocessing, k: r.k, rmse: r[col] }))), [trade])
  const optionA = useMemo(() => {
    const at = (p: string, k: number) => trade.find((r) => r.preprocessing === p && r.k === k)
    const c = at('spline', kSel.get('spline') ?? 4)
    if (!c) return null
    const ok = trade.filter((r) => (num(r['RMSE rest']) ?? Infinity) <= 1.05 * (num(c['RMSE rest']) ?? 0))
    const b = ok.reduce((a, r) => ((num(r['RMSE hard groups']) ?? Infinity) < (num(a['RMSE hard groups']) ?? Infinity) ? r : a), ok[0])
    return { c, b }
  }, [trade, kSel])

  // §3: bias and RMSE as % of mean TOR; slopes of the prediction on TOR and on IMPROVE HIPS (× MAC/10: x is Fabs/MAC)
  const groups = useMemo(() => rows(data, 'groups').map((r): Row => ({ ...r, biasPct: 100 * (num(r.rel_bias) ?? NaN), rmsePct: 100 * (num(r.rel_rmse) ?? NaN),
    slopeHips: m.slope(num(r['slope vs HIPS'])) })), [data, m])
  const g = (p: string, group: string) => groups.find((r) => r.preprocessing === p && r.group === group)
  // hard groups whose spline slope against HIPS (at the reader's MAC) is steeper than against TOR: where HIPS reads low
  const steeper = groups.filter((r) => r.preprocessing === 'spline' && r.group !== 'None of these' && (num(r.slopeHips) ?? -Infinity) > (num(r['slope vs TOR']) ?? Infinity))
    .map((r) => String(r.group).toLowerCase())
  const hasAllPreps = new Set(groups.map((r) => r.preprocessing)).size > 1

  // §4: the specialist rules against the full calibration set on their target groups
  const spec = rows(data, 'specialists')
  const localRaw = (group: string) => spec.find((r) => r.preprocessing === 'raw' && r.rule === 'local' && r.group === group)

  // §7: the Addis readout over k against HIPS BC at the reader's MAC, beside SPARTAN's published EC, refitted on the
  // filters that count under Filters → excluded (lib/markedRows useCountedFilter); the notebook's k curves are saved
  const isCounted = useCountedFilter()
  const addisK = useMemo(() => {
    const out: Row[] = []
    const add = (series: string, k: number, slope: number | null, intercept: number | null) => {
      out.push({ measure: `Deming slope against HIPS BC (${m.label})`, series, k, value: slope })
      out.push({ measure: 'Deming intercept (µg/m³)', series, k, value: intercept })
    }
    for (const r of rows(data, 'addis_by_k')) add(String(r.preprocessing), num(r.k) ?? 0, m.slope(num(r['Addis slope'])), num(r['Addis intercept']))
    const all = rows(data, 'addis_filters').filter((r) => num(r.fabs) != null && num(r.ec_published) != null)
    const af = all.filter((r) => isCounted(String(r.filter_id)))
    const pub = af.length ? deming(af.map((r) => num(r.fabs)! / m.mac), af.map((r) => num(r.ec_published)!), demingLambda(m.mac, lambda10)) : null
    const kMax = Math.max(0, ...out.map((r) => num(r.k) ?? 0))
    if (pub) for (let k = 1; k <= kMax; k++) add('SPARTAN published', k, pub.slope, pub.intercept)
    return { rows: out, pub, n: af.length, nOut: all.length - af.length }
  }, [data, m, lambda10, isCounted])

  if (!data) return <MlWaiting error={error} what="the decision workbook export" />
  const run = settings.run ? `results run ${settings.run}` : 'the executed notebook'
  const kText = [...kSel].map(([p, k]) => `${p} ${k}`).join(', ')

  return <>
    <PageIntro id="ml-decision-intro" title="Decision workbook — inside IMPROVE the reference model fails where Addis is; a local calibration helps there, but not at Addis" data={data} notebooks={NOTEBOOKS}>
      <FixedSetNote subject="this page" what="the notebook's tables on its own filter sets (the §6 network charts, rebuilt per filter, follow the switch)" />
      <p>The workbook tests four answers to "what do we do where the IMPROVE calibration does not transfer?": A, one better model; B, specialist calibrations and a rule to pick one; C, a flag on filters the calibration cannot be trusted on; D, new reference filters. Only IMPROVE has an independent EC truth (TOR), so A, B and C are judged inside IMPROVE with whole sites held out; at SPARTAN, HIPS BC = Fabs ÷ MAC is only a cross-check. This page reads {run}. Everything scored against TOR is the notebook's and does not involve the MAC; whatever is against HIPS is refitted or re-expressed at {m.label}{m.changed ? ' (the board\'s own numbers stay at MAC 10)' : ''}.</p>
      <p>Not repeated here: §5 and §6b, what the flag sees (its ablation ladder, coverage curves, missed failures and the loading × distance map) are on <a href="#tab=experimental&xp=flag">Experimental → What the flag sees</a>; §7b, the local calibration applied to Addis and the network, is on <a href="#tab=experimental&xp=analogs">Experimental → Loading-matched analogs</a>. §1 (where each SPARTAN site's loading and OC/BC sit in IMPROVE) saved no table.</p>
    </PageIntro>

    {sweep.length > 0
      ? <Figure kind="sweep" id="ml-decision-cv" data={sweep} x="k" y="rmse" series="preprocessing" facet="measure" colors={PREP_COLORS} chosen={chosen}
        xLabel="PLS components k" yLabel="Out-of-site RMSE (µg/filter)" height={240}
        title={`Out-of-site error flattens after a few components; the lowest on the hard groups without costing the rest more than 5 % is ${optionA ? `${optionA.b.preprocessing} k = ${optionA.b.k}` : '—'} (§2, §7)`}
        subtitle={`Every IMPROVE filter predicted by a model fitted without its site (5 site-grouped folds), against TOR EC per filter. Ringed: the one-standard-error choice of k on those folds (${kText}). The hard groups are the union of §3's five; the rest are the other ${'≈'}10,000 filters. The notebook's §2 curve is the mean of the five folds' RMSEs, which it did not save; this is the pooled RMSE over all out-of-site predictions, which at the chosen k is the RMSE it printed (${rows(data, 'cv_choice').map((r) => `${r.preprocessing} ${f2(r.rmse)}`).join(', ')}).`}
        provenance={provenance(data, ['tradeoff', 'cv_choice'])} />
      : <Figure kind="table" id="ml-decision-cv" data={rows(data, 'cv_choice')}
        columns={[{ key: 'preprocessing', label: 'Preprocessing' }, { key: 'k', label: 'k (one-SE rule)', format: 'int' }, { key: 'rmse', label: 'Out-of-site RMSE (µg/filter)', format: 'f2' },
          { key: 'bias', label: 'Bias (µg/filter)', format: 'f2' }, { key: 'r2', label: 'R²', format: 'f2' }]}
        title="Out-of-site predictions for every IMPROVE filter: the CV choice of k (§2)"
        subtitle="As printed. The RMSE curve over k needs the results run (section7_tradeoff.csv)." provenance={provenance(data, ['cv_choice'])} />}

    <Figure kind="intervals" id="ml-decision-groups" data={groups} row="group" series="preprocessing" colors={PREP_COLORS}
      panels={[{ value: 'biasPct', label: 'Bias, % of mean TOR EC', reference: 0 }, { value: 'rmsePct', label: 'RMSE, % of mean TOR EC', reference: 0 }]}
      aggregate={{ stat: 'median', interval: 'none' }} labelWidth={170}
      title={`The reference model under-predicts the filters most like Addis: heavy loading by ${pct0(-(num(g('spline', 'Heavy loading')?.biasPct) ?? NaN))} %, low OC/BC by ${pct0(-(num(g('spline', 'Low OC/BC')?.biasPct) ?? NaN))} % (§3)`}
      subtitle={`Out-of-site predictions at each preprocessing's CV choice of k. Each hard group is the most extreme 5 % of IMPROVE on a quantity a SPARTAN filter also has (HIPS loading per filter, TOR OC ÷ HIPS BC, non-soil K, the six urban sites, correlation with the median Addis spectrum), never on TOR EC itself. "None of these" is the other 10,027 filters: their positive bias and the heavy group's negative one are partly the usual shrinkage toward the mean; low OC/BC filters are not heavy and are still under-predicted, which points at composition.${hasAllPreps ? '' : ' Spline only: the notebook displayed the other preprocessings in its chart, not its table.'}`}
      provenance={provenance(data, ['groups'])} />

    <Figure kind="intervals" id="ml-decision-group-slopes" data={groups} row="group" series="preprocessing" colors={PREP_COLORS}
      panels={[{ value: 'slope vs TOR', label: 'Slope of FTIR EC on TOR EC', reference: 1 }, { value: 'slopeHips', label: `Slope of FTIR EC on IMPROVE HIPS BC (${m.label})`, reference: 1 }]}
      aggregate={{ stat: 'median', interval: 'none' }} labelWidth={170}
      title={`Against TOR every hard group's slope is well below 1; against HIPS at ${m.label} the spline model's slope is steeper for ${steeper.length ? steeper.join(', ') : 'no group'} (§3)`}
      subtitle={`OLS of the out-of-site prediction on the reference, both in µg per filter. The right panel is the only comparison SPARTAN can make: a slope against HIPS well above the one against TOR is HIPS reading low on those filters (the reference gap seen before, MAC 7.8–8.2 on smoke filters), not the calibration doing well. HIPS µg = Fabs ÷ MAC × volume, so that slope is the notebook's × ${m.mac}/10 (exact; intercepts unchanged).`}
      provenance={provenance(data, ['groups'])} />

    <Figure kind="table" id="ml-decision-groups-table" data={groups} choose={hasAllPreps ? [{ column: 'preprocessing', label: 'Preprocessing', default: 'spline' }] : undefined}
      columns={[{ key: 'group', label: 'Group' }, { key: 'n', label: 'Filters', format: 'int' }, { key: 'mean TOR µg', label: 'Mean TOR EC (µg)', format: 'f2' },
        { key: 'bias', label: 'Bias (µg)', format: 'f2' }, { key: 'rmse', label: 'RMSE (µg)', format: 'f2' }, { key: 'slope vs TOR', label: 'Slope vs TOR', format: 'f2' },
        { key: 'intercept vs TOR', label: 'Intercept vs TOR (µg)', format: 'f2' }, { key: 'slopeHips', label: `Slope vs HIPS (${m.label})`, format: 'f2' }, { key: 'intercept vs HIPS', label: 'Intercept vs HIPS (µg)', format: 'f2' }]}
      title="The failure groups in numbers (§3)" provenance={provenance(data, ['groups'])} />

    <Figure kind="intervals" id="ml-decision-specialists" data={spec} row="group" series="rule" block="preprocessing" colors={RULE_COLORS}
      panels={[{ value: 'change', lo: 'change lo', hi: 'change hi', label: 'RMSE vs the full set on its target, % (10th–90th)', reference: 0 },
        { value: 'outside', label: 'RMSE off its target, % change', reference: 0 }, { value: 'wins', label: 'Share of splits won', reference: 0.5 }]}
      aggregate={{ stat: 'median', interval: 'none' }} labelWidth={170}
      title={`Specialists barely help on spline; on raw spectra a per-filter local calibration wins everywhere, ordinary filters included (${pct0(num(localRaw('None of these')?.change))} %) (§4)`}
      subtitle={`Option B, inside IMPROVE: in each of 10 splits 30 % of sites are held out and each hard group (thresholds from the training sites) is predicted by the full training set or by a specialist of 500 training filters: same kind (the most extreme on the group's own variable), target analogs (spectra closest to the group's median), or local (each filter's own 500 nearest spectra, on a sample of 40). Left of 0, the specialist wins. A same-kind or analog model is much worse on the held-out filters it was not built for (middle panel), so its rule has to be right; the local model needs no rule, and it also beats the full set on ordinary filters, so it is really a candidate for option A.`}
      provenance={provenance(data, ['specialists'])} />

    <FlagOnSpartan data={data} />

    {addisK.rows.length > 0
      ? <Figure kind="sweep" id="ml-decision-addis-k" data={addisK.rows} x="k" y="value" series="series" facet="measure" colors={PREP_COLORS} chosen={chosen}
        xLabel="PLS components k" yLabel="" height={240}
        title="IMPROVE cannot choose k for Addis: the Addis intercept keeps moving with k while IMPROVE's error barely does (§7)"
        subtitle={`Option A's Addis readout: each model fitted on all of IMPROVE, applied to the ${addisK.n || 233} Addis filters and fitted against HIPS BC (Deming, the workbook's Addis error ratio, λ = ${f2(demingLambda(m.mac, lambda10))} at ${m.label}). Ringed: the CV choice of k. Grey: SPARTAN's published EC, refitted here at ${m.label}${addisK.pub ? ` (${f2(addisK.pub.slope)}·x ${addisK.pub.intercept < 0 ? '−' : '+'} ${f2(Math.abs(addisK.pub.intercept))})` : ''}. Slopes are the notebook's × ${m.mac}/10, exact with λ ∝ MAC²; intercepts do not move. ${optionA ? `The board's test: the best (preprocessing, k) on the hard groups that is no worse than +5 % on the rest is ${optionA.b.preprocessing} k = ${optionA.b.k} (hard ${f2(optionA.b['RMSE hard groups'])} vs ${f2(optionA.c['RMSE hard groups'])} µg for spline k = ${optionA.c.k}), and it moves the Addis intercept from ${f2(rows(data, 'addis_by_k').find((r) => r.preprocessing === 'spline' && r.k === optionA.c.k)?.['Addis intercept'])} to ${f2(rows(data, 'addis_by_k').find((r) => r.preprocessing === optionA.b.preprocessing && r.k === optionA.b.k)?.['Addis intercept'])} µg/m³.` : ''}`}
        provenance={provenance(data, ['addis_by_k', 'addis_filters'])}
        aside={addisK.nOut > 0 && <p className="chart-note">SPARTAN's published line leaves out {addisK.nOut} excluded filter{addisK.nOut === 1 ? '' : 's'} (Filters → excluded); the notebook's k curves are its saved fits on all {addisK.n + addisK.nOut}.</p>} />
      : <ChartFrame id="ml-decision-addis-k" title="Option A at Addis: slope and intercept against HIPS over k (§7)" exportable={false}>
        <Note>Needs the results run (section7_addis_by_k.csv); see the list at the top of the page.</Note>
      </ChartFrame>}

    <Figure kind="table" id="ml-decision-board" data={rows(data, 'board')}
      columns={[{ key: 'option', label: 'Option' }, { key: 'test', label: 'Test' }, { key: 'result', label: 'Result' }, { key: 'suggests', label: 'Suggests' }]}
      title="The decision board: B's local rule and A' are the same model, and it does not carry over to Addis (§8)"
      subtitle={`Filled by the notebook from this run, as written (its Addis numbers are against HIPS BC at MAC 10). B's best rule is the local calibration on raw spectra in every group; A' applies it to Addis, where it moves the Deming line further from HIPS than SPARTAN's published EC (Experimental → Loading-matched analogs). The flag adds information over any single quantity on IMPROVE, but at Addis it is high for everything. D: the Addis TOR filters, with pass criteria written before they arrive (|bias| ≤ 20 % of TOR, slope 0.8–1.2, |intercept| ≤ 0.5 µg/m³).`}
      provenance={provenance(data, ['board'])} />
  </>
}
