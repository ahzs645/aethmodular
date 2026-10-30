import { useMemo } from 'react'
import { Figure } from '@/components/Figure'
import { f2 } from '@/lib/format'
import { rmseAtMac, solveHipsMoments, useMac, type HipsMoments } from '@/lib/mac'
import { num, type Row } from '@/lib/table'
import type { SectionPageProps } from '../../types'
import { useCalibRuns } from '../../research/pages/spectralCommon'
import { MlWaiting, PageIntro, provenance, rows, useMl } from '../common'

const NOTEBOOKS = 'research/ftir_ec_phase3/ftir_42_target_definition_experiment, ftir_48_pyrolysis_split_target, ftir_41_adama_three_method_reconciliation, ftir_44_adama_through_locked_calibrations; ftir_40_quartz_tor_campaign_design as the design companion'

const TARGET: Record<string, string> = {
  EC_TOR: 'EC, TOR (the locked target)', EC_TOT: 'EC, TOT', OC_TOR: 'OC, TOR', TC: 'TC (OC + EC)',
  DELTA: 'Δ = OP_TOT − OP_TOR', OP_TOR: 'OP, TOR', OP_TOT: 'OP, TOT', DELTA_RATIO: 'Δ / EC_TOR',
}
const COHORT_COLORS = { 'Lowest-OC/EC (800) + AIRSpec': '#2C6E9E', 'Full pool (no selection) + AIRSpec': '#e15759' }
const COHORT_SHORT: Record<string, string> = { 'Lowest-OC/EC (800) + AIRSpec': 'Lowest-OC/EC 800', 'Full pool (no selection) + AIRSpec': 'Full pool' }
const ADAMA_COLORS = {
  'EC, TOR (quartz)': '#2C3E50', 'EC, TOT (quartz)': '#7f7f7f', 'deployed FTIR EC': '#59a14f', 'locked 800 + AIRSpec': '#2C6E9E',
  'sweep winner 440, k = 8': '#b07aa1', 'EC_TOT-target model': '#f28e2b', 'HIPS Fabs/MAC': '#e15759',
}

const dateLabel = (r: Row) => `${r.date}${r.flag ? ` · ${r.flag}` : ''}`

/**
 * ftir_42 and ftir_48 (target definitions) and ftir_41, ftir_44 (the five Adama TOR filters). Held-out scores are
 * against IMPROVE TOR/TOT and do not depend on the MAC. The Addis lines are against HIPS Fabs/10: their slopes scale by
 * MAC/10, and their RMSE moves through the fixed 190-filter cohort's HIPS mean and variance, which
 * calibration_runs.json gives exactly (its fixed Addis filters are the notebooks'), else solved from the saved fits.
 */
export function Targets(_props: SectionPageProps) {
  const { data, error } = useMl('targets')
  const { runs } = useCalibRuns()
  const m = useMac()

  // HIPS BC (Fabs/10) mean and variance over the fixed Addis cohort, from the explorer's readouts
  const exact = useMemo((): Pick<HipsMoments, 'mean' | 'variance'> | null => {
    const run = runs?.runs.find((r) => r.preset === 'ocec800_airspec' && r.target === 'addis')
    if (!run) return null
    const x = run.eval.ref.filter((_, i) => run.eval.fixed[i]).map((f) => f / 10)
    if (x.length !== 190) return null
    const mean = x.reduce((s, v) => s + v, 0) / x.length
    return { mean, variance: x.reduce((s, v) => s + (v - mean) ** 2, 0) / x.length }
  }, [runs])

  if (!data) return <MlWaiting error={error} what="the target-definition export" />

  const skill = [
    ...rows(data, 'target_definition_summary').filter((r) => r.target === 'OC_TOR' || r.target === 'TC').map((r): Row => ({ ...r, block: 'ftir_42 · what is predicted' })),
    ...rows(data, 'split_target_summary').map((r): Row => ({ ...r, block: r.target === 'EC_TOR' || r.target === 'EC_TOT' ? 'ftir_42 · what is predicted' : 'ftir_48 · the pyrolysis split' })),
  ].map((r): Row => ({ ...r, target: TARGET[String(r.target)] ?? String(r.target) }))
    .sort((p, q) => String(p.block).localeCompare(String(q.block)))

  const adamaDelta = rows(data, 'adama_delta_check').map((r) => num(r.DELTA_over_ECTR_meas)).filter((v): v is number => v != null).sort((p, q) => p - q)
  const deltaDist = [
    ...rows(data, 'delta_ratio_distribution').map((r): Row => ({ set: `${r.set} (n = ${num(r.n)?.toLocaleString()})`, mid: r.median, lo: r.q25, hi: r.q75, neg: num(r.share_negative) != null ? 100 * num(r.share_negative)! : null })),
    ...(adamaDelta.length ? [{ set: `Adama quartz, measured (n = ${adamaDelta.length}; full range)`, mid: adamaDelta[adamaDelta.length >> 1], lo: adamaDelta[0], hi: adamaDelta[adamaDelta.length - 1], neg: 0 }] : []),
  ]

  const routes = rows(data, 'ec_tot_routes')
  const rescale = rows(data, 'rescale_factor')

  // Addis by route, against HIPS Fabs/MAC
  const addisRows = rows(data, 'addis_ec_tot_routes')
  const moments = exact ?? solveHipsMoments(addisRows.map((r) => ({ slope: num(r.slope), intercept: num(r.intercept), r2: num(r.R2), rmse: num(r.RMSE) })))
  const addis = addisRows.map((r): Row => ({
    ...r, cohort: String(r.cohort),
    slope: m.slope(num(r.slope)),
    RMSE: m.changed ? rmseAtMac({ slope: num(r.slope), intercept: num(r.intercept), rmse: num(r.RMSE) }, moments, m.mac) : num(r.RMSE),
  }))

  // Adama: every EC reading per date (µg/m³); HIPS as Fabs/MAC
  const wide = rows(data, 'adama_models')
  const ADAMA: [string, string][] = [
    ['ECTR', 'EC, TOR (quartz)'], ['ECTT', 'EC, TOT (quartz)'], ['EC_ftir', 'deployed FTIR EC'],
    ['Locked: lowest-OC/EC 800 + AIRSpec (k=5)', 'locked 800 + AIRSpec'], ['Sweep winner: lowest-OC/EC 440 + AIRSpec (k=8)', 'sweep winner 440, k = 8'],
    ['EC_TOT-target, 800-intersection', 'EC_TOT-target model'],
  ]
  const adama = wide.flatMap((r): Row[] => [
    ...ADAMA.map(([col, label]) => ({ date: dateLabel(r), method: label, value: num(r[col]) })),
    { date: dateLabel(r), method: 'HIPS Fabs/MAC', value: m.fromFabs(num(r.Fabs)) },
  ])
  const carbon = wide.flatMap((r): Row[] => [
    { date: dateLabel(r), method: 'OC, TOR (quartz)', value: num(r.OCTR) }, { date: dateLabel(r), method: 'OC-target model', value: num(r['OC-target, 800-intersection']) },
    { date: dateLabel(r), method: 'deployed FTIR OC', value: num(r.OC_ftir) },
    { date: dateLabel(r), method: 'TC, thermal', value: num(r.TC) }, { date: dateLabel(r), method: 'TC-target model', value: num(r['TC-target, 800-intersection']) },
  ])

  const ref = rows(data, 'adama_reference').map((r): Row => ({ ...r, date: dateLabel({ ...r, flag: rows(data, 'adama_pairs').find((p) => p.date === r.date)?.flag ?? null }),
    op_ec: num(r.OPTR) != null && num(r.ECTR) ? num(r.OPTR)! / num(r.ECTR)! : null }))
  const budget = rows(data, 'adama_carbon_budget').flatMap((r): Row[] => [
    ['tc_ratio', 'TC: FTIR OC + EC / thermal'], ['oc_ratio_tor', 'OC: FTIR / TOR'], ['ec_ratio_tor', 'EC: FTIR / TOR'], ['ec_ratio_tot', 'EC: FTIR / TOT'],
  ].map(([c, label]) => ({ date: dateLabel(r), ratio: label, value: num(r[c]) })))
  const implied = rows(data, 'adama_implied_mac').flatMap((r): Row[] => [
    ['implied_MAC_vs_ECTOR', 'vs EC, TOR'], ['implied_MAC_vs_ECTOT', 'vs EC, TOT'], ['implied_MAC_vs_FTIR_EC', 'vs deployed FTIR EC'],
  ].map(([c, label]) => ({ date: dateLabel(r), definition: label, value: num(r[c]) })))
  const delta48 = rows(data, 'adama_delta_check').flatMap((r): Row[] => [
    ['DELTA_meas', 'measured Δ (OPTT − OPTR)'], ['DELTA_hat_800', 'Δ model, Lowest-OC/EC 800'], ['DELTA_hat_full', 'Δ model, full pool'],
  ].map(([c, label]) => ({ date: dateLabel(r), series: label, value: num(r[c]) })))
  const ratios = rows(data, 'adama_model_ratios')
  const lockedTor = num(ratios.find((r) => r.model === 'Locked 800 + AIRSpec' && r.reference === 'TOR')?.median_ratio_unflagged)
  // ftir_44 [11]: unflagged medians of the OC and TC ratios
  const unflagged = rows(data, 'adama_oc_tc').filter((r) => r.flag == null)
  const med = (c: string) => {
    const v = unflagged.map((r) => num(r[c])).filter((x): x is number => x != null).sort((p, q) => p - q)
    return v.length ? f2(v.length % 2 ? v[v.length >> 1] : (v[v.length / 2 - 1] + v[v.length / 2]) / 2) : '—'
  }
  const poolRatio = rows(data, 'pool_ratio')[0]

  return <>
    <PageIntro id="ml-targets-intro" title="What counts as EC — total carbon is easy, the OC/EC split is not, and Adama reads nothing like Addis" data={data} notebooks={NOTEBOOKS}>
      <p>The project calibrates FTIR against IMPROVE TOR EC. These notebooks keep the spectra, cohort, locked protocol and held-out-site folds fixed and change only the target: EC under the transmittance convention (TOT), OC, total carbon, and the pyrolysis correction Δ = OP_TOT − OP_TOR that separates the two EC conventions. Held-out scores are against IMPROVE and do not depend on the MAC; the Addis lines are against HIPS Fabs/MAC and follow the MAC set under Filters → units (now {m.mac}). Then the five Adama quartz/PTFE pairs, the only same-corridor place where thermal OC/EC under both conventions, FTIR carbon and HIPS sit on one table.</p>
      <p>The Adama TOR data also feed <a href="#tab=experimental&xp=tor">Experimental → Adama TOR study</a> and <a href="#tab=baseline">Baselines → external pilot</a> (AIRSpec/VIBES EC against ECTR/ECTT); their data folders are git-ignored, so those pages may be empty in this checkout. Everything below is read from the executed notebooks.</p>
    </PageIntro>

    <Figure kind="intervals" id="ml-targets-skill" data={skill} row="target" series="cohort" block="block" colors={COHORT_COLORS}
      panels={[{ value: 'heldout_R2', label: 'Held-out-site R²', reference: 0 }, { value: 'heldout_pct_RMSE', label: 'Held-out RMSE (% of the target mean)' }, { value: 'heldout_slope', label: 'Held-out slope', reference: 1 }]}
      aggregate={{ stat: 'median', interval: 'none' }} labelWidth={200}
      title="Total carbon is the easiest thing FTIR predicts; EC under the transmittance convention collapses without composition selection (ftir_42, ftir_48)"
      subtitle="Same spectra, cohort, protocol and folds; only the target changes (µg/filter on held-out IMPROVE sites). On the Lowest-OC/EC cohort TC and OC beat the locked EC_TOR target; EC_TOT is harder, and on the full pool its model picks k = 2 and fails. Of the split terms, OP_TOT and Δ are learnable on the cohort, OP_TOR weakly, the ratio Δ/EC_TOR not at all; on the full pool Δ is unlearnable."
      provenance={provenance(data, ['target_definition_summary', 'split_target_summary'])} />

    <Figure kind="table" id="ml-targets-tc" data={rows(data, 'tc_direct_vs_summed')}
      columns={[{ key: 'cohort', label: 'Cohort' }, { key: 'n_test', label: 'Held-out filters', format: 'int' }, { key: 'TC_direct_R2', label: 'TC direct R²', format: 'f3' }, { key: 'TC_direct_RMSE', label: 'TC direct RMSE (µg)', format: 'f2' },
        { key: 'TC_summed_R2', label: 'OC + EC models R²', format: 'f3' }, { key: 'TC_summed_RMSE', label: 'OC + EC models RMSE (µg)', format: 'f2' },
        { key: 'summed_vs_direct_RMSE_pct', label: 'Summed vs direct RMSE (%)', format: 'f2' }, { key: 'EC_share_R2', label: 'EC/TC share R²', format: 'f3' }]}
      title="The models predict amounts, not the OC/EC split: predicted EC/TC explains under 1 % of the observed share (ftir_42)"
      subtitle="A direct TC model and the sum of separate OC and EC models agree to 3 % (cohort) and 0.03 % (full pool) in RMSE, while the predicted EC share of TC has no out-of-site skill: the calibrations track loading through covariance, not the thermal partition. The per-filter crossplots are not saved by the notebook."
      provenance={provenance(data, ['tc_direct_vs_summed'])} />

    <Figure kind="intervals" id="ml-targets-delta" data={deltaDist} row="set"
      panels={[{ value: 'mid', lo: 'lo', hi: 'hi', label: 'Δ / EC_TOR (median and IQR)', reference: 0 }, { value: 'neg', label: 'Share with Δ < 0 (%)' }]} labelWidth={400}
      title="The split correction Δ is about 46 % of EC_TOR across IMPROVE, but only 19 % at Adama (ftir_48)"
      subtitle={`Δ = OP_TOT − OP_TOR = EC_TOR − EC_TOT. Over the usable IMPROVE mirror EC_TOT/EC_TOR has median ${f2(poolRatio?.median)} (IQR ${f2(poolRatio?.q25)}–${f2(poolRatio?.q75)}, ftir_42): the convention is no relabelling. Adama's five quartz filters (bottom row: median and full range) sit far below every IMPROVE set.`}
      provenance={provenance(data, ['delta_ratio_distribution', 'pool_ratio', 'adama_delta_check'])} />

    <Figure kind="intervals" id="ml-targets-routes" data={routes.map((r): Row => ({ ...r, cohort: String(r.cohort) }))} row="route" series="cohort" colors={COHORT_COLORS}
      panels={[{ value: 'R2', label: 'Held-out R² against EC_TOT', reference: 0 }, { value: 'pct_RMSE', label: 'RMSE (% of the EC_TOT mean)' }, { value: 'bias_ug', label: 'Bias (µg/filter)', reference: 0 }]}
      aggregate={{ stat: 'median', interval: 'none' }} labelWidth={300}
      title="EC_TOT is hard because of the carbon, not the correction: composing EC_TOR_hat − Δ_hat matches the direct model and rescues the full pool (ftir_48)"
      subtitle={`Four routes to EC_TOT on the same held-out filters, plus an oracle handed the true Δ. On the cohort the oracle is no better than the direct model (the error is the EC_TOR model's own, ~3 µg/filter), and subtracting a predicted Δ cancels shared error. The rescale a·EC_TOR_hat uses a fitted on training filters only (${rescale.map((r) => `${COHORT_SHORT[String(r.cohort)] ?? r.cohort} a = ${f2(r.a_ls_through_origin)}`).join(', ')}).`}
      provenance={provenance(data, ['ec_tot_routes', 'rescale_factor'])} />

    <Figure kind="table" id="ml-targets-decomposition" data={rows(data, 'composed_error_decomposition')}
      columns={[{ key: 'cohort', label: 'Cohort' }, { key: 'MSE_composed', label: 'MSE composed', format: 'f2' }, { key: 'MSE_EC_TOR_model (= oracle)', label: 'MSE EC_TOR model (= oracle)', format: 'f2' },
        { key: 'MSE_DELTA_model', label: 'MSE Δ model', format: 'f2' }, { key: 'minus_2cov', label: '− 2 cov', format: 'f2' }, { key: 'corr_errors', label: 'r(errors)', format: 'f2' },
        { key: 'MSE_direct', label: 'MSE direct EC_TOT', format: 'f2' }, { key: 'delta_share_of_composed_MSE', label: 'Δ share of composed MSE', format: 'f2' }]}
      title="Predicting Δ costs nothing on the cohort because its errors cancel the EC_TOR model's (ftir_48)"
      subtitle="MSE(EC_TOR_hat − Δ_hat) = MSE(EC_TOR_hat) + MSE(Δ_hat) − 2 cov. On the cohort the errors correlate at 0.55, so the composed MSE falls below the oracle's; on the full pool the Δ model is 36 % of the composed MSE (µg²/filter²)."
      provenance={provenance(data, ['composed_error_decomposition'])} />

    <Figure kind="heatmap" id="ml-targets-routes-sites" data={rows(data, 'ec_tot_routes_per_site').map((r): Row => ({ ...r, site: `${r.site} (n = ${r.n})` }))}
      row="site" column="route" value="RMSE_ug" ramp="sequential" format="f2" valueLabel="Held-out RMSE (µg/filter)" cell={[120, 22]}
      choose={[{ column: 'cohort', label: 'Cohort' }]}
      title="No route wins at every held-out IMPROVE site (ftir_48)"
      subtitle="Per-site RMSE against EC_TOT on the Lowest-OC/EC cohort's held-out sites (the table the notebook displayed; the full-pool cohort's is written only by a re-run). The rescaled route is most often the lowest; knowing the true Δ (the oracle) does not help at most sites."
      provenance={provenance(data, ['ec_tot_routes_per_site'])} />

    <Figure kind="intervals" id="ml-targets-addis" data={addis} row="route" series="cohort" colors={COHORT_COLORS}
      panels={[{ value: 'slope', label: `Slope vs HIPS Fabs/${m.mac}`, reference: 1 }, { value: 'intercept', label: 'Intercept (µg/m³)', reference: 0 },
        { value: 'RMSE', label: `RMSE against Fabs/${m.mac} (µg/m³)${m.changed && !moments ? ' (MAC 10)' : ''}` },
        { value: 'median_ratio_to_EC_TOR_model', label: 'Median prediction ÷ EC_TOR model', reference: 1 }]}
      aggregate={{ stat: 'median', interval: 'none' }} labelWidth={300}
      title="The TOT convention changes the Addis line beyond a rescale, and the negative offset survives every route (ftir_42, ftir_48)"
      subtitle={`The fixed 190-filter Addis cohort, OLS of each prediction on HIPS Fabs/${m.mac}. The direct EC_TOT model reads 0.76 of the EC_TOR model per filter, nothing like the IMPROVE target ratio of 0.54; the composed route lands between. Slopes follow the MAC (× MAC/10); R² and intercepts do not move${m.changed ? `; RMSE is recomputed from the cohort's HIPS mean and variance (${exact ? 'exact, from calibration_runs.json\'s same 190 filters' : moments ? 'solved from these fits' : 'unavailable, so left at MAC 10'}). ` : ''}. The notebooks' Deming slopes (fixed σx = 0.308, σy = 0.531 at MAC 10, so they do not rescale exactly) are in the export, not drawn. ftir_42's own Addis table reproduces the EC_TOR and direct EC_TOT rows exactly (the exporter checks).`}
      provenance={provenance(data, ['addis_ec_tot_routes', 'addis_by_ec_convention'], exact ? 'HIPS moments from calibration_runs.json (fixed Addis cohort).' : undefined)} />

    <Figure kind="table" id="ml-targets-adama-pairs" data={rows(data, 'adama_pairs')}
      columns={[{ key: 'date', label: 'Date' }, { key: 'FilterId_ptfe', label: 'PTFE filter' }, { key: 'FilterId_quartz', label: 'Quartz filter' }, { key: 'Volume_m3', label: 'Volume (m³)', format: 'f2' },
        { key: 'start_offset_min', label: 'Start offset (min)', format: 'f2' }, { key: 'volume_ratio', label: 'Volume ratio', format: 'f2' }, { key: 'flag', label: 'Flag' }]}
      title="The five Adama pairs: different physical filters on the same dates, two of them flagged (ftir_41)"
      subtitle="July 2024, Batch 54. A start offset of 40 minutes and a volume mismatch flag two pairs; the unflagged medians below use the other three. n = 5, one month: read every Adama number as a pilot."
      provenance={provenance(data, ['adama_pairs'])} />

    <Figure kind="intervals" id="ml-targets-adama-ref" data={ref} row="date"
      panels={[{ value: 'EC_TT_vs_TR_pct', label: 'EC_TOT vs EC_TOR (%)', reference: 0 }, { value: 'op_ec', label: 'Pyrolysed OC / EC, TOR' }]}
      aggregate={{ stat: 'median', interval: 'none' }} labelWidth={220}
      title="The reference is a split question: EC_TOT is 15–21 % below EC_TOR, and pyrolysis is half the EC (ftir_41)"
      subtitle="Thermal OC/EC on the quartz filters under both optical conventions. OC + EC is identical across conventions (to 1e-13 µg/m³), so no choice of split can close a total-carbon gap. The thermal-fraction shares (OC1–OC4, EC1–EC3) are written only by a re-run with the Drive."
      provenance={provenance(data, ['adama_reference'])} />

    <Figure kind="intervals" id="ml-targets-adama-budget" data={budget} row="date" series="ratio"
      panels={[{ value: 'value', label: 'FTIR ÷ thermal', reference: 1 }]} aggregate={{ stat: 'median', interval: 'none' }} labelWidth={220}
      title="The FTIR carbon sum is half the thermal TC, and the deficit is OC (ftir_41)"
      subtitle="Deployed FTIR OC + EC against thermal TC: 0.40–0.74. Against the transmittance convention FTIR EC is close to the thermal reference (median 0.86); OC is at 0.46. Part of that is expected media physics (uncorrected quartz adsorption inflates the denominator)."
      provenance={provenance(data, ['adama_carbon_budget'])} />

    <Figure kind="intervals" id="ml-targets-adama-mac" data={implied} row="date" series="definition"
      panels={[{ value: 'value', label: `Implied MAC = Fabs ÷ EC (m²/g); reference line: the reader's MAC ${m.mac}`, reference: m.mac }]} aggregate={{ stat: 'median', interval: 'none' }} labelWidth={220}
      title="Same-corridor absorption exceeds what any EC definition explains: implied MAC 15–25 m²/g (ftir_41)"
      subtitle="HIPS Fabs on the PTFE filter over each EC definition: far above the physical 4–13 m²/g, yet far below the ≈47 m²/g the ftir_16 Addis composition bridge implies. Absorption exceeds EC by 1.5–2×, not 4×. MAC-free."
      provenance={provenance(data, ['adama_implied_mac'])} />

    <Figure kind="intervals" id="ml-targets-adama-models" data={adama} row="date" series="method" colors={ADAMA_COLORS}
      panels={[{ value: 'value', label: `EC (µg/m³); HIPS as Fabs/${m.mac}`, reference: 0 }]} aggregate={{ stat: 'median', interval: 'none' }} labelWidth={220}
      title={`The composition-selected calibrations over-read Adama's thermal EC${lockedTor ? ` (${f2(lockedTor)}× EC_TOR)` : ''} and land with HIPS, not with TOR (ftir_44)`}
      subtitle={`Each Adama PTFE spectrum through the locked phase-3 calibrations. Adama's OC/EC ≈ 6 is far outside the cohort's training domain (OC/EC ≤ 2.27): a calibration built to match Addis's HIPS-implied EC behaves like HIPS/MAC on ordinary aerosol. The deployed SPARTAN calibration reads below TOR; the EC_TOT-target model is the only phase-3 model near the thermal reference. HIPS follows the MAC (Fabs/${m.mac}).`}
      provenance={provenance(data, ['adama_models'])} />

    <Figure kind="intervals" id="ml-targets-adama-ratios" data={ratios} row="model" series="reference"
      panels={[{ value: 'median_ratio_unflagged', label: 'Median model ÷ thermal EC, 3 unflagged pairs', reference: 1 }, { value: 'median_ratio_all5', label: 'All 5 pairs', reference: 1 }]}
      aggregate={{ stat: 'median', interval: 'none' }} labelWidth={220}
      title="Only the EC_TOT-target model reads near unity at Adama (ftir_44)"
      subtitle="Ratios against the quartz EC under each convention. MAC-free."
      provenance={provenance(data, ['adama_model_ratios'])} />

    <Figure kind="intervals" id="ml-targets-adama-carbon" data={[...carbon.filter((r) => /OC/.test(String(r.method))), ...carbon.filter((r) => /TC/.test(String(r.method)))]} row="date" series="method"
      panels={[{ value: 'value', label: 'Carbon (µg/m³)', reference: 0 }]} aggregate={{ stat: 'median', interval: 'none' }} labelWidth={220}
      title="OC and TC stay low whichever model reads them: the Adama carbon deficit is in the spectra or the media (ftir_44)"
      subtitle={`The OC-target and TC-target phase-3 models against thermal OC and TC, with the deployed FTIR OC. Unflagged medians: OC-target model ${med('OC_model_over_OCTR')}× OC_TOR, TC-target model ${med('TC_model_over_TC')}× thermal TC (deployed ${med('OC_deployed_over_OCTR')}× and ${med('TC_deployed_over_TC')}×).`}
      provenance={provenance(data, ['adama_models', 'adama_oc_tc'])} />

    <Figure kind="intervals" id="ml-targets-adama-delta" data={delta48} row="date" series="series"
      panels={[{ value: 'value', label: 'Δ = OP_TOT − OP_TOR (µg/m³)', reference: 0 }]} aggregate={{ stat: 'median', interval: 'none' }} labelWidth={220}
      title="Nothing tuned, the cohort's Δ model reads 0.4× Adama's measured Δ and the full-pool model 1.2× (ftir_48)"
      subtitle="The split model applied to the five Adama spectra. Adama's Δ is 0.15–0.21 of EC_TOR against the cohort's 0.45, so the composed EC_TOT overshoots ECTT (1.49× unflagged) more than the direct EC_TOT model does (1.27×)."
      provenance={provenance(data, ['adama_delta_check'])} />

    <Figure kind="table" id="ml-targets-campaign" data={rows(data, 'campaign_allocation').map((r): Row => ({ ...r, ...rows(data, 'campaign_seasons').find((s) => s.season === r.season) }))}
      columns={[{ key: 'season', label: 'Season' }, { key: 'months', label: 'Months' }, { key: 'primary_pairs', label: 'Paired days', format: 'int' }, { key: 'field_blanks', label: 'Field blanks', format: 'int' },
        { key: 'collocated_duplicates', label: 'Duplicates', format: 'int' }, { key: 'total_filters', label: 'Filters', format: 'int' }, { key: 'historical_filters', label: 'Historical Addis filters', format: 'int' },
        { key: 'Fabs_median', label: 'Fabs median (Mm⁻¹)', format: 'f2' }, { key: 'expected_MAC6_minus_MAC10_ugm3', label: 'MAC 6 − MAC 10 EC gap (µg/m³)', format: 'f2' }, { key: 'paired_sigma_ugm3', label: 'Paired σ (µg/m³)', format: 'f2' }]}
      title="The quartz campaign that would settle it: 36 paired days blocked over three seasons (ftir_40)"
      subtitle="The design companion (no model fitted): a MAC 6 vs MAC 10 EC gap of 2.9–3.7 µg/m³ by season against a paired σ of 0.31 µg/m³ (HIPS 0.29, and the Adama ECTR − ECTT spread 0.10) makes separation easy; seasonal coverage is the binding constraint. Plus six seasonal field blanks and six collocated duplicates. Research → MAC 6 vs 10 has the ftir_16 power curve."
      provenance={provenance(data, ['campaign_allocation', 'campaign_seasons'])} />
  </>
}
