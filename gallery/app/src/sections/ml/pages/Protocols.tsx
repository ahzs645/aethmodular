import { useMemo } from 'react'
import { Figure } from '@/components/Figure'
import { rmseAtMac, useMac } from '@/lib/mac'
import { num, type Row } from '@/lib/table'
import { useCalibRuns } from '../../research/pages/spectralCommon'
import type { SectionPageProps } from '../../types'
import { MlWaiting, PageIntro, provenance, rows, useMl } from '../common'
import { addisMoments, PROTOCOL_COLOR, protocolOf, ruleRows, seasonFabs, useCalib } from './components/calib'

const SHORT: Record<string, string> = { site_heldout: 'site-held-out', app: 'Calibration app' }
const SOURCE_COLOR: Record<string, string> = { 'ftir_22 · by protocol': '#4e79a7', 'ftir_15 · locked models': '#59a14f' }
const DRIVER_COLOR: Record<string, string> = { 'D²': '#e15759', Q: '#f28e2b', Fabs: '#76b7b2' }
const MODEL_COLOR: Record<string, string> = { 'spline baseline (AIRSpec df1=6), k = 5': '#4e79a7', 'raw, k = 6': '#e15759' }
/** ftir_15's model labels ("AIRSpec df1=6 (k=5)") and ftir_22's spectra labels to one name */
const spectraName = (s: unknown) => (String(s).startsWith('AIRSpec') ? 'spline baseline (AIRSpec df1=6)' : 'raw')
const kOf = (label: unknown) => Number(/k=(\d+)/.exec(String(label))?.[1] ?? NaN)

/**
 * Protocols end to end, transfer and bootstrap uncertainty (ftir_21, ftir_22, ftir_15, ftir_11, ftir_13, ftir_17,
 * ftir_18; ftir_31 §1.1/1.2/1.6/1.7/2.2/2.6 redraw ftir_21/22). The notebooks' tables come from
 * gallery/data/ml_pages/protocols.py, at MAC 10 on the fixed 190 Addis filters (ftir_18: the 239 pairs). Slopes
 * against HIPS follow the MAC set under Filters → units (slope × MAC/10); RMSE follows it through the evaluation
 * set's HIPS BC mean and variance, which calibration_runs.json's per-filter Fabs give exactly (lib/mac rmseAtMac);
 * intercepts do not move. A residual correlation or spread needs the per-filter residuals and stays at MAC 10.
 */
export function Protocols(_props: SectionPageProps) {
  const { data, error } = useMl('protocols')
  const { calib } = useCalib()
  const { runs } = useCalibRuns()
  const m = useMac()
  const moments = useMemo(() => addisMoments(runs, m.exportMac), [runs, m.exportMac])
  const seasons = useMemo(() => seasonFabs(runs), [runs])
  // ExternalFilterId "ETAD-0017-1" → calibration_runs id "ETAD-0017", which carries the fixed-cohort flag
  const fixedIds = useMemo(() => {
    const run = runs?.runs.find((r) => r.target === 'addis')
    return run ? new Set(run.eval.id.filter((_, i) => run.eval.fixed[i]).map(String)) : null
  }, [runs])

  const rmseAt = (slope: unknown, intercept: unknown, rmse: unknown, set: 'fixed' | 'all' = 'fixed') =>
    !m.changed ? num(rmse as number) : rmseAtMac({ slope: num(slope as number), intercept: num(intercept as number), rmse: num(rmse as number) }, moments[set], m.mac, m.exportMac)
  const rmseLabel = (what: string) => `${what} RMSE (µg/m³${m.changed ? `, ${m.label}` : ''})`
  const slopeLabel = (what = 'Addis slope') => `${what} vs Fabs/${m.mac}`

  // explorer: every cohort × spectra under the three protocols, at the rule k (calibration.json grid)
  const strip = useMemo(() => (calib ? ruleRows(calib) : []), [calib])
  const stripAt = strip.map((r): Row => ({ ...r, slope: m.slope(num(r.slope10), 10), rmseAt: rmseAt(r.slope10, r.intercept, r.rmse) }))

  if (!data) return <MlWaiting error={error} what="the protocol tables" />

  // ftir_21: the headline (fixed 190, MAC 10) joined to the TOR test of the same calibrations
  const tor = new Map(rows(data, 'tor_test').map((r) => [`${r.cohort}|${r.mode}`, r]))
  const headline = rows(data, 'headline').map((r): Row => {
    const t = tor.get(`${r.cohort}|${r.mode}`)
    return { cohort: r.cohort, protocol: protocolOf(r.mode), k: r.k, intercept: r.intercept, slope: m.slope(num(r.slope), 10), RMSE: rmseAt(r.slope, r.intercept, r.RMSE),
      floor: t?.pct_rmsecv_floor ?? null, tor_r2: t?.heldout_TOR_R2 ?? null }
  })

  // site-cluster bootstrap intervals: ftir_22 by protocol next to ftir_15's locked models (the site-held-out ones)
  const boot = [
    ...rows(data, 'boot_modes').map((r): Row => ({ row: `${spectraName(r.spectra)} · ${SHORT[String(r.mode)] ?? r.mode} (k = ${r.k})`, source: 'ftir_22 · by protocol', mid: r.median, lo: r.ci_low, hi: r.ci_high })),
    ...rows(data, 'boot_locked').filter((r) => r.metric === 'intercept').map((r): Row => ({ row: `${spectraName(r.model)} · site-held-out (k = ${kOf(r.model)})`, source: 'ftir_15 · locked models', mid: r.boot_median, lo: r['ci_2.5'], hi: r['ci_97.5'] })),
  ]
  const locked = [...new Set(rows(data, 'boot_locked').map((r) => String(r.model)))].map((model): Row => {
    const at = (metric: string) => rows(data, 'boot_locked').find((r) => r.model === model && r.metric === metric) ?? {}
    const i = at('intercept'), s = at('slope'), e = at('RMSE')
    return { model: `${spectraName(model)}, k = ${kOf(model)}`, intercept: i.boot_median ?? null, intercept_lo: i['ci_2.5'] ?? null, intercept_hi: i['ci_97.5'] ?? null,
      slope: m.slope(num(s.boot_median as number), 10), slope_lo: m.slope(num(s['ci_2.5'] as number), 10), slope_hi: m.slope(num(s['ci_97.5'] as number), 10),
      rmse: e.boot_median ?? null, rmse_lo: e['ci_2.5'] ?? null, rmse_hi: e['ci_97.5'] ?? null }
  })

  // residual drivers: ftir_22 (D² under each protocol) and ftir_15 (D², Q and Fabs for the locked models)
  const drivers = [
    ...rows(data, 'drivers_modes').map((r): Row => ({ block: 'ftir_22 · by protocol', row: `${spectraName(r.spectra)} · ${SHORT[String(r.mode)] ?? r.mode} (k = ${r.k})`, driver: 'D²',
      pearson: r.pearson_r_residual_vs_D2, spearman: r.spearman_r_residual_vs_D2 })),
    ...rows(data, 'drivers_locked').map((r): Row => ({ block: 'ftir_15 · locked models', row: `${spectraName(r.model)} · site-held-out (k = ${kOf(r.model)})`,
      driver: r.driver === 'D2' ? 'D²' : String(r.driver), pearson: r.pearson_r_vs_residual, spearman: r.spearman_r_vs_residual })),
  ]
  // a season's mean residual (prediction − Fabs/MAC) moves by that season's mean Fabs × (1/10 − 1/MAC): exact
  const season = rows(data, 'season_residuals').map((r): Row => {
    const f = seasons.get(String(r.season))
    const mean = num(r.mean_residual)
    return { season: r.season, model: `${spectraName(r.model)}, k = ${kOf(r.model)}`, n: r.n, median_D2: r.median_D2,
      mean_residual: !m.changed ? mean : mean != null && f != null ? mean + f / m.exportMac - f / m.mac : null }
  })

  const sweep = rows(data, 'size_sweep').map((r): Row => ({ size: `N = ${r.cohort_size}`, protocol: protocolOf(r.mode), k: r.k, intercept: r.intercept, slope: m.slope(num(r.slope), 10),
    RMSE: rmseAt(r.slope, r.intercept, r.RMSE), tor_r2: r.heldout_TOR_R2 }))
  const order = rows(data, 'row_order').map((r): Row => ({ ...r, protocol: protocolOf(r.mode) }))

  // every training-set choice scored on the fixed 190 Addis filters at MAC 10
  const fixed190 = (block: string, model: unknown, slope: unknown, intercept: unknown, r2: unknown, rmse: unknown): Row =>
    ({ block, model: String(model), intercept: num(intercept as number), slope: m.slope(num(slope as number), 10), R2: num(r2 as number), RMSE: rmseAt(slope, intercept, rmse) })
  const training = [
    ...rows(data, 'sizes_addis').map((r) => fixed190('ftir_11 · OC/EC cohort size and random controls (raw spectra)', r.model, r.slope, r.intercept, r.R2, r.RMSE)),
    ...rows(data, 'airspec_addis').map((r) => fixed190('ftir_13 · AIRSpec baselines (df1 = 6 and 8)', r.model, r.slope, r.intercept, r.R2, r.RMSE)),
    ...rows(data, 'fullpool').filter((r) => r.MAC_m2_g === 10).map((r) => fixed190('ftir_17 · no cleaning: the full 13k pool', `${r.model} (k = ${r.k})`, r.addis_slope, r.addis_intercept, r.addis_R2, r.addis_RMSE)),
    ...rows(data, 'hybrid').map((r) => fixed190('ftir_15 · hybrid OC/EC ∩ spectral-similarity cohort', r.model, r.addis_slope, r.addis_intercept, r.addis_R2, r.addis_RMSE)),
  ]
  // ftir_18 on the 239 pairs: a HIPS-native model predicts Fabs, divided by the MAC on both axes, so its slope and R²
  // do not move and its intercept and RMSE scale by exportMac / MAC; a TOR model's slope scales by MAC / exportMac
  const roundup = rows(data, 'roundup').map((r): Row => {
    const native = r.kind === 'HIPS-native', c = m.exportMac / m.mac
    const set = num(r.n) === moments.fixed?.n ? 'fixed' : 'all'
    return { model: `${r.model}${num(r.n) !== 239 ? ` (n = ${r.n})` : ''}`, kind: r.kind,
      intercept: native ? (num(r.intercept) ?? 0) * c : r.intercept, slope: native ? r.slope : m.slope(num(r.slope), 10), R2: r.R2,
      RMSE: native ? (num(r.RMSE) == null ? null : (num(r.RMSE) as number) * c) : rmseAt(r.slope, r.intercept, r.RMSE, set) }
  })

  // ftir_13 [17]: the tracked per-filter AIRSpec predictions, every panel at the reader's MAC
  const COLS: [string, string, string][] = [['lowest-OCEC 800, AIRSpec df1=6', 'Lowest-OC/EC (800)', 'df1 = 6'], ['Smoke 906, AIRSpec df1=6', 'Biomass-smoke (906)', 'df1 = 6'],
    ['lowest-OCEC 800, AIRSpec df1=8', 'Lowest-OC/EC (800)', 'df1 = 8'], ['Smoke 906, AIRSpec df1=8', 'Biomass-smoke (906)', 'df1 = 8']]
  const sets = fixedIds ? ['fixed 190 (as drawn)', 'all 239'] : ['all 239']
  const airspec = rows(data, 'airspec_filters').flatMap((r) => {
    const fabs = num(r.Fabs), id = String(r.ExternalFilterId), base = id.replace(/-\d+$/, '')
    if (fabs == null) return []
    return COLS.flatMap(([col, calibration, baseline]) => sets.filter((s) => s === 'all 239' || fixedIds?.has(base))
      .map((set): Row => ({ id, calibration, baseline, set, x: fabs / m.mac, y: num(r[col]) })))
  })

  return <>
    <PageIntro id="ml-protocols-intro" title="Protocols end to end — the protocol moves the Addis answer on raw spectra; the bootstrap and every training set tried bound it" data={data}
      notebooks="research/ftir_ec_phase3/ftir_21_both_protocols_end_to_end, ftir_22_figures_under_both_protocols, ftir_15_uncertainty_and_hybrid, ftir_11_ocec_ratio_cohort, ftir_13_airspec_corrected_calibrations, ftir_17_deck_figures_and_seasonal_spectra, ftir_18_transfer_roundup; ftir_31 §1.1, 1.2, 1.6, 1.7, 2.2 and 2.6 redraw ftir_21/22 from the same tables">
      <p>The Calibration app protocol fits on every cohort filter with k from interleaved 10-fold CV (first k within 5 % of the minimum); the
        site-held-out protocol fits on the training side of a site-disjoint 80/20 split with k from site-grouped 5-fold CV (first major minimum).
        Same cohorts, spectra and PLS maths. Every Addis number here is an OLS fit of predictions on HIPS: slopes are re-expressed at the MAC set under
        Filters → units (now {m.mac}), RMSE through the Addis filters' HIPS mean and variance, intercepts unchanged.
        Drawn elsewhere: the cohort-size sweep at every N under each protocol (<a href="#tab=calibration">Calibration tab</a>), the MAC pivot
        (ftir_22 [18]; <a href="#tab=research&amp;aep=mac-choice">MAC 6 vs 10</a>), ftir_51's protocol swing (<a href="#tab=research&amp;aep=generalise">Does it generalise</a>)
        and the provisional offset's bootstrap (<a href="#tab=research&amp;aep=provisional-ec">Provisional Addis EC</a>).</p>
    </PageIntro>

    {stripAt.length > 0 && <Figure kind="intervals" id="ml-protocols-strip" data={stripAt} row="cohort" series="protocol" colors={PROTOCOL_COLOR}
      choose={[{ column: 'spectra', label: 'Spectra', default: 'raw' }]}
      panels={[{ value: 'k', label: 'Rule k' }, { value: 'intercept', label: 'Addis intercept (µg/m³)', reference: 0 }, { value: 'slope', label: slopeLabel(), reference: 1 },
        { value: 'rmseAt', label: rmseLabel('Addis') }]}
      aggregate={{ stat: 'median', interval: 'none' }} labelWidth={240}
      title="Protocol strip, every cohort — on raw smoke the protocol moves the Addis intercept by 5.4 µg/m³, on spline-baselined spectra by at most 0.7"
      subtitle={`The explorer's fit at each protocol's rule k, on the fixed 190 Addis filters (OLS against HIPS Fabs/${m.mac}). This is ftir_21 [13]'s strip and ftir_31's setup matrix and intercept ladder (§1.1, 2.2, 1.6, 2.6), extended to the second-derivative spectra and to B2 (interleaved folds, phase 3's first-major-minimum rule). Switch Spectra: second-derivative intercepts swing up to 2.8 µg/m³ (smoke), spline-baselined ones at most 0.7.`}
      provenance={<>Read from <code>calibration.json</code> grid (target Addis, every lot, default cohort sizes, selected on raw spectra; ak, ob, om, rmse). RMSE at another MAC from the fixed 190 filters' Fabs in <code>calibration_runs.json</code>.</>} />}

    <Figure kind="intervals" id="ml-protocols-ftir21" data={headline} row="cohort" series="protocol" colors={PROTOCOL_COLOR}
      panels={[{ value: 'intercept', label: 'Addis intercept (µg/m³)', reference: 0 }, { value: 'slope', label: slopeLabel(), reference: 1 }, { value: 'RMSE', label: rmseLabel('Addis') },
        { value: 'floor', label: 'RMSECV floor, % of mean loading' }, { value: 'tor_r2', label: 'Held-out TOR test R²' }]}
      aggregate={{ stat: 'median', interval: 'none' }} labelWidth={290}
      title="Both protocols end to end (ftir_21) — the app protocol reports lower CV error yet gives steeper Addis slopes everywhere, and only site-held-out produces a TOR test"
      subtitle="The six setups run end to end twice. The app protocol fits on every cohort filter, so it has no held-out TOR test; the site-held-out test is near zero for Ethiopia-shaped smoke (R² 0.00). The site-held-out rows reproduce the locked ftir_11/ftir_13 calibrations exactly (the notebook asserts it). The per-filter clouds behind ftir_21 [11] (and ftir_31 §1.2) need a re-run: see the list above."
      provenance={provenance(data, ['headline', 'tor_test'])} />

    <Figure kind="intervals" id="ml-protocols-row-order" data={order} row="row order" series="protocol" colors={PROTOCOL_COLOR}
      panels={[{ value: 'k', label: 'k chosen' }, { value: 'intercept', label: 'Addis intercept (µg/m³)', reference: 0 }]}
      aggregate={{ stat: 'median', interval: 'none' }} labelWidth={200}
      title="Row-order sensitivity (ftir_22) — reordering the same 800 filters moves the app protocol's k from 15 to 19; site-held-out stays at 5"
      subtitle="Lowest-OC/EC (800) on spline-baselined spectra. Interleaved folds are assigned by position (every 10th row), so the app protocol's answer depends on the file's row order; site-grouped folds do not. This is why ftir_21 (k 19, −1.65) and ftir_22 (k 18, −1.78) differ on the same cohort."
      provenance={provenance(data, ['row_order'])} />

    <Figure kind="intervals" id="ml-protocols-bootstrap" data={boot} row="row" series="source" colors={SOURCE_COLOR}
      panels={[{ value: 'mid', lo: 'lo', hi: 'hi', label: 'Addis intercept (µg/m³): bootstrap median and 95 % interval', reference: 0 }]} labelWidth={330}
      title="Bootstrap intercept intervals (ftir_22, ftir_15) — the corrected intercept stays between −2.1 and −1.0 under either protocol; the raw one spans −6.2 to −2.8"
      subtitle="Training sites resampled with replacement (B = 200), refitted at each protocol's own k, the Addis intercept re-measured. ftir_15's [−1.78, −1.06] for the corrected model was conditional on k = 5; ftir_22 shows what it becomes under the app protocol. The intercept does not depend on the MAC. The draws themselves (for distributions) need a re-run."
      provenance={provenance(data, ['boot_modes', 'boot_locked'])} />

    <Figure kind="intervals" id="ml-protocols-bootstrap-locked" data={locked} row="model"
      panels={[{ value: 'intercept', lo: 'intercept_lo', hi: 'intercept_hi', label: 'Intercept (µg/m³)', reference: 0 },
        { value: 'slope', lo: 'slope_lo', hi: 'slope_hi', label: slopeLabel('Slope'), reference: 1 }, { value: 'rmse', lo: 'rmse_lo', hi: 'rmse_hi', label: 'RMSE (µg/m³, MAC 10)' }]} labelWidth={250}
      title="Locked models’ bootstrap (ftir_15) — the bootstrap brackets slope and scatter too: the corrected model trades intercept for a flatter slope and twice the RMSE"
      subtitle={`The two locked OC/EC-800 models (site-held-out protocol), 95 % intervals of B = 200 site-cluster refits. A slope interval is re-expressed exactly at another MAC (× MAC/10); the RMSE interval needs the draws, so it stays at MAC 10${m.changed ? ` while the page is at ${m.label}` : ''}.`}
      provenance={provenance(data, ['boot_locked'])} />

    <Figure kind="intervals" id="ml-protocols-drivers" data={drivers} row="row" series="driver" block="block" colors={DRIVER_COLOR}
      panels={[{ value: 'pearson', label: 'Pearson r with the Addis residual', reference: 0 }, { value: 'spearman', label: 'Spearman ρ', reference: 0 }]}
      aggregate={{ stat: 'median', interval: 'none' }} labelWidth={330}
      title="Residual drivers (ftir_22, ftir_15) — raw-model residuals track the distance from the calibration cloud (D², r 0.71–0.87) under both protocols; corrected-model residuals do not (r ≤ 0.33)"
      subtitle="Residual = prediction − Fabs/10 on the 239 Addis filters, against Mahalanobis D² in the model's score space, the Q residual off it, and Fabs. Extrapolation drives the raw model's errors; the corrected model's offset is something else. Correlations are at MAC 10: at another MAC they need each filter's residual, which only a re-run writes (residual vs D² per filter, ftir_22 [13] and ftir_31 §1.3)."
      provenance={provenance(data, ['drivers_modes', 'drivers_locked'])} />

    <Figure kind="intervals" id="ml-protocols-seasons" data={season} row="season" series="model" colors={MODEL_COLOR}
      panels={[{ value: 'mean_residual', label: `Mean residual, prediction − Fabs/${m.mac} (µg/m³)`, reference: 0 }, { value: 'median_D2', label: 'Median D² (score space)' }]}
      aggregate={{ stat: 'median', interval: 'none' }} labelWidth={170}
      title="Residual by season (ftir_15) — the corrected model's residual is a season-stable offset; the raw model's swings with the season and its D²"
      subtitle={`The two locked models' Addis residuals by Ethiopian season.${m.changed ? ` Each season's mean residual is moved from MAC 10 to ${m.label} by that season's mean HIPS Fabs (calibration_runs.json), which is exact.` : ''} The raw model's residual is highest in Kiremt, where its D² is largest; the corrected model's sits below zero in every season.`}
      provenance={provenance(data, ['season_residuals'])} />

    <Figure kind="intervals" id="ml-protocols-size" data={sweep} row="size" series="protocol" colors={PROTOCOL_COLOR}
      panels={[{ value: 'k', label: 'k chosen' }, { value: 'intercept', label: 'Addis intercept (µg/m³)', reference: 0 }, { value: 'slope', label: slopeLabel(), reference: 1 },
        { value: 'tor_r2', label: 'Held-out TOR test R²' }]}
      aggregate={{ stat: 'median', interval: 'none' }} labelWidth={110}
      title="Cohort size by protocol (ftir_22) — N = 800 is the best OC/EC cohort size only under site-held-out; under the app protocol the intercept just shrinks with N"
      subtitle="ftir_11's three lowest-OC/EC cohort sizes (raw spectra) re-derived under both protocols, fixed 190 Addis filters. Retire “800 is the sweet spot” as a protocol-free claim. Every size from 100 to 2000 under each protocol is in the Calibration tab's cohort-size sweep."
      provenance={provenance(data, ['size_sweep'])} />

    <Figure kind="intervals" id="ml-protocols-training" data={training} row="model" block="block"
      panels={[{ value: 'intercept', label: 'Addis intercept (µg/m³)', reference: 0 }, { value: 'slope', label: slopeLabel(), reference: 1 }, { value: 'R2', label: 'R²' },
        { value: 'RMSE', label: rmseLabel('Addis') }]}
      aggregate={{ stat: 'median', interval: 'none' }} labelWidth={330}
      title="Training sets on the same 190 filters (ftir_11, 13, 15, 17) — the training set moves the intercept from −5.3 to −0.6 µg/m³, and the intercepts nearest zero come with the flattest slopes"
      subtitle="Every training-set choice of ftir_11, 13, 15 and 17 under the site-held-out protocol: cohort size and five random 800s, AIRSpec baselines on the OC/EC and smoke cohorts, no cleaning at all on the 13k pool, and the hybrid OC/EC ∩ spectral-similarity cohort (whose held-out TOR test fails, R² 0.19). No choice gets the intercept to zero with the slope near 1."
      provenance={provenance(data, ['sizes_addis', 'airspec_addis', 'fullpool', 'hybrid'])} />

    <Figure kind="intervals" id="ml-protocols-roundup" data={roundup} row="model"
      panels={[{ value: 'intercept', label: `Addis intercept (µg/m³${m.changed ? `, ${m.label}` : ''})`, reference: 0 }, { value: 'slope', label: slopeLabel(), reference: 1 }, { value: 'R2', label: 'R²' },
        { value: 'RMSE', label: rmseLabel('Addis') }]}
      aggregate={{ stat: 'median', interval: 'none' }} labelWidth={300}
      title="Transfer roundup (ftir_18) — no transfer closes the gap: only the Addis-only nested-CV ceiling sits on the 1:1 line"
      subtitle={`Seven training choices on one axis, 239 Addis filters (deployed: its 190). Two predict HIPS Fabs directly (the IMPROVE HIPS-τ transfer and the Addis-only nested CV): divided by the MAC on both axes, their slope and R² do not move with the MAC and their intercept and RMSE scale by 10/MAC. The 13k-filter TOR model beats the 916-filter HIPS transfer on R² only because it has 14× the training filters. Per-filter predictions for the HIPS-τ and nested-CV models need a re-run.`}
      provenance={provenance(data, ['roundup'])} />

    <Figure kind="table" id="ml-protocols-hips-transfer" data={rows(data, 'hips_transfer')}
      columns={[{ key: 'model', label: 'Model' }, { key: 'k', label: 'k', format: 'int' }, { key: 'heldout_tau_R2', label: 'Held-out τ R² (IMPROVE)', format: 'f3' },
        { key: 'addis_median_Fabs_predicted', label: 'Addis median Fabs, predicted', format: 'f2' }, { key: 'addis_median_Fabs_observed', label: 'observed', format: 'f2' },
        { key: 'addis_bias', label: 'Bias (Mm⁻¹)', format: 'f2' }, { key: 'addis_RMSE', label: 'RMSE (Mm⁻¹)', format: 'f2' }]}
      title="HIPS-τ transfer (ftir_13) — even on AIRSpec spectra the IMPROVE HIPS-τ transfer predicts an Addis median Fabs of 12.5 against 47.1 observed"
      subtitle="Trained on IMPROVE filters' HIPS absorption and applied to Addis: baseline correction does not rescue the transfer. In Fabs units, so it does not depend on the MAC."
      provenance={provenance(data, ['hips_transfer'])} />

    {airspec.length > 0 && <Figure kind="crossplot" id="ml-protocols-airspec" data={airspec} x="x" y="y" facet="calibration" idColumn="id"
      choose={[{ column: 'baseline', label: 'AIRSpec baseline' }, { column: 'set', label: 'Filters' }]}
      xLabel={`HIPS Fabs / ${m.mac} (µg/m³)`} yLabel="Predicted EC (µg/m³)" sameScale
      title="AIRSpec crossplots (ftir_13 [17]) — on AIRSpec spectra both calibrations flatten at Addis; the Smoke-906 model reads nearly flat"
      subtitle={`Each Addis filter's prediction from the AIRSpec-corrected Lowest-OC/EC (800) and Biomass-smoke (906) calibrations against HIPS Fabs/${m.mac}; the fit line is the gallery's estimator. The OC/EC panel is the Calibration tab's ocec800_airspec preset; the Smoke-906 panel is in no other export. df1 = 8 changes nothing visible. The fixed 190 are the filters the notebook drew (they need calibration_runs.json; all 239 otherwise).`}
      provenance={provenance(data, ['airspec_filters'])} />}

    <Figure kind="table" id="ml-protocols-index" data={rows(data, 'figure_index')}
      columns={[{ key: 'notebook', label: 'Notebook' }, { key: 'original figure', label: 'Original figure' }, { key: 'sensitivity', label: 'Depends on k?' }, { key: 'updated version', label: 'Protocol-checked version' }]}
      title="Figure index (ftir_22) — which ftir_11–19 figures depend on the component choice, and where each protocol-checked version lives"
      subtitle="The notebooks' own map. Figures with no model in them (band identity, implied MAC, spectra) are k-free; every k-dependent one has a both-protocol version, drawn on this page or the Calibration tab where the gallery has its numbers."
      provenance={provenance(data, ['figure_index'])} />
  </>
}
