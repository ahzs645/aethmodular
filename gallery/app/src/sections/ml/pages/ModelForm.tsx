import { useMemo } from 'react'
import { Figure } from '@/components/Figure'
import { f2 } from '@/lib/format'
import { useMac } from '@/lib/mac'
import { num, type Row } from '@/lib/table'
import { themed } from '@/lib/theme'
import { useCalibRuns } from '../../research/pages/spectralCommon'
import type { SectionPageProps } from '../../types'
import { MlWaiting, PageIntro, provenance, rows, useMl } from '../common'
import { FoldedPlane } from './model-form/FoldedPlane'
import { chosenK } from './model-form/hips'
import { HipsTransfer } from './model-form/HipsTransfer'
import { VipSpectra, type VipSet } from './model-form/VipSpectra'
import { bestScoreCells, GUARDS, protocolRows, useCalib, W, type GuardKey } from './model-form/search'

const FORMS = ['PLS', 'HGB', 'RF', 'MLP', 'PLS→HGB']
const FORM_NAME: Record<string, string> = {
  PLS: 'PLS', HGB: 'HGB (histogram gradient boosting)', RF: 'RF (random forest)', MLP: 'MLP (neural network, 64 units)', 'PLS→HGB': 'PLS scores → HGB',
}
const FAMILY_COLOR: Record<string, string> = {
  'lowest-OC/EC 800 · corrected': '#2b6cb0', 'lowest-OC/EC 800 · deriv2': '#c026d3', 'full pool · corrected': '#7b8794',
}
const VIP_COLOR: Record<string, string> = { EC: '#2b6cb0', OC: '#e67e22', 'IMPROVE HIPS': themed('#34495e', '#9fb3c8'), 'Addis HIPS': '#c0392b' }
/** ftir_08's two component curves (RMSE of HIPS optical depth τ by k) */
const CURVES = [
  { table: 'hips_cv_improve', name: 'IMPROVE HIPS · whole sites held out', model: 'IMPROVE HIPS → Addis external test' },
  { table: 'hips_cv_addis', name: 'Addis HIPS · random 5-fold', model: 'Addis-only HIPS' },
]
const byForm = (a: Row, b: Row) => FORMS.indexOf(String(a.model)) - FORMS.indexOf(String(b.model))
const range = (v: (number | null)[]) => { const x = v.filter((n): n is number => n != null); return `${f2(Math.min(...x))}–${f2(Math.max(...x))}` }

/**
 * ftir_54 (model form), ftir_55 and ftir_51 05/06 (every combination tried), ftir_07/ftir_08 (VIP) and ftir_08 (HIPS
 * as the PLS target). The ftir_54 and VIP numbers are the executed notebooks' tables or the CSVs they wrote
 * (gallery/data/ml_pages/model_form.py); the combination search is redrawn from calibration.json, and ftir_08's
 * transfer crossplot and metrics from its per-filter predictions. Slopes against HIPS Fabs/10 follow the reader's MAC (× MAC/10, intercepts
 * unchanged); what cannot (an Addis RMSE or a residual rank against Fabs/10) is labelled MAC 10.
 */
export function ModelForm(_props: SectionPageProps) {
  const { data, error } = useMl('model-form')
  const { calib, error: calibError } = useCalib()
  const { runs } = useCalibRuns()
  const mac = useMac()

  const cells = useMemo(() => calib ? (Object.keys(GUARDS) as GuardKey[]).flatMap((g) =>
    bestScoreCells(calib, mac.mac, g).map((r): Row => ({ ...r, guardrails: GUARDS[g].label }))) : [], [calib, mac.mac])
  const protocols = useMemo(() => (calib ? protocolRows(calib, mac.mac) : []), [calib, mac.mac])

  if (!data) return <MlWaiting error={error} what="the model-form page" />

  // ftir_54: one row per cohort × representation × model form; the Addis slope is against Fabs/10
  const forms = rows(data, 'model_forms').map((r): Row => ({ ...r, family: `${r.cohort} · ${r.representation}`, form: FORM_NAME[String(r.model)] ?? r.model,
    addis_slope: mac.slope(num(r.addis_slope), 10) })).sort(byForm)
  const locked = forms.filter((r) => r.family === 'lowest-OC/EC 800 · corrected')
  const pls = locked.find((r) => r.model === 'PLS')
  const others = locked.filter((r) => r.model !== 'PLS')
  const intercepts = forms.map((r) => num(r.addis_intercept))
  const allNegative = intercepts.every((v) => v != null && v < 0)
  const extrap = rows(data, 'extrapolation').map((r): Row => ({ ...r, form: FORM_NAME[String(r.model)] ?? r.model })).sort(byForm)
  const edge = (test: string, m: string) => num(extrap.find((r) => String(r.test).startsWith(test) && r.model === m)?.slope)
  const trees = (test: string) => range(['HGB', 'RF'].map((m) => edge(test, m)))
  const spearman = rows(data, 'model_forms').filter((r) => r.cohort === 'lowest-OC/EC 800' && r.representation === 'corrected')
  const treeRho = range(spearman.filter((r) => r.model === 'HGB' || r.model === 'RF').map((r) => num(r.spearman_resid_vs_pcaD2)))
  const rf = num(rows(data, 'domain_spearman').find((r) => r.diagnostic === 'PCA D²' && String(r.against).startsWith('RF'))?.rho)
  const tuning = rows(data, 'hgb_tuning')

  const ftir55Cells = cells.filter((r) => r.guardrails === GUARDS.ftir55.label)
  const passCells = ftir55Cells.filter((r) => num(r.passing))
  const bestCell = passCells.reduce<Row | null>((b, r) => (!b || (num(r.best_score) ?? Infinity) < (num(b.best_score) ?? Infinity) ? r : b), null)
  const notebookCells = rows(data, 'ftir55_cells').filter((r) => num(r.passing))

  const raw = protocols.filter((r) => r.lot === 'all lots' && String(r.configuration).includes('800 × raw')).map((r) => num(r.intercept))
  const corr = protocols.filter((r) => r.lot === 'all lots' && String(r.configuration).includes('800 × spline')).map((r) => num(r.intercept))
  const spread = (v: (number | null)[]) => { const x = v.filter((n): n is number => n != null); return x.length ? Math.max(...x) - Math.min(...x) : null }

  const overlap = rows(data, 'vip_overlap')
  const ecOc = overlap[0], hips = overlap[1]
  const bands = rows(data, 'vip_bands')
  const share = (m: string, band: string) => num(bands.find((r) => r.model === m && r.band === band)?.vip2_share)
  const pct = (v: number | null) => (v == null ? '—' : `${(100 * v).toFixed(0)} %`)
  const hipsMetrics = rows(data, 'hips_transfer')
  const kOf = (model: string) => num(hipsMetrics.find((r) => r.model === model)?.components)
  const vipSets: VipSet[] = [
    { key: 'hips', label: 'IMPROVE vs Addis HIPS', table: 'vip_hips', models: ['IMPROVE HIPS', 'Addis HIPS'],
      k: [kOf('IMPROVE HIPS → Addis external test'), kOf('Addis-only HIPS')], cite: 'ftir_08 [12]' },
    { key: 'ec-oc', label: 'EC vs OC', table: 'vip_ec_oc', models: ['EC', 'OC'], k: [31, 53], cite: 'ftir_07 [5]' },
  ]

  // ftir_08 [7]/[9]: the component curves, mean ± 1 SD over the folds, and the k each one picks (lowest mean)
  const curves = CURVES.flatMap((c) => rows(data, c.table).map((r): Row => {
    const m = num(r.rmse_mean), sd = num(r.rmse_sd)
    return { curve: c.name, k: r.n_components, rmse: m, lo: m != null && sd != null ? m - sd : null, hi: m != null && sd != null ? m + sd : null }
  }))
  const picks = CURVES.map((c) => ({ ...c, k: chosenK(rows(data, c.table).map((r) => ({ k: num(r.n_components) ?? NaN, rmse: num(r.rmse_mean) ?? NaN }))), shown: kOf(c.model) }))
  const nested = rows(data, 'hips_nested_k').map((r) => num(r.chosen_n_components))
  const audit = (start: string) => num(rows(data, 'hips_audit').find((r) => String(r.cohort).startsWith(start))?.n)

  return <>
    <PageIntro id="ml-model-form-intro" title="Model form, every combination tried, and feature importance" data={data}
      notebooks="research/ftir_ec_phase3/ftir_54_model_form_robustness, ftir_55_every_combination_tried, ftir_51_priority_figure_export §5–6; research/ftir_hips_chem/ftir_07_current_pls_vip_diagnostics, ftir_08_hips_transfer_and_vip">
      <p>Three questions about the calibration's form rather than its cohort: would another learner do better on the
        same spectra (ftir_54), what did the search over every cohort, cutoff, spectra, protocol, lot and k actually
        find (ftir_55, ftir_51), which wavenumbers do the models lean on (ftir_07, ftir_08), and does a PLS model
        trained on HIPS rather than TOR EC carry from IMPROVE to Addis (ftir_08)? The ftir_54 and VIP numbers are the
        executed notebooks' tables or the CSVs they wrote; the search is redrawn from <code>calibration.json</code>,
        which the Calibration tab already loads, and ftir_08's transfer crossplot and its metrics from the
        notebook's per-filter predictions. Slopes against HIPS Fabs/10 are re-expressed at the MAC set under
        Filters → units (now {mac.label}); an Addis RMSE or a residual rank against Fabs/10 cannot be, and says MAC 10.</p>
      <p>Already drawn elsewhere: the full specification curve and the slope–intercept trap on the{' '}
        <a href="#tab=calibration">Calibration tab</a> (Specification curve; Slope vs intercept), and VIP profiles of the gallery's own calibrations on the{' '}
        <a href="#tab=meeting&meeting=meeting-20261028">28 Oct meeting page</a> (§8, What each calibration reads as EC) and{' '}
        <a href="#tab=meeting&meeting=satoshi">Satoshi's 2 Oct questions</a> (What PLS sees as EC).</p>
    </PageIntro>

    {/* ------------------------------------------------------------ 1. ftir_54 model form */}
    <Figure kind="intervals" id="ml-model-form-forms" data={forms} row="form" series="family" colors={FAMILY_COLOR}
      panels={[{ value: 'heldout_R2', label: 'Held-out IMPROVE R² (untouched sites)', reference: 1 },
        { value: 'addis_intercept', label: 'Addis Deming intercept (µg/m³)', reference: 0 },
        { value: 'addis_slope', label: `Addis Deming slope vs Fabs/${mac.mac}`, reference: 1 }]}
      aggregate={{ stat: 'median', interval: 'none' }} labelWidth={230}
      title={`${allNegative ? `All ${forms.length} fits put the Addis intercept below zero` : 'The Addis intercept by model form'}, and no other form beats PLS on held-out sites — ftir_54 [10], five model forms on the same split`}
      subtitle={`The same site-disjoint protocol split and the same 190 fixed Addis filters, five learners: PLS (k from the locked protocol rule), histogram gradient boosting (tuned below), random forest, a 64-unit MLP, and boosting on the PLS scores. On the locked Lowest OC/EC 800 cohort with corrected spectra PLS reaches held-out R² ${f2(pls?.heldout_R2)} against ${range(others.map((r) => num(r.heldout_R2)))}; the one near-tie (random forest on 2nd-derivative spectra) comes with a steeper Addis slope and a more negative intercept. Intercepts span ${range(intercepts)} µg/m³: the negative Addis offset does not depend on PLS. Slopes are re-expressed at ${mac.label}; the trees on corrected spectra land where the raw PLS model did (≈1.4–1.6x at MAC 10).`}
      provenance={provenance(data, ['model_forms'])} />

    <Figure kind="intervals" id="ml-model-form-extrapolation" data={extrap} row="form" block="test"
      panels={[{ value: 'RMSE_ug', label: 'RMSE on the held-out slice (µg/filter)' },
        { value: 'slope', label: 'Slope, predicted vs TOR EC', reference: 1 },
        { value: 'bias_ug', label: 'Mean bias (µg/filter)', reference: 0 }]}
      aggregate={{ stat: 'median', interval: 'none' }} labelWidth={230}
      title={`Where the truth is known, PLS extrapolates best: at the spectral edge it keeps slope ${f2(edge('B', 'PLS'))}, the trees ${trees('B')} — ftir_54 [13a], labelled extrapolation inside IMPROVE`}
      subtitle={`Addis has no labels, so extrapolation can only be scored inside the network. Each test holds out a slice of the full corrected pool and trains every form on the rest (no site restriction): A, the 800 lowest-OC/EC filters (the network's closest analogue of Addis); B, the 5 % with the largest PCA D²; C, the 5 % highest EC loadings. Every form under-predicts the low-OC/EC corner with slope ≈ 0.3 (a labelled echo of the Addis offset); at the loading edge the trees saturate (slope ${trees('C')}) while PLS keeps ${f2(edge('C', 'PLS'))}. Against TOR EC: no MAC.`}
      provenance={provenance(data, ['extrapolation'])} />

    <Figure kind="intervals" id="ml-model-form-diagnosability" data={forms} row="form" series="family" colors={FAMILY_COLOR}
      panels={[{ value: 'spearman_resid_vs_pcaD2', label: 'ρ, Addis residual (vs Fabs/10) and PCA D²', reference: 0 },
        { value: 'spearman_pred_vs_pcaD2', label: 'ρ, Addis prediction and PCA D²', reference: 0 }]}
      aggregate={{ stat: 'median', interval: 'none' }} labelWidth={230}
      title={`PLS's Addis residual ignores spectral distance; the trees' residuals track it (ρ ${treeRho}) — ftir_54 [15], diagnosability`}
      subtitle="Spearman correlation, over the 190 Addis filters, of each model's prediction and its residual against HIPS Fabs/10 (MAC 10: a residual's rank order changes with the MAC) with the model-agnostic PCA-10 D² of the spectrum. The corrected PLS residual is independent of distance (the ftir_15 constant-offset result); the tree residuals rise with it, but trees carry no internal flag. The per-filter scatter behind these numbers was never written (see the list above)."
      provenance={provenance(data, ['model_forms'])} />

    <Figure kind="heatmap" id="ml-model-form-flags" data={rows(data, 'domain_spearman')} row="diagnostic" column="against" value="rho" domain={[-1, 1]} format="f2"
      valueLabel="Spearman ρ over the 190 Addis filters" cell={[150, 30]}
      title={`Random-forest spread is a weak distance flag at Addis (ρ ${f2(rf)} with PCA D²) — ftir_54 [15], domain diagnostics`}
      subtitle="Three ways to say a filter is off-domain for the 800-cohort models: PCA-10 D² (any model), PLS score-space D² (PLS only) and the spread of the random forest's trees. PLS D² agrees with the model-agnostic distance; the tree spread barely does. Addis sits inside the envelope either way (a few per cent beyond the training p95)."
      provenance={provenance(data, ['domain_spearman'])} />

    <Figure kind="table" id="ml-model-form-hgb" data={tuning}
      columns={[{ key: 'learning_rate', label: 'Learning rate', format: 'f2' }, { key: 'max_leaf_nodes', label: 'Max leaves', format: 'int' }, { key: 'max_iter', label: 'Iterations', format: 'int' },
        { key: 'l2_regularization', label: 'L2', format: 'f2' }, { key: 'rmsecv_ug', label: 'Site-grouped 3-fold RMSECV (µg/filter)', format: 'f3' }]}
      title={`Boosting was tuned before it was compared: learning rate 0.05 wins, leaf count barely matters (${range(tuning.map((r) => num(r.rmsecv_ug)))} µg) — ftir_54 [6], HGB grid`}
      subtitle="Site-grouped 3-fold CV on the 800 cohort's training sites only, so no Addis or held-out site informs the choice; the best row was carried to every comparison above. No grid search on the full pool, no CNN or Gaussian process (the notebook's stated limits)."
      provenance={provenance(data, ['hgb_tuning'])} />

    {/* ------------------------------------------------------------ 2. ftir_55 / ftir_51: every combination tried */}
    {!calib ? <MlWaiting error={calibError} what="calibration.json (the explorer's summary fits)" /> : <>
      <Figure kind="heatmap" id="ml-model-form-best-score" data={cells} row="cohort" column="spectra" value="best_score" ramp="sequential" format="f2" cell={[150, 32]}
        choose={[{ column: 'guardrails', label: 'Guardrails', default: GUARDS.ftir55.label }]}
        variants={[{ key: 'measure', label: 'Show', default: 'score', choices: [
          { value: 'score', label: 'Best passing score', patch: { value: 'best_score', format: 'f2', domain: [0, 8], valueLabel: `best passing score |intercept| + ${W}·|slope − 1| (lower is closer to 1:1)` } },
          { value: 'passing', label: 'Configurations passing', patch: { value: 'passing', format: 'int', domain: undefined, valueLabel: 'configurations passing the guardrails' } },
          { value: 'tried', label: 'Configurations tried', patch: { value: 'tried', format: 'int', domain: undefined, valueLabel: 'configurations in calibration.json' } }] }]}
        title={`${passCells.length} of ${ftir55Cells.length} cohort × spectra cells hold a configuration that passes the guardrails${bestCell ? `; the best is ${bestCell.cohort} × ${String(bestCell.spectra).toLowerCase()} at ${f2(bestCell.best_score)}` : ''} — ftir_55 [11], best score by cohort × spectra (${mac.label}, ftir_55's slope box)`}
        subtitle={`Every Addis configuration in calibration.json scored on all its filters (each cutoff, selection space, protocol and training lot, at the rule k): the best explorer score |Deming intercept| + ${W}·|slope − 1| among those keeping held-out TOR R² ≥ ${calib.heldout_floor} and a slope inside the box, at ${mac.label} (slope × MAC/10, intercept unchanged, score and box re-judged). ftir_55 also required the extrapolation, Q-residual and negatives diagnostics, which calibration.json does not carry, and scored every k; an empty cell has no passing configuration.`}
        note={`ftir_55's own table (MAC 10, every k, all five guardrails): ${notebookCells.map((r) => `${r.cohort} × ${r.spectra} ${r.passing} pass, best ${f2(r.best_score)} at cutoff ${r.best_cutoff}, k = ${r.best_k} (${f2(r.best_slope)}x ${f2(r.best_intercept)})`).join('; ')}. Best here: ${passCells.map((r) => `${r.cohort} × ${r.spectra}: ${f2(r.best_score)}, ${f2(r.best_slope)}x ${f2(r.best_intercept)} (${r.best})`).join('; ') || 'none'}.`}
        provenance={<>ftir_55 §2c, redrawn from <code>gallery/app/public/data/calibration.json</code> (grid, target Addis, evaluation lot all). {provenance(data, ['ftir55_cells'])}</>} />

      <FoldedPlane calib={calib} provenance={<>ftir_55 §2a, redrawn from <code>gallery/app/public/data/calibration.json</code> (grid: Addis, site-held-out, lot all). The unfolded plane, with cohort colours and guardrails, is the Calibration tab's Slope vs intercept chart.</>} />

      <Figure kind="intervals" id="ml-model-form-protocols" data={protocols} row="configuration" series="protocol"
        choose={[{ column: 'lot', label: 'Training lot', default: 'all lots' }]}
        panels={[{ value: 'intercept', label: 'Addis Deming intercept, fixed 190 (µg/m³)', reference: 0 },
          { value: 'slope', label: `Addis Deming slope vs Fabs/${mac.mac}`, reference: 1 }]}
        aggregate={{ stat: 'median', interval: 'none' }} labelWidth={320}
        title={`The CV protocol moves a raw calibration's intercept by ${f2(spread(raw))} µg/m³ and the spline one's by ${f2(spread(corr))} — ftir_51 §5–6, protocols A, B, B2 and training lot`}
        subtitle={`ftir_51's variants (and the same cohorts on the other spectra) under the three component-selection protocols: A, site-grouped 5-fold CV, first major minimum (the deployed rule); B, interleaved 10-fold, within 5 % (the calibration app); B2, interleaved 10-fold, first major minimum. Training lot switches to the lot-248-only and lot-251-only fits of ftir_51 §6 (lot 248 is ~1,300 pool filters, used only around Jan–Feb 2021, so a lot-248 training set is also a winter-2021 one). Deming fits on the fixed 190 Addis filters; slopes at ${mac.label}.`}
        provenance={<>Redrawn from <code>calibration.json</code> (grid: Addis, evaluation lot all). {provenance(data, ['cv_protocols'])} The notebook's protocol table matches these rows to the stored digits.</>} />
    </>}

    {/* ------------------------------------------------------------ 3. ftir_07 / ftir_08: VIP */}
    <Figure kind="table" id="ml-model-form-vip-overlap" data={overlap}
      columns={[{ key: 'comparison', label: 'Models compared' }, { key: 'spearman_r', label: 'Spearman r of VIP profiles', format: 'f3' },
        { key: 'top_n_overlap', label: 'Top-200 wavenumbers shared', format: 'int' }, { key: 'important_jaccard', label: 'Jaccard, VIP ≥ 1 sets', format: 'f2' },
        { key: 'important_a_n', label: 'VIP ≥ 1, first', format: 'int' }, { key: 'important_b_n', label: 'VIP ≥ 1, second', format: 'int' },
        { key: 'a_vip_mass_on_b_important', label: 'First model’s VIP² where the second has VIP ≥ 1', format: 'pct' }]}
      title={`EC leans on OC's wavenumbers (${ecOc?.top_n_overlap ?? '—'} of the top 200 shared); the IMPROVE and Addis HIPS models share ${num(hips?.top_n_overlap) === 0 ? 'none' : hips?.top_n_overlap ?? '—'} of theirs — ftir_07 [5], ftir_08 [12], VIP overlap`}
      subtitle={`VIP (Takahama et al. 2019, Eq. 14) on the 2,722-point raw grid. ftir_07: the July 7 IMPROVE EC (k 31) and OC (k 53) models, reconstructed in Python to within 1×10⁻⁴ (EC) and 0.003 (OC) of the R fitted values; ${pct(num(ecOc?.a_vip_mass_on_b_important))} of EC's squared-VIP mass sits where OC's VIP ≥ 1. ftir_08: an IMPROVE HIPS model (k 1, site-held-out) and an Addis-only HIPS model (k 14, nested CV), both on filter optical depth: broadly correlated (r ${f2(hips?.spearman_r)}) yet with no top-200 wavenumber in common. VIP is a post hoc importance; it names no unique molecule.`}
      provenance={provenance(data, ['vip_overlap', 'vip_reconstruction'])} />

    <VipSpectra data={data} sets={vipSets} colors={VIP_COLOR} provenance={provenance(data, ['vip_hips', 'vip_ec_oc', 'vip_band_edges', 'vip_overlap'])}
      missing={<p className="subtitle">ftir_07's and ftir_08's per-wavenumber VIP profiles are local CSVs (see the list at the top for the commands that write them). The
        gallery's own VIP charts draw the same quantity for its calibrations: the <a href="#tab=meeting&meeting=meeting-20261028">28 Oct meeting page</a> §8 and{' '}
        <a href="#tab=meeting&meeting=satoshi">Satoshi's 2 Oct questions</a>.</p>} />

    <Figure kind="intervals" id="ml-model-form-vip-bands" data={bands} row="band" series="model" block="comparison" colors={VIP_COLOR}
      panels={[{ value: 'vip2_share', label: 'Share of squared-VIP mass in the band' }]}
      aggregate={{ stat: 'median', interval: 'none' }} labelWidth={230}
      title={`Addis HIPS moves VIP weight from the O–H/N–H region (${pct(share('IMPROVE HIPS', 'O-H / N-H region'))} → ${pct(share('Addis HIPS', 'O-H / N-H region'))}) to the fingerprint (${pct(share('IMPROVE HIPS', 'fingerprint region'))} → ${pct(share('Addis HIPS', 'fingerprint region'))}); EC and OC weight every band alike — ftir_07 [7], ftir_08 [12], VIP² by band`}
      subtitle={`Each model's squared-VIP mass by broad band (the shares do not sum to one: the rest falls between the named bands). EC puts ${pct(share('EC', 'fingerprint region'))} below 1500 cm⁻¹, so cutting the spectrum there would drop much of its information. Broad bands, not single peaks, because FTIR bands overlap.`}
      provenance={provenance(data, ['vip_bands'])} />

    <Figure kind="table" id="ml-model-form-shared-peaks" data={rows(data, 'shared_peaks')}
      columns={[{ key: 'wavenumber_cm-1', label: 'Wavenumber (cm⁻¹)', format: 'int' }, { key: 'EC_VIP', label: 'EC VIP', format: 'f2' }, { key: 'OC_VIP', label: 'OC VIP', format: 'f2' }, { key: 'score', label: 'EC VIP × OC VIP', format: 'f2' }]}
      title="The wavenumbers both EC and OC weight most lie below 1260 cm⁻¹ (1120–1255, 500–650), with 2917 C–H and 1712 carbonyl — ftir_07 [7], shared VIP peaks"
      subtitle="Spaced peaks of EC VIP × OC VIP (the notebook displayed the top 12 of 20). These are where an OC/EC composition shift would move an EC prediction; they are candidates for targeted transfer tests, not band assignments."
      provenance={provenance(data, ['shared_peaks'])} />

    {/* ------------------------------------------------------------ 4. ftir_08: HIPS as the PLS target */}
    {curves.length > 0 && <Figure kind="sweep" id="ml-model-form-hips-k" data={curves} x="k" y="rmse" lo="lo" hi="hi" series="curve" interval="whisker"
      choose={[{ column: 'curve', label: 'Model', default: CURVES[0].name }]}
      colors={{ [CURVES[0].name]: VIP_COLOR['IMPROVE HIPS'], [CURVES[1].name]: VIP_COLOR['Addis HIPS'] }}
      chosen={picks.flatMap((p) => (p.k != null ? [{ series: p.name, x: p.k }] : []))}
      xLabel="PLS components (k)" yLabel="CV RMSE of HIPS optical depth τ"
      title={`On held-out IMPROVE sites the HIPS model is best with ${picks[0].k === 1 ? 'a single component' : `k = ${picks[0].k}`}; Addis alone keeps improving to k = ${picks[1].k ?? '—'} — ftir_08 [7], HIPS as the PLS target`}
      subtitle={`PLS on the raw 2,722-point spectra with HIPS filter optical depth τ = Fabs × sampled volume / (100 × deposit area) as the response (MAC-free). IMPROVE: ${audit('IMPROVE matched') ?? '—'} of ${audit('IMPROVE spectra') ?? '—'} filters matched to HIPS Fabs, volume and area, 5-fold CV holding out whole sites, k 1–25 (the notebook's figure). Addis: the ${audit('Addis spectra + HIPS') ?? '—'} filters with spectra, HIPS and a positive volume, random 5-fold, k 1–15 (written, not drawn, by the notebook). Point: mean over folds; whisker: ±1 SD; the ringed point is the k with the lowest mean, which the notebook carried forward (${picks.map((p) => `${p.name.split(' · ')[0]} ${p.k ?? '—'}${p.shown != null ? `, table ${p.shown}` : ''}`).join('; ')}). Between IMPROVE sites the spread swamps the curve: one component is as good as any.${nested.length ? ` Inside the nested CV that produced the Addis predictions below, every outer fold's inner search chose k = ${[...new Set(nested)].join(', ')}.` : ''}`}
      provenance={provenance(data, ['hips_cv_improve', 'hips_cv_addis', 'hips_nested_k', 'hips_audit'])} />}

    <Figure kind="table" id="ml-model-form-hips-transfer" data={hipsMetrics}
      columns={[{ key: 'model', label: 'Model' }, { key: 'validation', label: 'Validation' }, { key: 'components', label: 'k', format: 'int' }, { key: 'n', label: 'n', format: 'int' },
        { key: 'slope', label: 'Slope', format: 'f2' }, { key: 'intercept', label: 'Intercept (Fabs)', format: 'f2' }, { key: 'R2', label: 'R²', format: 'f3' }, { key: 'RMSE', label: 'RMSE (Fabs)', format: 'f2' }, { key: 'bias', label: 'Bias (Fabs)', format: 'f2' }]}
      title="The IMPROVE HIPS model does not transfer to Addis; an Addis-only one fits — ftir_08 [9], transfer metrics"
      subtitle="Predicted against observed HIPS Fabs on the 239 Addis filters (Fabs itself, so no MAC), as the notebook tabulated them; the crossplot below redraws them per filter. The IMPROVE model, trained on optical depth and converted back, predicts a median Addis Fabs about a third of the observed; the Addis-only model is a mechanism probe, not an EC calibration."
      provenance={provenance(data, ['hips_transfer'])} />

    {rows(data, 'hips_addis_filters').length > 0
      ? <HipsTransfer filters={rows(data, 'hips_addis_filters')} shown={hipsMetrics} runs={runs}
        provenance={<>{provenance(data, ['hips_addis_filters', 'hips_transfer'])} The fixed-190 flag, season and date come from <code>gallery/app/public/data/calibration_runs.json</code> (Addis evaluation set).</>} />
      : null}

  </>
}
