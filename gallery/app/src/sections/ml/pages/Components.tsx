import { useMemo } from 'react'
import { Figure } from '@/components/Figure'
import { useMac } from '@/lib/mac'
import { num, type Row } from '@/lib/table'
import type { SweepRow } from '@/lib/types'
import { presetName, useCalibRuns } from '../../research/pages/spectralCommon'
import type { SectionPageProps } from '../../types'
import { MlWaiting, PageIntro, provenance, rows, useMl } from '../common'
import { COHORT_NAME, PROTOCOL_COLOR, protocolOf, ruleRows, SPECTRA_NAME, useCalib } from './components/calib'

/** ftir_20's four rule × scheme combinations and the locked k, in the order the notebook's strip reads them */
const RULE_COLOR: Record<string, string> = {
  'interleaved 10-fold + within 5 % (app)': '#e15759',
  'site-grouped 5-fold + first major minimum (phase 3)': '#4e79a7',
  'site-grouped 5-fold + within 5 %': '#76b7b2',
  'interleaved 10-fold + first major minimum': '#f28e2b',
  'locked k in use': '#59a14f',
}
const SCHEME_COLOR: Record<string, string> = {
  'interleaved 10-fold': '#e15759', 'site-grouped 5-fold': '#4e79a7',
  'interleaved · 10-fold': '#e15759', 'interleaved · 5-fold': '#ff9da7', 'site-grouped · 5-fold': '#4e79a7', 'site-grouped · 10-fold': '#76b7b2',
}
const CUT_COLOR: Record<string, string> = { 'N = 400': '#59a14f', 'N = 800': '#4e79a7', 'N = 1600': '#b07aa1' }
const CALIBRATION = '#tab=calibration'

/**
 * Component selection and cross-validation schemes (ftir_11 §8, ftir_20, ftir_23, ftir_32; ftir_31 §1.5 redraws
 * ftir_23). The notebooks' decision and verdict tables come from gallery/data/ml_pages/components.py; the
 * site-grouped curves and the rule k under each protocol are drawn from calibration.json and calibration_runs.json,
 * which the Calibration tab already loads. RMSECV is in µg per filter against IMPROVE TOR and does not depend on the
 * MAC; only the Addis slopes do, and they follow the MAC set under Filters → units.
 */
export function Components(_props: SectionPageProps) {
  const { data, error } = useMl('components')
  const { calib } = useCalib()
  const { runs } = useCalibRuns()
  const m = useMac()

  // explorer: the rule k for every cohort × spectra × protocol (calibration.json grid)
  const rule = useMemo(() => (calib ? ruleRows(calib) : []), [calib])
  // explorer: rule k against cohort size under each protocol (sizes swept for the three selected cohorts)
  const bySize = useMemo(() => !calib ? [] : calib.grid
    .filter((r) => r.tg === 'addis' && r.lot === 'all' && r.el === 'all' && (r.sel ?? 'raw') === 'raw' && r.cut != null && ['ocec', 'eth_shaped', 'analogs'].includes(r.co))
    .map((r): Row => ({ cohort: COHORT_NAME[r.co].replace(/ \(\d+\)$/, ''), spectra: SPECTRA_NAME[r.sp] ?? r.sp, protocol: protocolOf(r.mo), cut: r.cut, k: r.ak ?? r.k }))
    .sort((a, b) => (a.cut as number) - (b.cut as number)), [calib])
  // ftir_11 [17b]: site-grouped RMSECV at each k for the 400 / 800 / 1600 lowest-OC/EC cohorts (calibration.json sweeps)
  const ocecCurves = useMemo(() => !calib ? [] : (calib.sweeps as (SweepRow & { rmsecv?: number | null })[])
    .filter((s) => s.co === 'ocec' && (s.sel ?? 'raw') === 'raw' && [400, 800, 1600].includes(s.cut ?? -1) && s.rmsecv != null)
    .map((s): Row => ({ size: `N = ${s.cut}`, spectra: SPECTRA_NAME[s.sp] ?? s.sp, k: s.k, rmsecv: s.rmsecv ?? null, rule: s.ak }))
    .sort((a, b) => (a.k as number) - (b.k as number)), [calib])
  // ftir_20 [13], the site-grouped half: every explorer preset's CV curve as % of its mean training loading
  const presetCurves = useMemo(() => !runs ? [] : runs.runs.filter((r) => r.target === 'addis').flatMap((r) => {
    const preset = runs.presets.find((p) => p.key === r.preset)
    const loading = r.rmsecv_floor / (r.pct_rmsecv_floor / 100)
    return r.curve.filter((c) => c.rmsecv != null).map((c): Row => ({ calibration: preset ? presetName(preset) : r.preset, k: c.k, pct: 100 * (c.rmsecv as number) / loading, rule: r.auto_k }))
  }), [runs])

  if (!data) return <MlWaiting error={error} what="the component-selection tables" />

  const selection = rows(data, 'selection').map((r): Row => ({ ...r, protocol: protocolOf(r.mode), addis_slope: m.slope(num(r.addis_slope), 10) }))
  const curves20 = rows(data, 'rmsecv_curves')
  const curves23 = rows(data, 'selection_curves').map((r): Row => ({ ...r, protocol: protocolOf(r.mode) }))
  const chosen = (rs: Row[], series: string) => [...new Map(rs.map((r) => [String(r[series]), num(r.rule)])).entries()]
    .flatMap(([s, k]) => (k == null ? [] : [{ series: s, x: k }]))
  const ocecSpectra = SPECTRA_NAME.raw

  return <>
    <PageIntro id="ml-components-intro" title="Component selection and cross-validation — the number of PLS factors is a claim about a curve, and the curves differ" data={data}
      notebooks="research/ftir_ec_phase3/ftir_11_ocec_ratio_cohort §8, ftir_20_component_selection_across_setups, ftir_23_component_selection_by_protocol, ftir_32_cv_scheme_ablation; ftir_31 §1.5 redraws ftir_23">
      <p>Two choices decide k together: the cross-validation scheme (site-grouped 5-fold, whole IMPROVE sites held out, or the Calibration app's
        interleaved 10-fold, every 10th filter) and the stopping rule (the first major minimum within one standard error, or the first k within
        5 % of the minimum). The notebooks' tables are drawn as they ran, at MAC 10; the Addis slopes follow the MAC set under Filters → units
        (now {m.mac}). RMSECV is against IMPROVE TOR in µg per filter and does not depend on the MAC. One calibration's curve with its ±1 SE band,
        and what each k does at Addis, are on the <a href={CALIBRATION}>Calibration tab</a> (cross-validation curve and PLS-factor sweep). The June 2026 calibrations' component counts (ftir_ec_calibration_2026_06_25 05/06/09, CAL-0…CAL-7) are on the <a href="#tab=ml&amp;aep=june">June 2026 variants</a> page.</p>
    </PageIntro>

    <Figure kind="intervals" id="ml-components-k-rule" data={rows(data, 'k_by_rule')} row="cohort" series="rule" colors={RULE_COLOR}
      panels={[{ value: 'k', label: 'PLS factors chosen' }]} aggregate={{ stat: 'median', interval: 'none' }} labelWidth={290}
      title="k under each scheme and rule (ftir_20) — the two protocols disagree on k by up to 3×, and not always in the same direction"
      subtitle="Each setup's full cohort, raw spectra, under each scheme × rule. The app protocol (interleaved + within 5 %) takes 17–27 factors on four setups where phase 3's (site-grouped + first major minimum) takes 6–10; on Ethiopia-shaped smoke the order flips. The locked k (green) was chosen on each cohort's training split, so it need not equal the full-cohort phase-3 k."
      provenance={provenance(data, ['k_by_rule'])} />

    <Figure kind="intervals" id="ml-components-floors" data={rows(data, 'scheme_summary')} row="cohort" series="cv_scheme" colors={SCHEME_COLOR}
      choose={rows(data, 'scheme_summary').some((r) => r.preprocessing !== 'raw') ? [{ column: 'preprocessing', label: 'Spectra', default: 'raw' }] : undefined}
      panels={[{ value: 'pct_rmsecv_min', label: 'Lowest RMSECV, % of mean TOR EC loading' }, { value: 'k_within_5pct', label: 'k, within 5 %' }, { value: 'k_first_major_minimum', label: 'k, first major minimum' }]}
      aggregate={{ stat: 'median', interval: 'none' }} labelWidth={290}
      title="Error floor by CV scheme (ftir_20) — interleaved folds lower the error floor most where a cohort leans on its own sites"
      subtitle="The lowest point of each curve, as a share of the cohort's mean loading, under each scheme, and the k each rule reads off that same curve. Interleaved folds let filters from one IMPROVE site sit on both sides of a split: the smoke floor drops from 155 % to 98 %, while the OC/EC cohorts barely move."
      provenance={provenance(data, ['scheme_summary'])} />

    <Figure kind="intervals" id="ml-components-optimism" data={rows(data, 'optimism')} row="cohort"
      panels={[{ value: 'optimism_ratio', label: 'Site-grouped ÷ interleaved floor', reference: 1 }]} aggregate={{ stat: 'median', interval: 'none' }} labelWidth={290}
      title="Interleaving optimism (ftir_20) — the leakage penalty indicts the smoke lineage: ×1.59 for smoke, ×1.41 Ethiopia-shaped, ×1.01–1.07 elsewhere"
      subtitle="How much of each cohort's apparent skill under interleaved folds depends on having seen its own sites in training (raw spectra). Spectral analogs are the one cohort that does better when whole sites are held out."
      provenance={provenance(data, ['optimism'])} />

    {curves20.length > 0 && <Figure kind="sweep" id="ml-components-curves20" data={curves20} x="n_components" y="pct_rmsecv" series="cv_scheme" facet="cohort" colors={SCHEME_COLOR}
      choose={[{ column: 'preprocessing', label: 'Spectra', default: 'raw' }]} xLabel="PLS factors" yLabel="RMSECV, % of mean loading" height={200}
      title="RMSECV curves, both schemes (ftir_20 [13]) — the RMSECV curves under both schemes, one panel per setup"
      subtitle="Red interleaved 10-fold, blue site-grouped 5-fold, as the notebook drew them (its output/tables, read after a re-run)."
      provenance={provenance(data, ['rmsecv_curves'])} />}

    {presetCurves.length > 0 && <Figure kind="sweep" id="ml-components-preset-curves" data={presetCurves} x="k" y="pct" series="calibration"
      chosen={chosen(presetCurves, 'calibration')} xLabel="PLS factors" yLabel="Site-grouped RMSECV, % of mean training loading" height={340}
      title="Site-grouped CV curves, explorer calibrations — the composition cohorts flatten after 5–10 factors, the pool and smoke curves stay jagged above 100 %"
      subtitle="ftir_20 [13]'s site-grouped half, from the explorer's own runs (site-grouped 5-fold on each cohort's training split, the curve that chose k; ringed: the first major minimum). Each curve is divided by its cohort's mean TOR loading so the cohorts compare, as ftir_20 does. Toggle calibrations in the legend; one curve with its ±1 SE band is on the Calibration tab. The interleaved curves are in no gallery export (see the list above)."
      provenance={<>Read from <code>calibration_runs.json</code> runs[].curve, rmsecv_floor and pct_rmsecv_floor (target Addis; the curve does not depend on the target).</>} />}

    {ocecCurves.length > 0 && <Figure kind="sweep" id="ml-components-ocec-sizes" data={ocecCurves} x="k" y="rmsecv" series="size" colors={CUT_COLOR}
      choose={[{ column: 'spectra', label: 'Spectra', default: ocecSpectra }]} chosen={chosen(ocecCurves, 'size')} xLabel="PLS factors" yLabel="Site-grouped RMSECV (µg / filter)" height={300}
      title="RMSECV by OC/EC cohort size (ftir_11 [17b]) — the 400, 800 and 1600 lowest-OC/EC cohorts stop at 3, 6 and 9 factors"
      subtitle="Site-grouped RMSECV against k for the three cohort sizes ftir_11 compared, from the explorer's sweeps (ringed: each cohort's rule k). The explorer swept every k only for N = 800; for 400 and 1600 it holds the k values around the rule, so those lines are short. The cohort-size sweep at every N from 100 to 2000 is on the Calibration tab."
      provenance={<>Read from <code>calibration.json</code> sweeps (cohort ocec, selected on raw spectra; rmsecv and ak).</>} />}

    <Figure kind="table" id="ml-components-ocec-test" data={rows(data, 'ocec_sizes')}
      columns={[{ key: 'model', label: 'Cohort' }, { key: 'n_components', label: 'k', format: 'int' }, { key: 'n_test', label: 'Held-out filters', format: 'int' },
        { key: 'test_sites', label: 'Held-out sites', format: 'int' }, { key: 'slope', label: 'TOR test slope', format: 'f2' }, { key: 'R2', label: 'TOR test R²', format: 'f3' }, { key: 'RMSE', label: 'TOR test RMSE (µg/filter)', format: 'f2' }]}
      title="TOR test by cohort size (ftir_11) — at N = 800 the site-held-out TOR test passes (R² 0.91); 400 and 1600, and five random 800s, do worse"
      subtitle="Each cohort's own whole-site holdout (20 % of sites), at the k its site-grouped curve chose. A random 800 from the same pool keeps k = 6 and reaches R² 0.59–0.77, so the OC/EC selection, not the size, carries the test."
      provenance={provenance(data, ['ocec_sizes'])} />

    <Figure kind="intervals" id="ml-components-selection" data={selection} row="cohort" series="protocol" colors={PROTOCOL_COLOR}
      panels={[{ value: 'k', label: 'k chosen' }, { value: 'global_min_k', label: 'k at the curve’s minimum' }, { value: 'addis_intercept', label: 'Addis intercept (µg/m³)', reference: 0 },
        { value: 'addis_slope', label: `Addis slope vs Fabs/${m.mac}`, reference: 1 }]}
      aggregate={{ stat: 'median', interval: 'none' }} labelWidth={290}
      title="What each protocol selected (ftir_23) — the site-held-out rule stops at 4–15 factors, far short of its curve's minimum; the app rule stops at 9–27, and the Addis slope follows"
      subtitle={`Each protocol's own curve on the cohort's training split, its rule's choice, where that curve actually bottoms out, and what the chosen model does on the fixed 190 Addis filters (OLS against HIPS Fabs/${m.mac}${m.changed ? `; slopes re-expressed from the notebook's MAC 10` : ''}). Both rules stop early on purpose: the site-grouped ±1 SE band spans several µg per filter on the pool and smoke cohorts, so its minimum is not a statement the data support.`}
      provenance={provenance(data, ['selection'])} />

    {curves23.length > 0 && <Figure kind="sweep" id="ml-components-curves23" data={curves23} x="n_components" y="rmsecv" series="protocol" facet="cohort" colors={PROTOCOL_COLOR}
      xLabel="PLS factors" yLabel="RMSECV (µg / filter)" height={200}
      title="Selection curves (ftir_23 [9]) — each protocol's curve, one panel per cohort"
      subtitle="The curves the decisions above were read from (the notebook's output/tables, read after a re-run)."
      provenance={provenance(data, ['selection_curves'])} />}

    <Figure kind="intervals" id="ml-components-2x2" data={rows(data, 'scheme_2x2')} row="cohort" series="scheme" colors={SCHEME_COLOR}
      panels={[{ value: 'floor_pct_rmsecv', label: 'Lowest RMSECV, % of mean loading' }, { value: 'k_within_5pct', label: 'k, within 5 %' }, { value: 'k_global_min', label: 'k at the minimum' }]}
      aggregate={{ stat: 'median', interval: 'none' }} labelWidth={240}
      title="CV-scheme 2 × 2 (ftir_32) — the app protocol's optimism is interleaving, not its 10 folds: interleaved 5-fold matches 10-fold"
      subtitle="The CV-scheme 2 × 2 on the three cohorts the leak argument rests on. Interleaved 5- and 10-fold floors agree within 4 %; the site-grouped floor moves with the split count where sites are concentrated (smoke 155 % at 5 splits, 86 % at 10), and so does its k (AIRSpec within-5 % k 5 → 18). Quote every k with its full scheme, folds included."
      provenance={provenance(data, ['scheme_2x2'])} />

    <Figure kind="heatmap" id="ml-components-2x2-verdict" data={rows(data, 'verdict_2x2')} row="cohort" column="ratio" value="value" ramp="diverging" domain={[0.4, 1.6]} format="f2"
      valueLabel="ratio of RMSECV floors (1 = no effect)" cell={[150, 30]}
      title="2 × 2 verdict (ftir_32) — fold count does nothing to the interleaved floor; grouping matters only at 5 folds"
      subtitle="Each ratio divides one scheme's floor by another's. Count effect, interleaved 5 / 10: 1.00–1.04. Grouping effect at 5 folds: ×1.53 for smoke, ~1 for the OC/EC cohorts."
      provenance={provenance(data, ['verdict_2x2'])} />

    {rule.length > 0 && <Figure kind="intervals" id="ml-components-explorer-k" data={rule} row="cohort" series="protocol" colors={PROTOCOL_COLOR}
      choose={[{ column: 'spectra', label: 'Spectra', default: 'raw' }]}
      panels={[{ value: 'k', label: 'Rule k (PLS factors)' }, { value: 'rmsecv', label: 'RMSECV at the rule k (µg / filter)' }]}
      aggregate={{ stat: 'median', interval: 'none' }} labelWidth={240}
      title="Rule k by protocol, explorer — on raw spectra interleaved CV takes 11–27 factors where site-grouped CV takes 4–15"
      subtitle="Every cohort at its default size, selected on raw spectra, the training split's curve under each protocol. B2 keeps the app's interleaved folds but reads the curve with phase 3's first-major-minimum rule, so the gap between B and B2 is the rule and the gap between B2 and A is the folds. The same configurations' Addis intercepts and slopes are on the Protocols page."
      provenance={<>Read from <code>calibration.json</code> grid (target Addis, every lot; ak, rmsecv) — the explorer's batch results, the same rows as the Calibration tab. ftir_51 names the protocols A, B and B2.</>} />}

    {bySize.length > 0 && <Figure kind="sweep" id="ml-components-k-size" data={bySize} x="cut" y="k" series="protocol" facet="cohort" colors={PROTOCOL_COLOR}
      choose={[{ column: 'spectra', label: 'Spectra', default: SPECTRA_NAME.airspec }]} xLabel="Cohort size (IMPROVE filters)" yLabel="Rule k" height={220}
      title="Rule k against cohort size — on spline-baselined OC/EC cohorts the first-major-minimum rule holds k at 3–7 under either fold scheme; the within-5 % rule jumps between 3 and 19"
      subtitle="The explorer's rule k at every cohort size it ran under each protocol (spline-baselined spectra were swept at every size under all three; raw and second-derivative spectra under the interleaved protocols at five sizes only). Elsewhere k jumps from one size to the next under every protocol: a rule k is not a smooth function of the cohort, so a single k quoted for a cohort says little without its curve."
      provenance={<>Read from <code>calibration.json</code> grid (target Addis, every lot, selected on raw spectra).</>} />}
  </>
}
