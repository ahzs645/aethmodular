import { useMemo } from 'react'
import { Figure } from '@/components/Figure'
import { f2 } from '@/lib/format'
import { useMac } from '@/lib/mac'
import { num, type Row } from '@/lib/table'
import { INK } from '@/lib/theme'
import type { SectionPageProps } from '../../types'
import { MlWaiting, PageIntro, provenance, rows, useMl } from '../common'

const TARGET = 'target: slope 1, intercept 0'
const s = (v: unknown) => String(v ?? '')
const pct0 = (v: unknown) => (typeof v === 'number' ? `${v.toFixed(0)} %` : '—')
const last = <T,>(a: T[]): T | undefined => a[a.length - 1]
const median = (v: number[]) => { const a = [...v].sort((p, q) => p - q), n = a.length; return n ? (n % 2 ? a[(n - 1) / 2] : (a[n / 2 - 1] + a[n / 2]) / 2) : NaN }

/**
 * The June 2026 calibration notebooks' model-fitting results that Research → Ethiopia calibrations
 * does not show: CAL-0…CAL-7 on the 906 IMPROVE filters (research/ftir_ec_calibration_2026_06_25 05, 06,
 * 09_component_selection_fixed, 10_calibration_variants_effect), the ETAD-vs-HIPS line against (1, 0)
 * (11, 13), choosing the calibration data (14), and calibration stability (spartan_ec_2026_06_16 05, 07, 08).
 * Static: tracked tables and the executed notebooks' printed results (gallery/data/ml_pages/june.py).
 * Everything on the 906 filters and at Adama is TOR EC and does not depend on the MAC. The HIPS fits
 * (11, 13, 14) are OLS on Fabs/10: slopes follow the reader's MAC (× MAC/10, lib/mac), intercepts and R²
 * do not, and the (1, 0) target stays slope 1 at the reader's MAC, so its distance is recomputed.
 */
export function JuneVariants(_props: SectionPageProps) {
  const { data, error } = useMl('june')
  const m = useMac()

  const d = useMemo(() => {
    if (!data) return null
    const variants = rows(data, 'variants')
    const label = (r: Row) => `${r.variant} ${r.name} (n = ${r.n})`
    const role = (r: Row) => (r.role === 'candidate' ? 'Candidate calibrations' : 'Diagnostics: small or inverse training sets')
    const vrows = variants.map((r): Row => ({ ...r, label: label(r), block: role(r) }))
    const kRules: [string, string][] = [['k first min (06)', '06 · first major minimum (5-fold, 2 %)'], ['k Wold (06)', "06 · Wold's R"], ['k* (09/10)', '09/10 · k* (10-fold interleaved, 5 %)']]
    const kLong = vrows.flatMap((r) => kRules.map(([col, rule]): Row => ({ label: r.label, block: r.block, rule, k: num(r[col]) })))
    const effect = vrows.filter((r) => num(r['median predicted EC on the 906 (10, µg)']) != null)
    const cal0 = num(effect.find((r) => r.variant === 'CAL-0')?.['median predicted EC on the 906 (10, µg)'])

    // HIPS fits: slopes at the reader's MAC; the flag and distance to (1, 0) recomputed there
    const at = (r: Row): Row => {
      const slope = m.slope(num(r.slope), 10), b = num(r.intercept)
      return { ...r, slope, distance: slope != null && b != null ? Math.hypot(slope - 1, b) : null }
    }
    const target: Row = { label: TARGET, family: TARGET, 'training set': TARGET, slope: 1, intercept: 0 }
    const sweep = rows(data, 'hips_sweep').map((r): Row => {
      const a = at(r), sl = num(a.slope)
      return { ...a, label: `${r['training set']}, ${r.setting}`, flag: sl == null ? '' : sl < 0.5 ? 'flat / degenerate' : sl >= 0.75 && sl <= 1.3 ? 'near 1:1' : '—' }
    })
    const lot: Row[] = [
      ...rows(data, 'frontier').map((r): Row => ({ ...at(r), n: num(rows(data, 'selection').find((x) => x.selection === 'full set')?.n), family: r.correction === 'raw' ? 'raw spectra, full set (the reference)' : '13 · spectral correction', label: s(r.correction) })),
      ...rows(data, 'subsets').filter((r) => r.selection !== 'full set').map((r): Row => ({ ...at(r), family: s(r.family), label: s(r.selection) })),
      ...rows(data, 'selection').filter((r) => r.selection !== 'full set').map((r): Row => ({ ...at(r), family: '14 · designed subset', label: s(r.selection) })),
    ].sort((a, b) => (num(a.distance) ?? 9) - (num(b.distance) ?? 9))

    // cleaning: filters kept after each round, from round 0 (all 906)
    const runs = rows(data, 'cleaning_runs')
    const kept: Row[] = []
    for (const run of [...new Set(runs.map((r) => s(r.run)))]) {
      const rs = runs.filter((r) => r.run === run)
      kept.push({ run, round: 0, kept: num(rs[0]?.['n before']), threshold: null })
      for (const r of rs) kept.push({ run, round: r.round, kept: (num(r['n before']) ?? NaN) - (num(r.dropped) ?? NaN), threshold: num(r['3σ threshold (µg)']) })
    }
    const finals = [...new Set(kept.map((r) => s(r.run)))].map((run) => num(last(kept.filter((r) => r.run === run))?.kept)).filter((v): v is number => v != null)

    const adama = rows(data, 'adama')
    const byCal = [...new Set(adama.map((r) => s(r.calibration)))].map((c) => {
      const rs = adama.filter((r) => r.calibration === c)
      const ratio = rs.map((r) => (num(r.EC) ?? NaN) / (num(r['TOR EC']) ?? NaN)).filter(Number.isFinite)
      return { c, above: rs.filter((r) => (num(r.EC) ?? -Infinity) > (num(r['TOR EC']) ?? Infinity)).length, n: rs.length, ratio: median(ratio) }
    })
    return { vrows, kLong, effect, cal0, sweep, target, lot, kept, finals, byCal }
  }, [data, m])

  if (!data || !d) return <MlWaiting error={error} what="the June 2026 calibration variants" />

  const picks = rows(data, 'rule_picks')
  const raw5 = picks.filter((r) => s(r.cv).startsWith('05') && r.spectra === 'raw' && num(r.k) != null && num(r.k)! > 1)
  const kAt = (rule: string, spectra = 'raw') => num(picks.find((r) => s(r.rule).includes(rule) && r.spectra === spectra)?.k)
  const vr = (v: string) => d.vrows.find((r) => r.variant === v)
  const ratios = d.effect.map((r) => num(r['ratio to CAL-0 (10)'])).filter((v): v is number => v != null)
  const best = d.lot[0]
  const near = d.sweep.filter((r) => r.flag === 'near 1:1').map((r) => num(r.intercept)).filter((v): v is number => v != null)
  const ec50 = d.sweep.find((r) => r.label === 'train EC≤50, k=10'), cal010 = d.sweep.find((r) => r.label === 'CAL-0 all, k=10')
  const checks = rows(data, 'checks'), check = (w: string) => num(checks.find((r) => s(r.measure).startsWith(w))?.value)
  const loso = check('leave-one-site-out'), rand = check('random 5-fold')
  const appr = rows(data, 'approaches'), fits = rows(data, 'cleaning_fits')
  const june06 = rows(data, 'june06_clean')[0]
  const general = d.byCal.find((c) => c.c.startsWith('general')), biomass = d.byCal.filter((c) => c.c.startsWith('biomass'))
  const slopeLabel = `Slope vs HIPS Fabs/${m.mac} (OLS)`
  const macNote = m.changed ? `The notebooks fitted against Fabs/10; slopes here are × ${m.mac}/10, the target is slope 1 at MAC ${m.mac} (slope ${f2(10 / m.mac)} in the notebooks' Fabs/10 frame), intercepts and R² do not move.` : 'Fitted against HIPS Fabs/10 (MAC 10), the MAC set under Filters → units; another MAC rescales the slopes and moves the target.'

  return <>
    <PageIntro id="june-intro" title="June 2026 calibration variants and chasing the origin" data={data}
      notebooks="research/ftir_ec_calibration_2026_06_25 05_calibration_variants, 06_calibration_variants_components, 09_component_selection_fixed, 09_ethiopia_component_ec (panel A), 10_calibration_variants_effect, 11_hips_intercept_calibrations, 13_chase_origin_and_biomass, 14_choosing_calibration_data; research/spartan_ec_2026_06_16 05_calibration_optimization, 07_calibration_optimization_story, 08_outlier_removal_explainer">
      <p>The June notebooks were committed without their figures: their tables are tracked and their printed results are in the executed notebooks, and that is all this page draws. Nothing is refitted. The 906 IMPROVE filters (lot 251, TOR EC per filter) and the Adama filters are thermal-EC comparisons and do not depend on the MAC. The ETAD-vs-HIPS fits (notebooks 11, 13, 14) do: their slopes follow the MAC set under Filters → units (now {m.mac}).</p>
      <p>What these calibrations predict for the 319 ETAD spectra (by sample set, component count and filter lot, and against HIPS) is on <a href="#tab=research&aep=ethiopia-june">Research → Ethiopia calibrations (Jun 2026)</a>, with spartan_ec 08's removed filters. It is not repeated here.</p>
    </PageIntro>

    {/* ------------------------------------------------------------ CAL-0…CAL-7 on the 906 IMPROVE filters */}
    <Figure kind="sweep" id="june-rmsecv" data={rows(data, 'rmsecv_curve')} x="k" y="RMSECV" xLabel="PLS components (k)" yLabel="RMSECV (µg per filter)" height={260}
      xMarkers={[...new Set(raw5.map((r) => num(r.k)!))].map((k) => ({ at: k, label: `k = ${k}: ${raw5.filter((r) => r.k === k).map((r) => (s(r.rule).includes('5 %') ? '5 % rule' : s(r.rule).includes('2 %') ? '2 % rule' : 'minimum')).join(', ')}` }))}
      title={`CAL-0's cross-validation curve rises before it falls and is jagged, so the tolerance alone picks k = ${kAt('within 5 %') ?? '?'} or k = ${kAt('2 % tolerance') ?? '?'}`}
      subtitle="Notebook 05: RMSECV of PLS on all 906 IMPROVE filters (5-fold shuffled CV, the predecessor's _rmsep_by_ncomp), k = 1–30. The first deep dip at k = 13 is within 5 % of the k = 21 minimum, so a 2 % and a 5 % first-minimum rule disagree; the notebooks therefore fixed one rule for every variant. Notebook 09_ethiopia_component_ec recomputed this same curve (same folds and seed; its panel A) to pick k = 21 and build its ladder: 'Mark' shows those counts, whose ETAD EC is on Research → Ethiopia calibrations (k = 40 lies past this curve's range). TOR EC; no MAC."
      variants={[{ key: 'mark', label: 'Mark', choices: [{ value: 'rules', label: "05/06's rule picks" },
        { value: 'ladder', label: "09's component ladder", patch: { xMarkers: [3, 10, 18, 21, 25].map((k) => ({ at: k, label: k === 21 ? 'k = 21 (09)' : `k = ${k}` })) } }] }]}
      provenance={provenance(data, ['rmsecv_curve', 'rule_picks'], 'research/ftir_ec_calibration_2026_06_25/05_calibration_variants.ipynb fig05; 09_ethiopia_component_ec.ipynb fig09 panel A.')} />

    <Figure kind="intervals" id="june-rules" data={picks.map((r): Row => ({ ...r, row: `${r.rule} · ${r.spectra}` }))} row="row" block="cv"
      panels={[{ value: 'k', label: 'Components picked (CAL-0)' }, { value: 'RMSECV', label: 'Minimum of that RMSECV curve (µg per filter)' }]} aggregate={{ stat: 'median', interval: 'none' }} labelWidth={360}
      title={`On 2nd-derivative spectra CAL-0 reaches the same error with ${kAt('09 k*', '2nd derivative')}–${kAt('(06)', '2nd derivative')} components, against ${kAt('within 5 % of the minimum (06)')}–${kAt('2 % tolerance')} on raw spectra`}
      subtitle="Every rule the notebooks tried on CAL-0. 06 used the same 5-fold curve as 05; 09 replaced it with 10-fold interleaved CV (the app's R pls folds) and one 5 % tolerance. Wold's R needs a falling PRESS curve and stops at k = 1 on the raw curve, which rises first. Savitzky–Golay 2nd derivative, window 11, order 2 (Weakley's preprocessing, without his variable selection)."
      provenance={provenance(data, ['rule_picks'])} />

    <Figure kind="table" id="june-variant-list" data={d.vrows}
      columns={[{ key: 'variant', label: 'Variant' }, { key: 'name', label: 'Name' }, { key: 'rule', label: 'Training set' }, { key: 'role', label: 'Role' }, { key: 'n', label: 'n', format: 'int' }, { key: 'mean EC (µg)', label: 'Mean EC (µg)', format: 'f2' }, { key: 'k* (09/10)', label: 'k* (09/10)', format: 'int' }]}
      title="Eight calibrations from one set of 906 filters: keep everything, trim, keep only what is normally discarded, or cut by loading"
      subtitle={`06 defines CAL-0…CAL-5 from a baseline fit at k = 21 (the 3σ trim, its complement and the below-1:1 set); 10 adds CAL-6 (trim on the relative residual, fair to high-EC smoke filters) and CAL-7 (drop the 3 filters above 250 µg). CAL-5's band is a placeholder until the Ethiopia range is set. 05's first build: ${rows(data, 'built').map((r) => `${r.calib_id} (n = ${r.n_train}, RMSECV ${f2(r.RMSECV_at_k)} µg)`).join(', ')}.`}
      provenance={provenance(data, ['variants', 'built'])} />

    <Figure kind="sweep" id="june-retention" data={rows(data, 'retention')} x="EC_threshold_ug" y="n_retained" xLabel="Minimum measured EC kept (µg per filter)" yLabel="Filters kept (of 906)" height={220} xScale="log"
      title={`A ${rows(data, 'retention')[0]?.EC_threshold_ug ?? 70} µg floor keeps ${rows(data, 'retention')[0]?.n_retained ?? 22} of 906 filters, so CAL-4 is a diagnostic, not a calibration`}
      subtitle="Notebook 05: how many of the 906 filters survive each loading floor the meeting proposed. Its 10 µg variant (491 filters) is the smallest floor that leaves a usable set."
      provenance={provenance(data, ['retention'])} />

    <Figure kind="intervals" id="june-variant-fit" data={d.vrows} row="label" block="block"
      panels={[{ value: 'in-sample R² (06)', label: 'In-sample R² (06, k = 20)' }, { value: 'in-sample RMSE (06, µg)', label: 'In-sample RMSE (06, µg)', reference: 0 }, { value: '%RMSECV (09/10)', label: '%RMSECV at k* (09/10; RMSECV ÷ mean EC)', reference: 0 }]}
      aggregate={{ stat: 'median', interval: 'none' }} labelWidth={240}
      title={`CAL-1's 3σ trim fits best in sample (R² ${f2(vr('CAL-1')?.['in-sample R² (06)'])}) but cross-validates worse than keeping everything (%RMSECV ${pct0(vr('CAL-1')?.['%RMSECV (09/10)'])} vs ${pct0(vr('CAL-0')?.['%RMSECV (09/10)'])})`}
      subtitle="In-sample numbers (06, common k = 20; CAL-4 capped at 15) measure fit, not accuracy; %RMSECV (09 and 10, 10-fold interleaved, each variant at its own k*) is the comparison across variants. CAL-6 and CAL-7 were added in 10, so they have no 06 numbers. The diagnostics' small n makes their in-sample R² meaningless."
      provenance={provenance(data, ['variants'])} />

    <Figure kind="intervals" id="june-variant-k" data={d.kLong} row="label" block="block" series="rule"
      panels={[{ value: 'k', label: 'Components picked' }]} aggregate={{ stat: 'median', interval: 'none' }} labelWidth={240}
      colors={{ '06 · first major minimum (5-fold, 2 %)': '#4e79a7', "06 · Wold's R": '#bab0ac', '09/10 · k* (10-fold interleaved, 5 %)': '#f28e2b' }}
      title="Each variant's component count moves with the CV scheme (CAL-4 from 6 to 1, CAL-5 from 15 to 13), and Wold's R gives 1 almost everywhere"
      subtitle="The rules side by side, per variant (06's component table and 09/10's corrected k*). The trimmed and inverse sets pick k = 1: their curves never fall below the one-component error, which says the extra components fit noise for them."
      provenance={provenance(data, ['variants'])} />

    <Figure kind="intervals" id="june-effect" data={d.effect} row="label"
      panels={[{ value: 'median predicted EC on the 906 (10, µg)', label: 'Median predicted EC on the same 906 filters (µg)', reference: d.cal0 ?? undefined }, { value: '% negative on the 906 (10)', label: 'Filters predicted negative (%)', reference: 0 }]}
      aggregate={{ stat: 'median', interval: 'none' }} labelWidth={240}
      title={`Applied to the same 906 filters, the calibration choice moves the median EC by up to ${f2(Math.max(...ratios))}× CAL-0's: CAL-1's trim most, dropping the three extremes least`}
      subtitle="Notebook 10: each calibration, fitted on its own training set at its own k*, applied to all 906 filters (the reference line is CAL-0's median). CAL-5 and CAL-3 extrapolate on filters outside their training range: read them as sensitivity, not deployment. The per-filter distributions behind 10's box plot were not saved (see the intro)."
      provenance={provenance(data, ['variants'])} />

    <Figure kind="table" id="june-plan" data={rows(data, 'plan')}
      columns={[{ key: 'calib_id', label: 'calib_id (05)' }, { key: 'trainset', label: 'Training set' }, { key: 'filter', label: 'Filter' }, { key: 'purpose', label: 'Purpose' }]}
      title="Notebook 05's plan, before the CAL-x names: the meeting's experiments are about which samples the calibration sees"
      subtitle="The naming scheme (NAMING_SCHEME.md) the later notebooks replaced with CAL-0…CAL-7. The biomass (smoke-classifier) variant was never built: it waited on a classifier that notebook 13 later implemented."
      provenance={provenance(data, ['plan'])} />

    {/* ------------------------------------------------------------ chasing the origin (HIPS; MAC follows) */}
    <Figure kind="crossplot" id="june-sweep-plane" data={[...d.sweep, d.target]} x="slope" y="intercept" color="training set" idColumn="label" tipColumns={['training set', 'k', 'r2', 'flag']}
      xLabel={slopeLabel} yLabel="Intercept (µg/m³)" fit={false} sameScale={false} maxSide={440} colors={{ [TARGET]: INK.text }}
      title={near.length ? `Fewer components or a capped training range pull the ETAD intercept toward 0, but only by flattening the slope: the fits near slope 1 at ${m.label} still cross at ${f2(Math.max(...near))} to ${f2(Math.min(...near))} µg/m³` : `Fewer components or a capped training range pull the ETAD intercept toward 0, but only by flattening the slope: at ${m.label} no fit lands near slope 1`}
      subtitle={`Notebook 11: 14 calibrations trained on the 906 filters, applied to the ETAD spectra and regressed (OLS) on HIPS, n = 259 spectra. Fewer components flatten the model (k = 2 gives an intercept near 0 with slope ${f2(d.sweep.find((r) => r.label === 'CAL-0 all, k=2')?.slope)}, which says nothing); capping the training EC removes the few > 150 µg smoke filters that steepen it. ${macNote} The flag uses notebook 11's bands (slope < 0.5 flat, 0.75–1.3 near 1:1) at this MAC.`}
      provenance={provenance(data, ['hips_sweep'])} />

    <Figure kind="sweep" id="june-sweep-k" data={d.sweep} x="k" y="intercept" series="training set" xLabel="PLS components (k)" yLabel="Intercept vs HIPS (µg/m³)" reference={0} height={260}
      title={`The intercept follows the training range as much as k: at k = 10, EC ≤ 50 µg crosses at ${f2(ec50?.intercept)}, all 906 at ${f2(cal010?.intercept)} µg/m³`}
      subtitle="The same 14 fits, intercept only: it does not depend on the MAC (Fabs/MAC rescales x, not where the line crosses zero). ETAD's HIPS EC is a narrow 3.4–6.8 µg/m³ band (5–95 %), so a slope fitted through it is extrapolated a long way to reach x = 0."
      provenance={provenance(data, ['hips_sweep'])} />

    <Figure kind="intervals" id="june-offset" data={rows(data, 'offset_check').map((r): Row => ({ ...r, slope: m.slope(num(r.slope), 10) }))} row="calibration" series="spectra"
      panels={[{ value: 'slope', label: slopeLabel, reference: 1 }, { value: 'intercept', label: 'Intercept (µg/m³)', reference: 0 }]} aggregate={{ stat: 'median', interval: 'none' }} labelWidth={150}
      colors={{ raw: '#4e79a7', 'offset-corrected (3900–4000 cm⁻¹)': '#f28e2b' }}
      title="Subtracting the 3900–4000 cm⁻¹ baseline moves the intercept toward 0 only by flattening the slope as well"
      subtitle={`Notebook 11's exploratory check: the mean absorbance of the non-absorbing 3900–4000 cm⁻¹ window removed from every training and ETAD spectrum, then refitted. If the offset were a baseline mismatch the intercept would close at the same slope; it does not. ${m.changed ? `Slopes at MAC ${m.mac}.` : 'MAC 10.'}`}
      provenance={provenance(data, ['offset_check'])} />

    <Figure kind="crossplot" id="june-frontier" data={[...d.lot, { ...d.target }]} x="slope" y="intercept" color="family" idColumn="label" tipColumns={['family', 'r2', 'n']}
      xLabel={slopeLabel} yLabel="Intercept (µg/m³)" fit={false} sameScale={false} maxSide={460}
      colors={{ [TARGET]: INK.text, 'raw spectra, full set (the reference)': '#59a14f', '13 · spectral correction': '#4e79a7', '13 · organic-profile (biomass-like) subset': '#e15759', '14 · designed subset': '#f28e2b', '14 · provenance filter': '#b07aa1' }}
      title={`On the lot-248+251 calibration nothing reaches (1, 0): spectral corrections trade slope for intercept, and training-subset choices barely move the raw point or push it further away. Closest at ${m.label}: ${best?.label}, ${f2(best?.distance)} away`}
      subtitle={`Notebooks 13 and 14, all at k = 10 on the 13,012-filter lot-248+251 mirror, ETAD predictions regressed (OLS) on HIPS. 13's seven spectral corrections are its frontier (tables/hips_origin_frontier.csv); its organic-profile subsets and 14's designed subsets and provenance filter are printed fits. Dropping the Teflon CF band (1100–1300 cm⁻¹) holds the slope; SNV gets the intercept to ${f2(rows(data, 'frontier').find((r) => r.correction === 'SNV')?.intercept)} but under-reads. ${macNote}`}
      provenance={provenance(data, ['frontier', 'subsets', 'selection'])} />

    <Figure kind="table" id="june-frontier-table" data={d.lot}
      columns={[{ key: 'family', label: 'Change' }, { key: 'label', label: 'Calibration' }, { key: 'n', label: 'Training n', format: 'int' }, { key: 'slope', label: `Slope (${m.label})`, format: 'f2' }, { key: 'intercept', label: 'Intercept (µg/m³)', format: 'f2' }, { key: 'r2', label: 'R²', format: 'f2' }, { key: 'distance', label: `Distance to (1, 0) at ${m.label}`, format: 'f2' }]}
      title={`Ranked by distance to slope 1, intercept 0 at ${m.label}`}
      subtitle={`Distance = √((slope − 1)² + intercept²), recomputed at the reader's MAC (13 saved it at MAC 10). The full set itself (n = 13,012) is the reference row: slope ${f2(d.lot.find((r) => r.correction === 'raw')?.slope)}, intercept ${f2(d.lot.find((r) => r.correction === 'raw')?.intercept)}.`}
      provenance={provenance(data, ['frontier', 'subsets', 'selection'])} />

    <Figure kind="table" id="june-checks" data={checks}
      columns={[{ key: 'check', label: 'Check (14)' }, { key: 'measure', label: 'Measure' }, { key: 'value', label: 'Value', format: 'f2' }, { key: 'of', label: 'Of', format: 'int' }, { key: 'unit', label: 'Unit' }]}
      title={`ETAD sits inside the training domain and holding out whole sites costs ${loso != null && rand ? Math.round(100 * (loso / rand - 1)) : '?'} % RMSE, so the full lot-248+251 set is a defensible choice`}
      subtitle="Notebook 14's other checks on the lot-248+251 set (k = 10): filters whose comment history records a physical problem (dropping them barely moves the HIPS fit, above), leverage as whitened 10-component PCA distance to the training centre, and leave-one-site-out against random 5-fold CV on IMPROVE TOR EC (no MAC). EC-stratifying (316 filters) is the worst subset: it up-weights the rare high-EC filters that steepen the slope. The leverage histogram was not saved (see the intro)."
      provenance={provenance(data, ['checks'])} />

    {/* ------------------------------------------------------------ calibration stability (spartan_ec 05/07/08) */}
    <Figure kind="sweep" id="june-cleaning" data={d.kept} x="round" y="kept" series="run" xLabel="Cleaning round" yLabel="Filters kept (of 906)" height={260}
      variants={[{ key: 'show', label: 'Show', choices: [{ value: 'kept', label: 'Filters kept' }, { value: 'threshold', label: '3σ threshold', patch: { y: 'threshold', yLabel: '3σ residual threshold (µg)' } }] }]}
      title={`The same 3σ rule keeps ${Math.min(...d.finals, num(june06?.kept) ?? Infinity)} to ${Math.max(...d.finals, num(june06?.kept) ?? -Infinity)} of 906 filters: K and when the loop stops decide how many go`}
      subtitle={`Fit, drop every filter whose |residual| exceeds 3 × the kept residuals' spread, refit (spartan_ec 05 at K = 21, five rounds at most; 07 at K = 18, six; 08 at K = 20, until a round drops none). June 06's CAL-1 is the same loop at k = ${june06?.k ?? 21} stopped after three rounds: ${june06?.kept ?? 860} kept, as 05 after its third. The threshold falls from about 31 µg to 8 µg as the heavy filters leave, so later rounds trim ordinary ones.`}
      provenance={provenance(data, ['cleaning_runs', 'june06_clean'])} />

    <Figure kind="intervals" id="june-cleaning-fits" data={fits.map((r): Row => ({ ...r, row: `${r.run} · ${r.stage} (n = ${r.n}, K = ${r.K})` }))} row="row" block="run"
      panels={[{ value: 'R²', label: 'In-sample R²' }, { value: 'RMSE (µg)', label: 'In-sample RMSE (µg per filter)', reference: 0 }]} aggregate={{ stat: 'median', interval: 'none' }} labelWidth={330}
      title={`Cleaning lifts the in-sample R² from about ${f2(median(fits.filter((r) => s(r.stage).startsWith('before')).map((r) => num(r['R²']) ?? NaN)))} to ${f2(median(fits.filter((r) => !s(r.stage).startsWith('before')).map((r) => num(r['R²']) ?? NaN)))} and cuts the RMSE about fourfold, scored on the filters it kept`}
      subtitle={`In-sample, so it flatters the cleaned fit by construction. The only cross-validated number: 05's CV-RMSEP goes ${f2(fits.find((r) => r.stage === 'before cleaning' && r.run === 'spartan_ec 05')?.['CV-RMSEP (µg)'])} → ${f2(fits.find((r) => r.run === 'spartan_ec 05' && r.stage !== 'before cleaning')?.['CV-RMSEP (µg)'])} µg, with K re-picked at 39 on the cleaned set. TOR EC; no MAC.`}
      provenance={provenance(data, ['cleaning_fits'])} />

    <Figure kind="table" id="june-k-stability" data={rows(data, 'k_stability')}
      columns={[{ key: 'question', label: 'spartan_ec 07' }, { key: 'low', label: 'Low', format: 'f2' }, { key: 'high', label: 'High', format: 'f2' }, { key: 'value', label: 'Typical', format: 'f2' }, { key: 'unit', label: 'Of' }]}
      title="One CV run's best K lands anywhere from 14 to 28; averaged over runs the minimum is a flat valley, and the Adama result swings with K"
      subtitle="Notebook 07's printed summaries of its K-instability figures (10-fold CV with different random fold splits on the 906-filter set; the biomass calibration applied to the five Adama filters and divided by the general lot-241a calibration). The curves themselves were saved only as figures (see the intro)."
      provenance={provenance(data, ['k_stability'])} />

    <Figure kind="intervals" id="june-approaches" data={appr} row="approach"
      panels={[{ value: 'median ratio', label: 'Median Adama biomass / general EC', reference: 1 }, { value: 'spread (max − min over the 5 filters)', label: 'Spread over the 5 filters (max − min)', reference: 0 }]}
      aggregate={{ stat: 'median', interval: 'none' }} labelWidth={220}
      title={`Cleaning and a K = 15–25 ensemble pull the Adama biomass/general ratio from ${f2(appr[0]?.['median ratio'])} to ${f2(last(appr)?.['median ratio'])}, but the filter-to-filter spread barely narrows (${f2(appr[0]?.['spread (max − min over the 5 filters)'])} → ${f2(last(appr)?.['spread (max − min over the 5 filters)'])})`}
      subtitle="Notebook 07: the three approaches it compared for stabilising the calibration. Five filters, so the spread is the honest measure of what the ensemble buys."
      provenance={provenance(data, ['approaches'])} />

    <Figure kind="crossplot" id="june-adama" data={rows(data, 'adama')} x="TOR EC" y="EC" color="calibration" idColumn="FilterId" tipColumns={['calibration', 'char/soot']}
      xLabel="Adama quartz TOR EC (µg/m³)" yLabel="FTIR EC (µg/m³)" fit={false} maxSide={420}
      colors={{ 'general (lot 241a)': '#4e79a7', 'biomass, before cleaning (05)': '#e15759', 'biomass, cleaned, K re-picked (05)': '#f28e2b', 'biomass, cleaned, K 15–25 ensemble (07)': '#59a14f' }}
      title={`At Adama the general calibration reads below TOR on ${general ? general.n - general.above : '?'} of ${general?.n ?? 5} filters and every biomass calibration above it on ${biomass.length && biomass.every((c) => c.above === c.n) ? 'all' : 'most of'} ${biomass[0]?.n ?? 5}; cleaning and ensembling shrink the overshoot from ${f2(biomass[0]?.ratio)}× to ${f2(last(biomass)?.ratio)}× TOR`}
      subtitle="spartan_ec 05 and 07's tracked tables: the five Adama filters under the general SPARTAN calibration and the 906-filter IMPROVE (biomass, lot 251) calibration at each stage of 05's and 07's optimisation. Hover a filter for its Han char/soot ratio. Five filters, a lot mismatch (calibration lot 251, Adama lot 245): read them, do not fit them. TOR EC; no MAC."
      provenance={provenance(data, ['adama'])} />
  </>
}
