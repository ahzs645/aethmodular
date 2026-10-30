import { useMemo } from 'react'
import { ChartFrame } from '@/components/ChartFrame'
import { Crossplot, type CrossPoint } from '@/components/Crossplot'
import { Figure } from '@/components/Figure'
import { Key } from '@/components/Key'
import { f2 } from '@/lib/format'
import { useMac } from '@/lib/mac'
import { num, type Row } from '@/lib/table'
import type { SectionPageProps } from '../../types'
import { useCalibRuns } from '../../research/pages/spectralCommon'
import { MlWaiting, PageIntro, provenance, rows, useMl } from '../common'
import { ExcludedNote, FixedSetNote, useCountedFilter, useExclusionSplit } from '@/lib/markedRows'

const NOTEBOOKS = 'research/ftir_ec_phase3/ftir_43_residual_learner_null_control, ftir_45_residual_increment_attribution, ftir_46_paired_increment_and_mass (ftir_53 §5 re-draws its mass term)'

/** ftir_46's ladder rungs, bottom-up as the notebook draws them, in reader-facing words */
const RUNG: [string, string][] = [
  ['a  constant', 'a · constant offset'],
  ['b  f(X)', 'b · f(X) only'],
  ['c0 f(X)+season+volume (ftir_43 M2)', 'c0 · f(X) + season + volume'],
  ['c  c0+lot', 'c · c0 + lot (metadata base)'],
  ['c+C c+mass µg/m³', 'c+C · base + mass (µg/m³)'],
  ['c+L c+mass µg/filter', 'c+L · base + mass (µg/filter)'],
  ['e  spectra only (ftir_43 M3)', 'e · spectra only'],
  ['d  c+spectra (stacked)', 'd · base + spectra'],
  ['d+C c+C+spectra (stacked)', 'd+C · base + mass (µg/m³) + spectra'],
  ['d+L c+L+spectra (stacked)', 'd+L · base + mass (µg/filter) + spectra'],
]
const RUNG_LABEL = new Map(RUNG)
const RUNG_ORDER = new Map(RUNG.map(([k], i) => [k, i]))

/** ftir_46's null statistics: which measure each is, and what it compares */
const NULL_STAT: Record<string, [string, string]> = {
  R2_b: ['Residual R²', 'b · f(X) only'], R2_c: ['Residual R²', 'c · metadata base'], 'R2_c+': ['Residual R²', 'c+ · base + mass'],
  R2_d: ['Residual R²', 'd · base + spectra'], 'R2_d+': ['Residual R²', 'd+ · base + mass + spectra'], R2_e: ['Residual R²', 'e · spectra only'],
  dRMSE_d_c: ['ΔRMSE (µg/m³)', '(d)−(c) spectra beyond metadata'], 'dRMSE_d+_c+': ['ΔRMSE (µg/m³)', '(d+)−(c+) spectra beyond metadata + mass'],
  'dRMSE_c+_c': ['ΔRMSE (µg/m³)', '(c+)−(c) mass beyond metadata'], dRMSE_e_c: ['ΔRMSE (µg/m³)', '(e)−(c) spectra only vs metadata'],
  dR2_d_c: ['ΔR²', '(d)−(c) spectra beyond metadata'], 'dR2_d+_c+': ['ΔR²', '(d+)−(c+) spectra beyond metadata + mass'],
  'dR2_c+_c': ['ΔR²', '(c+)−(c) mass beyond metadata'], dR2_e_c: ['ΔR²', '(e)−(c) spectra only vs metadata'],
  folds_d_beats_c: ['Folds won (of 8)', '(d) beats (c)'],
}

const NO_ROWS: Row[] = []
const SITE_NAME: Record<string, string> = { addis: 'Addis Ababa', etbi: 'Bishoftu', chts: 'Beijing', indh: 'Delhi', uspa: 'JPL' }
const CONFIG_COLORS = { 'locked 800 + AIRSpec (rule k)': '#2C6E9E', 'winner 440 + AIRSpec k=8': '#e15759' }
const SERIES_COLORS = { blocked: '#2C6E9E', forward: '#f28e2b', observed: '#e15759', 'null (median, 5–95 %)': '#7f7f7f', 'null (median to q95)': '#7f7f7f' }

/** OLS y = a + b·x */
function ols(x: number[], y: number[]) {
  const n = x.length, mx = x.reduce((s, v) => s + v, 0) / n, my = y.reduce((s, v) => s + v, 0) / n
  let sxy = 0, sxx = 0
  for (let i = 0; i < n; i++) { sxy += (x[i] - mx) * (y[i] - my); sxx += (x[i] - mx) ** 2 }
  const b = sxy / sxx
  return { slope: b, intercept: my - b * mx }
}

/**
 * ftir_43, ftir_45, ftir_46: does a learner on the calibration residual find anything a no-chemistry null or the
 * metadata do not? Static: the residual learners were fitted on f(X) − Fabs/10, so their R² and RMSE stay at MAC 10.
 * What is exact at the reader's MAC moves: the residual-vs-f(X) panel (rebuilt from calibration_runs.json) and every
 * slope of the residual on mass at fixed f(X) (× 10/MAC). Filters → excluded acts on that panel only (by filter id):
 * the notebooks' tables are on their own fixed Addis set (FixedSetNote).
 */
export function Residual(_props: SectionPageProps) {
  const { data, error } = useMl('residual')
  const { runs } = useCalibRuns()
  const m = useMac()
  const at10 = m.changed ? ' (MAC 10)' : ''

  // ftir_43 [8a]: the locked 800 + AIRSpec prediction f(X) against its residual at Addis, from the explorer's readouts
  // (the same 239 filters and f(X): with every filter counted it reproduces the notebook's b = 0.802·f(X) + 2.851 at
  // MAC 10). Filters → excluded splits them by id: the fit is on the counted ones, the rest are × under Mark.
  const all = useMemo(() => {
    const run = runs?.runs.find((r) => r.preset === 'ocec800_airspec' && r.target === 'addis')
    if (!run) return null
    const pts: Row[] = []
    run.eval.pred.forEach((f, i) => {
      const b = m.fromFabs(run.eval.ref[i])
      if (b == null || !Number.isFinite(f)) return
      pts.push({ id: run.eval.id[i], season: run.eval.group[i], date: run.eval.date?.[i] ?? null, f, r: f - b, b })
    })
    return pts
  }, [runs, m])
  const idOf = (p: Row) => (p.id == null ? null : String(p.id))
  const split = useExclusionSplit(all ?? NO_ROWS, idOf)
  // the fit's filters by the one rule (lib/markedRows useCountedFilter, which also knows keep=)
  const isCounted = useCountedFilter()
  const live = useMemo(() => {
    if (!all) return null
    const pts = all.filter((p) => isCounted(idOf(p)))
    const fit = ols(pts.map((p) => p.f as number), pts.map((p) => p.b as number))
    return { pts, fit, marked: split.marked, nOut: all.length - pts.length, nAll: all.length }
  }, [all, split, isCounted]) // eslint-disable-line react-hooks/exhaustive-deps

  if (!data) return <MlWaiting error={error} what="the residual-learner export" />

  const models = rows(data, 'model_comparison')
  const noCorr = num(models.find((r) => String(r.model).startsWith('no correction'))?.HIPS_prediction_RMSE)
  const constant = num(models.find((r) => String(r.model).startsWith('M0'))?.HIPS_prediction_RMSE)
  const offsetShare = noCorr && constant ? Math.round(100 * (1 - constant / noCorr)) : null

  const null43 = rows(data, 'null_distribution_summary').flatMap((r): Row[] => [
    { pipeline: r.pipeline, series: 'null (median to q95)', mid: r.null_R2_median, lo: r.null_R2_median, hi: r.null_R2_q95 },
    { pipeline: r.pipeline, series: 'observed', mid: r.observed_R2, lo: null, hi: null },
  ]).map((r) => ({ ...r, pipeline: `${r.pipeline}  (p = ${f2(rows(data, 'null_distribution_summary').find((s) => s.pipeline === r.pipeline)?.p_null_ge_observed)})` }))

  const ladder = rows(data, 'ladder_summary').filter((r) => RUNG_ORDER.has(String(r.model)))
    .sort((p, q) => (RUNG_ORDER.get(String(p.model))! - RUNG_ORDER.get(String(q.model))!))
    .map((r): Row => ({ ...r, rung: RUNG_LABEL.get(String(r.model)) ?? String(r.model) }))
  const noCorrection = rows(data, 'ladder_summary').filter((r) => String(r.model).startsWith('0'))

  // folds won by each contrast under both schemes, in its label (one row per contrast, the schemes overlaid)
  const pairedRows = rows(data, 'paired_differences')
  const folds = (c: unknown) => pairedRows.filter((r) => r.contrast === c).map((r) => `${r.scheme} ${r.folds_hi_better}/${r.n_folds}`).join(', ')
  const paired = pairedRows.map((r): Row => ({ ...r, contrast: `${r.contrast}  (${folds(r.contrast)})` }))

  const nulls = rows(data, 'null_increment_summary').flatMap((r): Row[] => {
    const [measure, what] = NULL_STAT[String(r.stat)] ?? ['other', String(r.stat)]
    const label = `${what}  (p = ${num(r.p_value)?.toFixed(3) ?? '—'})`
    return [
      { measure, stat: label, series: 'null (median, 5–95 %)', mid: r.null_median, lo: r.null_q05, hi: r.null_q95 },
      { measure, stat: label, series: 'observed', mid: r.observed, lo: null, hi: null },
    ]
  })

  // residual (f(X) − Fabs/MAC) on mass at fixed f(X): −(coefficient of Fabs/MAC on mass), so × 10/MAC exactly; with no
  // f(X) in the design (the raw slope) it mixes f(X)'s own mass slope, which no export carries: MAC 10 only
  const partial = rows(data, 'addis_residual_vs_mass_partial_slopes').flatMap((r): Row[] => {
    const raw = String(r.conditioning).startsWith('none')
    if (raw && m.changed) return []
    const k = (v: unknown) => (raw ? num(v as number) : m.bc(num(v as number)))
    return [{ ...r, framing: String(r.framing), slope: k(r.slope), ci_lo: k(r.ci_lo), ci_hi: k(r.ci_hi), slope_x_IQR: k(r.slope_x_IQR) }]
  })

  // Fabs ~ f(X) + mass: per volume the coefficient is in Fabs units (Mm⁻¹ per µg/m³), MAC-free; per deposit it is
  // HIPS-EC-equivalent µg per µg mass (Fabs/10 × V/A), so × 10/MAC
  const sites = rows(data, 'cross_site_mass_slopes')
  const perVolume = sites.filter((r) => r.framing === 'per volume')
  const perDeposit = sites.filter((r) => r.framing === 'per deposit').map((r): Row => ({ ...r,
    ...Object.fromEntries(['mass_coef', 'mass_coef_lo', 'mass_coef_hi', 'mass_coef_addis_range', 'mass_coef_addis_range_lo', 'mass_coef_addis_range_hi']
      .map((c) => [c, m.bc(num(r[c]))])) }))
  const relation = rows(data, 'null_relation')[0]

  return <>
    <PageIntro id="ml-residual-intro" title="Residual learners and null controls — the spectra beat a no-chemistry null, not the metadata; the lead is mass" data={data} notebooks={NOTEBOOKS}>
      <FixedSetNote subject="this page" what="the notebooks' tables, on their own fixed Addis filters (the f(X) panel, rebuilt live, follows the switch)" />
      <p>A recurring proposal reads a model that predicts the FTIR − HIPS residual from the spectra as evidence of a missing absorber. The residual is r = f(X) − Fabs/MAC, and f(X) is itself a function of the spectra, so a learner can predict part of r by rebuilding f(X), with no chemistry at all. These notebooks run the controls on identical time-blocked folds (eight contiguous date blocks, so seasonal structure cannot leak). Nothing here is refitted: the learners were fitted on residuals at MAC 10 and their R² and RMSE are drawn as the notebooks printed them{m.changed ? <>, labelled MAC 10 while the reader's MAC is {m.mac}</> : ''}. Mass slopes of the residual at fixed f(X) follow the MAC exactly (× 10/MAC).</p>
    </PageIntro>

    {live && <ResidualVsFx pts={live.pts} marked={live.marked} nOut={live.nOut} nAll={live.nAll} slope={live.fit.slope} intercept={live.fit.intercept} mac={m.mac} changed={m.changed} relation={relation}
      provenance={<>Rebuilt live from <code>calibration_runs.json</code> (preset ocec800_airspec at Addis: the notebook's f(X), max |Δ| 5e-6 µg/m³ per ftir_46); follows the MAC. The notebook's relation: {provenance(data, ['null_relation'])}</>} />}

    <Figure kind="intervals" id="ml-residual-models" data={models} row="model"
      panels={[{ value: 'residual_R2', label: `Out-of-sample residual R²${at10}`, reference: 0 }, { value: 'HIPS_prediction_RMSE', label: `RMSE of the implied HIPS prediction (µg/m³)${at10}` }]}
      aggregate={{ stat: 'median', interval: 'none' }} labelWidth={240}
      title={`The constant offset does ${offsetShare ?? '~70'} % of the work; the spectra learner adds about 0.1 µg/m³ (ftir_43)`}
      subtitle="Five pipelines on the same eight time blocks: M0 the training-fold mean residual, M1 f(X) alone, M2 f(X) + season + volume, M3 a PLS residual learner on the corrected spectra, M4 spectra → HIPS directly. Residual R² is the number the external worklog quotes (0.32–0.40, not reproduced); RMSE of f(X) − r̂ against HIPS is the one deployment cares about. Predicting HIPS straight from the spectra (M4) is worse than any f(X)-anchored route."
      provenance={provenance(data, ['model_comparison'])} />

    <Figure kind="intervals" id="ml-residual-null43" data={null43} row="pipeline" series="series" colors={SERIES_COLORS}
      panels={[{ value: 'mid', lo: 'lo', hi: 'hi', label: `Residual R²${at10}`, reference: 0 }]} labelWidth={240}
      title="f(X) alone is indistinguishable from the no-chemistry null; the spectra learner clears it (ftir_43)"
      subtitle={`200 refits on simulated HIPS b* = a·f(X) + c + ε with a, c, σ matched to the observed relation (${relation ? `${f2(relation.a)}, ${f2(relation.c)}, ${f2(relation.sigma)}` : '—'}) and the spectra untouched: whatever such a learner finds is the proportional mechanism. Grey: the null's median to its 95th percentile (the notebook kept no other quantiles); red: observed. The draws themselves (the notebook's violin) are not saved.`}
      provenance={provenance(data, ['null_distribution_summary'])} />

    <Figure kind="intervals" id="ml-residual-ladder" data={ladder} row="rung" series="scheme" colors={SERIES_COLORS}
      panels={[{ value: 'pooled_RMSE', lo: 'rmse_ci_lo', hi: 'rmse_ci_hi', label: `Pooled RMSE of the HIPS prediction (µg/m³; block bootstrap 95 %)${at10}` },
        { value: 'pooled_R2', label: `Pooled residual R²${at10}`, reference: 0 }]} labelWidth={260}
      title="The ladder: only rungs with both mass and spectra pull clear of the metadata base (ftir_46)"
      subtitle={`Every rung on ftir_43's eight blocks (blocked, 239 filters) and forward-chained (each block predicted from earlier ones only, 179 filters); RMSE intervals resample whole blocks. The stacked rungs fit spectra on the base model's residual inside each training fold. Not drawn: no correction at all (f(X) as is), RMSE ${noCorrection.map((r) => `${f2(r.pooled_RMSE)} ${r.scheme}`).join(', ')}.`}
      provenance={provenance(data, ['ladder_summary'])} />

    <Figure kind="intervals" id="ml-residual-paired" data={paired} row="contrast" series="scheme" colors={SERIES_COLORS}
      panels={[{ value: 'pooled_dRMSE', lo: 'block_boot_ci_lo', hi: 'block_boot_ci_hi', label: `Paired ΔRMSE, higher rung − lower (µg/m³; block bootstrap 95 %)${at10}`, reference: 0 },
        { value: 'pooled_dR2', lo: 'dR2_block_ci_lo', hi: 'dR2_block_ci_hi', label: `Paired Δ residual R² (block bootstrap 95 %)${at10}`, reference: 0 }]} labelWidth={400}
      title="Spectra beyond metadata is not a paired result; with mass in the base it is (ftir_46)"
      subtitle="The test ftir_43 lacked: each contrast scored on the same filters and blocks, its interval from resampling whole blocks (folds won in brackets). Negative ΔRMSE favours the richer rung. Spectra over the metadata base: −0.06 with an interval across zero on both schemes. Mass alone buys about as much; mass and spectra together clear zero on both schemes, so the residual carries a mass × composition term."
      provenance={provenance(data, ['paired_differences'])} />

    <Figure kind="intervals" id="ml-residual-null46" data={nulls} row="stat" series="series" colors={SERIES_COLORS}
      choose={[{ column: 'measure', label: 'Statistic', default: 'ΔRMSE (µg/m³)' }]}
      panels={[{ value: 'mid', lo: 'lo', hi: 'hi', label: `Observed against 200 proportional-disagreement null draws${at10}`, reference: 0 }]} labelWidth={330}
      title="Every rung beyond f(X) beats the null, so the null cannot separate spectra from metadata (ftir_46)"
      subtitle="The same null as ftir_43, rerun through every rung and paired contrast. It answers a weaker question (could spectra with no information do this?) than the paired test above: the spectra-over-metadata increment clears it (0 of 200 draws) and still fails the pairing. Grey: the null's median and 5–95 %; red: observed; p in brackets (direction as the notebook tested it). The draws are written only by a re-run."
      provenance={provenance(data, ['null_increment_summary'])} />

    <Figure kind="intervals" id="ml-residual-stepwise" data={rows(data, 'stepwise_attribution')} row="model"
      panels={[{ value: 'residual_R2', label: `Out-of-sample residual R²${at10}`, reference: 0 }, { value: 'HIPS_RMSE', label: `RMSE of the HIPS prediction (µg/m³)${at10}` }]}
      aggregate={{ stat: 'median', interval: 'none' }} labelWidth={270}
      title="No named feature carries the increment: loading curvature, the char_06 flag and band heights add nothing (ftir_45)"
      subtitle="Stepwise linear models on ftir_43's folds. Neutral-baseline band heights (1617, O–H, C–H, C=O) overfit; all of them together fall below season + volume alone. The spectra learner's weight sits on band position and shape around 1500–1750 cm⁻¹ (its coefficient spectrum, written only by a re-run), which scalar heights cannot carry."
      provenance={provenance(data, ['stepwise_attribution'])} />

    <Figure kind="intervals" id="ml-residual-char06" data={rows(data, 'residual_by_char06_class').map((r): Row => ({ ...r, class: `${r.class} (n = ${r.n})` }))} row="class"
      panels={[{ value: 'residual_median', label: `Median residual f(X) − HIPS/10 (µg/m³)${at10}`, reference: 0 }, { value: 'adjusted_median', label: `Median after season + volume adjustment (µg/m³)${at10}`, reference: 0 }]}
      aggregate={{ stat: 'median', interval: 'none' }} labelWidth={200}
      title="The dry-season anomaly class is not where the offset hides (ftir_45)"
      subtitle="char_06's low-similarity filters carry the same offset as the rest; after the season adjustment both sit at zero. The per-filter histograms need each filter's char_06 class, which is written only by a re-run of char_06."
      provenance={provenance(data, ['residual_by_char06_class'])} />

    <Figure kind="intervals" id="ml-residual-mass-addis" data={partial} row="conditioning" block="framing"
      panels={[{ value: 'slope', lo: 'ci_lo', hi: 'ci_hi', label: `Residual slope on mass (µg EC per µg mass, ${m.label}; per-filter bootstrap 95 %)`, reference: 0 },
        { value: 'slope_x_IQR', label: `× mass IQR (µg/m³ or µg/cm², ${m.label})`, reference: 0 }]} labelWidth={260}
      title="At fixed f(X) the Addis residual falls with mass; the raw scatter hides it (ftir_46)"
      subtitle={`Mass and f(X) correlate at 0.84 at Addis, so the raw residual has no mass slope while the partial slope at fixed f(X) and season is clearly negative: heavier deposits read more HIPS per unit of FTIR EC. Per volume uses µg/m³; per deposit µg/cm² on the 3.53 cm² deposit area.${m.changed ? ` At ${m.label} the partial slopes are the notebook's × 10/${m.mac} (exact: f(X) is in the design); the raw slope needs f(X)'s own mass slope and is left out.` : ''} The scatter and added-variable plots need the notebook's per-filter mass, which only a re-run writes.`}
      provenance={provenance(data, ['addis_residual_vs_mass_partial_slopes'])} />

    <Figure kind="intervals" id="ml-residual-mass-sites" data={perVolume} row="site" series="config" colors={CONFIG_COLORS}
      panels={[{ value: 'mass_coef_addis_range', lo: 'mass_coef_addis_range_lo', hi: 'mass_coef_addis_range_hi', label: 'Mass coefficient inside the Addis loading range (Mm⁻¹ per µg/m³; bootstrap 95 %)', reference: 0 },
        { value: 'mass_coef', lo: 'mass_coef_lo', hi: 'mass_coef_hi', label: 'All filters (Mm⁻¹ per µg/m³)', reference: 0 }]} labelWidth={200}
      title="The mass term replicates in sign at every site but JPL, and Addis and Bishoftu carry two to three times the rest (ftir_46)"
      subtitle="Fabs ~ f(X) + mass per site with the locked 800 + AIRSpec calibration and the sweep winner (440, k = 8): the HIPS absorption added per µg/m³ of gravimetric mass at fixed FTIR EC. In Fabs units, so the MAC does not enter. Delhi's full-range coefficient is dominated by deposits up to 1700 µg/cm² where HIPS saturates; inside Addis's deposit range it is 0.31. JPL's mass IQR is about 5 µg/m³, so its near-zero term is underpowered. This is the panel ftir_53 §5 carries as committee backup (Research → Does the diagnosis generalise)."
      provenance={provenance(data, ['cross_site_mass_slopes', 'mass_term_ftir53'])} />

    <Figure kind="intervals" id="ml-residual-mass-deposit" data={perDeposit} row="site" series="config" colors={CONFIG_COLORS}
      panels={[{ value: 'mass_coef_addis_range', lo: 'mass_coef_addis_range_lo', hi: 'mass_coef_addis_range_hi', label: `µg HIPS-EC-equivalent per µg mass, Addis loading range (${m.label}; bootstrap 95 %)`, reference: 0 },
        { value: 'mass_coef', lo: 'mass_coef_lo', hi: 'mass_coef_hi', label: `All filters (${m.label})`, reference: 0 }]} labelWidth={200}
      title="Half of Addis's mass term is what HIPS does with any deposit; the other half is Ethiopian (ftir_46 per deposit)"
      subtitle={`The same fit per deposit (µg/cm²), the unit a scattering artifact lives in. The four non-Ethiopian sites pooled inside Addis's range give about half the Addis and Bishoftu coefficient. HIPS-EC-equivalent = Fabs/MAC, so these follow the MAC (× 10/${m.mac}).`}
      provenance={provenance(data, ['cross_site_mass_slopes'])} />

    <Figure kind="table" id="ml-residual-coverage" data={rows(data, 'cross_site_coverage').map((r): Row => ({ ...r, site: SITE_NAME[String(r.site)] ?? r.site }))} choose={[{ column: 'config', label: 'Calibration' }]}
      columns={[{ key: 'site', label: 'Site' }, { key: 'n', label: 'n', format: 'int' }, { key: 'k', label: 'k', format: 'int' }, { key: 'extrap_pct', label: '% outside training envelope', format: 'f2' },
        { key: 'volume_min', label: 'Volume min (m³)', format: 'f2' }, { key: 'volume_max', label: 'Volume max (m³)', format: 'f2' },
        { key: 'mass_conc_median', label: 'Mass median (µg/m³)', format: 'f2' }, { key: 'mass_area_median', label: 'Deposit median (µg/cm²)', format: 'f2' }, { key: 'mass_area_max', label: 'Deposit max (µg/cm²)', format: 'f2' },
        { key: 'n_in_addis_loading_range', label: 'In Addis deposit range', format: 'int' }, { key: 'corr_mass_fx', label: 'r(mass, f(X))', format: 'f2' }]}
      title="What each site's mass fit stands on: Delhi's deposits run to 16 times Addis's (ftir_46)"
      subtitle="The explorer's predictions at the five sites, joined to Fabs, volume and gravimetric mass. The Addis deposit range (16–107 µg/cm²) is the window the range-restricted coefficients use."
      provenance={provenance(data, ['cross_site_coverage'])} />
  </>
}

const SEASON_COLORS: Record<string, string> = { 'Dry (Oct-Feb)': '#4e79a7', 'Belg (Mar-May)': '#f28e2b', 'Kiremt (Jun-Sep)': '#59a14f' }
const NULL_LINE = '#b03a8c'

/**
 * ftir_43 [8a]: f(X) against its residual at the reader's MAC, with the proportional-disagreement null line
 * r = (1 − a)·f(X) − c from the OLS of HIPS/MAC on f(X). Drawn with that saved line rather than the reader's refit
 * estimators (a Deming refit of r on f(X) would answer another question).
 */
function ResidualVsFx({ pts, marked, nOut, nAll, slope, intercept, mac, changed, relation, provenance }: {
  pts: Row[]; marked: Row[]; nOut: number; nAll: number; slope: number; intercept: number; mac: number; changed: boolean; relation?: Row; provenance: React.ReactNode
}) {
  const toPoint = (p: Row): CrossPoint => ({ x: p.f as number, y: p.r as number, id: String(p.id), color: SEASON_COLORS[String(p.season)] ?? '#7f7f7f', season: String(p.season),
    tip: [String(p.id), `${p.season} · ${p.date ?? ''}`, `f(X): ${f2(p.f)} µg/m³`, `residual: ${f2(p.r)} µg/m³`] })
  const points = pts.map(toPoint)
  const a1 = 1 - slope
  return (
    <ChartFrame id="ml-residual-fx" title={`The residual tracks f(X) by construction: its ${a1.toFixed(2)}·f(X) slope needs no chemistry (ftir_43)`}
      subtitle={`The ${pts.length} Addis filters${nOut ? ` (${nOut} of the notebook's ${nAll} excluded under Filters → excluded)` : ''}. HIPS/${mac} = ${slope.toFixed(3)}·f(X) + ${intercept.toFixed(3)} (OLS)${(changed || nOut > 0) && relation ? `; the notebook's, on all ${nAll} at MAC 10: ${f2(relation.a)}·f(X) + ${f2(relation.c)}` : ''}. So even if the two methods disagree only proportionally, r = ${a1.toFixed(3)}·f(X) − ${intercept.toFixed(3)} − ε: the dashed line. A residual learner that reaches it has learned f(X), not a new absorber.`}
      provenance={provenance}>
      <Crossplot title="" points={points} marked={marked.map(toPoint)} fit={false} sameScale={false} maxSide={460} grouping="none"
        xLabel="f(X), locked 800 + AIRSpec prediction (µg/m³)" yLabel={`Residual f(X) − Fabs/${mac} (µg/m³)`}
        extraLines={[{ slope: a1, intercept: -intercept, color: NULL_LINE, label: 'proportional-disagreement null', dash: '6 4', method: 'ols', stats: null }]} />
      <ExcludedNote count={marked.length} drawn />
      <Key items={[...Object.entries(SEASON_COLORS).map(([label, color]) => ({ label, color })), { label: `null line r = ${a1.toFixed(3)}·f(X) − ${intercept.toFixed(3)}`, color: NULL_LINE, kind: 'dash' as const }]} />
    </ChartFrame>
  )
}
