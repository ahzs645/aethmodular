import type { MeetingSpec } from './types'

// "Which VIBES setting": the investigation of 16 Nov 2026, step by step. Which solver, rank and tau to use for new
// work, what each test measured and the choice it led to. Tables from
// gallery/data/export_meeting_20261116_vibes_settings.py (it reads the saved runs and fits nothing). Titles carry the
// message: section text shows only with Explain on.
const FROZEN = 'EC is µg per filter, read with the frozen k 4 IMPROVE model. That model was trained on spectra corrected at rank 70 and tau 0.1, so a change here is the cost of a mismatch, not a ranking of settings.'
const REBUILT = 'The calibration is rebuilt at each tau, so each tau is scored with its own model. Train split without one QC-failed spectrum (step 6).'
const HIPS = 'HIPS is optical and reads low on dark filters, so the Addis slope is a comparison, not a score against truth.'

/** the k the frozen selection rule picks, as the rings on the cross-validation curve (table "rule"; lib/vibesSettings.test.mjs) */
export const RULE_PICKS = {
  whole: { 'tau 0.05': 11, 'tau 0.1': 4, 'tau 0.2': 4 },
  qc: { 'tau 0.05': 8, 'tau 0.1': 4, 'tau 0.2': 3 },
}
const rings = (picks: Record<string, number>) => Object.entries(picks).map(([series, x]) => ({ series, x }))

export default {
  id: 'vibes-settings-20261116',
  label: 'Nov 16, 2026',
  date: '2026-11-16',
  short: 'VIBES settings',
  title: 'Which VIBES setting: the solver, the rank and tau, step by step',
  command: 'uv run python gallery/data/export_meeting_20261116_vibes_settings.py',
  intro: '@intro',
  sections: [
    {
      id: 'glance',
      title: 'The investigation at a glance: eight steps, each with its question, its finding and the choice',
      figures: [
        {
          kind: 'table', id: 'vs-glance', table: 'glance', title: 'Converged, rank 70, tau 0.1: what each step asked, found and settled',
          subtitle: 'Each row is a section below.',
          columns: [{ key: 'step', label: 'Step' }, { key: 'question', label: 'Question' }, { key: 'did', label: 'What was done' }, { key: 'found', label: 'What it found' }, { key: 'choice', label: 'Choice' }],
        },
        {
          kind: 'table', id: 'vs-terms', table: 'terms', title: 'The words on this page',
          columns: [{ key: 'term', label: 'Term' }, { key: 'meaning', label: 'What it is' }, { key: 'value', label: 'Used here' }],
        },
      ],
    },
    {
      id: 'solver',
      title: 'Step 1. The solver: converged changes EC on the 97 scans where the October run stopped early, and nowhere else',
      text: '@solver',
      figures: [
        {
          kind: 'table', id: 'vs-solver-groups', table: 'solver_groups', title: 'EC, October run against converged: unchanged on 97 % of PM2.5 scans, a median 5 % on the early stops',
          subtitle: 'Same background, rank 70 and tau 0.1 in both runs; only the solver differs. Percent changes use scans with EC of at least 1 µg.',
          columns: [{ key: 'group', label: 'Scans' }, { key: 'scans', label: 'Count', format: 'int' }, { key: 'median_abs_ug', label: 'Median |change| (µg)', format: 'coef' },
            { key: 'median_abs_pct', label: 'Median |change| (%)', format: 'f2' }, { key: 'largest_ug', label: 'Largest (µg)', format: 'f2' },
            { key: 'down', label: 'Moved down', format: 'int' }, { key: 'up', label: 'Moved up', format: 'int' }],
          noteAlways: true, note: FROZEN,
        },
        {
          kind: 'crossplot', id: 'vs-solver-scans', table: 'solver_scans', x: 'october', y: 'converged', color: 'group', idColumn: 'filter',
          tipColumns: ['analysis', 'site', 'change (%)', 'October steps'], fit: false, sameScale: true, maxSide: 520,
          xLabel: 'EC with the October baseline (µg)', yLabel: 'EC with the converged baseline (µg)',
          title: 'The 97 early stops: most move down, and all ten Addis filters move down',
          subtitle: 'One point per PM2.5 scan the October solver left within 30 steps. On the diagonal nothing changes.',
          noteAlways: true, note: 'The Addis early stops are flatter in a featureless window after converging (3 Nov tests). That their lower EC is more correct is not shown: 6 of 10 land closer to the site line.',
        },
        {
          kind: 'table', id: 'vs-solver-example', table: 'solver_example', title: 'One filter, CHTS-0830: the early-stop scan moves 3 %; the filter\'s other scan reads nearly three times higher under either solver',
          subtitle: 'Which scan of a filter is read can matter more than the solver. The gallery reads analysis 1975 (lab review of repeat scans).',
          columns: [{ key: 'scan', label: 'Scan' }, { key: 'early_stop', label: 'Early stop in October' }, { key: 'october_steps', label: 'October steps', format: 'int' },
            { key: 'october', label: 'EC, October (µg)', format: 'f2' }, { key: 'converged', label: 'EC, converged (µg)', format: 'f2' }, { key: 'change_pct', label: 'Change (%)', format: 'f2' }],
        },
      ],
    },
    {
      id: 'rank',
      title: 'Step 2. The rank: 40 gives the same EC as 70; only ranks 20 and 10 move it',
      text: '@rank',
      figures: [
        {
          kind: 'heatmap', id: 'vs-grid', table: 'settings', row: 'rank', column: 'tau', value: 'median_abs_pct', ramp: 'sequential', domain: [0, 5], format: 'f2',
          valueLabel: 'median |EC change| from rank 70, tau 0.1 (%)', cell: [112, 30],
          title: 'Every rank and tau against the network setting: the rank matters only below 40, tau a little everywhere',
          subtitle: 'Median EC change over 3,626 PM2.5 scans, each refitted to convergence at that setting. All 4,376 scans are certified at all 12 settings.',
          noteAlways: true, note: FROZEN,
        },
        {
          kind: 'table', id: 'vs-by-rank', table: 'by_rank', title: 'By rank at tau 0.1: field blanks spread out as the rank falls; repeat scans agree the same',
          subtitle: 'Blank spread: interquartile range of EC over 535 field blanks, which should all read alike. Repeat gap: median difference between two scans of one filter, 148 filters.',
          columns: [{ key: 'rank', label: 'Rank' }, { key: 'median_abs_pct', label: 'Median |EC change| (%)', format: 'f2' }, { key: 'moved_gt10_pct', label: 'Scans moved > 10 % (%)', format: 'f2' },
            { key: 'largest_ug', label: 'Largest (µg)', format: 'f2' }, { key: 'blank_iqr_ug', label: 'Blank spread (µg)', format: 'coef' }, { key: 'repeat_gap_ug', label: 'Repeat gap (µg)', format: 'coef' }],
        },
        {
          kind: 'sweep', id: 'vs-injection', table: 'injection', where: { height: '0.001' }, x: 'fwhm', y: 'median', series: 'rank', facet: 'group', xScale: 'log', reference: 1,
          xLabel: 'Width of the injected band (cm⁻¹)', yLabel: 'Share of the band left',
          title: 'Why: a band added to a real scan is kept the same at ranks 20, 40 and 70, unless the scan is a blank',
          subtitle: 'A faint band (height 0.001) of known width added to 40 scans per group and refitted (10 Nov test). 1 means the baseline left it alone.',
          noteAlways: true, note: 'A band about 1,500 cm⁻¹ wide is removed at every rank. That is the continuum loss, and it belongs to the background, not to the rank.',
        },
      ],
    },
    {
      id: 'tau-frozen',
      title: 'Step 3. tau with the frozen calibration: EC moves 1 to 2 %, because spectrum and calibration no longer match',
      text: '@tau_fixed',
      figures: [
        {
          kind: 'table', id: 'vs-by-tau', table: 'by_tau', title: 'By tau at rank 70: every scan moves a little, and a few move a lot',
          columns: [{ key: 'tau', label: 'tau' }, { key: 'median_abs_pct', label: 'Median |EC change| (%)', format: 'f2' }, { key: 'moved_gt10_pct', label: 'Scans moved > 10 % (%)', format: 'f2' },
            { key: 'largest_ug', label: 'Largest (µg)', format: 'f2' }, { key: 'addis_median_change_ug', label: 'Addis, median change (µg)', format: 'f2' }, { key: 'addis_slope', label: 'Addis slope against tau 0.1', format: 'f2' },
            { key: 'repeat_gap_ug', label: 'Repeat gap (µg)', format: 'coef' }, { key: 'repeat_gap_p90_ug', label: 'Repeat gap, 90th pct (µg)', format: 'f2' }],
          noteAlways: true, note: FROZEN,
        },
        {
          kind: 'bars', id: 'vs-noise', table: 'by_tau', label: 'tau', value: 'noise_e4', color: 'tau', format: 'f2', xLabel: 'Median fitted noise level, PM2.5 scans (×10⁻⁴ absorbance)',
          title: 'The fitted noise level follows tau: it roughly doubles when tau doubles',
          subtitle: 'So a noise level only compares between fits made at one tau.',
        },
      ],
    },
    {
      id: 'tau-rebuilt',
      title: 'Step 4. tau with the calibration rebuilt: no tau is better at every k',
      text: '@tau_refit',
      figures: [
        {
          kind: 'sweep', id: 'vs-heldout', table: 'heldout', where: { training: 'Without the QC-failed spectrum' }, x: 'k', y: 'RMSE', series: 'tau',
          title: 'Held-out IMPROVE error against k: the three tau curves cross, and all three fall with k',
          subtitle: '2,327 held-out IMPROVE filters against thermal EC. The gap between k 4 and k 11 is several times the gap between tau values.',
          xLabel: 'PLS components (k)', yLabel: 'Held-out RMSE (µg)', xMarkers: [{ at: 4, label: 'frozen model' }], height: 320,
          variants: [{ key: 'score', label: 'Score', default: 'RMSE', choices: [
            { value: 'RMSE', patch: { y: 'RMSE', yLabel: 'Held-out RMSE (µg)' } },
            { value: 'Median |error|', patch: { y: 'median_abs', yLabel: 'Held-out median |error| (µg)' } },
            { value: 'Mean |error|', patch: { y: 'MAE', yLabel: 'Held-out mean |error| (µg)' } }] }],
          noteAlways: true, note: REBUILT,
        },
        {
          kind: 'intervals', id: 'vs-boot', table: 'boot', row: 'row', series: 'tau', block: 'score', labelWidth: 120,
          panels: [{ value: 'difference', label: 'Error minus the error at tau 0.1 (µg)', reference: 0, lo: 'lo', hi: 'hi' }],
          title: 'Each tau against tau 0.1 on the same filters: tau 0.05 is never better; tau 0.2 is better at low k and worse at high k',
          subtitle: 'Left of zero: a smaller error than tau 0.1. Bars: 90 % interval with sites resampled 2,000 times. A bar that crosses zero is no different.',
          noteAlways: true, note: REBUILT,
        },
        {
          kind: 'table', id: 'vs-heldout-fixed', table: 'heldout_fixed', title: 'Held-out RMSE (µg) at four fixed k',
          columns: [{ key: 'k', label: 'k', format: 'int' }, { key: 'tau 0.05', label: 'tau 0.05', format: 'f2' }, { key: 'tau 0.1', label: 'tau 0.1', format: 'f2' }, { key: 'tau 0.2', label: 'tau 0.2', format: 'f2' }],
        },
        {
          kind: 'sweep', id: 'vs-addis', table: 'transfer', x: 'k', y: 'addis_slope', series: 'tau', reference: 1, height: 320,
          title: 'Addis against HIPS: the slope follows k, from 0.4 at k 4 to 1.8 at k 18, and the three tau values stay together',
          subtitle: 'A model fitted on all eligible IMPROVE filters at each tau, applied to 233 Addis spectra. HIPS BC is Fabs × volume ÷ 10.',
          xLabel: 'PLS components (k)', yLabel: 'Slope of FTIR EC on HIPS BC, Addis',
          variants: [{ key: 'read', label: 'Read', default: 'Slope', choices: [
            { value: 'Slope', patch: { y: 'addis_slope', yLabel: 'Slope of FTIR EC on HIPS BC, Addis', reference: 1 } },
            { value: 'Intercept', patch: { y: 'addis_intercept', yLabel: 'Intercept of FTIR EC on HIPS BC, Addis (µg)', reference: 0 } },
            { value: 'IMPROVE out-of-site error', patch: { y: 'oos_median_abs', yLabel: 'IMPROVE out-of-site median |error| (µg)', reference: undefined } }] }],
          noteAlways: true, note: HIPS,
        },
      ],
    },
    {
      id: 'k-rule',
      title: 'Step 5. The rule that picks k is not stable: it picks 4, 11, 8 or 3, depending on tau and on one spectrum',
      text: '@krule',
      figures: [
        {
          kind: 'table', id: 'vs-rule', table: 'rule', title: 'The k the frozen selection rule picks, and the held-out RMSE at that k',
          subtitle: 'The lower RMSE at tau 0.05 comes from the higher k, not from tau: at one k the three agree (step 4).',
          columns: [{ key: 'training', label: 'Train split' }, { key: 'tau 0.05', label: 'k at tau 0.05', format: 'int' }, { key: 'tau 0.1', label: 'k at tau 0.1', format: 'int' }, { key: 'tau 0.2', label: 'k at tau 0.2', format: 'int' },
            { key: 'RMSE at that k, tau 0.05', label: 'RMSE, tau 0.05', format: 'f2' }, { key: 'RMSE at that k, tau 0.1', label: 'RMSE, tau 0.1', format: 'f2' }, { key: 'RMSE at that k, tau 0.2', label: 'RMSE, tau 0.2', format: 'f2' }],
        },
        {
          kind: 'sweep', id: 'vs-cv', table: 'cv', x: 'k', y: 'rmse_whole', series: 'tau', chosen: rings(RULE_PICKS.whole), height: 320,
          xLabel: 'PLS components (k)', yLabel: 'Cross-validated RMSE on the train split (µg)',
          variants: [{ key: 'train', label: 'Train split', default: 'Whole', choices: [
            { value: 'Whole', label: 'Whole (frozen procedure)', patch: { y: 'rmse_whole', chosen: rings(RULE_PICKS.whole) } },
            { value: 'No QC-failed spectrum', patch: { y: 'rmse_qc', chosen: rings(RULE_PICKS.qc) } }] }],
          title: 'The curve the rule reads: an early dip near k 4 and a later one near k 8 to 11 are close in depth, so small changes swap them',
          subtitle: 'Site-grouped 5-fold cross-validation. Ring: the k the rule picks (the earliest dip within one standard error of the lowest).',
          noteAlways: true, note: 'Compare settings at a fixed k. A comparison at the k the rule picks compares k.',
        },
      ],
    },
    {
      id: 'qc-spectrum',
      title: 'Step 6. One QC-failed training spectrum: IMPROVE scores do not change, but SPARTAN EC from the k 4 model moves by a median 17 %',
      text: '@qc',
      figures: [
        {
          kind: 'table', id: 'vs-qc-groups', table: 'qc_groups', title: 'The k 4 model with and without IMPROVE filter 2008808, on every network scan: PM2.5 EC moves by a median 17 %, field blanks from 1.27 to 0.30 µg',
          subtitle: 'Both models are fitted on the train split of the converged tau 0.1 spectra; one training spectrum of 10,066 differs. Percent changes use scans reading at least 1 µg.',
          columns: [{ key: 'group', label: 'Scans' }, { key: 'scans', label: 'Count', format: 'int' }, { key: 'median_with', label: 'Median EC with it (µg)', format: 'f2' }, { key: 'median_without', label: 'without it (µg)', format: 'f2' },
            { key: 'median_abs_pct', label: 'Median |change| (%)', format: 'f2' }, { key: 'moved_gt10_pct', label: 'Scans moved > 10 % (%)', format: 'f2' }, { key: 'largest_ug', label: 'Largest (µg)', format: 'f2' }],
          noteAlways: true, note: 'The model with the spectrum is the 3 Nov rebuilt k 4 model, which matches the frozen model (coefficients r = 0.999998). Neither model has been checked against thermal EC on SPARTAN filters.',
        },
        {
          kind: 'bars', id: 'vs-qc-sites', table: 'qc_sites', label: 'site', value: 'change_pct', color: 'direction', format: 'f2', reference: 0,
          xLabel: 'Change in the site\'s median PM2.5 EC when the spectrum is left out (%)',
          title: 'By site: medians move from −64 % (PRFJ) to +51 % (ILHA); Addis (ETAD) moves 2 %',
          subtitle: 'The order of sites changes with one training spectrum, so a site-to-site comparison of EC from the k 4 model is not safe.',
        },
        {
          kind: 'table', id: 'vs-qc-checks', table: 'qc_checks', title: 'Which model is right? IMPROVE cannot tell them apart, and the other checks disagree',
          columns: [{ key: 'check', label: 'Check' }, { key: 'with', label: 'With the spectrum', format: 'coef' }, { key: 'without', label: 'Without it', format: 'coef' }, { key: 'reading', label: 'Reading' }],
          noteAlways: true, note: HIPS + ' A field blank should read near zero, but the blank level wanders with k for both models, so it does not settle this either.',
        },
        {
          kind: 'sweep', id: 'vs-qc-by-k', table: 'qc_by_k', x: 'k', y: 'median_abs_pct', series: 'scans', height: 300, xMarkers: [{ at: 4, label: 'frozen model' }],
          xLabel: 'PLS components (k)', yLabel: 'Median |EC change| (%)',
          title: 'The pull is strongest at low k: 17 % at k 4, 8 % at k 8, 3 % at k 12',
          subtitle: 'The same with-and-without comparison at every k, on SPARTAN PM2.5 scans and on IMPROVE\'s own held-out filters.',
        },
        {
          kind: 'table', id: 'vs-qc-leverage', table: 'qc_leverage', title: 'It is 364 times farther from the mean than a typical training spectrum; the next most extreme are heavy filters that passed QC, and one of them alone moves network EC by up to 6 %',
          subtitle: 'The train spectra farthest from the mean spectrum. Last column: network PM2.5 EC change from leaving that one spectrum out (for the QC-passed ones, out of the split that already lacks the failed one).',
          columns: [{ key: 'filter', label: 'IMPROVE filter' }, { key: 'site', label: 'Site' }, { key: 'raw_qc', label: 'Raw QC' }, { key: 'tor_ec_ug', label: 'Thermal EC (µg)', format: 'f2' },
            { key: 'distance', label: 'Distance from the mean (× typical)', format: 'f2' }, { key: 'network_median_abs_pct', label: 'Network median |EC change| (%)', format: 'f2' }],
        },
        {
          kind: 'table', id: 'vs-qc', table: 'qc_effect', title: 'The 3 Nov transfer table was made with it in training: without it the Addis slope at k 5 is 0.61, not 0.41',
          subtitle: 'tau 0.1, rank 70, converged in both; models fitted on all eligible IMPROVE filters. From k 12 up the two agree.',
          columns: [{ key: 'k', label: 'k', format: 'int' }, { key: 'slope_with', label: 'Addis slope, with it', format: 'f2' }, { key: 'slope_without', label: 'without it', format: 'f2' },
            { key: 'intercept_with', label: 'Addis intercept (µg), with it', format: 'f2' }, { key: 'intercept_without', label: 'without it', format: 'f2' },
            { key: 'error_with', label: 'IMPROVE out-of-site median |error| (µg), with it', format: 'f2' }, { key: 'error_without', label: 'without it', format: 'f2' }],
          noteAlways: true, note: HIPS,
        },
      ],
    },
    {
      id: 'training-set',
      title: 'Step 7. Training without the heaviest filters: one spectrum stops mattering, but the cut-off itself moves EC, and the Addis slope with it',
      text: '@domain',
      figures: [
        {
          kind: 'table', id: 'vs-domain', table: 'domain', choose: [{ column: 'k', label: 'PLS components', default: 'k 4' }],
          title: 'Six training sets: trimming cuts the pull of one spectrum from 6 % to 0.4 %, but EC still spreads 8 % when IMPROVE sites are resampled',
          subtitle: 'One spectrum: the largest network EC change from dropping any one of the set\'s six most extreme spectra. Spread: the standard deviation of EC over 30 refits on IMPROVE sites drawn with replacement, for a scan (median scan) and for a SPARTAN site\'s median.',
          columns: [{ key: 'training', label: 'Training set' }, { key: 'train_spectra', label: 'Spectra', format: 'int' }, { key: 'heaviest_ug', label: 'Heaviest (µg)', format: 'f2' },
            { key: 'one_spectrum_max_pct', label: 'One spectrum (%)', format: 'f2' }, { key: 'scan_spread_pct', label: 'Scan spread (%)', format: 'f2' },
            { key: 'site_spread_median_pct', label: 'Site spread, median (%)', format: 'f2' }, { key: 'site_spread_max_pct', label: 'worst site (%)', format: 'f2' }, { key: 'site_spread_max_site', label: 'Worst site' },
            { key: 'heldout_rmse_below25', label: 'Held-out RMSE below 25 µg', format: 'f2' }, { key: 'addis_slope', label: 'Addis slope on HIPS', format: 'f2' },
            { key: 'blank_median', label: 'Blank (µg)', format: 'f2' }, { key: 'change_from_all_pct', label: 'EC change from "all" (%)', format: 'f2' }],
          noteAlways: true, note: 'No network PM2.5 scan reads above 50 µg and 1.5 % read above 25 µg, so the heaviest training filters lie outside SPARTAN\'s range. ' + HIPS,
        },
        {
          kind: 'heatmap', id: 'vs-domain-pairs', table: 'domain_pairs', row: 'training', column: 'against', value: 'median_abs_pct', ramp: 'sequential', domain: [0, 30], format: 'f2',
          valueLabel: 'median |EC difference| between two training sets, network PM2.5 scans (%)', cell: [104, 30],
          title: 'Each training set gives a different network EC: the k 4 models are 3 to 30 % apart',
          subtitle: 'Row against column, as a percent of the row\'s EC. Trimming is a choice that moves EC, not only a clean-up.',
        },
        {
          kind: 'sweep', id: 'vs-domain-addis', table: 'domain_addis', x: 'k', y: 'addis_slope', lo: 'slope_lo', hi: 'slope_hi', series: 'training', reference: 1, height: 360,
          title: 'With the training set held fixed the Addis slope still follows k; the set trained below 25 µg reads about half the slope at every k and never reaches 1',
          subtitle: 'One model per k for each training set, applied to 233 Addis spectra. Click a legend entry to hide a set. The k where the slope reaches 1 is 8 to 11 for five sets, so "the k that matches HIPS" is not a property of k alone.',
          bandLabel: 'band: 5th to 95th percentile over 30 refits on IMPROVE sites drawn with replacement',
          xLabel: 'PLS components (k)', yLabel: 'Slope of FTIR EC on HIPS BC, Addis', xMarkers: [{ at: 4, label: 'frozen model' }],
          variants: [{ key: 'line', label: 'Read', default: 'Slope', choices: [
            { value: 'Slope', patch: { y: 'addis_slope', lo: 'slope_lo', hi: 'slope_hi', yLabel: 'Slope of FTIR EC on HIPS BC, Addis', reference: 1 } },
            { value: 'Intercept', patch: { y: 'addis_intercept', lo: 'intercept_lo', hi: 'intercept_hi', yLabel: 'Intercept of FTIR EC on HIPS BC, Addis (µg)', reference: 0 } },
            { value: 'Median EC', patch: { y: 'addis_median_ec', lo: undefined, hi: undefined, yLabel: 'Median FTIR EC of the Addis filters (µg)', reference: undefined } },
            { value: 'R²', patch: { y: 'addis_r2', lo: undefined, hi: undefined, yLabel: 'R² of FTIR EC against HIPS BC, Addis', reference: undefined } }] }],
          noteAlways: true, note: HIPS + ' R² stays between 0.6 and 0.8 from k 3 up under every set, so it cannot choose between them.',
        },
        {
          kind: 'bars', id: 'vs-domain-sites', table: 'domain_sites', label: 'site', value: 'spread_pct', format: 'f2', sort: 'desc', choose: [{ column: 'training', label: 'Training set', default: 'All QC-passed' }],
          xLabel: 'Spread of the site\'s median EC over refits on resampled IMPROVE sites (%)',
          title: 'Which sites are least pinned down: PRFJ spreads by 61 % with every spectrum in, and no site by more than 13 % below 25 µg',
          subtitle: 'k 4. Choose a training set.',
        },
      ],
    },
    {
      id: 'choices',
      title: 'What we chose, and what is still open',
      text: '@choices',
      figures: [
        {
          kind: 'table', id: 'vs-choices', table: 'choices', title: 'The solver, the rank and tau are settled; the calibration (its k and its training spectra) and the background are not',
          columns: [{ key: 'decision', label: 'Decision' }, { key: 'choice', label: 'Choice' }, { key: 'why', label: 'Why' }, { key: 'status', label: 'Status' }],
        },
      ],
    },
  ],
} satisfies MeetingSpec
