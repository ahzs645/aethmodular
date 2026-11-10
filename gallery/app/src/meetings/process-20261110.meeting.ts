import type { MeetingSpec } from './types'

// "The calibration, step by step": the whole process in order, with what was done before, what is done now and why
// at each step. Every table comes from gallery/data/export_meeting_20261110_process.py, which reads the two committed
// result files. Titles carry the message: section text shows only with Explain on.
const IMPROVE_ONLY = 'IMPROVE only, against thermal (TOR) EC, with whole sites held out. This is the calibration set: it does not show how FTIR behaves on Addis aerosol.'
const ADDIS = 'No Addis filter has thermal EC. HIPS is optical (MAC 10), and no aethalometer data are used.'

export default {
  id: 'process-20261110',
  label: 'Nov 10, 2026',
  date: '2026-11-10',
  short: 'the calibration, step by step',
  title: 'The calibration, step by step: what we chose, why, what changed, and what comes out',
  command: 'uv run python gallery/data/export_meeting_20261110_process.py',
  intro: '@intro',
  sections: [
    {
      id: 'choices',
      title: 'The choices at a glance',
      figures: [
        {
          kind: 'table', id: 'p1110-choices', table: 'choices', title: 'Six steps: what was done before, what is done now, and why',
          columns: [{ key: 'step', label: 'Step' }, { key: 'before', label: 'Before' }, { key: 'now', label: 'Now' }, { key: 'why', label: 'Why' }],
        },
      ],
    },
    {
      id: 'set',
      title: '1. The calibration set: which filters, and why all of them',
      text: '@set',
      figures: [
        {
          kind: 'table', id: 'p1110-population', table: 'population', title: 'What the calibration is built on',
          columns: [{ key: 'what', label: '' }, { key: 'value', label: '' }],
        },
        {
          kind: 'scatter', id: 'p1110-sites', table: 'by_site', x: 'median_tor_ocec', y: 'median_tor_ec_ug', color: 'kind',
          idColumn: 'site', tipColumns: ['site', 'n', 'ftir_over_tor'], maxSide: 520,
          title: 'Where the calibration filters sit: mostly light and organic-rich, with six urban sites at the other end',
          subtitle: 'Each dot is an IMPROVE site: median TOR EC per filter against median TOR OC/EC. Addis filters carry about 30 µg of EC at an OC/EC near 1, beyond the top left of this chart.',
          xLabel: 'Site median TOR OC/EC', yLabel: 'Site median TOR EC (µg per filter)',
          noteAlways: true,
          note: 'IMPROVE is a mostly rural and remote network; its six urban sites are Atlanta, Birmingham, Fresno, Phoenix, Pittsburgh and Seattle.',
        },
        {
          kind: 'histogram', id: 'p1110-composition', table: 'ocec_values', value: 'ftir_ocec', series: 'population', bins: 40, medians: true,
          xLabel: 'FTIR OC ÷ FTIR EC (both from IMPROVE-trained calibrations)',
          title: 'Addis is outside the calibration set’s composition: 99 % of Addis filters are below IMPROVE’s 5th percentile',
          subtitle: 'OC and EC predicted by calibrations fitted the same way on both networks. IMPROVE values are from sites held out of the fit.',
        },
      ],
    },
    {
      id: 'model',
      title: '2. The model: components and cross-validation',
      text: '@model',
      columns: 2,
      figures: [
        {
          kind: 'sweep', id: 'p1110-cv', table: 'selection_cv', x: 'k', y: 'rmse', series: 'scheme',
          xLabel: 'PLS components', yLabel: 'Cross-validation RMSE (µg per filter)', xMarkers: [{ at: 18, label: 'used' }],
          title: 'Whole sites held out against interleaved folds: the error by number of components',
          subtitle: 'Raw spectra, every IMPROVE filter. Interleaved folds leave each test filter with neighbours from its own site in training.',
          noteAlways: true,
          note: IMPROVE_ONLY,
        },
        {
          kind: 'sweep', id: 'p1110-gain-k', table: 'gain_k', x: 'k', y: 'gain', series: 'variant', reference: 1,
          xLabel: 'PLS components', yLabel: 'Response per µg of thermal EC', xMarkers: [{ at: 18, label: 'used' }],
          title: 'The response to EC rises with the component count and levels off below 1',
          subtitle: 'Gain from IMPROVE site means, for each way of training. Cross-validation error barely sees this, which is why the count matters at Addis.',
        },
      ],
    },
    {
      id: 'score',
      title: '3. How it is scored: before and now',
      text: '@score',
      figures: [
        {
          kind: 'sweep', id: 'p1110-selection', table: 'selection_bias', x: 'ocec', y: 'ftir_over_tor', series: 'chosen_by', reference: 1,
          xLabel: 'OC/EC used to choose the filters (bin median)', yLabel: 'Median FTIR EC ÷ TOR EC',
          title: 'The same filters, opposite answers: it depends on whose OC/EC you bin on',
          subtitle: 'Held-out IMPROVE filters binned by TOR’s own OC/EC, by FTIR’s own, and by the previous sample at the site.',
          noteAlways: true,
          note: 'Each ratio contains the error of the EC it is built from, so a low-OC/EC bin collects filters where that EC happened to read high.',
        },
        {
          kind: 'intervals', id: 'p1110-level', table: 'level', row: 'filters', series: 'grouping',
          title: 'Chosen fairly, FTIR reads about a tenth low on the most Addis-like IMPROVE filters, not a quarter',
          subtitle: 'Median held-out FTIR EC ÷ TOR EC for three ways of choosing the filters. Bars: 95 %, resampling sites.',
          panels: [{ value: 'ftir_over_tor', label: 'FTIR EC ÷ TOR EC', reference: 1, lo: 'lo', hi: 'hi' }],
          labelWidth: 330,
        },
        {
          kind: 'intervals', id: 'p1110-candidates', table: 'candidates', row: 'calibration', series: 'chosen',
          title: 'Scored the old way the saved calibrations spread from 0.65 to 0.85; scored fairly, 0.82 to 0.92',
          subtitle: 'Median held-out FTIR EC ÷ TOR EC for each calibration the 7 Nov closeout saved.',
          panels: [{ value: 'ftir_over_tor', label: 'FTIR EC ÷ TOR EC', reference: 1 }],
          labelWidth: 250,
        },
      ],
    },
    {
      id: 'response',
      title: '4. What the calibration responds to, and the correction',
      text: '@response',
      figures: [
        {
          kind: 'intervals', id: 'p1110-variants', table: 'variants', row: 'variant', series: 'step',
          title: 'Training variants do not move the response to EC; a recalibration fitted on other sites does',
          subtitle: 'Raw spectra, whole IMPROVE sites held out. Heavy and light filters are chosen by the previous sample at the site (TOR EC ≥ 15 µg, and below 3 µg).',
          panels: [{ value: 'gain', label: 'Response per µg of thermal EC', reference: 1, lo: 'gain_lo', hi: 'gain_hi' },
            { value: 'urban_level', label: 'Six urban IMPROVE sites: FTIR ÷ TOR', reference: 1 },
            { value: 'heavy_level', label: 'Heavy filters: FTIR ÷ TOR', reference: 1 },
            { value: 'light_median_abs_error', label: 'Light filters: median error (µg)' }],
          labelWidth: 220,
          noteAlways: true,
          note: IMPROVE_ONLY,
        },
        {
          kind: 'table', id: 'p1110-variants-table', table: 'variants', title: 'Every variant, with the blank reading and the cost',
          columns: [{ key: 'calibration', label: 'Calibration' }, { key: 'gain', format: 'f2' }, { key: 'offset', label: 'offset (µg)', format: 'f2' },
            { key: 'blank_reading', label: 'blank reads (µg)', format: 'f2' }, { key: 'urban_level', label: 'urban', format: 'f2' }, { key: 'heavy_level', label: 'heavy', format: 'f2' },
            { key: 'light_level', label: 'light', format: 'f2' }, { key: 'light_median_abs_error', label: 'light error (µg)', format: 'f2' },
            { key: 'overall_median_abs_error', label: 'median error (µg)', format: 'f2' }, { key: 'overall_rmse', label: 'RMSE (µg)', format: 'f2' }],
        },
      ],
    },
    {
      id: 'addis',
      title: '5. The output on filters never in training: Addis',
      text: '@addis',
      figures: [
        {
          kind: 'crossplot', id: 'p1110-addis', table: 'addis_points', x: 'x', y: 'ftir_ec', color: 'season', facet: 'comparator',
          choose: [{ column: 'calibration', label: 'FTIR EC', default: 'Raw spectra, recalibrated on IMPROVE' }],
          fit: true, errorRatio: 2.96, sameScale: false, idColumn: 'filter', maxSide: 400,
          title: 'Addis, filter by filter: published, as trained, and recalibrated, against HIPS as measured and linearised',
          subtitle: 'Pick the FTIR EC version. Left: HIPS Fabs / 10 as reported. Right: after inverting the HIPS response fitted on IMPROVE thermal EC. Deming fits; seasons Bega Oct–Jan, Belg Feb–May, Kiremt Jun–Sep.',
          xLabel: 'HIPS optical BC (µg/m³, MAC 10)', yLabel: 'FTIR EC (µg/m³)',
          noteAlways: true,
          note: ADDIS,
        },
        {
          kind: 'table', id: 'p1110-addis-fits', table: 'addis_fits', title: 'The fits',
          columns: [{ key: 'calibration', label: 'FTIR EC' }, { key: 'comparator', label: 'Against' }, { key: 'n', format: 'int' }, { key: 'slope', format: 'f2' },
            { key: 'intercept', label: 'intercept (µg/m³)', format: 'f2' }, { key: 'median_ec', label: 'median FTIR EC (µg/m³)', format: 'f2' }],
        },
      ],
    },
    {
      id: 'baselines',
      title: '6. Baselines: raw, spline, VIBES and PTFE-anchored VIBES',
      text: '@baselines',
      figures: [
        {
          kind: 'intervals', id: 'p1110-baselines', table: 'baselines', row: 'preprocessing', series: 'step',
          title: 'No baseline changes the calibration’s response to thermal EC; the same recalibration works on each',
          subtitle: 'The same IMPROVE filters, 18 components, whole sites held out, for each baseline, as trained and after removing the offset and dividing by the gain (fitted on other sites).',
          panels: [{ value: 'gain', label: 'Response per µg of thermal EC', reference: 1, lo: 'gain_lo', hi: 'gain_hi' },
            { value: 'urban_level', label: 'Six urban IMPROVE sites: FTIR ÷ TOR', reference: 1 },
            { value: 'light_median_abs_error', label: 'Light filters: median error (µg)' },
            { value: 'overall_median_abs_error', label: 'All filters: median error (µg)' }],
          labelWidth: 180,
          noteAlways: true,
          note: IMPROVE_ONLY,
        },
        {
          kind: 'sweep', id: 'p1110-baseline-k', table: 'baseline_k', x: 'k', y: 'gain', series: 'preprocessing', reference: 1,
          xLabel: 'PLS components', yLabel: 'Response per µg of thermal EC', xMarkers: [{ at: 18, label: 'used' }],
          title: 'Every baseline follows the same curve with components',
          subtitle: 'Gain from IMPROVE site means at 6, 12, 18 and 25 components.',
        },
        {
          kind: 'bars', id: 'p1110-baseline-addis', table: 'baseline_addis', label: 'calibration', value: 'intercept', color: 'comparator', sort: 'none', reference: 0, format: 'f2',
          choose: [{ column: 'comparator', label: 'Against', default: 'HIPS as measured' }],
          xLabel: 'Addis intercept of FTIR EC against HIPS BC (µg/m³)',
          title: 'At Addis the baselines only looked different because HIPS was the yardstick',
          subtitle: 'Deming intercept at Addis for each baseline’s saved calibration (7 Nov closeout). Switch "Against" to linearised HIPS: the spread between baselines collapses.',
          noteAlways: true,
          note: ADDIS,
        },
        {
          kind: 'table', id: 'p1110-baselines-table', table: 'baselines', title: 'The baselines, in numbers',
          columns: [{ key: 'preprocessing', label: 'Baseline' }, { key: 'step', label: '' }, { key: 'gain', format: 'f2' }, { key: 'gain_lo', label: 'low', format: 'f2' },
            { key: 'gain_hi', label: 'high', format: 'f2' }, { key: 'offset', label: 'offset (µg)', format: 'f2' }, { key: 'urban_level', label: 'urban', format: 'f2' },
            { key: 'heavy_level', label: 'heavy', format: 'f2' }, { key: 'light_median_abs_error', label: 'light error (µg)', format: 'f2' },
            { key: 'overall_median_abs_error', label: 'median error (µg)', format: 'f2' }, { key: 'overall_rmse', label: 'RMSE (µg)', format: 'f2' }],
        },
      ],
    },
  ],
} satisfies MeetingSpec
