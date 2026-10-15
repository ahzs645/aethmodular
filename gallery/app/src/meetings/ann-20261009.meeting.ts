import type { MeetingSpec } from './types'
import type { FigureVariant } from '@/components/Figure'

// a reader-facing variant (components/Figure `variants`): the loadings span three decades, so log by default
const LOG_AXIS: FigureVariant = { key: 'axis', label: 'x axis', default: 'log', choices: [
  { value: 'log', label: 'Log', patch: { log: true } }, { value: 'linear', label: 'Linear', patch: { log: false } },
] }

const SEASON_ROWS = ['All Addis', 'Dry (Oct-Feb)', 'Belg (Mar-May)', 'Kiremt (Jun-Sep)']

export default {
  id: 'ann-20261009',
  label: 'Ann 9 Oct',
  date: '2026-10-09',
  title: 'Questions from the 9 Oct meeting with Ann',
  command: 'cd research/ftir_hips_chem && ../../.venv/bin/python ../../gallery/data/export_meeting_20261009.py',
  intro: '@intro',
  sections: [
    {
      id: 'self',
      title: '1. Can IMPROVE predict itself?',
      text: '@self',
      figures: [
        {
          kind: 'intervals', id: 'a1009-self-intercepts', table: 'splits', where: { baseline: ['Spline', 'VIBES'], group: SEASON_ROWS },
          title: 'Held-out IMPROVE vs Addis intercepts — the calibration predicts its own held-out sites with a near-zero intercept; Addis does not',
          subtitle: 'Seasonal analog calibrations, 100 site-grouped splits each: about 20 % of the analogs held out by whole site, k chosen on the rest, the same calibration then predicting the held-out IMPROVE filters (vs TOR) and the season’s Addis filters (vs HIPS Fabs / MAC 10). Median and interquartile range over the splits.',
          row: 'group', series: 'baseline',
          panels: [
            { value: 'ftir_vs_tor_intercept', label: 'Held-out IMPROVE intercept (µg/m³)', reference: 0 },
            { value: 'addis_intercept', label: 'Addis intercept (µg/m³)', reference: 0 },
          ],
        },
        {
          kind: 'intervals', id: 'a1009-self-slopes', table: 'splits', where: { baseline: ['Spline', 'VIBES'], group: SEASON_ROWS },
          title: 'Slopes and cross-validation R² — VIBES predicts IMPROVE better than spline',
          subtitle: 'Same splits. Held-out IMPROVE slopes of 0.5–0.9 are the expected PLS shrinkage at low loadings.',
          row: 'group', series: 'baseline',
          panels: [
            { value: 'ftir_vs_tor_slope', label: 'Held-out IMPROVE slope', reference: 1 },
            { value: 'addis_slope', label: 'Addis slope', reference: 1 },
            { value: 'TOR_test_R2', label: 'Cross-validation TOR R²' },
          ],
        },
        {
          kind: 'table', id: 'a1009-allimprove', table: 'allimprove',
          title: 'All-IMPROVE models — three populations under one calibration',
          subtitle: 'The current-model reading (lot 251, and both lots): every held-out IMPROVE filter, the held-out filters that are also all-Addis analogs, and the Addis test set. Deming FTIR vs HIPS, single protocol split.',
          columns: [
            { key: 'model', label: 'Model' }, { key: 'baseline', label: 'Baseline' }, { key: 'k', label: 'k', format: 'int' },
            { key: 'population', label: 'Population' }, { key: 'slope', label: 'Slope', format: 'coef' },
            { key: 'intercept', label: 'Intercept (µg/m³)', format: 'coef' }, { key: 'n', label: 'n', format: 'int' },
          ],
        },
      ],
    },
    {
      id: 'window',
      title: '2. Is it the wavenumber range?',
      text: '@window',
      figures: [
        {
          kind: 'intervals', id: 'a1009-window', table: 'splits', where: { group: SEASON_ROWS },
          title: 'VIBES full vs VIBES cut at 1425 — cutting the fingerprint region closes little of the gap to spline',
          subtitle: 'Seasonal analog calibrations, median and interquartile range over 100 splits. “VIBES cut” is the same VIBES correction predicting from 4000–1425 cm⁻¹ only.',
          row: 'group', series: 'baseline',
          panels: [
            { value: 'addis_intercept', label: 'Addis intercept (µg/m³)', reference: 0 },
            { value: 'k', label: 'PLS factors chosen (k)' },
          ],
        },
        {
          kind: 'sweep', id: 'a1009-k', table: 'splits', where: { group: SEASON_ROWS },
          title: 'Addis intercept against the number of PLS factors — the bias follows k',
          subtitle: 'Each split chooses its own k; points are the median Addis intercept over the splits that chose that k, with the interquartile range. One panel per season.',
          x: 'k', y: 'addis_intercept', series: 'baseline', facet: 'group', aggregate: { stat: 'median', interval: 'iqr' },
          xLabel: 'PLS factors (k)', yLabel: 'Addis intercept (µg/m³)', reference: 0, interval: 'whisker',
        },
        {
          kind: 'table', id: 'a1009-allimprove-addis', table: 'allimprove_addis',
          title: 'All-IMPROVE models by season — the cut moves the intercept the way k moves',
          columns: [
            { key: 'model', label: 'Model' }, { key: 'group', label: 'Addis group' }, { key: 'baseline', label: 'Baseline' },
            { key: 'k', label: 'k', format: 'int' }, { key: 'slope', label: 'Slope', format: 'coef' }, { key: 'intercept', label: 'Intercept (µg/m³)', format: 'coef' },
          ],
        },
      ],
    },
    {
      id: 'mass',
      title: '3. Mass per filter instead of concentration',
      text: 'IMPROVE samples about 33 m³ per filter, SPARTAN at Addis about 7 m³. PLS predicts loading (µg per filter), so the question for extrapolation is whether Addis *loadings* fall inside the IMPROVE range.\n\n@mass',
      columns: 2,
      figures: [
        {
          kind: 'histogram', id: 'a1009-conc', table: 'loading', value: 'conc', series: 'population', log: true,
          title: 'Concentration', xLabel: 'HIPS Fabs / MAC 10 (µg/m³)',
          variants: [LOG_AXIS],
        },
        {
          kind: 'histogram', id: 'a1009-mass', table: 'loading', value: 'mass', series: 'population', log: true,
          title: 'Mass per filter', xLabel: 'HIPS Fabs / MAC 10 × volume (µg per filter)',
          variants: [LOG_AXIS],
        },
        {
          kind: 'table', id: 'a1009-mass-table', table: 'loading_summary', title: 'Where the Addis median sits in the IMPROVE distribution',
          columns: [
            { key: 'measure', label: 'Measure' }, { key: 'IMPROVE median', format: 'f2' }, { key: 'IMPROVE p95', format: 'f2' }, { key: 'IMPROVE p99', format: 'f2' },
            { key: 'Addis median', format: 'f2' }, { key: 'Addis median, IMPROVE percentile', label: 'Addis median: IMPROVE percentile', format: 'percent' },
            { key: '% of Addis above IMPROVE p99', label: 'Addis above IMPROVE p99', format: 'percent' },
          ],
        },
      ],
    },
    {
      id: 'hump',
      title: '4. The hump near 1000–1300 cm⁻¹ in the Addis spectra',
      text: 'Two checks on the VIBES-corrected spectra (the only baseline that reaches below 1425 cm⁻¹). **Where is the excess?** Median spectra, each filter scaled by its CH band (2800–3000 cm⁻¹) so loading drops out, Addis by season against the IMPROVE pool. **What does it track?** Within Addis, band areas per m³ against SPARTAN chemistry.\n\n@hump',
      figures: [
        {
          kind: 'spectrum', id: 'a1009-hump', table: 'hump', wn: 'wn', y: 'median', lo: 'q25', hi: 'q75', series: 'series',
          title: 'Addis vs IMPROVE below 1900 cm⁻¹ — CH-normalised median spectra',
          subtitle: 'VIBES full range, rank 70. Each filter divided by its CH band area (2800–3000 cm⁻¹), then the median and interquartile range per population. Shaded: reference band positions.',
          yLabel: 'Absorbance / CH band area', window: [1900, 550], hover: 'readout',
          annotations: [
            { lo: 600, hi: 640, label: 'SO₄ ν4' }, { lo: 1000, hi: 1060, label: 'Si-O' }, { lo: 1080, hi: 1140, label: 'SO₄ ν3' },
            { lo: 1150, hi: 1250, label: 'PTFE C-F', fill: '#f2e6c9' }, { lo: 1400, hi: 1430, label: 'NH₄⁺' },
          ],
        },
        {
          kind: 'heatmap', id: 'a1009-band-chem', table: 'band_chem', row: 'band', column: 'species', value: 'rho', domain: [-1, 1], format: 'f2', valueLabel: 'Spearman ρ',
          title: 'Band areas against SPARTAN chemistry — the Si-O band tracks Si and Al',
          subtitle: 'Within Addis: band area per m³ (VIBES-corrected) against SPARTAN chemistry (µg/m³), Spearman ρ, filters with both.',
        },
      ],
    },
    { id: 'open', title: 'Still open from the meeting', text: '@open' },
  ],
} satisfies MeetingSpec
