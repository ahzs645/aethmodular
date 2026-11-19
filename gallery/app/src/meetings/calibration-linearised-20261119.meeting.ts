import type { MeetingSpec } from './types'

// "Which calibration, once HIPS is linearised": the sweep of 19 Nov 2026. 33 IMPROVE calibrations (raw spectra or a
// baseline × eleven training sets) at 1 to 30 components, each fitted against Addis HIPS as measured and linearised,
// and scored against thermal EC on held-out IMPROVE sites. Tables from
// gallery/data/export_meeting_20261119_calibration_linearised.py (it reads the saved sweep and fits nothing). Titles
// carry the message: section text shows only with Explain on. The numbers in the titles are checked against the
// bundle by lib/calibrationLinearised.test.mjs.
const OPTICAL = 'HIPS is optical, so a slope against it compares two instruments. It is not a score against EC.'
const DEMING = 'Deming fits of FTIR EC on HIPS BC in µg/m³, 233 Addis filters, error ratio as the closeout.'
const HELD = 'IMPROVE columns: the same model on IMPROVE sites held out of its training, against thermal EC. 1.00 is right.'
/** the MAC measured against thermal EC at IMPROVE\'s six urban sites (10 Nov closure), and the error ratio there: 2.96 × (16.4 ÷ 10)² */
export const MAC_URBAN = 16.4
const LAMBDA_10 = 2.96
const AT_10 = 'MAC 10 (the convention)', AT_URBAN = `MAC ${MAC_URBAN} (urban IMPROVE, against thermal EC)`   // the exporter's panel names
const LINEARISED = 'Linearised, ceiling 2.26'
const BEST_COLUMNS = [{ key: 'rank', label: 'Rank', format: 'int' as const }, { key: 'Spectra', label: 'Spectra' }, { key: 'Training set', label: 'Training set' }, { key: 'n_training', label: 'Training filters', format: 'int' as const },
  { key: 'k', label: 'k', format: 'int' as const }, { key: 'slope', label: 'Slope', format: 'f2' as const }, { key: 'intercept', label: 'Intercept (µg/m³)', format: 'f2' as const }, { key: 'r2', label: 'R²', format: 'f2' as const },
  { key: 'urban', label: 'IMPROVE urban, FTIR ÷ thermal EC', format: 'f2' as const }, { key: 'heavy', label: 'IMPROVE heavy filters', format: 'f2' as const }, { key: 'error', label: 'IMPROVE median error (µg)', format: 'f2' as const },
  { key: 'median_ec', label: 'Addis median EC (µg/m³)', format: 'f2' as const }]

export default {
  id: 'calibration-linearised-20261119',
  label: 'Nov 19, 2026',
  date: '2026-11-19',
  short: 'calibration, HIPS linearised',
  title: 'Which calibration, once HIPS is linearised: none reaches a slope of 1 at MAC 10',
  command: 'uv run python gallery/data/export_meeting_20261119_calibration_linearised.py',
  intro: '@intro',
  sections: [
    {
      id: 'glance',
      title: 'At a glance: no calibration reaches a slope of 1 at MAC 10, and raw spectra are the ones that stay proportional',
      figures: [
        {
          kind: 'table', id: 'cl-glance', table: 'glance', title: 'What was asked, and what the sweep found',
          subtitle: 'Each row is a section below.',
          columns: [{ key: 'question', label: 'Question' }, { key: 'found', label: 'What it found' }],
          noteAlways: true, note: OPTICAL,
        },
      ],
    },
    {
      id: 'by-k',
      title: 'Against k: raw spectra keep the intercept near zero at every k; a baseline lowers the slope or bends the intercept',
      text: '@by_k',
      figures: [
        {
          kind: 'sweep', id: 'cl-by-k', table: 'by_k', x: 'k', y: 'slope', series: 'Spectra', reference: 1, height: 340,
          choose: [{ column: 'Training set', label: 'Training set', default: 'All IMPROVE filters' }, { column: 'HIPS', label: 'HIPS', default: LINEARISED }],
          title: 'With every IMPROVE filter in training: raw spectra rise from 0.30 to 0.57, VIBES to 0.60, the spline baseline stays under 0.25',
          subtitle: 'One model per number of components, applied to the Addis filters. The dashed line is a slope of 1 at MAC 10. Choose another training set, or HIPS as measured for the earlier reading.',
          xLabel: 'PLS components (k)', yLabel: 'Slope of FTIR EC on HIPS BC, Addis (MAC 10)', xMarkers: [{ at: 18, label: 'published-like' }],
          variants: [{ key: 'read', label: 'Read', default: 'Slope', choices: [
            { value: 'Slope', patch: { y: 'slope', yLabel: 'Slope of FTIR EC on HIPS BC, Addis (MAC 10)', reference: 1 } },
            { value: 'Intercept', patch: { y: 'intercept', yLabel: 'Intercept of FTIR EC on HIPS BC, Addis (µg/m³)', reference: 0 } },
            { value: 'R²', patch: { y: 'r2', yLabel: 'R² of FTIR EC against HIPS BC, Addis', reference: undefined } },
            { value: 'MAC for a slope of 1', patch: { y: 'mac_for_slope_1', yLabel: 'MAC at which the slope is 1 (m²/g)', reference: 10, yClip: [0, 60] } },
            { value: 'Median EC', patch: { y: 'median_ec_ugm3', yLabel: 'Median FTIR EC of the Addis filters (µg/m³)', reference: undefined } }] }],
          noteAlways: true, note: DEMING + ' ' + OPTICAL,
        },
        {
          kind: 'sweep', id: 'cl-by-training', table: 'by_k', where: { HIPS: LINEARISED }, x: 'k', y: 'slope', series: 'Training set', reference: 1, height: 380,
          choose: [{ column: 'Spectra', label: 'Spectra', default: 'Raw spectra' }],
          title: 'By training set, raw spectra: heavy filters in training raise the slope, a cut-off lowers it, and none passes 0.72',
          subtitle: 'Eleven training sets on the same spectra, against linearised HIPS (ceiling 2.26). Click a legend entry to hide a set.',
          xLabel: 'PLS components (k)', yLabel: 'Slope of FTIR EC on linearised HIPS BC, Addis (MAC 10)',
          variants: [{ key: 'read', label: 'Read', default: 'Slope', choices: [
            { value: 'Slope', patch: { y: 'slope', yLabel: 'Slope of FTIR EC on linearised HIPS BC, Addis (MAC 10)', reference: 1 } },
            { value: 'Intercept', patch: { y: 'intercept', yLabel: 'Intercept of FTIR EC on linearised HIPS BC, Addis (µg/m³)', reference: 0 } },
            { value: 'R²', patch: { y: 'r2', yLabel: 'R² of FTIR EC against linearised HIPS BC, Addis', reference: undefined } }] }],
          noteAlways: true, note: DEMING,
        },
      ],
    },
    {
      id: 'plane',
      title: 'Slope and intercept together: at MAC 10 every model is left of the target; at the urban IMPROVE MAC the raw-spectra models land on it',
      text: '@plane',
      figures: [
        {
          kind: 'scatter', id: 'cl-plane', table: 'plane', x: 'slope_at_mac', y: 'intercept', color: 'Spectra', facet: 'MAC', refX: [1], refY: [0],
          tipColumns: ['Training set', 'k', 'R²', 'IMPROVE urban FTIR ÷ thermal EC'],
          xLabel: 'Slope of FTIR EC on linearised HIPS BC, Addis', yLabel: 'Intercept (µg/m³)',
          title: '957 fits against linearised HIPS: the largest slope at MAC 10 is 0.72; at MAC 16.4 the raw-spectra fits reach 1 with an intercept near 0',
          subtitle: 'One dot per model and number of components (2 to 30), drawn at each MAC. The lines mark a slope of 1 and an intercept of 0.',
          noteAlways: true, note: 'A slope at another MAC is the slope at MAC 10 × MAC ÷ 10; the intercept does not move. 16.4 m²/g is Fabs ÷ thermal EC on light filters at IMPROVE\'s six urban sites. Ceiling 2.26. ' + OPTICAL,
        },
      ],
    },
    {
      id: 'best',
      title: 'The closest to a slope of 1 and an intercept of 0: 0.72x − 0.05 at MAC 10, and 1.00x + 0.02 at the urban IMPROVE MAC',
      text: '@best',
      figures: [
        {
          kind: 'table', id: 'cl-best-10', table: 'best_10',
          title: 'Closest at MAC 10: raw spectra trained on heavy filters only, 0.72x − 0.05, with more than four times the error on IMPROVE',
          subtitle: 'The twelve closest models, each at its best k, ranked by distance from a slope of 1 and an intercept of 0 against linearised HIPS (ceiling 2.26).',
          columns: BEST_COLUMNS,
          noteAlways: true, note: HELD + ' Heavy filters are chosen by the previous sample at the site (15 µg or more), never by their own thermal EC. The top rows are chosen on Addis, so they are fitted to the target they are scored on.',
        },
        {
          kind: 'table', id: 'cl-best-urban', table: 'best_urban',
          title: `Closest at MAC ${MAC_URBAN}: raw spectra reach 1.00x + 0.02, and seven of the first eight are raw spectra`,
          subtitle: `The same ranking with HIPS BC as Fabs ÷ ${MAC_URBAN}, the MAC measured against thermal EC at IMPROVE's six urban sites.`,
          columns: BEST_COLUMNS,
          noteAlways: true, note: HELD,
        },
        {
          kind: 'table', id: 'cl-plain', table: 'plain', title: 'For comparison, the plain models: raw spectra with every filter read 0.49x + 0.11 at MAC 10 and 0.81x + 0.11 at the urban IMPROVE MAC',
          subtitle: 'Every IMPROVE filter in training, at the number of components each is used at.',
          columns: [{ key: 'model', label: 'Model' }, { key: 'slope_10', label: 'Slope, MAC 10', format: 'f2' }, { key: 'slope_urban', label: `Slope, MAC ${MAC_URBAN}`, format: 'f2' },
            { key: 'intercept', label: 'Intercept (µg/m³)', format: 'f2' }, { key: 'r2', label: 'R²', format: 'f2' }, { key: 'mac_for_slope_1', label: 'MAC for a slope of 1 (m²/g)', format: 'f2' },
            { key: 'urban', label: 'IMPROVE urban, FTIR ÷ thermal EC', format: 'f2' }, { key: 'heavy', label: 'IMPROVE heavy filters', format: 'f2' }, { key: 'error', label: 'IMPROVE median error (µg)', format: 'f2' }],
          noteAlways: true, note: HELD,
        },
      ],
    },
    {
      id: 'cost',
      title: 'Where EC is known: the raw-spectra models that read thermal EC right on IMPROVE put Addis at a slope near 0.56',
      text: '@cost',
      figures: [
        {
          kind: 'scatter', id: 'cl-cost', table: 'cost', x: 'slope', y: 'improve_urban_level', color: 'Spectra', refY: [1], maxSide: 520,
          tipColumns: ['Training set', 'k', 'Median error on IMPROVE (µg)'],
          xLabel: 'Slope of FTIR EC on linearised HIPS BC, Addis (MAC 10)', yLabel: 'FTIR ÷ thermal EC, six urban IMPROVE sites (held out)',
          title: 'Addis slope against the level on IMPROVE: raw-spectra fits within 5 % of thermal EC give slopes of 0.39 to 0.71, median 0.56',
          subtitle: 'One dot per model and number of components. On the horizontal line the model reads thermal EC right at IMPROVE\'s urban sites; a slope of 1 at Addis would be off the right edge.',
          noteAlways: true, note: 'IMPROVE\'s urban sites have a thermal OC/EC near 3 and Addis is near 1, so this checks the level where EC is known, not at Addis composition. ' + OPTICAL,
        },
      ],
    },
    {
      id: 'drawn',
      title: 'Filter by filter: the chosen models on the Addis filters',
      text: '@drawn',
      figures: [
        {
          kind: 'crossplot', id: 'cl-drawn', table: 'drawn', x: 'hips', y: 'ftir', color: 'season', idColumn: 'filter', tipColumns: ['date'],
          choose: [{ column: 'Model', label: 'Model', default: 'Raw spectra, every filter, k 18' }],
          fit: true, errorRatio: LAMBDA_10, sameScale: true, maxSide: 520,
          xLabel: 'HIPS BC, linearised (µg/m³, MAC 10)', yLabel: 'FTIR EC from this model (µg/m³)',
          title: 'One model on 233 Addis filters, against linearised HIPS',
          subtitle: 'Each dot is a filter, coloured by season. Choose the model; switch the MAC to read the same filters against Fabs ÷ 16.4.',
          variants: [{ key: 'mac', label: 'MAC', default: AT_10, choices: [
            { value: AT_10, patch: { x: 'hips', xLabel: 'HIPS BC, linearised (µg/m³, MAC 10)', errorRatio: LAMBDA_10 } },
            { value: AT_URBAN, patch: { x: 'hips_urban', xLabel: `HIPS BC, linearised (µg/m³, MAC ${MAC_URBAN})`, errorRatio: LAMBDA_10 * (MAC_URBAN / 10) ** 2 } }] }],
          noteAlways: true, note: 'Ceiling 2.26, each filter by its own factor. Every Addis filter is past a 20 % expected under-read, so the linearised value is known for shape better than for level. ' + OPTICAL,
        },
      ],
    },
    {
      id: 'saved',
      title: 'The nine saved configurations, read again: the spline ones fall to a quarter, the raw ones keep an intercept near zero',
      text: '@saved',
      figures: [
        {
          kind: 'table', id: 'cl-saved', table: 'saved', title: 'Calibration tab, the nine saved configurations at Addis: as measured against linearised',
          subtitle: 'Every evaluation filter with a sampled volume (233 of 239). The Calibration tab draws these with Plot, HIPS, Linearised.',
          columns: [{ key: 'configuration', label: 'Configuration' }, { key: 'k', label: 'k', format: 'int' }, { key: 'measured', label: 'Against HIPS as measured' }, { key: 'linearised', label: 'Against linearised HIPS' },
            { key: 'r2', label: 'R², linearised', format: 'f2' }, { key: 'slope_urban', label: `Slope at MAC ${MAC_URBAN}`, format: 'f2' }, { key: 'mac_for_slope_1', label: 'MAC for a slope of 1 (m²/g)', format: 'f2' }],
          noteAlways: true, note: DEMING.replace('233 Addis filters', 'MAC 10') + ' ' + OPTICAL,
        },
      ],
    },
    {
      id: 'mac',
      title: 'Why 1 at MAC 10 is not the target: every MAC measured against thermal EC is above 10',
      text: '@mac',
      figures: [
        {
          kind: 'table', id: 'cl-mac', table: 'mac', title: 'With FTIR EC exactly right, the slope at MAC 10 would be 0.53 to 0.76',
          subtitle: 'MAC is Fabs ÷ thermal EC on filters light enough for HIPS to read in proportion (10 Nov closure).',
          columns: [{ key: 'where', label: 'Where' }, { key: 'mac', label: 'MAC (m²/g)', format: 'f2' }, { key: 'slope_if_ftir_right', label: 'Slope at MAC 10 if FTIR EC is right', format: 'f2' }, { key: 'source', label: 'From' }],
        },
      ],
    },
    { id: 'limits', title: 'Limits', text: '@limits' },
  ],
} satisfies MeetingSpec
