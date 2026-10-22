import type { MeetingSpec } from './types'
import type { FigureVariant } from '@/components/Figure'

const LOG_AXIS: FigureVariant = { key: 'axis', label: 'x axis', default: 'log', choices: [
  { value: 'log', label: 'Log', patch: { xScale: 'log' } }, { value: 'linear', label: 'Linear', patch: { xScale: 'linear' } },
] }

export default {
  id: 'meeting-20261028',
  label: '28 Oct',
  date: '2026-10-28',
  title: 'Why Dry and Belg fail, and why the look-alikes err differently: the AAAR to-do list',
  command: 'cd research/ftir_hips_chem && ~/anaconda3/bin/python workflows/run_meeting_20261028.py && ../../.venv/bin/python ../../gallery/data/export_meeting_20261028.py',
  intro: '@intro',
  sections: [
    {
      id: 'ocec',
      title: '1 and 9. Calibration filters chosen by OC/EC, and urban IMPROVE sites',
      text: '@ocec',
      figures: [
        {
          kind: 'intervals', id: 'm1028-cals', table: 'cals', row: 'season', series: 'calibration',
          title: 'Addis slope and intercept by season, for five ways of choosing the IMPROVE calibration filters',
          subtitle: 'Each calibration is fitted on baseline-corrected spectra with the poster’s protocol (site-held-out k; the urban set uses the AQRC app’s interleaved CV because six sites are too few to group). Deming FTIR EC vs HIPS EC (Fabs / MAC 10) on the 189 poster filters.',
          panels: [
            { value: 'slope', label: 'Slope', reference: 1 },
            { value: 'intercept', label: 'Intercept (µg/m³)', reference: 0 },
            { value: 'R2', label: 'R²' },
          ],
          labelWidth: 90,
        },
        {
          kind: 'histogram', id: 'm1028-ocec', table: 'ocec_dist', value: 'oc_ec', series: 'population', log: true,
          title: 'OC/EC: Addis sits below almost every IMPROVE filter',
          subtitle: 'IMPROVE: TOR OC / TOR EC, filters with at least 1 µg EC. Addis: SPARTAN OC (FTIR) / HIPS EC.',
          xLabel: 'OC / EC',
        },
        {
          kind: 'table', id: 'm1028-cals-table', table: 'cals', title: 'The same fits as a table',
          columns: [
            { key: 'calibration', label: 'Calibration filters' }, { key: 'season', label: 'Season' }, { key: 'k', label: 'k', format: 'int' },
            { key: 'TOR_test_R2', label: 'IMPROVE check: TOR R²', format: 'f2' }, { key: 'slope', label: 'Addis slope', format: 'f2' },
            { key: 'intercept', label: 'Addis intercept (µg/m³)', format: 'f2' }, { key: 'R2', label: 'Addis R²', format: 'f2' }, { key: 'n', label: 'n', format: 'int' },
          ],
          note: 'The IMPROVE check is held-out sites for the site-held-out fits and leave-one-site-out for the urban set; it is blank for the all-IMPROVE calibration here (see Figure 7).',
        },
      ],
    },
    {
      id: 'deficit',
      title: '2. Does the Addis error track the organic signal?',
      text: '@deficit',
      figures: [
        {
          kind: 'crossplot', id: 'm1028-deficit', table: 'deficit_points', x: 'ch_resid', y: 'error', color: 'season', facet: 'season', fit: false, sameScale: false,
          idColumn: 'MediaId', maxSide: 340,
          title: 'Error against the C–H band, both with HIPS EC’s effect removed (season-specific calibrations)',
          subtitle: 'y: FTIR EC minus the season’s Deming line. x: C–H band area (2800–3000 cm⁻¹, per m³) minus what HIPS EC predicts for it. Points to the right carry more organic material than their EC suggests.',
          xLabel: 'C–H band beyond HIPS EC (10⁻³ absorbance per m³)', yLabel: 'FTIR EC error (µg/m³)',
        },
        {
          kind: 'heatmap', id: 'm1028-deficit-rho', table: 'deficit', row: 'calibration', column: 'season', value: 'partial_rho', domain: [-1, 1], format: 'f2',
          valueLabel: 'partial Spearman ρ', title: 'FTIR EC against the organic index, HIPS EC held fixed',
          subtitle: 'Organic index: the mean percentile of the C–H, O–H and carbonyl band areas per m³.',
        },
      ],
    },
    {
      id: 'dust',
      title: '3. Dust by season',
      text: '@dust',
      columns: 2,
      figures: [
        {
          kind: 'table', id: 'm1028-dust', table: 'dust', title: 'SPARTAN dust elements and HIPS EC (medians, µg/m³)',
          columns: [
            { key: 'season', label: 'Season' }, { key: 'n', format: 'int' }, { key: 'Al', format: 'f2' }, { key: 'Si', format: 'f2' },
            { key: 'Fe', format: 'f2' }, { key: 'HIPS EC', format: 'f2' }, { key: 'Si / EC', format: 'f2' }, { key: 'Al / EC', format: 'f2' },
          ],
        },
        {
          kind: 'heatmap', id: 'm1028-dust-rho', table: 'dust_rho', row: 'calibration', column: 'season', value: 'partial_rho', domain: [-1, 1], format: 'f2',
          valueLabel: 'partial Spearman ρ', title: 'FTIR EC against Si, HIPS EC held fixed',
        },
      ],
    },
    {
      id: 'mac',
      title: '4. MAC by season',
      text: '@mac',
      figures: [
        {
          kind: 'table', id: 'm1028-mac', table: 'mac', title: 'The MAC each season would need, and the look-alikes’ own',
          subtitle: 'Look-alike MAC: median Fabs / TOR EC of that season’s 500 IMPROVE look-alikes (m²/g). Slopes and intercepts are Deming FTIR EC vs Fabs / MAC.',
          columns: [
            { key: 'season', label: 'Season' }, { key: 'calibration', label: 'Calibration' },
            { key: 'slope at MAC 10', format: 'f2' }, { key: 'intercept at MAC 10', format: 'f2' }, { key: 'MAC for slope 1', format: 'f2' },
            { key: 'look-alike MAC (Fabs / TOR EC)', label: 'look-alike MAC', format: 'f2' },
            { key: 'slope at look-alike MAC', format: 'f2' }, { key: 'intercept at look-alike MAC', format: 'f2' },
          ],
        },
      ],
    },
    {
      id: 'highwn',
      title: '5. The 3600–4000 cm⁻¹ residual',
      text: '@highwn',
      columns: 2,
      figures: [
        {
          kind: 'heatmap', id: 'm1028-highwn-rho', table: 'highwn_rho', row: 'against', column: 'season', value: 'rho', domain: [-1, 1], format: 'f2',
          valueLabel: 'Spearman ρ', title: '3600–4000 cm⁻¹ band area (per m³) against Addis chemistry and humidity',
        },
        {
          kind: 'crossplot', id: 'm1028-highwn-si', table: 'highwn_points', x: 'Si', y: 'highwn', color: 'season', fit: false, sameScale: false, idColumn: 'MediaId', maxSide: 380,
          title: 'The residual follows Si', xLabel: 'SPARTAN Si (µg/m³)', yLabel: '3600–4000 cm⁻¹ band (10⁻³ per m³)',
        },
      ],
    },
    {
      id: 'loading',
      title: '6. Loading: the heaviest IMPROVE filters as an out-of-range test',
      text: '@loading',
      figures: [
        {
          kind: 'sweep', id: 'm1028-loading', table: 'loading_bins', x: 'loading', y: 'ratio', lo: 'lo', hi: 'hi', series: 'series',
          title: 'FTIR EC / reference EC against EC on the filter',
          subtitle: 'Median and interquartile range in loading bins (0–2, 2–5, 5–10, 10–20, 20–40, 40–80, 80+ µg; bins with at least 5 filters). IMPROVE against TOR on the filter; Addis against HIPS EC × volume. “Out of range”: a baseline-corrected calibration trained on lot-251 filters below 20 µg, predicting the 367 above.',
          xLabel: 'EC on the filter (µg)', yLabel: 'FTIR / reference', xScale: 'log', reference: 1, interval: 'whisker',
          variants: [LOG_AXIS],
        },
        {
          kind: 'table', id: 'm1028-loading-fits', table: 'loading_fits', title: 'OLS fits in µg per filter',
          columns: [
            { key: 'population', label: 'Filters' }, { key: 'reference', label: 'Reference' }, { key: 'n', format: 'int' }, { key: 'k', format: 'int' },
            { key: 'slope', format: 'f2' }, { key: 'intercept µg', format: 'f2' }, { key: 'R2', label: 'R²', format: 'f2' },
            { key: 'median FTIR / reference', format: 'f2' },
          ],
        },
      ],
    },
    {
      id: 'dark',
      title: '7. TOR on dark filters (optical pyrolysis at zero)',
      text: '@dark',
      figures: [
        {
          kind: 'sweep', id: 'm1028-dark', table: 'dark_bins', x: 'loading', y: 'ratio', lo: 'lo', hi: 'hi', series: 'group',
          title: 'TOR EC / (HIPS Fabs / 10) against HIPS EC on the filter, IMPROVE library',
          subtitle: 'OP = 0: TOR’s optical pyrolysis (reflectance) reads zero. Median and interquartile range per loading bin (bins with at least 10 filters).',
          xLabel: 'HIPS EC on the filter (µg)', yLabel: 'TOR / HIPS', xScale: 'log', reference: 1, interval: 'whisker',
          variants: [LOG_AXIS],
        },
        {
          kind: 'table', id: 'm1028-dark-share', table: 'dark_share', title: 'How often the look-alikes are OP-zero filters',
          columns: [
            { key: 'set', label: 'Filters' }, { key: 'n', format: 'int' }, { key: 'OP = 0 share', format: 'pct' },
            { key: 'median TOR EC / (HIPS/10)', label: 'median TOR / (HIPS/10)', format: 'f2' },
          ],
        },
      ],
    },
    {
      id: 'vip',
      title: '8. What each calibration reads as EC (VIP)',
      text: '@vip',
      figures: [
        {
          kind: 'spectrum', id: 'm1028-vip-raw', table: 'vip', wn: 'wn', y: 'vip', series: 'model',
          where: { model: ['raw, SPARTAN-like (k 18)', 'baseline-corrected, all IMPROVE (k 8)'] },
          title: 'VIP: raw SPARTAN-like calibration vs the all-IMPROVE baseline-corrected one',
          subtitle: 'Variable importance in projection; 1 is the average. The baseline-corrected calibration only sees 1425–4000 cm⁻¹.',
          yLabel: 'VIP', window: [4000, 500], hover: 'readout', regions: ['ptfe'], zeroLine: false,
        },
        {
          kind: 'spectrum', id: 'm1028-vip-seasons', table: 'vip', wn: 'wn', y: 'vip', series: 'model',
          where: { model: ['Dry look-alikes', 'Belg look-alikes', 'Kiremt look-alikes', '500 lowest OC/EC'] },
          title: 'VIP of the season-specific look-alike calibrations and the low-OC/EC one',
          yLabel: 'VIP', window: [4000, 1425], hover: 'readout', zeroLine: false,
        },
        {
          kind: 'heatmap', id: 'm1028-vip-regions', table: 'vip_regions', row: 'region', column: 'model', value: 'share', ramp: 'sequential', format: 'pct',
          domain: [0, 0.5], valueLabel: 'share of VIP²', title: 'Share of VIP² by spectral region',
        },
      ],
    },
    {
      id: 'speciation',
      title: '10. Addis chemistry vs IMPROVE speciation',
      text: '@speciation',
      figures: [
        {
          kind: 'heatmap', id: 'm1028-speciation', table: 'speciation', row: 'population', column: 'species', value: 'median share of PM2.5',
          ramp: 'sequential', format: 'pct', domain: [0, 0.35], valueLabel: 'median share of PM2.5',
          title: 'Composition as a share of PM2.5',
          subtitle: 'IMPROVE from the validated FED archive (SO4f, NO3f, OCf, ECf, SOILf, MF) for the library filters. Addis: SPARTAN sulfate, nitrate and OC, HIPS EC, soil from the IMPROVE formula on SPARTAN Al, Si, Ca, Fe, Ti.',
        },
        {
          kind: 'table', id: 'm1028-speciation-table', table: 'speciation_table', title: 'The same shares, with OC/EC',
          columns: [
            { key: 'population', label: 'Filters' }, { key: 'EC', format: 'pct' }, { key: 'OC', format: 'pct' }, { key: 'sulfate', format: 'pct' },
            { key: 'nitrate', format: 'pct' }, { key: 'soil', format: 'pct' }, { key: 'OC / EC', format: 'f2' },
          ],
        },
      ],
    },
    {
      id: 'blh',
      title: '11. Normalising by boundary-layer height',
      text: '@blh',
      figures: [
        {
          kind: 'table', id: 'm1028-blh', table: 'blh', title: 'Addis by season: concentration, and concentration × boundary-layer height',
          subtitle: 'ERA5 boundary-layer height averaged over each filter’s sampling window; medians per season.',
          columns: [
            { key: 'quantity', label: 'Quantity' }, { key: 'season', label: 'Season' }, { key: 'BLH m', label: 'BLH (m)', format: 'int' },
            { key: 'median µg/m³', format: 'f2' }, { key: 'median mg/m² (× BLH)', format: 'f2' },
            { key: 'relative to Dry', format: 'f2' }, { key: 'relative to Dry (× BLH)', format: 'f2' },
          ],
        },
      ],
    },
    { id: 'next', title: 'What this suggests next', text: '@next' },
  ],
} satisfies MeetingSpec
