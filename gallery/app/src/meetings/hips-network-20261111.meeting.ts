import type { MeetingSpec } from './types'

// "HIPS across the network": the 10 Nov linearisation applied to every SPARTAN filter. Which sites, months and filters
// it touches, how much is the air and how much the sampled volume, whether each site's own data show it, and what that
// leaves us able to do. Every table comes from gallery/data/export_meeting_20261111.py, which reruns
// research/ftir_hips_chem/workflows/run_hips_network_20261111.py on committed files. Titles carry the message: section
// text shows only with Explain on.
const CURVE = 'Predicted from the IMPROVE curve (ceiling 2.26 [1.82, 2.70], measured against thermal EC). SPARTAN has no thermal EC to check it on its own filters.'
const OPTICAL = 'HIPS is optical and FTIR EC is FTIR-derived; no SPARTAN filter here has thermal EC. No aethalometer data are used.'

export default {
  id: 'hips-network-20261111',
  label: 'Nov 11, 2026',
  date: '2026-11-11',
  short: 'HIPS across the network',
  title: 'HIPS across the network: where it under-reads, why, and what we can do with it',
  command: 'uv run python gallery/data/export_meeting_20261111.py',
  intro: '@intro',
  sections: [
    {
      id: 'sites',
      title: '1. Where it bites: twelve sites lose more than a fifth on their median filter',
      text: '@sites',
      figures: [
        {
          kind: 'bars', id: 'h1111-sites', table: 'sites', label: 'site', value: 'under_read', color: 'band', format: 'pct',
          xLabel: 'share of the absorption HIPS does not report (median filter)',
          title: 'Addis Ababa loses the most, 42 %; Delhi 21 %; thirteen sites under a tenth',
          subtitle: 'Each site\'s median filter: HIPS optical depth ÷ ceiling. Colour is the band the median falls in.',
          noteAlways: true, note: CURVE,
        },
        {
          kind: 'table', id: 'h1111-sites-table', table: 'sites', title: 'Every site: the air, the volume, the optical depth and the share of filters past each mark',
          columns: [{ key: 'site', label: 'Site' }, { key: 'n', label: 'Filters', format: 'int' }, { key: 'median_fabs', label: 'Fabs reported (Mm⁻¹)', format: 'f2' },
            { key: 'linearised_fabs', label: 'Fabs linearised (Mm⁻¹)', format: 'f2' }, { key: 'median_volume', label: 'Volume (m³)', format: 'f2' },
            { key: 'median_tau', label: 'Optical depth', format: 'f2' }, { key: 'under_read', label: 'Under-read', format: 'pct' },
            { key: 'under_read_lo', label: 'low', format: 'pct' }, { key: 'under_read_hi', label: 'high', format: 'pct' },
            { key: 'share_over_20', label: 'Filters over a fifth', format: 'pct' }, { key: 'share_over_30', label: 'over three tenths', format: 'pct' },
            { key: 'share_beyond_improve', label: 'darker than IMPROVE\'s range', format: 'pct' }],
          subtitle: 'Medians per site. "Linearised" is what the median filter would report without the bend. Low and high use the ends of the ceiling\'s interval.',
        },
      ],
    },
    {
      id: 'volume',
      title: '2. Air or volume? Delhi\'s air is as heavy as Addis\'s; half the volume halves the loss',
      text: '@volume',
      figures: [
        {
          kind: 'scatter', id: 'h1111-air-volume', table: 'sites', x: 'median_volume', y: 'linearised_fabs', color: 'band', idColumn: 'site',
          tipColumns: ['site', 'median_fabs', 'median_tau', 'under_read'], maxSide: 560,
          xLabel: 'sampled volume, site median (m³)', yLabel: 'absorption in the air, linearised (Mm⁻¹)',
          title: 'Addis, Delhi and Jaipur have the heaviest air; only Addis draws the full 7.2 m³ through the filter',
          subtitle: 'One point per site. Up is heavier air, right is more air through the filter; the two multiply into the filter\'s optical depth.',
          noteAlways: true, note: CURVE,
        },
        {
          kind: 'table', id: 'h1111-replay', table: 'replay', title: 'The same air at each of SPARTAN\'s three volumes: Addis would lose 27 % at 3.6 m³ and 15 % at 1.8 m³',
          subtitle: 'Each filter\'s loading is taken back through the curve, rescaled to the volume, and read forward again; the site median is shown.',
          columns: [{ key: 'site', label: 'Site' }, { key: 'median_volume', label: 'Volume now (m³)', format: 'f2' }, { key: 'as_sampled', label: 'Under-read as sampled', format: 'pct' },
            { key: 'at_7.2', label: 'at 7.2 m³', format: 'pct' }, { key: 'at_3.6', label: 'at 3.6 m³', format: 'pct' }, { key: 'at_1.8', label: 'at 1.8 m³', format: 'pct' }],
        },
        {
          kind: 'scatter', id: 'h1111-history', table: 'points', x: 'year', y: 'under_read', color: 'regime', idColumn: 'id',
          tipColumns: ['id', 'date', 'fabs', 'volume', 'tau'], choose: [{ column: 'site', label: 'Site', default: 'Delhi, India' }], refY: [0.2, 0.3],
          xLabel: 'sampling date (22 = 2022)', yLabel: 'expected under-read',
          title: 'Every filter through time, coloured by sampled volume: Delhi\'s volume was cut twice, Addis\'s never',
          subtitle: 'One point per filter at the chosen site. Lines at a fifth and three tenths. Try Addis Ababa (one volume throughout), Jaipur and Dhaka.',
          noteAlways: true, note: 'Delhi\'s 2022–23 filters at 7 m³ read low on HIPS and on FTIR EC alike, so they do not separate the two instruments. ' + CURVE,
        },
        {
          kind: 'table', id: 'h1111-matched', table: 'matched', title: 'Delhi, same months at two volumes: more absorption per unit PM2.5 at the lower volume, as the curve predicts',
          subtitle: 'Median HIPS Fabs ÷ weighed PM2.5 on filters sampled at 1.8 m³, over the same on filters sampled at 3.6 m³. Interval: 90 % bootstrap over filters. No FTIR.',
          columns: [{ key: 'months', label: 'Months' }, { key: 'n_low_volume', label: 'Filters at 1.8 m³', format: 'int' }, { key: 'n_high_volume', label: 'at 3.6 m³', format: 'int' },
            { key: 'tau_low_volume', label: 'Optical depth at 1.8 m³', format: 'f2' }, { key: 'tau_high_volume', label: 'at 3.6 m³', format: 'f2' },
            { key: 'observed_ratio', label: 'Observed ratio', format: 'f2' }, { key: 'lo', label: 'low', format: 'f2' }, { key: 'hi', label: 'high', format: 'f2' },
            { key: 'predicted_ratio', label: 'Predicted', format: 'f2' }],
          noteAlways: true, note: 'The 1.8 m³ filters are from 2026 and the 3.6 m³ filters from 2024, so a change in the aerosol between those years would look the same.',
        },
        {
          kind: 'table', id: 'h1111-regimes', table: 'regimes', choose: [{ column: 'site', label: 'Site', default: 'Delhi, India' }],
          title: 'Sites that have sampled at more than one volume',
          columns: [{ key: 'regime', label: 'Volume' }, { key: 'n', label: 'Filters', format: 'int' }, { key: 'first', label: 'From' }, { key: 'last', label: 'To' },
            { key: 'fabs', label: 'Fabs reported (Mm⁻¹)', format: 'f2' }, { key: 'tau', label: 'Optical depth', format: 'f2' }, { key: 'under_read', label: 'Under-read', format: 'pct' }],
        },
      ],
    },
    {
      id: 'season',
      title: '3. Season: winter at Santiago, Pretoria, Johannesburg and Delhi; every month at Addis',
      text: '@season',
      figures: [
        {
          kind: 'heatmap', id: 'h1111-months', table: 'months', row: 'site', column: 'month_name', value: 'under_read', ramp: 'sequential', domain: [0, 0.5],
          format: 'pct', valueLabel: 'expected under-read (median filter of the month)', cell: [46, 22],
          title: 'Expected under-read by site and calendar month: Addis is above a third all year',
          subtitle: 'Median filter of each calendar month, all years together; months with fewer than four filters are blank. Sites in order of their overall median.',
          noteAlways: true, note: 'Months mix years and, at Delhi, Jaipur, Dhaka and Bujumbura, different sampled volumes. ' + CURVE,
        },
        {
          kind: 'table', id: 'h1111-swing', table: 'swing', title: 'How far each site swings between its best and worst month',
          columns: [{ key: 'site', label: 'Site' }, { key: 'best_month', label: 'Best month' }, { key: 'lowest', label: 'Under-read', format: 'pct' },
            { key: 'worst_month', label: 'Worst month' }, { key: 'highest', label: 'Under-read', format: 'pct' }, { key: 'swing', label: 'Swing', format: 'pct' }],
          subtitle: 'Sites with at least six months on file.',
        },
      ],
    },
    {
      id: 'group',
      title: '4. The affected filters as a group: defined by optical depth, and by how well the correction is known',
      text: '@group',
      figures: [
        {
          kind: 'bars', id: 'h1111-bands', table: 'bands', label: 'band', value: 'filters', color: 'band', format: 'int', xLabel: 'filters',
          title: 'Two thirds of the network\'s filters lose a fifth or less; 594 lose more than three tenths',
          subtitle: 'Every counted filter with HIPS and a sampled volume, by expected under-read.',
        },
        {
          kind: 'table', id: 'h1111-bands-table', table: 'bands', title: 'The correction is known to ±3 % up to a fifth, and to ±13 % beyond three tenths',
          subtitle: 'Factor: what the band\'s median filter is multiplied by to linearise it. ± is half the spread across the ceiling\'s interval; "by period" is the difference between IMPROVE 2005–2014 and 2015–2025.',
          columns: [{ key: 'band', label: 'Expected under-read' }, { key: 'tau_from', label: 'Optical depth from', format: 'f2' }, { key: 'tau_to', label: 'to', format: 'f2' },
            { key: 'filters', label: 'Filters', format: 'int' }, { key: 'share_of_network', label: 'Share', format: 'pct' }, { key: 'factor', label: 'Factor', format: 'f2' },
            { key: 'plus_minus', label: '±', format: 'pct' }, { key: 'by_period', label: 'by period', format: 'pct' },
            { key: 'beyond_improve', label: 'Darker than IMPROVE\'s range', format: 'pct' }, { key: 'mostly', label: 'Mostly' }],
          noteAlways: true, note: CURVE,
        },
        {
          kind: 'heatmap', id: 'h1111-bands-site', table: 'bands_by_site', row: 'site', column: 'band', value: 'share', ramp: 'sequential', domain: [0, 1], format: 'pct',
          valueLabel: 'share of the site\'s filters', cell: [84, 22],
          title: 'Where each site\'s filters fall: 95 % of Addis filters are in the last band',
          subtitle: 'Share of each site\'s filters in each band. Sites in order of their overall median.',
        },
        {
          kind: 'histogram', id: 'h1111-tau', table: 'points', value: 'tau', series: 'site', bins: 30, medians: true, xLabel: 'HIPS optical depth of the filter',
          choose: [{ column: 'site', label: 'Site', default: 'Addis Ababa, Ethiopia' }],
          title: 'One site\'s filters by optical depth: 0.45 is a fifth lost, 0.68 three tenths',
          subtitle: 'Choose a site. The optical depth is the laboratory\'s own HIPS tau, Fabs × volume ÷ 353.',
        },
      ],
    },
    {
      id: 'evidence',
      title: '5. Does each site\'s own data show it? Six of twelve as predicted; Addis, Delhi and the Chilean sites bend more',
      text: '@evidence',
      figures: [
        {
          kind: 'intervals', id: 'h1111-exponents', table: 'exponents', row: 'site', series: 'kind',
          title: 'Observed exponent of HIPS against FTIR EC beside what the curve predicts, for the twelve sites that can tell',
          subtitle: 'b in HIPS ∝ (FTIR EC)^b, with weighed PM2.5 as the instrument; 1 is proportional. Predicted: reported against linearised HIPS, same instrument, so volume changes are allowed for. Bars: 95 % bootstrap over filters (observed), the ceiling\'s interval (predicted). Heaviest sites first.',
          panels: [{ value: 'value', label: 'Response exponent', reference: 1, lo: 'lo', hi: 'hi' }],
          labelWidth: 220,
          noteAlways: true, note: 'An exponent against FTIR EC cannot separate a bend in HIPS from one in FTIR EC, and PM2.5 is a weak instrument where EC is a small or unsteady share of it (dust and sulfate sites). ' + OPTICAL,
        },
        {
          kind: 'scatter', id: 'h1111-agreement', table: 'told', x: 'predicted', y: 'observed', color: 'reading', idColumn: 'site', tipColumns: ['site', 'n', 'median_tau', 'lo', 'hi'],
          refX: [1], refY: [1], maxSide: 520, xLabel: 'exponent predicted from the IMPROVE curve', yLabel: 'exponent observed against FTIR EC',
          title: 'The twelve sites that can tell: observed follows predicted, with Addis, Delhi and the Chilean sites below and Johannesburg and Taipei above',
          subtitle: 'Sites whose interval is narrower than 0.6. On the diagonal the site bends as IMPROVE predicts.',
          noteAlways: true, note: OPTICAL,
        },
        {
          kind: 'table', id: 'h1111-evidence-table', table: 'evidence', title: 'All 23 sites with enough filters, including the eleven whose interval is too wide to read',
          columns: [{ key: 'site', label: 'Site' }, { key: 'n', label: 'Filters', format: 'int' }, { key: 'median_tau', label: 'Optical depth', format: 'f2' },
            { key: 'predicted', label: 'Predicted', format: 'f2' }, { key: 'observed', label: 'Observed', format: 'f2' }, { key: 'lo', label: 'low', format: 'f2' },
            { key: 'hi', label: 'high', format: 'f2' }, { key: 'observed_potassium', label: 'With potassium as instrument', format: 'f2' }, { key: 'reading', label: 'Reading' }],
        },
        {
          kind: 'table', id: 'h1111-tally', table: 'tally', title: 'The tally',
          columns: [{ key: 'reading', label: 'Reading' }, { key: 'sites', label: 'Sites', format: 'int' }, { key: 'which', label: 'Which' }],
        },
      ],
    },
    {
      id: 'decision',
      title: '6. What to do: the options, with the case for and against each',
      text: '@decision',
      figures: [
        {
          kind: 'table', id: 'h1111-options', table: 'options', title: 'Six ways to handle it, and where each stands on the evidence above',
          columns: [{ key: 'option', label: 'Option' }, { key: 'what', label: 'What it means' }, { key: 'for', label: 'For' }, { key: 'against', label: 'Against' }, { key: 'verdict', label: 'Where it stands' }],
          noteAlways: true, note: CURVE + ' ' + OPTICAL,
        },
      ],
    },
  ],
} satisfies MeetingSpec
