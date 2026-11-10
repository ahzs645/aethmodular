import type { MeetingSpec } from './types'

// "Which measurement carries the Addis offset": the answer to the question the 9 Nov status page left open, from
// data already in hand and without the aethalometer (the Addis MA350 is not a reference). A results page, not a
// recorded meeting; every table comes from gallery/data/export_meeting_20261110.py
// (workflow run_addis_reference_closure_20261110.py). Titles carry the message: section text shows only with Explain on.
const OPTICAL = 'HIPS is optical and no Addis filter has thermal (TOR) EC: this shows the shape of the comparison, not the Addis EC level. No aethalometer data are used.'

export default {
  id: 'closure-20261110',
  label: 'Nov 10, 2026',
  date: '2026-11-10',
  title: 'Which measurement carries the Addis offset: most of it is HIPS compression, shown with IMPROVE thermal EC',
  command: 'uv run python gallery/data/export_meeting_20261110.py',
  intro: '@intro',
  sections: [
    {
      id: 'improve',
      title: '1. IMPROVE thermal EC: HIPS falls behind as the filter darkens',
      text: '@improve',
      columns: 2,
      figures: [
        {
          kind: 'sweep', id: 'c1110-improve-ratio', table: 'strata', x: 'HIPS optical depth', y: 'Fabs ÷ TOR EC (m²/g)', series: 'population',
          xLabel: 'HIPS optical depth (group median)', yLabel: 'HIPS Fabs ÷ TOR EC (m²/g)',
          xMarkers: [{ at: 0.96, label: 'Addis median' }],
          title: 'Per unit of thermal EC, HIPS reads about a third less at Addis optical depth than on light filters',
          subtitle: 'IMPROVE samples with TOR EC and HIPS on the same day, 2005–2025, grouped by the previous sample’s EC loading at the same site so that neither instrument’s error decides the group. Each point is a group median.',
          noteAlways: true,
          note: 'IMPROVE only: thermal-optical (TOR) EC on quartz against HIPS on PTFE. No FTIR and no aethalometer. Heavy IMPROVE filters include smoke, whose composition differs from Addis.',
        },
        {
          kind: 'table', id: 'c1110-ceilings', table: 'ceilings',
          title: 'The ceiling is the same whichever IMPROVE sites are used, but it differs between 2005–2014 and 2015–2025',
          subtitle: 'tau = aL / (1 + aL / ceiling) through the group medians; intervals resample whole sites (90 %). The last three columns are what each ceiling implies on the Addis filters.',
          columns: [{ key: 'population', label: 'IMPROVE samples' }, { key: 'kind', label: '' }, { key: 'samples', format: 'int' }, { key: 'sites', format: 'int' },
            { key: 'tau_inf', label: 'ceiling', format: 'f2' }, { key: 'tau_inf_lo', label: 'low', format: 'f2' }, { key: 'tau_inf_hi', label: 'high', format: 'f2' },
            { key: 'unsaturated_fabs_over_ec', label: 'Fabs ÷ EC on light filters (m²/g)', format: 'f2' },
            { key: 'addis_predicted_exponent', label: 'Addis exponent predicted', format: 'f2' },
            { key: 'addis_slope', label: 'Addis slope after linearising', format: 'f2' }, { key: 'addis_intercept', label: 'Addis intercept after (µg/m³)', format: 'f2' }],
          noteAlways: true,
          note: 'The ceiling is the main uncertainty in everything below: across these rows the Addis intercept left after linearising runs from clearly positive to slightly negative. Measured Addis exponent: 0.41 to 0.46.',
        },
      ],
    },
    {
      id: 'addis',
      title: '2. At Addis, HIPS and FTIR EC are not proportional',
      text: '@addis',
      figures: [
        {
          kind: 'intervals', id: 'c1110-exponents', table: 'exponents', row: 'pair', series: 'instrument',
          title: 'HIPS rises as about the square root of FTIR EC; the IMPROVE curve predicts 0.56 for HIPS alone',
          subtitle: 'Response exponent b in Y ∝ X^b on the Addis filters, estimated with a third measurement as the instrument so noise in X does not flatten it. 1 = proportional. Bars: 95 % bootstrap over filters.',
          panels: [{ value: 'exponent', label: 'Response exponent', reference: 1, lo: 'lo', hi: 'hi' }],
          labelWidth: 400,
          noteAlways: true,
          note: 'PM2.5 mass is gravimetric and potassium is XRF: neither is optical or FTIR. EC’s share of PM2.5 can itself change with loading, so the PM2.5 rows show scaling, not a reference. ' + OPTICAL,
        },
      ],
    },
    {
      id: 'response',
      title: '3. Linearising HIPS with the IMPROVE ceiling takes the intercept to zero',
      text: '@response',
      columns: 2,
      figures: [
        {
          kind: 'crossplot', id: 'c1110-against-hips', table: 'against_hips', x: 'x', y: 'ftir_published', color: 'season', facet: 'comparator',
          fit: true, errorRatio: 2.96, sameScale: false, idColumn: 'filter', maxSide: 380,
          title: 'The same filters: 2.31x − 6.17 as measured, 0.55x + 0.01 once HIPS is linearised',
          subtitle: 'SPARTAN published FTIR EC against HIPS optical BC (MAC 10), as measured and after inverting the saturating response fitted on IMPROVE thermal EC (ceiling 2.26). Deming fits.',
          xLabel: 'HIPS optical BC (µg/m³, MAC 10)', yLabel: 'FTIR EC, SPARTAN published (µg/m³)',
          noteAlways: true,
          note: 'The ceiling’s interval is wide (1.82 to 2.70) and the intercept moves from +1.6 to −1.0 across it; the correction multiplies noise on the heaviest filters. ' + OPTICAL,
        },
        {
          kind: 'table', id: 'c1110-fits', table: 'fits', title: 'The fits, across the ceiling’s interval',
          columns: [{ key: 'comparator', label: 'Comparator' }, { key: 'model', label: 'FTIR EC' }, { key: 'n', format: 'int' },
            { key: 'slope', format: 'f2' }, { key: 'intercept', label: 'intercept (µg/m³)', format: 'f2' }],
        },
      ],
    },
    {
      id: 'season',
      title: '4. Seasons and filter lots',
      text: '@season',
      figures: [
        {
          kind: 'table', id: 'c1110-season', table: 'season',
          title: 'Linearising HIPS removes most of the seasonal swing; a smaller one remains, and lot 248 reads low',
          subtitle: 'Median ratios of the published FTIR EC by season and filter lot. Kiremt has the heaviest filters, so season and loading cannot be separated here.',
          columns: [{ key: 'group', label: 'Season, lot' }, { key: 'n', format: 'int' }, { key: 'median_optical_depth', label: 'HIPS optical depth', format: 'f2' },
            { key: 'published_over_hips', label: 'FTIR EC ÷ HIPS / 10', format: 'f2' },
            { key: 'published_over_linearised_hips', label: 'FTIR EC ÷ linearised HIPS / 10', format: 'f2' },
            { key: 'published_over_pm25', label: 'FTIR EC ÷ PM2.5 mass', format: 'f2' }],
          note: 'EC’s share of PM2.5 can really change with season, so the last column is context, not a test.',
        },
      ],
    },
    {
      id: 'level',
      title: '5. What is left: the slope is the FTIR level times the MAC',
      text: '@level',
      columns: 2,
      figures: [
        {
          kind: 'table', id: 'c1110-mass', table: 'mass',
          title: 'PM2.5 mass closure caps the level: a factor near 2 over-fills the mass',
          subtitle: 'Reconstructed mass (FTIR OM + EC + sulfate, nitrate, ammonium + soil + trace elements) over gravimetric PM2.5, for EC multipliers.',
          columns: [{ key: 'EC multiplier', format: 'f2' }, { key: 'n', format: 'int' },
            { key: 'reconstructed ÷ gravimetric PM2.5', format: 'f2' }, { key: 'EC share of PM2.5', format: 'f2' }],
          note: 'FTIR OM is below FTIR OC at Addis, so OM is probably low and the real room for EC is smaller.',
        },
        {
          kind: 'table', id: 'c1110-anchors', table: 'anchors', title: 'Every anchor on the level, and what limits it',
          columns: [{ key: 'anchor', label: 'Anchor' }, { key: 'published ÷ true', format: 'f2' }, { key: 'value', format: 'f2' }, { key: 'note', label: 'Limit' }],
          noteAlways: true,
          note: 'Working statement: Addis EC is the published value × 1.0 to 1.4, most likely about × 1.1. A range from soft anchors, not a validated correction.',
        },
      ],
    },
    {
      id: 'network',
      title: '6. Every SPARTAN site',
      text: '@network',
      figures: [
        {
          kind: 'scatter', id: 'c1110-network', table: 'network', x: 'median_optical_depth', y: 'exponent', color: 'what',
          idColumn: 'site', tipColumns: ['site', 'n', 'lo', 'hi'], refY: [1], maxSide: 460,
          title: 'The exponent falls with a site’s optical depth about as the IMPROVE thermal curve predicts; Addis is the heaviest site',
          subtitle: 'Exponent of HIPS against FTIR EC at each SPARTAN site, with PM2.5 mass and with potassium as the instrument, and the value the IMPROVE curve predicts for HIPS alone, against the site’s median HIPS optical depth. Sites with an interval wider than 0.6 are left off.',
          xLabel: 'Median HIPS optical depth', yLabel: 'Exponent of HIPS against FTIR EC',
          note: 'The two Chilean sites (0.56 at optical depth 0.2 and 0.4, with either instrument) sit far below the prediction; the South African sites sit above it.',
        },
        {
          kind: 'bars', id: 'c1110-site-intercepts', table: 'site_fits', label: 'site', value: 'intercept', color: 'comparator', sort: 'none', reference: 0, format: 'f2',
          choose: [{ column: 'comparator', label: 'Against', default: 'HIPS as measured' }],
          xLabel: 'Intercept of FTIR EC against HIPS BC (µg/m³)',
          title: 'Site by site: the intercept of FTIR EC against HIPS, as measured and once HIPS is linearised',
          subtitle: 'Deming fits per site (the Addis error ratio everywhere, so sites are comparable), ordered by median optical depth. Switch "Against" to see what one network-wide ceiling does: it removes the large negative intercepts and over-corrects some sites that were already near zero.',
          noteAlways: true,
          note: 'HIPS optical BC at MAC 10, not thermal EC. One ceiling from IMPROVE (2.26) is applied to every site; it is not a per-site correction.',
        },
        {
          kind: 'crossplot', id: 'c1110-site-points', table: 'site_points', x: 'x', y: 'ftir_ec', color: 'year', facet: 'comparator',
          choose: [{ column: 'site', label: 'Site', default: 'Addis Ababa, Ethiopia' }],
          fit: true, errorRatio: 2.96, sameScale: false, idColumn: 'id', tipColumns: ['date', 'tau'], maxSide: 400,
          title: 'One site’s filters: FTIR EC against HIPS as measured and linearised',
          subtitle: 'Pick a site. Each dot is a filter, coloured by year. Left: HIPS Fabs / 10 as reported. Right: after inverting the IMPROVE thermal response (ceiling 2.26).',
          xLabel: 'HIPS optical BC (µg/m³, MAC 10)', yLabel: 'FTIR EC (µg/m³)',
          noteAlways: true,
          note: OPTICAL,
        },
        {
          kind: 'table', id: 'c1110-site-table', table: 'site_table', title: 'Every site with at least 40 complete filters',
          columns: [{ key: 'site', label: 'Site' }, { key: 'n', format: 'int' }, { key: 'median_optical_depth', label: 'optical depth', format: 'f2' },
            { key: 'share_above_half', label: 'share above 0.5', format: 'pct' },
            { key: 'hips_exponent_vs_ftir_ec', label: 'exponent (PM2.5)', format: 'f2' }, { key: 'lo', format: 'f2' }, { key: 'hi', format: 'f2' },
            { key: 'exponent_potassium', label: 'exponent (potassium)', format: 'f2' }, { key: 'predicted_from_improve', label: 'predicted', format: 'f2' },
            { key: 'intercept_measured', label: 'intercept as measured', format: 'f2' }, { key: 'intercept_linearised', label: 'intercept linearised', format: 'f2' },
            { key: 'median_ftir_ec', label: 'median FTIR EC', format: 'f2' }],
        },
      ],
    },
    {
      id: 'candidates',
      title: '7. Every saved Addis calibration, by season',
      text: '@candidates',
      figures: [
        {
          kind: 'intervals', id: 'c1110-cand-season', table: 'cand_season', row: 'calibration', series: 'season',
          choose: [{ column: 'comparator', label: 'Against', default: 'HIPS as measured' }],
          title: 'The seasonal spread of every calibration shrinks once HIPS is linearised',
          subtitle: 'Median FTIR EC ÷ HIPS BC by season for each calibration the 7 Nov closeout saved per filter. Switch "Against" between HIPS as measured and linearised. Seasons: Bega Oct–Jan, Belg Feb–May, Kiremt Jun–Sep.',
          panels: [{ value: 'median_ratio', label: 'Median FTIR EC ÷ HIPS BC' }, { value: 'intercept', label: 'Intercept (µg/m³)', reference: 0 }],
          labelWidth: 230,
          noteAlways: true,
          note: 'Kiremt has the heaviest filters, so a remaining spread can be season or loading. ' + OPTICAL,
        },
        {
          kind: 'table', id: 'c1110-cand-overall', table: 'cand_overall', title: 'All seasons together',
          choose: [{ column: 'comparator', label: 'Against', default: 'HIPS linearised' }],
          columns: [{ key: 'calibration', label: 'Calibration' }, { key: 'n', format: 'int' }, { key: 'slope', format: 'f2' },
            { key: 'intercept', label: 'intercept (µg/m³)', format: 'f2' }, { key: 'median_ratio', label: 'median FTIR ÷ HIPS', format: 'f2' }],
        },
      ],
    },
    {
      id: 'source',
      title: '8. The FTIR side, tested against thermal EC',
      text: '@source',
      figures: [
        {
          kind: 'intervals', id: 'c1110-ftir-level', table: 'ftir_level', row: 'filters', series: 'grouping',
          title: 'FTIR reads about a tenth low on the most Addis-like IMPROVE filters; the larger figure came from how the filters were chosen',
          subtitle: 'Median FTIR EC ÷ TOR EC on held-out IMPROVE sites (raw spectra, 18 components, refitted with whole sites held out), for three ways of choosing the filters. Bars: 95 %, resampling sites.',
          panels: [{ value: 'ftir_over_tor', label: 'FTIR EC ÷ TOR EC', reference: 1, lo: 'lo', hi: 'hi' }],
          labelWidth: 330,
          noteAlways: true,
          note: 'Choosing filters by their own TOR EC or OC/EC selects on TOR’s measurement error and biases the ratio low. Choosing by site, or by the previous sample at the site, cannot. Even these groups have TOR OC/EC near 3, far above Addis.',
        },
        {
          kind: 'histogram', id: 'c1110-ftir-ocec', table: 'ftir_values', value: 'ftir_ocec', series: 'population', bins: 40, medians: true,
          xLabel: 'FTIR OC ÷ FTIR EC (both from IMPROVE-trained calibrations)',
          title: 'Addis is outside IMPROVE’s composition range: 99 % of Addis filters are below IMPROVE’s 5th percentile',
          subtitle: 'OC and EC predicted by calibrations fitted the same way on both networks, so the ratio means the same thing. IMPROVE values are held-out predictions, clipped to −1…15.',
          noteAlways: true,
          note: 'This is why IMPROVE cannot give the FTIR bias at Addis: no IMPROVE population has Addis’s composition.',
        },
        {
          kind: 'table', id: 'c1110-ftir-candidates', table: 'ftir_candidates',
          title: 'The saved calibrations differ little once the filters are chosen fairly',
          subtitle: 'Median held-out FTIR ÷ TOR EC per calibration (7 Nov closeout predictions): the six urban sites, and the closeout’s heavy low-OC/EC subset.',
          columns: [{ key: 'calibration', label: 'Calibration' }, { key: 'urban_n', label: 'urban n', format: 'int' }, { key: 'urban_sites', label: 'urban sites', format: 'f2' },
            { key: 'urban_sum_ratio', label: 'urban, sum ÷ sum', format: 'f2' }, { key: 'selected_n', label: 'subset n', format: 'int' },
            { key: 'selected_on_own_tor', label: 'chosen by own TOR', format: 'f2' }],
          note: 'The per-filter EPO predictions are not in the closeout export, so EPO is not re-scored here; its 0.90 was on the selected subset.',
        },
        {
          kind: 'intervals', id: 'c1110-source', table: 'source', row: 'filters', series: 'calibration',
          title: 'On held-out IMPROVE sites FTIR EC is proportional to thermal EC',
          subtitle: 'Exponent of the held-out FTIR prediction against TOR EC, instrument HIPS (cross-validation by site, 7 Nov closeout). 1 = proportional. Bars: 95 % bootstrap.',
          panels: [{ value: 'exponent', label: 'Exponent of FTIR EC against TOR EC', reference: 1, lo: 'lo', hi: 'hi' }],
          labelWidth: 170,
        },
        {
          kind: 'table', id: 'c1110-lots', table: 'lots', title: 'IMPROVE filters of lot 248 read thermal EC no lower than lot 251',
          subtitle: 'Median held-out FTIR ÷ TOR EC by filter lot.',
          columns: [{ key: 'filters', label: 'Filters' }, { key: 'lot', label: 'Lot' }, { key: 'n', format: 'int' }, { key: 'Raw spectra', format: 'f2' },
            { key: 'PTFE-anchored VIBES', format: 'f2' }, { key: 'Spline baseline', format: 'f2' }],
        },
      ],
    },
    {
      id: 'next',
      title: '9. What this changes, and what is still open',
      text: '@next',
      figures: [
        {
          kind: 'table', id: 'c1110-rereads', table: 'rereads', title: 'Earlier readings that used HIPS / 10 as the yardstick',
          columns: [{ key: 'earlier reading', label: 'Earlier reading' }, { key: 'with HIPS linearised', label: 'With HIPS linearised' }],
        },
        {
          kind: 'table', id: 'c1110-pathways', table: 'pathways', title: 'Pathways: done, open with data in hand, blocked',
          columns: [{ key: 'rank', label: '#' }, { key: 'pathway', label: 'Pathway' }, { key: 'status', label: 'Status' }],
        },
      ],
    },
  ],
} satisfies MeetingSpec
