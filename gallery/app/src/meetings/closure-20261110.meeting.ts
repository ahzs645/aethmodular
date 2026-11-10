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
          title: 'The saturation ceiling is the same whichever IMPROVE sites are used',
          subtitle: 'tau = aL / (1 + aL / ceiling) through the group medians; intervals resample whole sites (90 %).',
          columns: [{ key: 'population', label: 'IMPROVE sites' }, { key: 'samples', format: 'int' }, { key: 'sites', format: 'int' },
            { key: 'tau_inf', label: 'ceiling', format: 'f2' }, { key: 'tau_inf_lo', label: 'low', format: 'f2' }, { key: 'tau_inf_hi', label: 'high', format: 'f2' },
            { key: 'unsaturated_fabs_over_ec', label: 'Fabs ÷ EC on light filters (m²/g)', format: 'f2' }, { key: 'max_stratum_tau', label: 'highest group optical depth', format: 'f2' }],
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
      title: '5. What is left for FTIR: mainly one level factor',
      text: '@level',
      columns: 2,
      figures: [
        {
          kind: 'table', id: 'c1110-ocec', table: 'ocec',
          title: 'On held-out IMPROVE sites, FTIR reads thermal EC lower the lower the OC/EC',
          subtitle: 'Median FTIR ÷ TOR EC at TOR ≥ 15 µg per filter, by TOR OC/EC (cross-validation by site, 7 Nov closeout). Addis FTIR OC/EC has a median of 1.34.',
          columns: [{ key: 'tor_oc_over_ec', label: 'TOR OC/EC' }, { key: 'n', format: 'int' }, { key: 'sites', format: 'int' },
            { key: 'Raw spectra', format: 'f2' }, { key: 'PTFE-anchored VIBES', format: 'f2' }, { key: 'Spline baseline', format: 'f2' },
            { key: 'Anchored, OC/EC-balanced weights', format: 'f2' }],
        },
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
          note: 'Working statement: Addis EC is the published value × 1.0 to 1.5, most likely about × 1.2. A range from soft anchors, not a validated correction.',
        },
      ],
    },
    {
      id: 'network',
      title: '6. The same test across SPARTAN',
      text: '@network',
      figures: [
        {
          kind: 'scatter', id: 'c1110-network', table: 'network', x: 'median_optical_depth', y: 'exponent', color: 'what',
          idColumn: 'site', tipColumns: ['site', 'n', 'lo', 'hi'], refY: [1], maxSide: 460,
          title: 'The exponent falls with a site’s optical depth about as the IMPROVE thermal curve predicts; Addis is the heaviest site',
          subtitle: 'Exponent of HIPS against FTIR EC at each SPARTAN site (instrument: PM2.5 mass) and the value the IMPROVE curve predicts for HIPS alone, against the site’s median HIPS optical depth. Sites with an interval wider than 0.6 are left off.',
          xLabel: 'Median HIPS optical depth', yLabel: 'Exponent of HIPS against FTIR EC',
          note: 'A first pass. The two Chilean sites (0.56 at optical depth 0.2 and 0.4) sit far below the prediction and are unexplained.',
        },
      ],
    },
    {
      id: 'next',
      title: '7. What this changes, and what is still open',
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
