import type { MeetingSpec } from './types'

// "Which measurement carries the Addis offset": the answer to the question the 9 Nov status page left open, from
// data already in the repository. A results page, not a recorded meeting; every table comes from
// gallery/data/export_meeting_20261110.py (workflow run_addis_reference_closure_20261110.py).
// Titles carry the message, since section text shows only with Explain on.
const OPTICAL = 'Optical comparators, not thermal EC: no Addis filter has TOR EC, so this shows shape (who is proportional to whom), not the EC level.'

export default {
  id: 'closure-20261110',
  label: 'Nov 10, 2026',
  date: '2026-11-10',
  title: 'Which measurement carries the Addis offset: HIPS bends, FTIR EC does not',
  command: 'uv run python gallery/data/export_meeting_20261110.py',
  intro: '@intro',
  sections: [
    {
      id: 'linearity',
      title: '1. Four independent measurements: HIPS is the one that is not proportional',
      text: '@linearity',
      figures: [
        {
          kind: 'intervals', id: 'c1110-exponents', table: 'exponents', row: 'pair', series: 'instrument',
          title: 'HIPS rises as about the square root of every other measurement; FTIR EC is proportional to the aethalometer',
          subtitle: 'Response exponent b in Y ∝ X^b on the Addis filters, estimated with a third measurement as the instrument so noise in X does not flatten it. 1 = proportional. Bars: 95 % bootstrap over filters.',
          panels: [{ value: 'exponent', label: 'Response exponent', reference: 1, lo: 'lo', hi: 'hi' }],
          labelWidth: 400,
          noteAlways: true,
          note: 'PM2.5 mass is gravimetric and potassium is XRF: neither is optical or FTIR. ' + OPTICAL,
        },
        {
          kind: 'scatter', id: 'c1110-against-aeth', table: 'against_aeth', x: 'aethalometer', y: 'value', color: 'season', facet: 'measure',
          idColumn: 'filter', maxSide: 380,
          title: 'Against the aethalometer, HIPS flattens and FTIR EC runs straight through the origin',
          subtitle: 'Each dot is an Addis filter (172 with an aethalometer match), coloured by season. The aethalometer has its own calibration, so read the shape, not the slope.',
          xLabel: 'Aethalometer IR BCc (µg/m³, MA350)', yLabel: 'µg/m³',
        },
        {
          kind: 'intervals', id: 'c1110-by-season', table: 'by_season', row: 'season', series: 'y',
          title: 'The bend is inside every season, so it is not a between-season composition effect',
          subtitle: 'The same exponent against the aethalometer within each season (instrument: PM2.5 mass).',
          panels: [{ value: 'exponent', label: 'Response exponent', reference: 1, lo: 'lo', hi: 'hi' }],
          labelWidth: 80,
        },
      ],
    },
    {
      id: 'response',
      title: '2. Linearising HIPS with a constant fitted at other sites removes the offset',
      text: '@response',
      columns: 2,
      figures: [
        {
          kind: 'crossplot', id: 'c1110-against-hips', table: 'against_hips', x: 'x', y: 'ftir_published', color: 'season', facet: 'comparator',
          fit: true, errorRatio: 2.96, sameScale: false, idColumn: 'filter', maxSide: 380,
          title: 'The same filters: 2.31x − 6.17 as measured, 0.44x + 0.58 once HIPS is linearised',
          subtitle: 'SPARTAN published FTIR EC against HIPS optical BC (MAC 10), as measured and after inverting a saturating response whose ceiling was fitted on Beijing, Pasadena and Delhi with Addis held out. Deming fits.',
          xLabel: 'HIPS optical BC (µg/m³, MAC 10)', yLabel: 'FTIR EC, SPARTAN published (µg/m³)',
          noteAlways: true,
          note: 'The ceiling is empirical and its interval is wide (1.75 to 2.49); the correction multiplies noise on the heaviest filters. ' + OPTICAL,
        },
        {
          kind: 'table', id: 'c1110-fits', table: 'fits', title: 'The fits, across the ceiling’s interval',
          columns: [{ key: 'comparator', label: 'Comparator' }, { key: 'model', label: 'FTIR EC' }, { key: 'n', format: 'int' },
            { key: 'slope', format: 'f2' }, { key: 'intercept', label: 'intercept (µg/m³)', format: 'f2' }],
        },
      ],
    },
    {
      id: 'errors',
      title: '3. Random error with no reference',
      text: '@errors',
      figures: [
        {
          kind: 'intervals', id: 'c1110-errors', table: 'errors', row: 'measurement',
          title: 'FTIR EC is about as precise as the aethalometer; HIPS is somewhat noisier once its compressed range is allowed for',
          subtitle: 'Random error as a percentage, from three-way agreement alone (log-space triple collocation, 172 filters), on a common scale. Bars: 95 % bootstrap.',
          panels: [{ value: 'error_sd', label: 'Random error (%)', lo: 'error_sd_lo', hi: 'error_sd_hi' }],
          labelWidth: 220,
          note: 'Assumes the three errors are independent. Scatter about the shared signal, not accuracy of the level.',
        },
      ],
    },
    {
      id: 'season',
      title: '4. Seasons and filter lots, with the aethalometer as the comparator',
      text: '@season',
      figures: [
        {
          kind: 'table', id: 'c1110-season', table: 'season',
          title: 'Within lot 251 the published FTIR EC ÷ aethalometer is the same in every season',
          subtitle: 'Median ratio to the aethalometer IR BCc by season and filter lot.',
          columns: [{ key: 'group', label: 'Season, lot' }, { key: 'n', format: 'int' },
            { key: 'published_over_aethalometer', label: 'FTIR EC, published', format: 'f2' },
            { key: 'anchored_over_aethalometer', label: 'FTIR EC, PTFE-anchored VIBES', format: 'f2' },
            { key: 'hips_mac10_over_aethalometer', label: 'HIPS Fabs / 10', format: 'f2' }],
        },
      ],
    },
    {
      id: 'level',
      title: '5. What is left for FTIR: one level factor',
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
          kind: 'scatter', id: 'c1110-network', table: 'network', x: 'median_optical_depth', y: 'hips_exponent_vs_ftir_ec', color: 'which',
          idColumn: 'site', tipColumns: ['site', 'n', 'lo', 'hi'], refY: [1], maxSide: 460,
          title: 'Addis has the highest optical depth in the network and the lowest exponent',
          subtitle: 'Exponent of HIPS against FTIR EC at each SPARTAN site (instrument: PM2.5 mass), against the site’s median HIPS optical depth. Sites with an interval wider than 0.6 are left off.',
          xLabel: 'Median HIPS optical depth', yLabel: 'Exponent of HIPS against FTIR EC',
          note: 'A first pass. With two EC-like measures the exponent alone cannot say which one bends at a site; at Addis the aethalometer and potassium say it is HIPS.',
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
          columns: [{ key: 'earlier reading', label: 'Earlier reading' }, { key: 'with a linear comparator', label: 'With a linear comparator' }],
        },
        {
          kind: 'table', id: 'c1110-pathways', table: 'pathways', title: 'Pathways: done, open with data in hand, blocked',
          columns: [{ key: 'rank', label: '#' }, { key: 'pathway', label: 'Pathway' }, { key: 'status', label: 'Status' }],
        },
      ],
    },
  ],
} satisfies MeetingSpec
