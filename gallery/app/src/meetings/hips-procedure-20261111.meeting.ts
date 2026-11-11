import type { MeetingSpec } from './types'

// "HIPS on loaded filters: the procedure": the end product of the HIPS work as four steps that run on any set of
// filters (research/ftir_hips_chem/scripts/hips_loading.py, `aeth hips screen`, docs/hips-loading-protocol.md), shown
// here on every SPARTAN site. Tables from gallery/data/export_meeting_20261111_procedure.py. Titles carry the
// message: section text shows only with Explain on.
const CURVE = 'The ceiling is IMPROVE\'s, 2.26 [1.82, 2.70], measured against thermal EC. SPARTAN has no thermal EC to check it on its own filters.'
const REFERENCE = 'The site tests use FTIR EC, which is FTIR-derived and not thermal EC, so they cannot separate a bend in HIPS from one in FTIR EC. No aethalometer data are used.'

export default {
  id: 'hips-procedure-20261111',
  label: 'Nov 11, 2026',
  date: '2026-11-11',
  short: 'HIPS: the procedure',
  title: 'HIPS on loaded filters: identify, trace, correct, check',
  command: 'uv run python gallery/data/export_meeting_20261111_procedure.py',
  intro: '@intro',
  sections: [
    {
      id: 'steps',
      title: 'The procedure at a glance: four steps, one command, three tables',
      figures: [
        {
          kind: 'table', id: 'hp-steps', table: 'steps', title: 'Identify, trace, correct, check: what each step asks, computes, needs and gives',
          subtitle: 'Run with: aeth hips screen (every SPARTAN site), or aeth hips screen --input file.csv --fabs … --volume … for any other set of filters.',
          columns: [{ key: 'step', label: 'Step' }, { key: 'question', label: 'Question' }, { key: 'computed', label: 'What is computed' }, { key: 'needs', label: 'Needs' }, { key: 'gives', label: 'Gives' }],
          noteAlways: true, note: CURVE,
        },
      ],
    },
    {
      id: 'identify',
      title: 'Step 1. Identify: each filter by its optical depth, each site by its own data',
      text: '@identify',
      figures: [
        {
          kind: 'heatmap', id: 'hp-status', table: 'status', row: 'site', column: 'status', value: 'share', ramp: 'sequential', domain: [0, 1], format: 'pct',
          valueLabel: 'share of the site\'s filters', cell: [120, 22],
          title: 'Every filter gets a status: Addis has none it can use for level; thirteen sites can use almost all',
          subtitle: 'Use, linearised: expected under-read up to a fifth. Caution: a fifth to three tenths. Shape only: beyond. Sites in order of optical depth.',
          noteAlways: true, note: CURVE,
        },
        {
          kind: 'intervals', id: 'hp-ratios', table: 'ratios', row: 'site', series: 'kind',
          title: 'A test that assumes no curve: Fabs ÷ FTIR EC on the most loaded third of filters over the least loaded third',
          subtitle: 'Below 1, HIPS falls behind as filters load. Thirds are chosen by PM2.5 on the filter, not by HIPS or FTIR EC, which would select on their own errors. Bars: 90 % bootstrap. Beside each: what the curve predicts.',
          panels: [{ value: 'value', label: 'Heavy third ÷ light third', reference: 1, lo: 'lo', hi: 'hi' }], labelWidth: 220,
          noteAlways: true, note: REFERENCE,
        },
        {
          kind: 'table', id: 'hp-found', table: 'found', title: 'Each site\'s reading: expected and seen at eight, expected but untestable at five',
          columns: [{ key: 'site', label: 'Site' }, { key: 'filters', label: 'Filters', format: 'int' }, { key: 'median_tau', label: 'Optical depth', format: 'f2' },
            { key: 'expected_under_read', label: 'Expected under-read', format: 'pct' }, { key: 'share_use', label: 'Use', format: 'pct' }, { key: 'share_shape_only', label: 'Shape only', format: 'pct' },
            { key: 'exponent', label: 'Exponent', format: 'f2' }, { key: 'exponent_lo', label: 'low', format: 'f2' }, { key: 'exponent_hi', label: 'high', format: 'f2' },
            { key: 'ratio_heavy_over_light', label: 'Heavy ÷ light', format: 'f2' }, { key: 'ratio_predicted', label: 'predicted', format: 'f2' }, { key: 'reading', label: 'Reading' }],
          subtitle: 'Expected: the median filter loses more than a tenth. Seen: the exponent\'s interval is below 1, or the heavy-over-light interval is.',
        },
      ],
    },
    {
      id: 'trace',
      title: 'Step 2. Trace: what the bend does to each number read off HIPS',
      text: '@trace',
      figures: [
        {
          kind: 'table', id: 'hp-traced', table: 'traced', choose: [{ column: 'site', label: 'Site', default: 'Addis Ababa, Ethiopia' }],
          title: 'One site, every number: reported, linearised, and the range the ceiling allows',
          columns: [{ key: 'quantity', label: 'Read off HIPS' }, { key: 'reported', label: 'Reported', format: 'f2' }, { key: 'linearised', label: 'Linearised', format: 'f2' },
            { key: 'lo', label: 'low', format: 'f2' }, { key: 'hi', label: 'high', format: 'f2' }, { key: 'change', label: 'Change', format: 'pct' }],
          subtitle: 'Choose a site. BC is Fabs ÷ 10. The seasonal contrast is the three highest calendar-month medians over the three lowest.',
          noteAlways: true, note: CURVE,
        },
        {
          kind: 'heatmap', id: 'hp-change', table: 'change', row: 'site', column: 'quantity', value: 'change', ramp: 'diverging', domain: [-0.8, 0.8], format: 'pct',
          valueLabel: 'change when HIPS is linearised', cell: [128, 22],
          title: 'How far each number moves at each site: levels rise, seasons widen, slopes fall',
          subtitle: 'Linearised over reported, minus one. Blank where a site has too few filters with FTIR EC or too few months.',
        },
        {
          kind: 'crossplot', id: 'hp-ranks', table: 'ranks', x: 'reported', y: 'linearised', idColumn: 'site', tipColumns: ['site', 'rank_reported', 'rank_linearised', 'lo', 'hi'],
          xLabel: 'median BC as reported (µg/m³)', yLabel: 'median BC linearised (µg/m³)', sameScale: true, maxSide: 520,
          title: 'The order of sites changes: Delhi reports the most BC, but linearised Addis and Delhi are level',
          subtitle: 'One point per site, BC = Fabs ÷ 10. On the diagonal nothing changes; the heavy sites lift off it.',
          noteAlways: true, note: CURVE,
        },
        {
          kind: 'table', id: 'hp-fits', table: 'fits', title: 'Straight-line fits of FTIR EC on HIPS BC: the slope above 1 with a negative intercept is the bend\'s signature',
          columns: [{ key: 'site', label: 'Site' }, { key: 'slope_reported', label: 'Slope, reported', format: 'f2' }, { key: 'intercept_reported', label: 'Intercept', format: 'f2' },
            { key: 'slope_linearised', label: 'Slope, linearised', format: 'f2' }, { key: 'intercept_linearised', label: 'Intercept', format: 'f2' }],
          subtitle: 'Deming fits with the same error ratio at every site, so sites compare with each other. Sites with 40 filters carrying FTIR EC.',
          noteAlways: true, note: REFERENCE,
        },
      ],
    },
    {
      id: 'correct',
      title: 'Step 3. Correct: one formula, and a status that says what the answer may be used for',
      text: '@correct',
      figures: [
        {
          kind: 'table', id: 'hp-by-status', table: 'by_status', title: 'The correction is known to ±2 % where it is small and ±13 % where it is large',
          columns: [{ key: 'status', label: 'Status' }, { key: 'under_read', label: 'Expected under-read' }, { key: 'tau', label: 'Optical depth' }, { key: 'filters', label: 'Filters', format: 'int' },
            { key: 'share', label: 'Share', format: 'pct' }, { key: 'factor', label: 'Factor (median filter)', format: 'f2' }, { key: 'factor_lo', label: 'low', format: 'f2' },
            { key: 'factor_hi', label: 'high', format: 'f2' }, { key: 'plus_minus', label: '±', format: 'pct' }, { key: 'use', label: 'What it may be used for' }],
          noteAlways: true, note: CURVE,
        },
        {
          kind: 'scatter', id: 'hp-points', table: 'points', x: 'fabs', y: 'fabs_linearised', color: 'status', idColumn: 'id',
          tipColumns: ['id', 'date', 'volume', 'tau', 'fabs_linearised_lo', 'fabs_linearised_hi'], choose: [{ column: 'site', label: 'Site', default: 'Delhi, India' }],
          xLabel: 'Fabs as reported (Mm⁻¹)', yLabel: 'Fabs linearised (Mm⁻¹)', maxSide: 560,
          title: 'Every filter, reported against linearised: the correction grows with the filter\'s darkness, not the site',
          subtitle: 'One point per filter at the chosen site; hover for its range. At Delhi the same reported Fabs gets different corrections, because filters were sampled at three volumes.',
          noteAlways: true, note: CURVE,
        },
      ],
    },
    {
      id: 'check',
      title: 'Step 4. Check: after correcting, is the bend gone? Yes at six sites, not fully at four, too far at two',
      text: '@check',
      figures: [
        {
          kind: 'intervals', id: 'hp-after', table: 'after', row: 'site', series: 'kind',
          title: 'The exponent before and after correcting: it should reach 1',
          subtitle: 'b in HIPS ∝ (FTIR EC)^b with weighed PM2.5 as the instrument, on reported and on linearised HIPS. Bars: 95 % bootstrap. Sites in order of optical depth, heaviest first.',
          panels: [{ value: 'value', label: 'Response exponent', reference: 1, lo: 'lo', hi: 'hi' }], labelWidth: 220,
          noteAlways: true, note: REFERENCE,
        },
        {
          kind: 'table', id: 'hp-checked', table: 'checked', title: 'The verdict at every site',
          columns: [{ key: 'site', label: 'Site' }, { key: 'tested_filters', label: 'Filters tested', format: 'int' }, { key: 'exponent_before', label: 'Before', format: 'f2' },
            { key: 'exponent_predicted', label: 'Curve alone', format: 'f2' }, { key: 'exponent_after', label: 'After', format: 'f2' }, { key: 'after_lo', label: 'low', format: 'f2' },
            { key: 'after_hi', label: 'high', format: 'f2' }, { key: 'verdict', label: 'Verdict' }],
          subtitle: 'Before ÷ curve alone = after. "Cannot tell": the interval before correcting is wider than 0.6. "Cannot be tested": fewer than 40 filters with FTIR EC and PM2.5.',
        },
      ],
    },
    {
      id: 'product',
      title: 'What comes out, and when to run it again',
      figures: [
        {
          kind: 'table', id: 'hp-outputs', table: 'outputs', title: 'Three tables from one command',
          columns: [{ key: 'file', label: 'File' }, { key: 'rows', label: 'Rows' }, { key: 'columns', label: 'Columns' }, { key: 'use', label: 'What it is for' }],
          subtitle: 'aeth hips screen --out FOLDER',
        },
        {
          kind: 'table', id: 'hp-revisit', table: 'revisit', title: 'When to run it again, and what would change the answer',
          columns: [{ key: 'when', label: 'When' }, { key: 'what to do', label: 'What to do' }],
          noteAlways: true, note: CURVE + ' ' + REFERENCE,
        },
      ],
    },
  ],
} satisfies MeetingSpec
