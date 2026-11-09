import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { checkAddisStudy, checkSourceReview, eventHours, localAxisDate, localClockLabel, localRangeLabel } from './model.ts'

// Deliberately tiny contract fixture: one real zero-valued hour, with all other
// hours absent. Zero, no saved observation, and incomplete support must differ.
const empty = {
  expected_hours: 24, snapshot_hours: 0, valid_hours: 0, coverage_percent: 0, snapshot_coverage_percent: null,
  mean_bc_ug_m3: null, median_bc_ug_m3: null, sd_bc_ug_m3: null, max_bc_ug_m3: null, q25_bc_ug_m3: null, q75_bc_ug_m3: null,
}
const one = { ...empty, snapshot_hours: 1, valid_hours: 1, coverage_percent: 100 / 24, snapshot_coverage_percent: 100,
  mean_bc_ug_m3: 0, median_bc_ug_m3: 0, max_bc_ug_m3: 0, q25_bc_ug_m3: 0, q75_bc_ug_m3: 0 }
const start = '2024-02-01T00:00:00+03:00', end = '2024-02-02T00:00:00+03:00'
const day = { ...one, start_local: start, end_local_exclusive: end }
const calendar = (id) => ({ id, label: id, festival_exclusions_applied: false,
  seasons: [
    { ...one, id: 'bega', label: 'Bega', color: 'orange', months: id === 'belg_feb' ? [10, 11, 12, 1] : [10, 11, 12, 1, 2] },
    { ...empty, id: 'belg', label: 'Belg', color: 'green', months: id === 'belg_feb' ? [2, 3, 4, 5] : [3, 4, 5] },
    { ...empty, id: 'kiremt', label: 'Kiremt', color: 'blue', months: [6, 7, 8, 9] },
  ], diurnal: [{ ...one, season_id: 'bega', hour: 0 }],
})
function fixture() {
  const reported = { label: 'Reported site', bc_mean_ug_m3: 1, bc_sd_ug_m3: 1, bc_max_ug_m3: 2, completeness_percent: 90 }
  return {
    schema_version: 1, paper_reproduction_complete: false, independent_validation: false,
    deployment: { id: 'jacros', label: 'Jacros', site_code: 'ETAD', serial: 'fixture', instrument_model: 'MA350', timezone: 'Africa/Addis_Ababa', channel: 'IR', unit: 'ug/m3' },
    period: { start_local: start, end_local_exclusive: end, timezone: 'Africa/Addis_Ababa' },
    paper: { id: 'shlipak_2026_addis', start_local: start, end_local_exclusive: end, citation: { title: 'Test study', url: 'https://doi.org/10.1021/acsestair.6c00140' },
      source_model: { alpha_fossil: 1, alpha_biomass: 2 }, reported: { central: reported, jacros: reported, biomass_bc_percent_discussion: 5.8 }, unresolved_for_exact_reproduction: ['QC remains unresolved'] },
    source: { first_local: start, last_local: start, end_local_exclusive: '2024-02-01T01:00:00+03:00', output_unit: 'ug/m3', min_valid_ir_minutes: 30,
      input_file: 'fixture', input_sha256: 'fixture-hash', finite_handling: 'Finite signed values retained', timestamp_basis: 'Local standard hour start' },
    coverage: { ...one, outside_snapshot_hours: 23, missing_within_snapshot_hours: 0 },
    summary: day, daily: [day], monthly: [day],
    calendars: { belg_feb: calendar('belg_feb'), dry_feb: calendar('dry_feb') },
    events: [{ ...day, id: 'example-2024', label: 'Example', date_local: '2024-02-01', window_rule: 'Exploratory full day',
      available_hours: 1, incomplete: true, unavailable: false, trace: [{ elapsed_hour: 0, local: start, bc_ug_m3: 0, source_present: true }] }],
    capabilities: { central: { available: false, reason: 'Missing deployment' }, source_fractions: { available: false, reason: 'Missing optical pair' } },
    limitations: ['Descriptive context only'],
  }
}

test('Addis wall-clock axes keep the declared date and hour, including exclusive leap-month boundaries', () => {
  assert.equal(localAxisDate('2024-02-29T23:00:00+03:00').toISOString(), '2024-02-29T23:00:00.000Z')
  assert.equal(localClockLabel('2024-02-29T23:00:00+03:00'), '29 Feb 2024 23:00')
  assert.equal(localRangeLabel('2024-02-01T00:00:00+03:00', '2024-03-01T00:00:00+03:00'), '1 Feb 2024–29 Feb 2024')
  assert.equal(localRangeLabel(start, end), '1 Feb 2024')
})

test('event plotting keeps zero and signed values and inserts gaps for absent hours without mutating the input', () => {
  const event = fixture().events[0]
  event.trace = [
    { elapsed_hour: 3, local: start, bc_ug_m3: -0.5, source_present: true },
    { elapsed_hour: 0, local: start, bc_ug_m3: 0, source_present: true },
    { elapsed_hour: 2, local: start, bc_ug_m3: null, source_present: true },
  ]
  const before = JSON.stringify(event)
  const plotted = eventHours(event)
  assert.equal(plotted.length, 24)
  assert.equal(plotted[0].observation.bc_ug_m3, 0)
  assert.equal(plotted[1].observation, null)
  assert.equal(plotted[2].observation.bc_ug_m3, null)
  assert.equal(plotted[3].observation.bc_ug_m3, -0.5)
  assert.equal(plotted[23].observation, null)
  assert.equal(JSON.stringify(event), before)
})

test('the chart contract accepts empty bins and an incomplete event with a genuine zero measurement', () => {
  const data = fixture()
  data.daily.push({ ...empty, start_local: end, end_local_exclusive: '2024-02-03T00:00:00+03:00' })
  assert.equal(checkAddisStudy(data), null)
  data.daily[1].mean_bc_ug_m3 = 0
  assert.match(checkAddisStudy(data), /support|timeline/)
})

test('unsupported evidence, concentration units, timezones and contradictory coverage fail closed', () => {
  for (const mutate of [
    (x) => { x.schema_version = 2 },
    (x) => { x.independent_validation = true },
    (x) => { x.deployment.unit = 'ng/m3' },
    (x) => { x.period.start_local = '2024-02-01T00:00:00' },
    (x) => { x.period.timezone = 'UTC' },
    (x) => { x.coverage.valid_hours = 2 },
    (x) => { x.coverage.snapshot_hours = 25 },
    (x) => { x.events[0].incomplete = false },
    (x) => { x.events[0].available_hours = 24 },
    (x) => { x.events[0].trace.push({ ...x.events[0].trace[0] }) },
    (x) => { x.source = null },
  ]) {
    const x = fixture(); mutate(x)
    assert.notEqual(checkAddisStudy(x), null)
  }
  for (const value of [null, [], {}, { schema_version: 1 }]) assert.equal(typeof checkAddisStudy(value), 'string')
})

test('calendar month assignments and local-hour identities cannot silently overlap', () => {
  const overlap = fixture()
  overlap.calendars.belg_feb.seasons[0].months.push(2)
  assert.match(checkAddisStudy(overlap), /each month once/)
  const duplicated = fixture()
  duplicated.calendars.belg_feb.diurnal.push({ ...duplicated.calendars.belg_feb.diurnal[0] })
  assert.match(checkAddisStudy(duplicated), /local-hour/)
  const unavailable = fixture()
  unavailable.events[0] = { ...unavailable.events[0], ...empty, available_hours: 0, unavailable: true, trace: [] }
  assert.equal(checkAddisStudy(unavailable), null)
  assert.ok(eventHours(unavailable.events[0]).every((h) => h.observation === null))
})

function sourceReview() {
  const profile = JSON.parse(readFileSync(new URL('../../../../../../research/ftir_hips_chem/workflows/profiles/ec_bc_jpl_review.json', import.meta.url), 'utf8'))
  return { ...profile, applied_to_observations: false,
    loading_sensitivity: { basis: 'analytical_constant_k', equation: 'BCc/BC1 = 1/(1-k*ATN1)', atn_min: 0, atn_max: 60,
      scope: 'Analytical illustration', series: profile.loading_candidates.map((c) => ({ candidate_id: c.id, singular_atn1: 1 / c.k,
        points: [{ atn1: 0, multiplier: 1, denominator_valid: true }] })) } }
}

test('source materials keep historical summaries separate from observed concentrations', () => {
  const review = sourceReview()
  const before = JSON.stringify(review)
  assert.equal(checkSourceReview(review), null)
  const study = fixture()
  study.source_review = review
  assert.equal(checkAddisStudy(study), null)
  assert.equal(JSON.stringify(review), before)
  assert.equal(study.summary.mean_bc_ug_m3, 0)
})

test('source-review contract rejects ambiguous settings and fabricated evidence states', () => {
  for (const mutate of [
    (x) => { x.applied_to_observations = true },
    (x) => { x.sources[0].sha256 = 'unverified' },
    (x) => { x.loading_candidates[0].k = 0 },
    (x) => { x.loading_sensitivity.series[0].candidate_id = 'unknown' },
    (x) => { x.loading_sensitivity.series[0].points[0].denominator_valid = false },
    (x) => { x.optical_sensitivity.rows[0].biomass_percent = null },
    (x) => { x.optical_sensitivity.rows.push({ ...x.optical_sensitivity.rows[0] }) },
  ]) {
    const review = sourceReview(); mutate(review)
    assert.notEqual(checkSourceReview(review), null)
  }
})
