import assert from 'node:assert/strict'
import { test } from 'node:test'
import { chooseMLFold, mlConfusionMatrix, mlRegionSet, mlScoreValue, selectMLImportance, selectMLProfiles, selectMLRows, validateSeasonalMLData } from './seasonalML.ts'

const key = { task: 'season', calendar: 'belg_feb', cohort: 'direct232', model_id: 'rf_raw_full', fold_id: 'forward2024' }
const score = { ...key, n_train: 91, n_test: 97, balanced_accuracy: .7, accuracy: .75, macro_f1: .69 }
const complete = { id: 'forward2024', label: 'Cycle ending 2024', n_train: 91, n_test: 97, complete_test: true, missing_test_months: [] }
const sparse = { id: 'forward2025', label: 'Cycle ending 2025', n_train: 188, n_test: 30, complete_test: false, missing_test_months: [9] }
const importance = { ...key, kind: 'permutation', scheme: 'across_filters', region_id: 'ch', region_label: '2800–3000 cm⁻¹',
  low: 2800, high: 3000, n_channels: 26, delta_mean: -.03, delta_sd: .02, q05: -.06, q95: .01, repeats: 20, baseline_loss: .3, metric: 'balanced_error' }
const fixture = () => ({ schema_version: 1, generated_at: '2026-11-03', counts: { direct232: 232 },
  models: [{ id: 'rf_raw_full', label: 'Random forest · raw full', family: 'rf', representation: 'raw_full', covariates: [] }],
  representations: [{ id: 'raw_full', label: 'Raw full spectrum', n_channels: 2722 }], folds: [complete, sparse], scores: [score],
  confusion: [{ ...key, actual: 'Bega', predicted: 'Bega', n: 30 }, { ...key, actual: 'Bega', predicted: 'Belg', n: 7 }],
  importance: [importance], predictions: [{ ...key, id: 'ETAD-0139', date: '2023-11-01', season: 'Bega', actual: 'Bega', predicted: 'Belg' }],
  limitations: [], provenance: { inputs: {}, settings: {} } })

test('ML selectors never pool calendar, task, cohort or future-cycle scores', () => {
  const rows = [score, { ...score, calendar: 'dry_feb', balanced_accuracy: .9 },
    { ...score, cohort: 'weather188' }, { ...score, fold_id: 'forward2025', n_test: 30 },
    { ...score, task: 'disagreement', rmse: .4 }]
  assert.deepEqual(selectMLRows(rows, key), [score])
  assert.equal(selectMLRows(rows, { task: 'season', calendar: 'belg_feb', cohort: 'direct232' }).length, 2)
  assert.equal(rows.length, 5, 'selecting a graph never mutates the saved evaluation')
})

test('default ML fold prefers full test-month coverage, while a saved sparse-cycle selection remains explicit', () => {
  const folds = [sparse, complete]
  assert.equal(chooseMLFold(folds, folds.map(f => f.id), '').id, complete.id)
  assert.equal(chooseMLFold(folds, folds.map(f => f.id), sparse.id).id, sparse.id)
  assert.equal(chooseMLFold(folds, [sparse.id], complete.id).id, sparse.id)
  assert.equal(chooseMLFold(folds, [], ''), null)
})

test('task-specific metrics preserve a true zero and negative predictive R² without inventing unavailable scores', () => {
  assert.equal(mlScoreValue({ ...score, balanced_accuracy: 0 }, 'balanced_accuracy'), 0)
  assert.equal(mlScoreValue({ ...score, rmse: 0 }, 'rmse'), null, 'season classification cannot display concentration error')
  assert.equal(mlScoreValue({ ...score, task: 'disagreement', r2: -.6 }, 'r2'), -.6)
  assert.equal(mlScoreValue({ ...score, task: 'disagreement', rmse: null }, 'rmse'), null)
})

test('confusion fractions use actual-class denominators and do not turn an absent class into zero recall', () => {
  const rows = [{ ...key, actual: 'Belg', predicted: 'Bega', n: 1 }, { ...key, actual: 'Bega', predicted: 'Bega', n: 9 },
    { ...key, actual: 'Bega', predicted: 'Belg', n: 1 }, { ...key, actual: 'Belg', predicted: 'Belg', n: 1 },
    { ...key, actual: 'Kiremt', predicted: 'Kiremt', n: 0 }]
  const matrix = mlConfusionMatrix(rows)
  assert.deepEqual(matrix.classes, ['Bega', 'Belg', 'Kiremt'])
  assert.deepEqual(matrix.cells[0][0], { n: 9, total: 10, fraction: .9 })
  assert.deepEqual(matrix.cells[1][1], { n: 1, total: 2, fraction: .5 })
  assert.deepEqual(matrix.cells[2][2], { n: 0, total: 0, fraction: null })
})

test('region views retain negative importance and exact repeat spread, separating conditional permutations and refitted ablations', () => {
  const rows = [importance, { ...importance, scheme: 'within_season', delta_mean: .01 },
    { ...importance, kind: 'ablation', delta_mean: .04, repeats: 1 },
    { ...importance, fold_id: sparse.id, delta_mean: .5 }, { ...importance, delta_mean: null }]
  const before = structuredClone(rows)
  const selected = selectMLImportance(rows, key, 'permutation', 'across_filters')
  assert.equal(selected.length, 1)
  assert.equal(selected[0].delta_mean, -.03)
  assert.deepEqual([selected[0].q05, selected[0].q95], [-.06, .01])
  assert.equal(selectMLImportance(rows, key, 'ablation', 'across_filters')[0].delta_mean, .04)
  const support = selectMLImportance([{ ...importance, region_id: 'observed-zero', delta_mean: 0 },
    { ...importance, region_id: 'unsupported', delta_mean: null }], key, 'permutation', 'across_filters')
  assert.deepEqual(support.map(row => [row.region_id, row.delta_mean]), [['observed-zero', 0]], 'unsupported regions never acquire zero importance')
  assert.deepEqual(rows, before)
})

test('validation rejects repeated physical-filter predictions within an evaluation but permits the same filter in a separate cycle', () => {
  const data = fixture()
  assert.equal(validateSeasonalMLData(data), null)
  const repeated = { ...data, predictions: [...data.predictions, { ...data.predictions[0] }] }
  assert.match(validateSeasonalMLData(repeated), /duplicate.*physical-filter/)
  const secondCycle = { ...data, predictions: [...data.predictions, { ...data.predictions[0], fold_id: sparse.id }] }
  assert.equal(validateSeasonalMLData(secondCycle), null)
  assert.match(validateSeasonalMLData({ ...data, scores: [score, { ...score }] }), /duplicate model score/)
})

test('whole-spectrum importance stays an overall control instead of appearing as a localized middle-wavenumber effect', () => {
  const rows = [{ ...importance, region_id: 'whole_spectrum', low: 500, high: 4000 },
    { ...importance, region_id: 'bin_2800_2900' }, { ...importance, region_id: '2800–3000 · C–H stretching region' }]
  assert.deepEqual(rows.map(mlRegionSet), ['whole', 'bins', 'named'])
  assert.equal(rows.filter(row => mlRegionSet(row) === 'bins').length, 1)
  assert.equal(rows.filter(row => mlRegionSet(row) === 'named').length, 1)
})

test('per-season importance never pools with all-season rows or changes the declared held-out season count', () => {
  const base = { ...importance, task: 'disagreement', metric: 'rmse', scheme: 'within_season' }
  const rows = [base, { ...base, season: 'Bega (Oct-Jan, dry)', n_test: 37, delta_mean: .04 },
    { ...base, season: 'Belg (Feb-May, short rains)', n_test: 34, delta_mean: -.02 },
    { ...base, season: 'Kiremt (Jun-Sep, long rains)', n_test: 26, delta_mean: .01 }]
  const selection = { ...key, task: 'disagreement' }
  assert.deepEqual(selectMLImportance(rows, selection, 'permutation', 'within_season'), [base])
  const bega = selectMLImportance(rows, selection, 'permutation', 'within_season', 'Bega (Oct-Jan, dry)')
  assert.equal(bega.length, 1)
  assert.equal(bega[0].n_test, 37)
  assert.equal(bega[0].delta_mean, .04)
  assert.equal(selectMLImportance(rows, selection, 'ablation', 'within_season', 'Bega (Oct-Jan, dry)').length, 0)
})

test('validation rejects nonfinite results and concentration/classification unit mix-ups while retaining signed effects', () => {
  const data = fixture()
  assert.equal(validateSeasonalMLData(data), null)
  assert.match(validateSeasonalMLData({ ...data, scores: [{ ...score, balanced_accuracy: Infinity }] }), /nonfinite/)
  assert.match(validateSeasonalMLData({ ...data, scores: [{ ...score, balanced_accuracy: 1.1 }] }), /range/)
  assert.match(validateSeasonalMLData({ ...data, importance: [{ ...importance, metric: 'rmse' }] }), /metric units/)
  assert.match(validateSeasonalMLData({ ...data, importance: [{ ...importance, q05: Infinity }] }), /invalid spectral/)
  assert.match(validateSeasonalMLData({ ...data, predictions: [{ ...data.predictions[0], task: 'disagreement', actual: 'Bega' }] }), /values/)
})

test('descriptive seasonal spectra retain their full-cohort count, signed medians and gap positions separately from ML folds', () => {
  const profile = { representation: 'spline_common', season: 'Bega (Oct-Jan, dry)', n: 81,
    wavenumbers: [3200, 2800, 1800, 1600], median: [.01, -.02, null, .005], q25: [.005, -.03, null, 0], q75: [.02, -.01, null, .01] }
  const rows = [profile, { ...profile, representation: 'raw_full' }]
  const selected = selectMLProfiles(rows, 'spline_common')
  assert.deepEqual(selected, [profile])
  assert.equal(selected[0].n, 81, 'descriptive Bega count is not replaced by the37 held-out Bega filters')
  assert.deepEqual(selected[0].median, [.01, -.02, null, .005])
  const data = { ...fixture(), seasonal_spectra: rows }
  assert.equal(validateSeasonalMLData(data), null)
  assert.match(validateSeasonalMLData({ ...data, seasonal_spectra: [profile, { ...profile }] }), /duplicate seasonal spectrum/)
  assert.match(validateSeasonalMLData({ ...data, seasonal_spectra: [{ ...profile, median: [.01, Infinity, null, .005] }] }), /seasonal spectrum/)
  assert.match(validateSeasonalMLData({ ...data, seasonal_spectra: [{ ...profile, wavenumbers: [3200, 2800, 2800, 1600] }] }), /ordered/)
})
