import assert from 'node:assert/strict'
import { test } from 'node:test'
import { checkRegionSet, ecLineFor, pairResidualScores, selectResidualImportance, validateSeasonalChecks } from './seasonalChecks.ts'

const line = { task: 'disagreement', calendar: 'belg_feb', cohort: 'direct232', fold_id: 'forward_2024', model_id: 'ec_line',
  n_train: 91, n_test: 97, rmse: .488, mae: .38, r2: .87, bias: -.1, slope: .9, intercept: -1.2 }
const score = (model_id, target, rmse, representation = 'spline_common') => ({ fold_id: 'forward_2024', model_id, family: model_id.split('_')[0],
  representation, target, n_train: 91, n_test: 97, rmse, mae: rmse, r2: .8, bias: 0 })
const imp = (region_id, target, extra = {}) => ({ fold_id: 'forward_2024', model_id: 'rf_spline_common', family: 'rf', representation: 'spline_common',
  target, season: null, region_id, region_label: region_id, low: 1800, high: 2250, n_channels: 200, n_test: 97, delta_mean: .1,
  q05: .05, q95: .15, repeats: 10, baseline_loss: .43, metric: 'rmse', ...extra })
const fixture = () => ({ schema_version: 1, generated_at: '2026-11-03', lots: { counts: [{ cycle_end: 2023, season: 'Bega', lot_group: '248', n: 18 }], sources: {}, note: '' },
  ec_line: [line], disagreement_ec_correlation: { 2024: .94 }, disagreement_by_season: [],
  residual_scores: [score('rf_spline_common', 'disagreement', .636), score('rf_spline_common', 'residual', .427)],
  residual_importance: [imp('1800–2250 · middle continuum', 'disagreement'), imp('1800–2250 · middle continuum', 'residual')],
  lot_control: [{ test_id: 'rev', test_label: 'x', family: 'rf', representation: 'raw_common', model_id: 'rf_raw_common', n_train: 97, n_test: 34,
    lots_train: '251', lots_test: '248', accuracy: .6, balanced_accuracy: .6 }],
  lot_separability: [], saved_forward2024_recall: [], labels: { family: {}, representation: {}, named_regions: [] },
  reproduction: { compared_models: 1, max_abs_rmse_difference: 0, common_channels: 1897 }, limitations: [], provenance: {} })

test('the EC line is found for exactly one saved evaluation and never borrowed from another cohort or cycle', () => {
  assert.equal(ecLineFor([line], { calendar: 'belg_feb', cohort: 'direct232', fold_id: 'forward_2024' }), line)
  assert.equal(ecLineFor([line], { calendar: 'belg_feb', cohort: 'weather188', fold_id: 'forward_2024' }), null)
  assert.equal(ecLineFor([line], { calendar: 'belg_feb', cohort: 'direct232', fold_id: 'forward_2025' }), null)
  assert.equal(ecLineFor(undefined, { calendar: 'belg_feb', cohort: 'direct232', fold_id: 'forward_2024' }), null)
})

test('direct and residual scores pair by model within one fold', () => {
  const rows = [...fixture().residual_scores, { ...score('rf_spline_common', 'residual', .5), fold_id: 'forward_2025' }, score('rf_raw_full', 'disagreement', .578, 'raw_full')]
  const pairs = pairResidualScores(rows, 'forward_2024', 'all')
  assert.equal(pairs.length, 2)
  assert.equal(pairs[0].direct.rmse, .636)
  assert.equal(pairs[0].residual.rmse, .427)
  assert.equal(pairs[1].residual, null, 'a missing residual stays missing')
  assert.equal(pairResidualScores(rows, 'forward_2024', 'raw_full').length, 1)
})

test('region selection keeps target, season and region set apart', () => {
  const rows = [imp('bin_1800_1900', 'residual'), imp('3500–4000', 'residual', { low: 3500, high: 4000 }), imp('whole_spectrum', 'residual', { low: null, high: null }),
    imp('3500–4000', 'disagreement'), imp('3500–4000', 'residual', { season: 'Kiremt (Jun-Sep, long rains)' }), imp('1560–1700', 'residual', { delta_mean: null })]
  assert.deepEqual(rows.map(checkRegionSet), ['bins', 'named', 'whole', 'named', 'named', 'named'])
  const named = selectResidualImportance(rows, { fold_id: 'forward_2024', target: 'residual', representation: 'spline_common', season: null, set: 'named' })
  assert.deepEqual(named.map(r => r.region_id), ['3500–4000'], 'non-finite importance and other seasons/targets are excluded')
  assert.equal(selectResidualImportance(rows, { fold_id: 'forward_2024', target: 'residual', representation: 'spline_common', season: 'Kiremt (Jun-Sep, long rains)', set: 'named' }).length, 1)
})

test('validation refuses duplicates, impossible scores and wrong units', () => {
  assert.equal(validateSeasonalChecks(fixture()), null)
  assert.match(validateSeasonalChecks({ ...fixture(), ec_line: [line, line] }), /EC-line/)
  assert.match(validateSeasonalChecks({ ...fixture(), residual_scores: [score('rf_x', 'residual', -1)] }), /residual score/)
  assert.match(validateSeasonalChecks({ ...fixture(), residual_importance: [imp('a', 'residual', { metric: 'balanced_error' })] }), /region test/)
  assert.match(validateSeasonalChecks({ ...fixture(), lot_control: [{ ...fixture().lot_control[0], accuracy: 1.2 }] }), /lot-controlled/)
  assert.match(validateSeasonalChecks({ schema_version: 2 }), /unsupported/)
})
