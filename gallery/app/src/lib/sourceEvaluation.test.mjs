import test from 'node:test'
import assert from 'node:assert/strict'
import { epoSourceVerified, medianAbsoluteError, sourceScoreValue, sweepMedianAbsoluteError } from './sourceEvaluation.ts'

test('loading a historical snapshot with newer UI cannot validate legacy EPO scores', () => {
  const row = { model: 'epo', improve_error: 1.24, improve_heavy_ratio: 0.91, improve_mae: 1.24 }
  for (const receipt of [undefined, {}, { status: 'legacy_unverified' }, { status: 'corrected' }]) {
    assert.equal(epoSourceVerified(receipt), false)
    assert.equal(medianAbsoluteError(row, receipt), null)
    assert.equal(sourceScoreValue(row, 'improve_heavy_ratio', receipt), null)
    assert.equal(sweepMedianAbsoluteError(row, receipt), null)
  }
  assert.equal(medianAbsoluteError({ ...row, model: 'vibes' }), 1.24)
})

test('a corrected source receipt exposes distinct median and mean errors', () => {
  const receipt = { status: 'corrected', protocol: 'fixed_k_common_physical_filter_metrics_v1' }
  const row = { model: 'epo', evaluation_status: 'corrected', improve_median_absolute_error: 1.1, improve_mae: 2.9, improve_error: 1.1 }
  assert.equal(medianAbsoluteError(row, receipt), 1.1)
  assert.equal(medianAbsoluteError({ ...row, evaluation_status: 'legacy_withheld' }, receipt), null)
  assert.equal(medianAbsoluteError({ ...row, evaluation_status: undefined }, receipt), null)
  assert.equal(sweepMedianAbsoluteError(row, receipt), 1.1)
  assert.equal(sourceScoreValue(row, 'improve_mae', receipt), 2.9)
  assert.equal(sourceScoreValue({ ...row, improve_rmse: null }, 'improve_rmse', receipt), null)
})
