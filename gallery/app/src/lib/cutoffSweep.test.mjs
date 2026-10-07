import assert from 'node:assert/strict'
import { test } from 'node:test'
import { calibrationReadout, cutoffPosition, fittingCounts, cutoffConfig } from './cutoffSweep.ts'

const row = { co: 'ocec', cut: 800, sel: 'raw', sp: 'airspec', ho: 0.8,
  rmse: 9.5, cv_rmse: 0.4, rmsecv: 0.5, n: 183, n_cohort: 798, n_fit: 621, n_heldout: 177 }

test('IMPROVE RMSE is distinct from target RMSE and internal CV RMSE', () => {
  assert.equal(calibrationReadout(row, 'cv_rmse'), 0.4)
  assert.equal(calibrationReadout(row, 'rmsecv'), 0.5)
  assert.equal(calibrationReadout(row, 'ho'), 0.8)
  assert.equal(calibrationReadout({ rmse: 9.5 }, 'cv_rmse'), null)
  assert.equal(calibrationReadout({ cv_rmse: NaN }, 'cv_rmse'), null)
})

test('fitted-count axis preserves cutoff selection and uses explicit fit counts only', () => {
  assert.equal(cutoffPosition(row, 'fitted'), 621)
  assert.equal(cutoffPosition(row, 'cutoff'), 800)
  assert.deepEqual(cutoffConfig(row), { co: 'ocec', cut: 800, sel: 'raw', sp: 'airspec' })
  assert.equal(cutoffPosition({ cut: 800, n: 183 }, 'fitted'), null)
  assert.equal(cutoffPosition({ ...row, n_fit: Infinity }, 'fitted'), null)
})

test('cohort and holdout counts cannot be inferred from cutoff or test pairs', () => {
  assert.deepEqual(fittingCounts(row), { cohort: 798, fitted: 621, heldout: 177 })
  assert.deepEqual(fittingCounts({ cut: 800, n: 183 }), { cohort: null, fitted: null, heldout: null })
  assert.deepEqual(fittingCounts({ n_fit: 798, n_heldout: 0 }), { cohort: null, fitted: 798, heldout: 0 })
  assert.deepEqual(fittingCounts({ n_cohort: -1, n_fit: 1.5, n_heldout: NaN }), { cohort: null, fitted: null, heldout: null })
})
