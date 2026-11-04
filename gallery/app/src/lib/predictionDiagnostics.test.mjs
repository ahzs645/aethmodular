import assert from 'node:assert/strict'
import { test } from 'node:test'
import { ecdf } from './stats.ts'
import { residualRows } from './predictionDiagnostics.ts'
import { timeTickFormat } from './timeTicks.ts'

test('ECDF ties form one jump, missing values are not zero observations', () => {
  assert.deepEqual(ecdf([null, NaN, Infinity, 3, 1, 1, 0]), [
    { x: 0, y: .25, n: 4 }, { x: 1, y: .75, n: 4 }, { x: 3, y: 1, n: 4 },
  ])
  assert.deepEqual(ecdf([]), [])
  assert.deepEqual(ecdf([0, 1, 3], [1, 2, 1]), ecdf([0, 1, 1, 3]))
  assert.throws(() => ecdf([0, 1], [1]))
})

test('a MAC change moves the optical reference, never the fitted prediction', () => {
  const source = [{ Fabs: 20, residual_ugm3: 1 }, { Fabs: null, residual_ugm3: 2 }]
  const at10 = residualRows(source, 10), at5 = residualRows(source, 5)
  assert.equal(at10[0].predicted, 3)
  assert.equal(at5[0].predicted, 3)
  assert.equal(at5[0].observed, 4)
  assert.equal(at5[0].residual, -1)
  assert.equal(at5[1].residual, null)
  assert.equal(source[0].residual_ugm3, 1)
  assert.throws(() => residualRows(source, 0))
  assert.throws(() => residualRows(source, 10, Infinity))
})

test('short time spans label clock time rather than repeating the month', () => {
  const start = new Date(2026, 3, 12, 9), end = new Date(2026, 3, 12, 18)
  assert.equal(timeTickFormat([+start, +end])(start), '09:00')
  assert.equal(timeTickFormat([+start, +end])(end), '18:00')
  assert.equal(timeTickFormat([+start, +start + 365 * 86400000])(start), 'Apr 2026')
})
