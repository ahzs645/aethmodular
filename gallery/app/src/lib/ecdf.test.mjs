import assert from 'node:assert/strict'
import { test } from 'node:test'
import { ecdf, ecdfAt, ecdfSteps } from './ecdf.ts'

test("the ECDF is the notebooks' arange(1, n + 1) / n over the sorted values; ties share their last step", () => {
  assert.deepEqual(ecdf([3, 1, NaN, 2, 4]), { x: [1, 2, 3, 4], p: [0.25, 0.5, 0.75, 1] })
  assert.deepEqual(ecdf([1, 2, 2, 3]).p, [0.25, 0.75, 0.75, 1])
  assert.deepEqual(ecdf([]), { x: [], p: [] })
})

test('F(v) counts the values at or below v', () => {
  const s = [1, 2, 2, 3]
  assert.deepEqual([0, 1, 1.5, 2, 2.5, 3, 9].map((v) => ecdfAt(s, v)), [0, 0.25, 0.25, 0.75, 0.75, 1, 1])
  assert.ok(Number.isNaN(ecdfAt([], 1)))
})

test('the step path rises at each value inside the range and keeps its level at the clipped ends', () => {
  assert.deepEqual(ecdfSteps([1, 2, 2, 3], 0, 4), [[0, 0], [1, 0], [1, 0.25], [2, 0.25], [2, 0.75], [3, 0.75], [3, 1], [4, 1]])
  // clipped at 1.5 … 2.5: starts at F(1.5) and ends at F(2.5)
  assert.deepEqual(ecdfSteps([1, 2, 2, 3], 1.5, 2.5), [[1.5, 0.25], [2, 0.25], [2, 0.75], [2.5, 0.75]])
})
