import assert from 'node:assert/strict'
import { test } from 'node:test'
import { regression } from './stats.ts'
import { confidence, lineInterval, monthBlocks } from './regressionCI.ts'
const x = Array.from({ length: 24 }, (_, i) => i / 3 + 1)
const y = x.map((v, i) => 1.3 * v - .4 + Math.sin(i) * .8)
const opts = { errorsInVariables: true, lambda: 2.96 }
test('bootstrap is deterministic, cached, finite, and leaves point fits unchanged', () => {
  const s = regression(x, y, opts), before = JSON.stringify(s)
  for (const method of ['ols', 'deming', 'york']) {
    const ci = confidence(s, method)
    assert.equal(ci.draws.length, 500)
    assert.equal(confidence(s, method), ci)
    assert.deepEqual(confidence(regression(x, y, opts), method), ci)
    assert.ok(ci.slope[0] < ci.slope[1])
    assert.ok(ci.intercept.every(Number.isFinite))
  }
  assert.equal(JSON.stringify(s), before)
})
test('line bounds retain slope/intercept covariance rather than combining marginal extremes', () => {
  const ci = { draws: [[1, 9], [2, 8], [3, 7]], slope: [1, 3], intercept: [7, 9] }
  assert.deepEqual(lineInterval(ci, 1), [10, 10])
  assert.deepEqual(lineInterval(ci, 0), [7.05, 8.95])
})
test('exact lines have zero-width intervals for every estimator', () => {
  const s = regression(x, x.map(v => 2 * v - 3), opts)
  for (const method of ['ols', 'deming', 'york']) {
    const ci = confidence(s, method), [lo, hi] = lineInterval(ci, 4)
    assert.ok(Math.abs(lo - 5) < 1e-9 && Math.abs(hi - 5) < 1e-9)
  }
})
test('month blocks are site-qualified and sparse site strata do not get fabricated intervals', () => {
  const rows = x.map((_, i) => ({ site: 'A', date: `2021-${String(Math.floor(i / 3) + 1).padStart(2, '0')}-01` }))
  const groups = monthBlocks(rows)
  const ci = confidence(regression(x, y, { ...opts, bootstrapGroups: groups }), 'deming')
  assert.match(ci.description, /month-block bootstrap within site/)
  assert.notDeepEqual(ci.slope, confidence(regression(x, y, opts), 'deming').slope)
  assert.equal(confidence(regression(x, y, { ...opts, bootstrapGroups: x.map(() => 'A|2021-01') }), 'deming'), null)
  assert.equal(monthBlocks([{ site: 'A', date: null }]), undefined)
  assert.deepEqual(monthBlocks([{ site: 'A', date: '2021-01-01' }, { site: 'B', date: '2021-01-01' }]), ['A|2021-01', 'B|2021-01'])
})
test('unavailable sources, few pairs, invalid York errors, and unstable resampling give no CI', () => {
  assert.equal(confidence(null, 'deming'), null)
  assert.equal(confidence({ n: 50 }, 'deming'), null)
  assert.equal(confidence(regression(x.slice(0, 7), y.slice(0, 7), opts), 'deming'), null)
  assert.equal(confidence(regression(x, y, { ...opts, sigmaX: x.map(() => 0) }), 'york'), null)
  const unstable = regression(Array(19).fill(1).concat(2), Array(19).fill(1).concat(3), opts)
  assert.equal(confidence(unstable, 'deming'), null)
})
test('missing x/y observations remove matching block and uncertainty entries together', () => {
  const groups = x.map((_, i) => `A|${Math.floor(i / 3)}`)
  const a = confidence(regression([...x, NaN], [...y, 1000], { ...opts, bootstrapGroups: [...groups, 'bad'], sigmaX: [...x.map(() => .2), 999] }), 'york')
  const b = confidence(regression(x, y, { ...opts, bootstrapGroups: groups, sigmaX: x.map(() => .2) }), 'york')
  assert.deepEqual(a, b)
})
