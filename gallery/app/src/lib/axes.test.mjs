import assert from 'node:assert/strict'
import { test } from 'node:test'
import { DEFAULT_AXES, pairDomain } from './axes.ts'

test('default origin axes hide negative points only from display and count them', () => {
  const d = pairDomain([1, 2, 3], [-4, 2, 5], DEFAULT_AXES)
  assert.equal(DEFAULT_AXES.mode, 'From 0,0')
  assert.equal(d.x[0], 0)
  assert.equal(d.y[0], 0)
  assert.equal(d.dropped, 1)
})

test('intercept view retains negative observations without borrowing their y extent for x', () => {
  const d = pairDomain([0.1, 2, 10], [-4, 2, 13], { ...DEFAULT_AXES, mode: 'Show intercept' }, [-0.581])
  assert.ok(d.x[0] > -0.5)
  assert.ok(d.y[0] < -4)
  assert.equal(d.dropped, 0)
  assert.ok(Math.abs((d.x[1] - d.x[0]) - (d.y[1] - d.y[0])) < 1e-10)
})

test('intercept view includes every selected intercept, including those outside the observed range', () => {
  for (const identity of [true, false]) {
    const d = pairDomain([1, 2, 3], [1, 2, 4], { ...DEFAULT_AXES, mode: 'Show intercept', identity }, [-6, 9])
    assert.ok(d.y[0] < -6)
    assert.ok(d.y[1] > 9)
    assert.equal(d.dropped, 0)
  }
})

test('robustDomain clips to quantiles, keeps required values and counts what it leaves out', async () => {
  const { robustDomain } = await import('./axes.ts')
  const v = Array.from({ length: 101 }, (_, i) => i)          // 0..100
  assert.deepEqual(robustDomain(v, 0.02, 0.98).domain, [2, 98])
  assert.deepEqual(robustDomain(v, 0.02, 0.98), { domain: [2, 98], below: 2, above: 2 })
  assert.deepEqual(robustDomain(v, 0, 1).domain, [0, 100])
  assert.deepEqual(robustDomain(v, 0.5, 0.5, { include: [0] }).domain, [0, 50])
  assert.deepEqual(robustDomain(v, 0, 1, { limit: [10, 40] }), { domain: [10, 40], below: 10, above: 60 })
  assert.deepEqual(robustDomain([null, NaN, undefined], 0.02, 0.98, { fallback: [-12, 3] }).domain, [-12, 3])
})
