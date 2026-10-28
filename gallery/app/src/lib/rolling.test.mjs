import test from 'node:test'
import assert from 'node:assert/strict'
import { ROLL_DEFAULT, clampDays, rollLabel, rollingByTime } from './rolling.ts'

const day = (n) => new Date(Date.UTC(2024, 0, 1) + n * 864e5)
const pts = (vals, every = 1) => vals.map((v, i) => ({ d: day(i * every), v }))
const run = (p, spec) => rollingByTime(p, (x) => x.d, (x) => x.v, { ...ROLL_DEFAULT, ...spec }).map((r) => r.v)

test('trailing is pandas rolling("ND"): the window (t − N days, t]', () => {
  // daily points: a 3-day trailing window holds today and the two days before, never the day 3 back
  assert.deepEqual(run(pts([1, 2, 3, 4, 5]), { days: 3, minN: 1, stat: 'mean' }), [1, 1.5, 2, 3, 4])
})

test('min_periods: fewer finite values than minN give null', () => {
  assert.deepEqual(run(pts([1, 2, 3, 4, 5, 6]), { days: 45, minN: 5 }), [null, null, null, null, 3, 3.5])
  assert.deepEqual(run(pts([1, NaN, 3]), { days: 45, minN: 2, stat: 'mean' }), [null, null, 2])
})

test('median averages the two middle values of an even window, as pandas does', () => {
  assert.deepEqual(run(pts([4, 1, 3, 2]), { days: 10, minN: 1 }), [4, 2.5, 3, 2.5])
})

test('centred puts the same width around each point', () => {
  // 4-day centred window on daily points: two days either side
  assert.deepEqual(run(pts([1, 2, 3, 4, 5]), { days: 4, minN: 1, stat: 'mean', align: 'centred' }), [2, 2.5, 3, 3.5, 4])
})

test('the default reproduces ftir_29: a trailing 45-day median from the 5th filter on', () => {
  // filters every 9 days: the 45-day window (t − 45, t] holds 5 filters once there are 5
  const v = run(pts([10, 20, 30, 40, 50, 60, 70], 9), {})
  assert.deepEqual(v, [null, null, null, null, 30, 40, 50])
})

test('labels and typed day counts', () => {
  assert.equal(rollLabel(ROLL_DEFAULT), '45-day rolling median')
  assert.equal(rollLabel({ ...ROLL_DEFAULT, days: 30, stat: 'mean', align: 'centred' }), '30-day centred rolling mean')
  assert.equal(clampDays('60.4', 45), 60)
  assert.equal(clampDays('0', 45), 45)
  assert.equal(clampDays('x', 45), 45)
  assert.equal(clampDays('99999', 45), 3650)
})
