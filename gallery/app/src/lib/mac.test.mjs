import test from 'node:test'
import assert from 'node:assert/strict'
import { demingLambda, slopeAt, bcAt, macView } from './macMath.ts'
import { regression } from './stats.ts'

test('λ scales with MAC², 1 off an Fabs axis', () => {
  assert.equal(demingLambda(10, 2.96), 2.96)
  assert.ok(Math.abs(demingLambda(6, 2.96) - 2.96 * 0.36) < 1e-12)
  assert.equal(demingLambda(null, 2.96), 1)
})

test('a fit against Fabs/MAC: slope scales by MAC/10, intercept stays (OLS and Deming with λ ∝ MAC²)', () => {
  let seed = 3
  const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647)
  const fabs = Array.from({ length: 60 }, () => 20 + 60 * rnd())
  const y = fabs.map((f) => 0.18 * f - 2 + rnd())
  const at = (mac) => regression(fabs.map((f) => f / mac), y, { errorsInVariables: true, lambda: demingLambda(mac, 2.96) })
  const s10 = at(10), s6 = at(6)
  assert.ok(Math.abs(slopeAt(s10.slope, 6) - s6.slope) < 1e-9)
  assert.ok(Math.abs(slopeAt(s10.demingSlope, 6) - s6.demingSlope) < 1e-9)
  assert.ok(Math.abs(s10.demingIntercept - s6.demingIntercept) < 1e-9)
  assert.ok(Math.abs(s10.intercept - s6.intercept) < 1e-9)
})

test('HIPS BC and the view object', () => {
  assert.equal(bcAt(5, 5), 10)
  assert.equal(bcAt(null, 5), null)
  const v = macView(12.5, 10, 2.96)
  assert.equal(v.changed, true)
  assert.equal(v.fromFabs(50), 4)
  assert.equal(v.slope(1), 1.25)
  assert.equal(v.label, 'MAC 12.5')
})
