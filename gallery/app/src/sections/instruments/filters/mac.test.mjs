// node --test src/sections/instruments/filters/*.test.mjs   (from gallery/app)
import assert from 'node:assert/strict'
import { test } from 'node:test'
import { closestTo, iqrFences, ironSweep, linspace, macMethods, ols2, seasonRuns } from './mac.ts'

const near = (a, b, tol = 1e-9) => assert.ok(Math.abs(a - b) < tol, `${a} != ${b}`)

test('the four MAC methods agree on a line through the origin', () => {
  const ec = [1, 2, 3, 4, 5]
  const m = macMethods(ec, ec.map((v) => 10 * v))
  near(m.meanOfRatios, 10); near(m.ratioOfMeans, 10); near(m.origin, 10)
  near(m.ols.slope, 10); near(m.ols.intercept, 0); near(m.ols.r2, 1); near(m.xAtZero, 0)
})

test('an offset separates the methods as AAARpos describes', () => {
  // y = 4x + 28: OLS recovers it, and its line reaches zero absorption at EC = -7
  const ec = [1, 2, 3, 4]
  const y = ec.map((v) => 4 * v + 28)
  const m = macMethods(ec, y)
  near(m.ols.slope, 4); near(m.ols.intercept, 28); near(m.xAtZero, -7)
  near(m.ratioOfMeans, 38 / 2.5)
  near(m.meanOfRatios, (32 + 18 + 40 / 3 + 11) / 4)
  near(m.origin, (32 + 72 + 120 + 176) / 30)
})

test('two-predictor least squares recovers exact coefficients', () => {
  const ec = [1, 2, 3, 4, 5, 6, 7], fe = [0.3, 0.1, 0.5, 0.2, 0.9, 0.4, 0.6]
  const y = ec.map((v, i) => 26 + 3.5 * v + 14 * fe[i])
  const f = ols2(y, ec, fe)
  near(f.c, 26, 1e-9); near(f.a, 3.5, 1e-9); near(f.b, 14, 1e-9); near(f.r2, 1, 1e-12)
  f.residuals.forEach((e) => near(e, 0, 1e-9))
  assert.equal(ols2([1, 2, 3, 4], [1, 2, 3, 4], [2, 4, 6, 8]), null) // collinear
})

test('subtracting the true iron term restores the EC slope; closestTo picks it', () => {
  const ec = [1, 2, 3, 4, 5, 6], fe = [0.5, 0.1, 0.8, 0.3, 0.7, 0.2]
  const y = ec.map((v, i) => 10 * v + 20 * fe[i])
  const steps = ironSweep(ec, y, fe, linspace(0, 40, 41))
  const best = closestTo(steps, 10)
  near(best.k, 20); near(best.slope, 10); near(best.intercept, 0, 1e-9)
})

test('IQR fences use linear quantiles', () => {
  // pandas: Q1 = 2, Q3 = 4 for [1..5] -> fences -1 and 7
  const [lo, hi] = iqrFences([5, 1, 3, 2, 4])
  near(lo, -1); near(hi, 7)
})

test('season runs split at long gaps and drop short runs', () => {
  const pts = [
    ['2023-01-01', 'Dry', 10], ['2023-01-10', 'Dry', 12], ['2023-01-19', 'Dry', 14],
    ['2023-11-01', 'Dry', 20], ['2023-11-10', 'Dry', 22],
    ['2023-04-01', 'Belg', 8],
  ].map(([d, s, v]) => ({ d, s, v }))
  const runs = seasonRuns(pts, (p) => p.s, (p) => p.d, (p) => p.v, { gapDays: 60, minN: 2 })
  assert.deepEqual(runs.map((r) => [r.season, r.date, r.n, r.mean]), [['Dry', '2023-01-10', 3, 12], ['Dry', '2023-11-10', 2, 21]])
})
