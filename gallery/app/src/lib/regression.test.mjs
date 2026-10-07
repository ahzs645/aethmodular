import assert from 'node:assert/strict'
import { test } from 'node:test'
import { regression, york } from './stats.ts'
import { parseEstimators, estimatorParam } from './regressionMethods.ts'
import { fitUncertainty } from './fitUncertainty.ts'

const near = (actual, expected, tol = 1e-9) => assert.ok(Math.abs(actual - expected) <= tol, `${actual} != ${expected}`)
const x = [1, 2, 3, 4, 6], y = [2.1, 3.9, 6.4, 7.8, 12.6]

test('Deming is the default; every subset including no fits survives a copied URL', () => {
  assert.deepEqual(parseEstimators(null), ['deming'])
  assert.deepEqual(parseEstimators('invalid'), ['deming'])
  for (let mask = 0; mask < 8; mask++) {
    const methods = ['ols', 'deming', 'york'].filter((_, i) => mask & (1 << i))
    assert.deepEqual(parseEstimators(estimatorParam(methods)), methods)
  }
})

test('OLS, Deming and York recover an exact line', () => {
  const s = regression(x, x.map(v => 2 * v - 3), { errorsInVariables: true, lambda: 2.96 })
  near(s.slope, 2); near(s.intercept, -3); near(s.r2, 1)
  near(s.demingSlope, 2); near(s.demingIntercept, -3)
  near(s.yorkSlope, 2); near(s.yorkIntercept, -3)
})

test('constant-error York equals Deming for positive and negative associations and unequal error ratios', () => {
  for (const sign of [-1, 1]) for (const lambda of [0.25, 1, 2.96, 50]) {
    const s = regression(x, y.map(v => sign * v), { errorsInVariables: true, lambda })
    near(s.yorkSlope, s.demingSlope); near(s.yorkIntercept, s.demingIntercept)
    assert.match(s.yorkAssumption, /constant errors/)
  }
})

test('heteroscedastic York matches the existing Python York implementation', () => {
  // calibration_explorer/hips_lab.py:york on this exact fixture.
  const sx = [.1, .4, .15, .6, .2], sy = [.2, .25, .5, .3, .6]
  const fit = york(x, y, sx, sy)
  near(fit.slope, 2.09352139474352)
  near(fit.intercept, -0.012250266704079138)
  const s = regression(x, y, { errorsInVariables: true, sigmaX: sx, sigmaY: sy })
  near(s.yorkSlope, fit.slope)
  assert.ok(Math.abs(s.yorkSlope - s.demingSlope) > 1e-3)
  assert.match(s.yorkAssumption, /supplied point/)
})

test('missing paired values are removed together, including uncertainty entries', () => {
  const a = regression([...x, NaN], [...y, 100], { errorsInVariables: true, sigmaX: [.1, .4, .15, .6, .2, 999], sigmaY: [.2, .25, .5, .3, .6, 999] })
  assert.equal(a.n, 5)
  near(a.yorkSlope, 2.09352139474352)
  assert.equal(regression(x, [1]), null)
})

test('York refuses invalid uncertainties and degenerate inputs', () => {
  assert.equal(york(x, y, [0, 1, 1, 1, 1], [1, 1, 1, 1, 1]), null)
  assert.equal(york(x, y, [1], [1]), null)
  assert.equal(york([1, 1, 1], [1, 2, 3], [1, 1, 1], [1, 1, 1]), null)
  assert.equal(regression([1, 1, 1], [1, 2, 3]), null)
})

test('HIPS York weights use paired per-filter uncertainty and the displayed MAC', () => {
  const rows = [1, 2, 3].map(v => ({ 'HIPS Fabs uncertainty': v }))
  const opts = fitUncertainty(rows, 'HIPS BC', 'EC (FTIR)', { mac_value: 10 }, 2.96)
  opts.sigmaX.forEach((v, i) => near(v, [.5, 1, 1.5][i]))
  assert.equal(opts.lambda, 2.96)
  const incomplete = fitUncertainty([...rows, {}], 'HIPS BC', 'EC (FTIR)', { mac_value: 10 })
  assert.equal(incomplete.sigmaX, undefined)
})
