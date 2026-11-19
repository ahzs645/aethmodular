// A saved calibration run under the linearised HIPS view (lib/hipsRun): made-up filters whose Fabs follows the
// saturating IMPROVE curve, so the answer is known. No data files.
import assert from 'node:assert/strict'
import { test } from 'node:test'
import { CENTRAL_CEILING, DEPOSIT_AREA_CM2, saturate } from './hipsLinearise.ts'
import { lineariseRun, runHipsMark, siteInScope } from './hipsRun.ts'

const near = (a, b, tol, what) => assert.ok(Math.abs(a - b) <= tol, `${what}: ${a} vs ${b}`)
const VOLUME = 7.2, MAC_TRUE = 10, LAMBDA_10 = 2.96
// true EC 1 to 12 µg/m³; the prediction is 0.6 of it (a pure level error); Fabs is what a saturating HIPS reports
const ec = Array.from({ length: 60 }, (_, i) => 1 + (11 * i) / 59)
const tauOf = (c) => saturate((c * MAC_TRUE * VOLUME) / (100 * DEPOSIT_AREA_CM2), CENTRAL_CEILING)
const fabs = ec.map((c) => (tauOf(c) * 100 * DEPOSIT_AREA_CM2) / VOLUME)
const row = (set, MAC) => ({ evaluation_set: set, MAC, n: 0, ols_slope: null, ols_intercept: null, deming_slope: null, deming_intercept: null })
const run = (over = {}) => ({
  preset: 'p', target: 'addis', k: 4, auto_k: 4, n_cohort: 1, n_train: 1, n_train_sites: 1, rmsecv_floor: 1, pct_rmsecv_floor: 1, curve: [], heldout: null,
  metrics: [row('all', 10), row('fixed', 10), row('all', 17)], split_check: [], plausibility: null, extrap_pct: null, q_residual_pct: null, ref_kind: 'fabs',
  eval: { id: ec.map((_, i) => `ETAD-${String(i).padStart(4, '0')}`), ref: fabs, pred: ec.map((c) => 0.6 * c), group: ec.map(() => 'Bega'), date: ec.map(() => '2024-01-01'),
    deployed: null, fixed: ec.map((_, i) => i % 2 === 0) },
  ...over,
})
const volumes = (missing = new Set()) => (id) => (id && !missing.has(id) ? VOLUME : null)

test('linearising takes a compressed reference back to a proportional one', () => {
  const lin = lineariseRun(run(), volumes(), CENTRAL_CEILING, LAMBDA_10)
  // each filter's linearised Fabs ÷ MAC is its true EC
  lin.eval.ref.forEach((f, i) => near(f / MAC_TRUE, ec[i], 1e-9, `filter ${i}`))
  const all = lin.metrics.find((m) => m.evaluation_set === 'all' && m.MAC === 10)
  near(all.deming_slope, 0.6, 1e-9, 'slope at MAC 10'); near(all.deming_intercept, 0, 1e-9, 'intercept'); assert.equal(all.n, 60)
  // the slope follows the MAC of the row, the intercept does not
  const at17 = lin.metrics.find((m) => m.MAC === 17)
  near(at17.deming_slope, 0.6 * 1.7, 1e-9, 'slope at MAC 17'); near(at17.deming_intercept, 0, 1e-9, 'intercept at MAC 17')
  assert.equal(lin.metrics.find((m) => m.evaluation_set === 'fixed').n, 30)
  assert.deepEqual(lin.hips_linearised, { ceiling: CENTRAL_CEILING, dropped: 0 })
  assert.equal(runHipsMark(lin), ', linearised'); assert.equal(runHipsMark(run()), '')
})

test('as measured, the same run reads steep with a negative intercept: the sign the protocol names', () => {
  const xs = fabs.map((f) => f / 10), ys = ec.map((c) => 0.6 * c)
  const mx = xs.reduce((a, b) => a + b) / xs.length, my = ys.reduce((a, b) => a + b) / ys.length
  const slope = xs.reduce((a, x, i) => a + (x - mx) * (ys[i] - my), 0) / xs.reduce((a, x) => a + (x - mx) ** 2, 0)
  assert.ok(slope > 0.9 && my - slope * mx < -0.5, `${slope}, ${my - slope * mx}`)
})

test('a filter without a sampled volume, or past the ceiling, drops out with its prediction', () => {
  const missing = new Set(['ETAD-0003', 'ETAD-0010'])
  const lin = lineariseRun(run(), volumes(missing), CENTRAL_CEILING, LAMBDA_10)
  assert.equal(lin.eval.id.length, 58); assert.equal(lin.eval.pred.length, 58); assert.equal(lin.hips_linearised.dropped, 2)
  assert.ok(!lin.eval.id.includes('ETAD-0003'))
  // a ceiling below the darkest filters leaves only the lighter ones
  const low = lineariseRun(run(), volumes(), 1.0, LAMBDA_10)
  const kept = fabs.filter((f) => (f * VOLUME) / (100 * DEPOSIT_AREA_CM2) < 1.0).length
  assert.ok(kept < 60); assert.equal(low.eval.ref.length, kept); assert.ok(low.eval.ref.every(Number.isFinite))
})

test('a run scored against a reference EC is left alone, and the input run is not changed', () => {
  const ecRun = run({ ref_kind: 'ec' })
  assert.equal(lineariseRun(ecRun, volumes(), CENTRAL_CEILING, LAMBDA_10), ecRun)
  const saved = run(), before = [...saved.eval.ref]
  lineariseRun(saved, volumes(), CENTRAL_CEILING, LAMBDA_10)
  assert.deepEqual(saved.eval.ref, before); assert.equal(saved.hips_linearised, undefined)
})

test('the scope reads the 11 Nov screen: Addis and Delhi are seen, Bishoftu expected only, Pasadena neither', () => {
  for (const code of ['ETAD', 'INDH', 'ETBI', 'USPA', 'CHTS', 'XXXX', undefined]) assert.equal(siteInScope(code, 'all'), true)
  assert.deepEqual(['ETAD', 'INDH', 'CHTS', 'ETBI', 'USPA'].map((c) => siteInScope(c, 'seen')), [true, true, true, false, false])
  assert.deepEqual(['ETAD', 'INDH', 'CHTS', 'ETBI', 'USPA'].map((c) => siteInScope(c, 'expected')), [true, true, true, true, false])
  assert.equal(siteInScope('XXXX', 'expected'), false); assert.equal(siteInScope(undefined, 'seen'), false)
})
