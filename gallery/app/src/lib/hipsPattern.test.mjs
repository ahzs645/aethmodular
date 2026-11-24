// The patterned-deposit model (lib/hipsPattern.ts) against the exporter and the notebook: the saturating curve is exactly
// a clumped deposit, the models invert, and the Addis fits reproduce the Python ones.
import assert from 'node:assert/strict'
import fs from 'node:fs'
import { test } from 'node:test'
import * as P from './hipsPattern.ts'

const read = (p) => JSON.parse(fs.readFileSync(new URL(`../../public/data/hips_linearise/${p}`, import.meta.url), 'utf8'))
const lin = read('linearise.json'), w = read('warren.json')
const near = (a, b, d, m) => assert.ok(Math.abs(a - b) < d, `${m ?? ''}: ${a} vs ${b}`)

test('the saturating curve is the Laplace transform of a Poisson number of exponential clumps', () => {
  const c = 2.2596
  const z = P.clumpedDeposit(c, 200000)
  near(z.reduce((a, b) => a + b, 0) / z.length, 1, 1e-9)
  for (const x of [0.3, 1, 2, 4]) near(-Math.log(z.reduce((a, b) => a + Math.exp(-x * b), 0) / z.length), x / (1 + x / c), 0.01, `x ${x}`)
  near(z.reduce((a, b) => a + (b - 1) ** 2, 0) / z.length, P.curveVariance(c), 0.02, 'variance')
  near(z.filter((v) => v === 0).length / z.length, P.curveClearShare(c), 0.004, 'clear share')
})

test('limits: no pattern is Beer–Lambert, full pixelation stops at the screen, the early bend is the thickness variance', () => {
  near(P.twoZone(1.3, 1), 1.3, 1e-12)
  near(P.twoZone(50, P.FULL_ALPHA), P.pixelationLimit(), 1e-6)
  near(P.pixelationLimit(), w.limit_tau, 1e-4)
  near(P.patternVariance(P.FULL_ALPHA), (1 - 0.638) / 0.638, 1e-6)
  for (const m of [P.screenModel('a', '', '', 10), P.curveModel('c', '', '', 2.2596), P.compositeModel('k', '', '', 3, 2.9)]) {
    const x = 1e-3
    near((2 * (x - m.response(x))) / (x * x), m.variance, 0.01, m.key)
    for (const v of [0.05, 0.3, 0.97, 1.4]) { const t = m.response(v); if (Number.isFinite(m.invert(t))) near(m.invert(t), v, 2e-3, `${m.key} inverts ${v}`) }
  }
  assert.ok(Number.isNaN(P.screenModel('f', '', '', P.FULL_ALPHA).invert(1.2)))
})

test('the Addis fits are those of the notebook (Python)', () => {
  const fit = (m) => P.fitModel(lin.filters, lin.lambda, m, lin.mac)
  const raw = fit(P.EVEN), ten = fit(P.screenModel('a', '', '', 10)), three = fit(P.screenModel('b', '', '', 3)), curve = fit(P.curveModel('c', '', '', 2.2596))
  near(raw.slope, 2.3139, 1e-3); near(raw.intercept, -6.1693, 5e-3)
  near(ten.slope, 0.7465, 3e-3); near(ten.intercept, -0.467, 0.02); near(ten.mac, 13.396, 0.06)
  near(three.slope, 1.7432, 3e-3); near(three.intercept, -4.4507, 0.02)
  near(curve.slope, 0.5458, 3e-3); near(curve.intercept, 0.0089, 0.02)
  assert.equal(fit(P.screenModel('f', '', '', P.FULL_ALPHA)).past, 76)
})

test('every split of the IMPROVE bend between a screen and the rest fits the medians as well', () => {
  const best = Math.min(...w.splits.map((s) => s.aic))
  for (const s of w.splits) assert.ok(s.aic - best < 2, `α ${s.alpha}`)
  assert.ok(w.fits.find((f) => f.key === 'alpha10').aic - w.fits.find((f) => f.key === 'curve').aic > 10)
})
