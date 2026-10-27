// The Research diagnostics tab's fits and screens (src/sections/research/common/stats.ts), against scipy / numpy values
import assert from 'node:assert/strict'
import { test } from 'node:test'
import { bestLowThreshold, correlationP, deming, influence, multiOls, normalCdf, ols, originFit, quadratic, quantile, slidingWindow, spearman, splitScan, splitSweep, steiger, theilSen } from '../sections/research/common/stats.ts'

const near = (a, b, tol = 1e-6) => assert.ok(Math.abs(a - b) <= tol, `${a} vs ${b}`)
const x = [1, 2, 3, 4, 5, 6, 7, 8.5, 9, 11]
const y = [2.1, 3.9, 6.2, 8.1, 9.8, 12.5, 13.9, 17.2, 18.1, 22.9]

test('OLS matches scipy.stats.linregress, standard errors included', () => {
  const f = ols(x, y)
  near(f.slope, 2.053617469087082); near(f.intercept, -0.13293870034201127); near(f.r, 0.9991678350429498)
  near(f.seSlope, 0.029639129972410326); near(f.seIntercept, 0.1907641271498007)
  assert.equal(ols([1, 1, 1], [1, 2, 3]), null)
  assert.equal(ols([1, 2, NaN, 4], [1, 2, 3, NaN]), null)   // two finite pairs only
})

test('Theil–Sen, quadratic and quantiles match scipy / numpy', () => {
  const t = theilSen(x, y)
  near(t.slope, 2.046153846153846); near(t.intercept, -0.10384615384615259)
  const q = quadratic(x, y)
  near(q.coef[2], 0.01551101, 1e-7); near(q.coef[1], 1.87118865, 1e-7); near(q.coef[0], 0.25524054, 1e-7)
  const v = [3, 1, 4, 1, 5, 9, 2, 6]
  near(quantile(v, 0.25), 1.75); near(quantile(v, 0.5), 3.5); near(quantile(v, 0.9), 6.9)
})

test('through-origin slope and uncentred R²', () => {
  const f = originFit([1, 2, 3], [2, 4, 6])
  near(f.slope, 2); near(f.r2, 1)
})

test('Deming spans OLS (λ → ∞) to inverse OLS (λ → 0)', () => {
  const f = ols(x, y)
  near(deming(x, y, 1e9).slope, f.slope, 1e-6)
  near(deming(x, y, 1e-9).slope, 1 / ols(y, x).slope, 1e-6)
  const d = deming(x, y, 1)
  near(d.intercept, y.reduce((a, b) => a + b) / 10 - d.slope * x.reduce((a, b) => a + b) / 10)
})

test('Spearman with ties, Pearson p and the normal CDF', () => {
  near(spearman([1, 2, 2, 3, 5], [2, 1, 4, 4, 6]), 0.7631578947368421)
  const yy = [2.1 + 0.5, 3.9 - 1, 6.2 + 2, 8.1 - 0.3, 9.8 + 1, 12.5 - 2]
  near(correlationP(0.9322050434932286, 6), 0.006738436089242768, 1e-6)
  near(normalCdf(1.96), 0.9750021048517795, 2e-7); near(normalCdf(-0.5), 0.3085375387259869, 2e-7)
  assert.ok(yy.length === 6)
})

test('Steiger: equal correlations give z = 0; a stronger one gives a positive z', () => {
  near(steiger(0.6, 0.6, 0.8, 100).z, 0)
  const s = steiger(0.66, 0.52, 0.89, 150)
  assert.ok(s.z > 0 && s.p < 0.05)
})

test('multiple OLS recovers known coefficients', () => {
  const a = [1, 2, 3, 4, 5, 6], b = [2, 1, 4, 3, 6, 5]
  const yy = a.map((v, i) => 1 + 2 * v - 0.5 * b[i])
  const f = multiOls([a, b], yy)
  near(f.coef[0], 1); near(f.coef[1], 2); near(f.coef[2], -0.5); near(f.r2, 1)
})

test('split sweep: low side is s ≤ t, both sides need minN, best threshold is the low slope nearest 1', () => {
  // slope 1 where s ≤ 50, slope 0.5 above
  const pts = Array.from({ length: 40 }, (_, i) => ({ x: i % 10, y: (i < 20 ? 1 : 0.5) * (i % 10), s: i < 20 ? 30 + i : 60 + i }))
  const sw = splitSweep(pts, [40, 49, 85], 5)
  assert.deepEqual(sw.map((s) => [s.nLow, s.nHigh]), [[11, 29], [20, 20], [26, 14]])
  near(sw[1].low.slope, 1); near(sw[1].high.slope, 0.5)
  const best = bestLowThreshold(sw, 5)
  near(best.low.slope, 1); assert.notEqual(best.threshold, 85)   // 40 and 49 tie at slope 1; the first wins
})

test('split scan scores divergence × coverage at quantiles 0.2 … 0.8', () => {
  const pts = Array.from({ length: 100 }, (_, i) => ({ x: i % 10, y: (i < 50 ? 1 : 2) * (i % 10), s: i }))
  const rows = splitScan('v', pts, 100, 10)
  assert.equal(rows.length, 7)
  const best = rows.slice().sort((a, b) => b.score - a.score)[0]
  near(best.quantile, 0.5); near(best.divergence, 1)
  assert.deepEqual(splitScan('v', pts.map((p) => ({ ...p, s: p.s % 3 })), 100, 10), [])
})

test('sliding window and influence', () => {
  const pts = x.map((v, i) => ({ x: v, y: y[i] }))
  const w = slidingWindow(pts, 5, 2)
  assert.equal(w.length, 3)
  near(w[0].center, 3)
  const inf = influence(pts)
  near(inf.reduce((a, r) => a + r.h, 0), 2)    // leverages sum to the number of parameters
})
