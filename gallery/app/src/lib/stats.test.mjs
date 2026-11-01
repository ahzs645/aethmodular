// lib/stats as the gallery's one statistics library: the merged copies agree with each other and with d3
import assert from 'node:assert/strict'
import { test } from 'node:test'
import * as d3 from 'd3'
import {
  boxStats, deming, describe, goldenJitter, IQR_MIN_N, jitter, mean, median, meanSe, monthlyMeans, monthlyStats, orNull,
  pearson, quantile, quantileSorted, regression, sd, spearman,
} from './stats.ts'

const near = (a, b, tol = 1e-12) => assert.ok(Math.abs(a - b) <= tol * Math.max(1, Math.abs(b)), `${a} vs ${b}`)
let seed = 7
const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647

/** paired data sets: a clean line, a noisy calibration-like cloud, a negative slope, ties, and gaps */
const sets = [
  { x: [1, 2, 3, 4, 5, 6, 7, 8.5, 9, 11], y: [2.1, 3.9, 6.2, 8.1, 9.8, 12.5, 13.9, 17.2, 18.1, 22.9] },
  (() => { const x = Array.from({ length: 200 }, () => rnd() * 8); return { x, y: x.map((v) => 0.9 * v + 1.3 + (rnd() - 0.5) * 3) } })(),
  { x: [0.5, 1.5, 2, 2, 3, 4.5, 6], y: [9, 7.5, 7, 6.8, 5, 2.5, 1] },
  { x: [1, 2, null, 4, 5, NaN, 7], y: [1.2, 1.9, 3, undefined, 5.3, 6, 6.8] },
]

test('regression()’s Deming is deming(): the same slope and intercept, λ = 1 and λ ≠ 1', () => {
  for (const { x, y } of sets) {
    const xs = x.map((v) => v ?? NaN), ys = y.map((v) => v ?? NaN)
    for (const lambda of [1, 0.25, 4, 37, 1e-3]) {
      const r = regression(xs, ys, { errorsInVariables: true, lambda }), d = deming(x, y, lambda)
      assert.equal(r.demingSlope, d.slope)
      assert.equal(r.demingIntercept, d.intercept)
    }
  }
})

test('pearson and spearman: null-tolerant, a minN floor, spearman is pearson of the ranks', () => {
  const { x, y } = sets[3]
  near(pearson(x, y), pearson([1, 2, 5, 7], [1.2, 1.9, 5.3, 6.8]))
  assert.equal(pearson([1, 2, 3, 4], [2, 4, 6, 8], { minN: 5 }), null)
  near(pearson([1, 2, 3, 4, 5], [2, 4, 6, 8, 11], { minN: 5 }), 22 / Math.sqrt(488))
  assert.equal(pearson([1, 1, 1], [1, 2, 3]), null)
  // the weather tab's floor (lib/weather WEATHER_MIN_PAIRS) and its tie-averaged value
  near(spearman([1, 2, 2, 3, 4, NaN], [1, 2, 3, 4, 5, 9], { minN: 5 }), 0.9746794344808963)
  assert.equal(spearman([1, 2, 3, 4], [1, 2, 3, 4], { minN: 5 }), null)
  near(spearman([1, 2, 3, 4, 5, 6], [1, 8, 27, 64, 125, 216]), 1)
})

test('quantile, median, mean and sd agree with d3 (numpy linear quantiles, sample SD)', () => {
  for (let t = 0; t < 200; t++) {
    const n = 1 + Math.floor(rnd() * 40)
    const v = Array.from({ length: n }, () => (rnd() < 0.1 ? null : Math.round(rnd() * 1e4) / 100))
    const f = v.filter((x) => x != null)
    if (!f.length) continue
    for (const p of [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]) {
      near(quantile(v, p), d3.quantile(f, p))
      near(quantileSorted(f.slice().sort((a, b) => a - b), p), d3.quantileSorted(f.slice().sort((a, b) => a - b), p))
    }
    near(median(v), d3.median(f))
    near(mean(v), d3.mean(f))
    if (f.length > 1) near(sd(v), d3.deviation(f), 1e-10)
  }
  assert.ok(Number.isNaN(quantile([], 0.5)) && Number.isNaN(median([null, NaN])) && Number.isNaN(mean([])))
  assert.equal(orNull(median([])), null)
  near(median([1, 2, 3, 4]), 2.5)      // numpy.median: the mean of the middle two
  near(sd([2, 4, 4, 4, 5, 5, 7, 9], 0), 2)   // population SD (np.std)
})

test('meanSe and describe', () => {
  const m = meanSe([2, 4, null, 6])
  near(m.mean, 4); near(m.se, 2 / Math.sqrt(3)); assert.equal(m.n, 3)
  assert.deepEqual(meanSe([]), { mean: null, se: null, n: 0 })
  const s = describe([5, 1, null, 3, 2, 4])
  assert.equal(s.n, 5); near(s.median, 3); near(s.q25, 2); near(s.q90, 4.6); near(s.sd, Math.sqrt(2.5))
  assert.equal(describe([null]), null)
})

test('boxStats: quartiles, 1.5·IQR whiskers and fences, a minN', () => {
  const b = boxStats([1, 2, 3, 4, 5, 6, 7, 100])
  near(b.q1, 2.75); near(b.median, 4.5); near(b.q3, 6.25)
  assert.equal(b.max, 7); assert.deepEqual(b.outliers, [100])
  near(b.loFence, 2.75 - 1.5 * 3.5); near(b.hiFence, 6.25 + 1.5 * 3.5)
  assert.equal(boxStats([1, 2, 3]), null)
  near(boxStats([1, 2, 3], { minN: 1 }).median, 2)
})

test('monthlyStats: median and mean per month, quartiles from IQR_MIN_N values', () => {
  const rows = [[1, 3], [1, 1], [1, 2], [2, 5], [2, 1], [2, 2], [2, 4], [3, null], [13, 9]].map(([month, v]) => ({ month, v }))
  const m = monthlyStats(rows, (r) => r.month, (r) => r.v)
  assert.equal(m.length, 12)
  assert.deepEqual(m[0], { month: 1, n: 3, median: 2, mean: 2, q1: null, q3: null })
  assert.equal(IQR_MIN_N, 4)
  near(m[1].q1, d3.quantile([1, 2, 4, 5], 0.25)); near(m[1].q3, d3.quantile([1, 2, 4, 5], 0.75))
  assert.deepEqual(m[2], { month: 3, n: 0, median: null, mean: null, q1: null, q3: null })
  assert.equal(monthlyStats(rows, (r) => r.month, (r) => r.v, { iqrMinN: 3 })[0].q1, 1.5)
  const mm = monthlyMeans(['2023-01-02', '2024-01-05', '2023-02-01'], [1, 3, null])
  assert.deepEqual(mm[0], { mean: 2, n: 2 }); assert.deepEqual(mm[1], { mean: null, n: 0 })
})

test('jitter is deterministic and inside its width', () => {
  for (let i = 0; i < 500; i++) {
    const a = jitter(i, 10), g = goldenJitter(i, 10, i / 7)
    assert.equal(a, jitter(i, 10))
    assert.ok(a >= -5 && a < 5 && g >= -5 && g < 5)
  }
  assert.equal(goldenJitter(3, 8), ((3 * 0.6180339) % 1 - 0.5) * 8)
})
