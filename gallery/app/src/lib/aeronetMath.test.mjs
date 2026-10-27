import assert from 'node:assert/strict'
import { test } from 'node:test'
import { cazorlaClass, exclusionCurve, fineSoil, meanSe, monthlyMeans, nonAbsorbingFraction, pearson, quantileSorted } from './aeronetMath.ts'

const near = (a, b, tol = 1e-9) => assert.ok(Math.abs(a - b) < tol, `${a} != ${b}`)

test('fine soil is Malm 1994 on the five elements, preferring elemental calcium', () => {
  const r = { Aluminum: 1, Silicon: 1, Calcium: 9, 'Calcium (element)': 1, Iron: 1, Titanium: 1 }
  near(fineSoil(r), 2.2 + 2.49 + 1.63 + 2.42 + 1.94)
  near(fineSoil({ ...r, 'Calcium (element)': null }), 2.2 + 2.49 + 1.63 * 9 + 2.42 + 1.94)
  assert.equal(fineSoil({ ...r, Titanium: null }), null)
})

test('w is ammonium sulfate and nitrate over themselves plus BC', () => {
  // AS = 1.375, AN = 1.29, BC = 2.665: w = 0.5
  near(nonAbsorbingFraction({ Sulfate: 1, Nitrate: 1, 'EC (FTIR)': 2.665 }, 'EC (FTIR)'), 0.5)
  assert.equal(nonAbsorbingFraction({ Sulfate: 1, Nitrate: null, 'EC (FTIR)': 1 }, 'EC (FTIR)'), null)
})

test('pearson and the numpy-style quantile', () => {
  near(pearson([1, 2, 3, 4], [2, 4, 6, 8]), 1)
  assert.equal(pearson([1, 1, 1], [1, 2, 3]), null)
  near(quantileSorted([1, 2, 3, 4, 5], 0.2), 1.8)
})

test('the exclusion curve drops the lowest BC/AOD and recomputes r', () => {
  // the two low-ratio filters break an otherwise perfect BC–AOD line
  const pts = [
    { ratio: 0.1, x: 1, ys: { aod: 9 } }, { ratio: 0.2, x: 2, ys: { aod: 8 } },
    ...[3, 4, 5, 6, 7, 8, 9, 10].map((x) => ({ ratio: 1 + x, x, ys: { aod: x / 10 } })),
  ]
  const steps = exclusionCurve(pts, ['aod'], [0, 20])
  assert.equal(steps[0].n, 10)
  assert.ok(steps[0].r.aod.r < 0.5)
  assert.equal(steps[1].n, 8)          // the 20th percentile sits between the 2nd and 3rd ratio
  near(steps[1].r.aod.r, 1)
})

test('Cazorla matrix classes', () => {
  assert.equal(cazorlaClass(0.5, 2), 'Dust dominated')
  assert.equal(cazorlaClass(1.2, 2), 'OC/Dust mix')
  assert.equal(cazorlaClass(2, 2), 'OC dominated')
  assert.equal(cazorlaClass(0.5, 1.2), 'Dust/EC mix')
  assert.equal(cazorlaClass(1.2, 1.2), 'Mix')
  assert.equal(cazorlaClass(2, 1.2), 'EC/OC mixture')
  assert.equal(cazorlaClass(1.2, 0.8), 'Coated large particles')
  assert.equal(cazorlaClass(2, 0.8), 'EC dominated')
  assert.equal(cazorlaClass(NaN, 1), null)
})

test('monthly means and mean ± SE', () => {
  const m = monthlyMeans(['2023-01-02', '2024-01-05', '2023-02-01'], [1, 3, null])
  assert.deepEqual(m[0], { mean: 2, n: 2 })
  assert.deepEqual(m[1], { mean: null, n: 0 })
  const s = meanSe([1, 2, 3, null])
  near(s.mean, 2); near(s.se, 1 / Math.sqrt(3)); assert.equal(s.n, 3)
})
