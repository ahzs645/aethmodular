// node --test src/sections/instruments/sources/aae.test.mjs
import assert from 'node:assert/strict'
import { test } from 'node:test'
import { MA350_NM, aae, aaeWithRatio, bccOffset, classShares, classify, crosstab, cell, dominantOf, ma350Ratio, rollingMean, thresholdSweep } from './aae.ts'

const close = (a, b, tol = 1e-3) => assert.ok(Math.abs(a - b) < tol, `${a} is not ${b}`)

test('classify follows optics.classify_aae: both cut points are inclusive at their own end', () => {
  assert.deepEqual([0.5, 0.9, 0.91, 1.49, 1.5, 2.2].map((v) => classify(v)), ['Fossil fuel', 'Fossil fuel', 'Mixed', 'Mixed', 'Biomass', 'Biomass'])
  assert.equal(classify(null), null)
  assert.equal(classify(NaN), null)
  assert.equal(classify(1.2, { fossil_max: 1.0, biomass_min: 1.2 }), 'Biomass')
})

test('classShares are fractions of the finite values, with the median', () => {
  const s = classShares([0.5, 1.0, 1.2, 1.6, 2.0, null, NaN])
  assert.equal(s.n, 5)
  close(s.shares['Fossil fuel'], 0.2); close(s.shares.Mixed, 0.4); close(s.shares.Biomass, 0.4)
  assert.equal(s.median, 1.2)
  assert.equal(classShares([]).n, 0)
})

test('aae is positive when the short wavelength absorbs more, and matches optics.aae', () => {
  // UV/IR absorption ratio 5.5 is an AAE of about 2 (the optics.py example)
  close(aae(5.5, 1, 375, 880), 2.0, 0.01)
  close(aae(1, 1, 375, 880), 0)
  assert.equal(aae(0, 1, 375, 880), null)
  assert.equal(aae(-1, 1, 375, 880), null)
})

test('the BCc offset is the optics.py identity: -1.016 for UV/IR, -0.967 for Red/IR', () => {
  close(bccOffset(ma350Ratio('UV', 'IR'), MA350_NM.UV, MA350_NM.IR), -1.016)
  close(bccOffset(ma350Ratio('Red', 'IR'), MA350_NM.Red, MA350_NM.IR), -0.967)
  // an AAE from BCc, turned back into absorption with the MA350 ratio, is the absorption AAE
  const uvAbs = 3 * 24.069, irAbs = 1 * 10.120
  const fromBcc = aae(3, 1, 375, 880), fromAbs = aae(uvAbs, irAbs, 375, 880)
  close(aaeWithRatio(fromBcc, ma350Ratio('UV', 'IR'), 375, 880), fromAbs, 1e-9)
  // ratio 1 is BCc itself
  close(aaeWithRatio(fromBcc, 1, 375, 880), fromBcc, 1e-12)
})

test('rollingMean is pandas rolling(7, min_periods=3).mean() over entries', () => {
  const out = rollingMean([1, 2, null, 4, 5, 6, 7, 8, 9], 7, 3)
  assert.deepEqual(out.slice(0, 3), [null, null, null])
  close(out[3], (1 + 2 + 4) / 3)
  close(out[8], (4 + 5 + 6 + 7 + 8 + 9) / 6)   // window 2..8 holds one missing value
})

test('dominantOf picks the largest value and its share of the total', () => {
  const d = dominantOf({ a: 1, b: 3, c: null, d: 0 }, ['a', 'b', 'c', 'd'])
  assert.equal(d.source, 'b'); close(d.share, 0.75)
  assert.equal(dominantOf({ a: null }, ['a']), null)
})

test('crosstab counts pairs and skips missing ones', () => {
  const t = crosstab([{ a: 'x', b: 'p' }, { a: 'x', b: 'p' }, { a: 'y', b: 'q' }, { a: null, b: 'q' }])
  assert.equal(cell(t, 'x', 'p'), 2); assert.equal(cell(t, 'y', 'q'), 1); assert.equal(cell(t, 'y', 'p'), 0)
})

test('thresholdSweep splits at or below the threshold vs above and fits each side', () => {
  const pairs = Array.from({ length: 10 }, (_, i) => ({ x: i, y: i < 5 ? 2 * i : 3 * i, t: i / 10 }))
  const [s] = thresholdSweep(pairs, [0.4])
  assert.equal(s.nLow, 5); assert.equal(s.nHigh, 5)
  close(s.low.slope, 2); close(s.high.slope, 3)
  const [edge] = thresholdSweep(pairs, [0.85])
  assert.equal(edge.nHigh, 1); assert.equal(edge.high, null)   // fewer than three pairs: no fit
})
