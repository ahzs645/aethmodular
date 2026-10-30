// node --test src/sections/instruments/diurnal/*.test.mjs   (from gallery/app)
import assert from 'node:assert/strict'
import { test } from 'node:test'
import { bccAae, blockSummary, dailyCv, deltaC, derive, hourMedians, median, normalisedProfile, overIr, pearson, quantileSorted, representativeDays, wavelengthCv } from './spectral.ts'

const near = (a, b, tol = 1e-9) => assert.ok(Math.abs(a - b) < tol, `${a} != ${b}`)

test('Delta-C, ratios and BCc AAE follow the notebooks, floors included', () => {
  near(deltaC(3, 2), 1)
  assert.equal(deltaC(null, 2), null)
  near(overIr(3, 2), 1.5)
  assert.equal(overIr(3, 0.05), null)
  // a BCc ratio of (375/880)^-1 is an AAE of 1
  near(bccAae(880 / 375, 1, 375, 880), 1)
  assert.equal(bccAae(0.05, 1, 375, 880), null)
})

test('the wavelength CV is SD (n − 1) over mean of the channels present, above the floor', () => {
  // mean 3, sample SD √2.5
  near(wavelengthCv([1, 2, 3, 4, 5]), Math.sqrt(2.5) / 3)
  near(wavelengthCv([1, null, 3, null, 5]), 2 / 3)
  assert.equal(wavelengthCv([0.01, 0.02, 0.03, 0.04, 0.05]), null)
  assert.equal(wavelengthCv([1, null, null, null, null]), null)
  assert.equal(wavelengthCv([2, 2, 2, 2, 2]), 0)
})

test('quantiles are linear between ranks, as pandas', () => {
  near(quantileSorted([1, 2, 3, 4], 0.25), 1.75)
  near(median([4, null, 1, 3, 2]), 2.5)
  assert.equal(median([]), null)
})

test('derive builds every column per day-hour', () => {
  const d = derive({ UV: [2, null], Blue: [1.5, 1], Green: [1.2, 1], Red: [1.1, 1], IR: [1, 1] })
  near(d.deltaC[0], 1); assert.equal(d.deltaC[1], null)
  near(d.uvIr[0], 2); near(d.blueIr[0], 1.5); near(d.redIr[1], 1)
  near(d.aaeUvIr[0], -Math.log(2) / Math.log(375 / 880))
  near(d.cv[1], 0)
})

test('the normalised profile divides each hour by the mean of the hourly medians', () => {
  const hours = [0, 0, 0, 1, 1, 2]
  const v = [1, 2, 3, 4, 4, 7]
  const { median: m, n } = hourMedians(hours, v)
  assert.deepEqual(m.slice(0, 3), [2, 4, 7]); assert.deepEqual(n.slice(0, 3), [3, 2, 1])
  assert.equal(m[3], null)
  const { ratio, dailyMean } = normalisedProfile(m)
  near(dailyMean, 13 / 3); near(ratio[0], 6 / 13); near(ratio[2], 21 / 13); assert.equal(ratio[5], null)
  // a minimum count drops thin hours before the mean
  assert.equal(hourMedians(hours, v, () => true, 2).median[2], null)
})

test('daily CV keeps days with enough IR minutes; representative days skip the five extremes', () => {
  const days = [], cv = [], ir = []
  for (let k = 0; k < 30; k++) for (let h = 0; h < 24; h++) {
    days.push(`2024-01-${String(k + 1).padStart(2, '0')}`); cv.push(k / 100 + (h % 2) * 0.001); ir.push(h < 20 || k === 0 ? 1 : null)
  }
  const daily = dailyCv(days, cv, ir, () => 60)
  assert.equal(daily.length, 30)
  assert.equal(dailyCv(days, cv, ir, () => 59).length, 1)
  const rep = representativeDays(daily)
  assert.deepEqual(rep.map((r) => r.day.day), ['2024-01-25', '2024-01-16', '2024-01-06'])
  assert.equal(representativeDays(daily.slice(0, 9)), null)
})

test('the block summary takes medians per time block over the kept day-hours', () => {
  const hours = [0, 1, 6, 7, 12, 22]
  const ch = { UV: [2, 4, 3, 3, 1, 5], Blue: [1, 1, 1, 1, 1, 1], Green: [1, 1, 1, 1, 1, 1], Red: [1, 1, 1, 1, 1, 1], IR: [1, 2, 1, null, 1, 1] }
  const rows = blockSummary(hours, ch.IR, derive(ch), (i) => i !== 0, () => 60)
  assert.equal(rows[0].n, 1); near(rows[0].deltaC, 2); near(rows[0].ir, 2)
  assert.equal(rows[1].n, 1); assert.equal(rows[1].minutes, 60); near(rows[1].uvIr, 3)
  assert.equal(rows[3].n, 0); assert.equal(rows[3].ir, null)
})

test('pearson', () => {
  near(pearson([1, 2, 3, null], [2, 4, 6, 1]), 1)
  assert.equal(pearson([1, 2], [1, 2]), null)
})
