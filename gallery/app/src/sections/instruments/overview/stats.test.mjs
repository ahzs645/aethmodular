// node --test src/sections/instruments/overview/stats.test.mjs   (from gallery/app)
import assert from 'node:assert/strict'
import { test } from 'node:test'
import { dayToDay, movingMean, powerLawExponent, quantileSorted, relativeSpectra, SIGMA, summarize, toDays, weekendEffect } from './stats.ts'

const near = (a, b, tol = 1e-9) => assert.ok(Math.abs(a - b) < tol, `${a} != ${b}`)
const SEASONS = [
  { name: 'Dry', months: [10, 11, 12, 1, 2], color: '#000' },
  { name: 'Belg', months: [3, 4, 5], color: '#111' },
  { name: 'Kiremt', months: [6, 7, 8, 9], color: '#222' },
]
const file = (starts, ir, extra = {}) => ({
  start: starts, UV: ir, Blue: ir, Green: ir, Red: ir, IR: ir,
  coverage: starts.map(() => null), flow: starts.map(() => null), ...extra,
})

test('a bin counts for the day its middle falls on', () => {
  // Addis bins start 09:00: the same date; Beijing's 15:00 bins: the next date
  const { days } = toDays(file(['2023-02-28 09:00', '2023-02-28 15:00', '2023-09-30 15:00'], [1, 2, 3]), SEASONS, 'none')
  assert.deepEqual(days.map((d) => d.date), ['2023-02-28', '2023-03-01', '2023-10-01'])
  assert.deepEqual(days.map((d) => d.season.name), ['Dry', 'Belg', 'Dry'])
  // 28 Feb 2023 was a Tuesday, 1 Mar a Wednesday, 1 Oct a Sunday
  assert.deepEqual(days.map((d) => d.dow), [1, 2, 6])
})

test('the addis_02 screen drops negatives, then values above mean + 3 SD', () => {
  const ir = [...Array(20).fill(1), -0.5, 50]
  const starts = ir.map((_, i) => `2023-01-${String(i + 1).padStart(2, '0')} 09:00`)
  const neg = toDays(file(starts, ir), SEASONS, 'neg')
  assert.equal(neg.days.filter((d) => d.bc.IR === null).length, 1)
  assert.deepEqual(neg.dropped.IR, { neg: 1, high: 0 })
  const sd3 = toDays(file(starts, ir), SEASONS, '3sd')
  // mean and SD over the 21 non-negative values put 50 above the limit
  assert.deepEqual(sd3.dropped.IR, { neg: 1, high: 1 })
  assert.equal(toDays(file(starts, ir), SEASONS, 'none').days.filter((d) => d.bc.IR === null).length, 0)
})

test('quantiles interpolate like pandas', () => {
  near(quantileSorted([1, 2, 3, 4], 0.25), 1.75)
  const s = summarize([4, null, 1, 3, 2, NaN])
  assert.equal(s.n, 4)
  near(s.mean, 2.5)
  near(s.sd, Math.sqrt(5 / 3))
  near(s.median, 2.5)
  assert.equal(summarize([null]), null)
})

test('the moving mean counts calendar days, not rows, and needs enough of them', () => {
  const starts = ['2023-01-01 09:00', '2023-01-02 09:00', '2023-01-03 09:00', '2023-03-01 09:00']
  const { days } = toDays(file(starts, [1, 2, 3, 100]), SEASONS, 'none')
  const m = movingMean(days, (d) => d.bc.IR, 1, 2)
  assert.deepEqual(m, [1.5, 2, 2.5, null])
})

test('day-to-day change only across bins 24 h apart', () => {
  const { days } = toDays(file(['2023-01-01 09:00', '2023-01-02 09:00', '2023-01-04 09:00', '2023-01-05 09:00'], [1, 3, 10, 4]), SEASONS, 'none')
  assert.deepEqual(dayToDay(days, (d) => d.bc.IR).map((x) => x.dv), [2, -6])
})

test('weekend effect: change relative to the weekday mean', () => {
  // two weeks from Monday 2 Jan 2023: weekdays 2, weekends 1
  const starts = [], ir = []
  for (let i = 0; i < 14; i++) {
    starts.push(`2023-01-${String(i + 2).padStart(2, '0')} 09:00`)
    ir.push((i % 7 >= 5 ? 1 : 2) + (i % 3) * 0.01)
  }
  const { days } = toDays(file(starts, ir), SEASONS, 'none')
  const w = weekendEffect(days, (d) => d.bc.IR)
  assert.equal(w.weekday.n, 10)
  assert.equal(w.weekend.n, 4)
  near(w.change, ((w.weekend.mean - w.weekday.mean) / w.weekday.mean) * 100)
  assert.ok(w.change < -45 && w.lo < w.change && w.hi > w.change && w.p < 0.001)
})

test('a λ⁻¹ absorption spectrum has exponent 1, and BCc ratios divide σ out', () => {
  const lambda = { UV: 375, Blue: 470, Green: 528, Red: 625, IR: 880 }
  // BCc for b_abs ∝ λ⁻¹: b/σ
  const bc = Object.fromEntries(Object.entries(lambda).map(([c, l]) => [c, [880 / l / SIGMA[c]]]))
  const { days } = toDays({ start: ['2023-01-01 09:00'], ...bc, coverage: [null], flow: [null] }, SEASONS, 'none')
  const abs = relativeSpectra(days, 'absorption')
  near(abs.UV[0], 880 / 375)
  near(powerLawExponent(Object.values(lambda), Object.keys(lambda).map((c) => abs[c][0])), 1, 1e-12)
  const ratio = relativeSpectra(days, 'bcc')
  near(ratio.UV[0], (880 / 375) * (SIGMA.IR / SIGMA.UV))
})
