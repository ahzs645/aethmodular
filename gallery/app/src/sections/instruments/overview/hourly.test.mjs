// node --test src/sections/instruments/overview/hourly.test.mjs   (from gallery/app)
import assert from 'node:assert/strict'
import { test } from 'node:test'
import { dayShapes, episodeRate, hourQuantile, hourShares, toEpisodes, toHours, weekHourGrid } from './hourly.ts'

const near = (a, b, tol = 1e-9) => assert.ok(Math.abs(a - b) < tol, `${a} != ${b}`)
const SEASONS = [
  { name: 'Dry', months: [10, 11, 12, 1, 2], color: '#000' },
  { name: 'Wet', months: [3, 4, 5, 6, 7, 8, 9], color: '#111' },
]

test('hours run on from the first, skip nulls, and clip by date', () => {
  // 2023-02-28 22:00 (a Tuesday), then 23:00 missing, then 1 Mar 00:00 and 01:00
  const f = { site: 'X', first: '2023-02-28 22:00', offset: 3, unit: 'ng/m3', IR: [1000, null, 3000, 4000] }
  const h = toHours(f, SEASONS)
  assert.deepEqual(h.map((x) => [x.date, x.hour, x.dow, x.season.name, x.v]), [
    ['2023-02-28', 22, 1, 'Dry', 1], ['2023-03-01', 0, 2, 'Wet', 3], ['2023-03-01', 1, 2, 'Wet', 4]])
  assert.equal(toHours(f, SEASONS, '2023-03-01', null).length, 2)
  assert.equal(toHours(f, SEASONS, null, '2023-02-28').length, 1)
})

test('a day needs enough hours; its peak, peak hour and peak-to-mean', () => {
  const IR = Array.from({ length: 48 }, (_, i) => (i === 7 ? 9000 : 1000))
  IR.splice(30, 10, ...Array(10).fill(null))   // day 2 keeps only 14 hours
  const days = dayShapes(toHours({ site: 'X', first: '2023-01-02 00:00', offset: 3, unit: 'ng/m3', IR }, SEASONS), 18)
  assert.equal(days.length, 1)
  const d = days[0]
  assert.equal(d.peakHour, 7)
  near(d.mean, (23 + 9) / 24)
  near(d.ratio, 9 / ((23 + 9) / 24))
})

test('hour shares and the hourly quantile', () => {
  const s = hourShares([7, 7, 19, 7])
  near(s[7], 0.75)
  near(s[19], 0.25)
  assert.equal(s[0], 0)
  assert.deepEqual(hourShares([]).slice(0, 2), [null, null])
  const h = toHours({ site: 'X', first: '2023-01-02 00:00', offset: 0, unit: 'ng/m3', IR: [1000, 2000, 3000, 4000] }, SEASONS)
  near(hourQuantile(h, 0.5), 2.5)
})

test('the weekday-by-hour grid averages each cell', () => {
  // Monday 2 Jan 2023 00:00 and 01:00, then Monday 9 Jan 00:00 (168 h later)
  const IR = Array(169).fill(null)
  IR[0] = 1000; IR[1] = 2000; IR[168] = 3000
  const g = weekHourGrid(toHours({ site: 'X', first: '2023-01-02 00:00', offset: 0, unit: 'ng/m3', IR }, SEASONS))
  assert.equal(g.mean[0][0], 2)
  assert.equal(g.n[0][0], 2)
  assert.equal(g.mean[0][1], 2)
  assert.equal(g.mean[1][0], null)
  near(g.overall, 2)
})

test('episodes are dated, seasoned and rated per 30 days of data', () => {
  const cols = { start: ['2023-01-02 07:00', '2023-06-05 19:00'], hours: [3, 2], peak: [30000, 20000], peak_at: [1, 0], mean: [25000, 18000], excess: [9000, 3000] }
  const eps = toEpisodes(cols, SEASONS)
  assert.deepEqual(eps.map((e) => [e.date, e.startHour, e.dow, e.season.name, e.peak]), [['2023-01-02', 7, 0, 'Dry', 30], ['2023-06-05', 19, 0, 'Wet', 20]])
  assert.equal(toEpisodes(cols, SEASONS, '2023-02-01').length, 1)
  const r = episodeRate(eps, 1440)
  assert.equal(r.n, 2)
  near(r.per30, 1)
  near(r.hours.median, 2.5)
})
