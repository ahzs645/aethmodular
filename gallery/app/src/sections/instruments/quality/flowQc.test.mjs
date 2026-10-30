// flowQc.ts against pandas, numpy and scipy: synthetic cases, and the exported minute and daily records against
// values computed in Python on the same data (the reference numbers below) and the figures in notebooks/qc/figures
import assert from 'node:assert/strict'
import fs from 'node:fs'
import { test } from 'node:test'
import { breakGaps, dailyGaps, expandHours, kdeFromHist, quantileLinear, ratioClass, rollingCentred, tukeyBox, unionDays } from './flowQc.ts'

const path = (f) => new URL(`../../../../public/data/aethalometer/quality/${f}`, import.meta.url)
const data = (f) => (fs.existsSync(path(f)) ? JSON.parse(fs.readFileSync(path(f), 'utf8')) : null)
const near = (a, b, tol, what) => assert.ok(Math.abs(a - b) <= tol, `${what}: ${a} vs ${b}`)

test('rollingCentred matches pandas rolling(center=True) with its default min_periods', () => {
  const s = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  // pandas: Series(range(10)).rolling(4, center=True).median() and .rolling(3, center=True)
  assert.deepEqual(rollingCentred(s, 4, 'median'), [null, null, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, null])
  assert.deepEqual(rollingCentred(s, 3, 'median'), [null, 1, 2, 3, 4, 5, 6, 7, 8, null])
  assert.deepEqual(rollingCentred([1, null, 3, 4, 5], 2, 'mean'), [null, null, null, 3.5, 4.5])
})

test('quantiles and the Tukey box follow numpy and matplotlib', () => {
  near(quantileLinear([1, 2, 3, 4], 0.25), 1.75, 1e-12, 'q25')
  const b = tukeyBox([1, 2, 3, 4, 5, 6, 7, 8, 100])
  assert.equal(b.q25, 3); assert.equal(b.q75, 7); assert.equal(b.hi, 8); assert.equal(b.above, 1); assert.equal(b.lo, 1)
  assert.equal(tukeyBox([null, NaN]), null)
})

test('ratio classes: pass within 5 %, warn within 10 %', () => {
  assert.equal(ratioClass(1.04, 1), 'pass'); assert.equal(ratioClass(0.92, 1), 'warn'); assert.equal(ratioClass(1.2, 1), 'fail')
})

test('a Gaussian KDE of a histogram is a density and is within 2 % of scipy on the Pasadena minutes', (t) => {
  const site = data('minute/USPA.json')
  if (!site) return t.skip('minute export absent')
  const f = site.flow_qc
  const grid = Array.from({ length: 401 }, (_, i) => 1.8 + (i * 1.4) / 400)
  const d = kdeFromHist(f.R_hist, 0.15997886, grid)
  // the 1 % of minutes beyond the exported range carry no kernel: the curve integrates to the 99 % inside it
  near(d.reduce((s, v) => s + v, 0) * (1.4 / 400), 0.99, 0.01, 'integral')
  // scipy.stats.gaussian_kde on the 471,232 minute ratios, evaluated at 2.2, 2.4, 2.467, 2.6, 2.8
  const ref = [0.5715, 2.9294, 2.7309, 1.7945, 0.73]
  // the exported bins (~0.009) are close to the bandwidth (~0.012), so the binned curve sits within 2 % of scipy's
  kdeFromHist(f.R_hist, 0.15997886, [2.2, 2.4, 2.467, 2.6, 2.8]).forEach((v, i) => near(v, ref[i], 0.02 * ref[i], `kde ${i}`))
})

test('the hourly rows and their rolling windows match pandas on the Pasadena record', (t) => {
  const fl = data('minute/USPA.flow.json')
  if (!fl) return t.skip('flow detail absent')
  const hours = expandHours(fl.hourly)
  assert.equal(hours.length, 7862); assert.equal(fl.hourly.R.length, 7862)
  assert.equal(hours[0].toISOString().slice(0, 16), '2023-06-23T08:00')
  // pandas on the unrounded hourly medians: rolling(24, center=True).median() and rolling(168).mean() of the pass share
  const r24 = rollingCentred(fl.hourly.R, 24, 'median'), p168 = rollingCentred(fl.hourly.pass, 168, 'mean')
  for (const [i, v] of [[100, 2.4209], [1000, 2.3859], [5000, 2.4494]]) near(r24[i], v, 6e-4, `R 24 h at ${i}`)
  for (const [i, v] of [[100, 0.8694], [1000, 0.3381], [5000, 0.929]]) near(p168[i] / 100, v, 5e-3, `pass 7 d at ${i}`)
})

test('the all-sites records reproduce the figures: Beijing from its daily bins, Pasadena (JPL) as a union', (t) => {
  const chts = data('CHTS.json'), uspa = data('USPA.json'), fl = data('minute/USPA.flow.json')
  if (!chts || !uspa || !fl) return t.skip('exports absent')
  const daily = (q) => new Map(q.start.map((s, i) => [s.slice(0, 10), q.series.flow1[i] / q.series.flow2[i]]).filter(([, v]) => Number.isFinite(v)))
  const bj = tukeyBox([...daily(chts).values()])
  // all_sites_flow_ratio_4sites.png: Beijing n = 222 days, median 1.83, box 1.74–1.95
  assert.equal(bj.n, 222); near(bj.median, 1.833, 1e-3, 'Beijing median'); near(bj.q25, 1.741, 1e-3, 'q25'); near(bj.q75, 1.946, 1e-3, 'q75')
  const minute = new Map(fl.daily9.day.map((d, i) => [d, fl.daily9.R_med[i]]))
  assert.equal(minute.size, 331)                                   // the 6-site figure's Pasadena: n = 331 d, median 2.47
  near(tukeyBox([...minute.values()]).median, 2.469, 1e-3, 'Pasadena median')
  const both = unionDays(minute, daily(uspa))
  const b = tukeyBox([...both.values()])
  // the figure's Pasadena (JPL): 629 days (one more than here), median 2.28, IQR 1.32–2.49, 2021-11-08 → 2024-08-14
  assert.equal(both.size, 628); near(b.median, 2.28, 0.005, 'union median'); near(b.q25, 1.316, 1e-3, 'q25'); near(b.q75, 2.485, 1e-3, 'q75')
  assert.equal([...both.keys()][0], '2021-11-08'); assert.equal([...both.keys()].at(-1), '2024-08-14')
  const pts = breakGaps(both, 7)
  assert.ok(pts.some((p) => p.y === null) && pts.filter((p) => p.y !== null).length === 628)
})

test('dailyGaps breaks a daily line only across a missing day', () => {
  const d = (s) => new Date(`${s}T00:00:00Z`)
  const out = dailyGaps([{ x: d('2024-01-01'), y: 1 }, { x: d('2024-01-02'), y: 2 }, { x: d('2024-01-05'), y: 3 }])
  assert.deepEqual(out.map((p) => p.y), [1, 2, null, 3])
})
