import test from 'node:test'
import assert from 'node:assert/strict'
import { cellsFromRose8, cellsFromSparse, classOf, concentrationEdges, polarGrid, pollutionRose, ringLabelBearing, speedBins, weightedQuantile } from './windRose.ts'

const near = (a, b, eps = 1e-9) => assert.ok(Math.abs(a - b) < eps, `${a} ≉ ${b}`)
const filter = (id, value, cells, calm = 0) => ({ id, value, cells: cells.map(([sector, speed, hours]) => ({ sector, speed, hours })), calm })

test('roses.json counts decode to cells: 8 sectors × 3 speed classes, then calm', () => {
  const counts = new Array(25).fill(0)
  counts[0] = 2; counts[3 * 3 + 2] = 5; counts[24] = 4
  assert.deepEqual(cellsFromRose8(counts), { cells: [{ sector: 0, speed: 0, hours: 2 }, { sector: 3, speed: 2, hours: 5 }], calm: 4 })
})

test('wind.json sparse cells decode: calm first, then cell = sector × bins + bin', () => {
  // export_weather.wind_cells on [0.2 m/s, 0.7 m/s @ 355°, 7 m/s @ 12°, 3.2 m/s @ 180°] gives this
  assert.deepEqual(cellsFromSparse([1, 0, 1, 19, 1, 85, 1], 10), {
    calm: 1, cells: [{ sector: 0, speed: 0, hours: 1 }, { sector: 1, speed: 9, hours: 1 }, { sector: 8, speed: 5, hours: 1 }],
  })
})

test('weighted quantile: equal weights give the ordinary median; weights pull it', () => {
  near(weightedQuantile([1, 2, 3, 4], [1, 1, 1, 1], 0.5), 2.5)
  near(weightedQuantile([3, 1, 2], [1, 1, 1], 0.5), 2)
  // most of the weight on 10: 1 sits at 0.05 of the weight, 10 at 0.55, so the median is 0.9 of the way
  near(weightedQuantile([1, 10], [1, 9], 0.5), 9.1)
  assert.equal(weightedQuantile([1, 2], [0, 0], 0.5), null)
  near(weightedQuantile([1, 2, 3], [1, 1, 1], 0), 1)
  near(weightedQuantile([1, 2, 3], [1, 1, 1], 1), 3)
})

test('class edges follow the quantiles, rounded, and only rise', () => {
  assert.deepEqual(concentrationEdges([1, 2, 3, 4, 5, 6, 7, 8, 9]), [3, 5, 7, 8.2])
  assert.deepEqual(concentrationEdges([5, 5, 5, 5]), [5])
  assert.equal(classOf(1, [3, 5]), 0)
  assert.equal(classOf(3, [3, 5]), 1)
  assert.equal(classOf(9, [3, 5]), 2)
})

test('pollution rose: a sector is the hour-weighted statistic of the filters whose air came from it', () => {
  const fs = [
    filter('a', 10, [[0, 0, 18], [2, 1, 6]], 0),
    filter('b', 20, [[0, 1, 6], [2, 0, 18]]),
    filter('c', 40, [[2, 2, 24]], 3),
  ]
  const rose = pollutionRose(fs, 4, { stat: 'mean', edges: [15, 30], minFilters: 1 })
  // N: 18 h at 10 and 6 h at 20 → 12.5; E: 6 h at 10, 18 at 20, 24 at 40 → 28.75
  near(rose.sectors[0].value, 12.5)
  near(rose.sectors[2].value, 28.75)
  assert.equal(rose.sectors[1].value, null)
  assert.deepEqual(rose.sectors[2].classHours, [6, 18, 24])
  assert.equal(rose.hours, 72)
  assert.equal(rose.calm, 3)
  assert.equal(rose.sectors[2].filters, 3)
  // the median over E's hours: 24 of 48 h are at 40, so it sits between 20 and 40
  const med = pollutionRose(fs, 4, { stat: 'median', edges: [], minFilters: 1 }).sectors[2].value
  assert.ok(med > 20 && med < 40)
  // fewer filters than minFilters: no statistic, but the hours still count
  const strict = pollutionRose(fs, 4, { stat: 'mean', edges: [], minFilters: 3 })
  assert.equal(strict.sectors[0].value, null)
  assert.equal(strict.sectors[0].hours, 24)
})

test('polar grid: raw cells are hour-weighted means; smoothing spreads into neighbours and wraps north', () => {
  const fs = [filter('a', 10, [[0, 1, 10]]), filter('b', 30, [[0, 1, 10]]), filter('c', 50, [[3, 1, 20]])]
  const raw = polarGrid(fs, 4, 3, { smooth: false, minHours: 5, minFilters: 2 })
  near(raw.find((c) => c.sector === 0 && c.speed === 1).value, 20)
  // one filter only: below minFilters
  assert.equal(raw.find((c) => c.sector === 3 && c.speed === 1).value, null)
  const sm = polarGrid(fs, 4, 3, { smooth: true, minHours: 1, minFilters: 1 })
  const at = (s, k) => sm.find((c) => c.sector === s && c.speed === k)
  // sector 3 is next to sector 0 across north: both pull towards each other
  assert.ok(at(0, 1).value > 20 && at(0, 1).value < 50)
  assert.ok(at(3, 1).value > 20 && at(3, 1).value < 50)
  assert.ok(at(0, 0).hours > 0)
})

test('speed bins: median and IQR (from 4) of the filters in each bin', () => {
  const pts = [0.2, 0.4, 0.6, 0.8, 1.2].map((speed, i) => ({ speed, value: i + 1 }))
  const b = speedBins(pts, 1)
  assert.equal(b.length, 2)
  assert.deepEqual([b[0].n, b[0].median, b[0].q1, b[0].q3], [4, 2.5, 1.75, 3.25])
  assert.deepEqual([b[1].n, b[1].median, b[1].q1], [1, 5, null])
})

test('ring labels sit in a gap between sectors, off the north axis', () => {
  assert.equal(ringLabelBearing(8), 22.5)
  assert.equal(ringLabelBearing(16), 33.75)
  assert.equal(ringLabelBearing(8, 60), 67.5)
})
