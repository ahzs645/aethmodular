import test from 'node:test'
import assert from 'node:assert/strict'
import { lineRuns, nearestIndex, stackGeometry } from './stacked.ts'

test('natural heights: each panel its height or base × share, gaps between, margins outside', () => {
  const g = stackGeometry([{}, { share: 2 }, { height: 50 }], { fit: null, base: 90, gap: 10, top: 8, bottom: 30 })
  assert.deepEqual(g.heights, [90, 180, 50])
  assert.deepEqual(g.tops, [8, 108, 298])
  assert.equal(g.total, 8 + 30 + 320 + 20)
})

test('a fitted height is shared by share, never below the minimum', () => {
  // the timelines' old rule: (fit − margins − gaps) / n, at least 40
  const g = stackGeometry([{}, {}, {}, {}, {}], { fit: 500, base: 92, gap: 14, top: 8, bottom: 30, min: 40 })
  const each = (500 - 38 - 4 * 14) / 5
  g.heights.forEach((h) => assert.equal(h, each))
  assert.equal(g.total, 500)
  const tiny = stackGeometry([{}, { share: 3 }], { fit: 100, base: 92, gap: 10, top: 0, bottom: 0, min: 40 })
  assert.deepEqual(tiny.heights, [40, 67.5])
})

test('an empty stack is just its margins', () => {
  assert.deepEqual(stackGeometry([], { fit: null, base: 90, gap: 10, top: 8, bottom: 30 }), { heights: [], tops: [], total: 38 })
})

const day = (n) => new Date(Date.UTC(2024, 0, 1) + n * 864e5)
test('lineRuns breaks at a missing value and at a gap longer than the limit', () => {
  const pts = [[0, 1], [1, 2], [2, null], [3, 4], [4, 5], [9, 6], [10, NaN], [11, 7]].map(([d, v]) => ({ t: day(d), v }))
  const runs = lineRuns(pts, (p) => p.t, (p) => p.v, 1.5).map((r) => r.map((q) => q.v))
  assert.deepEqual(runs, [[1, 2], [4, 5], [6], [7]])
  // no limit: only the missing values break it
  assert.deepEqual(lineRuns(pts, (p) => p.t, (p) => p.v).map((r) => r.length), [2, 3, 1])
})

test('nearestIndex is the nearest item, the earlier one on a tie', () => {
  const xs = [0, 10, 20, 30]
  assert.equal(nearestIndex(xs, (x) => x, -5), 0)
  assert.equal(nearestIndex(xs, (x) => x, 14), 1)
  assert.equal(nearestIndex(xs, (x) => x, 15), 1)
  assert.equal(nearestIndex(xs, (x) => x, 16), 2)
  assert.equal(nearestIndex(xs, (x) => x, 99), 3)
  assert.equal(nearestIndex([], (x) => x, 1), -1)
})
