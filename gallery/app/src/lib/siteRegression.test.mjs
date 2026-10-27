import assert from 'node:assert/strict'
import { test } from 'node:test'
import { bandRegressions, groupedRegressions } from './siteRegression.ts'
import { regression, regressionSamples } from './stats.ts'

const near = (a, b) => assert.ok(Math.abs(a - b) < 1e-9, `${a} != ${b}`)
const access = {
  x: p => p.x, y: p => p.y, site: p => p.site, color: p => p.color,
  options: members => ({ errorsInVariables: true, lambda: 2.96,
    sigmaX: members.map(p => p.sx), sigmaY: members.map(p => p.sy),
    bootstrapGroups: members.map(p => p.month) }),
}
const rows = ['A', 'B'].flatMap((site, i) => [1, 2, 3, 4].map(x => ({
  site, x: x + i * 10, y: (i + 1) * (x + i * 10) - i * 3,
  color: i ? '#bb0000' : '#0000bb', sx: x / 10, sy: x / 5, month: `${site}:2026-0${x}`,
})))

test('separate site fits recover different lines and their own observed ranges', () => {
  const groups = groupedRegressions(rows, 'site', access)
  assert.equal(groups.length, 2)
  for (const [i, g] of groups.entries()) {
    assert.equal(g.name, i ? 'B' : 'A')
    assert.equal(g.stats.n, 4)
    near(g.stats.demingSlope, i + 1); near(g.stats.demingIntercept, -i * 3)
    near(g.stats.yorkSlope, i + 1); near(g.stats.yorkIntercept, -i * 3)
    assert.deepEqual(g.domain, i ? [11, 14] : [1, 4])
    assert.equal(g.color, rows[i * 4].color)
  }
})

test('pair filtering preserves per-site uncertainty and bootstrap alignment', () => {
  const missing = [{ ...rows[0], x: null }, { ...rows[1], y: NaN }, { ...rows[2], x: Infinity }]
  const groups = groupedRegressions([...missing, ...rows], 'site', access)
  for (const g of groups) {
    const expected = rows.filter(p => p.site === g.name)
    const sample = regressionSamples.get(g.stats)
    assert.deepEqual(sample.x, expected.map(p => p.x))
    assert.deepEqual(sample.y, expected.map(p => p.y))
    assert.deepEqual(sample.opts.sigmaX, expected.map(p => p.sx))
    assert.deepEqual(sample.opts.sigmaY, expected.map(p => p.sy))
    assert.deepEqual(sample.opts.bootstrapGroups, expected.map(p => p.month))
    assert.equal(sample.opts.lambda, 2.96)
  }
})

test('pooled mode reproduces the original regression of all paired filters', () => {
  const [g] = groupedRegressions(rows, 'pooled', access)
  assert.equal(g.name, 'All selected sites')
  assert.equal(g.color, undefined)
  assert.deepEqual(g.stats, regression(rows.map(p => p.x), rows.map(p => p.y), access.options(rows)))
  assert.deepEqual(g.domain, [1, 14])
})

test('sparse or constant sites cannot create misleading fitted lines', () => {
  const sparse = rows.filter(p => p.site === 'A').slice(0, 2)
  const constant = rows.filter(p => p.site === 'B').map(p => ({ ...p, x: 1 }))
  assert.deepEqual(groupedRegressions([...sparse, ...constant], 'site', access), [])
  assert.deepEqual(groupedRegressions([], 'site', access), [])
})

test('one fit per band: in the bands\' order and colours, points in no band left out', () => {
  // two bands with exact lines (y = 2x and y = 3x + 1) and a point in no band
  const pts = [...[1, 2, 3, 4].map(x => ({ x, y: 2 * x, band: 'low' })), ...[5, 6, 7, 8].map(x => ({ x, y: 3 * x + 1, band: 'high' })), { x: 9, y: 0, band: null }]
  const band = { labels: ['low', 'high'], labelOf: p => p.band, colorOf: l => (l === 'low' ? '#111111' : '#222222') }
  const fits = bandRegressions([...pts].reverse(), band, { x: p => p.x, y: p => p.y, options: () => ({}) })
  assert.deepEqual(fits.map(f => [f.name, f.color, f.stats.n]), [['low', '#111111', 4], ['high', '#222222', 4]])
  near(fits[0].stats.slope, 2); near(fits[1].stats.slope, 3); near(fits[1].stats.intercept, 1)
})
