import assert from 'node:assert/strict'
import { test } from 'node:test'
import { mostVisibleChart, readChartDisplay, sharedChartDisplay } from './chartDisplay.ts'

test('shared changes remove only the corresponding per-chart exceptions', () => {
  const before = { controlsSide: 'top', charts: { scatter: { controlsSide: 'left', density: 'readable' }, hexbin: { controlsSide: 'right' } } }
  assert.deepEqual(sharedChartDisplay(before, { controlsSide: 'right' }), { controlsSide: 'right', charts: { scatter: { density: 'readable' } } })
  assert.equal(before.charts.scatter.controlsSide, 'left')
  assert.deepEqual(sharedChartDisplay(before, { density: 'compact' }).charts.scatter, { controlsSide: 'left' })
})

test('stored settings validate values and retain explicit false and compact overrides', () => {
  assert.deepEqual(readChartDisplay({ controlsSide: 'elsewhere', charts: { scatter: { explain: false, density: 'compact', theme: 'dark' }, bad: { theme: 'sepia', controlsSide: 'bottom' }, broken: null } }), {
    controlsSide: 'top', charts: { scatter: { explain: false, density: 'compact', theme: 'dark' } },
  })
  for (const bad of [null, false, [], { charts: [] }]) assert.deepEqual(readChartDisplay(bad), { controlsSide: 'top', charts: {} })
})

const viewport = { left: 0, top: 60, right: 1000, bottom: 800 }
const chart = (key, top, bottom, extra = {}) => ({ key, title: key, rect: { left: 0, right: 800, top, bottom }, ...extra })

test('target follows the largest visible area, excluding the sticky header and offscreen charts', () => {
  assert.equal(mostVisibleChart([chart('above', -200, 200), chart('main', 220, 900), chart('below', 900, 1500)], viewport)?.key, 'main')
  assert.equal(mostVisibleChart([chart('tall', 0, 1800), chart('small', 200, 400)], viewport)?.key, 'tall')
  assert.equal(mostVisibleChart([chart('hidden', 900, 1500)], viewport), null)
  assert.equal(mostVisibleChart([], viewport), null)
})

test('focused charts win over the page behind them and stable order breaks equal-area ties', () => {
  assert.equal(mostVisibleChart([chart('behind', 60, 800), chart('focused', 200, 600, { focused: true })], viewport)?.key, 'focused')
  assert.equal(mostVisibleChart([chart('first', 200, 400), chart('second', 200, 400)], viewport)?.key, 'first')
  assert.equal(mostVisibleChart([chart('clipped', 200, 100)], viewport), null)
})
