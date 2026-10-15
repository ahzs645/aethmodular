import assert from 'node:assert/strict'
import { test } from 'node:test'
import { meanDirection, rainClass, spearman, weatherField } from './weather.ts'

const near = (a, b, tol = 1e-9) => assert.ok(Math.abs(a - b) < tol, `${a} != ${b}`)

test('spearman is 1 for any monotone relation and -1 reversed', () => {
  const x = [1, 2, 3, 4, 5, 6]
  near(spearman(x, x.map((v) => v ** 3)), 1)
  near(spearman(x, x.map((v) => -Math.exp(v))), -1)
})

test('spearman averages tied ranks and skips non-finite pairs', () => {
  // ranks x: 1, 2.5, 2.5, 4, 5 ; y: 1..5 -> rho = 0.975
  near(spearman([1, 2, 2, 3, 4, NaN], [1, 2, 3, 4, 5, 9]), 0.9746794344808963)
  assert.equal(spearman([1, 2, 3], [1, 2, 3]), null)
})

test('rain classes follow the addis_03 daily bins', () => {
  assert.equal(rainClass(0), 'No rain')
  assert.equal(rainClass(0.05), 'No rain')
  assert.equal(rainClass(2.4), 'Light (< 2.5 mm/day)')
  assert.equal(rainClass(2.5), 'Moderate (2.5–10)')
  assert.equal(rainClass(10), 'Heavy (≥ 10)')
})

test('mean direction wraps through north', () => {
  near(meanDirection([350, 10]), 0, 1e-9)
  near(meanDirection([80, 100]), 90, 1e-9)
})

test('weather fields map per source, with no station boundary-layer height', () => {
  assert.equal(weatherField('rh', 'ERA5'), 'Relative humidity (ERA5)')
  assert.equal(weatherField('pres', 'station'), 'Sea-level pressure (station)')
  assert.equal(weatherField('blh', 'station'), null)
})
