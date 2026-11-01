import assert from 'node:assert/strict'
import { test } from 'node:test'
import { scaleLinear, scaleLog } from 'd3'
import { orientErrorRatio, orientPoints, sourceScales } from './plotOrientation.ts'
import { regression } from './stats.ts'

test('swapping preserves filter records, exclusions and the source points; swapping twice restores them', () => {
  const record = { id: 'ETAD-0035' }
  const points = [{ x: -2, y: 0, id: record.id, rec: record, excluded: true }]
  assert.equal(orientPoints(points, false), points)
  const swapped = orientPoints(points, true)
  assert.deepEqual([swapped[0].x, swapped[0].y], [0, -2])
  assert.equal(swapped[0].rec, record)
  assert.equal(swapped[0].excluded, true)
  assert.deepEqual(orientPoints(swapped, true), points)
  assert.deepEqual([points[0].x, points[0].y], [-2, 0])
})

test('live fits reverse the measurement error ratio and refit OLS rather than reciprocating it', () => {
  const x = [1, 2, 3, 4, 7], y = [2, 3, 5, 4, 9], lambda = 2.96
  const original = regression(x, y, { errorsInVariables: true, lambda })
  const reversed = regression(y, x, { errorsInVariables: true, lambda: orientErrorRatio(lambda, true) })
  assert.ok(Math.abs(original.demingSlope * reversed.demingSlope - 1) < 1e-12)
  assert.ok(Math.abs(reversed.demingIntercept + original.demingIntercept / original.demingSlope) < 1e-12)
  assert.ok(Math.abs(original.slope * reversed.slope - original.r2) < 1e-12)
  assert.ok(Math.abs(reversed.slope - 1 / original.slope) > 0.01)
  assert.equal(orientErrorRatio(lambda, false), lambda)
  assert.equal(orientErrorRatio(null, true), 1)
})

test('saved layers retain their domains and log mapping when transposed, including horizontal lines', () => {
  const side = 400
  const x = scaleLog().domain([1, 100]).range([0, side])
  const y = scaleLinear().domain([-4, 12]).range([side, 0])
  const original = sourceScales(x, y, false)
  assert.equal(original.x, x)
  assert.equal(original.y, y)
  const source = sourceScales(x, y, true)
  // OrientedPlotLayer maps (px, py) to (side - py, side - px).
  for (const [oldX, oldY] of [[-4, 1], [0, 10], [12, 100]]) {
    assert.ok(Math.abs(side - source.y(oldY) - x(oldY)) < 1e-12)
    assert.ok(Math.abs(side - source.x(oldX) - y(oldX)) < 1e-12)
  }
  assert.deepEqual(source.x.domain(), y.domain())
  assert.deepEqual(source.y.domain(), x.domain())
  const verticalX = side - source.y(10)
  assert.ok(Number.isFinite(verticalX))
  assert.notEqual(side - source.x(-4), side - source.x(12))
  assert.deepEqual(x.range(), [0, side])
  assert.deepEqual(y.range(), [side, 0])
})
