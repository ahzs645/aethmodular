import assert from 'node:assert/strict'
import { test } from 'node:test'
import { EC, FABS, defaultMoisture, macPoints, trimBySite } from './moistureMac.ts'

const AERONET = 'Precipitable water (AERONET)', ERA5 = 'Precipitable water (ERA5)'
const row = (site, ec, fabs, moisture = {}) => ({ site, [EC]: ec, [FABS]: fabs, ...moisture })
const many = (site, n, moisture) => Array.from({ length: n }, (_, i) => row(site, 1 + i / 10, 20 + i, moisture(i)))

test('a filter needs Fabs, positive EC and the moisture', () => {
  const pts = macPoints([row('A', 2, 30, { x: 1 }), row('A', 0, 30, { x: 1 }), row('A', 2, null, { x: 1 }), row('A', 2, 30)], 'x')
  assert.equal(pts.length, 1)
  assert.equal(pts[0].mac, 15)
})

test('the trim is per site, so a site reads the same alone or among others', () => {
  const site = (name, shift) => Array.from({ length: 200 }, (_, i) => row(name, 1, shift + i, { x: 1 }))
  const a = site('A', 10), b = site('B', 1000)
  const alone = trimBySite(macPoints(a, 'x'))
  const together = trimBySite(macPoints([...a, ...b], 'x')).filter((p) => p.r.site === 'A')
  assert.deepEqual(together.map((p) => p.r), alone.map((p) => p.r))
  assert.ok(alone.length < 200 && alone.length >= 196)
})

test('AERONET stays the default where it covers the filters', () => {
  const rows = many('Addis', 20, (i) => ({ [AERONET]: 1 + i, [ERA5]: 1 + i }))
  assert.equal(defaultMoisture(rows, [AERONET, ERA5]), AERONET)
})

test('ERA5 opens where the photometer leaves too few filters', () => {
  const rows = many('Bishoftu', 21, (i) => ({ [AERONET]: i < 5 ? 1 : null, [ERA5]: 1 + i }))
  assert.equal(defaultMoisture(rows, [AERONET, ERA5]), ERA5)
})

test('with several sites the default has to serve each of them', () => {
  const rows = [...many('Addis', 20, (i) => ({ [AERONET]: 1 + i, [ERA5]: 1 + i })), ...many('Bishoftu', 21, (i) => ({ [AERONET]: i < 5 ? 1 : null, [ERA5]: 1 + i }))]
  assert.equal(defaultMoisture(rows, [AERONET, ERA5]), ERA5)
})

test('a site with few filters only asks to keep them, and nothing usable falls back', () => {
  assert.equal(defaultMoisture(many('Small', 4, () => ({ [AERONET]: 1 })), [AERONET, ERA5]), AERONET)
  assert.equal(defaultMoisture(many('Small', 4, (i) => ({ [AERONET]: i ? null : 1 })), [AERONET]), AERONET)
  assert.equal(defaultMoisture([], [AERONET]), AERONET)
  assert.equal(defaultMoisture(many('Small', 4, () => ({})), [AERONET]), null)
})
