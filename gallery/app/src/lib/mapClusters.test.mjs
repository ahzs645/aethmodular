import test from 'node:test'
import assert from 'node:assert/strict'
import { worldPixel, fromWorldPixel, groupOverlapping } from './mapClusters.ts'

test('worldPixel and fromWorldPixel round-trip', () => {
  for (const [lon, lat, z] of [[38.76, 9.02, 3], [-63.6, 44.6, 6.5], [77.2, 28.6, 0]]) {
    const [x, y] = worldPixel(lon, lat, z)
    const [lo, la] = fromWorldPixel(x, y, z)
    assert.ok(Math.abs(lo - lon) < 1e-9 && Math.abs(la - lat) < 1e-9)
  }
  assert.deepEqual(worldPixel(0, 0, 0), [256, 256])
})

test('groupOverlapping merges overlapping discs and splits them when zoomed in', () => {
  // Delhi and Kanpur are ~400 km apart: 20 px discs collide at a world view, not at a regional one
  const sites = [{ code: 'DEL', lon: 77.2, lat: 28.6 }, { code: 'KAN', lon: 80.3, lat: 26.5 }, { code: 'ADD', lon: 38.8, lat: 9.0 }]
  const far = groupOverlapping(sites, 1, () => 20)
  assert.deepEqual(far.map((g) => g.members.map((m) => m.code)), [['DEL', 'KAN'], ['ADD']])
  assert.ok(far[0].lon > 77.2 && far[0].lon < 80.3)
  assert.deepEqual(far[1], { members: [sites[2]], lon: 38.8, lat: 9.0 })
  const near = groupOverlapping(sites, 6, () => 20)
  assert.equal(near.length, 3)
})

test('a group that grows can take in a further neighbour', () => {
  // points at 0, 20 and 60 px: radius-12 singles join only the first two, and the pair (radius 40) reaches the third
  const z = 4, [x0, y0] = worldPixel(0, 0, z)
  const at = (dx) => { const [lon, lat] = fromWorldPixel(x0 + dx, y0, z); return { lon, lat, k: dx } }
  const items = [at(0), at(20), at(60)]
  const radius = (m) => (m.length === 1 ? 12 : 40)
  assert.deepEqual(groupOverlapping(items, z, radius).map((g) => g.members.map((m) => m.k)), [[0, 20, 60]])
  assert.equal(groupOverlapping(items, z, () => 12).length, 2)
})
