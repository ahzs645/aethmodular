import test from 'node:test'
import assert from 'node:assert/strict'
import { addTile, arrange, dashParams, firstFit, nextTileId, parseDash, parseSheets, parseTiles, removeTile, rowsUsed, sheetsParam, tilesParam, updateTile } from './dashboard.ts'

const T = (id, x, y, w, h, kind = 'histogram', fields = []) => ({ id, kind, x, y, w, h, fields })
const at = (tiles) => Object.fromEntries(tiles.map((t) => [t.id, [t.x, t.y, t.w, t.h]]))

test('tiles and sheets round-trip through the URL, names and fields with separators included', () => {
  const tiles = [T('t1', 0, 0, 6, 4, 'scatter', ['HIPS BC', 'EC (FTIR, spline)']), T('t2', 6, 0, 6, 4, 'histogram', ['a:b / c,d;e'])]
  assert.deepEqual(parseTiles(tilesParam(tiles)), tiles)
  const sheets = [{ id: 's1', name: 'Carbon: EC; OC', tiles }, { id: 's2', name: 'Empty', tiles: [] }]
  assert.deepEqual(parseSheets(sheetsParam(sheets)), sheets)
  assert.equal(parseSheets(null), null)
  assert.equal(sheetsParam(null), null)
})

test('a mangled or duplicate tile is dropped; sizes and places are clamped', () => {
  assert.deepEqual(parseTiles('t1:histogram:0:0:6:4:%E0%A4%A,t2:histogram:0:0:6:4:x,t2:scatter:0:0:12:6').map((t) => t.id), ['t2'])
  assert.deepEqual(at(parseTiles('t1:map:11:0:40:0')), { t1: [0, 0, 12, 1] })
})

test('the dashboard state leaves its defaults out of the URL and reads back', () => {
  const d = parseDash(new URLSearchParams(''))
  assert.deepEqual([d.scope, d.layout, d.sheet], ['sites', 'grid', 0])
  assert.deepEqual(Object.values(dashParams(d)).filter(Boolean), [])
  const d2 = { scope: 'site', layout: 'focus', sheet: 1, sheets: { site: [{ id: 's1', name: 'A', tiles: [T('t1', 0, 0, 3, 2)] }], sites: null, network: null } }
  const p = new URLSearchParams(Object.entries(dashParams(d2)).filter(([, v]) => v !== null))
  assert.deepEqual(parseDash(p), d2)
})

test('arrange: overlaps are pushed down, then everything floats up', () => {
  // dropping a on top of b pushes b under it
  assert.deepEqual(at(arrange([T('a', 0, 0, 6, 3), T('b', 0, 0, 6, 3)], 'a')), { a: [0, 0, 6, 3], b: [0, 3, 6, 3] })
  // a hole closes: c floats up under a
  assert.deepEqual(at(arrange([T('a', 0, 0, 6, 3), T('c', 0, 7, 6, 2)])), { a: [0, 0, 6, 3], c: [0, 3, 6, 2] })
  // side by side stays side by side
  assert.deepEqual(at(arrange([T('a', 0, 0, 6, 3), T('b', 6, 0, 6, 3)])), { a: [0, 0, 6, 3], b: [6, 0, 6, 3] })
  // moving b below a and to the left swaps their rows
  assert.deepEqual(at(updateTile([T('a', 0, 0, 12, 2), T('b', 0, 2, 12, 2)], 'b', { y: 0 })), { b: [0, 0, 12, 2], a: [0, 2, 12, 2] })
  // a tile wider than the room left is pulled back inside the grid
  assert.deepEqual(at(arrange([T('a', 10, 0, 6, 2)])), { a: [6, 0, 6, 2] })
})

test('firstFit fills the first gap; add, remove and ids', () => {
  const tiles = [T('t1', 0, 0, 6, 4), T('t3', 6, 0, 3, 4)]
  assert.deepEqual(firstFit(tiles, 3, 2), { x: 9, y: 0 })
  assert.deepEqual(firstFit(tiles, 4, 2), { x: 0, y: 4 })
  assert.equal(nextTileId(tiles), 't2')
  const added = addTile(tiles, { kind: 'scatter', w: 3, h: 4, fields: [] })
  assert.deepEqual(at(added).t2, [9, 0, 3, 4])
  assert.deepEqual(at(removeTile(added, 't1')), { t3: [6, 0, 3, 4], t2: [9, 0, 3, 4] })
  assert.equal(rowsUsed([T('a', 0, 0, 12, 5), T('b', 0, 5, 4, 7)]), 12)
})
