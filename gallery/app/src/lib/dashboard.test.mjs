import test from 'node:test'
import assert from 'node:assert/strict'
import { addTile, arrange, placeNew, placeTile, stepTile, dashParams, firstFit, nextTileId, parseDash, parseSheets, parseTiles, removeTile, rowsUsed, sheetsParam, tilesParam, updateTile } from './dashboard.ts'

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

test('placeTile: onto a tile of the same size the two swap; otherwise the others make room', () => {
  const tiles = [T('a', 0, 0, 6, 4), T('b', 6, 0, 6, 4), T('c', 0, 4, 12, 3)]
  assert.deepEqual(at(placeTile(tiles, 'a', 6, 0)), { b: [0, 0, 6, 4], a: [6, 0, 6, 4], c: [0, 4, 12, 3] })
  // different sizes: no swap (it would overlap), the one in the way moves down
  const t2 = [T('a', 0, 0, 3, 2), T('b', 3, 0, 9, 2)]
  assert.deepEqual(at(placeTile(t2, 'a', 3, 0)), { a: [3, 0, 3, 2], b: [3, 2, 9, 2] })
  assert.deepEqual(at(placeTile(tiles, 'c', 0, 0)), { c: [0, 0, 12, 3], a: [0, 3, 6, 4], b: [6, 3, 6, 4] })
  assert.equal(placeTile(tiles, 'a', 0, 0), tiles)
})

test('stepTile: an arrow trades places with the neighbour on that side, else moves a cell', () => {
  const row = [T('a', 0, 0, 4, 4), T('b', 4, 0, 4, 4), T('c', 8, 0, 4, 4)]
  assert.deepEqual(at(stepTile(row, 'a', 'right')), { b: [0, 0, 4, 4], a: [4, 0, 4, 4], c: [8, 0, 4, 4] })
  assert.deepEqual(at(stepTile(row, 'c', 'left')), { a: [0, 0, 4, 4], c: [4, 0, 4, 4], b: [8, 0, 4, 4] })
  const col = [T('a', 0, 0, 12, 2), T('b', 0, 2, 12, 5)]
  assert.deepEqual(at(stepTile(col, 'a', 'down')), { b: [0, 0, 12, 5], a: [0, 5, 12, 2] })
  assert.deepEqual(at(stepTile(col, 'b', 'up')), { b: [0, 0, 12, 5], a: [0, 5, 12, 2] })
  // nothing to the right: one cell
  assert.deepEqual(at(stepTile([T('a', 0, 0, 4, 2)], 'a', 'right')), { a: [1, 0, 4, 2] })
})

test('placeNew: a free gap, else shrunk into the largest gap, else half of the largest tile; never below the window', () => {
  // room: the first gap
  assert.deepEqual(placeNew([T('a', 0, 0, 6, 12)], 4, 5), { tiles: [T('a', 0, 0, 6, 12)], x: 6, y: 0, w: 4, h: 5 })
  // only a 4 x 3 hole left: the 6 x 5 tile shrinks into it
  const holey = [T('a', 0, 0, 12, 9), T('b', 0, 9, 8, 3)]
  const p1 = placeNew(holey, 6, 5)
  assert.deepEqual([p1.x, p1.y, p1.w, p1.h], [8, 9, 4, 3])
  // full: the largest tile gives up half (across its longer side)
  const full = [T('a', 0, 0, 8, 12), T('b', 8, 0, 4, 12)]
  const p2 = placeNew(full, 4, 5)
  assert.deepEqual(at(p2.tiles), { a: [0, 0, 4, 12], b: [8, 0, 4, 12] })
  assert.deepEqual([p2.x, p2.y, p2.w, p2.h], [4, 0, 4, 12])
  // a tall narrow tile splits top and bottom
  const p3 = placeNew([T('a', 0, 0, 4, 12), T('b', 4, 0, 8, 12)].map((t) => ({ ...t })), 3, 3)
  assert.deepEqual(at(p3.tiles).b, [4, 0, 4, 12])
  // through addTile: a full sheet stays inside the window
  const added = addTile(full, { kind: 'scatter', w: 4, h: 5, fields: [] })
  assert.equal(rowsUsed(added), 12)
  assert.equal(added.length, 3)
})
