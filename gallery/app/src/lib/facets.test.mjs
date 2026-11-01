import assert from 'node:assert/strict'
import { test } from 'node:test'
import { extentOf, facetGrid, groupExtents, panelRow, panelWrap, pickPanels, unionExtent } from './facets.ts'
import { panelAxesKey } from './axes.ts'

test('facetGrid fits as many equal columns as the minimum cell allows', () => {
  assert.deepEqual(facetGrid(1000, 6, { minCell: 300, gap: 16 }), { cols: 3, cell: 322, gap: 16 })
  assert.equal(facetGrid(200, 3, { minCell: 300 }).cols, 1)
})

test('panelWrap keeps one row while every min fits, as panelRow does', () => {
  const p = [{ min: 300 }, { min: 300 }]
  assert.deepEqual(panelWrap(800, p, 18).widths, panelRow(800, p, 18).widths)
  assert.equal(panelWrap(800, p, 18).rows, 1)
})

test('panelWrap flows into equal columns like repeat(auto-fit, minmax(min, 1fr))', () => {
  const p = Array.from({ length: 4 }, () => ({ min: 340 }))
  const w = panelWrap(1100, p, 18)
  assert.equal(w.cols, 3)
  assert.equal(w.rows, 2)
  assert.equal(w.stacked, false)
  assert.deepEqual(w.widths, [354, 354, 354, 354])
  const narrow = panelWrap(500, p, 18)
  assert.equal(narrow.stacked, true)
  assert.equal(narrow.cols, 1)
  assert.equal(narrow.rows, 4)
})

test('pickPanels filters by show and orders by order, the rest after in their own order', () => {
  const keys = ['a', 'b', 'c', 'd']
  assert.deepEqual(pickPanels(keys), [0, 1, 2, 3])
  assert.deepEqual(pickPanels(keys, ['c', 'a', 'zz']), [0, 2])
  assert.deepEqual(pickPanels(keys, [], ['d', 'b']), [3, 1, 0, 2])
  assert.deepEqual(pickPanels(keys, ['a', 'b', 'd'], ['d']), [3, 0, 1])
})

test('unionExtent and extentOf ignore missing and non-finite values', () => {
  assert.deepEqual(unionExtent([[1, 3], null, [-2, 2], [NaN, 5]]), [-2, 3])
  assert.equal(unionExtent([null]), null)
  assert.deepEqual(extentOf([3, null, -1, Infinity, 2]), [-1, 3])
  assert.equal(extentOf([]), null)
})

test('groupExtents shares a range within each Panel axes group', () => {
  const panels = [{ key: 'A-dry', site: 'A', y: [0, 4] }, { key: 'A-wet', site: 'A', y: [1, 9] }, { key: 'B-dry', site: 'B', y: [2, 3] }]
  const regs = (mode) => panels.map((p) => ({ group: panelAxesKey(mode, p), axis: 'y', extent: p.y }))
  assert.deepEqual([...groupExtents(regs('Shared'))], [['|y', [0, 9]]])
  assert.deepEqual(groupExtents(regs('Per site')).get('A|y'), [0, 9])
  assert.deepEqual(groupExtents(regs('Per site')).get('B|y'), [2, 3])
  assert.deepEqual(groupExtents(regs('Own')).get('A-dry|y'), [0, 4])
})

test('panelWrap caps the columns at maxCols', () => {
  const p = Array.from({ length: 4 }, () => ({ min: 200 }))
  assert.equal(panelWrap(1200, p, 16).cols, 4)
  const two = panelWrap(1200, p, 16, 2)
  assert.equal(two.cols, 2)
  assert.equal(two.rows, 2)
  assert.deepEqual(two.widths, [592, 592, 592, 592])
})

test('panelWrap with one column stacks full-width panels', () => {
  const w = panelWrap(900, [{ min: 200 }, { min: 200, max: 500 }], 16, 1)
  assert.equal(w.stacked, true)
  assert.deepEqual(w.widths, [900, 500])
})
