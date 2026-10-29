import test from 'node:test'
import assert from 'node:assert/strict'
import { rowsOf, where, uniq, summarize, densities } from './table.ts'
import { nearestIndex, channelQuantiles, iqrRows } from './spectra.ts'
import { facetGrid, panelRow, panelLetter, fitColumns } from './facets.ts'

test('rowsOf accepts column and row tables', () => {
  const cols = { a: [1, 2], b: ['x', 'y'] }
  assert.deepEqual(rowsOf(cols), [{ a: 1, b: 'x' }, { a: 2, b: 'y' }])
  assert.deepEqual(rowsOf([{ a: 1 }]), [{ a: 1 }])
  assert.deepEqual(rowsOf(null), [])
})

test('where filters on values and value lists; uniq keeps first-seen order', () => {
  const rows = rowsOf({ g: ['b', 'a', 'b', 'c'], v: [1, 2, 3, 4] })
  assert.deepEqual(where(rows, { g: 'b' }).map((r) => r.v), [1, 3])
  assert.deepEqual(where(rows, { g: ['a', 'c'] }).map((r) => r.v), [2, 4])
  assert.deepEqual(uniq(rows, 'g'), ['b', 'a', 'c'])
})

test('summarize gives median and interval per group, skipping missing values', () => {
  const rows = rowsOf({ g: ['a', 'a', 'a', 'a', 'a', 'b'], v: [1, 2, 3, 4, null, 7] })
  const [a, b] = summarize(rows, ['g'], 'v')
  assert.equal(a.g, 'a'); assert.equal(a.mid, 2.5); assert.equal(a.lo, 1.75); assert.equal(a.hi, 3.25); assert.equal(a.n, 4)
  assert.equal(b.mid, 7); assert.equal(b.lo, 7)
  const [m] = summarize(rows, ['g'], 'v', { stat: 'mean', interval: 'minmax' })
  assert.equal(m.mid, 2.5); assert.equal(m.lo, 1); assert.equal(m.hi, 4)
})

test('densities share edges and each integrates to one', () => {
  const d = densities([{ key: 'a', values: [1, 10, 100] }, { key: 'b', values: [5, 5, 50, 500] }], { bins: 10, log: true })
  assert.equal(d.edges.length, 11)
  assert.ok(Math.abs(d.edges[0] - 1) < 1e-9 && Math.abs(d.edges[10] - 500) < 1e-6)
  const step = (Math.log10(500) - Math.log10(1)) / 10
  for (const s of d.series) assert.ok(Math.abs(s.density.reduce((a, c) => a + c, 0) * step - 1) < 1e-9)
  assert.equal(d.series[1].median, 27.5)
})

test('nearestIndex works on ascending and descending grids', () => {
  assert.equal(nearestIndex([500, 600, 700], 640), 1)
  assert.equal(nearestIndex([700, 600, 500], 640), 1)
  assert.equal(nearestIndex([700, 600, 500], 9999), 0)
  assert.equal(nearestIndex([700, 600, 500], -5), 2)
  assert.equal(nearestIndex([], 1), -1)
})

test('channelQuantiles and iqrRows', () => {
  const q = channelQuantiles([[1, 10], [2, 20], [3, NaN]], [0, 0.5, 1])
  assert.deepEqual(q, [[1, 10], [2, 15], [3, 20]])
  assert.deepEqual(iqrRows([[0], [1], [2], [3], [4]]), { lo: [1], mid: [2], hi: [3] })
  assert.deepEqual(iqrRows([[1], [2], [3]]), { lo: [1], mid: [2], hi: [3] })
  assert.throws(() => iqrRows([[1]]))
})

test('facetGrid fits columns to the width', () => {
  assert.deepEqual(facetGrid(1200, 8, { minCell: 260, maxCols: 4, gap: 16 }), { cols: 4, cell: 288, gap: 16 })
  assert.equal(facetGrid(600, 8).cols, 2)
  assert.equal(facetGrid(200, 3).cols, 1)
  assert.equal(facetGrid(2000, 2).cols, 2)
})

test('facetGrid caps a cell at maxCell', () => {
  assert.equal(facetGrid(1200, 2, { maxCell: 360 }).cell, 360)
})

test('panelRow puts panels side by side when their mins fit, else stacks them', () => {
  // a fixed square beside a growing sweep: the sweep takes the spare room
  assert.deepEqual(panelRow(1000, [{ min: 300, basis: 420, max: 420, grow: 0 }, { min: 260, basis: 360 }], 24), { stacked: false, widths: [420, 556] })
  // too narrow for both mins: stacked, each as wide as the container up to its max
  assert.deepEqual(panelRow(500, [{ min: 300, max: 420, grow: 0 }, { min: 260 }], 24), { stacked: true, widths: [420, 500] })
  // bases too wide: each shrinks toward its min in proportion to its slack
  const r = panelRow(700, [{ min: 300, basis: 500 }, { min: 300, basis: 500 }], 20)
  assert.equal(r.stacked, false)
  assert.deepEqual(r.widths, [340, 340])
  // spare room spills over once a panel reaches its max
  assert.deepEqual(panelRow(1000, [{ min: 200, basis: 200, max: 300 }, { min: 200, basis: 200 }], 0).widths, [300, 700])
  // one panel never counts as stacked
  assert.deepEqual(panelRow(400, [{ min: 240, max: 360 }]), { stacked: false, widths: [360] })
})

test('panelLetter runs a..z then aa', () => {
  assert.deepEqual([0, 1, 25, 26, 27].map(panelLetter), ['a', 'b', 'z', 'aa', 'ab'])
})

test('fitColumns puts boxes side by side when their text fits', () => {
  assert.equal(fitColumns(466, 190, 2), 2)   // two short readouts in a 466 px strip
  assert.equal(fitColumns(300, 190, 2), 1)   // too narrow: stacked
  assert.equal(fitColumns(1200, 190, 9), 4)  // many groups: rows of at most four
  assert.equal(fitColumns(1200, 190, 1), 1)
})
