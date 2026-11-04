import test from 'node:test'
import assert from 'node:assert/strict'
import { arrangeBlocks, flatKeys, mergeOrder, moved, orderBy, readLegendOrders } from './legendOrder.ts'

const row = (key) => ({ key, label: key, color: '#000', shape: 'dot', off: false })
const self = (s) => s

test('an order sorts a copy and leaves the array the chart draws from alone', () => {
  const items = ['a', 'b', 'c']
  assert.deepEqual(orderBy(items, ['c', 'a', 'b'], self), ['c', 'a', 'b'])
  assert.deepEqual(items, ['a', 'b', 'c'])
  assert.deepEqual(orderBy(items, undefined, self), items)
  assert.notEqual(orderBy(items, undefined, self), items)
})

test('items the order does not name follow in their own order; stale labels are ignored', () => {
  assert.deepEqual(orderBy(['a', 'b', 'c', 'd'], ['gone', 'c'], self), ['c', 'a', 'b', 'd'])
})

test('groups follow their first member, loose items stay first as the legend draws them', () => {
  const blocks = [{ name: null, rows: [row('x')] }, { name: 'A', rows: [row('a1'), row('a2')] }, { name: 'B', rows: [row('b1'), row('b2')] }]
  const out = arrangeBlocks(blocks, ['b2', 'b1', 'a2', 'a1', 'x'])
  assert.deepEqual(out.map((b) => b.name), [null, 'B', 'A'])
  assert.deepEqual(flatKeys(out), ['x', 'b2', 'b1', 'a2', 'a1'])
  assert.deepEqual(arrangeBlocks(blocks, undefined).map((b) => b.name), [null, 'A', 'B'])
})

test('a group no order names keeps its place behind the named ones', () => {
  const blocks = [{ name: 'A', rows: [row('a')] }, { name: 'B', rows: [row('b')] }, { name: 'C', rows: [row('c')] }]
  assert.deepEqual(arrangeBlocks(blocks, ['c']).map((b) => b.name), ['C', 'A', 'B'])
})

test('moved puts an item at the index it was dropped on, either way', () => {
  assert.deepEqual(moved(['a', 'b', 'c', 'd'], 0, 2), ['b', 'c', 'a', 'd'])
  assert.deepEqual(moved(['a', 'b', 'c', 'd'], 3, 1), ['a', 'd', 'b', 'c'])
})

test('reordering a smaller subset keeps the labels it hid', () => {
  assert.deepEqual(mergeOrder(['b', 'a'], ['a', 'c', 'b']), ['b', 'a', 'c'])
})

test('saved orders drop malformed entries', () => {
  const saved = readLegendOrders({ good: { order: ['a', 3, 'b'], page: true }, plain: { order: ['a'] }, empty: { order: [] }, bad: { order: 'a' }, none: null })
  assert.deepEqual([...saved], [['good', { order: ['a', 'b'], page: true }], ['plain', { order: ['a'], page: false }]])
  assert.equal(readLegendOrders(['a']).size, 0)
})
