// Samples on a page's own items, undated ones included (sampleView.ts)
import assert from 'node:assert/strict'
import { test } from 'node:test'
import { placedIn } from './sampleView.ts'

const items = [
  { id: 'a', code: 'ETAD', date: '2023-01-01' },
  { id: 'b', code: 'ETAD', date: '2024-01-01' },
  { id: 'c', code: 'ETAD', date: null },
  { id: 'd', code: 'ETBI', date: '2025-11-01' },
  { id: 'e', code: 'ETBI', date: '' },
]
const ids = (xs) => xs.map((x) => x.id).join('')
const at = (t) => t

test('placedIn: a Samples set that narrows nothing keeps the undated items too', () => {
  assert.equal(ids(placedIn(items, at, () => true)), 'abcde')
})

test('placedIn: narrowing a site by date takes its undated items out, and no other site\'s', () => {
  assert.equal(ids(placedIn(items, at, (code, date) => code !== 'ETAD' || date < '2024')), 'ade')
})

test('placedIn: a site left out of Samples takes its undated items with it', () => {
  assert.equal(ids(placedIn(items, at, (code) => code === 'ETBI')), 'de')
  assert.equal(ids(placedIn(items, at, () => false)), '')
})
