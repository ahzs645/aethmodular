import { test } from 'node:test'
import assert from 'node:assert/strict'
import { hasSsr, keepUnderSsr, parseSsrHide, ssrCounts, ssrOnly } from './ssrFilters.ts'

const both = { 'Equivalent BC (SSR)': 1.2, 'HIPS BC': 1.1 }
const ssr = { 'BC (public SSR estimate)': 0.8 }
const hips = { 'HIPS BC': 2 }
const nothing = { 'Equivalent BC (SSR)': null }

test('SSR filters are told apart from HIPS ones, and nulls do not count', () => {
  assert.equal(hasSsr(both), true)
  assert.equal(ssrOnly(both), false)
  assert.equal(ssrOnly(ssr), true)
  assert.equal(hasSsr(nothing), false)
  assert.deepEqual(ssrCounts([both, ssr, hips, nothing]), { any: 2, only: 1 })
})

test('each choice hides what it says', () => {
  const rows = [both, ssr, hips, nothing]
  assert.deepEqual(rows.filter(keepUnderSsr('')), rows)
  assert.deepEqual(rows.filter(keepUnderSsr('only')), [both, hips, nothing])
  assert.deepEqual(rows.filter(keepUnderSsr('any')), [hips, nothing])
  assert.equal(parseSsrHide('only'), 'only')
  assert.equal(parseSsrHide('x'), '')
  assert.equal(parseSsrHide(null), '')
})
