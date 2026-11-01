import assert from 'node:assert/strict'
import { test } from 'node:test'
import { codesOf, errorText, keyOf, loadDerived, settle } from './data.ts'

test('settle shows only the current key, or the last data under keep', () => {
  const last = { key: 'a', data: 1, error: '' }
  assert.deepEqual(settle(last, 'a'), { data: 1, error: '', loading: false, key: 'a' })
  assert.deepEqual(settle(last, 'b'), { data: null, error: '', loading: true, key: null })
  assert.deepEqual(settle(last, 'b', true), { data: 1, error: '', loading: true, key: 'a' })
  assert.deepEqual(settle(null, 'a', true), { data: null, error: '', loading: true, key: null })
  assert.deepEqual(settle(last, null, true), { data: null, error: '', loading: false, key: null })
})

test('settle reports the current key\'s error and never an old one', () => {
  const failed = { key: 'a', data: null, error: 'a.json: HTTP 404' }
  assert.deepEqual(settle(failed, 'a'), { data: null, error: 'a.json: HTTP 404', loading: false, key: 'a' })
  assert.deepEqual(settle(failed, 'b', true), { data: null, error: '', loading: true, key: null })
})

test('keyOf ignores order and repeats; codesOf inverts it', () => {
  assert.equal(keyOf(['ETAD', 'AEAZ', 'ETAD']), 'AEAZ,ETAD')
  assert.deepEqual(codesOf(keyOf(['B', 'A'])), ['A', 'B'])
  assert.deepEqual(codesOf(keyOf([])), [])
})

test('errorText prints an Error by its message', () => {
  assert.equal(errorText(new Error('x.json: HTTP 500')), 'x.json: HTTP 500')
  assert.equal(errorText('plain'), 'plain')
})

test('loadDerived builds once per key and forgets a failure', async () => {
  let built = 0
  const build = async () => ++built
  const [a, b] = await Promise.all([loadDerived('test:once', build), loadDerived('test:once', build)])
  assert.equal(a, 1); assert.equal(b, 1); assert.equal(built, 1)
  let tries = 0
  const flaky = async () => { if (++tries === 1) throw new Error('first try fails'); return 'ok' }
  await assert.rejects(loadDerived('test:retry', flaky), /first try fails/)
  await new Promise((r) => setTimeout(r, 0))
  assert.equal(await loadDerived('test:retry', flaky), 'ok')
})
