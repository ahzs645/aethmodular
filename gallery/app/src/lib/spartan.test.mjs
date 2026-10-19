import test from 'node:test'
import assert from 'node:assert/strict'
import { gzipSync } from 'node:zlib'
import { loadJson, loadBinary } from './data.ts'
import { nativeTrace } from './spartan.ts'

test('gzip data works on plain static hosts and hosts that already decoded it; concurrent reads share a fetch', async () => {
  const original = globalThis.fetch
  let calls = 0
  globalThis.fetch = async url => {
    calls++
    if (String(url).includes('already-decoded')) return new Response('{"ok":true}')
    if (String(url).includes('float')) return new Response(gzipSync(new Uint8Array([0, 0, 128, 63])))
    return new Response(gzipSync('{"ok":true}'))
  }
  try {
    assert.deepEqual(await Promise.all([loadJson('test-gzip.json.gz'), loadJson('test-gzip.json.gz')]), [{ ok: true }, { ok: true }])
    assert.equal(calls, 1)
    assert.deepEqual(await loadJson('already-decoded.json.gz'), { ok: true })
    assert.equal(new DataView(await loadBinary('float.bin.gz')).getFloat32(0, true), 1)
  } finally { globalThis.fetch = original }
})

test('failed requests are retryable, including malformed compressed data', async () => {
  const original = globalThis.fetch
  let calls = 0
  globalThis.fetch = async () => ++calls === 1 ? new Response('absent', { status: 404 }) : new Response(gzipSync('{"retry":true}'))
  try {
    await assert.rejects(loadJson('retry.json.gz'), /404/)
    assert.deepEqual(await loadJson('retry.json.gz'), { retry: true })
  } finally { globalThis.fetch = original }
})

test('native spectra keep analysis offsets, descending grids, negative absorbance and gaps', () => {
  const values = new Float32Array([99, 98, .25, -.5, NaN])
  const a = { first: 4000, last: 500, n: 3, offset: 2 }
  const result = nativeTrace(values.buffer, a)
  assert.deepEqual(result.wn, [4000, 2250, 500])
  assert.equal(result.y[0], .25)
  assert.equal(result.y[1], -.5)
  assert.ok(Number.isNaN(result.y[2]))
  assert.throws(() => nativeTrace(values.buffer, { ...a, n: 4 }), /length/)
  assert.throws(() => nativeTrace(values.buffer, { ...a, offset: -1 }), /length/)
})

test('site selection fetches only selected filter shards, keeps the whole site index and never requests spectra', async () => {
  const { loadSpartanFilters } = await import('./spartan.ts')
  const original = globalThis.fetch
  const calls = []
  const index = { schema_version: 1, fields: [], field_units: {}, sites: [
    { code: 'AAAA', filters: 'sites/test-A.json.gz', native: 'native/A.json.gz' },
    { code: 'BBBB', filters: 'sites/test-B.json.gz', native: 'native/B.json.gz' },
  ] }
  globalThis.fetch = async url => {
    calls.push(String(url))
    if (String(url).endsWith('index.json')) return new Response(JSON.stringify(index))
    const code = String(url).includes('test-A') ? 'AAAA' : 'BBBB'
    return new Response(gzipSync(JSON.stringify({ rows: [{ id: `${code}-0001`, code, excluded: true, exclusion_reason: 'source QC' }] })))
  }
  try {
    const empty = await loadSpartanFilters([])
    assert.equal(empty.index.sites.length, 2)
    assert.equal(empty.rows.length, 0)
    assert.equal(calls.length, 1)
    const one = await loadSpartanFilters(['AAAA'])
    assert.equal(one.rows[0].code, 'AAAA')
    assert.equal(one.rows[0].excluded, true)
    assert.equal(calls.length, 2)
    await loadSpartanFilters(['AAAA'])
    assert.equal(calls.length, 2)
    const all = await loadSpartanFilters('all')
    assert.equal(all.rows.length, 2)
    assert.equal(calls.length, 3)
    assert.ok(calls.every(url => !url.includes('/native/') && !url.includes('/spectra/')))
  } finally { globalThis.fetch = original }
})
