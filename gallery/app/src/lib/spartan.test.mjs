import test from 'node:test'
import assert from 'node:assert/strict'
import { gzipSync } from 'node:zlib'
import { loadJson, loadBinary } from './data.ts'
import { decodeTrace, nativeTraces } from './spartan.ts'

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

test('an HTML fallback for missing JSON or spectra is reported as an unavailable export and remains retryable', async () => {
  const original = globalThis.fetch
  globalThis.fetch = async () => new Response('<!doctype html><html>Gallery</html>', { headers: { 'content-type': 'text/html; charset=utf-8' } })
  try {
    await assert.rejects(loadJson('missing-index.json'), /data export is not available/)
    await assert.rejects(loadJson('missing-native.json.gz'), /data export is not available/)
    await assert.rejects(loadBinary('missing-spectrum.bin.gz'), /data export is not available/)
    globalThis.fetch = async () => new Response('{"restored":true}', { headers: { 'content-type': 'application/json' } })
    assert.deepEqual(await loadJson('missing-index.json'), { restored: true })
  } finally { globalThis.fetch = original }
})

test('native spectra keep analysis offsets, descending grids, negative absorbance and gaps', () => {
  const values = new Float32Array([99, 98, .25, -.5, NaN])
  const a = { first: 4000, last: 500, n: 3, offset: 2 }
  const result = nativeTraces(values.buffer, a).raw
  assert.deepEqual(result.wn, [4000, 2250, 500])
  assert.equal(result.y[0], .25)
  assert.equal(result.y[1], -.5)
  assert.ok(Number.isNaN(result.y[2]))
  assert.throws(() => nativeTraces(values.buffer, { ...a, n: 4 }), /length/)
  assert.throws(() => nativeTraces(values.buffer, { ...a, offset: -1 }), /length/)
})

// the layout gallery/data/spectrum_codec.py writes: integer steps, differences, four byte planes
const planes = (diffs) => {
  const n = diffs.length, out = new Uint8Array(4 * n), view = new DataView(new ArrayBuffer(4))
  diffs.forEach((d, i) => { view.setInt32(0, d, true); for (let k = 0; k < 4; k++) out[k * n + i] = view.getUint8(k) })
  return out
}

test('encoded traces decode to steps, keep negatives and gaps, and check their length', () => {
  // 0.25, -0.5, then a gap stored as the previous value
  const body = planes([25_000_000, -75_000_000, 0])
  const buf = new Uint8Array(8 + body.length); buf.set(body, 8)
  const y = decodeTrace(buf.buffer, { offset: 8, n: 3, step: 1e-8, nan: [[2, 1]] })
  assert.ok(Math.abs(y[0] - .25) < 1e-12 && Math.abs(y[1] + .5) < 1e-12)
  assert.ok(Number.isNaN(y[2]))
  assert.throws(() => decodeTrace(buf.buffer, { offset: 8, n: 4, step: 1e-8 }), /length/)
})

test('schema 2 analyses give raw, AIRSpec (raw minus its baseline over its span) and VIBES on its own grid', () => {
  const raw = planes([30, 10, 10, 10]), base = planes([10, 0]), vib = planes([5, 5, 5])
  const buf = new Uint8Array(raw.length + base.length + vib.length)
  buf.set(raw, 0); buf.set(base, raw.length); buf.set(vib, raw.length + base.length)
  const a = { first: 4000, last: 1000, n: 4, invalid_points: 0, series: {
    raw: { offset: 0, n: 4, step: 1 }, airspec_baseline: { offset: raw.length, n: 2, step: 1, start: 1 },
    vibes_corrected: { offset: raw.length + base.length, n: 3, step: 1 } } }
  const t = nativeTraces(buf.buffer, a, { first: 3000, last: 1000, n: 3 })
  assert.deepEqual(Array.from(t.raw.y), [30, 40, 50, 60])
  assert.deepEqual(t.airspec.wn, [3000, 2000])
  assert.deepEqual(Array.from(t.airspec.y), [30, 40])
  assert.deepEqual(t.vibes.wn, [3000, 2000, 1000])
  assert.deepEqual(Array.from(t.vibes.y), [5, 10, 15])
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

test('VIBES provenance names the background lots, whether the filter lot is among them, and the held-out check', async () => {
  const { vibesProvenance } = await import('./spartan.ts')
  const grid = { first: 4000, last: 500, n: 3, model: 'm', model_rank: 70, model_blanks: 87, resampling: '',
    models: {
      b87: { label: '87-blank background', rank: 70, n_blanks: 87, lots: ['248', '251', '253'], blanks: '59 IMPROVE + 28 Addis', source: 's1' },
      lot241_r70: { label: '87 + lot 241/241a blanks', rank: 70, n_blanks: 96, lots: ['241', '241a', '248', '251', '253'], blanks: 'b87 + 9', note: 'LOBO for blanks.', source: 's2' },
    },
    lot_check: {
      '250': { heldout_blanks: 60, model: 'b87', ptfe_rms_b87: 0.000089, ptfe_rms: 0.000089, rms_b87: 0.000034, rms: 0.000034 },
      '241': { heldout_blanks: 4, model: 'lot241_r70', ptfe_rms_b87: 0.00116, ptfe_rms: 0.000047, rms_b87: 0.00033, rms: 0.000018 },
    } }
  assert.equal(vibesProvenance({ lot: '251' }, grid), null)                       // no VIBES trace
  assert.match(vibesProvenance({ lot: '251', vibes_model: 'b87' }, grid).line, /lots 248, 251, 253.*lot 251 is among them/)
  const out = vibesProvenance({ lot: '250', vibes_model: 'b87' }, grid)
  assert.match(out.line, /lot 250 is not among them; held-out lot-250 field blanks still correct as cleanly \(median PTFE-band residue 0\.00009\)/)
  const lot = vibesProvenance({ lot: '241', vibes_model: 'lot241_r70' }, grid)
  assert.match(lot.line, /87 \+ lot 241\/241a blanks .*lot 241, which the 87-blank background has no blank of/)
  assert.match(lot.detail, /with the 87-blank background they kept 0\.0003 and 0\.001/)
  assert.match(vibesProvenance({ lot: null, vibes_model: 'b87' }, grid).line, /no lot on record/)
  // an export from before the lot test: the old one-line description, no claim about lots
  assert.equal(vibesProvenance({ vibes_model: 'b87' }, { ...grid, models: undefined }).line, 'VIBES background: 87 blanks (IMPROVE and Addis), rank 70.')
})
