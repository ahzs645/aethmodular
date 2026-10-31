import test from 'node:test'
import assert from 'node:assert/strict'
import { parseCohort, selectCohort, rowsCsv } from './dashboardCohort.ts'
import { measuredRows, normalizeSiteIdentity, metricDefinition } from './metricDefinition.ts'
import { readSnapshot, snapshotDigest } from './dashboardSnapshot.ts'
const row = (id, site, date, a, b, extra = {}) => ({ id, site, date, a, b, season: 'Dry', excluded: false, exclusion_reason: null, ...extra })
const rows = [row('a', 'Addis', '2023-01-01', 0, 2), row('b', 'Beijing', '2023-02-01', null, 3), row('c', 'Addis', '2023-03-01', 4, null)]
test('tile filters narrow the page; retained IDs use source membership, including explicit empty membership', () => {
  assert.deepEqual(selectCohort(rows, [], { site: 'Addis', from: '2023-02-01' }).map(r => r.id), ['c'])
  assert.deepEqual(selectCohort([], rows, { ids: ['a', 'missing'] }).map(r => r.id), ['a'])
  assert.deepEqual(selectCohort(rows, rows, { ids: [] }), [])
  assert.equal(selectCohort(rows, rows, { from: '2024-01-01', to: '2023-01-01' }).length, 0)
})
test('cohort URL input rejects malformed structures and keeps deduplicated reference IDs', () => {
  for (const value of ['null', '[1]', '{bad', '3']) assert.deepEqual(parseCohort(value), {})
  assert.deepEqual(parseCohort('{"ids":["a","a"],"from":"invalid","site":12}'), { ids: ['a'] })
})
test('contributing rows require finite pairs, preserve zero, and count physical IDs once', () => {
  assert.deepEqual(measuredRows([...rows, rows[0], row('d', 'Addis', '2023-02-01', Infinity, 2)], ['a', 'b']).map(r => r.id), ['a'])
  assert.equal(measuredRows([row('e', 'Addis', '', 1, 2)], ['a'], true).length, 0)
})
test('CSV keeps absent values blank, escapes delimiters and neutralizes spreadsheet formulas', () => {
  const csv = rowsCsv([row('=1+2', 'A,"B', '2023-01-01', -2, null, { exclusion_reason: 'line\nbreak' })], ['a', 'b'])
  assert.match(csv, /"'=1\+2"/); assert.match(csv, /"A,""B"/); assert.match(csv, /"-2",""/); assert.match(csv, /"line\nbreak"/)
})
test('normalization preserves unchanged branches and does not mutate the source', () => {
  const unchanged = { a: [1, 2] }; const input = { unchanged, sites: [{ name: 'Pasadena' }], bySite: { Pasadena: 5 } }
  const output = normalizeSiteIdentity(input)
  assert.equal(output.unchanged, unchanged); assert.equal(output.sites[0].name, 'JPL'); assert.equal(output.bySite.JPL, 5)
  assert.equal(input.sites[0].name, 'Pasadena'); assert.equal(normalizeSiteIdentity(unchanged), unchanged)
})
test('metric identity distinguishes FTIR, optical and TOR methods', () => {
  const meta = { field_units: { 'EC (FTIR)': 'ug/m3' } }
  assert.match(metricDefinition('EC (FTIR)', meta).caveat, /not independent/)
  assert.match(metricDefinition('HIPS BC', meta).method, /optical/)
  assert.match(metricDefinition('EC (TOR)', meta).method, /TOR/)
})
test('snapshot versions and checksums are validated before any view can be restored', async () => {
  const snapshot = { schema: 'aeth-dashboard', version: 1, created: '2026-10-30', revision: 'test', view: { h: { tab: 'dashboard' }, q: {} },
    dash: { scope: 'sites' }, sheet: { name: 'Test', id: 's1', tiles: [] }, provenance: {}, tiles: [], externalTiles: [] }
  const envelope = { snapshot, sha256: await snapshotDigest(snapshot) }
  assert.deepEqual(await readSnapshot(JSON.stringify(envelope)), snapshot)
  await assert.rejects(readSnapshot(JSON.stringify({ ...envelope, sha256: 'wrong' })), /checksum/)
  await assert.rejects(readSnapshot(JSON.stringify({ snapshot: { ...snapshot, version: 2 } })), /version/)
  const invalidHeader = { ...snapshot, revision: 42 }
  await assert.rejects(readSnapshot(JSON.stringify({ snapshot: invalidHeader, sha256: await snapshotDigest(invalidHeader) })), /definition/)
  const bad = { ...snapshot, tiles: [{ id: 't1', kind: 'stat', fields: [], rows: [{ id: 'bad', site: {}, date: '2023-01-01' }], meta: { fields: [], field_units: {}, sites: [], seasons: [] } }] }
  await assert.rejects(readSnapshot(JSON.stringify({ snapshot: bad, sha256: await snapshotDigest(bad) })), /filter record/)
})
