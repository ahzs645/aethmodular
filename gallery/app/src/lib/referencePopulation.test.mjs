import assert from 'node:assert/strict'
import { test } from 'node:test'
import fs from 'node:fs'
import { cleanReference, placements, stripPosition, ratioReference } from './referencePopulation.ts'

const rows = [1, 2, 3, 4].map((v, i) => ({ id: `A-${i}`, site: 'A', code: 'AAAA', date: '2020-01-01', excluded: false, x: v, y: 2 * v }))
rows.push({ ...rows[0], id: 'bad', x: 1000, excluded: true })
rows.push(...[10, 20, 30, 40].map((v, i) => ({ id: `B-${i}`, site: 'B', code: 'BBBB', date: '2020-01-01', excluded: false, x: v, y: v })))
const selected = rows[1]
test('switching reference recomputes percentile and quartiles without changing the filter or the source rows', () => {
  const before = JSON.stringify(rows)
  const a = placements(selected, ['x'], cleanReference(rows, 'A')).x
  assert.deepEqual(a, { v: 2, pct: 25, min: 1, q1: 1.75, med: 2.5, q3: 3.25, max: 4, n: 4 })
  const b = placements(selected, ['x'], cleanReference(rows, 'B')).x
  assert.equal(b.pct, 0); assert.equal(b.med, 25); assert.equal(b.v, 2)
  const pooled = placements(selected, ['x'], cleanReference(rows)).x
  assert.equal(pooled.n, 8); assert.equal(pooled.med, 7); assert.equal(pooled.pct, 12.5)
  assert.equal(JSON.stringify(rows), before)
})
test('coverage is per field; missing, infinite, and absent sample measurements cannot fabricate a percentile', () => {
  const ref = cleanReference(rows, 'A').map((r, i) => ({ ...r, x: i < 2 ? null : r.x, absent: NaN, infinite: Infinity }))
  assert.deepEqual(placements(selected, ['x', 'absent', 'infinite'], ref), {})
  assert.deepEqual(placements({ ...selected, x: null }, ['x'], rows), {})
  assert.deepEqual(cleanReference(rows, 'nonexistent'), [])
})
test('out-of-range dots stay inside the strip, including constant reference ranges', () => {
  assert.equal(stripPosition(-5, { min: 1, max: 4 }), 0)
  assert.equal(stripPosition(10, { min: 1, max: 4 }), 100)
  assert.equal(stripPosition(2.5, { min: 1, max: 4 }), 50)
  assert.equal(stripPosition(1, { min: 2, max: 2 }), 0)
  assert.equal(stripPosition(3, { min: 2, max: 2 }), 100)
  assert.equal(stripPosition(2, { min: 2, max: 2 }), 50)
})
test('reference ratio medians use per-filter pairs, not ratios of marginal medians', () => {
  const ref = [{ x: 1, y: 1 }, { x: 10, y: 2 }, { x: 100, y: 100 }, { x: 3, y: 0 }, { x: null, y: 2 }]
  assert.deepEqual(ratioReference(ref, 'x', 'y'), { n: 3, median: 1 })
  assert.deepEqual(ratioReference(ref, 'z', 'y'), { n: 0, median: null })
})
test('public network export counts unique physical filters and does not substitute optical or functional-group fields', () => {
  const network = JSON.parse(fs.readFileSync(new URL('../../public/data/spartan_reference.json', import.meta.url)))
  assert.equal(network.schema_version, 1)
  assert.equal(network.n, network.rows.length)
  assert.equal(new Set(network.rows.map(r => `${r.code}|${r.id}`)).size, network.n)
  assert.equal(new Set(network.rows.map(r => r.code)).size, network.n_sites)
  assert.ok(network.n_sites > 4)
  assert.ok(!network.fields.includes('Alkane CH'))
  // HIPS BC is the in-house HIPS (Fabs / MAC 10), never the public BC put in its place
  for (const r of network.rows) {
    if (r['HIPS BC'] === null) continue
    assert.equal(typeof r['HIPS Fabs'], 'number', `${r.id}: HIPS BC without its Fabs`)
    assert.ok(Math.abs(r['HIPS BC'] - r['HIPS Fabs'] / 10) < 1e-5, `${r.id}: HIPS BC is not Fabs / 10`)
  }
  for (const r of network.rows) {
    assert.equal(r['EC (FTIR)'], r['EC (ChemSpec FTIR)'])
    assert.equal(r['OC (FTIR)'], r['OC (ChemSpec FTIR)'])
  }
  const clean = cleanReference(network.rows)
  assert.ok(clean.length > 0)
  assert.ok(clean.every(r => !r.excluded))
  const gallery = JSON.parse(fs.readFileSync(new URL('../../public/data/filters.json', import.meta.url)))
  // the network's HIPS is the same table as the focus sites' HIPS
  const byId = new Map(network.rows.map(r => [r.id, r]))
  for (const r of gallery.rows.filter(r => typeof r['HIPS Fabs'] === 'number')) {
    const n = byId.get(r.id)
    if (n && n['HIPS Fabs'] !== null) assert.ok(Math.abs(n['HIPS Fabs'] - r['HIPS Fabs']) < 1e-4, `${r.id}: HIPS Fabs differs from filters.json`)
  }
  for (const r of gallery.rows.filter(r => r.excluded)) {
    const publicRow = network.rows.find(p => p.id === r.id)
    if (publicRow) assert.equal(publicRow.excluded, true, `Known exclusion ${r.id} must carry into the public reference`)
  }
})
