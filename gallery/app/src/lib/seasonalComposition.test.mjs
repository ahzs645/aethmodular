import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { test } from 'node:test'
import { seasonalComposition } from './seasonalComposition.ts'

const meta = JSON.parse(readFileSync(new URL('../../public/data/meta.json', import.meta.url), 'utf8'))
const filters = JSON.parse(readFileSync(new URL('../../public/data/filters.json', import.meta.url), 'utf8'))

test('Addis seasonal chemistry uses the canonical local calendar and once-only carbon', () => {
  const result = seasonalComposition(filters.rows, meta)
  assert.deepEqual(result.stats.map(s => s.n), [55, 69, 65])
  assert.equal(result.rows.length, 189)
  assert.equal(result.species.filter(f => f.startsWith('EC')).length, 1)
  assert.equal(result.species.filter(f => f.startsWith('OC')).length, 1)
  assert.ok(!result.species.includes('HIPS BC'))
  assert.ok(!result.species.includes('OM (FTIR)'))
  const feb = result.rows.filter(r => r.month === 2)
  assert.ok(feb.length > 0)
  assert.ok(feb.every(r => r.season.startsWith('Belg')))
  for (const s of result.stats) {
    assert.ok(s.start && s.end)
    assert.ok(s.fields.find(f => f.field === 'EC (FTIR)').n > 0)
  }
})

test('switching to meeting bins moves February without changing measurements or source labels', () => {
  const before = JSON.stringify(filters.rows)
  const local = seasonalComposition(filters.rows, meta)
  const meeting = seasonalComposition(filters.rows, meta, 'dry_feb')
  assert.equal(meeting.rows.length, local.rows.length)
  assert.ok(meeting.rows.filter(r => r.month === 2).every(r => r.season.startsWith('Dry')))
  assert.equal(meeting.stats[0].n - local.stats[0].n, local.rows.filter(r => r.month === 2).length)
  assert.equal(JSON.stringify(filters.rows), before)
})

test('exclusions, missing species, zero and negative concentrations keep distinct meanings', () => {
  const row = filters.rows.find(r => r.site === 'Addis Ababa')
  const rows = [
    { ...row, id: 'missing', 'EC (FTIR)': null, 'OC (FTIR)': null },
    { ...row, id: 'zero', 'EC (FTIR)': 0, 'OC (FTIR)': null },
    { ...row, id: 'negative', 'EC (FTIR)': -2, 'OC (FTIR)': null },
    { ...row, id: 'excluded', excluded: true, 'EC (FTIR)': 1000 },
    { ...row, id: 'other-site', site: 'Beijing', 'EC (FTIR)': 1000 },
  ]
  const result = seasonalComposition(rows, meta)
  assert.equal(result.rows.length, 3)
  const fields = result.stats.find(s => s.n > 0).fields
  assert.deepEqual(fields.find(f => f.field === 'EC (FTIR)'), { field: 'EC (FTIR)', n: 2, negative: 1, mean: -1 })
  assert.deepEqual(fields.find(f => f.field === 'OC (FTIR)'), { field: 'OC (FTIR)', n: 0, negative: 0, mean: null })
})
