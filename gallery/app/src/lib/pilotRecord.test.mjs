import assert from 'node:assert/strict'
import { existsSync, readFileSync } from 'node:fs'
import { test } from 'node:test'
import { pilotRecord } from './pilotRecord.ts'
import { placements } from './referencePopulation.ts'

const data = (path) => new URL(`../../public/data/${path}`, import.meta.url)
const skip = !existsSync(data('pilot_filters.json')) || !existsSync(data('pilot_spectra.json'))
const read = (path) => JSON.parse(readFileSync(data(path), 'utf8'))

test('a pilot record is found by its own id or by the SPARTAN base id, and nothing else matches', () => {
  const d = { filters: [{ id: 'J1233' }, { id: 'ETBI-0002-2' }] }
  assert.equal(pilotRecord(d, 'J1233')?.id, 'J1233')
  assert.equal(pilotRecord(d, 'ETBI-0002')?.id, 'ETBI-0002-2')
  assert.equal(pilotRecord(d, 'ETBI-0002-2')?.id, 'ETBI-0002-2')
  assert.equal(pilotRecord(d, 'J1703'), null)
  assert.equal(pilotRecord(null, 'J1233'), null)
})

test('the export holds the 5 Adama and 26 Bishoftu filters, each with one corrected spectrum on both grids', { skip }, () => {
  const p = read('pilot_filters.json'), s = read('pilot_spectra.json')
  assert.equal(p.filters.filter((f) => f.campaign === 'Adama').length, 5)
  assert.equal(p.filters.filter((f) => f.campaign === 'Bishoftu').length, 26)
  assert.equal(new Set(p.filters.map((f) => f.id)).size, 31)
  assert.deepEqual(p.filters.map((f) => f.spectrum).sort(), [...s.ids].sort())
  for (const m of ['AIRSpec', 'VIBES']) for (const y of s[m].y) assert.equal(y.length, s[m].bin_wn.length)
  // every field a record carries has a unit and a group, so the drawer can lay it out
  const grouped = new Set(p.groups.flatMap((g) => g.fields))
  for (const f of p.filters) for (const k of Object.keys(f.values)) { assert.ok(k in p.units, k); assert.ok(grouped.has(k), k) }
})

test('J1233 carries the loading plane\'s TOR values, its quartz partner and the provisional spectrum map', { skip }, () => {
  const p = read('pilot_filters.json')
  const r = p.filters.find((f) => f.id === 'J1233')
  assert.equal(r.partner.id, 'J1703')
  assert.equal(r.spectrum, 'adama:4744')
  assert.equal(r.spectrum_map, 'provisional')
  // the TOR study's loading plane: EC 1.862 µg/m³ over 2.583 m³ of quartz, OC/EC 7.229
  assert.ok(Math.abs(r.values['EC (TOR)'] - 1.862) < 1e-3)
  assert.ok(Math.abs(r.values['Quartz sampled volume'] - 2.583) < 1e-3)
  assert.ok(Math.abs(r.values['OC/EC (TOR)'] - 7.2295) < 1e-3)
  // the thermal fractions add up to the TOR totals
  const oc = ['OC1', 'OC2', 'OC3', 'OC4', 'OP'].reduce((a, k) => a + r.values[k], 0)
  assert.ok(Math.abs(oc - r.values['OC (TOR)']) < 1e-2, `${oc} vs ${r.values['OC (TOR)']}`)
})

test('a pilot filter is placed against the 253 Addis filters on the measurements they share', { skip }, () => {
  const p = read('pilot_filters.json')
  assert.equal(p.reference.rows.length, 253)
  const r = p.filters.find((f) => f.id === 'J1233')
  const placed = placements({ id: r.id, site: r.site, code: r.code, date: r.date, excluded: false, ...r.values }, p.reference.fields, p.reference.rows.filter((x) => !x.excluded))
  assert.deepEqual(Object.keys(placed).sort(), ['FTIR EC, VIBES baseline', 'FTIR EC, spline baseline'])
  for (const v of Object.values(placed)) assert.ok(v.pct >= 0 && v.pct <= 100 && v.n > 200)
  const b = p.filters.find((f) => f.id === 'ETBI-0003-3')
  const placedB = placements({ id: b.id, site: b.site, code: b.code, date: b.date, excluded: false, ...b.values }, p.reference.fields, p.reference.rows)
  assert.ok('HIPS Fabs' in placedB && 'Band 2920, VIBES baseline' in placedB)
})
