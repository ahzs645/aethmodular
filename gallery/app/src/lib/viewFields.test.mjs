import assert from 'node:assert/strict'
import { test } from 'node:test'
import { parseRatios, ratioKey, ratiosParam, ratioUnit, viewRows, viewUnits } from './viewFields.ts'

const units = { 'EC (FTIR)': 'ug/m3', 'PM2.5 mass': 'µg/m³', 'HIPS Fabs': 'Mm-1', 'HIPS BC': 'ug/m3', 'BC (public HIPS)': 'µg/m³', 'Sampled volume': 'm³' }
const base = { mac: 10, exportMac: 10, basis: 'conc', units, ratios: [] }
const row = { id: 'A', 'EC (FTIR)': 2, 'PM2.5 mass': 20, 'HIPS Fabs': 30, 'HIPS BC': 3, 'BC (public HIPS)': 3.1, 'Sampled volume': 7.2 }

test('ratios round-trip through the URL, names with commas and slashes included', () => {
  const defs = [{ num: 'EC (FTIR)', den: 'PM2.5 mass' }, { num: 'a,b:c~d', den: 'x / y' }]
  assert.deepEqual(parseRatios(ratiosParam(defs)), defs)
  assert.deepEqual(parseRatios(null), [])
  // a field over itself, repeats and half pairs are dropped
  assert.deepEqual(parseRatios('A:A,A:B,A:B,C:'), [{ num: 'A', den: 'B' }])
})

test('an unchanged view returns the same rows', () => {
  const rows = [row]
  assert.equal(viewRows(rows, base)[0], row)
})

test('MAC rescales HIPS BC and the public HIPS BC, nothing else', () => {
  const [r] = viewRows([row], { ...base, mac: 15 })
  assert.equal(r['HIPS BC'], 2)
  assert.ok(Math.abs(r['BC (public HIPS)'] - 3.1 * 10 / 15) < 1e-12)
  assert.equal(r['EC (FTIR)'], 2)
})

test('mass per filter multiplies concentrations by the volume; no volume, no value', () => {
  const [r, s] = viewRows([row, { ...row, id: 'B', 'Sampled volume': null }], { ...base, basis: 'mass' })
  assert.ok(Math.abs(r['EC (FTIR)'] - 14.4) < 1e-12)
  assert.equal(r['HIPS Fabs'], 30)
  assert.equal(r['Sampled volume'], 7.2)
  assert.equal(s['EC (FTIR)'], null)
  const u = viewUnits({ ...base, basis: 'mass' })
  assert.equal(u['EC (FTIR)'], 'µg/filter')
  assert.equal(u['HIPS Fabs'], 'Mm-1')
})

test('ratios use the values after MAC and basis, and skip a zero or missing denominator', () => {
  const ratios = [{ num: 'EC (FTIR)', den: 'PM2.5 mass' }, { num: 'HIPS BC', den: 'EC (FTIR)' }, { num: 'EC (FTIR)', den: 'Missing' }]
  const [r] = viewRows([row], { ...base, mac: 20, ratios })
  assert.equal(r[ratioKey(ratios[0])], 0.1)
  assert.equal(r[ratioKey(ratios[1])], 0.75)
  assert.equal(r[ratioKey(ratios[2])], null)
  const [z] = viewRows([{ ...row, 'PM2.5 mass': 0 }], { ...base, ratios })
  assert.equal(z[ratioKey(ratios[0])], null)
  assert.equal(viewUnits({ ...base, ratios })[ratioKey(ratios[0])], 'ratio')
  assert.equal(ratioUnit('ug/m3', 'Mm-1'), 'µg/m³ per Mm⁻¹')
})
