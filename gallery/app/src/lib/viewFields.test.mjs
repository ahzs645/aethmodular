import assert from 'node:assert/strict'
import { test } from 'node:test'
import { HIPS_TAU, HIPS_UNDER_READ, hipsFactor, hipsTau, isPerFilter, parseRatios, ratioKey, ratiosParam, ratioUnit, viewRows, viewUnits, withHipsLoading } from './viewFields.ts'
import { baseUnit } from './theme.ts'
import { sameUnits } from './axes.ts'

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

test('HIPS linearised: each filter by its own optical depth, the same factor on Fabs, BC and the public BC', () => {
  const ceiling = 2.2596
  const [r] = viewRows([row], { ...base, hipsCeiling: ceiling })
  // tau = 30 Mm-1 x 7.2 m3 / 353 = 0.6119; factor = 1 / (1 - tau / ceiling)
  const k = 1 / (1 - (30 * 7.2 / 353) / ceiling)
  assert.ok(Math.abs(hipsFactor(row, ceiling) - k) < 1e-12)
  assert.ok(Math.abs(r['HIPS Fabs'] - 30 * k) < 1e-9)
  assert.ok(Math.abs(r['HIPS BC'] - 3 * k) < 1e-9)
  assert.ok(Math.abs(r['BC (public HIPS)'] - 3.1 * k) < 1e-9)
  assert.equal(r['EC (FTIR)'], 2)
  // half the volume gives half the optical depth and a smaller factor on the same Fabs
  const [half] = viewRows([{ ...row, 'Sampled volume': 3.6 }], { ...base, hipsCeiling: ceiling })
  assert.ok(half['HIPS Fabs'] < r['HIPS Fabs'] && half['HIPS Fabs'] > 30)
  // the MAC still applies, to the linearised absorption
  const [m] = viewRows([row], { ...base, hipsCeiling: ceiling, mac: 15 })
  assert.ok(Math.abs(m['HIPS BC'] - (30 * k) / 15) < 1e-9)
})

test('HIPS linearised: no sampled volume or past the ceiling drops the HIPS fields, never keeps the measured value', () => {
  const o = { ...base, hipsCeiling: 2.2596 }
  const [noVolume] = viewRows([{ ...row, 'Sampled volume': null }], o)
  assert.equal(noVolume['HIPS Fabs'], null)
  assert.equal(noVolume['HIPS BC'], null)
  assert.equal(noVolume['BC (public HIPS)'], null)
  assert.equal(noVolume['EC (FTIR)'], 2)
  // 120 Mm-1 x 7.2 / 353 = 2.45, past the ceiling
  const [dark] = viewRows([{ ...row, 'HIPS Fabs': 120 }], o)
  assert.equal(dark['HIPS Fabs'], null)
  // a filter with only the public BC is linearised from it (Fabs = BC x 10)
  const [pub] = viewRows([{ id: 'B', 'BC (public HIPS)': 3, 'Sampled volume': 7.2 }], o)
  assert.ok(Math.abs(pub['BC (public HIPS)'] - 3 / (1 - (30 * 7.2 / 353) / 2.2596)) < 1e-9)
  // as measured is untouched
  assert.equal(viewRows([row], { ...base, hipsCeiling: null })[0], row)
})

test('HIPS linearised: units say so, and still match their own unit for the 1:1 line and mass per filter', () => {
  const u = viewUnits({ ...base, hipsCeiling: 2.2596, ratios: [{ num: 'HIPS BC', den: 'EC (FTIR)' }, { num: 'EC (FTIR)', den: 'PM2.5 mass' }] })
  assert.equal(u['HIPS BC'], 'ug/m3, linearised')
  assert.equal(u['HIPS Fabs'], 'Mm-1, linearised')
  assert.equal(u['BC (public HIPS)'], 'µg/m³, linearised')
  assert.equal(u['EC (FTIR)'], 'ug/m3')
  assert.equal(u['HIPS BC / EC (FTIR)'], 'ratio, HIPS linearised')
  assert.equal(u['EC (FTIR) / PM2.5 mass'], 'ratio')
  assert.ok(sameUnits(u['HIPS BC'], u['EC (FTIR)']))
  assert.ok(!sameUnits(u['HIPS Fabs'], u['EC (FTIR)']))
  const mass = viewUnits({ ...base, basis: 'mass', hipsCeiling: 2.2596 })
  assert.equal(mass['HIPS BC'], 'µg/filter, linearised')
  assert.ok(isPerFilter(mass['HIPS BC']) && isPerFilter(mass['EC (FTIR)']) && !isPerFilter(mass['HIPS Fabs']))
})

test('HIPS linearised at some sites: the others keep HIPS as measured, and the units say which sites', () => {
  const ceiling = 2.2596, k = 1 / (1 - (30 * 7.2 / 353) / ceiling)
  const dark = { ...row, id: 'D', code: 'DARK' }, light = { ...row, id: 'L', code: 'LITE' }
  const o = { ...base, hipsCeiling: ceiling, hipsScope: 'seen', hipsSites: new Set(['DARK']) }
  const [d, l] = viewRows([dark, light], o)
  assert.ok(Math.abs(d['HIPS BC'] - 3 * k) < 1e-9)
  // outside the scope: the same object, nothing dropped, even without a sampled volume
  assert.equal(l, light)
  const [noVolume] = viewRows([{ ...light, 'Sampled volume': null }], o)
  assert.equal(noVolume['HIPS BC'], 3)
  // the MAC still applies on both sides
  const [dm, lm] = viewRows([dark, light], { ...o, mac: 15 })
  assert.ok(Math.abs(dm['HIPS BC'] - (30 * k) / 15) < 1e-9)
  assert.equal(lm['HIPS BC'], 2)
  // no sites named: every site, as before
  assert.ok(Math.abs(viewRows([light], { ...o, hipsSites: null })[0]['HIPS BC'] - 3 * k) < 1e-9)
  const u = viewUnits({ ...o, ratios: [{ num: 'HIPS BC', den: 'EC (FTIR)' }] })
  assert.equal(u['HIPS BC'], 'ug/m3, linearised where seen')
  assert.equal(u['HIPS BC / EC (FTIR)'], 'ratio, HIPS linearised where seen')
  assert.equal(baseUnit(u['HIPS BC']), 'ug/m3')
  assert.ok(sameUnits(u['HIPS BC'], u['EC (FTIR)']))
  assert.equal(viewUnits({ ...o, hipsScope: 'expected' })['HIPS Fabs'], 'Mm-1, linearised where expected')
  assert.equal(viewUnits({ ...o, hipsScope: 'all' })['HIPS Fabs'], 'Mm-1, linearised')
})

test('HIPS loading: the optical depth and expected under-read of each filter with HIPS and a sampled volume', () => {
  const tau = 30 * 7.2 / 353
  assert.ok(Math.abs(hipsTau(row) - tau) < 1e-12)
  const [r, noVolume, pub, negative] = withHipsLoading([row, { ...row, 'Sampled volume': null }, { id: 'B', 'BC (public HIPS)': 3, 'Sampled volume': 7.2 }, { ...row, 'HIPS Fabs': -1 }])
  assert.ok(Math.abs(r[HIPS_TAU] - tau) < 1e-12)
  assert.ok(Math.abs(r[HIPS_UNDER_READ] - 100 * tau / 2.2596) < 1e-9)
  assert.equal(r['HIPS Fabs'], 30)
  assert.equal(HIPS_TAU in noVolume, false)
  assert.ok(Math.abs(pub[HIPS_TAU] - tau) < 1e-12)
  assert.equal(HIPS_TAU in negative, false)
  // the fields are read off the measured Fabs: the linearised view leaves them alone
  const [lin] = viewRows([r], { ...base, hipsCeiling: 2.2596 })
  assert.equal(lin[HIPS_TAU], r[HIPS_TAU])
  assert.equal(lin[HIPS_UNDER_READ], r[HIPS_UNDER_READ])
})

test('HIPS linearised past a τ: only filters at or past the cut, every site; a filter with no τ drops out', () => {
  // row: 30 Mm-1 at 7.2 m3, τ 0.612; a light twin at 10 Mm-1, τ 0.204
  const o = { ...base, hipsCeiling: 2.2596, hipsScope: 'past', hipsCut: 0.45 }
  const [dark, light, noVolume] = viewRows([row, { ...row, id: 'L', 'HIPS Fabs': 10, 'HIPS BC': 1, 'BC (public HIPS)': 1 }, { ...row, id: 'N', 'Sampled volume': null }], o)
  assert.ok(dark['HIPS Fabs'] > 30)
  assert.equal(light['HIPS Fabs'], 10)
  assert.equal(light['HIPS BC'], 1)
  assert.equal(noVolume['HIPS Fabs'], null)
  // a higher cut leaves the darker filter as measured too
  assert.equal(viewRows([row], { ...o, hipsCut: 0.7 })[0]['HIPS Fabs'], 30)
  const u = viewUnits(o)
  assert.equal(u['HIPS Fabs'], 'Mm-1, linearised past τ 0.45')
  assert.equal(baseUnit(u['HIPS Fabs']), 'Mm-1')
})
