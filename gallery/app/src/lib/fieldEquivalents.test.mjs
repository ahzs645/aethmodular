import test from 'node:test'
import assert from 'node:assert/strict'
import { EQUIVALENTS, resolveField, whyMissing } from './fieldEquivalents.ts'

const having = (...fields) => (f) => fields.includes(f)

test('a measured field stays itself', () => {
  assert.deepEqual(resolveField('HIPS BC', having('HIPS BC', 'Equivalent BC (SSR)')), { field: 'HIPS BC' })
})

test('HIPS BC moves to the same instrument first, then the reflectometer', () => {
  assert.equal(resolveField('HIPS BC', having('BC (public HIPS)', 'Equivalent BC (SSR)')).field, 'BC (public HIPS)')
  // ARCB: the HIPS–SSR estimate covers 35 of 43 filters, the reflectometer all 43; the calibrated estimate wins
  const arcb = resolveField('HIPS BC', having('BC (public SSR estimate)', 'Equivalent BC (SSR)'))
  assert.equal(arcb.field, 'BC (public SSR estimate)')
  assert.equal(arcb.equivalent.for, 'HIPS BC')
  assert.match(arcb.equivalent.basis, /HIPS–SSR curve/)
  assert.equal(resolveField('HIPS BC', having('Equivalent BC (SSR)')).field, 'Equivalent BC (SSR)')
})

test('FTIR carbon moves to the public FTIR product, another baseline, then IMPROVE TOR', () => {
  assert.equal(resolveField('EC (FTIR)', having('EC (ChemSpec FTIR)', 'EC (TOR)')).field, 'EC (ChemSpec FTIR)')
  assert.equal(resolveField('EC (FTIR)', having('EC (FTIR, VIBES)')).field, 'EC (FTIR, VIBES)')
  assert.equal(resolveField('OC (FTIR)', having('OC (TOR)')).field, 'OC (TOR)')
})

test('no documented stand-in keeps the field, so the tile can say why it is empty', () => {
  // EC is not black carbon: an SSR site has no EC equivalent
  assert.deepEqual(resolveField('EC (FTIR)', having('Equivalent BC (SSR)', 'PM2.5 mass')), { field: 'EC (FTIR)' })
  assert.deepEqual(resolveField('Silicon', having('Aluminum')), { field: 'Silicon' })
  for (const list of Object.values(EQUIVALENTS)) for (const e of list) assert.ok(e.source.url.startsWith('https://'))
})

test('the reason cites the method that explains the gap, and claims dates only from the record', () => {
  const arcb = { name: 'Buenos Aires', date_min: '2014-10-02', date_max: '2016-10-13' }
  const ssr = (f) => f === 'Equivalent BC (SSR)'
  const ec = whyMissing('EC (FTIR)', arcb, 'spartan', ssr)
  assert.match(ec.text, /Buenos Aires \(2014–2016\)/)
  assert.match(ec.text, /stretched Teflon/)
  assert.match(ec.text, /optical measurement rather than EC/)
  assert.match(ec.source.label, /Parameters & Methods/)
  assert.match(whyMissing('HIPS BC', arcb, 'spartan', ssr).text, /reflectometer/)
  assert.doesNotMatch(whyMissing('HIPS BC', arcb).text, /reflectometer/)
  assert.match(whyMissing('Silicon', arcb).text, /all predate that/)
  // a 2025–26 site: the XRF date is no explanation, and nothing is said to predate anything
  const ussl = { name: 'St Louis', date_min: '2025-01-10', date_max: '2026-03-01' }
  assert.doesNotMatch(whyMissing('Silicon', ussl).text, /predate/)
  assert.doesNotMatch(whyMissing('EC (FTIR)', ussl).text, /predate/)
  assert.match(whyMissing('Silicon', arcb).source.label, /XRF SOP/)
  assert.match(whyMissing('EC (FTIR)', { name: 'Mammoth Cave' }, 'improve').text, /thermal-optical/)
  assert.equal(whyMissing('Nitrate', arcb).source, undefined)
})
