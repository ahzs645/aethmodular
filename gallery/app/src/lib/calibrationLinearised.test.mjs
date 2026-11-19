// The "Which calibration, once HIPS is linearised" meeting page (meetings/calibration-linearised-20261119.meeting.ts)
// writes its findings into its titles and names table values in its defaults. The numbers are data (the page's
// bundle, from gallery/data/export_meeting_20261119_calibration_linearised.py), so check they still agree.
import assert from 'node:assert/strict'
import fs from 'node:fs'
import test from 'node:test'

const spec = fs.readFileSync(new URL('../meetings/calibration-linearised-20261119.meeting.ts', import.meta.url), 'utf8')
const bundle = JSON.parse(fs.readFileSync(new URL('../../public/data/meetings/calibration-linearised-20261119.json', import.meta.url), 'utf8'))
const rows = (name) => { const t = bundle.tables[name], keys = Object.keys(t); return t[keys[0]].map((_, i) => Object.fromEntries(keys.map((k) => [k, t[k][i]]))) }
const f2 = (v) => v.toFixed(2)
const line = (s, i) => `${f2(s)}x ${i >= 0 ? '+' : '−'} ${f2(Math.abs(i))}`
const says = (text) => assert.ok(spec.includes(text), `the page says "${text}"`)
const LINEARISED = 'Linearised, ceiling 2.26', ALL = 'All IMPROVE filters'

test('every table and text key the page names is in its bundle, and every default it names is a value there', () => {
  const tables = [...spec.matchAll(/table: '([\w]+)'/g)].map((m) => m[1])
  assert.ok(tables.length >= 10)
  for (const name of tables) assert.ok(bundle.tables[name], `table "${name}"`)
  for (const [, key] of spec.matchAll(/'@([\w-]+)'/g)) assert.ok(bundle.text[key], `text "${key}"`)
  const byK = bundle.tables.by_k
  assert.ok(byK.HIPS.includes(LINEARISED) && byK['Training set'].includes(ALL) && byK.Spectra.includes('Raw spectra'))
  says(`const LINEARISED = '${LINEARISED}'`); says(`default: '${ALL}'`)
  assert.ok(bundle.tables.drawn.Model.includes('Raw spectra, every filter, k 18'))
  // the two MAC panels are named in the exporter; the spec's MAC is the one in the second name
  const macs = [...new Set(bundle.tables.plane.MAC)]
  assert.equal(macs.length, 2); assert.equal(macs[0], 'MAC 10 (the convention)')
  const urban = /^MAC ([\d.]+) /.exec(macs[1])[1]
  says(`export const MAC_URBAN = ${urban}`)
})

test('the titles carry the sweep\'s numbers', () => {
  const central = rows('by_k').filter((r) => r.HIPS === LINEARISED && r.k >= 2)
  const top = Math.max(...central.map((r) => r.slope))
  assert.equal(central.length, 957); says('957 fits')
  says(`the largest slope at MAC 10 is ${f2(top)}`); says(`none passes ${f2(top)}`)
  assert.ok(top < 0.9, 'no fit reaches 0.9 at MAC 10')
  const at = (spectra, k) => central.find((r) => r.Spectra === spectra && r['Training set'] === ALL && r.k === k).slope
  says(`raw spectra rise from ${f2(at('Raw spectra', 4))} to ${f2(at('Raw spectra', 30))}, VIBES to ${f2(Math.max(...central.filter((r) => r.Spectra === 'VIBES baseline' && r['Training set'] === ALL).map((r) => r.slope)))}`)
  assert.ok(Math.max(...central.filter((r) => r.Spectra === 'Spline baseline' && r['Training set'] === ALL).map((r) => r.slope)) < 0.25)

  const b10 = rows('best_10'), bu = rows('best_urban')
  assert.equal(b10[0].Spectra, 'Raw spectra'); assert.equal(b10[0]['Training set'], 'Thermal EC 15 µg or more')
  says(`Closest at MAC 10: raw spectra trained on heavy filters only, ${line(b10[0].slope, b10[0].intercept)}`)
  const plain = rows('plain')[0]
  assert.ok(b10[0].error / plain.error > 4, 'more than four times the IMPROVE error of the plain raw model')
  says(`raw spectra reach ${line(bu[0].slope, bu[0].intercept)}, and seven of the first eight are raw spectra`)
  assert.equal(bu.slice(0, 8).filter((r) => r.Spectra === 'Raw spectra').length, 7)
  says(`raw spectra with every filter read ${line(plain.slope_10, plain.intercept)} at MAC 10 and ${line(plain.slope_urban, plain.intercept)} at the urban IMPROVE MAC`)

  const right = rows('cost').filter((r) => r.Spectra === 'Raw spectra' && r.improve_urban_level >= 0.95 && r.improve_urban_level <= 1.05).map((r) => r.slope).sort((a, b) => a - b)
  const median = (right[Math.floor((right.length - 1) / 2)] + right[Math.ceil((right.length - 1) / 2)]) / 2
  says(`raw-spectra fits within 5 % of thermal EC give slopes of ${f2(right[0])} to ${f2(right.at(-1))}, median ${f2(median)}`)
  const mac = rows('mac').filter((r) => r.mac > 10).map((r) => r.slope_if_ftir_right)
  says(`the slope at MAC 10 would be ${f2(Math.min(...mac))} to ${f2(Math.max(...mac))}`)
})

test('a slope at the urban MAC is the slope at MAC 10 scaled, with the same intercept', () => {
  const plane = rows('plane'), urban = Number(/^MAC ([\d.]+) /.exec(plane.at(-1).MAC)[1])
  for (const r of plane) assert.ok(Math.abs(r.slope_at_mac - r.slope * (r.MAC.startsWith('MAC 10 ') ? 1 : urban / 10)) < 2e-4)
})
