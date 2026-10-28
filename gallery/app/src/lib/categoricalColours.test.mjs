import test from 'node:test'
import assert from 'node:assert/strict'
import { categoricalScale, categoricalStops, CATEGORICAL_COLOUR_SETS, hasNamedColours, SPARTAN_COMPOSITION_COLOURS } from './categoricalColours.ts'

test('SPARTAN component colours stay attached to their names across charts, order and subsets', () => {
  const bars = categoricalScale('SPARTAN composition', ['Sulfate', 'OC (ChemSpec FTIR)', 'EC (FTIR)', 'Nitrate', 'Ammonium', 'Potassium', 'Sodium'])
  const map = categoricalScale('SPARTAN composition', ['bc', 'no3', 'oc', 'so4', 'nh4', 'k', 'na'])
  for (const [field, key, hex] of [
    ['Sulfate', 'so4', '#ec2227'], ['OC (ChemSpec FTIR)', 'oc', '#2e6b3c'], ['EC (FTIR)', 'bc', '#1a1a1a'],
    ['Nitrate', 'no3', '#f47c20'], ['Ammonium', 'nh4', '#ee5fa7'], ['Potassium', 'k', '#b48ad8'], ['Sodium', 'na', '#2f4ea0'],
  ]) {
    assert.equal(bars(field), hex)
    assert.equal(map(key), hex)
    assert.equal(categoricalScale('SPARTAN composition', [field])(field), hex)
  }
  assert.equal(categoricalScale('SPARTAN composition', ['Metals & crustal'])('Metals & crustal'), '#f9e81b')
})

test('Gallery retains the original composition range and other category sets are available', () => {
  const fields = ['OC', 'EC', 'Sulfate', 'Nitrate']
  const gallery = categoricalScale('Gallery', fields)
  assert.deepEqual(fields.map(gallery), ['#4e79a7', '#f28e2c', '#e15759', '#76b7b2'])
  assert.equal(categoricalStops('Gallery').length, 22)
  const families = ['Carbon', 'Ions', 'Metals & crustal']
  const familyScale = categoricalScale('Gallery', families, false, 'family')
  assert.deepEqual(families.map(familyScale), ['#5b6470', '#2171b5', '#b2182b'])
  for (const name of Object.keys(CATEGORICAL_COLOUR_SETS)) {
    assert.ok(categoricalStops(name).every(c => /^#[0-9a-f]{6}$/i.test(c)), name)
    const colours = categoricalScale(name, fields)
    assert.ok(fields.every(f => /^#[0-9a-f]{6}$/i.test(colours(f))), name)
  }
})

test('semantic sets keep their component mapping; generic sets can reverse without mutating the registry', () => {
  const before = categoricalStops('Gallery')
  assert.deepEqual(categoricalStops('Gallery', true), [...before].reverse())
  assert.deepEqual(categoricalStops('Gallery'), before)
  assert.equal(categoricalScale('Gallery', ['A'], true)('A'), before[before.length - 1])
  assert.equal(categoricalScale('SPARTAN composition', ['Sulfate'], true)('Sulfate'), '#ec2227')
  assert.deepEqual(categoricalStops('SPARTAN composition', true), Object.values(SPARTAN_COMPOSITION_COLOURS))
  assert.equal(hasNamedColours('SPARTAN composition'), true)
  assert.equal(hasNamedColours('Gallery'), false)
})

test('additional species use categorical colours without being assigned an unrelated reconstructed component', () => {
  const extra = categoricalScale('SPARTAN composition', ['Sulfate', 'Iron', 'Calcium'])
  const fallback = categoricalScale('Gallery', ['Sulfate', 'Iron', 'Calcium'])
  assert.equal(extra('Iron'), fallback('Iron'))
  assert.equal(extra('Calcium'), fallback('Calcium'))
  assert.notEqual(extra('Iron'), SPARTAN_COMPOSITION_COLOURS.dust)
  assert.deepEqual(categoricalStops('unknown set'), categoricalStops('Gallery'))
})
