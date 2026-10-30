import test from 'node:test'
import assert from 'node:assert/strict'
import { lab } from 'd3'
import { FTIR_GROUP_COLOURS, FTIR_GROUP_ORDER, PALETTES, SPARTAN_COMPOSITION_COLOURS, categoricalScale, getPalette, hasNamedColours, paletteColours, paletteNames, paletteNote } from './palettes.ts'

const HEX = /^#[0-9a-f]{6}$/i

test('every palette is well formed: unique names, hex colours, a source, a Gallery default', () => {
  for (const [kind, list] of Object.entries(PALETTES)) {
    const names = list.map(p => p.name)
    assert.equal(new Set(names).size, names.length, `${kind} names repeat`)
    assert.ok(names.includes('Gallery'), `${kind} has no Gallery palette`)
    for (const p of list) {
      assert.ok(p.colours.every(c => HEX.test(c)), `${kind} ${p.name}`)
      assert.ok(p.source, `${kind} ${p.name} has no source`)
      if (kind === 'sequential') assert.equal(p.colours.length, 8, p.name)
      if (kind === 'diverging') assert.equal(p.colours.length, 7, p.name)
      if (kind !== 'categorical') assert.ok(!p.named && !p.familyColours && !p.dark, `${p.name}: named and dark-twin colours are categorical only`)
      for (const [light, twin] of Object.entries(p.dark ?? {})) {
        assert.ok(p.colours.includes(light), `${p.name}: dark twin for a colour it does not have`)
        assert.ok(HEX.test(twin) && lab(twin).l > 50, `${p.name}: ${twin} would not show on the dark panel`)
      }
    }
  }
})

test('sequential ramps run one way in lightness, so more always reads as further along', () => {
  for (const p of PALETTES.sequential) {
    const l = p.colours.map(c => lab(c).l)
    const up = l[l.length - 1] > l[0]
    assert.ok(l.every((v, i) => i === 0 || (up ? v > l[i - 1] : v < l[i - 1])), `${p.name}: ${l.map(Math.round)}`)
  }
})

test('the published sets keep their published colours', () => {
  assert.deepEqual(paletteColours('categorical', 'Okabe–Ito'), ['#e69f00', '#56b4e9', '#009e73', '#f0e442', '#0072b2', '#d55e00', '#cc79a7', '#000000'])
  assert.deepEqual(paletteColours('categorical', 'Tol bright'), ['#4477aa', '#ee6677', '#228833', '#ccbb44', '#66ccee', '#aa3377', '#bbbbbb'])
  assert.equal(paletteColours('categorical', 'Tol muted').length, 9)
  const batlow = paletteColours('sequential', 'Batlow')
  assert.equal(batlow[0], '#011959')
  assert.equal(batlow[7], '#faccfa')
  assert.deepEqual(paletteColours('sequential', 'Gallery'), ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#084594'])
  assert.equal(paletteColours('diverging', 'Gallery')[3], '#f7f7f7')
})

test('notes come from the flags, and categorical sets say how many colours they have', () => {
  assert.equal(paletteNote('sequential', 'Viridis'), 'colour-vision safe')
  assert.equal(paletteNote('sequential', 'Greys'), 'prints in black and white')
  assert.equal(paletteNote('diverging', 'Greys'), 'prints in black and white')
  assert.equal(paletteNote('sequential', 'Plasma'), undefined)
  assert.equal(paletteNote('categorical', 'Tol bright'), 'colour-vision safe; 7 colours')
  assert.equal(paletteNote('categorical', 'Okabe–Ito'), 'colour-vision safe; 8 colours')
  assert.equal(paletteNote('categorical', 'Set 3'), '12 colours; pale: for filled areas, not lines or points')
  assert.equal(paletteNote('categorical', 'SPARTAN composition'), 'fixed component colours; additional species use Gallery colours')
  assert.equal(paletteNote('categorical', 'nothing'), undefined)
})

test('an unknown name falls back to the Gallery palette of its kind', () => {
  for (const kind of Object.keys(PALETTES)) assert.equal(getPalette(kind, 'unknown set').name, 'Gallery')
  assert.deepEqual(paletteColours('categorical', 'unknown set'), paletteColours('categorical', 'Gallery'))
  assert.deepEqual(paletteNames('categorical').slice(0, 2), ['SPARTAN composition', 'Gallery'])
})

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
  assert.equal(categoricalScale('SPARTAN composition', ['Crustal elements'])('Crustal elements'), '#f9e81b')
  assert.equal(categoricalScale('SPARTAN composition', ['Trace metals'])('Trace metals'), '#8c8c8c')
})

test('Gallery retains the original composition range and every set builds a scale', () => {
  const fields = ['OC', 'EC', 'Sulfate', 'Nitrate']
  const gallery = categoricalScale('Gallery', fields)
  assert.deepEqual(fields.map(gallery), ['#4e79a7', '#f28e2c', '#e15759', '#76b7b2'])
  assert.equal(paletteColours('categorical', 'Gallery').length, 22)
  const families = ['Carbon', 'Ions', 'Crustal elements']
  const familyScale = categoricalScale('Gallery', families, false, 'family')
  assert.deepEqual(families.map(familyScale), ['#5b6470', '#2171b5', '#b2182b'])
  for (const name of paletteNames('categorical')) {
    const colours = categoricalScale(name, fields)
    assert.ok(fields.every(f => HEX.test(colours(f))), name)
  }
})

test('semantic sets keep their component mapping; generic sets can reverse without mutating the registry', () => {
  const before = paletteColours('categorical', 'Gallery')
  assert.deepEqual(paletteColours('categorical', 'Gallery', true), [...before].reverse())
  assert.deepEqual(paletteColours('categorical', 'Gallery'), before)
  assert.equal(categoricalScale('Gallery', ['A'], true)('A'), before[before.length - 1])
  assert.equal(categoricalScale('SPARTAN composition', ['Sulfate'], true)('Sulfate'), '#ec2227')
  assert.deepEqual(paletteColours('categorical', 'SPARTAN composition', true), Object.values(SPARTAN_COMPOSITION_COLOURS))
  assert.equal(hasNamedColours('SPARTAN composition'), true)
  assert.equal(hasNamedColours('Gallery'), false)
})

test('additional species use categorical colours without being assigned an unrelated reconstructed component', () => {
  const extra = categoricalScale('SPARTAN composition', ['Sulfate', 'Iron', 'Calcium'])
  const fallback = categoricalScale('Gallery', ['Sulfate', 'Iron', 'Calcium'])
  assert.equal(extra('Iron'), fallback('Iron'))
  assert.equal(extra('Calcium'), fallback('Calcium'))
  assert.notEqual(extra('Iron'), SPARTAN_COMPOSITION_COLOURS.dust)
})

test('the FTIR functional groups keep the IMPROVE figure colours by default and in their own set, in its stack order', () => {
  assert.deepEqual(FTIR_GROUP_ORDER, ['Alkane CH', 'Carboxylic COOH', 'Non-acid CO', 'Alcohol COH', 'Amine NH2'])
  const groups = ['Alcohol COH', 'Non-acid CO', 'Carboxylic COOH', 'Alkane CH']
  for (const name of ['SPARTAN composition', 'FTIR functional groups']) {
    const scale = categoricalScale(name, groups)
    for (const g of groups) assert.equal(scale(g), FTIR_GROUP_COLOURS[g], `${name}: ${g}`)
  }
  assert.equal(categoricalScale('FTIR functional groups', ['carbonylCO'])('carbonylCO'), FTIR_GROUP_COLOURS['Non-acid CO'])
})
