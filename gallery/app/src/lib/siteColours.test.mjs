import test from 'node:test'
import assert from 'node:assert/strict'
import { colourGroups, colouring, namedColor, seasonColor, siteCodeColor, siteColor } from './siteColours.ts'
import { INK, MONTHS, MONTH_LETTERS, lotColor, LOT_UNKNOWN, preprocessingColor, PREPROCESSING_COLOR, BASELINE_COLOR, prettyUnit, withUnit } from './theme.ts'

const meta = {
  sites: [{ name: 'Addis Ababa', code: 'ETAD', color: '#F39C12' }, { name: 'Delhi', code: 'INDH', color: '#3498DB' }],
  seasons: [{ name: 'Dry (Oct-Feb)', months: [10, 11, 12, 1, 2], color: '#E67E22' }, { name: 'Belg (Mar-May)', months: [3, 4, 5], color: '#27AE60' }],
}

test('site and season colours come from meta, with a muted or given fallback', () => {
  assert.equal(siteColor(meta, 'Delhi'), '#3498DB')
  assert.equal(siteColor(meta, 'Nowhere'), INK.muted)
  assert.equal(siteColor(meta, 'Nowhere', '#123456'), '#123456')
  assert.equal(siteCodeColor(meta, 'ETAD'), '#F39C12')
  assert.equal(siteCodeColor(meta, 'XXXX', INK.accent), INK.accent)
  assert.equal(seasonColor(meta, 'Belg (Mar-May)'), '#27AE60')
  assert.equal(seasonColor(meta, null), INK.muted)
  assert.equal(namedColor(meta.seasons, 'Dry (Oct-Feb)'), '#E67E22')
  assert.equal(namedColor(undefined, 'Dry (Oct-Feb)', INK.text), INK.text)
})

test('colouring goes by season for one site and by site for several', () => {
  const one = [{ site: 'Addis Ababa', season: 'Belg (Mar-May)' }, { site: 'Addis Ababa', season: 'Dry (Oct-Feb)' }]
  const c1 = colouring(one, meta)
  assert.equal(c1.by, 'season')
  assert.deepEqual(c1.order(one), ['Dry (Oct-Feb)', 'Belg (Mar-May)'])
  assert.equal(c1.color('Dry (Oct-Feb)'), '#E67E22')
  assert.equal(c1.label(''), 'No season')
  const two = [...one, { site: 'Delhi', season: 'Dry (Oct-Feb)' }]
  const c2 = colouring(two, meta)
  assert.equal(c2.by, 'site')
  assert.deepEqual(c2.order(two), ['Addis Ababa', 'Delhi'])
  assert.equal(c2.color('Delhi'), '#3498DB')
})

test('colourGroups keeps first-appearance order and pairs each group with its colour', () => {
  const rows = [{ site: 'Delhi', season: 'Belg (Mar-May)' }, { site: 'Addis Ababa', season: 'Dry (Oct-Feb)' }, { site: 'Delhi', season: 'Dry (Oct-Feb)' }]
  assert.deepEqual(colourGroups(rows, 'site', meta).map(([k, c, rs]) => [k, c, rs.length]), [['Delhi', '#3498DB', 2], ['Addis Ababa', '#F39C12', 1]])
  assert.deepEqual(colourGroups(rows, 'season', meta).map(([k, c, rs]) => [k, c, rs.length]), [['Belg (Mar-May)', '#27AE60', 1], ['Dry (Oct-Feb)', '#E67E22', 2]])
})

test('shared constants: months, units, lots and preprocessing colours', () => {
  assert.equal(MONTHS.length, 12)
  assert.equal(MONTHS[0], 'Jan')
  assert.equal(MONTH_LETTERS.join(''), 'JFMAMJJASOND')
  assert.equal(prettyUnit('ug/m3'), 'µg/m³')
  assert.equal(prettyUnit('µg/m3'), 'µg/m³')
  assert.equal(prettyUnit('ng/m3'), 'ng/m³')
  assert.equal(prettyUnit('Mm-1'), 'Mm⁻¹')
  assert.equal(prettyUnit('ug/m3 per ug/m3'), 'µg/m³ per µg/m³')
  assert.equal(prettyUnit(undefined), '')
  assert.equal(prettyUnit('%'), '%')
  assert.equal(withUnit('EC (FTIR)', { 'EC (FTIR)': 'ug/m3' }), 'EC (FTIR) (µg/m³)')
  assert.equal(lotColor(251), lotColor('251'))
  assert.equal(lotColor('999'), LOT_UNKNOWN)
  // shared names agree with the baseline colours; the workbook's vibes is the current (rank-70) VIBES
  assert.equal(PREPROCESSING_COLOR.spline, BASELINE_COLOR.AIRSpec)
  assert.equal(PREPROCESSING_COLOR.airspec, BASELINE_COLOR.airspec)
  assert.equal(PREPROCESSING_COLOR.vibes, BASELINE_COLOR['VIBES-full-r70'])
  assert.equal(preprocessingColor('VIBES-full'), BASELINE_COLOR['VIBES-full'])
  assert.equal(preprocessingColor('nope'), INK.axis)
})
