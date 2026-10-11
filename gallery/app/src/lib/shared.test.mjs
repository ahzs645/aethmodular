import test from 'node:test'
import assert from 'node:assert/strict'
import { fmt, fixed, f2, f3, signed, pct, sci, fitLine } from './format.ts'
import { ALL_ADDIS, inSeasonGroup, isAllAddis, normalizeSeasonName, seasonOfMonth, seasonPhrase, shortSeason } from './seasons.ts'
import { BASELINE_COLOR, baselineColor } from './theme.ts'
import { baselineName, ecAxis } from './labels.ts'

test('formatters print a dash for missing values and keep their digit defaults', () => {
  for (const missing of [null, undefined, '', NaN, Infinity, 'x']) {
    assert.equal(fmt(missing), '—'); assert.equal(f2(missing), '—'); assert.equal(signed(missing), '—'); assert.equal(pct(missing), '—')
  }
  assert.equal(fmt(0.12345), '0.123')
  assert.equal(fixed(0.12345), '0.12')
  assert.equal(f2('1.5'), '1.50')
  assert.equal(f3(2), '2.000')
  assert.equal(signed(0.4), '+0.40')
  assert.equal(signed(-0.4), '−0.40')
  assert.equal(signed(-1.2345, 3), '−1.234')
  assert.equal(pct(0.256), '26 %')
  assert.equal(sci(0.000123), '1.2e-4')
  assert.equal(fitLine(0.82, -1.06), '0.82x − 1.06')
  assert.equal(fitLine(1, 0.5), '1.00x + 0.50')
})

test('season helpers agree on names, the all-Addis group and month bins', () => {
  assert.equal(normalizeSeasonName('Dry (Oct–Feb)'), 'Dry (Oct-Feb)')
  assert.equal(normalizeSeasonName('Dry (Oct-Feb)'), 'Dry (Oct-Feb)')
  assert.equal(shortSeason('Kiremt (Jun-Sep)'), 'Kiremt')
  assert.equal(shortSeason(ALL_ADDIS), ALL_ADDIS)
  assert.equal(seasonPhrase(ALL_ADDIS), 'all seasons')
  assert.ok(isAllAddis('all') && isAllAddis(ALL_ADDIS) && !isAllAddis('Dry (Oct-Feb)'))
  assert.ok(inSeasonGroup('Belg (Mar-May)', ALL_ADDIS))
  assert.ok(inSeasonGroup('Belg (Mar-May)', 'Belg (Mar-May)'))
  assert.ok(!inSeasonGroup('Belg (Mar-May)', 'Dry (Oct-Feb)'))
  const dryFeb = [{ name: 'Dry (Oct-Feb)', months: [10, 11, 12, 1, 2] }, { name: 'Belg (Mar-May)', months: [3, 4, 5] }]
  assert.equal(seasonOfMonth(2, dryFeb)?.name, 'Dry (Oct-Feb)')
  assert.equal(seasonOfMonth(7, dryFeb), null)
  assert.equal(seasonOfMonth(null, dryFeb), null)
})

test('a VIBES run is coloured and named by its rank', () => {
  assert.equal(baselineColor('AIRSpec'), '#2C6E9E')
  assert.equal(baselineColor('VIBES'), BASELINE_COLOR.VIBES)
  assert.equal(baselineColor('VIBES', 70), BASELINE_COLOR['VIBES-full-r70'])
  assert.equal(baselineColor('VIBES', 28), BASELINE_COLOR.VIBES)
  assert.notEqual(BASELINE_COLOR.AIRSpec, '#2171b5', 'the accent / OLS blue is not a baseline colour')
  assert.equal(baselineName('VIBES', { backgroundRank: 70 }), baselineName('VIBES-full-r70'))
  assert.equal(baselineName('VIBES-full-r70', { short: true }), 'VIBES')
  assert.equal(baselineName('unknown-method'), 'unknown-method')
  assert.equal(ecAxis('hips'), 'HIPS Fabs / MAC 10 (µg/m³)')
  assert.equal(ecAxis('ftir', 'µg per filter'), 'FTIR EC (µg per filter)')
})

test('cell labels pick readable ink from the fill', async () => {
  const { inkOn, INK } = await import('./theme.ts')
  assert.equal(inkOn('#ffffff'), INK.text)
  assert.equal(inkOn('#f7f7f7'), INK.text)
  assert.equal(inkOn('#2166ac'), '#fff')
  assert.equal(inkOn('#b2182b'), '#fff')
  assert.equal(inkOn('rgb(8, 48, 107)'), '#fff')
  assert.equal(inkOn('rgb(222, 235, 247)'), INK.text)
  assert.equal(inkOn('not a colour'), INK.text)
})
