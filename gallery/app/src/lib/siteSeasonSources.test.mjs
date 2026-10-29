import test from 'node:test'
import assert from 'node:assert/strict'
import { calendarNote, calendarVariants, localSeasonsFor, seasonSources } from '../siteSeasons.ts'

test('papers carry the audit qualifications and agencies are listed beside them', () => {
  const addis = seasonSources('ETAD')
  const shlipak = addis.find((s) => s.cite.startsWith('Shlipak'))
  assert.deepEqual([shlipak.match, shlipak.scope, shlipak.checked], ['exact', 'local', 'full text'])
  assert.ok(addis.some((s) => !s.paper && s.url.includes('ethiomet.gov.et')))
  // the same Addis paper is borrowed evidence at Bishoftu
  assert.equal(seasonSources('ETBI').find((s) => s.cite.startsWith('Shlipak')).scope, 'transferred')
  // a source that is one of the papers is not listed twice
  const ilha = seasonSources('ILHA')
  assert.equal(ilha.filter((s) => s.cite.startsWith('Alpert')).length, 1)
})

test('a calendar variant shows only its own references', () => {
  const dry = seasonSources('ETAD', 'dry_feb')
  assert.deepEqual(dry.map((s) => s.cite), ['LaPolt et al. (2025)', 'Yirga et al. (2024)'])
  assert.equal(dry.find((s) => s.cite.startsWith('Yirga')).match, 'inferred')
  const jaipur = seasonSources('INJA', 'imd_jaipur')
  assert.ok(jaipur.length === 1 && !jaipur[0].paper && jaipur[0].url.includes('mausam.imd.gov.in/jaipur'))
  assert.ok(seasonSources('INJA').some((s) => s.cite.startsWith('Pandey')))
  assert.equal(seasonSources('TWTA', 'outlook')[0].cite, 'Chou et al. (2010)')
  assert.match(calendarNote('CODC'), /unresolved/)
})

test('the new alternatives are offered and partition the year', () => {
  for (const [code, key, feb] of [['INJA', 'imd_jaipur', 'Winter (Dec–Feb)'], ['INKA', 'dec_winter', 'Winter (Dec–Feb)'], ['BIBU', 'nkunzimana', 'Semidry (Jan–Feb)']]) {
    assert.ok(calendarVariants('', code).some((v) => v.key === key), code)
    const cal = localSeasonsFor('x', { season_conventions: {} }, code, key)
    assert.deepEqual(cal.flatMap((s) => s.months).sort((a, b) => a - b), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    assert.equal(cal.find((s) => s.months.includes(2)).name, feb)
  }
})
