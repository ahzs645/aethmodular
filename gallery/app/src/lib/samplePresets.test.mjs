import assert from 'node:assert/strict'
import test from 'node:test'
import { linkSetsSamples, NO_PRESET_KEY, noPresetMark, samplesFor, samplesIn, sameSamples, withPreset } from './samplePresets.ts'

const ADDIS = { label: 'Addis Ababa', samples: { ssites: 'ETAD', excl: '1' } }

test('a bare link opens on the page\'s preset', () => {
  const p = withPreset(new URLSearchParams('tab=research&aep=rh-sweep'), ADDIS)
  assert.equal(p.get('ssites'), 'ETAD')
  assert.equal(p.get('excl'), '1')
  assert.equal(p.get('aep'), 'rh-sweep')
})

test('a link that names samples keeps them, whole: the preset does not fill the rest', () => {
  const p = withPreset(new URLSearchParams('tab=research&ssites=ETAD,ETBI,CHTS'), ADDIS)
  assert.equal(p.get('ssites'), 'ETAD,ETBI,CHTS')
  assert.equal(p.get('excl'), null)
})

test('hand exclusions and page controls do not count as naming samples', () => {
  assert.equal(linkSetsSamples(new URLSearchParams('tab=research&drop=ETAD-0001&keep=ETAD-0002&grp=x&fits=ols')), false)
})

test('a page left at every default says so, so a reload does not bring the preset back', () => {
  assert.equal(noPresetMark({}, ADDIS), '1')
  assert.equal(noPresetMark({ ssites: 'CHTS' }, ADDIS), null)
  assert.equal(noPresetMark({}, { label: 'Focus sites', samples: {} }), null)
  assert.equal(noPresetMark({}, null), null)
  const p = new URLSearchParams(`tab=research&${NO_PRESET_KEY}=1`)
  assert.equal(withPreset(p, ADDIS).get('ssites'), null)
})

test('a page change: the link, then what the reader left, then the preset', () => {
  const remembered = new Map([['research/rh-sweep', { ssites: 'ETAD,CHTS' }]])
  assert.deepEqual(samplesFor('research/rh-sweep', ADDIS, remembered, { ssites: 'INDH' }), { ssites: 'INDH' })
  assert.deepEqual(samplesFor('research/rh-sweep', ADDIS, remembered), { ssites: 'ETAD,CHTS' })
  assert.deepEqual(samplesFor('research/other', ADDIS, remembered), ADDIS.samples)
  assert.deepEqual(samplesFor('', null, remembered), {})
})

test('sets compare by value, and an unset key equals an empty one', () => {
  const m = new Map([['ssites', 'ETAD'], ['cal', ''], ['drop', 'x']])
  assert.deepEqual(samplesIn(m), { ssites: 'ETAD' })
  assert.equal(sameSamples({ ssites: 'ETAD' }, samplesIn(m)), true)
  assert.equal(sameSamples({ ssites: 'ETAD', excl: '1' }, samplesIn(m)), false)
})
