import assert from 'node:assert/strict'
import { test } from 'node:test'
import { readQueryValue } from './url.ts'

test('a key reads its own value first, then its legacy keys in order', () => {
  const q = new URLSearchParams('aq.site=ETAD&research.site=Addis')
  assert.equal(readQueryValue(q, 'site', ['aq.site']), 'ETAD')
  assert.equal(readQueryValue(new URLSearchParams('site=X&aq.site=ETAD'), 'site', ['aq.site']), 'X')
  assert.equal(readQueryValue(new URLSearchParams('b=2&a=1'), 'k', ['a', 'b']), '1')
  assert.equal(readQueryValue(new URLSearchParams(''), 'k', ['a']), null)
})

test('legacy keys are read under the same tile prefix', () => {
  const q = new URLSearchParams('t3.season=Dry&season=Wet')
  assert.equal(readQueryValue(q, 't3.weather-corr.season', ['season'], 't3.'), 'Dry')
})
