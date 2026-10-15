import assert from 'node:assert/strict'
import { test } from 'node:test'
import { resolveOptions } from './optionRules.ts'

// the scatterplot's rules, as declared in charts/correlation/Scatterplot.tsx
const SCATTER = [
  { key: 'layout', choices: ['Combined', 'Per site', 'Per season'], default: 'Combined' },
  { key: 'fitLines', toggle: true, default: true },
  { key: 'grouping', default: 'pooled', hidden: 'pooled',
    when: (c) => c.fitLines && c.estimators > 0 && c.layout !== 'Per season',
    hiddenNote: (c) => (c.fitLines && c.estimators > 0 && c.layout === 'Per season' ? 'one fit per panel' : null),
    choices: [{ value: 'pooled', label: (c) => (c.layout === 'Per site' ? 'One fit per site' : 'One pooled fit') },
      { value: 'site', when: (c) => c.layout === 'Combined' }, { value: 'season' }] },
]
const item = (r, key) => r.items.find((i) => i.def.key === key)

test('combined offers every grouping and keeps the reader\'s choice', () => {
  const r = resolveOptions(SCATTER, ['Combined', true, 'site'], { estimators: 1 })
  assert.equal(r.values.grouping, 'site')
  assert.deepEqual(item(r, 'grouping').offered.map((c) => c.value), ['pooled', 'site', 'season'])
})

test('per site drops "per site" (it is the pooled fit) and relabels pooled', () => {
  const r = resolveOptions(SCATTER, ['Per site', true, 'site'], { estimators: 1 })
  assert.equal(r.values.grouping, 'pooled')
  assert.deepEqual(item(r, 'grouping').offered, [{ value: 'pooled', label: 'One fit per site' }, { value: 'season', label: 'season' }])
})

test('per season hides grouping with a note, fit lines off hides it silently', () => {
  const season = resolveOptions(SCATTER, ['Per season', true, 'season'], { estimators: 1 })
  assert.equal(item(season, 'grouping').kind, 'hidden')
  assert.equal(item(season, 'grouping').note, 'one fit per panel')
  assert.equal(season.values.grouping, 'pooled')
  const off = resolveOptions(SCATTER, ['Combined', false, 'season'], { estimators: 1 })
  assert.equal(item(off, 'grouping').kind, 'hidden')
  assert.equal(item(off, 'grouping').note, null)
})

test('no estimator ticked hides grouping', () => {
  const r = resolveOptions(SCATTER, ['Combined', true, 'site'], { estimators: 0 })
  assert.equal(item(r, 'grouping').kind, 'hidden')
})

test('an unknown stored value falls back to the default', () => {
  const r = resolveOptions(SCATTER, ['Sideways', true, 'pooled'], { estimators: 1 })
  assert.equal(r.values.layout, 'Combined')
})
